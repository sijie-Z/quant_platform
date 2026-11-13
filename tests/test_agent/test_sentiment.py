"""Tests for LLM sentiment factor."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_platform.agent.sentiment_factor import (
    SAMPLE_HEADLINES_POOL,
    KeywordSentimentAnalyzer,
    LLMSentimentFactor,
    SentimentAnalyzer,
)


class TestKeywordSentimentAnalyzer:
    """Test the keyword-based sentiment analyzer (default backend)."""

    def test_bullish_headline(self):
        analyzer = KeywordSentimentAnalyzer()
        score = analyzer.analyze("净利润同比增长50%，业绩超预期")
        assert score > 0, f"Expected positive score, got {score}"

    def test_bearish_headline(self):
        analyzer = KeywordSentimentAnalyzer()
        score = analyzer.analyze("大股东减持100万股，收到监管问询函")
        assert score < 0, f"Expected negative score, got {score}"

    def test_neutral_headline(self):
        analyzer = KeywordSentimentAnalyzer()
        score = analyzer.analyze("公布定期报告，召开股东大会")
        assert -0.3 <= score <= 0.3, f"Expected neutral score, got {score}"

    def test_empty_headline(self):
        analyzer = KeywordSentimentAnalyzer()
        score = analyzer.analyze("某公司发布公告")
        assert score == 0.0

    def test_score_range(self):
        """All scores should be in [-1, 1]."""
        analyzer = KeywordSentimentAnalyzer()
        for template, _ in SAMPLE_HEADLINES_POOL:
            score = analyzer.analyze(template)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {template}"


class TestLLMSentimentFactor:
    """Test the LLM sentiment factor class."""

    def test_instantiation(self):
        factor = LLMSentimentFactor(use_real_llm=False)
        assert factor.name == "llm_sentiment"
        assert factor.category.value == "custom"

    def test_instantiation_with_real_llm_no_key(self):
        """Should fall back to keyword analyzer when no API key."""
        factor = LLMSentimentFactor(use_real_llm=True)
        assert isinstance(factor.analyzer, KeywordSentimentAnalyzer)

    def test_compute_returns_dataframe(self, prices):
        """Factor.compute() should return a DataFrame."""
        factor = LLMSentimentFactor(use_real_llm=False)
        result = factor.compute(prices)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == prices.shape

    def test_compute_values_in_range(self, prices):
        """Sentiment scores should be in [-1, 1]."""
        factor = LLMSentimentFactor(use_real_llm=False)
        result = factor.compute(prices)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1.0
            assert valid.max().max() <= 1.0

    def test_run_method_delegation(self, prices):
        """The .run() method should delegate to .compute()."""
        factor = LLMSentimentFactor(use_real_llm=False)
        result = factor.run(prices)
        assert result.name == "llm_sentiment"
        assert isinstance(result.values, pd.DataFrame)

    def test_lookback_smoothing(self, prices):
        """Longer lookback should produce smoother output."""
        f1 = LLMSentimentFactor(use_real_llm=False, lookback_days=1, name="s1")
        f5 = LLMSentimentFactor(use_real_llm=False, lookback_days=5, name="s5")
        r1 = f1.compute(prices)
        r5 = f5.compute(prices)
        # Rolling mean should have fewer extreme values
        if r1.dropna().size > 0 and r5.dropna().size > 0:
            assert r5.std().std() <= r1.std().std() * 1.2  # Allow small jitter

    def test_different_names(self, prices):
        """Factors with different names should not collide."""
        f1 = LLMSentimentFactor(use_real_llm=False, name="sent_a")
        f2 = LLMSentimentFactor(use_real_llm=False, name="sent_b")
        assert f1.name == "sent_a"
        assert f2.name == "sent_b"

    def test_fill_template(self, prices):
        """Template filling should produce strings without placeholders."""
        factor = LLMSentimentFactor(use_real_llm=False)
        rng = np.random.default_rng(42)
        for template, _ in SAMPLE_HEADLINES_POOL[:5]:
            filled = factor._fill_template(template, rng)
            assert "{" not in filled, f"Unfilled placeholder in: {filled}"


class _CountingAnalyzer(SentimentAnalyzer):
    """Deterministic analyzer that records every call."""

    def __init__(self, score: float = 0.5, fallback: bool = False):
        self.score = score
        self.calls: list[tuple[str, str]] = []
        self.last_analyze_fallback = fallback

    def analyze(self, headline: str, asset: str = "") -> float:
        self.calls.append((headline, asset))
        return self.score


class TestCacheKeyIsolation:
    """The old key was md5(headline|date): a score computed for one stock was
    replayed for another, and a keyword fallback was served as if it were the
    model's answer. The key must cover everything that changes the score."""

    def test_different_assets_do_not_share_an_entry(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=0.5)
        factor.analyzer = analyzer
        date = pd.Timestamp("2024-01-02")

        factor._analyze_headline("业绩超预期", "600000.SH", date)
        factor._analyze_headline("业绩超预期", "000001.SZ", date)

        assert len(analyzer.calls) == 2
        assert len(factor._cache) == 2

    def test_different_analyzers_do_not_share_an_entry(self, tmp_path):
        date = pd.Timestamp("2024-01-02")

        keyword_factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        keyword_factor._analyze_headline("净利润增长", "600000.SH", date)
        keyword_factor._save_cache()

        other = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=-0.9)
        other.analyzer = analyzer
        score = other._analyze_headline("净利润增长", "600000.SH", date)

        assert len(analyzer.calls) == 1, "keyword answer leaked to another analyzer"
        assert score == pytest.approx(-0.9)

    def test_different_models_do_not_share_an_entry(self, tmp_path):
        date = pd.Timestamp("2024-01-02")

        first = LLMSentimentFactor(
            model="deepseek-chat", cache_dir=str(tmp_path),
        )
        first.analyzer = _CountingAnalyzer(score=0.5)
        first._analyze_headline("业绩超预期", "600000.SH", date)
        first._save_cache()

        second = LLMSentimentFactor(
            model="deepseek-reasoner", cache_dir=str(tmp_path),
        )
        analyzer = _CountingAnalyzer(score=-0.4)
        second.analyzer = analyzer
        score = second._analyze_headline("业绩超预期", "600000.SH", date)

        assert len(analyzer.calls) == 1
        assert score == pytest.approx(-0.4)

    def test_prompt_change_invalidates_the_entry(self, tmp_path):
        date = pd.Timestamp("2024-01-02")
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        factor.analyzer = _CountingAnalyzer(score=0.5)
        factor._analyze_headline("业绩超预期", "600000.SH", date)

        analyzer = _CountingAnalyzer(score=0.5)
        analyzer.SYSTEM_PROMPT = "different instructions"
        factor.analyzer = analyzer
        factor._analyze_headline("业绩超预期", "600000.SH", date)

        assert len(analyzer.calls) == 1

    def test_persisted_entries_are_reused(self, tmp_path):
        date = pd.Timestamp("2024-01-02")
        first = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        first._analyze_headline("净利润增长", "600000.SH", date)
        first._save_cache()

        second = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        calls = []

        def fail_on_miss(*args, **kwargs):
            calls.append(args)
            raise AssertionError("the persisted entry was not reused")

        second.analyzer.analyze = fail_on_miss  # type stays KeywordSentimentAnalyzer
        score = second._analyze_headline("净利润增长", "600000.SH", date)

        assert calls == []
        assert score == pytest.approx(KeywordSentimentAnalyzer().analyze("净利润增长"))


class TestFallbackScoresAreNotPersisted:
    """A DeepSeek outage used to be cached forever: the keyword fallback was
    stored exactly like a model answer. Fallbacks stay retryable."""

    def test_fallback_is_kept_out_of_the_cache_file_and_retried(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=0.1, fallback=True)
        factor.analyzer = analyzer
        date = pd.Timestamp("2024-01-02")

        first = factor._analyze_headline("业绩超预期", "600000.SH", date)
        assert first == pytest.approx(0.1)
        assert next(iter(factor._cache.values()))["retryable"] is True

        factor._save_cache()
        payload = json.loads(
            (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8")
        )
        assert payload["entries"] == {}, "fallback score must not be persisted"

        factor._analyze_headline("业绩超预期", "600000.SH", date)
        assert len(analyzer.calls) == 2, "a fallback entry must be retried"

    def test_normal_scores_are_persisted(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        factor.analyzer = _CountingAnalyzer(score=0.7, fallback=False)
        factor._analyze_headline("业绩超预期", "600000.SH", pd.Timestamp("2024-01-02"))
        factor._save_cache()

        payload = json.loads(
            (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8")
        )
        assert len(payload["entries"]) == 1
        assert next(iter(payload["entries"].values()))["score"] == pytest.approx(0.7)


class TestCorruptCacheFile:
    def test_unreadable_file_does_not_raise_or_wipe_the_in_memory_cache(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        factor._cache["kept"] = {"score": 0.4, "retryable": False}
        cache_file = tmp_path / "sentiment_cache.json"
        cache_file.write_text("{ this is not json", encoding="utf-8")

        factor._load_cache()  # must not raise

        assert "kept" in factor._cache
        assert cache_file.read_text(encoding="utf-8") == "{ this is not json"

    def test_older_key_layout_is_ignored_without_wiping_memory(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        factor._cache["kept"] = {"score": 0.4, "retryable": False}
        (tmp_path / "sentiment_cache.json").write_text(
            json.dumps({"abcd": 0.5}), encoding="utf-8"
        )

        factor._load_cache()

        assert factor._cache == {"kept": {"score": 0.4, "retryable": False}}


class TestAtomicCacheWrite:
    def test_save_replaces_the_file_and_leaves_no_temp_files(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        factor._cache["k"] = {"score": 0.5, "retryable": False}

        factor._save_cache()

        payload = json.loads(
            (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8")
        )
        assert payload["version"] == factor._CACHE_VERSION
        assert payload["entries"]["k"]["score"] == pytest.approx(0.5)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path, monkeypatch):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        factor._cache["k"] = {"score": 0.5, "retryable": False}
        factor._save_cache()
        good = (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8")

        factor._cache["k2"] = {"score": 0.6, "retryable": False}

        import quant_platform.agent.sentiment_factor as module

        def broken_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(module.os, "replace", broken_replace)
        factor._save_cache()  # must not raise

        assert (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8") == good
        assert list(tmp_path.glob("*.tmp")) == []
