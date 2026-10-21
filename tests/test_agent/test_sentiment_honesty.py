"""Honesty and robustness tests for the LLM sentiment factor.

Covers audit items R6 (requested vs. actual backend), R7 (fabricated news),
R14 (cache poisoning) and R15 (TTL + keyword double-count).
"""

import json
import logging
import time

import numpy as np
import pandas as pd
import pytest

from quant_platform.agent.sentiment_factor import (
    KeywordSentimentAnalyzer,
    LLMSentimentFactor,
    OpenAISentimentAnalyzer,
    SentimentAnalyzer,
)


class _CountingAnalyzer(SentimentAnalyzer):
    """Deterministic analyzer that records every call."""

    def __init__(self, score: float = 0.5, fallback: bool = False):
        self.score = score
        self.calls: list[tuple[str, str]] = []
        self.last_analyze_fallback = fallback

    def analyze(self, headline: str, asset: str = "") -> float:
        self.calls.append((headline, asset))
        return self.score


@pytest.fixture
def small_prices():
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "600000.SH": np.linspace(10.0, 11.0, len(dates)),
            "000001.SZ": np.linspace(20.0, 21.0, len(dates)),
        },
        index=dates,
    )


# ----------------------------------------------------------------------
# R6: requested vs. actual backend
# ----------------------------------------------------------------------

class TestBackendHonesty:
    def test_no_key_downgrade_state_and_logs(self, tmp_path, monkeypatch, caplog):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        caplog.set_level(logging.INFO)

        factor = LLMSentimentFactor(use_real_llm=True, cache_dir=str(tmp_path))
        prices = pd.DataFrame(
            {"600000.SH": [10.0, 10.1]},
            index=pd.bdate_range("2024-01-02", periods=2),
        )
        factor.compute(prices)

        assert factor.requested_use_real_llm is True
        assert factor.use_real_llm is False
        assert factor.is_llm is False
        assert factor.backend == "keyword"
        assert factor.backend_downgraded is True
        assert factor.params["effective_backend"] == "keyword"

        messages = " | ".join(record.getMessage() for record in caplog.records)
        backend_lines = [message for message in messages.split(" | ") if "backend=" in message]
        assert any("backend=keyword" in message for message in backend_lines)
        assert not any("DeepSeek" in message for message in backend_lines)
        assert "downgraded" in messages.lower()

    def test_real_backend_label_when_key_present(self, tmp_path):
        factor = LLMSentimentFactor(
            use_real_llm=True, api_key="sk-test", cache_dir=str(tmp_path),
        )
        assert factor.is_llm is True
        assert factor.use_real_llm is True
        assert factor.backend == "llm:deepseek-chat"
        assert factor.backend_downgraded is False

    def test_keyword_request_never_claims_llm(self, tmp_path):
        factor = LLMSentimentFactor(use_real_llm=False, cache_dir=str(tmp_path))
        assert factor.requested_use_real_llm is False
        assert factor.is_llm is False
        assert factor.backend == "keyword"
        assert factor.backend_downgraded is False


# ----------------------------------------------------------------------
# R7: fabricated news vs. caller-supplied news
# ----------------------------------------------------------------------

class TestSyntheticDemoHonesty:
    def test_no_news_default_is_flagged_synthetic(self, tmp_path, small_prices, caplog):
        caplog.set_level(logging.WARNING)
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))

        factor.compute(small_prices)

        assert factor.is_synthetic is True
        assert factor.data_source == "synthetic_templates"
        assert factor.params["is_synthetic"] is True
        assert factor.params["data_source"] == "synthetic_templates"
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "SYNTHETIC" in messages

    def test_real_llm_without_news_fails_closed(self, tmp_path, small_prices):
        factor = LLMSentimentFactor(
            use_real_llm=True, api_key="sk-test", cache_dir=str(tmp_path),
        )
        with pytest.raises(ValueError, match="no caller-supplied news"):
            factor.compute(small_prices)

    def test_real_llm_explicit_synthetic_opt_in_is_labeled(
        self, tmp_path, monkeypatch,
    ):
        calls = []

        def fake_analyze(self, headline, asset=""):
            calls.append((headline, asset))
            return 0.5

        monkeypatch.setattr(OpenAISentimentAnalyzer, "analyze", fake_analyze)
        factor = LLMSentimentFactor(
            use_real_llm=True,
            api_key="sk-test",
            cache_dir=str(tmp_path),
            allow_synthetic_llm=True,
        )
        dates = pd.bdate_range("2024-01-02", periods=30)
        prices = pd.DataFrame(
            {"600000.SH": np.linspace(10.0, 11.0, len(dates))}, index=dates,
        )

        factor.compute(prices)

        assert factor.is_synthetic is True
        assert factor.data_source == "synthetic_templates"
        assert calls, "the real-LLM analyzer should have scored demo headlines"
        assert factor.n_analyzer_calls == len(calls) > 0


class TestCallerSuppliedNews:
    def test_supplied_headlines_are_scored_and_not_synthetic(
        self, tmp_path, small_prices,
    ):
        factor = LLMSentimentFactor(lookback_days=1, cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=0.7)
        factor.analyzer = analyzer

        news = {
            "600000.SH": {
                small_prices.index[0]: ["净利润同比增长50%"],
                small_prices.index[1]: "大股东减持",  # single string accepted
            },
        }
        result = factor.compute(small_prices, news=news)

        assert factor.is_synthetic is False
        assert factor.data_source == "caller_supplied_news"
        assert factor.params["is_synthetic"] is False
        assert analyzer.calls == [
            ("净利润同比增长50%", "600000.SH"),
            ("大股东减持", "600000.SH"),
        ]
        assert result.loc[small_prices.index[0], "600000.SH"] == pytest.approx(0.7)
        assert result.loc[small_prices.index[1], "600000.SH"] == pytest.approx(0.7)
        assert result["600000.SH"].isna().sum() == 3
        assert result["000001.SZ"].isna().all()

    def test_headlines_alias_is_accepted(self, tmp_path, small_prices):
        factor = LLMSentimentFactor(lookback_days=1, cache_dir=str(tmp_path))
        factor.analyzer = _CountingAnalyzer(score=0.2)
        date_key = str(small_prices.index[0].date())

        result = factor.compute(
            small_prices, headlines={"000001.SZ": {date_key: ["公告"]}},
        )

        assert factor.is_synthetic is False
        assert result.loc[small_prices.index[0], "000001.SZ"] == pytest.approx(0.2)

    def test_news_and_headlines_together_are_rejected(self, tmp_path, small_prices):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="not both"):
            factor.compute(small_prices, news={}, headlines={})

    def test_empty_news_does_not_fabricate_headlines(self, tmp_path, small_prices):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=0.7)
        factor.analyzer = analyzer

        result = factor.compute(small_prices, news={})

        assert analyzer.calls == []
        assert result.isna().to_numpy().all()
        assert factor.is_synthetic is False
        assert factor.data_source == "caller_supplied_news"

    def test_unknown_asset_is_ignored(self, tmp_path, small_prices):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=0.7)
        factor.analyzer = analyzer

        result = factor.compute(
            small_prices,
            news={"NOT_IN_PANEL": {small_prices.index[0]: ["公告"]}},
        )

        assert analyzer.calls == []
        assert result.isna().to_numpy().all()

    def test_dict_headline_bucket_is_rejected(self, tmp_path, small_prices):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=0.7)
        factor.analyzer = analyzer

        result = factor.compute(
            small_prices,
            news={"600000.SH": {small_prices.index[0]: {"headline": "x"}}},
        )

        assert analyzer.calls == []
        assert result.isna().to_numpy().all()

    def test_earlier_result_provenance_is_not_retroactively_relabeled(
        self, tmp_path, small_prices,
    ):
        factor = LLMSentimentFactor(lookback_days=1, cache_dir=str(tmp_path))

        synthetic_result = factor.run(small_prices)
        assert synthetic_result.params["is_synthetic"] is True

        news_result = factor.run(
            small_prices,
            news={"600000.SH": {small_prices.index[0]: ["公告"]}},
        )

        assert news_result.params["is_synthetic"] is False
        assert synthetic_result.params["is_synthetic"] is True
        assert synthetic_result.params is not news_result.params


# ----------------------------------------------------------------------
# R14: cache validation / poisoning
# ----------------------------------------------------------------------

class TestCacheValidation:
    @staticmethod
    def _write_entries(tmp_path, entries):
        payload = {
            "version": LLMSentimentFactor._CACHE_VERSION,
            "entries": entries,
        }
        (tmp_path / "sentiment_cache.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )

    def test_poisoned_persisted_entries_are_rejected(self, tmp_path):
        now = time.time()
        version = LLMSentimentFactor._CACHE_VERSION
        self._write_entries(tmp_path, {
            "good": {"version": version, "score": 0.5, "ts": now},
            "high": {"version": version, "score": 123.45, "ts": now},
            "nan": {"version": version, "score": float("nan"), "ts": now},
            "inf": {"version": version, "score": float("inf"), "ts": now},
            "string": {"version": version, "score": "0.5", "ts": now},
            "bool": {"version": version, "score": True, "ts": now},
            "no_version": {"score": 0.5, "ts": now},
            "old_version": {"version": version - 1, "score": 0.5, "ts": now},
            "future_ts": {"version": version, "score": 0.5, "ts": now + 30 * 86400},
            "expired": {"version": version, "score": 0.5, "ts": now - 30 * 86400},
            "persisted_fallback": {
                "version": version, "score": 0.5, "ts": now, "retryable": True,
            },
        })

        factor = LLMSentimentFactor(cache_dir=str(tmp_path))

        assert set(factor._cache) == {"good"}
        assert factor._cache["good"]["score"] == pytest.approx(0.5)

    def test_poisoned_in_memory_entry_is_not_served(self, tmp_path):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        analyzer = _CountingAnalyzer(score=-0.2)
        factor.analyzer = analyzer
        date = pd.Timestamp("2024-01-02")
        key = factor._make_cache_key("公告", "600000.SH", date)
        factor._cache[key] = {"score": 123.45, "ts": time.time(), "retryable": False}

        score = factor._analyze_headline("公告", "600000.SH", date)

        assert score == pytest.approx(-0.2)
        assert len(analyzer.calls) == 1
        assert factor._cache[key]["score"] == pytest.approx(-0.2)

    def test_save_refuses_invalid_scores(self, tmp_path):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        now = time.time()
        factor._cache["high"] = {"score": 5.0, "ts": now, "retryable": False}
        factor._cache["nan"] = {"score": float("nan"), "ts": now, "retryable": False}
        factor._cache[123] = {"score": 0.5, "ts": now, "retryable": False}

        factor._save_cache()

        payload = json.loads(
            (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8")
        )
        assert payload["entries"] == {}

    def test_invalid_analyzer_output_is_not_cached(self, tmp_path):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path))
        factor.analyzer = _CountingAnalyzer(score=42.0)

        score = factor._analyze_headline(
            "公告", "600000.SH", pd.Timestamp("2024-01-02"),
        )

        assert score == 0.0
        entry = next(iter(factor._cache.values()))
        assert entry["retryable"] is True

    def test_oversized_cache_file_is_ignored(self, tmp_path):
        (tmp_path / "sentiment_cache.json").write_text(
            "{}" + " " * 100, encoding="utf-8",
        )

        factor = LLMSentimentFactor(
            cache_dir=str(tmp_path), max_cache_file_bytes=10,
        )

        assert factor._cache == {}

    def test_cache_evicts_oldest_entries_at_capacity(self, tmp_path):
        factor = LLMSentimentFactor(
            cache_dir=str(tmp_path), max_cache_entries=10,
        )
        now = time.time()
        for index in range(9):
            factor._cache[f"k{index}"] = {
                "score": 0.1, "ts": now - index, "retryable": False,
            }
        factor.analyzer = _CountingAnalyzer(score=0.5)
        date = pd.Timestamp("2024-01-02")

        factor._analyze_headline("公告一", "600000.SH", date)
        factor._analyze_headline("公告二", "600000.SH", date)

        assert len(factor._cache) <= 10
        assert "k8" not in factor._cache, "oldest entry should be evicted"

    def test_expired_entries_are_not_persisted(self, tmp_path):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path), cache_ttl_days=1)
        factor._cache["stale"] = {
            "score": 0.5, "ts": time.time() - 2 * 86400, "retryable": False,
        }

        factor._save_cache()

        payload = json.loads(
            (tmp_path / "sentiment_cache.json").read_text(encoding="utf-8")
        )
        assert payload["entries"] == {}

    def test_load_respects_max_cache_entries(self, tmp_path):
        now = time.time()
        version = LLMSentimentFactor._CACHE_VERSION
        self._write_entries(tmp_path, {
            f"k{index}": {"version": version, "score": 0.5, "ts": now - index}
            for index in range(20)
        })

        factor = LLMSentimentFactor(cache_dir=str(tmp_path), max_cache_entries=5)

        assert len(factor._cache) == 5
        assert set(factor._cache) == {f"k{index}" for index in range(5)}


class TestRunLevelFallbackVisibility:
    """A real-LLM run whose API calls failed must not claim LLM output."""

    class _FailingLLM(OpenAISentimentAnalyzer):
        def __init__(self):
            super().__init__("sk-test")

        def analyze(self, headline, asset=""):
            self.last_analyze_fallback = True
            self.last_fallback_error = "api down"
            return 0.1

    def test_failed_llm_calls_are_counted_and_surfaced(
        self, tmp_path, small_prices, caplog,
    ):
        caplog.set_level(logging.WARNING)
        factor = LLMSentimentFactor(lookback_days=1, cache_dir=str(tmp_path))
        factor.analyzer = self._FailingLLM()

        factor.compute(
            small_prices,
            news={"600000.SH": {small_prices.index[0]: ["公告"]}},
        )

        assert factor.is_llm is True
        assert factor.backend == "llm:deepseek-chat"
        assert factor.n_analyzer_calls == 1
        assert factor.n_fallback_scores == 1
        assert factor.last_fallback_reason == "api down"
        assert factor.params["n_fallback_scores"] == 1
        assert factor.params["fallback_reason"] == "api down"
        assert any(
            "partially degraded" in record.getMessage()
            for record in caplog.records
        )


# ----------------------------------------------------------------------
# R15: TTL and keyword de-duplication
# ----------------------------------------------------------------------

class TestCacheTTL:
    def test_expired_entry_is_recomputed(self, tmp_path):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path), cache_ttl_days=1)
        analyzer = _CountingAnalyzer(score=-0.2)
        factor.analyzer = analyzer
        date = pd.Timestamp("2024-01-02")
        key = factor._make_cache_key("公告", "600000.SH", date)
        factor._cache[key] = {
            "score": 0.9, "ts": time.time() - 2 * 86400, "retryable": False,
        }

        score = factor._analyze_headline("公告", "600000.SH", date)

        assert score == pytest.approx(-0.2)
        assert len(analyzer.calls) == 1

    def test_ttl_none_disables_expiry(self, tmp_path):
        factor = LLMSentimentFactor(cache_dir=str(tmp_path), cache_ttl_days=None)
        analyzer = _CountingAnalyzer(score=-0.2)
        factor.analyzer = analyzer
        date = pd.Timestamp("2024-01-02")
        key = factor._make_cache_key("公告", "600000.SH", date)
        factor._cache[key] = {
            "score": 0.9, "ts": time.time() - 999 * 86400, "retryable": False,
        }

        score = factor._analyze_headline("公告", "600000.SH", date)

        assert score == pytest.approx(0.9)
        assert analyzer.calls == []


class TestKeywordScoring:
    def test_keyword_lists_are_deduplicated(self):
        for keywords in (
            KeywordSentimentAnalyzer.BULLISH_KEYWORDS,
            KeywordSentimentAnalyzer.BEARISH_KEYWORDS,
        ):
            assert len(keywords) == len(set(keywords))

    def test_mixed_headline_is_not_double_counted(self):
        score = KeywordSentimentAnalyzer().analyze(
            "净利润增长，控股股东增持，大股东减持"
        )
        # 2 bullish hits / 3 total hits; the old duplicate "增持" entry scored 0.2.
        assert score == pytest.approx(1.0 / 3.0, abs=1e-4)
