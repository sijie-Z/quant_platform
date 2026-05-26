"""Tests for LLM sentiment factor."""

import numpy as np
import pandas as pd

from quant_platform.agent.sentiment_factor import (
    SAMPLE_HEADLINES_POOL,
    KeywordSentimentAnalyzer,
    LLMSentimentFactor,
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
