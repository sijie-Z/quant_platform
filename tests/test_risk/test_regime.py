"""Tests for risk.regime — Market regime detection."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.risk.regime import (
    VolatilityRegimeDetector, TrendRegimeDetector,
    CorrelationRegimeDetector, CompositeRegimeDetector, RegimeType,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.02, 500), index=dates)


@pytest.fixture
def sample_prices(sample_returns):
    return (1 + sample_returns).cumprod() * 100


@pytest.fixture
def sample_returns_matrix():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    data = np.random.normal(0.0005, 0.02, (500, 20))
    return pd.DataFrame(data, index=dates, columns=[f"stock_{i}" for i in range(20)])


class TestVolatilityDetector:
    def test_detect_returns_regime(self, sample_returns):
        detector = VolatilityRegimeDetector(lookback=252, vol_window=21)
        result = detector.detect(sample_returns)
        assert result["regime"] in [
            RegimeType.LOW_VOL, RegimeType.MEDIUM_VOL,
            RegimeType.HIGH_VOL, RegimeType.EXTREME_VOL,
        ]
        assert "percentile" in result
        assert "confidence" in result

    def test_insufficient_data(self):
        detector = VolatilityRegimeDetector(lookback=252)
        short_returns = pd.Series(np.random.normal(0, 0.02, 50))
        result = detector.detect(short_returns)
        assert result["regime"] == RegimeType.MEDIUM_VOL

    def test_percentile_bounds(self, sample_returns):
        detector = VolatilityRegimeDetector(lookback=252, vol_window=21)
        result = detector.detect(sample_returns)
        assert 0 <= result["percentile"] <= 1


class TestTrendDetector:
    def test_detect_trend(self, sample_prices):
        detector = TrendRegimeDetector(short_window=50, long_window=200)
        result = detector.detect(sample_prices)
        assert result["regime"] in [RegimeType.BULL, RegimeType.BEAR, RegimeType.SIDEWAYS]
        assert "confidence" in result

    def test_insufficient_data(self):
        detector = TrendRegimeDetector(short_window=50, long_window=200)
        short_prices = pd.Series(range(50))
        result = detector.detect(short_prices)
        assert result["regime"] == RegimeType.SIDEWAYS


class TestCorrelationDetector:
    def test_detect_correlation(self, sample_returns_matrix):
        detector = CorrelationRegimeDetector(lookback=63, threshold=0.5)
        result = detector.detect(sample_returns_matrix)
        assert result["regime"] in [RegimeType.NORMAL, RegimeType.STRESSED]
        assert "avg_correlation" in result

    def test_insufficient_data(self):
        detector = CorrelationRegimeDetector(lookback=63)
        short = pd.DataFrame(np.random.randn(10, 20))
        result = detector.detect(short)
        assert result["regime"] == RegimeType.NORMAL


class TestCompositeDetector:
    def test_composite_detect(self, sample_returns, sample_prices):
        detector = CompositeRegimeDetector()
        result = detector.detect(sample_returns, sample_prices)
        assert "overall_regime" in result
        assert result["overall_regime"] in ["risk_on", "neutral", "cautious", "risk_off"]
        assert "composite_risk_score" in result
        assert "volatility" in result
        assert "trend" in result

    def test_composite_with_correlation(self, sample_returns, sample_prices, sample_returns_matrix):
        detector = CompositeRegimeDetector()
        result = detector.detect(sample_returns, sample_prices, sample_returns_matrix)
        assert "correlation" in result
