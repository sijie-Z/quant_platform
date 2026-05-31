"""Tests for the Lookahead Bias Detector (inspired by freqtrade)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.risk.lookahead_detector import LookaheadDetector


@pytest.fixture
def sample_data():
    """Small synthetic dataset for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    assets = [f"A{i:04d}" for i in range(5)]
    prices = 100 * (1 + np.random.randn(60, 5) * 0.02).cumprod(axis=0)
    prices = pd.DataFrame(prices, index=dates, columns=assets)
    returns = prices.pct_change(fill_method=None)
    return prices, returns


class TestLookaheadDetector:
    def test_init(self):
        d = LookaheadDetector()
        assert d.threshold == 1e-4
        assert d.max_check_dates == 50

    def test_init_custom(self):
        d = LookaheadDetector(threshold=0.01, max_check_dates=10)
        assert d.threshold == 0.01
        assert d.max_check_dates == 10

    def test_select_dates_limits(self, sample_data):
        prices, _ = sample_data
        d = LookaheadDetector(max_check_dates=10)
        dates = d._select_dates(prices)
        assert len(dates) <= 10

    def test_report_no_bias(self):
        d = LookaheadDetector()
        result = {
            "has_bias": False,
            "dates_checked": 20,
            "biased_dates": [],
            "max_signal_diff": 0.0,
            "threshold": 1e-4,
            "signal_diff_summary": pd.DataFrame(),
            "factor_bias_report": {},
        }
        d.print_report(result)
        assert d.suggest_fixes(result) == ["No bias detected — no fixes needed."]
