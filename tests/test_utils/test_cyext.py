"""Tests for Cython-accelerated hot paths (with Python fallback)."""

import numpy as np
import pytest

from quant_platform.utils.cyext import (
    HAS_CYTHON,
    benchmark_cython_speedup,
    rank_ic,
    rank_ic_py,
    rolling_max_drawdown,
    rolling_max_drawdown_py,
    rolling_momentum,
    rolling_momentum_py,
    rolling_volatility,
    rolling_volatility_py,
    zscore_cross_section,
    zscore_cross_section_py,
)
from quant_platform.utils.cyext._fast_rank import batch_rank_ic_cy
from quant_platform.utils.cyext._fast_zscore import winsorize_cy, zscore_panel_cy


# ── Rolling Momentum Tests ──


class TestRollingMomentum:
    def test_shape_preserved(self):
        prices = np.random.lognormal(4, 0.3, (100, 10))
        result = rolling_momentum(prices, period=20)
        assert result.shape == prices.shape

    def test_nan_for_insufficient_history(self):
        prices = np.random.lognormal(4, 0.3, (30, 5))
        result = rolling_momentum(prices, period=20)
        # First 20 rows should be NaN
        assert np.all(np.isnan(result[:20]))

    def test_values_after_period(self):
        prices = np.random.lognormal(4, 0.3, (50, 5))
        result = rolling_momentum(prices, period=20)
        # After period, should have some non-NaN values
        valid = result[20:]
        assert np.sum(~np.isnan(valid)) > 0

    def test_log_return_correctness(self):
        prices = np.array([[100, 200], [110, 220], [121, 242]], dtype=np.float64)
        result = rolling_momentum_py(prices, period=2)
        # Row 2: log(121/100) = log(1.21), log(242/200) = log(1.21)
        assert abs(result[2, 0] - np.log(1.21)) < 1e-10
        assert abs(result[2, 1] - np.log(1.21)) < 1e-10


# ── Rolling Volatility Tests ──


class TestRollingVolatility:
    def test_shape_preserved(self):
        returns = np.random.randn(100, 10) * 0.02
        result = rolling_volatility(returns, period=20)
        assert result.shape == returns.shape

    def test_positive_values(self):
        returns = np.random.randn(100, 5) * 0.02
        result = rolling_volatility(returns, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_constant_returns_zero_vol(self):
        returns = np.full((50, 3), 0.01)
        result = rolling_volatility(returns, period=20)
        # Constant returns → zero volatility
        valid = result[20:]
        assert np.allclose(valid[~np.isnan(valid)], 0, atol=1e-10)


# ── Rolling Max Drawdown Tests ──


class TestRollingMaxDrawdown:
    def test_shape_preserved(self):
        equity = np.cumsum(np.random.randn(100)) + 1000
        result = rolling_max_drawdown(equity, period=20)
        assert len(result) == len(equity)

    def test_negative_values(self):
        equity = np.array([100, 110, 105, 95, 100, 108])
        result = rolling_max_drawdown(equity, period=3)
        valid = result[~np.isnan(result)]
        assert np.all(valid <= 0)

    def test_monotonic_increase_zero_dd(self):
        equity = np.arange(100, dtype=float)
        result = rolling_max_drawdown(equity, period=10)
        valid = result[10:]
        assert np.allclose(valid, 0)


# ── Rank IC Tests ──


class TestRankIC:
    def test_perfect_correlation(self):
        factor = np.arange(100, dtype=float)
        returns = np.arange(100, dtype=float)
        ic = rank_ic(factor, returns)
        assert abs(ic - 1.0) < 0.01

    def test_negative_correlation(self):
        factor = np.arange(100, dtype=float)
        returns = np.arange(100, 0, -1, dtype=float)
        ic = rank_ic(factor, returns)
        assert abs(ic - (-1.0)) < 0.01

    def test_random_near_zero(self):
        np.random.seed(42)
        factor = np.random.randn(1000)
        returns = np.random.randn(1000)
        ic = rank_ic(factor, returns)
        assert abs(ic) < 0.1

    def test_with_nans(self):
        factor = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        returns = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10, 11, 12])
        ic = rank_ic(factor, returns)
        assert not np.isnan(ic)


# ── Z-Score Tests ──


class TestZScore:
    def test_mean_zero(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
        result = zscore_cross_section(values)
        valid = result[~np.isnan(result)]
        assert abs(np.mean(valid)) < 1e-10

    def test_std_near_one(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
        result = zscore_cross_section(values)
        valid = result[~np.isnan(result)]
        assert abs(np.std(valid, ddof=1) - 1.0) < 0.1

    def test_with_nans(self):
        values = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        result = zscore_cross_section(values)
        assert np.isnan(result[2])  # NaN preserved
        assert not np.isnan(result[0])

    def test_constant_values_zero(self):
        values = np.full(20, 5.0)
        result = zscore_cross_section(values)
        assert np.allclose(result, 0)


# ── Panel Z-Score Tests ──


class TestZScorePanel:
    def test_panel_shape(self):
        panel = np.random.randn(10, 50)
        result = zscore_panel_cy(panel)
        assert result.shape == panel.shape

    def test_each_row_normalized(self):
        panel = np.random.randn(10, 50) * 100
        result = zscore_panel_cy(panel)
        for i in range(10):
            row = result[i]
            valid = row[~np.isnan(row)]
            if len(valid) > 2:
                assert abs(np.mean(valid)) < 1e-8


# ── Winsorize Tests ──


class TestWinsorize:
    def test_clips_outliers(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
        result = winsorize_cy(values, lower=0.05, upper=0.95)
        assert result[-1] < 100  # Outlier clipped

    def test_preserves_shape(self):
        values = np.random.randn(50)
        result = winsorize_cy(values)
        assert result.shape == values.shape


# ── Benchmark ──


class TestBenchmark:
    def test_benchmark_runs(self):
        result = benchmark_cython_speedup(n=100, n_assets=50, period=10)
        assert "rolling_momentum" in result
        assert "rolling_volatility" in result
        assert "rank_ic" in result
        assert result["has_cython"] in (True, False)

    def test_python_fallback_works(self):
        """Ensure Python fallback produces same results as auto-select."""
        prices = np.random.lognormal(4, 0.3, (50, 10))
        result_auto = rolling_momentum(prices, 20)
        result_py = rolling_momentum_py(prices, 20)
        np.testing.assert_array_equal(result_auto, result_py)
