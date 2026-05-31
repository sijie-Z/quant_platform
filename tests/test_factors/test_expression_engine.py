"""Tests for the expression-based factor engine (inspired by vnpy.alpha)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.factors.expression_engine import (
    DataProxy,
    ExpressionFactor,
    calculate_by_expression,
    register_expression_functions,
)
from quant_platform.factors.expressions import (
    ALL_FUNCTIONS,
    register_expression_functions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """10 assets × 100 days of synthetic price data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    assets = [f"A{i:04d}" for i in range(10)]
    # Random walk starting at 100
    returns = np.random.randn(100, 10) * 0.02
    prices = 100 * (1 + returns).cumprod(axis=0)
    return pd.DataFrame(prices, index=dates, columns=assets)


# ---------------------------------------------------------------------------
# Test DataProxy
# ---------------------------------------------------------------------------


class TestDataProxy:
    def test_add(self, sample_prices):
        a = DataProxy(sample_prices)
        b = DataProxy(sample_prices)
        c = (a + b).df
        pd.testing.assert_frame_equal(c, sample_prices * 2)

    def test_sub(self, sample_prices):
        a = DataProxy(sample_prices)
        b = DataProxy(sample_prices)
        c = (a - b).df
        pd.testing.assert_frame_equal(c, sample_prices * 0)

    def test_mul(self, sample_prices):
        a = DataProxy(sample_prices)
        b = DataProxy(sample_prices)
        c = (a * b).df
        pd.testing.assert_frame_equal(c, sample_prices ** 2)

    def test_div(self, sample_prices):
        a = DataProxy(sample_prices)
        b = DataProxy(sample_prices)
        c = (a / b).df
        pd.testing.assert_frame_equal(c, pd.DataFrame(1.0, index=sample_prices.index, columns=sample_prices.columns))

    def test_add_scalar(self, sample_prices):
        a = DataProxy(sample_prices)
        c = (a + 10).df
        pd.testing.assert_frame_equal(c, sample_prices + 10)

    def test_gt(self, sample_prices):
        """Comparison returns 0/1 int DataFrame."""
        a = DataProxy(sample_prices)
        mid = sample_prices.mean().mean()
        result = (a > mid).df
        assert result.dtypes.iloc[0] == np.dtype("float64")
        assert set(result.values.flatten()) <= {0.0, 1.0}

    def test_abs(self, sample_prices):
        neg_prices = -sample_prices
        a = DataProxy(neg_prices)
        result = abs(a).df
        pd.testing.assert_frame_equal(result, sample_prices.abs())


# ---------------------------------------------------------------------------
# Test expression evaluation
# ---------------------------------------------------------------------------


class TestExpressionEvaluation:
    def test_simple_expression(self, sample_prices):
        """close_pct * 2 should work."""
        data = {"close": sample_prices, "close_pct": sample_prices.pct_change(fill_method=None)}
        result = calculate_by_expression(data, "close_pct * 2")
        expected = sample_prices.pct_change(fill_method=None) * 2
        pd.testing.assert_frame_equal(result, expected)

    def test_nested_expression(self, sample_prices):
        """close / ts_delay(close, 1) - 1 should compute daily returns."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "close / ts_delay(close, 1) - 1")
        expected = sample_prices.pct_change(fill_method=None)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ts_mean(self, sample_prices):
        """ts_mean(close, 5) should compute 5-day moving average."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "ts_mean(close, 5)")
        expected = sample_prices.T.rolling(5, axis=1).mean().T
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ts_sum(self, sample_prices):
        """ts_sum(close_pct, 21) should compute 21-day cumulative return."""
        data = {"close": sample_prices, "close_pct": sample_prices.pct_change(fill_method=None)}
        result = calculate_by_expression(data, "ts_sum(close_pct, 21)")
        expected = sample_prices.pct_change(fill_method=None).T.rolling(21, axis=1).sum().T
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_ts_std(self, sample_prices):
        """ts_std(close_pct, 20) should compute 20-day volatility."""
        data = {"close_pct": sample_prices.pct_change(fill_method=None)}
        result = calculate_by_expression(data, "ts_std(close_pct, 20)")
        expected = sample_prices.pct_change(fill_method=None).T.rolling(20, axis=1).std(ddof=0).T
        pd.testing.assert_frame_equal(result, expected, rtol=1e-8)

    def test_ts_delay(self, sample_prices):
        """ts_delay(close, 1) should shift by 1 day."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "ts_delay(close, 1)")
        expected = sample_prices.shift(1)
        pd.testing.assert_frame_equal(result, expected)

    def test_ts_delta(self, sample_prices):
        """ts_delta(close, 1) should be daily change."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "ts_delta(close, 1)")
        expected = sample_prices.diff(1)
        pd.testing.assert_frame_equal(result, expected)

    def test_ts_min_max(self, sample_prices):
        """ts_min and ts_max should work."""
        data = {"close": sample_prices}
        result_min = calculate_by_expression(data, "ts_min(close, 5)")
        result_max = calculate_by_expression(data, "ts_max(close, 5)")
        expected_min = sample_prices.T.rolling(5, axis=1).min().T
        expected_max = sample_prices.T.rolling(5, axis=1).max().T
        pd.testing.assert_frame_equal(result_min, expected_min, rtol=1e-10)
        pd.testing.assert_frame_equal(result_max, expected_max, rtol=1e-10)

    def test_complex_expression(self, sample_prices):
        """Complex alpha-like expression."""
        data = {"close": sample_prices, "close_pct": sample_prices.pct_change(fill_method=None)}
        # Rank of (5-day return / 20-day volatility) over past 10 days
        expr = "ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)"
        result = calculate_by_expression(data, expr)
        assert result.shape == sample_prices.shape
        assert not result.isna().all().all()  # Should have some valid values after burn-in

    def test_cs_rank(self, sample_prices):
        """Cross-sectional rank should rank across assets per date."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "cs_rank(close)")
        expected = sample_prices.rank(axis=1, pct=True)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_cs_zscore(self, sample_prices):
        """Cross-sectional z-score."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "cs_zscore(close)")
        # Per date, mean should be ~0
        date_means = result.mean(axis=1)
        assert date_means.abs().max() < 1e-10

    def test_invalid_expression_raises(self, sample_prices):
        data = {"close": sample_prices}
        with pytest.raises(ValueError, match="Expression evaluation failed"):
            calculate_by_expression(data, "nonexistent_function(close)")

    def test_unknown_column_raises(self, sample_prices):
        data = {"close": sample_prices}
        with pytest.raises(ValueError):
            calculate_by_expression(data, "unknown_column + 1")

    def test_log(self, sample_prices):
        """log(close) should compute natural log."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "log(close)")
        expected = np.log(sample_prices)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-10)

    def test_if_else(self, sample_prices):
        """if_else(close > 100, 1, 0) should return 1 where close > 100."""
        data = {"close": sample_prices}
        result = calculate_by_expression(data, "if_else(close > 100, 1, 0)")
        expected = (sample_prices > 100).astype(float)
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Test ExpressionFactor
# ---------------------------------------------------------------------------


class TestExpressionFactor:
    def test_basic_compute(self, sample_prices):
        factor = ExpressionFactor(
            name="test_momentum",
            expression="ts_sum(close_pct, 21)",
        )
        result = factor.compute(sample_prices)
        assert result.shape == sample_prices.shape
        assert factor.name == "test_momentum"
        assert factor.expression == "ts_sum(close_pct, 21)"

    def test_complex_factor(self, sample_prices):
        """ExpressionFactor with complex WorldQuant-style alpha."""
        factor = ExpressionFactor(
            name="alpha_001",
            expression="ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)",
        )
        result = factor.compute(sample_prices)
        assert result.shape == sample_prices.shape
        # After burn-in, should have non-NaN values
        last_valid = result.iloc[-1].dropna()
        assert len(last_valid) > 0

    def test_factor_with_params(self, sample_prices):
        factor = ExpressionFactor(
            name="custom_vol",
            expression="ts_std(close_pct, 20)",
            params={"period": 20},
        )
        result = factor.compute(sample_prices)
        assert result.shape == sample_prices.shape

    def test_expression_factor_with_financials(self, sample_prices):
        """Expression factor using financials columns."""
        fin = pd.DataFrame(
            np.random.randn(100, 10),
            index=sample_prices.index,
            columns=sample_prices.columns,
        )
        fin.name = "pe_ratio"
        expression = "cs_rank(pe_ratio)"
        factor = ExpressionFactor(name="rank_pe", expression=expression)
        result = factor.compute(sample_prices, financials=fin)
        assert result.shape == sample_prices.shape


# ---------------------------------------------------------------------------
# Test expression from config
# ---------------------------------------------------------------------------


def test_create_expression_factors(sample_prices):
    """create_expression_factors should produce callable factors."""
    from quant_platform.factors.expression_engine import create_expression_factors

    config = {
        "my_momentum": "ts_sum(close_pct, 21)",
        "my_volatility": "ts_std(close_pct, 20)",
    }
    factors = create_expression_factors(config)
    assert len(factors) == 2
    assert factors[0].name == "my_momentum"

    results = [f.compute(sample_prices) for f in factors]
    assert results[0].shape == sample_prices.shape
    assert results[1].shape == sample_prices.shape


def test_expression_factor_matches_classical(sample_prices):
    """Expression-based ts_std should match classical Volatility20D."""
    from quant_platform.factors.technical import Volatility20D

    # Expression approach: ts_std(close_pct, 20)
    expr_factor = ExpressionFactor(
        name="expr_vol",
        expression="ts_std(close_pct, 20)",
    )
    expr_result = expr_factor.compute(sample_prices)

    # Classical approach
    classical = Volatility20D()
    classical_result = classical.compute(sample_prices)

    # ts_std uses ddof=0 (population), Volatility20D uses ddof=1 (sample)
    # So they won't match exactly but should be very close
    mask = classical_result.notna() & expr_result.notna()
    diff = (expr_result.where(mask) - classical_result.where(mask)).abs()
    assert diff.max().max() < 1e-3
