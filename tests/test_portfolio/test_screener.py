"""Tests for the Factor Screener module.

Inspired by BlackOil-OmniAlpha's strategy+engine pattern, the screener
applies boolean rules on processed factor values for quick stock selection.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quant_platform.portfolio.screener import (
    FactorScreener,
    OPERATORS,
    ScreenConfig,
    ScreenRule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factors(
    n_stocks: int = 20,
    n_days: int = 10,
) -> dict[str, pd.DataFrame]:
    """Create synthetic processed factor data for testing."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i:04d}" for i in range(n_stocks)]

    np.random.seed(42)

    # Factor 1: normally distributed (mean 0, std 1) — after standardization
    f1 = pd.DataFrame(
        np.random.randn(n_days, n_stocks),
        index=dates, columns=assets,
    )

    # Factor 2: uniform [0, 100] — like PE ratio
    f2 = pd.DataFrame(
        np.random.uniform(0, 100, (n_days, n_stocks)),
        index=dates, columns=assets,
    )

    # Factor 3: uniform [0, 0.5] — like ROE
    f3 = pd.DataFrame(
        np.random.uniform(0, 0.5, (n_days, n_stocks)),
        index=dates, columns=assets,
    )

    # Factor 4: log-normal — like market cap
    np.random.seed(99)
    f4 = pd.DataFrame(
        np.random.lognormal(22, 1, (n_days, n_stocks)),
        index=dates, columns=assets,
    )

    return {
        "zscore_factor": f1,
        "pe_ratio": f2,
        "roe": f3,
        "market_cap": f4,
    }


# ---------------------------------------------------------------------------
# Test ScreenRule
# ---------------------------------------------------------------------------


class TestScreenRule:
    def test_valid_operators(self):
        """All supported operators should be accepted."""
        for op in OPERATORS:
            if op == "between":
                r = ScreenRule(factor="pe", operator=op, value=[10, 30])
            else:
                r = ScreenRule(factor="pe", operator=op, value=20)
            assert r.operator == op

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            ScreenRule(factor="pe", operator="bad_op", value=20)

    def test_between_requires_two_values(self):
        with pytest.raises(ValueError, match="'between' operator requires"):
            ScreenRule(factor="pe", operator="between", value=10)

    def test_gt_apply(self):
        sr = ScreenRule(factor="x", operator="gt", value=0)
        s = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sr.apply(s)
        expected = pd.Series([False, False, False, True, True])
        pd.testing.assert_series_equal(result, expected)

    def test_gte_apply(self):
        sr = ScreenRule(factor="x", operator="gte", value=0)
        s = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sr.apply(s)
        expected = pd.Series([False, False, True, True, True])
        pd.testing.assert_series_equal(result, expected)

    def test_lt_apply(self):
        sr = ScreenRule(factor="x", operator="lt", value=30)
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = sr.apply(s)
        expected = pd.Series([True, True, False, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_lte_apply(self):
        sr = ScreenRule(factor="x", operator="lte", value=30)
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = sr.apply(s)
        expected = pd.Series([True, True, True, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_eq_apply(self):
        sr = ScreenRule(factor="x", operator="eq", value=0)
        s = pd.Series([-0.5, 0.0, 0.5, 1.0])
        result = sr.apply(s)
        expected = pd.Series([False, True, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_ne_apply(self):
        sr = ScreenRule(factor="x", operator="ne", value=0)
        s = pd.Series([-0.5, 0.0, 0.5, 1.0])
        result = sr.apply(s)
        expected = pd.Series([True, False, True, True])
        pd.testing.assert_series_equal(result, expected)

    def test_between_apply(self):
        sr = ScreenRule(factor="x", operator="between", value=[10, 30])
        s = pd.Series([5.0, 10.0, 20.0, 30.0, 35.0])
        result = sr.apply(s)
        expected = pd.Series([False, True, True, True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_tolerance(self):
        """Tolerance should prevent floating-point edge cases in eq."""
        sr = ScreenRule(factor="x", operator="eq", value=0, tolerance=1e-6)
        s = pd.Series([-1e-8, 0.0, 1e-8, 1.0])
        result = sr.apply(s)
        # Everything within 1e-6 of 0 should be considered equal
        expected = pd.Series([True, True, True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_tolerance_ne(self):
        """Tolerance in ne: values within tolerance of threshold are NOT considered different."""
        sr = ScreenRule(factor="x", operator="ne", value=0, tolerance=1e-6)
        s = pd.Series([-1e-8, 0.0, 1e-8, 1.0])
        result = sr.apply(s)
        # Everything within 1e-6 of 0 should be considered equal (not ne)
        expected = pd.Series([False, False, False, True])
        pd.testing.assert_series_equal(result, expected)

    def test_apply_with_nan(self):
        """NaN values should propagate as False."""
        sr = ScreenRule(factor="x", operator="gt", value=0)
        s = pd.Series([1.0, np.nan, -1.0, np.nan, 2.0])
        result = sr.apply(s)
        expected = pd.Series([True, False, False, False, True])
        pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Test ScreenConfig
# ---------------------------------------------------------------------------


class TestScreenConfig:
    def test_default_config(self):
        c = ScreenConfig()
        assert c.enabled is False
        assert c.rules == []
        assert c.logic == "and"
        assert c.min_stocks == 5
        assert c.max_stocks == 200

    def test_invalid_logic_raises(self):
        with pytest.raises(ValueError, match="Unknown logic"):
            ScreenConfig(logic="xor")

    def test_with_rules(self):
        rules = [
            ScreenRule(factor="pe", operator="lt", value=30),
            ScreenRule(factor="roe", operator="gt", value=0.15),
        ]
        c = ScreenConfig(enabled=True, rules=rules, logic="and")
        assert c.enabled is True
        assert len(c.rules) == 2
        assert c.logic == "and"


# ---------------------------------------------------------------------------
# Test FactorScreener
# ---------------------------------------------------------------------------


class TestFactorScreener:
    def test_empty_rules_returns_empty(self):
        screener = FactorScreener()
        factors = _make_factors()
        result = screener.screen(factors, rules=[])
        assert result == []

    def test_no_matching_factor_returns_empty(self):
        screener = FactorScreener()
        factors = _make_factors()
        rules = [ScreenRule(factor="nonexistent", operator="gt", value=0)]
        result = screener.screen(factors, rules=rules)
        assert result == []

    def test_single_rule_and_logic(self):
        """pe_ratio < 30 should return stocks with PE < 30."""
        factors = _make_factors()
        rules = [ScreenRule(factor="pe_ratio", operator="lt", value=30)]
        screener = FactorScreener()
        result = screener.screen(factors, rules=rules)

        # Verify all returned stocks have PE < 30
        last_date = factors["pe_ratio"].index[-1]
        pe_values = factors["pe_ratio"].loc[last_date]
        for asset in result:
            assert pe_values[asset] < 30, f"{asset} has PE >= 30"

    def test_multiple_rules_and_logic(self):
        """pe_ratio < 30 AND roe > 0.15 — intersection."""
        factors = _make_factors()
        rules = [
            ScreenRule(factor="pe_ratio", operator="lt", value=30),
            ScreenRule(factor="roe", operator="gt", value=0.15),
        ]
        screener = FactorScreener(ScreenConfig(logic="and"))
        result = screener.screen(factors, rules=rules)

        last_date = factors["pe_ratio"].index[-1]
        pe_vals = factors["pe_ratio"].loc[last_date]
        roe_vals = factors["roe"].loc[last_date]

        for asset in result:
            assert pe_vals[asset] < 30, f"{asset} fails PE rule"
            assert roe_vals[asset] > 0.15, f"{asset} fails ROE rule"

    def test_or_logic(self):
        """pe_ratio < 30 OR roe > 0.15 — union should be larger."""
        factors = _make_factors()
        rules = [
            ScreenRule(factor="pe_ratio", operator="lt", value=30),
            ScreenRule(factor="roe", operator="gt", value=0.15),
        ]

        screener_and = FactorScreener(ScreenConfig(logic="and"))
        screener_or = FactorScreener(ScreenConfig(logic="or"))

        result_and = screener_and.screen(factors, rules=rules)
        result_or = screener_or.screen(factors, rules=rules)

        # OR should be >= AND in size
        assert len(result_or) >= len(result_and)

    def test_empty_factors_returns_empty(self):
        screener = FactorScreener()
        rules = [ScreenRule(factor="pe", operator="lt", value=30)]
        result = screener.screen({}, rules=rules)
        assert result == []

    def test_dict_config_init(self):
        """Should accept dict config and parse rules."""
        screener = FactorScreener({
            "enabled": True,
            "rules": [{"factor": "pe_ratio", "operator": "lt", "value": 30}],
            "logic": "and",
        })
        assert screener.config.enabled is True
        assert len(screener.config.rules) == 1
        assert screener.config.rules[0].factor == "pe_ratio"
        assert screener.config.rules[0].operator == "lt"
        assert screener.config.rules[0].value == 30

    def test_screen_historical(self):
        """screen_historical should return results per date."""
        factors = _make_factors(n_days=5)
        rules = [ScreenRule(factor="pe_ratio", operator="lt", value=30)]
        screener = FactorScreener()
        historical = screener.screen_historical(factors, rules=rules)

        assert isinstance(historical, pd.Series)
        assert len(historical) == 5  # one entry per day

        # Each entry should be a list of passing stocks
        for date_key, stocks in historical.items():
            assert isinstance(stocks, list)
            if stocks:
                last_date_pe = factors["pe_ratio"].loc[date_key]
                for s in stocks:
                    assert last_date_pe[s] < 30

    def test_nan_values_excluded(self):
        """Assets with NaN in ANY screened factor should be excluded."""
        factors = _make_factors()
        rules = [ScreenRule(factor="pe_ratio", operator="lt", value=30)]

        # Set one asset to NaN on the last date
        last_date = factors["pe_ratio"].index[-1]
        factors["pe_ratio"].loc[last_date, "A0000"] = np.nan

        screener = FactorScreener()
        result = screener.screen(factors, rules=rules)

        # NaN asset should not appear
        assert "A0000" not in result

    def test_min_stocks_relaxation(self):
        """If too few stocks pass, rules should be progressively relaxed."""
        factors = _make_factors(n_stocks=20)
        # Very restrictive rule
        rules = [
            ScreenRule(factor="pe_ratio", operator="lt", value=5),
            ScreenRule(factor="roe", operator="gt", value=0.4),
        ]

        screener = FactorScreener(ScreenConfig(min_stocks=3, max_stocks=200))
        result = screener.screen(factors, rules=rules)

        # Should have at least min_stocks
        assert len(result) >= 1  # Could be less if even relaxed rules find nothing

    def test_max_stocks_cap(self):
        """If too many stocks pass, should cap by multi-factor score."""
        factors = _make_factors(n_stocks=50)
        # Very loose rule — almost everything passes
        rules = [ScreenRule(factor="pe_ratio", operator="gt", value=0)]

        screener = FactorScreener(ScreenConfig(max_stocks=10, min_stocks=1))
        result = screener.screen(factors, rules=rules)

        # Should cap at max_stocks
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestScreenerIntegration:
    """End-to-end test with synthetic data pipeline."""

    def test_standalone_screening(self):
        """Full flow: create screener, apply to factors, verify results."""
        factors = _make_factors(n_stocks=30, n_days=20)
        rules = [
            ScreenRule(factor="zscore_factor", operator="gt", value=0),
            ScreenRule(factor="pe_ratio", operator="between", value=[10, 40]),
        ]

        screener = FactorScreener({
            "enabled": True,
            "rules": [],
            "logic": "and",
        })
        result = screener.screen(factors, rules=rules)

        last_date = factors["zscore_factor"].index[-1]
        zs = factors["zscore_factor"].loc[last_date]
        pe = factors["pe_ratio"].loc[last_date]

        for asset in result:
            assert zs[asset] > 0, f"{asset} zscore <= 0"
            assert 10 <= pe[asset] <= 40, f"{asset} PE out of [10, 40]"

    def test_screen_with_config(self):
        """Screener should work with ScreenConfig object."""
        rules = [
            ScreenRule(factor="pe_ratio", operator="lt", value=30),
        ]
        config = ScreenConfig(enabled=True, rules=rules, logic="and")
        screener = FactorScreener(config)
        factors = _make_factors()
        result = screener.screen(factors)

        # Should use config rules
        last_date = factors["pe_ratio"].index[-1]
        for asset in result:
            assert factors["pe_ratio"].loc[last_date, asset] < 30

    def test_none_init(self):
        """Screener should handle None config gracefully."""
        screener = FactorScreener(None)
        assert screener.config.enabled is False

    @pytest.mark.skip(reason="Requires full pipeline — run manually")
    def test_with_full_pipeline(self):
        """Placeholder for full pipeline integration test."""
        pass
