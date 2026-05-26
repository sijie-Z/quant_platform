"""Tests for VaR and CVaR calculations."""

import numpy as np
import pandas as pd
from quant_platform.risk.var import (
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_var,
    var_summary,
)


def _make_returns(n_days=500):
    np.random.seed(42)
    # Fat-tailed returns: t-distribution
    daily_ret = 0.0004 + 0.015 * np.random.standard_t(df=5, size=n_days)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    return pd.Series(daily_ret, index=dates)


def test_historical_var_positive():
    sr = _make_returns()
    var1 = historical_var(sr, confidence=0.95)
    # VaR should be positive (represents a loss)
    assert var1 > 0


def test_var_increases_with_confidence():
    sr = _make_returns()
    var95 = historical_var(sr, confidence=0.95)
    var99 = historical_var(sr, confidence=0.99)
    assert var99 > var95  # Higher confidence = larger VaR


def test_parametric_var():
    sr = _make_returns()
    var1 = parametric_var(sr, confidence=0.95)
    assert var1 > 0


def test_monte_carlo_var():
    sr = _make_returns()
    var1 = monte_carlo_var(sr, confidence=0.95, n_simulations=10_000)
    assert var1 > 0


def test_cvar_greater_than_var():
    sr = _make_returns()
    var95 = historical_var(sr, confidence=0.95)
    cvar95 = historical_cvar(sr, confidence=0.95)
    # CVaR should be >= VaR (it's the mean beyond VaR)
    assert cvar95 >= var95


def test_var_summary():
    sr = _make_returns()
    summary = var_summary(sr, confidence=0.95)
    assert "historical_var" in summary
    assert "parametric_var" in summary
    assert "monte_carlo_var" in summary
    assert "historical_cvar" in summary


def test_var_horizon_scaling():
    sr = _make_returns()
    var1d = historical_var(sr, horizon=1)
    var5d = historical_var(sr, horizon=5)
    # Multi-day VaR scales with sqrt(horizon)
    assert 2.0 < var5d / var1d < 2.5  # sqrt(5) ~= 2.236
