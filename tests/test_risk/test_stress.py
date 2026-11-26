"""Stress scenarios must be shocks, not random walks that can end positive."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.risk.stress import run_stress_test


def _zero_vol_returns(n: int = 252) -> pd.Series:
    return pd.Series(np.zeros(n), index=pd.bdate_range("2023-01-02", periods=n))


def test_zero_vol_path_equals_the_declared_cumulative_shock():
    result = run_stress_test(_zero_vol_returns(), "2015_ashare_crash")

    assert result["estimated_cumulative_return"] == pytest.approx(-0.08, abs=1e-10)


def test_crash_scenarios_never_end_positive():
    returns = pd.Series(
        np.random.default_rng(0).normal(0, 0.01, 252),
        index=pd.bdate_range("2023-01-02", periods=252),
    )

    for scenario in ("2008_financial_crisis", "2015_ashare_crash", "2020_covid_crash"):
        result = run_stress_test(returns, scenario)
        assert result["estimated_cumulative_return"] < 0
        assert result["worst_cumulative_return"] <= result["estimated_cumulative_return"]
        assert result["estimated_max_drawdown"] <= 0
        assert result["worst_max_drawdown"] <= result["estimated_max_drawdown"]


def test_drawdown_includes_the_starting_peak():
    result = run_stress_test(_zero_vol_returns(), "2015_ashare_crash")

    # Constant -8%/30 days still has a path drawdown, not zero.
    assert result["estimated_max_drawdown"] < 0
