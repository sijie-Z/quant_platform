"""Pure-function tests for the first real research runner (no network)."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.lab.runs.first_real_research_80d_reversal import (
    backtest,
    cross_sectional_ic,
    ic_stats,
    performance,
)


def _panel(n_dates=24, n_assets=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    factor = pd.DataFrame(
        rng.normal(size=(n_dates, n_assets)), index=dates, columns=assets,
    )
    forward = factor * 0.01 + pd.DataFrame(
        rng.normal(0, 0.02, (n_dates, n_assets)), index=dates, columns=assets,
    )
    eligible = pd.DataFrame(True, index=dates, columns=assets)
    return factor, forward, eligible


def test_ic_detects_a_planted_relation():
    factor, forward, eligible = _panel()

    ic = cross_sectional_ic(factor, forward, eligible)

    assert len(ic) == len(factor)
    assert ic.mean() > 0.2


def test_ic_requires_enough_eligible_names():
    factor, forward, eligible = _panel()
    eligible.iloc[:, :40] = False  # only 20 names left, below MIN_NAMES

    ic = cross_sectional_ic(factor, forward, eligible)

    assert ic.empty


def test_ic_stats_reports_significance():
    ic = pd.Series([0.05, 0.04, 0.06, 0.03, 0.05, 0.02])

    stats = ic_stats(ic)

    assert stats["n_months"] == 6
    assert stats["mean_ic"] == pytest.approx(0.0416667, rel=1e-4)
    assert stats["t_stat"] > 5
    assert stats["positive_ratio"] == 1.0


def test_backtest_longs_the_top_quintile_and_charges_turnover():
    factor, _, eligible = _panel(n_assets=100)
    forward = factor * 0.01  # no noise: the top quintile must win

    gross, net, bench, turnover = backtest(factor, forward, eligible)

    assert len(gross) == len(factor) - 1
    assert (gross > 0).all()
    assert (net <= gross).all()
    # One-sided turnover: cash -> fully invested is 0.5 (half the two-way
    # traded value). Cost uses 2 * turnover at COST_BPS per traded value.
    assert turnover.iloc[0] == pytest.approx(0.5)
    assert (turnover >= 0).all()
    assert net.iloc[0] == pytest.approx(
        gross.iloc[0] - turnover.iloc[0] * 2 * 15 / 10000
    )


def test_performance_of_constant_returns():
    monthly = pd.Series([0.01] * 12)

    stats = performance(monthly)

    assert stats["annual_return"] == pytest.approx(1.01**12 - 1)
    assert stats["annual_vol"] == pytest.approx(0.0)
    assert stats["sharpe"] == 0.0
    assert stats["max_drawdown"] == pytest.approx(0.0)
