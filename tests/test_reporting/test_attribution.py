"""Attribution turnover must count names entering and leaving."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest.metrics import infer_periods_per_year
from quant_platform.reporting.attribution import factor_attribution, turnover_analysis


def test_turnover_counts_entering_and_leaving_names():
    first, second = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    previous = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    current = pd.Series({"A": 0.25, "B": 0.25, "E": 0.25, "F": 0.25})

    result = turnover_analysis({first: previous, second: current})

    # Half of the book was replaced: one-sided turnover 0.5. The intersection
    # diff saw only A and B and reported 0.0.
    assert result.loc[second, "turnover"] == pytest.approx(0.5)


def test_complete_rotation_with_one_name_per_period_does_not_crash():
    first, second = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")

    result = turnover_analysis({
        first: pd.Series({"A": 1.0}),
        second: pd.Series({"B": 1.0}),
    })

    assert list(result.columns) == ["turnover", "n_assets"]
    assert result.loc[second, "turnover"] == pytest.approx(1.0)


def _attribution_fixture(seed: int = 7, n: int = 400):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    factor_returns = pd.DataFrame({
        "value": rng.normal(0.0002, 0.010, n),
        "momentum": rng.normal(0.0001, 0.008, n),
    }, index=dates)
    strategy = (
        0.0003
        + 1.2 * factor_returns["value"]
        - 0.5 * factor_returns["momentum"]
        + rng.normal(0, 0.002, n)
    )
    return strategy, factor_returns


def test_attribution_has_an_intercept_row_and_shares_sum_to_100():
    strategy, factor_returns = _attribution_fixture()

    frame = factor_attribution(strategy, factor_returns)

    assert "intercept" in frame.index
    assert frame.loc["intercept", "contribution_annual"] == pytest.approx(
        frame.loc["intercept", "beta"] * infer_periods_per_year(strategy.dropna())
    )
    assert frame["contribution_pct"].sum() == pytest.approx(100.0)


def test_contribution_annual_uses_inferred_periods_per_year():
    strategy, factor_returns = _attribution_fixture()

    frame = factor_attribution(strategy, factor_returns)
    ppy = infer_periods_per_year(strategy.dropna())

    for factor in factor_returns.columns:
        expected = (
            frame.loc[factor, "beta"]
            * frame.loc[factor, "factor_mean_daily"]
            * ppy
        )
        assert frame.loc[factor, "contribution_annual"] == pytest.approx(expected)
    # The old code hardcoded 252 regardless of the sample's calendar span.
    assert ppy != 252


def test_cancelling_factors_report_nan_shares_instead_of_exploding():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", periods=400)
    first = rng.normal(0, 0.01, 400)
    second = rng.normal(0, 0.01, 400)
    # Offset the means so the net factor contribution is ~3% of the gross
    # churn: shares would print in the hundreds of percent. The old formula
    # reported +755.4% / -655.4% here.
    first = first - first.mean() + 0.001
    second = second - second.mean() - 0.00095
    factor_returns = pd.DataFrame({"long": first, "short": second}, index=dates)
    strategy = pd.Series(
        first + second + rng.normal(0, 0.0005, 400), index=dates,
    )

    frame = factor_attribution(strategy, factor_returns)

    assert frame["contribution_pct"].isna().all()
    assert not np.isinf(frame["contribution_pct"].to_numpy(dtype=float)).any()


def test_exactly_cancelling_factors_report_nan_shares():
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2022-01-03", periods=400)
    first = rng.normal(0, 0.01, 400)
    second = rng.normal(0, 0.01, 400)
    first = first - first.mean() + 0.001
    second = second - second.mean() - 0.001
    factor_returns = pd.DataFrame({"long": first, "short": second}, index=dates)
    strategy = pd.Series(first + second, index=dates)

    frame = factor_attribution(strategy, factor_returns)

    assert frame["contribution_pct"].isna().all()
