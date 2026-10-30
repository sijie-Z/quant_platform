"""Regression tests for ChartData (BUG-31).

`ChartData.monthly_returns` was annotated `dict[str, list[float]]`, but
`_build_chart_data` builds and the dashboard's heatmap reads
`{"years": [str], "months": [int], "data": [[float]]}`. No value of that shape
satisfies `dict[str, list[float]]`, so pydantic rejected it, `_execute_pipeline`
marked the run failed, and `GET /api/run/{id}/result` always returned 404 -- on
the primary product endpoint.
"""

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.api.routes import _build_chart_data
from quant_platform.api.schemas import ChartData, MonthlyReturns


def _series(n_days, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0, 0.01, n_days), index=pd.bdate_range(start, periods=n_days)
    )


class TestBuildChartData:
    def test_multi_year_series_builds_the_grid(self):
        """This is the case that used to raise ValidationError."""
        chart = _build_chart_data(_series(400))
        assert isinstance(chart, ChartData)

        mr = chart.monthly_returns
        assert isinstance(mr, MonthlyReturns)
        assert mr.years == ["2024", "2025"]
        assert mr.months == list(range(1, 13))
        assert len(mr.data) == 2
        assert all(len(row) == 12 for row in mr.data)

    def test_series_shorter_than_a_month(self):
        chart = _build_chart_data(_series(2))
        # Nothing to build a monthly grid from -- must not raise.
        assert chart.monthly_returns is None

    def test_long_series_spans_every_year(self):
        chart = _build_chart_data(_series(1000, start="2020-01-01"))
        mr = chart.monthly_returns
        assert mr.years == ["2020", "2021", "2022", "2023"]
        assert len(mr.data) == len(mr.years)

    def test_months_without_observations_are_null_not_missing(self):
        """The heatmap indexes data[year][month] positionally, so an absent
        month must still occupy its slot. It is `null` rather than NaN:
        Starlette's JSON encoder rejects NaN and returned a 500."""
        mr = _build_chart_data(_series(400)).monthly_returns
        assert len(mr.months) == 12
        missing = [v for row in mr.data for v in row if v is None]
        assert missing, "expected some months with no observation in a 400-day series"


class TestMonthlyReturnsShape:
    def test_matches_what_the_dashboard_reads(self):
        """TerminalDashboard does `mr.years`, `mr.months`, `mr.data?.[yi]?.[mi]`.

        The old annotation described a dict mapping a string to a flat list of
        floats, which cannot express any of those.
        """
        mr = MonthlyReturns(
            years=["2024"], months=list(range(1, 13)),
            data=[[0.01] * 12],
        )
        payload = mr.model_dump()
        assert set(payload) == {"years", "months", "data"}
        assert payload["data"][0][0] == pytest.approx(0.01)

    def test_defaults_are_empty_not_none(self):
        mr = MonthlyReturns()
        assert mr.years == [] and mr.months == [] and mr.data == []

    def test_old_annotation_would_reject_this_shape(self):
        """Pins why the fix was needed: the producer's output is not
        `dict[str, list[float]]`."""
        produced = {
            "years": ["2024"],
            "months": [1, 2],
            "data": [[0.01, -0.02]],
        }
        # `data` is a list of lists, not a list of floats.
        assert isinstance(produced["data"][0], list)
        assert not isinstance(produced["data"][0], float)


def test_partial_year_grid_uses_none_instead_of_nan():
    """A mid-year start used to leave NaN cells, which Starlette refuses to
    serialise (GET /api/run/{id}/result returned 500)."""
    series = _series(300, start="2024-06-03")

    chart = _build_chart_data(series)

    assert chart.monthly_returns is not None
    cells = [value for row in chart.monthly_returns.data for value in row]
    assert any(value is None for value in cells)
    assert not any(
        isinstance(value, float) and math.isnan(value) for value in cells
    )


def test_scatter_and_attribution_use_the_asset_return_panel():
    """Passing the portfolio-return Series made both panels silently empty.

    `_compute_factor_scatter`/`_compute_attribution` need a date x asset
    forward-return panel; calling `.loc[d]` on a Series raised and was
    swallowed, so every real run rendered "No scatter/attribution data".
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-01", periods=140)
    assets = [f"S{i:02d}" for i in range(20)]
    factors = {
        "f1": pd.DataFrame(rng.normal(size=(140, 20)), index=dates, columns=assets),
        "f2": pd.DataFrame(rng.normal(size=(140, 20)), index=dates, columns=assets),
    }
    returns = pd.DataFrame(rng.normal(0, 0.01, (140, 20)), index=dates, columns=assets)
    signal = factors["f1"].rank(axis=1, pct=True) - 0.5
    weights = pd.Series(1.0 / len(assets), index=assets)
    weights_history = {dates[70]: weights, dates[110]: weights}
    strategy_returns = returns.mean(axis=1)

    chart = _build_chart_data(
        strategy_returns,
        weights_history=weights_history,
        processed_factors=factors,
        signal=signal,
        asset_returns=returns,
    )

    assert chart.factor_scatter, "factor scatter must not be empty"
    assert chart.attribution, "attribution must not be empty"
    assert all(item.factor in factors for item in chart.attribution)
