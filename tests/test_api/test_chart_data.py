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

    def test_months_without_observations_are_nan_not_missing(self):
        """The heatmap indexes data[year][month] positionally, so an absent
        month must still occupy its slot."""
        mr = _build_chart_data(_series(400)).monthly_returns
        assert len(mr.months) == 12
        nanos = [v for row in mr.data for v in row if math.isnan(v)]
        assert nanos, "expected some months with no observation in a 400-day series"


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
