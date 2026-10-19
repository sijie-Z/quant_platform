"""Attribution turnover must count names entering and leaving."""

import pandas as pd
import pytest

from quant_platform.reporting.attribution import turnover_analysis


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
