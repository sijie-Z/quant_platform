"""Exposure report contract."""

import pandas as pd
import pytest
from quant_platform.risk.exposure import exposure_report


def test_report_includes_the_sector_map_used_by_consumers():
    """`sector_exposure` was computed and dropped, so API/HTML sector blocks
    were always empty on real runs."""
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Financials"})

    report = exposure_report(weights, sector_map=sectors)

    assert report["sector_exposure"].to_dict() == pytest.approx(
        {"Tech": 0.8, "Financials": 0.2},
    )
    assert report["max_sector_exposure"] == pytest.approx(0.8)
    assert report["top_sector"] == "Tech"
    assert report["n_sectors"] == 2


def test_no_sector_map_means_no_sector_key():
    report = exposure_report(pd.Series({"A": 1.0}))
    assert "sector_exposure" not in report
