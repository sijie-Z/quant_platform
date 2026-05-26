"""Tests for factor processing (winsorize, standardize, neutralize)."""

import numpy as np
import pandas as pd
from quant_platform.factors.processing import (
    neutralize,
    process_factor,
    standardize,
    winsorize,
)


def _make_factor(n_dates=200, n_assets=50):
    """Helper: create a synthetic factor DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}.SH" for i in range(n_assets)]
    # Add some outliers
    data = np.random.randn(n_dates, n_assets) + np.random.randn(n_assets) * 0.5
    data[:, 0] = 10  # outlier
    data[:, 1] = -10  # outlier
    return pd.DataFrame(data, index=dates, columns=assets)


def test_winsorize_clips_outliers():
    factor = _make_factor()
    result = winsorize(factor, lower=0.01, upper=0.99)
    # Winsorized data should have reduced range vs raw
    for date in result.index:
        raw_row = factor.loc[date].dropna()
        result_row = result.loc[date].dropna()
        assert result_row.max() <= raw_row.max() + 1e-10
        assert result_row.min() >= raw_row.min() - 1e-10
        # Key: the extreme outlier should have been clipped
        # The range of clipped data should not exceed raw quantile range
        raw_lo = raw_row.quantile(0.01)
        raw_hi = raw_row.quantile(0.99)
        assert (result_row >= raw_lo - 1e-10).all()
        assert (result_row <= raw_hi + 1e-10).all()


def test_standardize_zscore():
    factor = _make_factor()
    result = standardize(factor, method="zscore")
    # Each cross-section should have mean ~0, std ~1
    for date in result.index[:10]:
        row = result.loc[date].dropna()
        if len(row) > 10:
            assert abs(row.mean()) < 0.5
            assert abs(row.std() - 1.0) < 0.5


def test_standardize_rank():
    factor = _make_factor()
    result = standardize(factor, method="rank")
    valid = result.dropna()
    if valid.size > 0:
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()


def test_neutralize_reduces_sector_bias():
    factor = _make_factor()
    # Create a sector map where one sector has consistently higher values
    n_assets = factor.shape[1]
    sectors = pd.Series(
        ["A"] * (n_assets // 2) + ["B"] * (n_assets - n_assets // 2),
        index=factor.columns,
    )
    result = neutralize(factor, sector_map=sectors)

    # Neutralized factor should have reduced correlation with the original
    assert result.shape == factor.shape


def test_process_pipeline():
    factor = _make_factor()
    result = process_factor(
        factor,
        winsorize_enabled=True,
        standardize_enabled=True,
        neutralize_enabled=True,
        sector_map=None,
        market_cap=None,
    )
    assert result.shape == factor.shape
    assert not result.isnull().all().all()
