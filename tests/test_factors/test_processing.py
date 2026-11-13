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
    # One sector carries a large constant premium.
    n_assets = factor.shape[1]
    sectors = pd.Series(
        ["A"] * (n_assets // 2) + ["B"] * (n_assets - n_assets // 2),
        index=factor.columns,
    )
    biased = factor.copy()
    biased.loc[:, sectors == "B"] += 3.0

    result = neutralize(biased, sector_map=sectors)

    assert result.shape == factor.shape
    # A residual regression on sector dummies has zero group mean by
    # construction. Checking only the shape would pass an identity return.
    for date in result.index[:10]:
        row = result.loc[date]
        assert abs(row[sectors == "A"].mean()) < 1e-8
        assert abs(row[sectors == "B"].mean()) < 1e-8
    assert not np.allclose(result.to_numpy(), biased.to_numpy())


def test_process_pipeline():
    factor = _make_factor()
    sectors = pd.Series(
        ["A"] * (factor.shape[1] // 2)
        + ["B"] * (factor.shape[1] - factor.shape[1] // 2),
        index=factor.columns,
    )
    result = process_factor(
        factor,
        winsorize_enabled=True,
        standardize_enabled=True,
        neutralize_enabled=True,
        sector_map=sectors,
        market_cap=None,
    )
    assert result.shape == factor.shape
    assert not result.isnull().all().all()
    # The pipeline output is winsorized (the +-10 outliers are clipped) and
    # standardized, so it must no longer contain the raw values.
    assert not np.allclose(result.to_numpy(), factor.to_numpy())
    assert result.abs().to_numpy().max() < 6.0


def test_winsorize_preserves_a_sparse_binary_factor():
    """With 2 non-zero names out of 200, both 1%/99% quantiles are 0.

    Clipping flattened the signal to all zeros, so `breakout_ignition` (a
    binary event factor) had 1,304 dates with >=30 "valid" values but not a
    single evaluable IC observation in the default pipeline.
    """
    dates = pd.bdate_range("2024-01-02", periods=60)
    assets = [f"S{i:03d}" for i in range(500)]
    data = np.zeros((60, 500))
    data[:, 10] = 1.0
    data[:, 450] = 1.0
    factor = pd.DataFrame(data, index=dates, columns=assets)

    import quant_platform.utils.numba_accelerator as accel

    out_numba = winsorize(factor, 0.01, 0.99)
    out_pandas = accel.winsorize_pandas(factor, 0.01, 0.99)

    for out in (out_numba, out_pandas):
        assert out.to_numpy().sum() == 120.0, "the two event names were clipped"


def test_log_market_cap_is_not_neutralized_against_itself():
    """The size factor must not be regressed on its own market cap.

    ``log_market_cap`` neutralized with ``market_cap=`` used to be regressed
    on itself: the residual is numerically zero and the processed
    cross-sectional std collapsed from ~0.77 (raw log market cap) to ~0.0097.
    The fix skips only the market-cap leg for this factor; sector
    neutralization still applies.
    """
    rng = np.random.default_rng(7)
    n_dates, n_assets = 40, 80
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    assets = [f"S{i:03d}" for i in range(n_assets)]

    base = rng.lognormal(np.log(5e9), 0.8, size=n_assets)
    mcap = pd.DataFrame(
        np.tile(base, (n_dates, 1))
        * rng.lognormal(0.0, 0.05, size=(n_dates, n_assets)),
        index=dates,
        columns=assets,
    )
    factor = np.log(mcap)
    sectors = pd.Series(
        np.array(["Tech", "Finance", "Energy", "Health"])[
            np.arange(n_assets) % 4
        ],
        index=assets,
    )
    raw_std = factor.stack().std()

    processed = process_factor(
        factor,
        winsorize_enabled=False,
        standardize_enabled=False,
        neutralize_enabled=True,
        sector_map=sectors,
        market_cap=mcap,
        factor_name="log_market_cap",
    )
    assert processed.stack().std() > 0.5 * raw_std

    # The name-aware skip is the fix: without it the factor still collapses.
    collapsed = process_factor(
        factor,
        winsorize_enabled=False,
        standardize_enabled=False,
        neutralize_enabled=True,
        sector_map=sectors,
        market_cap=mcap,
    )
    assert collapsed.stack().std() < 0.1 * raw_std


def test_sparse_event_factor_survives_processing_and_yields_ic():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=80)
    assets = [f"S{i:03d}" for i in range(200)]
    data = np.zeros((80, 200))
    data[:, [5, 100]] = 1.0
    factor = pd.DataFrame(data, index=dates, columns=assets)
    forward = pd.DataFrame(
        rng.normal(0, 0.01, (80, 200)), index=dates, columns=assets,
    )
    forward.iloc[:, 5] += 0.02
    forward.iloc[:, 100] += 0.02

    from quant_platform.factors.evaluation import rank_ic

    processed = process_factor(factor)
    ic = rank_ic(processed, forward)

    assert len(ic) > 0, "processing erased the sparse event signal"
