"""Covariance estimation must not drop every asset with one suspension."""

import numpy as np
import pandas as pd
from quant_platform.portfolio.covariance import estimate_covariance


def _returns(periods: int = 252, assets: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        rng.normal(0, 0.01, (periods, assets)),
        index=dates,
        columns=[f"S{i}" for i in range(assets)],
    )


def test_a_single_missing_day_does_not_remove_the_asset():
    returns = _returns()
    returns.iloc[10, 2] = np.nan

    cov = estimate_covariance(returns, method="sample", lookback=252)

    assert "S2" in cov.columns
    assert cov.shape == (5, 5)
    assert np.isfinite(cov.to_numpy()).all()


def test_an_asset_missing_most_of_the_window_is_dropped():
    returns = _returns()
    returns["S1"] = np.nan

    cov = estimate_covariance(returns, method="sample", lookback=252)

    assert "S1" not in cov.columns
    assert cov.shape == (4, 4)
