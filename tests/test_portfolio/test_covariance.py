"""Covariance estimation must not drop every asset with one suspension."""

import numpy as np
import pandas as pd
import pytest
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


def _heterogeneous_returns(periods: int = 500, assets: int = 6) -> pd.DataFrame:
    """Correlated assets whose volatilities differ by up to 6x."""
    rng = np.random.default_rng(11)
    market = rng.normal(0, 1, (periods, 1))
    scales = np.linspace(0.01, 0.06, assets)
    data = (0.7 * market + 0.7 * rng.normal(0, 1, (periods, assets))) * scales
    return pd.DataFrame(
        data,
        index=pd.bdate_range("2023-01-02", periods=periods),
        columns=[f"S{i}" for i in range(assets)],
    )


def test_ledoit_wolf_is_not_an_identity_matrix():
    """The default covariance method must use the data's scale and structure.

    A shutdown/identity return satisfies shape and finiteness checks, which is
    all the previous tests asserted, while silently turning MVO into an
    equal-weight L2 penalty and RiskParity into equal weight.
    """
    returns = _heterogeneous_returns()
    sample = estimate_covariance(returns, method="sample", lookback=500).to_numpy()
    lw = estimate_covariance(returns, method="ledoit_wolf", lookback=500).to_numpy()

    assert not np.allclose(lw, np.eye(lw.shape[0]))
    # Both shrinkage targets preserve the total variance (trace).
    assert np.trace(lw) == pytest.approx(np.trace(sample), rel=0.1)
    # Diagonal entries carry each asset's own variance, not a common constant.
    assert lw[5, 5] > lw[0, 0] * 4
    # The shared market factor leaves positive off-diagonal covariance.
    assert lw[0, 1] > 0
    assert np.allclose(lw, lw.T, atol=1e-12)
    assert np.linalg.eigvalsh(lw).min() > -1e-12


def test_ledoit_wolf_sklearn_fallback_keeps_the_same_properties(monkeypatch):
    """The non-numba branch must not become a silent identity either."""
    from quant_platform.utils import numba_accelerator

    monkeypatch.setattr(numba_accelerator, "HAS_NUMBA", False)
    returns = _heterogeneous_returns()
    sample = estimate_covariance(returns, method="sample", lookback=500).to_numpy()
    lw = estimate_covariance(returns, method="ledoit_wolf", lookback=500).to_numpy()

    assert not np.allclose(lw, np.eye(lw.shape[0]))
    assert np.trace(lw) == pytest.approx(np.trace(sample), rel=0.1)
    assert lw[5, 5] > lw[0, 0] * 4
    assert np.linalg.eigvalsh(lw).min() > -1e-12
