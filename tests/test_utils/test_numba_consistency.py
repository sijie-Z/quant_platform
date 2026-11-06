"""The numba kernels must agree with the pandas/scipy definitions."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.utils import numba_accelerator as accel


def test_numba_momentum_matches_pandas():
    if not accel.HAS_NUMBA:
        pytest.skip("numba unavailable")

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=120)
    assets = [f"S{i:02d}" for i in range(12)]
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (120, 12)), index=dates, columns=assets,
    )

    accelerated = accel.momentum_factor_numba(returns, 21)
    reference = accel.momentum_factor_pandas(returns, 21)

    np.testing.assert_allclose(
        accelerated.to_numpy(), reference.to_numpy(),
        rtol=1e-10, atol=1e-12, equal_nan=True,
    )


def test_rank_ic_numba_averages_ties_like_scipy():
    if not accel.HAS_NUMBA:
        pytest.skip("numba unavailable")

    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    factor_row = np.array([0.0, 0.3, 1.0] * 14)[:40].astype(float)
    ret_row = rng.normal(size=40)
    date = pd.Timestamp("2024-01-02")
    assets = [f"S{i:02d}" for i in range(40)]
    factor = pd.DataFrame([factor_row], index=[date], columns=assets)
    returns = pd.DataFrame([ret_row], index=[date], columns=assets)

    accelerated = float(accel.rank_ic_numba(factor, returns).iloc[0])
    expected = float(spearmanr(factor_row, ret_row).statistic)

    assert accelerated == pytest.approx(expected, abs=1e-12)

    order = rng.permutation(40)
    permuted = float(
        accel.rank_ic_numba(factor.iloc[:, order], returns.iloc[:, order]).iloc[0]
    )
    assert permuted == pytest.approx(accelerated, abs=1e-12)


def test_rank_ic_paths_agree_for_a_small_universe(monkeypatch):
    """The kernel used to hardcode a 30-asset floor, so a 30-stock universe
    yielded no IC at all. Both paths must apply the same shared breadth rule
    (``min_assets_for_ic``) and agree on the evaluable cross-sections."""
    if not accel.HAS_NUMBA:
        pytest.skip("numba unavailable")

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=40)
    assets = [f"S{i:02d}" for i in range(30)]
    factor = pd.DataFrame(rng.normal(size=(40, 30)), index=dates, columns=assets)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (40, 30)), index=dates, columns=assets,
    )
    for i in range(40):
        missing = rng.choice(30, size=3, replace=False)
        factor.iloc[i, missing] = np.nan
        returns.iloc[i, missing] = np.nan

    accelerated = accel.rank_ic_numba(factor, returns)

    monkeypatch.setattr(accel, "HAS_NUMBA", False)
    from quant_platform.factors.evaluation import rank_ic

    fallback = rank_ic(factor, returns)

    assert len(accelerated) == len(fallback) == 40
    pd.testing.assert_series_equal(
        accelerated, fallback, check_exact=False, rtol=1e-9, atol=1e-12,
    )


def test_ledoit_wolf_kernel_compiles_and_matches_sklearn():
    """`returns.mean(axis=0)` is not a valid numba overload.

    The kernel raised TypingError for every call with >=5 assets; the engine
    swallowed the exception and ran the backtest with `cov=None` (identity
    covariance for MVO, equal-weight fallback for risk parity).
    """
    if not accel.HAS_NUMBA:
        pytest.skip("numba unavailable")

    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(0)
    n, p = 252, 20
    market = rng.normal(0, 0.01, (n, 1))
    returns = pd.DataFrame(
        market + rng.normal(0, 0.02, (n, p)),
        index=pd.bdate_range("2023-01-02", periods=n),
        columns=[f"S{i:02d}" for i in range(p)],
    )

    cov = accel.covariance_numba(returns)
    reference = LedoitWolf().fit(returns.values).covariance_

    assert cov.shape == (p, p)
    assert np.allclose(cov.values, cov.values.T)
    assert np.linalg.eigvalsh(cov.values).min() > -1e-12
    # Different shrinkage targets (constant-correlation vs scaled identity),
    # so exact equality is not expected; the estimates must be close.
    np.testing.assert_allclose(cov.values, reference, atol=1e-3)


def test_estimate_covariance_ledoit_wolf_does_not_raise():
    from quant_platform.portfolio.covariance import estimate_covariance

    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (120, 12)),
        index=pd.bdate_range("2023-01-02", periods=120),
        columns=[f"S{i:02d}" for i in range(12)],
    )

    cov = estimate_covariance(returns, method="ledoit_wolf")

    assert cov.shape == (12, 12)
    assert np.isfinite(cov.values).all()
