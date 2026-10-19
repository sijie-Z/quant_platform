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
