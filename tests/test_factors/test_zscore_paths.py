"""Z-score convention: population std (ddof=0) on both the numba and pandas paths.

``_zscore_numba`` divides by ``np.std(valid)`` (population std), while the
pandas fallback inside ``zscore_numba`` divided by ``Series.std()`` (sample
std). The paths disagreed by sqrt(n/(n-1)) -- 5.1% at 10 assets -- and no test
ever compared them. The convention is now pinned to population std, which is
what the accelerated path (the one that produces production factor values)
already used.
"""

import numpy as np
import pandas as pd
import pytest
import quant_platform.utils.numba_accelerator as accel
from quant_platform.factors.processing import standardize

N_ASSETS = 10
N_DATES = 5
POPULATION_ROW_STD = 1.0
SAMPLE_ROW_STD = float(np.sqrt(N_ASSETS / (N_ASSETS - 1)))


def _cross_section() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=N_DATES)
    assets = [f"S{i:02d}" for i in range(N_ASSETS)]
    # Distinct per-row location and scale so every row exercises the kernel.
    data = rng.normal(size=(N_DATES, N_ASSETS))
    data = data * np.linspace(0.5, 2.0, N_DATES)[:, None] + 100.0
    return pd.DataFrame(data, index=dates, columns=assets)


def _row_stats(frame: pd.DataFrame, row: int = 0):
    values = frame.iloc[row].dropna()
    return values.mean(), values.std(ddof=0), values.std(ddof=1)


def test_accelerated_and_pandas_paths_agree_to_1e12(monkeypatch):
    factor = _cross_section()

    accelerated = standardize(factor, method="zscore")

    monkeypatch.setattr(accel, "HAS_NUMBA", False)
    fallback = standardize(factor, method="zscore")

    pd.testing.assert_frame_equal(
        accelerated, fallback, rtol=1e-12, atol=1e-14,
    )


def test_the_direct_kernel_agrees_with_the_pandas_fallback(monkeypatch):
    """Exercise ``_zscore_numba`` itself even if numba is not installed."""
    factor = _cross_section()
    kernel_values = accel._zscore_numba(factor.to_numpy(dtype=float))
    kernel = pd.DataFrame(
        kernel_values, index=factor.index, columns=factor.columns,
    )

    monkeypatch.setattr(accel, "HAS_NUMBA", False)
    fallback = standardize(factor, method="zscore")

    pd.testing.assert_frame_equal(kernel, fallback, rtol=1e-12, atol=1e-14)


def test_each_standardized_row_has_population_std_one():
    factor = _cross_section()

    accelerated = standardize(factor, method="zscore")

    for row in range(N_DATES):
        mean, population_std, sample_std = _row_stats(accelerated, row)
        assert mean == pytest.approx(0.0, abs=1e-12)
        assert population_std == pytest.approx(POPULATION_ROW_STD, rel=1e-12)
        # The competing convention is measurably different at 10 assets.
        assert sample_std == pytest.approx(SAMPLE_ROW_STD, rel=1e-12)


def test_pandas_fallback_also_uses_population_std(monkeypatch):
    factor = _cross_section()

    monkeypatch.setattr(accel, "HAS_NUMBA", False)
    fallback = standardize(factor, method="zscore")

    for row in range(N_DATES):
        _, population_std, _ = _row_stats(fallback, row)
        assert population_std == pytest.approx(POPULATION_ROW_STD, rel=1e-12)
