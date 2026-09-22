"""Output-shape guard for every registered factor.

`BaseFactor.compute` promises a (date x asset) frame whose columns are the
input panel's assets. Nothing checked it. `MAConvergenceFactor` collapsed the
asset axis -- `pd.concat([gap1, gap2], axis=1).max(axis=1)` returns one value
per date -- and `process_factor` wrapped the Series in a one-column frame whose
only label was the literal string 'factor' (factors/processing.py:171). That
label then reached `_align_factors`, which intersected dates but never columns,
and pandas' label-union alignment in `_build_row` made every cell of the
combined signal NaN. On the default 21-factor config the equal-weighted alpha
signal had 0 of 546,376 cells populated, and the backtest's turnover over its
59 rebalances summed to 0.5 -- the initial build, and no trade after it.

This module tests the promise directly, for every factor in the registry
including ones added later, without naming any of them.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.fundamental import register_all as register_fundamental
from quant_platform.factors.registry import get_registry
from quant_platform.factors.technical import register_all as register_technical

N_DAYS = 300
N_ASSETS = 40


def _panel(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)
    cols = [f"S{i:03d}" for i in range(N_ASSETS)]
    steps = rng.normal(0, 0.012, (N_DAYS, N_ASSETS))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=cols
    )


def _turnover(prices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.uniform(0.005, 0.05, prices.shape),
        index=prices.index,
        columns=prices.columns,
    )


def _financials(prices: pd.DataFrame) -> pd.DataFrame:
    """The (field, asset) column layout `_compute_factors` hands fundamental factors."""
    rng = np.random.default_rng(9)
    fields = {
        "market_cap": rng.uniform(1e9, 5e11, prices.shape),
        "pb_ratio": rng.uniform(0.5, 8.0, prices.shape),
        "pe_ratio": rng.uniform(5.0, 60.0, prices.shape),
        "roe": rng.uniform(-0.1, 0.35, prices.shape),
        "asset_growth": rng.uniform(-0.2, 0.6, prices.shape),
    }
    return pd.concat(
        {
            field: pd.DataFrame(values, index=prices.index, columns=prices.columns)
            for field, values in fields.items()
        },
        axis=1,
    )


def _registered_factors() -> list[str]:
    register_technical()
    register_fundamental()
    return sorted(get_registry().list_all())


def _compute(name: str, prices, turnover, financials) -> object:
    """Factor output on the guard's panel, supplying whatever its category needs."""
    inst = get_registry().get(name)()
    if inst.category.value == "fundamental":
        return inst.compute(prices, financials=financials)
    return inst.compute(prices, turnover=turnover)


def _shape_violation(out: object, prices: pd.DataFrame) -> str | None:
    """Why `out` breaks the (date x asset) contract, or None when it honours it."""
    if not isinstance(out, pd.DataFrame):
        return (
            f"returned a {type(out).__name__} of shape {getattr(out, 'shape', None)}, "
            "not a (date x asset) DataFrame"
        )
    if not out.columns.equals(prices.columns):
        unexpected = [c for c in out.columns if c not in prices.columns][:3]
        missing = [c for c in prices.columns if c not in out.columns][:3]
        return (
            f"has {len(out.columns)} columns for {len(prices.columns)} assets "
            f"(unexpected {unexpected}, missing {missing})"
        )
    if not out.index.isin(prices.index).all():
        return "has dates that are not in the input panel"
    return None


FACTORS = _registered_factors()


@pytest.mark.parametrize("name", FACTORS)
def test_compute_returns_a_date_by_asset_frame(name):
    prices = _panel()
    try:
        out = _compute(name, prices, _turnover(prices), _financials(prices))
    except Exception as exc:
        pytest.fail(f"{name}.compute raised on the guard's panel: {exc!r}")

    violation = _shape_violation(out, prices)
    assert violation is None, f"{name}.compute {violation}"


class TestTheGuardItselfWorks:
    """A shape check that cannot fail is worse than none: these pin down that
    the check discriminates, on the exact shapes the defect produced."""

    def test_it_catches_the_collapsed_series(self):
        prices = _panel()
        collapsed = pd.concat([prices, prices], axis=1).max(axis=1)
        assert _shape_violation(collapsed, prices) is not None

    def test_it_catches_a_one_column_frame_labelled_factor(self):
        prices = _panel()
        wrapped = prices.mean(axis=1).to_frame("factor")
        violation = _shape_violation(wrapped, prices)
        assert violation is not None
        assert "'factor'" in violation

    def test_it_accepts_a_well_formed_frame(self):
        prices = _panel()
        assert _shape_violation(prices.rolling(5).mean(), prices) is None

    def test_it_rejects_dates_outside_the_panel(self):
        prices = _panel()
        shifted = prices.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)
        assert _shape_violation(shifted, prices) is not None

    def test_the_sweep_covers_the_default_universe(self):
        assert len(FACTORS) >= 20, f"only {len(FACTORS)} factors were exercised"
