"""P8: a corporate action on a suspension day must not move the stale mark.

The synthetic provider applies `adj_factor` to the raw price path and then
blanks/forward-fills suspended rows.  It used to leave the new adjustment
factor on the suspended row, so `close_adj = close / adj_factor` jumped even
though the clean adjusted path never moved (measured 65.50 -> 76.56 on
000007.SH, 2022-03-07).  Returns masked those rows, but `get_close()` -- the
valuation mark used by the daemon/monitoring -- showed the fake price.

The fix freezes `adj_factor` on suspended rows before the forward-fill, so the
stale mark equals the last clean adjusted close exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider

START, END = "2021-01-01", "2024-12-31"
#: Seed 1 puts a corporate action on a suspension day for 000097.SH
#: (2023-08-30); the assertion below re-derives the collision and fails if
#: the generator no longer produces one.
SEED = 1
N_STOCKS = 150


@pytest.fixture(scope="module")
def provider():
    prov = SyntheticDataProvider(
        n_stocks=N_STOCKS, start_date=START, end_date=END, seed=SEED
    )
    prov._generate_all()
    return prov


@pytest.fixture(scope="module")
def pipeline(provider):
    pipe = DataPipeline(
        provider, START, END, exclude_st=False, exclude_suspended=False
    )
    pipe.run()
    return pipe


def _action_suspension_collisions(provider, prices) -> list[tuple[str, pd.Timestamp]]:
    suspended = prices["is_suspended"].unstack("asset").fillna(False).astype(bool)
    collisions = []
    first_date = suspended.index[0]
    for asset, days in provider._corporate_action_days.items():
        for day in days:
            if (
                day in suspended.index
                and asset in suspended.columns
                and bool(suspended.loc[day, asset])
                and day != first_date
            ):
                collisions.append((asset, day))
    return collisions


def test_the_fixture_still_contains_an_action_on_a_suspension_day(provider):
    prices = provider.get_prices(START, END)
    collisions = _action_suspension_collisions(provider, prices)
    assert collisions, (
        "seed/generator no longer puts a corporate action on a suspended "
        "day; this test would be vacuous"
    )


def test_stale_mark_equals_last_clean_adjusted_close(provider):
    prices = provider.get_prices(START, END)
    close_adj = (prices["close"] / prices["adj_factor"]).unstack("asset")
    suspended = prices["is_suspended"].unstack("asset").fillna(False).astype(bool)

    collisions = _action_suspension_collisions(provider, prices)
    for asset, day in collisions:
        clean_before = close_adj[asset].loc[:day].iloc[:-1]
        clean_before = clean_before[~suspended[asset].reindex(clean_before.index)]
        last_clean = clean_before.dropna().iloc[-1]
        assert close_adj.loc[day, asset] == pytest.approx(last_clean, abs=1e-12), (
            f"{asset} {day.date()}: stale mark {close_adj.loc[day, asset]} != "
            f"last clean adjusted close {last_clean}"
        )


def test_no_suspension_row_jumps_in_the_adjusted_path(provider):
    prices = provider.get_prices(START, END)
    close_adj = (prices["close"] / prices["adj_factor"]).unstack("asset")
    suspended = prices["is_suspended"].unstack("asset").fillna(False).astype(bool)

    for asset in close_adj.columns:
        series = close_adj[asset]
        previous = series.shift(1)
        check = suspended[asset] & previous.notna() & series.notna()
        diff = (series - previous).abs().where(check).dropna()
        assert (diff < 1e-12).all(), (
            f"{asset}: adjusted close jumps on a suspension day "
            f"(max |diff| {diff.max()})"
        )


def test_suspended_action_raw_mark_is_post_action(provider):
    """The stale raw mark must be the post-action level, not the pre-action one.

    A paper book holding this stock applies the split/dividend on the ex-date.
    If the feed kept the pre-action raw close while the share count doubled,
    the position's market value would double until the resume session.
    `close_adj` staying continuous (tests above) is not sufficient: the raw
    close itself must be scaled by the action factor (F_ex / F_previous
    traded day), which is exactly what a real exchange's adjusted reference
    price does for a suspended stock.
    """
    prices = provider.get_prices(START, END)
    close = prices["close"].unstack("asset")
    factor = prices["adj_factor"].unstack("asset")
    suspended = prices["is_suspended"].unstack("asset").fillna(False).astype(bool)

    collisions = _action_suspension_collisions(provider, prices)
    assert collisions
    for asset, day in collisions:
        position = close.index.get_loc(day)
        prior = close[asset].iloc[:position][~suspended[asset].iloc[:position]]
        prior_factor = factor[asset].iloc[:position][
            ~suspended[asset].iloc[:position]
        ]
        if prior.dropna().empty:
            continue
        prev_raw = prior.dropna().iloc[-1]
        prev_factor = prior_factor.dropna().iloc[-1]
        assert close.loc[day, asset] == pytest.approx(
            prev_raw * factor.loc[day, asset] / prev_factor, rel=1e-9,
        )


def test_pipeline_valuation_mark_is_the_last_clean_close(pipeline, provider):
    """`get_close()` is what the daemon marks positions with."""
    close = pipeline.get_close()
    suspended = pipeline._suspended

    collisions = _action_suspension_collisions(
        provider, pipeline.prices.reset_index().set_index(["date", "asset"])
    )
    assert collisions
    for asset, day in collisions:
        clean_before = close[asset].loc[:day].iloc[:-1]
        clean_before = clean_before[~suspended[asset].reindex(clean_before.index)]
        last_clean = clean_before.dropna().iloc[-1]
        assert close.loc[day, asset] == pytest.approx(last_clean, abs=1e-12)

        # And the suspended-day return is not fabricated.
        assert pd.isna(pipeline.returns.loc[day, asset]) or day == pipeline.returns.index[-1]
