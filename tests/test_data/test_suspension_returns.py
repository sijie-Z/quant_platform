"""Suspension forward-fill must not fabricate a reversal signal (issue #58).

The synthetic provider forward-fills suspended days. Differencing the filled
path books a fake zero on the suspension day and the compounded multi-day move
on the resume day -- adjacent observations that produce a spurious negative
lag-1 IC on pure noise. Returns and factor prices now exclude both boundaries.
"""

import numpy as np
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.factors.evaluation import rank_ic


@pytest.fixture(scope="module")
def pipeline():
    provider = SyntheticDataProvider(
        n_stocks=120,
        start_date="2021-01-01",
        end_date="2025-12-31",
        seed=42,
        embedded_alpha=False,
    )
    pipe = DataPipeline(
        provider,
        "2021-01-01",
        "2025-12-31",
        exclude_st=False,
        exclude_suspended=False,
    )
    pipe.run()
    return pipe


def test_provider_marks_suspensions_and_keeps_the_stale_price(pipeline):
    prices = pipeline.prices
    assert "is_suspended" in prices.columns
    assert prices["is_suspended"].dtype == bool

    rate = float(prices["is_suspended"].mean())
    assert 0.005 < rate < 0.05, f"suspension rate {rate:.3f} looks wrong"

    # Realism is preserved: a suspended row carries the previous close.
    # Rows without a prior close (first observation of an asset) are not
    # comparable and are excluded.
    previous_close = prices["close"].groupby(level="asset").shift(1)
    suspended_close = prices.loc[prices["is_suspended"], "close"]
    prior = previous_close[prices["is_suspended"]]
    comparable = prior.notna() & suspended_close.notna()
    assert comparable.sum() > 0
    assert (suspended_close[comparable] == prior[comparable]).mean() > 0.99


def test_returns_never_span_a_forward_fill(pipeline):
    suspended = (
        pipeline.prices["is_suspended"].unstack("asset").fillna(False).astype(bool)
    )
    # returns[t] spans t -> t+1, so either endpoint being stale contaminates it.
    bad = suspended | suspended.shift(-1)
    assert pipeline.returns.where(bad).isna().all().all()


def test_factor_prices_are_blank_on_suspended_days(pipeline):
    suspended = (
        pipeline.prices["is_suspended"].unstack("asset").fillna(False).astype(bool)
    )

    factor_prices = pipeline.get_factor_prices()
    assert factor_prices.where(suspended).isna().all().all()

    # Valuation still keeps the forward-filled mark.
    assert pipeline.get_close().where(suspended).notna().any().any()


def test_adjusted_returns_have_no_corporate_action_jumps(pipeline):
    """adj_factor moved without the raw price, so close_adj jumped ±20%."""
    close = pipeline.prices["close_adj"].unstack("asset")
    returns = close.pct_change(fill_method=None).abs()

    assert returns.max().max() < 0.12


def test_lagged_return_ic_is_zero_on_pure_noise(pipeline):
    returns = pipeline.returns
    lagged = returns.shift(1)
    ic = rank_ic(lagged, returns)

    assert len(ic) > 200, "not enough observations for a meaningful null control"
    # Not every asset trades for the whole sample (pre-listing rows are NaN),
    # but the panel must stay mostly populated for the null control to mean
    # anything.
    assert returns.notna().mean().mean() > 0.70

    mean_ic = float(ic.mean())
    t_stat = mean_ic / float(ic.std()) * np.sqrt(len(ic))
    assert abs(mean_ic) < 0.004, (
        f"mean IC {mean_ic:.4f} looks like the forward-fill artifact"
    )
    assert t_stat > -2.0, f"t-stat {t_stat:.2f} is significantly negative"
