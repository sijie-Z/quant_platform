"""Tests for data pipeline."""


import pandas as pd
from quant_platform.data.pipeline import DataPipeline


def test_pipeline_run(data_pipeline):
    assert data_pipeline.prices is not None
    assert data_pipeline.financials is not None
    assert data_pipeline.benchmark is not None
    assert data_pipeline.metadata is not None


def test_pipeline_returns_shape(data_pipeline):
    returns = data_pipeline.returns
    assert returns is not None
    assert returns.shape[0] > 100  # At least 100 days
    assert returns.shape[1] > 10   # At least 10 stocks


def test_pipeline_excludes_st(synthetic_provider):
    """ST names are screened out per date, not for the whole sample.

    This replaces ``assert not meta["is_st"].any()``.  That assertion was the
    look-ahead written as a specification: the synthetic provider's ``is_st``
    flag means "ever becomes ST", with the trigger drawn from the middle 90%
    of the sample, so demanding ``is_st`` be empty forced stocks out of the
    universe from day one for events years in their future.

    What must hold instead: the run produces a per-date mask, every asset-day
    the mask rejects has no price and no return, and the assets that are never
    ST stay eligible throughout.  The look-ahead property itself is pinned by
    ``tests/test_data/test_pipeline_pit_st_mask.py``.
    """
    pipeline = DataPipeline(
        provider=synthetic_provider,
        start_date="2023-01-01",
        end_date="2024-12-31",
        exclude_st=True,
    )
    pipeline.run()

    mask = pipeline.st_eligible
    assert mask is not None
    # The mask spans every date and every asset the backtest can see.
    assert mask.index.equals(pipeline.returns.index)
    assert list(pipeline.returns.columns) == list(mask.columns)

    # Nothing ST is tradable while it is ST — no price, no return.
    assert not (pipeline.get_close().where(~mask).notna().any().any())
    assert not (pipeline.returns.where(~mask).notna().any().any())

    # ...and the mask is not the old whole-sample exclusion in disguise.
    ever_st = mask.columns[~mask.all(axis=0)]
    assert len(ever_st) > 0, "fixture no longer exercises the ST path"
    for asset in ever_st:
        assert mask[asset].iloc[0], (
            f"{asset} is excluded before its ST event — that is the look-ahead"
        )

    # Assets with no ST episode anywhere keep their price history.
    never_st = mask.columns[mask.all(axis=0)]
    assert len(never_st) > 0
    assert pipeline.get_close()[never_st].dropna(how="all").shape[1] > 0

    # The static flag is preserved for downstream consumers, but no longer
    # drives the universe: it is a whole-sample property.
    assert "is_st" in pipeline.metadata.columns


def test_pipeline_close_prices(data_pipeline):
    close = data_pipeline.get_close()
    # No all-NaN columns
    assert close.dropna(axis=1, how="all").shape[1] > 0


def test_pipeline_volume(data_pipeline):
    vol = data_pipeline.get_volume()
    valid = vol.dropna()
    assert (valid >= 0).all().all()


def test_remove_long_suspensions_drops_asset():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "asset"])
    close = [1.0, 2.0, None, 11.0, None, 12.0,
             None, 13.0, None, 14.0, 15.0, 16.0]
    df = pd.DataFrame({"close": close}, index=index)

    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.max_suspension_days = 3
    result = pipeline._remove_long_suspensions(df)

    assert "A" not in result.index.get_level_values("asset")
    assert "B" in result.index.get_level_values("asset")


def test_price_limit_flags_are_boolean(data_pipeline):
    prices = data_pipeline.prices
    assert "is_limit_up" in prices.columns
    assert "is_limit_down" in prices.columns
    assert prices["is_limit_up"].dtype == bool
    assert prices["is_limit_down"].dtype == bool
