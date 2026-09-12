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
    pipeline = DataPipeline(
        provider=synthetic_provider,
        start_date="2023-01-01",
        end_date="2024-12-31",
        exclude_st=True,
    )
    pipeline.run()
    meta = pipeline.metadata
    assert not meta["is_st"].any()


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
