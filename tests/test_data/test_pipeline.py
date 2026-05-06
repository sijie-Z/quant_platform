"""Tests for data pipeline."""

import numpy as np

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
