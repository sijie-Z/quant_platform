"""Shared test fixtures for the quant platform."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider


@pytest.fixture(scope="session")
def synthetic_provider():
    """Session-scoped synthetic data provider (small scale for tests)."""
    return SyntheticDataProvider(
        n_stocks=100,
        start_date="2023-01-01",
        end_date="2024-12-31",
        seed=123,
        embedded_alpha=True,  # Keep embedded alpha for test data stability
    )


@pytest.fixture(scope="session")
def data_pipeline(synthetic_provider):
    """Session-scoped data pipeline with pre-run data."""
    pipeline = DataPipeline(
        provider=synthetic_provider,
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    pipeline.run()
    return pipeline


@pytest.fixture
def prices(data_pipeline):
    """Close prices DataFrame (date x asset)."""
    return data_pipeline.get_close()


@pytest.fixture
def returns(data_pipeline):
    """Daily returns DataFrame (date x asset)."""
    return data_pipeline.returns


@pytest.fixture
def benchmark(data_pipeline):
    """Benchmark daily returns."""
    return data_pipeline.benchmark


@pytest.fixture
def metadata(data_pipeline):
    """Stock metadata."""
    return data_pipeline.metadata


@pytest.fixture
def financials(data_pipeline):
    """Financial data (unstacked)."""
    return data_pipeline.financials.unstack("asset")


@pytest.fixture
def sample_factor(prices):
    """A sample factor: 20-day returns (momentum proxy)."""
    return prices.pct_change(fill_method=None).rolling(20).apply(lambda x: (1 + x).prod() - 1)
