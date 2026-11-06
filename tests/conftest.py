"""Shared test fixtures for the quant platform."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# CI runs the `pytest` console script, which does not put the repo root on
# sys.path (unlike `python -m pytest`). Tests that import sibling helpers as
# `tests.test_data...` then fail during collection under importmode=importlib.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quant_platform.data.pipeline import DataPipeline  # noqa: E402
from quant_platform.data.providers.synthetic import SyntheticDataProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _keep_platform_logger_propagating():
    """Keep pytest's ``caplog`` working after any test calls setup_logging().

    ``setup_logging()`` sets ``propagate = False`` on the package logger, so
    once a logging/CLI test runs, caplog's root handler stops seeing platform
    records and later warning assertions fail depending on test order. Set
    propagation for each test and restore the previous value afterwards.
    """
    logger = logging.getLogger("quant_platform")
    previous = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = previous


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
