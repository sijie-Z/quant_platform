"""Tests for synthetic data provider."""

import pandas as pd
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.data.schema import SECTORS


def test_provider_creation():
    provider = SyntheticDataProvider(n_stocks=50, start_date="2023-01-01", end_date="2023-12-31")
    assert provider.n_stocks == 50
    assert provider.start_date == pd.Timestamp("2023-01-01")


def test_get_prices(synthetic_provider):
    prices = synthetic_provider.get_prices("2023-01-01", "2023-06-30")
    assert prices is not None
    assert len(prices) > 0
    assert isinstance(prices.index, pd.MultiIndex)
    assert "close" in prices.columns
    assert "volume" in prices.columns


def test_price_columns(synthetic_provider):
    prices = synthetic_provider.get_prices("2023-01-01", "2024-12-31")
    expected = {"open", "high", "low", "close", "volume", "turnover", "adj_factor", "vwap"}
    assert expected.issubset(set(prices.columns))


def test_get_financials(synthetic_provider):
    fin = synthetic_provider.get_financials("2023-01-01", "2023-12-31")
    assert fin is not None
    assert "market_cap" in fin.columns
    assert "roe" in fin.columns
    assert "pb_ratio" in fin.columns


def test_get_benchmark(synthetic_provider):
    bench = synthetic_provider.get_benchmark("2023-01-01", "2023-12-31")
    assert len(bench) > 0
    assert bench.name == "benchmark"


def test_get_metadata(synthetic_provider):
    meta = synthetic_provider.get_metadata()
    assert len(meta) == 100
    assert "sector" in meta.columns
    assert "market_cap_group" in meta.columns
    assert "is_st" in meta.columns
    # All sectors should be valid
    assert all(s in SECTORS for s in meta["sector"])


def test_prices_no_negative(synthetic_provider):
    prices = synthetic_provider.get_prices("2023-01-01", "2024-12-31")
    for col in ["open", "high", "low", "close"]:
        valid = prices[col].dropna()
        assert (valid > 0).all(), f"{col} has non-positive values"


def test_reproducibility():
    """Same seed should produce identical data."""
    p1 = SyntheticDataProvider(n_stocks=20, seed=42)
    p2 = SyntheticDataProvider(n_stocks=20, seed=42)
    prices1 = p1.get_prices("2023-01-01", "2023-12-31")
    prices2 = p2.get_prices("2023-01-01", "2023-12-31")
    pd.testing.assert_frame_equal(prices1, prices2)


def test_returns_in_range(synthetic_provider):
    """Daily returns should be within price limit bounds (-10%, +10%)."""
    prices = synthetic_provider.get_prices("2023-01-01", "2024-12-31")
    close = prices["close"].unstack("asset")
    returns = close.pct_change(fill_method=None)
    valid = returns.dropna()
    assert (valid >= -0.11).all().all()  # Allow small numerical error
    assert (valid <= 0.11).all().all()
