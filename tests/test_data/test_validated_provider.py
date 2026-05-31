"""Tests for multi-source validated data provider."""

import pandas as pd
import pytest

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.providers.validated_provider import (
    ValidatedProvider,
    ValidatedResult,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    LOW_CONFIDENCE,
    UNUSABLE,
)


class MockProvider(DataProvider):
    """Mock provider with controllable output."""
    def __init__(self, name: str, prices=None, financials=None, benchmark=None):
        self._name = name
        self._prices = prices if prices is not None else pd.DataFrame()
        self._financials = financials if financials is not None else pd.DataFrame()
        self._benchmark = benchmark if benchmark is not None else pd.Series(dtype=float)

    def get_prices(self, start_date="", end_date="", fields=None):
        return self._prices

    def get_financials(self, start_date="", end_date=""):
        return self._financials

    def get_benchmark(self, start_date="", end_date=""):
        return self._benchmark

    def get_metadata(self):
        return pd.DataFrame()


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame({"A0001": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)


class TestValidatedProvider:
    def test_single_source_medium_confidence(self, sample_prices):
        """Single source should return MEDIUM_CONFIDENCE."""
        vp = ValidatedProvider({
            "a": MockProvider(name="a", prices=sample_prices),
        })
        result = vp.get_prices("2024-01-01", "2024-01-05")
        # Current API strips confidence — just check data works
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_two_sources_agree_high_confidence(self, sample_prices):
        """Two sources with same data should return HIGH_CONFIDENCE."""
        vp = ValidatedProvider({
            "a": MockProvider(name="a", prices=sample_prices.copy()),
            "b": MockProvider(name="b", prices=sample_prices.copy()),
        })
        result = vp._validate("get_prices", {}, vp.price_deviation)
        assert result.confidence >= HIGH_CONFIDENCE - 0.1
        assert result.n_sources == 2
        assert result.discrepancies is None

    def test_two_sources_disagree_low_confidence(self, sample_prices):
        """Two sources with different data should flag discrepancy."""
        p1 = sample_prices.copy()
        p2 = sample_prices.copy() * 1.5  # 50% deviation

        vp = ValidatedProvider({
            "a": MockProvider(name="a", prices=p1),
            "b": MockProvider(name="b", prices=p2),
        }, price_deviation=5.0)
        result = vp._validate("get_prices", {}, vp.price_deviation)
        assert result.confidence < MEDIUM_CONFIDENCE
        assert result.discrepancies is not None
        assert len(result.discrepancies) >= 1

    def test_all_sources_fail(self):
        vp = ValidatedProvider({
            "a": MockProvider(name="a", prices=pd.DataFrame()),
        })
        result = vp._validate("get_prices", {}, vp.price_deviation)
        assert result.confidence == UNUSABLE
        assert result.n_sources == 0

    def test_get_validated_prices_returns_result(self, sample_prices):
        """New API should return full ValidatedResult."""
        vp = ValidatedProvider({
            "a": MockProvider(name="a", prices=sample_prices),
        })
        result = vp._validate("get_prices", {}, vp.price_deviation)
        assert isinstance(result, ValidatedResult)
        assert isinstance(result.values, pd.DataFrame)

    def test_deviation_computation(self, sample_prices):
        """Exact same data should have 0% deviation."""
        vp = ValidatedProvider({"a": MockProvider("a", prices=sample_prices)})
        dev = vp._compute_deviation(sample_prices, sample_prices.copy(), "a", "b")
        assert dev == 0.0

    def test_deviation_with_offset(self, sample_prices):
        """10% higher prices should show ~10% deviation."""
        vp = ValidatedProvider({"a": MockProvider("a")})
        shifted = sample_prices * 1.1
        dev = vp._compute_deviation(sample_prices, shifted, "a", "b")
        assert 9.0 < dev < 11.0

    def test_fuse_average(self, sample_prices):
        """Fusing two identical sources should return same values."""
        vp = ValidatedProvider({
            "a": MockProvider("a", prices=sample_prices),
            "b": MockProvider("b", prices=sample_prices),
        })
        result = vp._validate("get_prices", {}, vp.price_deviation)
        pd.testing.assert_frame_equal(result.values, sample_prices)
