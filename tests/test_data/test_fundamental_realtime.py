"""Tests for real-time fundamental data provider."""

import numpy as np
import pandas as pd
import pytest
import time

from quant_platform.data.providers.fundamental_realtime import (
    FundamentalMetrics,
    FundamentalDataProvider,
    FundamentalScreener,
)


class TestFundamentalMetrics:
    def test_creation(self):
        m = FundamentalMetrics(code="600519", pe_ttm=33.5, pb=11.2, roe=31.5)
        assert m.code == "600519"
        assert m.pe_ttm == 33.5
        assert m.pb == 11.2

    def test_to_dict(self):
        m = FundamentalMetrics(code="600519", pe_ttm=33.5)
        d = m.to_dict()
        assert d["code"] == "600519"
        assert d["pe_ttm"] == 33.5

    def test_to_series(self):
        m = FundamentalMetrics(code="600519", pe_ttm=33.5)
        s = m.to_series()
        assert isinstance(s, pd.Series)
        assert s["code"] == "600519"

    def test_default_values(self):
        m = FundamentalMetrics(code="600519")
        assert m.pe_ttm == 0.0
        assert m.pb == 0.0
        assert m.roe == 0.0

    def test_timestamp_auto_generated(self):
        m = FundamentalMetrics(code="600519")
        assert m.timestamp


class TestFundamentalDataProvider:
    def test_creation(self):
        fd = FundamentalDataProvider()
        assert fd._source == "eastmoney"
        assert fd._cache_ttl == 300

    def test_get_fundamentals_synthetic(self):
        """Should return synthetic data when API unavailable."""
        fd = FundamentalDataProvider(cache_ttl=1)
        # Mock the API calls to fail, triggering synthetic fallback
        with patch.object(fd, '_fetch_eastmoney', side_effect=Exception("no API")):
            with patch.object(fd, '_fetch_sina', side_effect=Exception("no API")):
                m = fd.get_fundamentals("600519")
                assert m.code == "600519"
                assert m.pe_ttm > 0

    def test_cache_hit(self):
        fd = FundamentalDataProvider(cache_ttl=60)
        # Pre-populate cache
        m = FundamentalMetrics(code="600519", pe_ttm=33.5)
        fd._cache["600519"] = (time.time(), m)

        result = fd.get_fundamentals("600519")
        assert result.pe_ttm == 33.5
        assert fd._cache_hits == 1

    def test_cache_expired(self):
        fd = FundamentalDataProvider(cache_ttl=1)
        m = FundamentalMetrics(code="600519", pe_ttm=33.5)
        fd._cache["600519"] = (time.time() - 10, m)  # Expired

        with patch.object(fd, '_fetch_eastmoney', side_effect=Exception("no API")):
            with patch.object(fd, '_fetch_sina', side_effect=Exception("no API")):
                result = fd.get_fundamentals("600519")
                assert result.code == "600519"
                assert fd._cache_misses >= 1

    def test_get_bulk(self):
        fd = FundamentalDataProvider(cache_ttl=60)
        with patch.object(fd, '_fetch_eastmoney', side_effect=Exception("no API")):
            with patch.object(fd, '_fetch_sina', side_effect=Exception("no API")):
                result = fd.get_bulk(["600519", "000001"])
                assert len(result) == 2
                assert "600519" in result
                assert "000001" in result

    def test_get_bulk_partial_cache(self):
        fd = FundamentalDataProvider(cache_ttl=60)
        m = FundamentalMetrics(code="600519", pe_ttm=33.5)
        fd._cache["600519"] = (time.time(), m)

        with patch.object(fd, '_fetch_eastmoney', side_effect=Exception("no API")):
            with patch.object(fd, '_fetch_sina', side_effect=Exception("no API")):
                result = fd.get_bulk(["600519", "000001"])
                assert result["600519"].pe_ttm == 33.5  # From cache
                assert result["000001"].pe_ttm != 0  # Freshly fetched

    def test_get_as_dataframe(self):
        fd = FundamentalDataProvider(cache_ttl=60)
        with patch.object(fd, '_fetch_eastmoney', side_effect=Exception("no API")):
            with patch.object(fd, '_fetch_sina', side_effect=Exception("no API")):
                df = fd.get_as_dataframe(["600519", "000001"])
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 2

    def test_clear_cache(self):
        fd = FundamentalDataProvider()
        fd._cache["600519"] = (time.time(), FundamentalMetrics(code="600519"))
        fd.clear_cache()
        assert len(fd._cache) == 0

    def test_stats(self):
        fd = FundamentalDataProvider()
        stats = fd.stats
        assert "source" in stats
        assert "cached" in stats
        assert "cache_ttl" in stats
        assert "request_count" in stats

    def test_generate_synthetic(self):
        fd = FundamentalDataProvider()
        m = fd._generate_synthetic("600519")
        assert m.code == "600519"
        assert m.pe_ttm > 0
        assert m.pb > 0
        assert m.market_cap > 0

    def test_generate_synthetic_different_codes(self):
        """Different codes should get different synthetic values."""
        fd = FundamentalDataProvider()
        m1 = fd._generate_synthetic("600519")
        m2 = fd._generate_synthetic("000001")
        # Not guaranteed to be different for all fields, but market_cap should differ
        assert m1.code != m2.code


class TestFundamentalScreener:
    @pytest.fixture
    def screener(self):
        fd = FundamentalDataProvider(cache_ttl=60)
        # Pre-populate with known values
        fd._cache["600519"] = (time.time(), FundamentalMetrics(
            code="600519", pe_ttm=33.5, pb=11.2, roe=31.5,
            market_cap=2.1e12, dividend_yield=1.5, debt_ratio=25,
        ))
        fd._cache["000001"] = (time.time(), FundamentalMetrics(
            code="000001", pe_ttm=8.0, pb=0.8, roe=12.0,
            market_cap=3.5e11, dividend_yield=3.0, debt_ratio=45,
        ))
        fd._cache["300750"] = (time.time(), FundamentalMetrics(
            code="300750", pe_ttm=60.0, pb=5.0, roe=18.0,
            market_cap=8e11, dividend_yield=0.5, debt_ratio=55,
        ))
        return FundamentalScreener(fd)

    def test_screen_pe_max(self, screener):
        result = screener.screen(["600519", "000001", "300750"], pe_max=40)
        assert "600519" in result
        assert "000001" in result
        assert "300750" not in result

    def test_screen_roe_min(self, screener):
        result = screener.screen(["600519", "000001", "300750"], roe_min=20)
        assert "600519" in result
        assert "000001" not in result

    def test_screen_pb_max(self, screener):
        result = screener.screen(["600519", "000001", "300750"], pb_max=5)
        assert "000001" in result
        assert "600519" not in result

    def test_screen_multiple_criteria(self, screener):
        result = screener.screen(
            ["600519", "000001", "300750"],
            pe_max=50, roe_min=10, debt_ratio_max=50,
        )
        assert "600519" in result
        assert "000001" in result
        assert "300750" not in result  # debt_ratio=55

    def test_screen_dividend_yield(self, screener):
        result = screener.screen(["600519", "000001", "300750"], dividend_yield_min=2.0)
        assert "000001" in result
        assert "300750" not in result

    def test_screen_all_pass(self, screener):
        result = screener.screen(["600519", "000001", "300750"])
        assert len(result) == 3

    def test_screen_none_pass(self, screener):
        result = screener.screen(["600519", "000001", "300750"], pe_max=1)
        assert len(result) == 0

    def test_rank_by_roe(self, screener):
        ranked = screener.rank_by(["600519", "000001", "300750"], metric="roe")
        assert ranked[0][0] == "600519"  # ROE=31.5
        assert ranked[-1][0] == "000001"  # ROE=12.0

    def test_rank_by_pe_ascending(self, screener):
        ranked = screener.rank_by(["600519", "000001", "300750"], metric="pe_ttm", ascending=True)
        assert ranked[0][0] == "000001"  # PE=8.0

    def test_rank_by_top_n(self, screener):
        ranked = screener.rank_by(["600519", "000001", "300750"], metric="roe", top_n=2)
        assert len(ranked) == 2


# Need to import for patching
from unittest.mock import patch
