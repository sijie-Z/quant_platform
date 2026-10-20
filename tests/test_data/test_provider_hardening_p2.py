"""Regression tests for the P2 provider hardening fixes.

Covers D21 (Adata contract), D24 (synthetic ST accessors / fields), D37
(connection-pool empty-frame failover, provider-identity cache key), D38
(fundamental provenance + strict mode) and D43 (bounded WebSocket
reconnects with backoff and a health signal).

No network access.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pandas as pd
import pytest
from quant_platform.data.providers.adata_provider import AdataProvider
from quant_platform.data.providers.base import DataFetchIncompleteError
from quant_platform.data.providers.connection_pool import (
    CACHE_SCHEMA_VERSION,
    DataProviderPool,
)
from quant_platform.data.providers.fundamental_realtime import (
    FundamentalDataProvider,
)
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.data.providers.websocket_provider import WebSocketQuoteProvider

# ----------------------------------------------------------------------
# D24: synthetic provider
# ----------------------------------------------------------------------

def test_fresh_instance_st_timeseries_does_not_raise():
    provider = SyntheticDataProvider(
        n_stocks=3, start_date="2024-01-01", end_date="2024-03-31", seed=7,
    )
    st = provider.get_st_timeseries()
    assert isinstance(st, pd.DataFrame)
    assert {"asset", "announce_date", "end_date", "is_st"} <= set(st.columns)


def test_empty_st_timeseries_keeps_end_date():
    provider = SyntheticDataProvider(
        n_stocks=2, start_date="2024-01-01", end_date="2024-03-31", seed=7,
    )
    provider._dates = pd.DatetimeIndex(
        pd.bdate_range("2024-01-01", "2024-03-31"), name="date"
    )
    provider._assets = ["000001.SH", "000002.SH"]
    provider._metadata = pd.DataFrame(
        {"is_st": [False, False]}, index=pd.Index(provider._assets, name="asset"),
    )

    st = provider._generate_st_timeseries()

    assert st.empty
    assert "end_date" in st.columns


def test_synthetic_fields_contract():
    provider = SyntheticDataProvider(
        n_stocks=5, start_date="2024-01-01", end_date="2024-03-31", seed=1,
    )
    sub = provider.get_prices(
        "2024-01-01", "2024-03-31", fields=["close"],
    )
    assert list(sub.columns) == ["close"]

    with pytest.raises(ValueError, match="Unknown price fields"):
        provider.get_prices("2024-01-01", "2024-03-31", fields=["nope"])


# ----------------------------------------------------------------------
# D21: adata provider
# ----------------------------------------------------------------------

class FakeAdataMarket:
    def __init__(self, fail_codes=(), ignore_end_date=False):
        self.fail_codes = set(fail_codes)
        self.ignore_end_date = ignore_end_date

    def get_market(self, stock_code, k_type=1, start_date="", end_date=None):
        if stock_code in self.fail_codes:
            raise RuntimeError("adata offline")
        end = "2024-01-10" if self.ignore_end_date or end_date is None else end_date
        dates = pd.bdate_range(start_date, end)
        return pd.DataFrame({
            "trade_date": [str(d.date()) for d in dates],
            "open": [10.0 + i for i in range(len(dates))],
            "high": [10.5 + i for i in range(len(dates))],
            "low": [9.5 + i for i in range(len(dates))],
            "close": [10.2 + i for i in range(len(dates))],
            "volume": [1000] * len(dates),
            "amount": [10_000.0] * len(dates),
        })

    def get_market_index(self, index_code="000300", start_date="", end_date=None):
        dates = pd.bdate_range(start_date, end_date or "2024-01-10")
        return pd.DataFrame({
            "trade_date": [str(d.date()) for d in dates],
            "close": [100.0 * (1.01 ** i) for i in range(len(dates))],
        })


class FakeAdata:
    def __init__(self, **kwargs):
        self.stock = types.SimpleNamespace(
            market=FakeAdataMarket(**kwargs),
            info=types.SimpleNamespace(all_code=lambda: pd.DataFrame({
                "stock_code": ["600001", "600002"],
                "short_name": ["AAA", "BBB"],
                "exchange": ["SH", "SH"],
            })),
        )


def adata_provider(**kwargs) -> AdataProvider:
    provider = AdataProvider()
    provider._adata = FakeAdata(**kwargs)
    return provider


def test_adata_prices_are_a_multiindex_panel_and_sliced():
    provider = adata_provider(ignore_end_date=True)

    prices = provider.get_prices("2024-01-02", "2024-01-04")

    assert isinstance(prices.index, pd.MultiIndex)
    assert list(prices.index.names) == ["date", "asset"]
    assert {"open", "high", "low", "close", "volume"} <= set(prices.columns)
    # The fake ignored end_date; the provider must slice locally.
    assert prices.index.get_level_values("date").max() == pd.Timestamp("2024-01-04")

    sub = provider.get_prices("2024-01-02", "2024-01-04", fields=["close"])
    assert list(sub.columns) == ["close"]
    with pytest.raises(ValueError, match="Unknown price fields"):
        provider.get_prices("2024-01-02", "2024-01-04", fields=["nope"])


def test_adata_symbol_failure_raises_incomplete():
    provider = adata_provider(fail_codes={"600002"})
    with pytest.raises(DataFetchIncompleteError):
        provider.get_prices("2024-01-02", "2024-01-04")


def test_adata_financials_raise_instead_of_returning_a_fake_panel():
    provider = adata_provider()
    with pytest.raises(NotImplementedError):
        provider.get_financials("2024-01-01", "2024-01-04")


def test_adata_metadata_contract():
    provider = adata_provider()
    meta = provider.get_metadata()
    assert meta.index.name == "asset"
    assert {"sector", "listing_date", "delisting_date"} <= set(meta.columns)
    assert "is_st" not in meta.columns
    assert provider.st_available is False


def test_adata_benchmark_is_sliced_series():
    provider = adata_provider()
    bench = provider.get_benchmark("2024-01-02", "2024-01-04")
    assert isinstance(bench, pd.Series)
    assert bench.index.max() <= pd.Timestamp("2024-01-04")
    assert bench.iloc[0] == pytest.approx(0.01)


# ----------------------------------------------------------------------
# D37: connection pool
# ----------------------------------------------------------------------

class PoolProvider:
    def __init__(self, name, frame=None, fail=False):
        self.name = name
        self.frame = frame
        self.fail = fail
        self.calls = 0

    def get_prices(self, start_date, end_date, **kwargs):
        self.calls += 1
        if self.fail:
            raise ConnectionError(f"{self.name} is down")
        return self.frame.copy()

    def get_financials(self, start_date, end_date, **kwargs):
        return self.get_prices(start_date, end_date, **kwargs)


def test_pool_fails_over_on_empty_frame():
    empty = PoolProvider("empty", pd.DataFrame())
    good = PoolProvider("good", pd.DataFrame({"close": [1.0, 2.0]}))
    pool = DataProviderPool(
        {"empty": empty, "good": good}, priorities={"empty": 1, "good": 2},
    )

    result = pool.get_prices("2024-01-01", "2024-01-02")

    assert len(result) == 2
    assert good.calls == 1
    health = pool.health_check()
    assert health["empty"]["failure_count"] == 1
    assert "empty" in health["empty"]["last_error"]


def test_pool_accepts_empty_from_the_only_source():
    only = PoolProvider("only", pd.DataFrame())
    pool = DataProviderPool({"only": only}, priorities={"only": 1})

    result = pool.get_prices("2024-01-01", "2024-01-02")

    assert result is not None and result.empty
    assert pool.health_check()["only"]["failure_count"] == 0


def test_pool_min_rows_triggers_failover():
    short = PoolProvider("short", pd.DataFrame({"close": [1.0]}))
    good = PoolProvider("good", pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
    pool = DataProviderPool(
        {"short": short, "good": good}, priorities={"short": 1, "good": 2},
        min_rows=2,
    )

    result = pool.get_prices("2024-01-01", "2024-01-02")

    assert len(result) == 3
    assert good.calls == 1


def test_pool_cache_key_includes_provider_identity_and_schema_version():
    provider = PoolProvider("p", pd.DataFrame({"close": [1.0]}))
    pool = DataProviderPool({"p": provider}, priorities={"p": 1})

    key = pool._make_cache_key("prices", "2024-01-01", "2024-01-02", {})

    assert CACHE_SCHEMA_VERSION in key
    assert "PoolProvider" in key


# ----------------------------------------------------------------------
# D38: fundamental realtime provenance / strict mode
# ----------------------------------------------------------------------

def _failing_eastmoney(code):
    raise RuntimeError("live API down")


def test_strict_mode_raises_instead_of_fabricating(monkeypatch):
    provider = FundamentalDataProvider(rate_limit=0.0, strict=True)
    monkeypatch.setattr(provider, "_fetch_eastmoney", _failing_eastmoney)

    with pytest.raises(RuntimeError, match="strict mode"):
        provider.get_fundamentals("600519")
    assert provider.stats["synthetic_count"] == 0


def test_provenance_survives_bulk_dataframe_and_to_frame(monkeypatch):
    provider = FundamentalDataProvider(rate_limit=0.0)
    monkeypatch.setattr(provider, "_fetch_eastmoney", _failing_eastmoney)

    metrics = provider.get_fundamentals("600519")
    assert metrics.is_synthetic is True
    assert metrics.source == "synthetic"

    frame = provider.get_as_dataframe(["600519"])
    assert bool(frame.loc["600519", "is_synthetic"]) is True
    assert frame.loc["600519", "source"] == "synthetic"

    single = metrics.to_frame()
    assert bool(single.loc[0, "is_synthetic"]) is True
    assert single.loc[0, "source"] == "synthetic"


# ----------------------------------------------------------------------
# D43: WebSocket reconnect bounds / backoff / health signal
# ----------------------------------------------------------------------

def test_websocket_reconnect_cap_defaults_to_finite():
    with patch(
        "quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True,
    ):
        ws = WebSocketQuoteProvider()
    assert ws._max_reconnect == 10


def test_websocket_backoff_is_exponential_and_capped():
    with patch(
        "quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True,
    ):
        ws = WebSocketQuoteProvider(
            reconnect_interval=5.0, max_reconnect=10, max_backoff=60.0,
        )
    ws._reconnect_count = 1
    assert ws._backoff_delay() == pytest.approx(5.0)
    ws._reconnect_count = 2
    assert ws._backoff_delay() == pytest.approx(10.0)
    ws._reconnect_count = 10
    assert ws._backoff_delay() == pytest.approx(60.0)


def test_websocket_loop_exhausts_and_reports_health():
    with patch(
        "quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True,
    ):
        ws = WebSocketQuoteProvider(reconnect_interval=0.0, max_reconnect=3)

    def boom():
        raise RuntimeError("endpoint down")

    ws._connect = boom
    ws._running = True
    ws._run_loop()

    assert ws._exhausted is True
    assert ws._running is False
    assert ws._reconnect_count == 3
    stats = ws.stats
    assert stats["status"] == "exhausted"
    assert stats["exhausted"] is True
    assert stats["last_error"] == "endpoint down"
