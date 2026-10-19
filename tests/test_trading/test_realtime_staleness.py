"""A cached market frame must not be presented as a live quote."""

import pandas as pd
import pytest

import quant_platform.trading.realtime as rt_mod

pytestmark = pytest.mark.skipif(
    not rt_mod.HAS_AKSHARE, reason="akshare not installed",
)


def _snapshot_frame():
    return pd.DataFrame({
        "代码": ["600519"],
        "名称": ["贵州茅台"],
        "最新价": [100.0],
        "涨跌幅": [1.0],
        "涨跌额": [1.0],
        "成交量": [1000.0],
        "成交额": [100_000.0],
        "最高": [101.0],
        "最低": [98.0],
        "今开": [99.0],
        "昨收": [99.0],
        "换手率": [0.5],
        "市盈率-动态": [20.0],
        "市净率": [5.0],
        "总市值": [1e12],
        "流通市值": [1e12],
        "振幅": [3.0],
    })


def test_stale_cache_is_flagged_and_quotes_keep_the_fetch_time(monkeypatch):
    state = {"fail": False}

    def fetch():
        if state["fail"]:
            raise ConnectionError("east money unavailable")
        return _snapshot_frame()

    monkeypatch.setattr(
        rt_mod.ak, "stock_zh_a_spot_em", fetch, raising=False,
    )

    market = rt_mod.RealTimeMarket(cache_ttl=0)
    market.get_market_snapshot()
    fetched_at = market.snapshot_fetched_at

    assert fetched_at
    assert market.snapshot_is_stale is False

    state["fail"] = True
    market.get_market_snapshot()

    assert market.snapshot_is_stale is True
    # The timestamp stays at the original fetch; it must not advance to now.
    assert market.snapshot_fetched_at == fetched_at

    quotes = market.get_quotes(["600519"])
    assert quotes and quotes[0].timestamp == fetched_at
