"""Eastmoney field mapping must not report market cap as margins."""

import json

from quant_platform.data.providers.fundamental_realtime import FundamentalDataProvider


class _Response:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_eastmoney_maps_market_cap_fields(monkeypatch):
    provider = FundamentalDataProvider()
    # `stock/get` fields: f162 PE (x100), f167 PB (x100), f116/f117 market
    # caps. f114/f115 are not margins and must not be mapped to them.
    payload = {"data": {"f58": "贵州茅台", "f162": 2500.0, "f167": 300.0,
                        "f114": 24.0, "f115": 26.0,
                        "f116": 1_650_000_000_000.0, "f117": 1_200_000_000_000.0}}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(payload),
    )

    metrics = provider._fetch_eastmoney("600519")

    assert metrics.market_cap == 1_650_000_000_000.0
    assert metrics.circulating_cap == 1_200_000_000_000.0
    assert metrics.pe_ttm == 25.0
    assert metrics.pb == 3.0
    assert metrics.net_margin == 0.0
    assert metrics.revenue_growth == 0.0
    assert metrics.roe == 0.0
    assert metrics.source == "eastmoney"
