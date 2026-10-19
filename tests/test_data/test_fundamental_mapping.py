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
    payload = {"data": {"f9": 25.0, "f23": 3.0, "f114": 24.0, "f115": 26.0,
                        "f116": 1_650_000_000_000.0, "f117": 1_200_000_000_000.0}}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(payload),
    )

    metrics = provider._fetch_eastmoney("600519")

    assert metrics.market_cap == 1_650_000_000_000.0
    assert metrics.circulating_cap == 1_200_000_000_000.0
    assert metrics.net_margin == 0.0
    assert metrics.revenue_growth == 0.0
    assert metrics.roe == 0.0
