"""The Tushare client must never send the token over plain HTTP.

tushare 1.4.x hard-codes ``http://api.waditu.com/dataapi``; the provider
forces the TLS endpoint before building the client so credentials do not
travel in clear text.
"""

from quant_platform.data.providers.tushare_loader import TushareProvider


def test_tushare_client_uses_https(monkeypatch):
    from tushare.pro.client import DataApi

    monkeypatch.setattr(DataApi, "_DataApi__http_url", "http://api.waditu.com/dataapi")
    provider = TushareProvider(token="test-token-12345")
    api = provider.ts_api
    assert api is not None
    assert DataApi._DataApi__http_url.startswith("https://")
