"""`config.data.provider` must be the selector.

`_load_data` defaulted `use_tushare=True` and tested
`configured_provider == "tushare" or use_tushare`, so a Tushare token in the
environment replaced a configured synthetic/postgres provider.
"""

from pathlib import Path

import pytest
from quant_platform.main import _load_data
from quant_platform.utils.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _small_config(provider: str):
    config = load_config(str(DEFAULT_CONFIG))
    config.data.provider = provider
    config.data.start_date = "2024-01-01"
    config.data.end_date = "2024-06-30"
    config.universe.n_stocks = 20
    return config


def test_synthetic_provider_is_not_replaced_by_a_tushare_token(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "dummy-token")
    constructed = []

    class FakeTushare:
        def __init__(self, *args, **kwargs):
            constructed.append("tushare")

    monkeypatch.setattr(
        "quant_platform.data.providers.tushare_loader.TushareProvider",
        FakeTushare,
    )

    prices, *_ = _load_data(_small_config("synthetic"))

    assert constructed == []
    assert not prices.empty


def test_a_configured_tushare_provider_is_attempted(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "dummy-token")
    constructed = []

    class FakeTushare:
        def __init__(self, *args, **kwargs):
            constructed.append("tushare")
            raise RuntimeError("offline in tests")

    monkeypatch.setattr(
        "quant_platform.data.providers.tushare_loader.TushareProvider",
        FakeTushare,
    )

    prices, *_ = _load_data(_small_config("tushare"))

    assert constructed == ["tushare"]  # attempted, then fell back to synthetic
    assert not prices.empty
