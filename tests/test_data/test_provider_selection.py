"""`config.data.provider` must be the selector, and real providers fail closed.

`_load_data` defaulted `use_tushare=True` and tested
`configured_provider == "tushare" or use_tushare`, so a Tushare token in the
environment replaced a configured synthetic/postgres provider.

A configured real provider that cannot be constructed must raise
`DataProviderUnavailableError` instead of silently serving synthetic data;
synthetic is used only after an explicit `allow_synthetic_fallback` opt-in
(audit S8/D08).
"""

from pathlib import Path

import pytest
import quant_platform.main as main_module
from quant_platform.main import DataProviderUnavailableError, _load_data
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
    """An unavailable configured provider fails closed instead of going synthetic."""
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

    with pytest.raises(DataProviderUnavailableError) as exc_info:
        _load_data(_small_config("tushare"))

    assert constructed == ["tushare"]  # attempted, then refused to go synthetic
    message = str(exc_info.value)
    assert "tushare" in message
    assert "offline in tests" in message
    assert "--allow-synthetic-fallback" in message


def test_explicit_fallback_opt_in_allows_synthetic(monkeypatch):
    """`allow_synthetic_fallback` is the only way a failed real provider goes synthetic."""
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
    config = _small_config("tushare")
    config.data.allow_synthetic_fallback = True

    prices, *_ = _load_data(config)

    assert constructed == ["tushare"]
    assert not prices.empty
    assert main_module._LAST_DATA_PROVIDER == "synthetic"
    provenance = main_module._LAST_DATA_PROVENANCE
    assert provenance["requested_provider"] == "tushare"
    assert provenance["effective_provider"] == "synthetic"
    assert provenance["synthetic"] is True
    assert provenance["fallback_allowed"] is True
