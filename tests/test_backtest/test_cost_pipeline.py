"""R9: the cost-model main pipeline wiring.

`_load_data` must attach the share-volume panel the engine needs (next to
the raw close it already attached), and a failure to produce it must leave
the attribute absent (the cost model then warns and falls back to fixed
slippage) rather than inventing a panel.
"""

import logging

import pandas as pd
import pytest
import quant_platform.data.pipeline as pipeline_module
import quant_platform.main as cli
from quant_platform.utils.config import load_config


def _frames():
    dates = pd.bdate_range("2024-01-02", periods=3)
    assets = ["A", "B"]
    returns = pd.DataFrame(0.0, index=dates, columns=assets)
    benchmark = pd.Series(0.0, index=dates, name="benchmark_return")
    volume = pd.DataFrame(
        [[10_000.0, 20_000.0], [11_000.0, 21_000.0], [12_000.0, 22_000.0]],
        index=dates, columns=assets,
    )
    raw_close = pd.DataFrame(
        [[10.0, 20.0], [10.5, 20.5], [11.0, 21.0]],
        index=dates, columns=assets,
    )
    prices = pd.DataFrame({"A": [1.0]}, index=dates[-1:])
    return dates, assets, returns, benchmark, volume, raw_close, prices


class _FakePipeline:
    """Minimal DataPipeline double exposing the `_load_data` surface."""

    volume_error: Exception | None = None
    last_instance = None

    def __init__(self, **kwargs):
        (
            _dates, assets, self.returns, self.benchmark,
            self.volume, self.raw_close, self.factor_prices,
        ) = _frames()
        self.metadata = pd.DataFrame({"asset": assets})
        self.financials = pd.DataFrame()
        type(self).last_instance = self

    def run(self):
        pass

    def get_sector_panel(self):
        return None

    def has_industry_history(self):
        return False

    def get_execution_gates(self):
        return {}

    def get_factor_ohlc(self):
        return {}

    def get_factor_prices(self):
        return self.factor_prices

    def get_raw_close(self):
        return self.raw_close

    def get_volume(self):
        if type(self).volume_error is not None:
            raise type(self).volume_error
        return self.volume

    def get_turnover(self):
        return None


def _config():
    return load_config(raw={
        "universe": {"n_stocks": 2},
        "data": {
            "provider": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        },
    })


@pytest.fixture
def fake_pipeline(monkeypatch):
    def install(pipeline_cls):
        monkeypatch.setattr(pipeline_module, "DataPipeline", pipeline_cls)
        monkeypatch.setattr(
            cli, "_resolve_provider",
            lambda *args, **kwargs: {
                "provider": object(),
                "provider_name": "synthetic",
                "requested_provider": "synthetic",
                "effective_provider": "synthetic",
                "identity": {"synthetic": True, "adjustment": "hfq"},
                "universe": ["A", "B"],
            },
        )
    return install


def test_load_data_attaches_daily_volume_panel(fake_pipeline):
    fake_pipeline(_FakePipeline)

    _prices, _returns, _benchmark, metadata, _financials, _turnover = (
        cli._load_data(_config())
    )

    volume = metadata.attrs.get("daily_volume")
    assert volume is not None
    # The panel mounted is exactly what pipeline.get_volume() returns.
    assert volume is _FakePipeline.last_instance.volume
    # Share volume, not a derived CNY figure: the engine does the conversion.
    assert volume.iloc[0].tolist() == [10_000.0, 20_000.0]
    assert metadata.attrs["raw_prices"]["close"] is (
        _FakePipeline.last_instance.raw_close
    )


def test_load_data_volume_failure_is_loud_and_fabricates_nothing(
    fake_pipeline, caplog,
):
    class ExplodingVolume(_FakePipeline):
        volume_error = RuntimeError("volume source down")

    fake_pipeline(ExplodingVolume)

    with caplog.at_level(logging.WARNING):
        _prices, _returns, _benchmark, metadata, _financials, _turnover = (
            cli._load_data(_config())
        )

    assert "daily_volume" not in metadata.attrs
    assert any(
        "Trading volume unavailable" in record.getMessage()
        for record in caplog.records
    )
