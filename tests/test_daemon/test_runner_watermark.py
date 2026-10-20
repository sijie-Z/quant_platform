"""S18: daemon watermark persistence and bounded catch-up.

The watermark used to live only in memory (a restart re-pulled the same day),
only the latest data row was ever processed (missed days after downtime were
never replayed), and a data-load failure was returned without any log/alert.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import quant_platform.daemon.runner as runner_mod
from quant_platform.core.store import Store
from quant_platform.daemon.runner import Daemon


def _make_data(n_days=200, n_assets=5):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=assets,
    )
    closes = 10.0 * (1 + returns).cumprod()

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.returns = returns
            self._closes = closes

        def run(self):
            pass

        def get_close(self):
            return self._closes

    return FakePipeline, dates


def _make_daemon(store, monkeypatch, **kwargs):
    fake_pipeline, dates = _make_data()
    monkeypatch.setattr(
        runner_mod, "DataPipeline", lambda **kw: fake_pipeline(**kw),
    )
    monkeypatch.setattr(
        runner_mod, "BaostockDataProvider", lambda **kw: object(),
    )
    return Daemon(store=store, **kwargs), dates


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "daemon.db"))


def test_watermark_survives_a_process_restart(store, monkeypatch):
    monkeypatch.setattr(runner_mod, "is_trading_day", lambda day: True)
    first, _ = _make_daemon(store, monkeypatch)
    assert first.run_once()["status"] == "ok"

    # A new process (new Daemon, same store/state dir) must not re-pull today.
    second, _ = _make_daemon(store, monkeypatch, run_hour=0)

    assert second.last_run_date == first.last_run_date
    assert second.should_run_today() is False


def test_catch_up_replays_every_unprocessed_exchange_day(store, monkeypatch):
    daemon, dates = _make_daemon(store, monkeypatch)
    seeded = {
        "version": 1,
        "cash": 10_000_000.0,
        "positions": [],
        "equity_peak": 10_000_000.0,
        "current_equity": 10_000_000.0,
        "total_trades": 0,
        "last_rebalance": str(dates[-5])[:10],
        "next_rebalance": "2099-01-01",
        "vol_series": [],
        "status": "running",
        "last_processed_date": str(dates[-5])[:10],
        "day_count": 10,
        "safety": {"kill_switch_triggered": False},
    }
    store.save_engine_state(seeded, "live_engine_alpha_v1")

    result = daemon.run_once()

    assert result["status"] == "ok"
    assert result["days_processed"] == 4
    snapshot = store.get_engine_state("live_engine_alpha_v1")
    assert snapshot["day_count"] == 14
    assert snapshot["last_processed_date"] == str(dates[-1])[:10]


def test_catch_up_is_bounded(store, monkeypatch):
    monkeypatch.setattr(runner_mod, "MAX_CATCHUP_DAYS", 2)
    daemon, dates = _make_daemon(store, monkeypatch)
    store.save_engine_state({
        "version": 1,
        "cash": 10_000_000.0,
        "positions": [],
        "equity_peak": 10_000_000.0,
        "current_equity": 10_000_000.0,
        "total_trades": 0,
        "last_rebalance": str(dates[0])[:10],
        "next_rebalance": "2099-01-01",
        "vol_series": [],
        "status": "running",
        "last_processed_date": str(dates[-11])[:10],
        "day_count": 0,
        "safety": {"kill_switch_triggered": False},
    }, "live_engine_alpha_v1")

    result = daemon.run_once()

    assert result["days_processed"] == 2
    assert any("bounded" in alert.lower() for alert in result["alerts"])


def test_data_failure_logs_an_alert(store, monkeypatch, caplog):
    class BrokenPipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            raise ConnectionError("provider down")

    monkeypatch.setattr(runner_mod, "DataPipeline", BrokenPipeline)
    monkeypatch.setattr(runner_mod, "BaostockDataProvider", lambda **kw: object())
    daemon = Daemon(store=store)

    # The "quant_platform" logger sets propagate=False once setup_logging()
    # has run in this process, so attach caplog's handler directly.
    logger = logging.getLogger(runner_mod.__name__)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.ERROR, logger=runner_mod.__name__):
            result = daemon.run_once()
    finally:
        logger.removeHandler(caplog.handler)

    assert result["status"] == "data_failed"
    assert any(
        "ALERT" in record.getMessage() and "data load failed" in record.getMessage()
        for record in caplog.records
    )
    # A failed data pull must not consume the day's watermark.
    assert daemon.last_run_date == ""
    assert not daemon._state_path.exists()
