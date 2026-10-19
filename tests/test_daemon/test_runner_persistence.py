"""Daemon state persistence across runs and process restarts (issues #3/#4).

The daemon used to rebuild `LiveEngine` from scratch on every run, so cash
reset to 10M and positions to empty. These tests pin the persisted-state
contract and the processed-date watermark.
"""

import numpy as np
import pandas as pd
import pytest

import quant_platform.daemon.runner as runner_mod
from quant_platform.core.store import Store
from quant_platform.daemon.runner import Daemon


def _make_data(n_days=200, n_assets=10):
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


def _make_daemon(store, monkeypatch):
    fake_pipeline, dates = _make_data()
    monkeypatch.setattr(
        runner_mod, "DataPipeline", lambda **kwargs: fake_pipeline(**kwargs),
    )
    monkeypatch.setattr(
        runner_mod, "BaostockDataProvider", lambda **kwargs: object(),
    )
    return Daemon(store=store), dates


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "daemon.db"))


def test_run_persists_state_and_does_not_replay_the_same_day(store, monkeypatch):
    daemon, dates = _make_daemon(store, monkeypatch)

    first = daemon.run_once()

    assert first["status"] == "ok"
    snapshot = store.get_engine_state(daemon.strategy_id)
    assert snapshot is not None
    assert snapshot["last_processed_date"] == str(dates[-1])[:10]

    # A new process (new Daemon object, same store) sees the same day again.
    second_daemon, _ = _make_daemon(store, monkeypatch)
    second = second_daemon.run_once()

    assert second["status"] == "ok"
    after = store.get_engine_state(second_daemon.strategy_id)
    assert after["last_processed_date"] == snapshot["last_processed_date"]
    assert after["day_count"] == snapshot["day_count"]
    assert after["cash"] == snapshot["cash"]
    assert after["total_trades"] == snapshot["total_trades"]


def test_restart_rehydrates_positions_cash_and_watermark(store, monkeypatch):
    _, dates = _make_data()
    seeded = {
        "version": 1,
        "cash": 9_999_000.0,
        "positions": [{
            "asset": "S00", "entry_date": "2024-01-02",
            "exit_date": "2099-01-01", "shares": 100, "entry_price": 10.0,
        }],
        "equity_peak": 10_000_000.0,
        "current_equity": 10_000_000.0,
        "total_trades": 7,
        "last_rebalance": "2024-05-01",
        "next_rebalance": "2024-08-01",
        "vol_series": [],
        "status": "running",
        "last_processed_date": str(dates[-2])[:10],
        "day_count": 42,
        "safety": {"kill_switch_triggered": False},
    }
    store.save_engine_state(seeded, "live_engine_alpha_v1")

    daemon, _ = _make_daemon(store, monkeypatch)
    result = daemon.run_once()

    assert result["status"] == "ok"
    snapshot = store.get_engine_state(daemon.strategy_id)
    assert snapshot["day_count"] == 43, "exactly one new day should be processed"
    assert snapshot["cash"] == pytest.approx(9_999_000.0)
    assert snapshot["positions"][0]["asset"] == "S00"
    assert snapshot["total_trades"] == 7
    assert snapshot["next_rebalance"] == "2024-08-01"


def test_killed_snapshot_blocks_trading(store, monkeypatch):
    seeded = {"version": 1, "status": "killed", "cash": 1.0}
    store.save_engine_state(seeded, "live_engine_alpha_v1")

    daemon, _ = _make_daemon(store, monkeypatch)
    result = daemon.run_once()

    assert result["status"] == "killed"
    assert store.get_engine_state(daemon.strategy_id) == seeded


def test_corrupt_snapshot_is_not_overwritten(store, monkeypatch):
    store.save_engine_state({"version": 99}, "live_engine_alpha_v1")

    daemon, _ = _make_daemon(store, monkeypatch)
    result = daemon.run_once()

    assert result["status"] == "state_corrupt"
    assert store.get_engine_state("live_engine_alpha_v1") == {"version": 99}
