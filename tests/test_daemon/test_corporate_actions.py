"""Daemon: the daily LiveEngine run loads and applies the corporate-action feed.

The daemon constructed ``LiveEngine()`` without a feed, so a split/dividend
was never applied to the persisted paper book.  The feed now comes from the
same ``DataPipeline`` that produced the price panel, is applied as replayed
exchange days advance, and is reported in the run result
(``corporate_actions_applied``: count / last ex-date / source).  A pipeline
without a feed must keep the old behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_platform.daemon.runner as runner_mod
from quant_platform.core.store import Store
from quant_platform.daemon.runner import Daemon


def _make_data(n_days=200, n_assets=4, actions=None):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=assets,
    )
    closes = 10.0 * (1 + returns).cumprod()
    frame = pd.DataFrame(actions or [], columns=[
        "ex_date", "asset", "kind", "cash_per_share", "split_ratio",
    ])

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.returns = returns
            self._closes = closes
            self.provider = object()
            self.corporate_actions_available = bool(len(frame))

        def run(self):
            pass

        def get_close(self):
            return self._closes

        def get_corporate_actions(self):
            return frame.copy()

    return FakePipeline, dates


def _make_daemon(store, monkeypatch, actions=None):
    fake_pipeline, dates = _make_data(actions=actions)
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


def test_daemon_applies_pipeline_feed_and_reports_provenance(
    store, monkeypatch,
):
    _, dates = _make_data()
    ex_date = str(dates[-1])[:10]
    daemon, _ = _make_daemon(store, monkeypatch, actions=[{
        "ex_date": ex_date, "asset": "S00", "kind": "split",
        "cash_per_share": 0.0, "split_ratio": 2.0,
    }])
    store.save_engine_state({
        "version": 2,
        "cash": 9_999_000.0,
        "positions": [{
            "asset": "S00", "entry_date": "2024-01-02",
            "exit_date": "2099-01-01", "shares": 100, "entry_price": 10.0,
        }],
        "equity_peak": 10_000_000.0,
        "current_equity": 10_000_000.0,
        "total_trades": 7,
        "last_rebalance": ex_date,
        "next_rebalance": "2024-08-01",
        "vol_series": [],
        "status": "running",
        "last_processed_date": str(dates[-2])[:10],
        "day_count": 42,
    }, "live_engine_alpha_v1")

    result = daemon.run_once()

    assert result["status"] == "ok"
    snapshot = store.get_engine_state(daemon.strategy_id)
    position = snapshot["positions"][0]
    assert position["shares"] == 200, "the daemon did not apply the split"
    assert position["entry_price"] == pytest.approx(5.0)
    assert snapshot["applied_corporate_actions"] == [["S00", ex_date]]
    assert result["corporate_actions_feed_rows"] == 1
    assert result["corporate_actions_applied"] == {
        "count": 1,
        "last_ex_date": ex_date,
        "source": result["corporate_actions_applied"]["source"],
    }
    assert result["corporate_actions_applied"]["source"].startswith(
        "pipeline:"
    )


def test_daemon_catch_up_does_not_pay_positions_entered_after_the_ex_date(
    store, monkeypatch,
):
    _, dates = _make_data()
    ex_date = str(dates[-1])[:10]
    daemon, _ = _make_daemon(store, monkeypatch, actions=[{
        "ex_date": ex_date, "asset": "S00", "kind": "split",
        "cash_per_share": 0.0, "split_ratio": 2.0,
    }])
    store.save_engine_state({
        "version": 2,
        "cash": 9_999_000.0,
        # Entered *on* the ex-date: not entitled to a caught-up action.
        "positions": [{
            "asset": "S00", "entry_date": ex_date,
            "exit_date": "2099-01-01", "shares": 100, "entry_price": 10.0,
        }],
        "equity_peak": 10_000_000.0,
        "current_equity": 10_000_000.0,
        "total_trades": 7,
        "last_rebalance": ex_date,
        "next_rebalance": "2024-08-01",
        "vol_series": [],
        "status": "running",
        "last_processed_date": str(dates[-2])[:10],
        "day_count": 42,
    }, "live_engine_alpha_v1")

    result = daemon.run_once()

    assert result["status"] == "ok"
    snapshot = store.get_engine_state(daemon.strategy_id)
    assert snapshot["positions"][0]["shares"] == 100, (
        "a caught-up past action paid a position entered on the ex-date"
    )
    event = result["corporate_actions_applied"]
    assert event["count"] == 1  # the ex-date was processed ...
    assert event["last_ex_date"] == ex_date


def test_daemon_without_a_feed_runs_unchanged(store, monkeypatch):
    fake_pipeline, dates = _make_data()

    class NoFeedPipeline(fake_pipeline):
        get_corporate_actions = None

    monkeypatch.setattr(
        runner_mod, "DataPipeline", lambda **kwargs: NoFeedPipeline(**kwargs),
    )
    monkeypatch.setattr(
        runner_mod, "BaostockDataProvider", lambda **kwargs: object(),
    )
    daemon = Daemon(store=store)

    result = daemon.run_once()

    assert result["status"] == "ok"
    assert result["corporate_actions_feed_rows"] == 0
    assert result["corporate_actions_applied"]["count"] == 0
    snapshot = store.get_engine_state(daemon.strategy_id)
    assert snapshot["applied_corporate_actions"] == []
