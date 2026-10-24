"""D8/D9/D10 regression: LiveRunner persistence, failure isolation, TCA.

D8: the runner wrote only P&L/session rows -- two executed orders and two
fills left orders=0, trades=0, signals=0 in the Store. D9: one broker error
aborted the whole multi-day run and `_end_session()` never ran, leaving the
session `active` and `_running=True` forever. D10: the dual-track
PaperBroker book was never seeded, so every paper order returned zero fills
and the TCA comparison was empty.
"""

from __future__ import annotations

import pytest
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.live_runner import LiveRunner


def _runner(tmp_path, dual_track: bool = False) -> LiveRunner:
    return LiveRunner(
        broker_type="simulated",
        initial_cash=10_000_000,
        dual_track=dual_track,
        risk_monitor=RiskMonitor(),
        store=Store(str(tmp_path / "runner.db")),
    )


def _fixed_buy_signals(target_value: float = 400_000.0):
    return lambda: [{
        "code": "600519",
        "side": "buy",
        "target_value": target_value,
        "strength": 0.6,
        "reason": "test signal",
    }]


def test_run_once_persists_signals_orders_and_trades(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.set_universe(["600519"])
    runner.set_prices({"600519": 1800.0})
    monkeypatch.setattr(runner, "_generate_signals", _fixed_buy_signals())

    report = runner.run_once(date="2026-01-15")

    assert report.n_orders == 1
    signals = runner._store.get_signals(limit=10)
    orders = runner._store.get_orders()
    trades = runner._store.get_trades()
    assert signals, "generated signals were not persisted"
    assert len(orders) == 1, "executed order was not persisted"
    assert orders[0]["status"] == "filled"
    assert len(trades) == 1, "fill was not persisted as a trade"
    assert trades[0]["order_id"] == orders[0]["order_id"]
    assert trades[0]["quantity"] == 200
    assert runner._store.get_events(topic="audit.order_filled")
    assert report.n_fills == 1


def test_run_survives_a_broker_error_and_always_closes_the_session(
    tmp_path, monkeypatch,
):
    runner = _runner(tmp_path)
    runner.set_universe(["600519", "000001", "002001"])
    monkeypatch.setattr(
        runner, "_generate_signals",
        lambda: [
            {"code": code, "side": "buy", "target_value": 100_000.0,
             "strength": 0.6, "reason": "test signal"}
            for code in runner._universe
        ],
    )

    state = {"raised": False}
    real_place = runner._broker.place_order

    def flaky_place(order):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("broker connection dropped")
        return real_place(order)

    monkeypatch.setattr(runner._broker, "place_order", flaky_place)

    report = runner.run(days=3)

    assert state["raised"], "the injected broker failure never fired"
    assert report.days_traded == 2, (
        "days after the failed one must still run"
    )
    assert runner._running is False, "the runner was left running"
    sessions = runner._store.get_sessions(limit=5)
    assert sessions and sessions[0]["status"] == "completed", (
        "the session must always be closed, even after a broker error"
    )


def test_dual_track_paper_broker_produces_fills(tmp_path, monkeypatch):
    runner = _runner(tmp_path, dual_track=True)
    runner.set_universe(["600519"])
    runner.set_prices({"600519": 1800.0})
    monkeypatch.setattr(
        runner, "_generate_signals", _fixed_buy_signals(400_000.0),
    )

    report = runner.run_once(date="2026-01-15")

    assert report.paper_metrics is not None
    assert report.paper_metrics["total_orders"] >= 1
    assert report.paper_metrics["total_trades"] > 0, (
        "the dual-track PaperBroker produced no fills; TCA would be empty"
    )
    assert report.paper_metrics["avg_fill_pct"] > 0
