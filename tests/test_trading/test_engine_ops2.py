"""Ops wave 2 regressions — live engine start/stop lifecycle (S1/S4/S16)."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import quant_platform.main as main_module
from quant_platform.core.state_machine import PortfolioState
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine
from quant_platform.utils.metrics import get_metrics

TZ = ZoneInfo("Asia/Shanghai")


class FakeBroker:
    """Minimal broker that is NOT a SimulatedBroker.

    The engine loop only runs cycles outside market hours for simulated
    brokers; using this broker lets a pre-market test assert that no cycle
    (and therefore no P&L row) ran.
    """

    def __init__(self, equity: float = 500_000.0):
        self._equity = equity
        self.orders = []

    def connect(self):
        return True

    def get_account(self):
        return {
            "cash": self._equity, "total_equity": self._equity,
            "n_positions": 0, "connected": True,
        }

    def get_positions(self):
        return []

    def new_trading_day(self):
        pass

    def place_order(self, order):  # pragma: no cover - no signals in these tests
        self.orders.append(order)
        raise AssertionError("no order expected in this test")


def _engine(tmp_path, broker, monkeypatch, interval=300):
    engine = LiveTradingEngine(
        broker=broker,
        store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
        rebalance_interval=interval,
    )
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    return engine


def _clock(engine, value):
    engine._scheduler._now = lambda: value


def test_pre_market_start_does_not_enter_trading_or_write_eod(tmp_path, monkeypatch):
    engine = _engine(tmp_path, FakeBroker(), monkeypatch)
    _clock(engine, datetime(2026, 10, 20, 9, 0, tzinfo=TZ))

    engine.start()
    try:
        assert engine._sm.state in (
            PortfolioState.READY, PortfolioState.PRE_MARKET,
        )
        assert engine._sm.state not in (
            PortfolioState.TRADING, PortfolioState.POST_MARKET,
        )
        assert engine._store.get_pnl_history(
            start_date="2026-10-20", end_date="2026-10-20",
        ) == []
        assert engine._store.has_eod_reconciliation("2026-10-20") is False
    finally:
        engine.stop()

    # The exchange closes: one scheduler tick reconciles exactly once.
    _clock(engine, datetime(2026, 10, 20, 15, 1, tzinfo=TZ))
    engine._scheduler._tick()
    engine._scheduler._tick()

    rows = engine._store.get_pnl_history(
        start_date="2026-10-20", end_date="2026-10-20",
    )
    assert len(rows) == 1
    assert rows[0]["total_equity"] == pytest.approx(500_000.0)


def test_market_open_start_enters_trading(tmp_path, monkeypatch):
    engine = _engine(tmp_path, FakeBroker(), monkeypatch)
    _clock(engine, datetime(2026, 10, 20, 10, 0, tzinfo=TZ))

    engine.start()
    try:
        # The loop may already have started a cycle (TRADING -> REBALANCING);
        # either way the engine entered the trading session at the open.
        assert engine._sm.state in (
            PortfolioState.TRADING, PortfolioState.REBALANCING,
        )
    finally:
        engine.stop()


def test_stop_persists_state_even_when_the_scheduler_stop_fails(tmp_path, monkeypatch):
    """S16: scheduler.stop() raising used to skip the final snapshot/session."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)
    engine.start()
    original_stop = engine._scheduler.stop
    try:
        def _boom():
            raise RuntimeError("scheduler stop exploded")

        monkeypatch.setattr(engine._scheduler, "stop", _boom)
        engine.stop()
    finally:
        # Restore the real stop so this test cannot leave a scheduler thread
        # alive for the rest of the pytest process.
        engine._scheduler.stop = original_stop
        release = getattr(engine, "_thread", None)
        if release is not None and release.is_alive():
            engine.stop()
        engine._scheduler.stop()

    assert engine._thread is None or not engine._thread.is_alive()
    assert not engine._scheduler._thread or not engine._scheduler._thread.is_alive()
    assert engine._store.get_engine_state("live_engine:default") is not None
    sessions = engine._store.get_sessions()
    assert sessions
    assert sessions[0]["status"] == "stopped"
    assert sessions[0]["stopped_at"], "completed session row must carry stopped_at"


def test_stop_mid_cycle_leaves_a_completed_session_and_no_thread(tmp_path, monkeypatch):
    """Simulated SIGTERM mid-cycle: stop() waits, then persists everything."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)
    in_cycle = threading.Event()
    release = threading.Event()

    def blocking_fetch():
        in_cycle.set()
        release.wait(timeout=10)

    engine.start()
    try:
        _wait_until(
            lambda: engine._cycle_count >= 1 and not engine._cycle_lock.locked()
        )
        monkeypatch.setattr(engine, "_fetch_prices", blocking_fetch)
        threading.Thread(target=engine.run_once, daemon=True).start()
        assert in_cycle.wait(timeout=5)
        threading.Timer(0.2, release.set).start()
        engine.stop()
    finally:
        release.set()
        if engine._thread is not None and engine._thread.is_alive():
            engine.stop()

    assert engine._thread is None or not engine._thread.is_alive()
    assert engine._store.get_engine_state("live_engine:default") is not None
    session = engine._store.get_sessions()[0]
    assert session["status"] == "stopped"
    assert session["stopped_at"]


def test_engine_start_is_aborted_when_the_audit_write_fails(tmp_path, monkeypatch):
    """S16: the ENGINE_START audit is written before the loop thread exists."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)

    def _fail(*_args, **_kwargs):
        raise RuntimeError("store is gone")

    monkeypatch.setattr(engine._audit, "log", _fail)

    with pytest.raises(RuntimeError, match="ENGINE_START audit failed"):
        engine.start()

    assert engine._thread is None
    assert engine._running is False
    assert engine._scheduler._running is False
    session = engine._store.get_sessions()[0]
    assert session["status"] == "failed"


def test_cycle_heartbeat_gauges_are_exported(tmp_path, monkeypatch):
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)

    engine.run_once()

    snapshot = get_metrics().get_snapshot()
    assert snapshot["last_engine_cycle_timestamp"]["value"] > 0
    assert "trading_equity" in snapshot
    assert snapshot["risk_kill_switch_active"]["value"] in (0.0, 1.0)


def test_deactivation_after_stop_is_not_undone_by_stale_engine_state(
    tmp_path, monkeypatch,
):
    """S21 follow-up: risk_state is authoritative over the engine snapshot."""
    db = str(tmp_path / "deactivate.db")
    engine1 = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=100_000),
        store=Store(db),
        risk_monitor=RiskMonitor(),
    )
    engine1._session_id = "session-1"
    engine1._risk.activate_kill_switch("armed", actor="operator:alice")
    engine1._persist_engine_state()  # engine_state now records kill=true
    # The operator deactivates from the dashboard while the engine is stopped.
    engine1._risk.deactivate_kill_switch("resolved", actor="operator:bob")

    engine2 = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=100_000),
        store=Store(db),
        risk_monitor=RiskMonitor(),
    )
    engine2.start()
    try:
        assert engine2._risk.kill_switch_active is False, (
            "a durable deactivation must not be undone by a stale engine row"
        )
    finally:
        engine2.stop()


def test_pnl_rows_use_the_exchange_timestamp_basis(tmp_path, monkeypatch):
    """S17: P&L rows are written in Asia/Shanghai naive time, not host time."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)
    fixed = datetime(2026, 10, 20, 10, 0, tzinfo=TZ)
    monkeypatch.setattr(engine, "_exchange_now", lambda: fixed)

    engine.run_once()

    rows = engine._store.get_pnl_history(
        start_date="2026-10-20", end_date="2026-10-20",
    )
    assert rows
    assert rows[-1]["timestamp"] == "2026-10-20T10:00:00"


def test_sigterm_handler_mid_cycle_completes_the_session(tmp_path, monkeypatch):
    """S16 end-to-end: the CLI SIGTERM handler stops the engine mid-cycle and
    the final session/engine-state writes still happen."""
    import signal

    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)
    in_cycle = threading.Event()
    release = threading.Event()

    def blocking_fetch():
        in_cycle.set()
        release.wait(timeout=10)

    previous = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    cleanup = None
    engine.start()
    try:
        _wait_until(
            lambda: engine._cycle_count >= 1 and not engine._cycle_lock.locked()
        )
        monkeypatch.setattr(engine, "_fetch_prices", blocking_fetch)
        threading.Thread(target=engine.run_once, daemon=True).start()
        assert in_cycle.wait(timeout=5)
        threading.Timer(0.2, release.set).start()

        cleanup = main_module._install_shutdown_handlers(engine.stop)
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
    finally:
        release.set()
        if cleanup is not None:
            cleanup()
        for sig, prev in previous.items():
            signal.signal(sig, prev)
        if engine._thread is not None and engine._thread.is_alive():
            engine.stop()

    assert engine._thread is None or not engine._thread.is_alive()
    assert engine._store.get_engine_state("live_engine:default") is not None
    session = engine._store.get_sessions()[0]
    assert session["status"] == "stopped"
    assert session["stopped_at"]


def test_a_failed_session_write_does_not_leave_a_zombie_engine(
    tmp_path, monkeypatch,
):
    """C5: `_running=True` used to survive a session-write failure, blocking
    every retry with no loop behind it."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000), monkeypatch)
    original_save_session = engine._store.save_session
    calls = {"n": 0}

    def flaky_save_session(session):
        if session.get("status") == "active" and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("database is locked")
        return original_save_session(session)

    monkeypatch.setattr(engine._store, "save_session", flaky_save_session)

    with pytest.raises(OSError):
        engine.start()
    assert engine._running is False
    assert engine._thread is None

    # The guard must not block a retry once the store recovers.
    engine.start()
    try:
        assert engine._running is True
    finally:
        engine.stop()


def test_a_newer_engine_snapshot_restores_a_failed_risk_write(
    tmp_path, monkeypatch,
):
    """Fail-safe: if arming could not be persisted to risk_state but a later
    engine cycle recorded the armed flag, the restart honours the newer row."""
    db = str(tmp_path / "newer-engine.db")
    store = Store(db)
    store.save_engine_state(
        {
            "version": 1, "tenant_id": "default", "risk_level": "green",
            "kill_switch_active": False, "breaches": [],
            "portfolio_value": 0.0, "peak_value": 0.0, "daily_pnl": 0.0,
            "updated_at": "2026-10-20T09:00:00",
        },
        "risk_state:default",
    )
    store.save_engine_state(
        {
            "tenant_id": "default", "kill_switch_active": True,
            "portfolio_value": 100_000.0, "cash": 100_000.0,
            "positions": [], "updated_at": "2026-10-20T10:00:00",
        },
        "live_engine:default",
    )

    engine = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=100_000),
        store=Store(db),
        risk_monitor=RiskMonitor(),
    )
    engine.start()
    try:
        assert engine._risk.kill_switch_active is True, (
            "the newer armed engine snapshot must not be ignored"
        )
    finally:
        engine.stop()


def test_cli_shutdown_handler_runs_cleanup_once_and_exits(tmp_path):
    """The main.py trade handler unwinds via SystemExit and cleans up once."""
    import signal

    calls = []
    previous = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        cleanup = main_module._install_shutdown_handlers(lambda: calls.append(1))
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as excinfo:
            handler(signal.SIGTERM, None)
        cleanup()
        cleanup()
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)

    assert excinfo.value.code == 128 + int(signal.SIGTERM)
    assert calls == [1]


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
