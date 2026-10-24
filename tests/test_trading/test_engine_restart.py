"""D4/D5/D7/D12: restart restore, reconciliation and exchange calendar.

D4: a fresh engine + fresh SimulatedBroker on the same Store was refused by
the health check's position-sync test (broker=0 local=2000) and there was no
restore path. D5: kill switch / peak / day baseline / cash were in memory
only, so a restart silently resumed trading and reset the account. D7: the
new engine's empty `_trading_day` called `new_trading_day()` on the first
cycle, releasing today's T+1 buys. D12: the rollover used the host-local
calendar date, not the exchange's.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime as real_datetime
from zoneinfo import ZoneInfo

import pytest

import quant_platform.trading.engine as engine_mod
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.risk.healthcheck import SystemBlockError
from quant_platform.trading.broker import Order, OrderSide, OrderStatus, SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine, TradeSignal

# Fixed exchange session for the T+1 rollover tests: Monday 2026-10-26,
# 15:00 Asia/Shanghai. Pinning the exchange clock removes the run-date
# dependency: `_execute_cycle_body` only advances the trading day when the
# exchange calendar says today is a session, so running the suite on a
# weekend/holiday used to leave T+1 locked. 2026-10-23 is the previous
# exchange session (Friday).
_FIXED_SESSION = real_datetime(2026, 10, 26, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
_FIXED_SESSION_DAY = "2026-10-26"
_PREVIOUS_SESSION_DAY = "2026-10-23"


def _buy(broker: SimulatedBroker, quantity: int = 2000, price: float = 10.0):
    return broker.place_order(
        Order(code="600519", side=OrderSide.BUY, quantity=quantity, price=price)
    )


def _engine(broker, store, monkeypatch) -> LiveTradingEngine:
    engine = LiveTradingEngine(
        broker=broker, store=store, risk_monitor=RiskMonitor(),
    )
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    return engine


def _wait_for_cycle(engine: LiveTradingEngine, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while engine._cycle_count < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert engine._cycle_count >= 1, "engine loop did not run a cycle"


def test_restart_restores_open_positions_into_a_fresh_broker(tmp_path, monkeypatch):
    db = str(tmp_path / "restart.db")
    store1 = Store(db)
    broker1 = SimulatedBroker(initial_cash=1_000_000)
    broker1.connect()
    assert _buy(broker1).status == OrderStatus.FILLED
    store1.save_position({
        **broker1.get_positions()[0].to_dict(), "tenant_id": "default",
    })

    # A crash leaves only the positions table: no engine_state row yet.
    broker2 = SimulatedBroker(initial_cash=1_000_000)
    engine2 = _engine(broker2, Store(db), monkeypatch)
    engine2.start()
    try:
        positions = broker2.get_positions()
        assert [p.code for p in positions] == ["600519"]
        assert positions[0].quantity == 2000
        assert [row["quantity"] for row in Store(db).get_positions("default")] == [2000]
    finally:
        engine2.stop()


def test_restart_refuses_when_broker_and_store_disagree(tmp_path, monkeypatch):
    db = str(tmp_path / "mismatch.db")
    store = Store(db)
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    assert _buy(broker, quantity=100).status == OrderStatus.FILLED
    store.save_position({
        "code": "600519", "quantity": 2000, "available": 2000,
        "avg_cost": 10.0, "current_price": 10.0, "market_value": 20000.0,
    })

    engine = _engine(broker, store, monkeypatch)
    with pytest.raises(SystemBlockError, match="reconciliation"):
        engine.start()

    assert engine._thread is None
    assert engine._running is False
    assert _baseline_rows(store) == [], (
        "a blocked start must not leave a ledger baseline behind"
    )


def test_restart_restores_kill_switch_cash_and_positions(tmp_path, monkeypatch):
    db = str(tmp_path / "kill.db")
    store1 = Store(db)
    broker1 = SimulatedBroker(initial_cash=1_000_000)
    broker1.connect()
    assert _buy(broker1).status == OrderStatus.FILLED

    engine1 = LiveTradingEngine(
        broker=broker1, store=store1, risk_monitor=RiskMonitor(),
    )
    engine1._session_id = "session-1"
    # Same exchange day: the restored T+1 lock must keep today's shares on
    # the buy side (otherwise the restart cycle de-risks them immediately).
    engine1._trading_day = engine1._exchange_trading_date()
    engine1._persist_engine_state()
    engine1._risk.activate_kill_switch("test kill")
    engine1._persist_engine_state()
    cash_after_buy = broker1.get_account()["cash"]
    assert cash_after_buy < 1_000_000

    broker2 = SimulatedBroker(initial_cash=1_000_000)
    engine2 = _engine(broker2, Store(db), monkeypatch)
    engine2.start()
    try:
        _wait_for_cycle(engine2)
        assert engine2._risk.kill_switch_active is True, (
            "a persisted kill switch must be restored"
        )
        assert broker2.get_account()["cash"] == pytest.approx(cash_after_buy, abs=0.01)
        positions = broker2.get_positions()
        assert positions and positions[0].quantity == 2000

        blocked = engine2.manual_order("600519", "buy", 100, 10.0)
        assert blocked["status"] == "rejected"
        assert "Kill switch" in blocked["error_msg"]

        engine2._risk.deactivate_kill_switch()
        accepted = engine2.manual_order("600519", "buy", 100, 10.0)
        assert accepted["status"] == "filled"
    finally:
        engine2.stop()


def test_same_day_restart_does_not_unlock_todays_buys(tmp_path, monkeypatch):
    db = str(tmp_path / "t1.db")
    store1 = Store(db)
    broker1 = SimulatedBroker(initial_cash=1_000_000)
    broker1.connect()
    assert _buy(broker1).status == OrderStatus.FILLED
    assert broker1.get_positions()[0].available == 0  # T+1 lock

    engine1 = LiveTradingEngine(
        broker=broker1, store=store1, risk_monitor=RiskMonitor(),
    )
    monkeypatch.setattr(engine1, "_exchange_now", lambda: _FIXED_SESSION)
    engine1._trading_day = _FIXED_SESSION_DAY
    engine1._persist_engine_state()

    broker2 = SimulatedBroker(initial_cash=1_000_000)
    engine2 = _engine(broker2, Store(db), monkeypatch)
    monkeypatch.setattr(engine2, "_exchange_now", lambda: _FIXED_SESSION)
    engine2.start()
    try:
        _wait_for_cycle(engine2)
        positions = broker2.get_positions()
        assert positions and positions[0].quantity == 2000
        assert positions[0].available == 0, (
            "a new engine on the same exchange day must not release T+1"
        )
        assert engine2._trading_day == engine1._trading_day

        # Sizing must refuse the sell because available is still 0.
        engine2._current_prices["600519"] = 10.0
        engine2._price_last_seen["600519"] = time.monotonic()
        signal = TradeSignal(
            code="600519", side=OrderSide.SELL, target_weight=0.0,
            current_weight=0.02, signal_strength=0.0, reason="exit",
        )
        assert engine2._signal_to_order(signal, 1_000_000, positions) is None
    finally:
        engine2.stop()


def test_crash_before_first_state_write_keeps_same_day_t1_lock(
    tmp_path, monkeypatch,
):
    """No engine_state row yet (crash on the first cycle): the position row's
    timestamp must still prevent a same-day T+1 release."""
    db = str(tmp_path / "crash_t1.db")
    broker1 = SimulatedBroker(initial_cash=1_000_000)
    broker1.connect()
    assert _buy(broker1).status == OrderStatus.FILLED
    store1 = Store(db)
    store1.save_position({
        **broker1.get_positions()[0].to_dict(), "tenant_id": "default",
    })
    assert store1.get_engine_state("live_engine:default") is None

    broker2 = SimulatedBroker(initial_cash=1_000_000)
    engine2 = _engine(broker2, Store(db), monkeypatch)
    # Align the exchange clock with the store row's host timestamp for this
    # environment; production derives both from the exchange calendar.
    monkeypatch.setattr(engine2, "_exchange_now", lambda: real_datetime.now())
    engine2.start()
    try:
        _wait_for_cycle(engine2)
        positions = broker2.get_positions()
        assert positions and positions[0].available == 0, (
            "a same-day crash before the first engine_state write must not "
            "release today's T+1 lock"
        )
    finally:
        engine2.stop()


def test_next_day_restart_releases_t1(tmp_path, monkeypatch):
    db = str(tmp_path / "nextday.db")
    store1 = Store(db)
    broker1 = SimulatedBroker(initial_cash=1_000_000)
    broker1.connect()
    assert _buy(broker1).status == OrderStatus.FILLED

    engine1 = LiveTradingEngine(
        broker=broker1, store=store1, risk_monitor=RiskMonitor(),
    )
    monkeypatch.setattr(engine1, "_exchange_now", lambda: _FIXED_SESSION)
    engine1._trading_day = _PREVIOUS_SESSION_DAY  # the prior exchange session
    engine1._persist_engine_state()

    broker2 = SimulatedBroker(initial_cash=1_000_000)
    engine2 = _engine(broker2, Store(db), monkeypatch)
    monkeypatch.setattr(engine2, "_exchange_now", lambda: _FIXED_SESSION)
    engine2.start()
    try:
        _wait_for_cycle(engine2)
        positions = broker2.get_positions()
        assert positions and positions[0].available == 2000, (
            "T+1 must be released once the exchange date advances"
        )
    finally:
        engine2.stop()


def test_weekend_date_advance_does_not_release_t1(tmp_path, monkeypatch):
    """A weekend/holiday date change must not release Friday's buys.

    Paper mode trades continuously (it bypasses the market-hours gate), so a
    naive calendar-date rollover would let it sell Friday's shares on
    Saturday.
    """
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    assert _buy(broker).status == OrderStatus.FILLED
    engine = _engine(broker, Store(str(tmp_path / "weekend.db")), monkeypatch)
    saturday = real_datetime(2026, 10, 24, 10, 0,
                             tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(engine, "_exchange_now", lambda: saturday)
    engine._trading_day = "2026-10-23"  # Friday

    engine._execute_cycle()

    assert broker.get_positions()[0].available == 0, (
        "a Saturday date advance must not release T+1"
    )
    assert engine._trading_day == "2026-10-23"


def test_rollover_uses_exchange_clock_not_host_clock(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    assert _buy(broker).status == OrderStatus.FILLED
    engine = _engine(broker, Store(str(tmp_path / "tz.db")), monkeypatch)

    exchange_day = real_datetime(2026, 10, 21, 10, 0,
                                 tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(engine, "_exchange_now", lambda: exchange_day)
    engine._trading_day = "2026-10-21"
    engine._day_start_equity = 123.0

    class HostAheadDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return real_datetime.now(tz)
            # The host (UTC-3) crossed local midnight into the next day while
            # the exchange is still in the same trading session.
            return real_datetime(2026, 10, 22, 0, 1)

    monkeypatch.setattr(engine_mod, "datetime", HostAheadDateTime)

    engine._execute_cycle()

    positions = broker.get_positions()
    assert positions and positions[0].available == 0, (
        "host-local midnight must not unlock same-session T+1 buys"
    )
    assert engine._trading_day == "2026-10-21"
    assert engine._day_start_equity == 123.0, (
        "the daily P&L baseline must not reset on a host-local date change"
    )


# ── R12: ledger baseline written once per successful start ──

def _baseline_rows(store) -> list[dict]:
    conn = sqlite3.connect(store._db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM ledger_baselines ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()


def test_start_writes_one_ledger_baseline(tmp_path, monkeypatch):
    db = str(tmp_path / "baseline.db")
    store = Store(db)
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    assert _buy(broker).status == OrderStatus.FILLED
    engine = _engine(broker, store, monkeypatch)

    engine.start()
    try:
        rows = _baseline_rows(store)
        assert len(rows) == 1
        baseline = store.get_latest_ledger_baseline(
            "default", "live_engine:default",
        )
        assert baseline["cash"] == pytest.approx(
            broker.get_account()["cash"],
        )
        assert baseline["positions"]["600519"]["quantity"] == 2000
        assert baseline["tenant_id"] == "default"
    finally:
        engine.stop()

    assert len(_baseline_rows(store)) == 1


def test_restart_writes_a_new_baseline_row(tmp_path, monkeypatch):
    db = str(tmp_path / "baseline_restart.db")
    store1 = Store(db)
    broker1 = SimulatedBroker(initial_cash=1_000_000)
    broker1.connect()
    assert _buy(broker1).status == OrderStatus.FILLED
    engine1 = _engine(broker1, store1, monkeypatch)
    engine1.start()
    try:
        first = store1.get_latest_ledger_baseline(
            "default", "live_engine:default",
        )
    finally:
        engine1.stop()

    broker2 = SimulatedBroker(initial_cash=1_000_000)
    store2 = Store(db)
    engine2 = _engine(broker2, store2, monkeypatch)
    engine2.start()
    try:
        latest = store2.get_latest_ledger_baseline(
            "default", "live_engine:default",
        )
        rows = _baseline_rows(store2)
    finally:
        engine2.stop()

    assert len(rows) == 2, "a restart appends, it does not replace"
    assert latest["id"] > first["id"]
    assert latest["positions"]["600519"]["quantity"] == 2000


def test_unknown_broker_positions_write_no_baseline(
    tmp_path, monkeypatch, caplog,
):
    store = Store(str(tmp_path / "baseline_unknown.db"))
    broker = SimulatedBroker(initial_cash=100_000)
    broker.connect()
    monkeypatch.setattr(broker, "get_positions", lambda: None)
    engine = _engine(broker, store, monkeypatch)

    with caplog.at_level(logging.WARNING, logger="quant_platform"):
        engine._write_ledger_baseline()

    assert store.get_latest_ledger_baseline() is None
    assert any(
        "Ledger baseline not written" in record.getMessage()
        for record in caplog.records
    )


def test_missing_broker_cash_fails_loud_and_writes_nothing(
    tmp_path, monkeypatch,
):
    store = Store(str(tmp_path / "baseline_nocash.db"))
    broker = SimulatedBroker(initial_cash=100_000)
    broker.connect()
    monkeypatch.setattr(broker, "get_account", lambda: {"total_equity": 1.0})
    engine = _engine(broker, store, monkeypatch)

    with pytest.raises(RuntimeError, match="cash"):
        engine._write_ledger_baseline()

    assert store.get_latest_ledger_baseline() is None


def test_a_disconnected_broker_is_not_baselined(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "baseline_disconnected.db"))
    broker = SimulatedBroker(initial_cash=100_000)
    broker.connect()
    monkeypatch.setattr(
        broker, "get_account",
        lambda: {"connected": False, "cash": 1_000_000.0},
    )
    engine = _engine(broker, store, monkeypatch)

    with pytest.raises(RuntimeError, match="disconnected"):
        engine._write_ledger_baseline()

    assert store.get_latest_ledger_baseline() is None
