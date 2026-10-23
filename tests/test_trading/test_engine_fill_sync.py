"""R13: asynchronous broker fills are booked into the trade ledger.

``place_order`` only persists fills it observes synchronously (FILLED /
PARTIAL). QMT/XTP report most executions after returning SUBMITTED, so
without the fill sync those trades never reach the ``trades`` table and the
R12 EOD trade ledger fails against a stale baseline.

Accounting contract under test:
* delta = broker cumulative ``filled_quantity`` - persisted trade quantity;
* a replayed broker snapshot is a no-op (deterministic trade ids);
* the synchronous ``{order_id}:fill`` row already counts toward persisted;
* fees are recorded as 0 (an order-level total cannot be split per print);
* ``order.filled`` is never republished (cumulative payload would double
  count downstream) -- ``order.synced`` carries the delta;
* a broken/NULL broker book never takes down the trading cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_platform.core.events import EventBus
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.risk.healthcheck import SystemBlockError
from quant_platform.risk.realtime_engine import RealTimeRiskEngine
from quant_platform.trading.broker import (
    BrokerOrder,
    OrderSide,
    OrderStatus,
    Position,
    SimulatedBroker,
)
from quant_platform.trading.engine import LiveTradingEngine

TZ = ZoneInfo("Asia/Shanghai")


class FakeBroker:
    """Minimal broker exposing the book/account the sync and startup read."""

    def __init__(
        self, orders=None, *, orders_result: str = "list",
        positions=None, cash: float = 100_000.0,
    ):
        self.orders = list(orders or [])
        self.orders_result = orders_result
        self.positions = list(positions or [])
        self.cash = cash
        self.fail = False

    def connect(self):
        return True

    def get_orders(self, status=None):
        if self.fail:
            raise ConnectionError("broker link down")
        if self.orders_result == "none":
            return None
        return list(self.orders)

    def get_positions(self):
        return list(self.positions)

    def get_account(self):
        market_value = sum(float(p.market_value or 0) for p in self.positions)
        return {
            "cash": self.cash,
            "market_value": market_value,
            "total_equity": self.cash + market_value,
            "n_positions": len(self.positions),
            "connected": True,
        }

    def place_order(self, order):  # pragma: no cover - no orders in these tests
        raise AssertionError("no order expected in this test")


def _engine(tmp_path, broker, *, risk=None, bus=None) -> LiveTradingEngine:
    return LiveTradingEngine(
        broker=broker,
        store=Store(str(tmp_path / "engine.db")),
        bus=bus if bus is not None else EventBus(),
        risk_monitor=risk or RiskMonitor(),
    )


def _order(**overrides) -> BrokerOrder:
    fields = {
        "order_id": "o-async", "code": "600519", "side": OrderSide.BUY,
        "quantity": 100, "price": 10.0, "status": OrderStatus.SUBMITTED,
    }
    fields.update(overrides)
    return BrokerOrder(**fields)


def _simulated_engine(tmp_path) -> LiveTradingEngine:
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    engine = _engine(tmp_path, broker)
    engine._fetch_prices = lambda: None
    engine._trading_day = engine._exchange_trading_date()
    return engine


def test_zero_fill_books_nothing_then_the_fill_is_booked_once(tmp_path):
    order = _order()
    engine = _engine(tmp_path, FakeBroker([order]))

    assert engine._sync_broker_fills() == 0, "nothing filled yet"
    assert engine._store.get_trades() == []
    assert engine._store.get_orders() == []

    order.filled_quantity = 100
    order.filled_price = 10.5
    order.status = OrderStatus.FILLED
    order.commission = 5.0
    order.tax = 1.0

    assert engine._sync_broker_fills() == 1
    trades = engine._store.get_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "o-async:sync:100"
    assert trades[0]["quantity"] == 100
    assert trades[0]["price"] == pytest.approx(10.5)
    # Fees on the order total cannot be split per incremental print.
    assert trades[0]["commission"] == 0
    assert trades[0]["tax"] == 0
    order_row = engine._store.get_orders()[0]
    assert order_row["status"] == "filled"
    assert order_row["filled_quantity"] == 100
    assert order_row["filled_price"] == pytest.approx(10.5)

    assert engine._sync_broker_fills() == 0, "replay must be idempotent"
    assert len(engine._store.get_trades()) == 1
    # Compliance: a synced fill is a real execution and must be audited.
    assert engine._store.get_events(topic="audit.order_filled"), (
        "synced fills must reach the audit trail"
    )


def test_partial_batches_book_incremental_deltas(tmp_path):
    order = _order(
        filled_quantity=40, filled_price=10.0, status=OrderStatus.PARTIAL,
    )
    engine = _engine(tmp_path, FakeBroker([order]))

    assert engine._sync_broker_fills() == 1
    order.filled_quantity = 100
    order.filled_price = 10.2
    order.status = OrderStatus.FILLED
    assert engine._sync_broker_fills() == 1

    trades = {t["trade_id"]: t for t in engine._store.get_trades(limit=10)}
    assert trades["o-async:sync:40"]["quantity"] == 40
    assert trades["o-async:sync:100"]["quantity"] == 60
    assert engine._store.get_trade_quantity_for_order("o-async") == 100

    assert engine._sync_broker_fills() == 0
    assert engine._store.get_trade_quantity_for_order("o-async") == 100


def test_synchronous_fill_row_counts_toward_the_persisted_quantity(tmp_path):
    broker = FakeBroker()
    engine = _engine(tmp_path, broker)
    filled = _order(
        filled_quantity=100, filled_price=10.0, status=OrderStatus.FILLED,
        broker_order_id="7777",
    )
    engine._persist_fill_result(filled)
    assert engine._store.get_trade_quantity_for_order("o-async") == 100
    assert engine._store.get_orders()[0]["broker_order_id"] == "7777", (
        "the sync's cross-restart dedupe depends on the wire id being stored"
    )

    broker.orders = [filled]
    assert engine._sync_broker_fills() == 0, "delta is already zero"
    trades = engine._store.get_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "o-async:fill"


def test_unknown_order_book_is_a_noop(tmp_path):
    engine = _engine(tmp_path, FakeBroker(orders_result="none"))
    assert engine._sync_broker_fills() == 0
    assert engine._store.get_trades() == []


def test_get_orders_failure_warns_and_the_cycle_still_completes(
    tmp_path, monkeypatch, caplog,
):
    engine = _simulated_engine(tmp_path)
    monkeypatch.setattr(engine, "_generate_signals", lambda positions=None: [])

    def boom(status=None):
        raise ConnectionError("link down")

    monkeypatch.setattr(engine._broker, "get_orders", boom)

    with caplog.at_level(logging.WARNING):
        cycle = engine._execute_cycle()

    assert "Broker fill sync skipped" in caplog.text
    assert cycle.portfolio_value == pytest.approx(1_000_000.0)
    assert engine._store.get_trades() == []


def _freeze_pre_market(engine, monkeypatch) -> None:
    """No network fetch and no cycles: 09:00 is outside the session."""
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    engine._scheduler._now = lambda: datetime(2026, 10, 20, 9, 0, tzinfo=TZ)


def test_start_books_downtime_fills_explained_by_the_broker_book(
    tmp_path, monkeypatch,
):
    """B1: a fill while the process was down moves the broker book; startup
    adopts it because the broker's unbooked fills explain the difference."""
    store = Store(str(tmp_path / "engine.db"))
    store.save_position({
        "code": "600519", "quantity": 100, "available": 100,
        "avg_cost": 10.0, "current_price": 10.0, "market_value": 1000.0,
    })
    recovered = _order(
        order_id="fresh-after-restart", code="600519", quantity=200,
        price=10.0, filled_quantity=100, filled_price=10.0,
        status=OrderStatus.PARTIAL, broker_order_id="qmt-42",
    )
    broker = FakeBroker(
        [recovered],
        positions=[Position(
            code="600519", quantity=200, available=200, avg_cost=10.0,
            current_price=10.0, market_value=2000.0,
        )],
        cash=98_000.0,
    )
    engine = LiveTradingEngine(
        broker=broker, store=store, bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    _freeze_pre_market(engine, monkeypatch)

    engine.start()
    try:
        assert engine._running
        trades = store.get_trades()
        assert [t["trade_id"] for t in trades] == [
            "fresh-after-restart:sync:100",
        ]
        assert store.get_positions()[0]["quantity"] == 200
        baseline = store.get_latest_ledger_baseline()
        assert baseline is not None, "a confirmed startup keeps the baseline"
        assert baseline["positions"]["600519"]["quantity"] == 200
    finally:
        engine.stop()


def test_start_adopts_a_flat_book_explained_by_downtime_sells(
    tmp_path, monkeypatch,
):
    """B1 (flat branch): a sell-to-flat while down explains an empty broker
    book; a live adapter has no ``restore_positions`` to fall back on."""
    store = Store(str(tmp_path / "engine.db"))
    store.save_position({
        "code": "600519", "quantity": 100, "available": 0,
        "avg_cost": 10.0, "current_price": 10.0, "market_value": 1000.0,
    })
    recovered = _order(
        order_id="fresh-sell", code="600519", side=OrderSide.SELL,
        quantity=100, price=10.5, filled_quantity=100, filled_price=10.5,
        status=OrderStatus.FILLED, broker_order_id="qmt-43",
    )
    broker = FakeBroker([recovered], positions=[], cash=101_000.0)
    engine = LiveTradingEngine(
        broker=broker, store=store, bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    _freeze_pre_market(engine, monkeypatch)

    engine.start()
    try:
        assert store.get_positions() == []
        trades = store.get_trades()
        assert [t["trade_id"] for t in trades] == ["fresh-sell:sync:100"]
        assert trades[0]["side"] == "sell"
    finally:
        engine.stop()


def test_start_still_refuses_unexplained_position_differences(
    tmp_path, monkeypatch,
):
    store = Store(str(tmp_path / "engine.db"))
    store.save_position({
        "code": "600519", "quantity": 100, "available": 100,
        "avg_cost": 10.0, "current_price": 10.0, "market_value": 1000.0,
    })
    broker = FakeBroker(
        [],  # readable order book, but no fill explains the difference
        positions=[Position(
            code="600519", quantity=150, available=150, avg_cost=10.0,
            current_price=10.0, market_value=1500.0,
        )],
    )
    engine = LiveTradingEngine(
        broker=broker, store=store, bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    _freeze_pre_market(engine, monkeypatch)

    with pytest.raises(SystemBlockError, match="reconciliation"):
        engine.start()
    assert engine._thread is None


def test_cross_restart_recovered_order_is_not_rebooked(tmp_path):
    """M1: QMT rebuilds a previous process's order with a fresh local id;
    the persisted ``broker_order_id`` keeps the delta at zero."""
    store = Store(str(tmp_path / "engine.db"))
    store.save_order_with_trade(
        {
            "order_id": "old-local", "code": "600519", "side": "buy",
            "quantity": 100, "price": 10.0, "status": "filled",
            "filled_quantity": 100, "filled_price": 10.5,
            "broker_order_id": "9001",
        },
        {
            "trade_id": "old-local:fill", "order_id": "old-local",
            "code": "600519", "side": "buy", "quantity": 100, "price": 10.5,
        },
    )
    recovered = _order(
        order_id="new-local", filled_quantity=100, filled_price=10.5,
        status=OrderStatus.FILLED, broker_order_id="9001",
    )
    engine = LiveTradingEngine(
        broker=FakeBroker([recovered]), store=store,
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )

    assert engine._sync_broker_fills() == 0
    assert len(store.get_trades()) == 1
    assert store.get_trade_quantity_for_order("new-local") == 0


def test_broker_id_dedupe_does_not_swallow_a_new_days_fill(tmp_path):
    """N1: broker wire ids are reused across days; a same-id row from an
    earlier day (or another asset/side) must not explain today's fill."""
    store = Store(str(tmp_path / "engine.db"))
    store.save_order_with_trade(
        {
            "order_id": "yesterday", "code": "600519", "side": "buy",
            "quantity": 100, "price": 10.0, "status": "filled",
            "filled_quantity": 100, "filled_price": 10.5,
            "broker_order_id": "9001",
            "created_at": "2000-01-01T09:30:00",
        },
        {
            "trade_id": "yesterday:fill", "order_id": "yesterday",
            "code": "600519", "side": "buy", "quantity": 100, "price": 10.5,
        },
    )
    store.save_order_with_trade(
        {
            "order_id": "other-code", "code": "000001", "side": "buy",
            "quantity": 100, "price": 10.0, "status": "filled",
            "filled_quantity": 100, "filled_price": 10.5,
            "broker_order_id": "9001",
        },
        {
            "trade_id": "other-code:fill", "order_id": "other-code",
            "code": "000001", "side": "buy", "quantity": 100, "price": 10.5,
        },
    )
    recovered = _order(
        order_id="today", code="600519", filled_quantity=100,
        filled_price=10.5, status=OrderStatus.FILLED,
        broker_order_id="9001",
    )
    engine = LiveTradingEngine(
        broker=FakeBroker([recovered]), store=store,
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )

    assert engine._sync_broker_fills() == 1, (
        "an old-day or other-asset row with the same wire id must not "
        "swallow today's fill"
    )
    assert store.get_trade_quantity_for_order("today") == 100


def test_start_withholds_the_baseline_when_the_order_book_is_unreadable(
    tmp_path, monkeypatch,
):
    """M2: a baseline whose book may contain unbooked fills would make EOD
    count them twice; withholding it yields unknown (loud) instead."""
    engine = LiveTradingEngine(
        broker=FakeBroker(orders_result="none"),
        store=Store(str(tmp_path / "engine.db")),
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    _freeze_pre_market(engine, monkeypatch)

    engine.start()
    try:
        assert engine._running
        assert engine._store.get_latest_ledger_baseline() is None
    finally:
        engine.stop()


class _UnreadableResyncBroker(FakeBroker):
    """Mimics QMT after a failed order query: ``get_orders()`` serves its
    in-memory cache (empty on a fresh process), so only the resync knows."""

    def _resync_from_broker(self):
        return False


def test_start_withholds_the_baseline_when_the_resync_query_fails(
    tmp_path, monkeypatch,
):
    engine = LiveTradingEngine(
        broker=_UnreadableResyncBroker(),
        store=Store(str(tmp_path / "engine.db")),
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    _freeze_pre_market(engine, monkeypatch)

    engine.start()
    try:
        assert engine._running
        assert engine._store.get_latest_ledger_baseline() is None, (
            "an empty cached book from a failed resync must not count as "
            "confirmed"
        )
    finally:
        engine.stop()


class _ExplodingBookBroker(FakeBroker):
    def get_orders(self, status=None):
        def gen():
            raise RuntimeError("iterator blew up")
            yield  # pragma: no cover

        return gen()


def test_get_orders_iteration_failure_is_contained(tmp_path, caplog):
    """N6: non-TypeError iterator errors must not escape the sync."""
    engine = _engine(tmp_path, _ExplodingBookBroker())
    with caplog.at_level(logging.WARNING):
        assert engine._sync_broker_fills() == 0
    assert "get_orders returned" in caplog.text


def test_start_syncs_fills_before_stamping_the_ledger_baseline(
    tmp_path, monkeypatch,
):
    """Fills absorbed from the broker at start are part of the baseline, so
    the EOD ledger must not replay them (R12 boundary)."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000))
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    calls: list[str] = []
    stamps: dict = {}

    def sync(executed_at=None):
        # The real sync confirms the (readable) order book; the fake must
        # too, otherwise start() withholds the baseline by design. The loop
        # may sync again after start(); only the first call is inspected.
        engine._fill_sync_confirmed = True
        engine._fill_sync_blocked = False
        if not calls:
            calls.append("sync")
            stamps["sync"] = executed_at
        return 0

    def baseline(not_before=None):
        calls.append("baseline")
        stamps["baseline"] = not_before

    monkeypatch.setattr(engine, "_sync_broker_fills", sync)
    monkeypatch.setattr(engine, "_write_ledger_baseline", baseline)

    engine.start()
    try:
        assert calls[:2] == ["sync", "baseline"]
        # The baseline stamp is forced strictly after the sync's stamp.
        assert stamps["sync"] is not None
        assert stamps["baseline"] is not None
        assert stamps["baseline"].isoformat() == stamps["sync"]
    finally:
        engine.stop()


def test_startup_synced_fills_are_not_replayed_by_the_eod_ledger(tmp_path):
    """The R12 ledger applies only trades at/after the baseline stamp.

    Both guards are load-bearing and this test is sensitive to each:
    * the broker's ``updated_at`` is in the future (QMT stamps observation
      time, and a skewed clock must not move a pre-baseline fill into the
      ledger) -- only the ``executed_at`` override keeps it out;
    * ``not_before`` is in the future, so the clamp (not the wall clock)
      must place ``created_at`` strictly after the trade.
    """
    from quant_platform.operations.reconciliation import ledger_from_trades

    store = Store(str(tmp_path / "engine.db"))
    recovered = _order(
        filled_quantity=100, filled_price=10.0, status=OrderStatus.FILLED,
        updated_at="2999-01-02T00:00:00",
    )
    engine = LiveTradingEngine(
        broker=FakeBroker([recovered]), store=store,
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )

    startup_sync_at = datetime(2999, 1, 1, 0, 0, 0)
    assert engine._sync_broker_fills(
        executed_at=startup_sync_at.isoformat(),
    ) == 1
    engine._write_ledger_baseline(not_before=startup_sync_at)

    baseline = store.get_latest_ledger_baseline()
    trades = store.get_trades_for_export(tenant_id="default")
    assert len(trades) == 1
    assert trades[0]["executed_at"] == startup_sync_at.isoformat(), (
        "the startup override must win over the broker's observation stamp"
    )
    assert baseline["created_at"] > startup_sync_at.isoformat(), (
        "the not_before clamp must force the baseline strictly later"
    )
    assert ledger_from_trades(baseline, trades)["positions"] == {}, (
        "a startup-synced fill must not be replayed on top of the baseline"
    )


def test_each_cycle_syncs_before_saving_the_pnl_snapshot(
    tmp_path, monkeypatch,
):
    engine = _simulated_engine(tmp_path)
    monkeypatch.setattr(engine, "_generate_signals", lambda positions=None: [])
    calls: list[str] = []
    monkeypatch.setattr(
        engine, "_sync_broker_fills", lambda: calls.append("sync") or 0,
    )
    real_save = engine._store.save_pnl_snapshot

    def save(snapshot):
        calls.append("pnl")
        return real_save(snapshot)

    monkeypatch.setattr(engine._store, "save_pnl_snapshot", save)

    engine._execute_cycle()

    assert calls == ["sync", "pnl"]


def test_kill_switch_cycles_still_sync_fills(tmp_path, monkeypatch):
    """A sustained kill switch returns before the normal end-of-cycle sync."""
    engine = _simulated_engine(tmp_path)
    engine._risk.activate_kill_switch("test")
    calls: list[str] = []
    monkeypatch.setattr(
        engine, "_sync_broker_fills",
        lambda *args, **kwargs: calls.append("sync") or 0,
    )

    engine._execute_cycle()

    assert calls == ["sync"]


def test_stale_quote_cycles_still_sync_fills(tmp_path, monkeypatch):
    """M4: a dead feed returns before Step 4; fills must still be booked."""
    engine = _simulated_engine(tmp_path)
    engine._last_prices_update = "2026-10-20T09:30:00"
    monkeypatch.setattr(engine, "_aggregate_quote_age", lambda: 9999.0)
    calls: list[str] = []
    monkeypatch.setattr(
        engine, "_sync_broker_fills",
        lambda *args, **kwargs: calls.append("sync") or 0,
    )

    engine._execute_cycle()

    assert calls == ["sync"]


def test_unknown_position_book_cycles_still_sync_fills(tmp_path, monkeypatch):
    """M4: the order book is a separate endpoint from the failed position query."""
    engine = _simulated_engine(tmp_path)
    monkeypatch.setattr(
        engine, "_query_positions", lambda trust_flat=False: (None, False),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        engine, "_sync_broker_fills",
        lambda *args, **kwargs: calls.append("sync") or 0,
    )

    engine._execute_cycle()

    assert calls == ["sync"]


def test_stop_syncs_fills_after_the_final_cycle(tmp_path, monkeypatch):
    """M4: the live loop stops cycling after the close; stop() closes the ledger."""
    engine = _simulated_engine(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        engine, "_sync_broker_fills",
        lambda *args, **kwargs: calls.append("sync") or 0,
    )

    engine.stop()

    assert calls == ["sync"]


class _SyncDuringPlaceBroker(FakeBroker):
    """A broker whose place_order lets a cycle sync interleave (M5)."""

    def __init__(self):
        super().__init__()
        self.engine: LiveTradingEngine | None = None

    def place_order(self, order):
        result = BrokerOrder(
            order_id=order.order_id, code=order.code, side=order.side,
            quantity=order.quantity, price=order.price,
            filled_quantity=order.quantity, filled_price=order.price,
            status=OrderStatus.FILLED,
        )
        self.orders = [result]
        # Another thread's cycle sync runs before the engine persists the
        # synchronous fill.
        self.engine._sync_broker_fills()
        return result


def test_manual_order_fill_in_flight_is_not_double_booked(tmp_path):
    """M5: manual_order persists outside the cycle lock; the in-flight marker
    must keep a concurrent sync from booking the same fill as a delta."""
    broker = _SyncDuringPlaceBroker()
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(tmp_path / "engine.db")),
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    broker.engine = engine

    result = engine.manual_order("600519", "buy", 100, 10.0)

    assert result["status"] == "filled"
    trades = engine._store.get_trades()
    assert [t["trade_id"] for t in trades] == [f"{result['order_id']}:fill"]
    assert engine._store.get_trade_quantity_for_order(result["order_id"]) == 100
    assert engine._fills_in_flight == 0, "the in-flight marker must be cleared"


class _RecoveredDuringPlaceBroker(FakeBroker):
    """M5/N3: place_order swaps in a recovered order with a different local id
    *inside* the call; an id-based skip cannot see that window."""

    def __init__(self):
        super().__init__()
        self.engine: LiveTradingEngine | None = None

    def place_order(self, order):
        result = BrokerOrder(
            order_id="recovered-id", code=order.code, side=order.side,
            quantity=order.quantity, price=order.price,
            filled_quantity=order.quantity, filled_price=order.price,
            status=OrderStatus.FILLED, broker_order_id="9002",
        )
        self.orders = [result]
        self.engine._sync_broker_fills()
        return result


def test_manual_order_recovered_id_window_is_covered(tmp_path):
    broker = _RecoveredDuringPlaceBroker()
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(tmp_path / "engine.db")),
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )
    broker.engine = engine

    result = engine.manual_order("600519", "buy", 100, 10.0)

    assert result["order_id"] == "recovered-id"
    trades = engine._store.get_trades()
    assert [t["trade_id"] for t in trades] == ["recovered-id:fill"]
    assert engine._store.get_trade_quantity_for_order("recovered-id") == 100
    assert engine._fills_in_flight == 0


def test_synced_fill_emits_order_synced_never_order_filled(tmp_path):
    bus = EventBus()
    order = _order(
        filled_quantity=100, filled_price=10.5, status=OrderStatus.FILLED,
    )
    engine = _engine(tmp_path, FakeBroker([order]), bus=bus)

    assert engine._sync_broker_fills() == 1

    synced = bus.get_history(topic="order.synced")
    assert len(synced) == 1
    assert synced[0]["data"] == {
        "order_id": "o-async", "code": "600519", "side": "buy",
        "delta": 100, "price": 10.5,
    }
    assert bus.get_history(topic="order.filled") == [], (
        "republishing the cumulative fill would double count downstream"
    )


def test_real_time_risk_engine_receives_the_delta_not_the_cumulative(
    tmp_path,
):
    risk = RealTimeRiskEngine()
    order = _order(
        quantity=300, filled_quantity=100, filled_price=10.0,
        status=OrderStatus.PARTIAL,
    )
    engine = _engine(tmp_path, FakeBroker([order]), risk=risk)

    assert engine._sync_broker_fills() == 1
    assert risk._positions["600519"]["quantity"] == 100

    order.filled_quantity = 250
    order.filled_price = 10.4
    order.status = OrderStatus.FILLED
    assert engine._sync_broker_fills() == 1
    assert risk._positions["600519"]["quantity"] == 250
