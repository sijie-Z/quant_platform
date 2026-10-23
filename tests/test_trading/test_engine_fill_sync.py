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

import pytest

from quant_platform.core.events import EventBus
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.risk.realtime_engine import RealTimeRiskEngine
from quant_platform.trading.broker import (
    BrokerOrder,
    OrderSide,
    OrderStatus,
    SimulatedBroker,
)
from quant_platform.trading.engine import LiveTradingEngine


class FakeBroker:
    """Minimal broker exposing only the order book the sync reads."""

    def __init__(self, orders=None, *, orders_result: str = "list"):
        self.orders = list(orders or [])
        self.orders_result = orders_result
        self.fail = False

    def get_orders(self, status=None):
        if self.fail:
            raise ConnectionError("broker link down")
        if self.orders_result == "none":
            return None
        return list(self.orders)


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
    )
    engine._persist_fill_result(filled)
    assert engine._store.get_trade_quantity_for_order("o-async") == 100

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


def test_startup_catch_up_books_fills_that_happened_while_down(tmp_path):
    """A fresh process adopts a broker-filled order the old one left SUBMITTED."""
    store = Store(str(tmp_path / "engine.db"))
    store.save_order({
        "order_id": "o-down", "code": "000001", "side": "buy",
        "quantity": 200, "price": 9.0, "status": "submitted",
    })
    recovered = _order(
        order_id="o-down", code="000001", quantity=200, price=9.0,
        filled_quantity=200, filled_price=9.1,
        status=OrderStatus.FILLED, broker_order_id="qmt-42",
    )
    engine = LiveTradingEngine(
        broker=FakeBroker([recovered]), store=store,
        bus=EventBus(), risk_monitor=RiskMonitor(),
    )

    assert engine._sync_broker_fills() == 1
    trades = store.get_trades()
    assert len(trades) == 1
    assert trades[0]["trade_id"] == "o-down:sync:200"
    assert trades[0]["quantity"] == 200
    row = store.get_orders()[0]
    assert row["status"] == "filled"
    assert row["filled_quantity"] == 200
    assert row["broker_order_id"] == "qmt-42"


def test_start_syncs_fills_before_stamping_the_ledger_baseline(
    tmp_path, monkeypatch,
):
    """Fills absorbed from the broker at start are part of the baseline, so
    the EOD ledger must not replay them (R12 boundary)."""
    engine = _engine(tmp_path, SimulatedBroker(initial_cash=100_000))
    calls: list[str] = []
    monkeypatch.setattr(
        engine, "_sync_broker_fills", lambda: calls.append("sync") or 0,
    )
    monkeypatch.setattr(
        engine, "_write_ledger_baseline", lambda: calls.append("baseline"),
    )

    engine.start()
    try:
        assert calls[:2] == ["sync", "baseline"]
    finally:
        engine.stop()


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
        filled_quantity=100, filled_price=10.0, status=OrderStatus.PARTIAL,
    )
    engine = _engine(tmp_path, FakeBroker([order]), risk=risk)

    assert engine._sync_broker_fills() == 1
    assert risk._positions["600519"]["quantity"] == 100

    order.filled_quantity = 250
    order.filled_price = 10.4
    order.status = OrderStatus.FILLED
    assert engine._sync_broker_fills() == 1
    assert risk._positions["600519"]["quantity"] == 250
