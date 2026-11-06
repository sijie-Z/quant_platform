"""D2/D3 regression: partial fills and audit-failure durability.

D2: `_execute_cycle_body` only handled FILLED/REJECTED; a PARTIAL result fell
through untouched, so a real partial execution produced no order row, no
trade row, no audit event and no trade count. De-risking rejections were
audited as ORDER_SUBMITTED, hiding a failed exit.

D3: the old path called `audit.log_order()` (which writes the order+trade)
and any exception aborted the cycle before P&L/position persistence. A
transient audit failure permanently lost a broker fill from the ledger.
"""

from __future__ import annotations

import time

import pytest

from quant_platform.core.audit import AuditAction
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import (
    BrokerOrder,
    Order,
    OrderSide,
    OrderStatus,
    SimulatedBroker,
)
from quant_platform.trading.engine import LiveTradingEngine, TradeSignal


def _make_engine(tmp_path, monkeypatch, broker=None) -> LiveTradingEngine:
    broker = broker or SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    engine = LiveTradingEngine(
        broker=broker,
        store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
    )
    # No network in tests; quotes are injected directly.
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    engine._trading_day = engine._exchange_trading_date()
    return engine


def _fresh_quote(engine: LiveTradingEngine, code: str = "600519",
                 price: float = 10.0, age: float = 0.0) -> None:
    engine._current_prices[code] = price
    engine._price_last_seen[code] = time.monotonic() - age


def _buy_signal(weight: float = 0.02) -> TradeSignal:
    return TradeSignal(
        code="600519", side=OrderSide.BUY, target_weight=weight,
        current_weight=0.0, signal_strength=0.5, reason="test signal",
    )


def _partial_result(order) -> BrokerOrder:
    return BrokerOrder(
        order_id=order.order_id, code=order.code, side=order.side,
        order_type=order.order_type, quantity=order.quantity,
        price=order.price, filled_quantity=1000, filled_price=10.0,
        status=OrderStatus.PARTIAL, commission=5.0, tax=0.0,
    )


def test_partial_fill_is_persisted_counted_and_audited(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=1_000_000)
    engine = _make_engine(tmp_path, monkeypatch, broker)
    _fresh_quote(engine)
    monkeypatch.setattr(
        engine, "_generate_signals", lambda positions=None: [_buy_signal()],
    )
    monkeypatch.setattr(broker, "place_order", _partial_result)

    cycle = engine._execute_cycle()

    orders = engine._store.get_orders()
    trades = engine._store.get_trades()
    assert len(orders) == 1, "partial fill must persist an order row"
    assert orders[0]["status"] == "partial"
    assert orders[0]["filled_quantity"] == 1000
    assert len(trades) == 1, "partial fill must persist a trade row"
    assert trades[0]["order_id"] == orders[0]["order_id"]
    assert trades[0]["quantity"] == 1000
    assert engine._trade_count == 1
    assert engine._store.get_events(topic="audit.order_filled"), (
        "partial fill must be audited as ORDER_FILLED"
    )
    assert cycle.orders and cycle.orders[0]["status"] == "partial"


def test_derisking_rejection_is_audited_as_rejected(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    broker.place_order(
        Order(code="600519", side=OrderSide.BUY, quantity=1000, price=10.0)
    )
    broker.new_trading_day()
    engine = _make_engine(tmp_path, monkeypatch, broker)
    _fresh_quote(engine)

    def reject(order):
        return BrokerOrder(
            order_id=order.order_id, code=order.code, side=order.side,
            order_type=order.order_type, quantity=order.quantity,
            price=order.price, status=OrderStatus.REJECTED,
            error_msg="No liquidity available",
        )

    monkeypatch.setattr(broker, "place_order", reject)
    engine._risk.activate_kill_switch("test")

    cycle = engine._execute_cycle()

    assert cycle.orders and cycle.orders[0]["status"] == "rejected"
    assert engine._store.get_events(topic="audit.order_rejected"), (
        "a rejected de-risking sell must be audited as ORDER_REJECTED"
    )
    assert not engine._store.get_events(topic="audit.order_submitted"), (
        "a rejected de-risking sell was previously audited as ORDER_SUBMITTED"
    )
    rows = engine._store.get_orders()
    assert rows and rows[0]["status"] == "rejected"


def test_fill_survives_audit_event_failure(tmp_path, monkeypatch):
    """An exception from the audit sink must not lose the broker fill."""
    broker = SimulatedBroker(initial_cash=1_000_000)
    engine = _make_engine(tmp_path, monkeypatch, broker)
    _fresh_quote(engine)
    monkeypatch.setattr(
        engine, "_generate_signals", lambda positions=None: [_buy_signal()],
    )

    real_log = engine._audit.log

    def failing_log(action, component, details=None, reason="", result="success"):
        if action == AuditAction.ORDER_FILLED:
            raise RuntimeError("audit store unavailable")
        return real_log(action, component, details=details, reason=reason,
                        result=result)

    monkeypatch.setattr(engine._audit, "log", failing_log)

    cycle = engine._execute_cycle()

    assert cycle.orders and cycle.orders[0]["status"] == "filled"
    assert engine._store.get_orders(), "order row lost when audit raised"
    assert engine._store.get_trades(), "trade row lost when audit raised"
    assert engine._store.get_pnl_history(days=1), (
        "the cycle aborted before P&L persistence when the audit sink raised"
    )


def test_fill_survives_log_order_failure(tmp_path, monkeypatch):
    """The original D3 repro: one `log_order` exception lost the fill."""
    broker = SimulatedBroker(initial_cash=1_000_000)
    engine = _make_engine(tmp_path, monkeypatch, broker)
    _fresh_quote(engine)
    monkeypatch.setattr(
        engine, "_generate_signals", lambda positions=None: [_buy_signal()],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("audit log_order down")

    monkeypatch.setattr(engine._audit, "log_order", boom)

    cycle = engine._execute_cycle()

    assert cycle.orders and cycle.orders[0]["status"] == "filled"
    assert len(engine._store.get_orders()) == 1
    assert len(engine._store.get_trades()) == 1
    assert engine._store.get_pnl_history(days=1), (
        "the cycle was aborted by the audit exception"
    )
