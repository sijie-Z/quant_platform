"""Tests for core.audit — Audit log."""

import pytest

from quant_platform.core.audit import AuditAction, AuditLog
from quant_platform.core.events import EventBus
from quant_platform.core.store import Store


@pytest.fixture
def audit(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    bus = EventBus()
    return AuditLog(store=store, bus=bus), store, bus


class TestAuditLog:
    def test_log_basic(self, audit):
        al, store, bus = audit
        al.log(AuditAction.ORDER_SUBMITTED, component="test", details={"code": "600519"})
        events = store.get_events()
        assert len(events) == 1
        assert "audit.order_submitted" in events[0]["topic"]

    def test_log_signal(self, audit):
        al, store, bus = audit
        al.log_signal("600519", "buy", 0.8, strategy="momentum", factors={"mom": 0.5})
        signals = store.get_signals()
        assert len(signals) == 1
        assert signals[0]["code"] == "600519"

    def test_log_order(self, audit):
        al, store, bus = audit
        order = {
            "order_id": "o1", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0,
        }
        al.log_order(order, AuditAction.ORDER_SUBMITTED, reason="signal")
        orders = store.get_orders()
        assert len(orders) == 1

    def test_log_order_filled_creates_trade(self, audit):
        al, store, bus = audit
        order = {
            "order_id": "o1", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0,
            "filled_quantity": 100, "filled_price": 1800.0,
        }
        al.log_order(order, AuditAction.ORDER_FILLED)
        trades = store.get_trades()
        assert len(trades) == 1

    def test_log_position(self, audit):
        al, store, bus = audit
        al.log_position({"code": "600519", "quantity": 100, "avg_cost": 1800.0}, AuditAction.POSITION_OPENED)
        positions = store.get_positions()
        assert len(positions) == 1

    def test_log_position_closed(self, audit):
        al, store, bus = audit
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 100})
        al.log_position({"code": "600519"}, AuditAction.POSITION_CLOSED)
        assert len(store.get_positions()) == 0

    def test_log_state_change(self, audit):
        al, store, bus = audit
        al.log_state_change("init", "ready", reason="startup")
        events = store.get_events()
        assert len(events) == 1

    def test_log_risk_breach(self, audit):
        al, store, bus = audit
        al.log_risk_breach("position_limit", {"ticker": "600519"}, severity="warning")
        events = store.get_events()
        assert len(events) == 1

    def test_get_recent(self, audit):
        al, store, bus = audit
        al.log(AuditAction.ORDER_SUBMITTED, "test")
        al.log(AuditAction.ORDER_FILLED, "test")
        al.log(AuditAction.RISK_BREACH, "test")
        assert len(al.get_recent(limit=10)) == 3

    def test_bus_publishes_events(self, audit):
        al, store, bus = audit
        received = []
        bus.subscribe("audit.*", lambda e: received.append(e))
        al.log(AuditAction.ENGINE_START, "test")
        assert len(received) == 1
