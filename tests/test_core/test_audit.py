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


class TestTenantAttribution:
    """Audit writes used the store default tenant.

    Engine logs come from a worker thread where the constructor's ContextVar
    is not visible, so the AuditLog must carry the tenant explicitly.
    """

    def test_orders_trades_and_events_carry_the_tenant(self, tmp_path):
        store = Store(str(tmp_path / "tenant.db"))
        al = AuditLog(store, EventBus(), tenant_id="fund_a")

        al.log_order({
            "order_id": "o1", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0,
            "filled_quantity": 100, "filled_price": 1800.0,
        }, AuditAction.ORDER_FILLED)

        assert [o["order_id"] for o in store.get_orders(tenant_id="fund_a")] == ["o1"]
        assert len(store.get_trades(tenant_id="fund_a")) == 1
        assert len(store.get_events(tenant_id="fund_a")) == 1
        assert store.get_orders(tenant_id="fund_b") == []

    def test_closing_a_position_spares_other_tenants(self, tmp_path):
        store = Store(str(tmp_path / "positions.db"))
        store.save_position({
            "code": "600519", "tenant_id": "fund_a", "quantity": 100,
            "avg_cost": 10.0,
        })
        store.save_position({
            "code": "600519", "tenant_id": "fund_b", "quantity": 200,
            "avg_cost": 20.0,
        })
        al = AuditLog(store, EventBus(), tenant_id="fund_a")

        al.log_position({"code": "600519", "quantity": 0},
                        AuditAction.POSITION_CLOSED)

        assert {p["tenant_id"] for p in store.get_positions()} == {"fund_b"}

    def test_signals_carry_the_tenant(self, tmp_path):
        store = Store(str(tmp_path / "signals.db"))
        al = AuditLog(store, EventBus(), tenant_id="fund_a")

        al.log_signal("600519", "buy", 0.5, strategy="mom")

        assert len(store.get_signals(tenant_id="fund_a")) == 1


class TestRepeatedFillsOnOneOrder:
    """A second ORDER_FILLED used to fail after the order row was written.

    `log_order` passed `trade_id = order_id`, so the second fill hit the
    `trades` UNIQUE constraint; because order and trade were separate
    transactions, the order had already been updated and the fill (plus its
    audit event) was lost.
    """

    @staticmethod
    def _fill(**overrides):
        order = {
            "order_id": "o1", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0,
            "filled_quantity": 50, "filled_price": 1800.0,
            "status": "partially_filled", "strategy_id": "momentum",
        }
        order.update(overrides)
        return order

    def test_both_fills_land_in_the_trades_table(self, audit):
        al, store, _ = audit

        al.log_order(self._fill(), AuditAction.ORDER_FILLED)
        al.log_order(self._fill(filled_quantity=100, filled_price=1801.0,
                                status="filled"), AuditAction.ORDER_FILLED)

        trades = store.get_trades()
        assert len(trades) == 2
        assert sorted(t["quantity"] for t in trades) == [50, 100]

    def test_both_fill_events_are_logged(self, audit):
        al, store, _ = audit

        al.log_order(self._fill(), AuditAction.ORDER_FILLED)
        al.log_order(self._fill(status="filled"), AuditAction.ORDER_FILLED)

        assert len(store.get_events(topic="audit.order_filled")) == 2

    def test_the_fill_dto_does_not_erase_the_strategy_id(self, audit):
        al, store, _ = audit
        al.log_order(self._fill(), AuditAction.ORDER_SUBMITTED)

        # The broker's fill DTO omits attribution fields.
        al.log_order({
            "order_id": "o1", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0,
            "filled_quantity": 100, "filled_price": 1800.0, "status": "filled",
        }, AuditAction.ORDER_FILLED)

        row = store.get_orders()[0]
        assert row["strategy_id"] == "momentum"
        assert row["status"] == "filled"

