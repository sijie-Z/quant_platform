"""Tests for core.store — SQLite persistence."""

import json

import pytest

from quant_platform.core.store import Store


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Store(db_path)


class TestStoreOrders:
    def test_save_and_get_order(self, store):
        order = {
            "order_id": "ord-001", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0, "status": "pending",
        }
        store.save_order(order)
        orders = store.get_orders()
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ord-001"

    def test_get_orders_by_status(self, store):
        store.save_order({"order_id": "o1", "code": "A", "side": "buy",
                          "quantity": 100, "price": 10, "status": "pending"})
        store.save_order({"order_id": "o2", "code": "B", "side": "sell",
                          "quantity": 200, "price": 20, "status": "filled"})
        assert len(store.get_orders(status="pending")) == 1
        assert len(store.get_orders(status="filled")) == 1

    def test_get_orders_by_code(self, store):
        store.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 10, "status": "pending"})
        store.save_order({"order_id": "o2", "code": "000001", "side": "buy",
                          "quantity": 100, "price": 10, "status": "pending"})
        assert len(store.get_orders(code="600519")) == 1


class TestStorePositions:
    def test_save_and_get_position(self, store):
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 1800.0})
        positions = store.get_positions()
        assert len(positions) == 1
        assert positions[0]["code"] == "600519"

    def test_delete_position(self, store):
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 1800.0})
        store.delete_position("600519")
        assert len(store.get_positions()) == 0

    def test_update_position(self, store):
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 1800.0})
        store.save_position({"code": "600519", "quantity": 200, "avg_cost": 1850.0})
        positions = store.get_positions()
        assert len(positions) == 1
        assert positions[0]["quantity"] == 200


class TestStoreTrades:
    def test_save_and_get_trade(self, store):
        store.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 1800, "status": "filled"})
        store.save_trade({
            "trade_id": "t1", "order_id": "o1", "code": "600519",
            "side": "buy", "quantity": 100, "price": 1800.0,
        })
        trades = store.get_trades()
        assert len(trades) == 1

    def test_get_trades_by_code(self, store):
        store.save_order({"order_id": "o1", "code": "A", "side": "buy",
                          "quantity": 100, "price": 10, "status": "filled"})
        store.save_trade({"trade_id": "t1", "order_id": "o1", "code": "A",
                          "side": "buy", "quantity": 100, "price": 10})
        store.save_order({"order_id": "o2", "code": "B", "side": "buy",
                          "quantity": 100, "price": 10, "status": "filled"})
        store.save_trade({"trade_id": "t2", "order_id": "o2", "code": "B",
                          "side": "buy", "quantity": 100, "price": 10})
        assert len(store.get_trades(code="A")) == 1


class TestStorePnL:
    def test_save_and_get_pnl(self, store):
        store.save_pnl_snapshot({
            "total_equity": 10_000_000, "cash": 5_000_000,
            "market_value": 5_000_000, "daily_pnl": 50_000, "n_positions": 10,
        })
        history = store.get_pnl_history(days=1)
        assert len(history) == 1
        assert history[0]["total_equity"] == 10_000_000


class TestStoreSignals:
    def test_save_and_get_signal(self, store):
        store.save_signal({
            "signal_id": "s1", "code": "600519",
            "direction": "buy", "strength": 0.8,
            "factor_values": {"momentum": 0.5},
        })
        signals = store.get_signals()
        assert len(signals) == 1
        assert signals[0]["factor_values"]["momentum"] == 0.5

    def test_filter_consumed_signals(self, store):
        store.save_signal({"signal_id": "s1", "code": "A",
                           "direction": "buy", "strength": 0.5, "consumed": 0})
        store.save_signal({"signal_id": "s2", "code": "B",
                           "direction": "sell", "strength": 0.3, "consumed": 1})
        assert len(store.get_signals(consumed=0)) == 1
        assert len(store.get_signals(consumed=1)) == 1
        assert len(store.get_signals(consumed=-1)) == 2


class TestStoreSessions:
    def test_save_and_get_session(self, store):
        store.save_session({"session_id": "sess-1", "broker": "simulated", "status": "active"})
        sessions = store.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "sess-1"


class TestStoreEvents:
    def test_log_and_get_event(self, store):
        store.log_event({"event_id": "e1", "topic": "order.filled", "data": {"code": "600519"}})
        events = store.get_events()
        assert len(events) == 1
        assert events[0]["topic"] == "order.filled"

    def test_filter_events_by_topic(self, store):
        store.log_event({"event_id": "e1", "topic": "order.filled", "data": {}})
        store.log_event({"event_id": "e2", "topic": "risk.breach", "data": {}})
        assert len(store.get_events(topic="order.filled")) == 1


class TestStoreConfig:
    def test_save_and_get_config(self, store):
        config = {"optimizer": "mvo", "n_stocks": 300}
        store.save_config_snapshot(config, description="test")
        configs = store.get_config_snapshots()
        assert len(configs) == 1
        stored = json.loads(configs[0]["config"]) if isinstance(configs[0]["config"], str) else configs[0]["config"]
        assert stored["optimizer"] == "mvo"


class TestStoreStats:
    def test_get_stats(self, store):
        store.save_order({"order_id": "o1", "code": "A", "side": "buy",
                          "quantity": 100, "price": 10, "status": "pending"})
        store.save_position({"code": "A", "quantity": 100, "avg_cost": 10})
        stats = store.get_stats()
        assert stats["orders"] == 1
        assert stats["positions"] == 1
