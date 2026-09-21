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


class TestStoppingASessionKeepsTheStartRecord:
    """A session is written twice under one id: by `start()` and by `stop()`.

    `stop()` supplies only `{session_id, status, stopped_at}` (see
    core/scheduler.py:106 and trading/engine.py:241). `INSERT OR REPLACE`
    deleted the row and reinserted it from that dict, so the defaults for every
    absent key overwrote what `start()` had written.
    """

    def _start(self, store, **overrides):
        session = {
            "session_id": "sess-1",
            "broker": "SimulatedBroker",
            "status": "active",
            "started_at": "2026-01-05T09:30:00",
            "total_trades": 7,
            "total_pnl": 1234.5,
            "config": {"n_stocks": 300},
        }
        session.update(overrides)
        store.save_session(session)

    def _only(self, store) -> dict:
        sessions = store.get_sessions()
        assert len(sessions) == 1, "stopping must not create a second session"
        return sessions[0]

    def test_started_at_is_not_rewritten_to_the_stop_time(self, store):
        self._start(store)
        store.save_session({
            "session_id": "sess-1",
            "status": "stopped",
            "stopped_at": "2026-01-05T15:00:00",
        })

        row = self._only(store)
        assert row["started_at"] == "2026-01-05T09:30:00"
        assert row["stopped_at"] == "2026-01-05T15:00:00"
        assert row["status"] == "stopped"

    def test_the_other_start_fields_survive(self, store):
        self._start(store)
        store.save_session({
            "session_id": "sess-1",
            "status": "stopped",
            "stopped_at": "2026-01-05T15:00:00",
        })

        row = self._only(store)
        assert row["broker"] == "SimulatedBroker"
        assert row["total_trades"] == 7
        assert row["total_pnl"] == 1234.5
        assert json.loads(row["config"]) == {"n_stocks": 300}

    def test_a_supplied_field_still_overwrites(self, store):
        """The engine's stop() passes a final trade count; that must land."""
        self._start(store)
        store.save_session({
            "session_id": "sess-1",
            "status": "stopped",
            "stopped_at": "2026-01-05T15:00:00",
            "total_trades": 19,
        })

        row = self._only(store)
        assert row["total_trades"] == 19
        assert row["started_at"] == "2026-01-05T09:30:00", "and nothing else moved"

    def test_saving_twice_without_a_stop_time_keeps_the_row(self, store):
        self._start(store)
        store.save_session({"session_id": "sess-1", "status": "active"})

        row = self._only(store)
        assert row["started_at"] == "2026-01-05T09:30:00"
        assert row["broker"] == "SimulatedBroker"


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


class TestTwoTenantsCanHoldTheSameTicker:
    """`positions` was keyed by `code` alone.

    A `tenant_id` column existed and was indexed, but was not part of the
    primary key, and `INSERT OR REPLACE` matches on the key. So the second
    tenant to save a ticker replaced the first tenant's row outright --
    `get_positions(tenant_id=...)` could not recover it, because the row was
    gone.
    """

    @staticmethod
    def _save(store, tenant_id, quantity, code="600519"):
        store.save_position({
            "tenant_id": tenant_id,
            "code": code,
            "quantity": quantity,
            "available": quantity,
            "avg_cost": 10.0,
        })

    def test_both_tenants_rows_survive(self, store):
        self._save(store, "fund_a", 100)
        self._save(store, "fund_b", 200)

        assert len(store.get_positions()) == 2

    def test_each_tenant_reads_only_its_own(self, store):
        self._save(store, "fund_a", 100)
        self._save(store, "fund_b", 200)

        assert [p["quantity"] for p in store.get_positions("fund_a")] == [100]
        assert [p["quantity"] for p in store.get_positions("fund_b")] == [200]

    def test_a_tenant_can_update_its_own_row(self, store):
        self._save(store, "fund_a", 100)
        self._save(store, "fund_b", 200)

        self._save(store, "fund_a", 150)

        assert [p["quantity"] for p in store.get_positions("fund_a")] == [150]
        assert [p["quantity"] for p in store.get_positions("fund_b")] == [200]

    def test_deleting_one_tenant_position_leaves_the_other(self, store):
        self._save(store, "fund_a", 100)
        self._save(store, "fund_b", 200)

        store.delete_position("600519", tenant_id="fund_a")

        assert store.get_positions("fund_a") == []
        assert [p["quantity"] for p in store.get_positions("fund_b")] == [200]


class TestThePrimaryKeyMigration:
    """An existing database must be rebuilt, since SQLite cannot alter a
    primary key."""

    @staticmethod
    def _legacy_db(path):
        """A database with the old `code TEXT PRIMARY KEY` schema."""
        import sqlite3
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE positions (
                code TEXT PRIMARY KEY,
                tenant_id TEXT DEFAULT 'default',
                name TEXT DEFAULT '',
                quantity INTEGER NOT NULL,
                available INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                current_price REAL DEFAULT 0,
                market_value REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO positions (code, tenant_id, quantity, available, avg_cost, updated_at)"
            " VALUES ('600519', 'default', 100, 100, 10.0, '2026-01-01')"
        )
        conn.commit()
        conn.close()

    def test_existing_rows_are_kept_and_the_key_becomes_composite(self, tmp_path):
        db = str(tmp_path / "legacy.db")
        self._legacy_db(db)

        store = Store(db)  # opening runs the migration

        kept = store.get_positions()
        assert len(kept) == 1
        assert kept[0]["code"] == "600519"
        assert kept[0]["quantity"] == 100

        # The new key admits a second tenant on the same ticker.
        store.save_position({"tenant_id": "fund_b", "code": "600519",
                             "quantity": 200, "available": 200, "avg_cost": 11.0})
        assert len(store.get_positions()) == 2

    def test_reopening_does_not_migrate_twice(self, tmp_path):
        db = str(tmp_path / "legacy.db")
        self._legacy_db(db)

        Store(db).save_position({"tenant_id": "fund_b", "code": "600519",
                                 "quantity": 200, "available": 200, "avg_cost": 11.0})
        reopened = Store(db)

        assert len(reopened.get_positions()) == 2
