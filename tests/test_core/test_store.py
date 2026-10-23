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
        store.delete_position("600519", tenant_id="default")
        assert len(store.get_positions()) == 0

    def test_delete_position_requires_a_tenant(self, store):
        """An unscoped delete used to remove every tenant's row."""
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 1800.0})
        with pytest.raises(ValueError):
            store.delete_position("600519")
        assert len(store.get_positions()) == 1

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


class TestEngineState:
    def test_round_trip(self, store):
        state = {"version": 1, "cash": 1.0, "positions": [{"asset": "A"}]}
        store.save_engine_state(state, "strategy-a")
        assert store.get_engine_state("strategy-a") == state

    def test_missing_returns_none(self, store):
        assert store.get_engine_state("does-not-exist") is None

    def test_overwrite_keeps_one_row(self, store):
        store.save_engine_state({"v": 1}, "s")
        store.save_engine_state({"v": 2}, "s")
        assert store.get_engine_state("s") == {"v": 2}

    def test_survives_reopen(self, tmp_path):
        db = str(tmp_path / "state.db")
        Store(db).save_engine_state({"v": 3}, "s")
        assert Store(db).get_engine_state("s") == {"v": 3}


class TestPartialOrderSavesKeepTheRestOfTheRow:
    """A broker fill DTO carries only the fields the broker knows about.

    `save_order` used `INSERT OR REPLACE`, which deletes the row and
    reinserts it from the incoming dict, so every column absent from a
    later partial save was reset to its default. `OrderResult.to_dict()`
    (trading/broker.py) has no strategy_id/signal_id/broker_order_id, so a
    fill silently erased the attribution written when the order was
    submitted.
    """

    @staticmethod
    def _submitted(store, **overrides):
        order = {
            "order_id": "o1", "code": "600519", "side": "buy",
            "quantity": 100, "price": 1800.0, "status": "submitted",
            "strategy_id": "momentum", "signal_id": "sig-1",
            "broker_order_id": "broker-9", "tenant_id": "fund_a",
            "created_at": "2026-01-05T09:30:00",
        }
        order.update(overrides)
        store.save_order(order)

    @staticmethod
    def _fill_dto(**overrides):
        """The shape `trading/broker.py:OrderResult.to_dict()` returns."""
        dto = {
            "order_id": "o1", "code": "600519", "side": "buy",
            "order_type": "limit", "quantity": 100, "price": 1800.0,
            "filled_quantity": 100, "filled_price": 1800.0,
            "status": "filled", "commission": 5.4, "tax": 0.0,
            "slippage": 0.0, "error_msg": "",
            "created_at": "2026-01-05T09:30:00",
            "updated_at": "2026-01-05T15:00:00",
        }
        dto.update(overrides)
        return dto

    def test_a_fill_does_not_blank_the_attribution_columns(self, store):
        self._submitted(store)

        store.save_order(self._fill_dto())

        row = store.get_orders()[0]
        assert row["strategy_id"] == "momentum"
        assert row["signal_id"] == "sig-1"
        assert row["broker_order_id"] == "broker-9"

    def test_supplied_columns_still_overwrite(self, store):
        self._submitted(store)

        store.save_order(self._fill_dto())

        row = store.get_orders()[0]
        assert row["status"] == "filled"
        assert row["filled_quantity"] == 100
        assert row["commission"] == 5.4
        assert row["updated_at"] == "2026-01-05T15:00:00"

    def test_a_missing_tenant_does_not_rewrite_the_row_tenant(self, store):
        self._submitted(store)

        store.save_order(self._fill_dto())  # no tenant_id in the DTO

        assert store.get_orders()[0]["tenant_id"] == "fund_a"

    def test_created_at_survives_an_update_that_omits_it(self, store):
        self._submitted(store)

        store.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 1800.0,
                          "filled_quantity": 100, "status": "filled"})

        assert store.get_orders()[0]["created_at"] == "2026-01-05T09:30:00"

    def test_a_first_insert_uses_the_sql_defaults_it_does_not_supply(self, store):
        store.save_order({"order_id": "o2", "code": "000001", "side": "sell",
                          "quantity": 200, "price": 10.0})

        row = store.get_orders()[0]
        assert row["tenant_id"] == "default"
        assert row["order_type"] == "limit"
        assert row["status"] == "pending"
        assert row["filled_quantity"] == 0
        assert row["strategy_id"] == ""
        assert row["created_at"], "created_at has no SQL default and must be set"
        assert row["updated_at"]


class TestMultipleFillsForOneOrder:
    """`trade_id` used to default to `order_id`.

    An order filled in several prints got one UNIQUE conflict per extra
    print: by then `save_order` had already replaced the order row, and the
    trade (and its audit event) were lost.
    """

    def _order(self, store):
        store.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 1800.0, "status": "submitted"})

    def test_two_fills_are_both_recorded(self, store):
        self._order(store)
        base = {"order_id": "o1", "code": "600519", "side": "buy", "quantity": 50}

        store.save_trade({**base, "price": 1800.0})
        store.save_trade({**base, "price": 1801.0})

        trades = store.get_trades()
        assert len(trades) == 2
        assert len({t["trade_id"] for t in trades}) == 2
        assert sorted(t["price"] for t in trades) == [1800.0, 1801.0]

    def test_an_explicit_trade_id_is_kept(self, store):
        self._order(store)
        store.save_trade({"trade_id": "t1", "order_id": "o1", "code": "600519",
                          "side": "buy", "quantity": 50, "price": 1800.0})

        assert store.get_trades()[0]["trade_id"] == "t1"


class TestSaveOrderWithTrade:
    def test_writes_the_order_and_the_trade_together(self, store):
        store.save_order_with_trade(
            {"order_id": "o1", "code": "600519", "side": "buy",
             "quantity": 100, "price": 1800.0, "status": "filled"},
            {"order_id": "o1", "code": "600519", "side": "buy",
             "quantity": 100, "price": 1800.0},
        )

        assert len(store.get_orders()) == 1
        assert len(store.get_trades()) == 1
        assert store.get_trades()[0]["trade_id"] != "o1", "uuid4, not the order id"

    def test_a_bad_trade_rolls_the_order_back(self, store):
        """One transaction: no half-written order when the fill is invalid."""
        with pytest.raises(KeyError):
            store.save_order_with_trade(
                {"order_id": "o1", "code": "600519", "side": "buy",
                 "quantity": 100, "price": 1800.0, "status": "filled"},
                {"order_id": "o1", "code": "600519", "side": "buy"},  # no quantity
            )

        assert store.get_orders() == []
        assert store.get_trades() == []

    def test_a_second_fill_updates_the_order_and_appends_the_trade(self, store):
        order = {"order_id": "o1", "code": "600519", "side": "buy",
                 "quantity": 100, "price": 1800.0, "status": "partially_filled",
                 "strategy_id": "momentum"}
        store.save_order_with_trade(
            order,
            {"order_id": "o1", "code": "600519", "side": "buy",
             "quantity": 50, "price": 1800.0},
        )
        store.save_order_with_trade(
            {**order, "status": "filled", "filled_quantity": 100},
            {"order_id": "o1", "code": "600519", "side": "buy",
             "quantity": 50, "price": 1801.0},
        )

        assert len(store.get_orders()) == 1
        assert len(store.get_trades()) == 2
        row = store.get_orders()[0]
        assert row["status"] == "filled"
        assert row["strategy_id"] == "momentum"


class TestEodReconciliationMarker:
    def test_marker_round_trip(self, store):
        assert store.has_eod_reconciliation("2026-10-20") is False

        store.mark_eod_reconciliation("2026-10-20", "2026-10-20T15:30:00")

        assert store.has_eod_reconciliation("2026-10-20") is True
        assert store.has_eod_reconciliation("2026-10-19") is False

    def test_marking_the_same_day_twice_keeps_one_row(self, tmp_path):
        db = str(tmp_path / "eod.db")
        store = Store(db)
        store.mark_eod_reconciliation("2026-10-20", "first")
        store.mark_eod_reconciliation("2026-10-20", "second")

        import sqlite3
        conn = sqlite3.connect(db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM eod_reconciliation").fetchone()[0]
        finally:
            conn.close()
        assert count == 1

    def test_an_existing_database_gains_the_table(self, tmp_path):
        """Databases written before the table existed must migrate on open."""
        import sqlite3

        db = str(tmp_path / "legacy.db")
        first = Store(db)
        first.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 1800.0})
        first.close()

        # Simulate a database created before the eod_reconciliation table.
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE eod_reconciliation")
        conn.commit()
        conn.close()

        reopened = Store(db)
        assert reopened.has_eod_reconciliation("2026-10-20") is False
        reopened.mark_eod_reconciliation("2026-10-20")

        assert reopened.has_eod_reconciliation("2026-10-20") is True
        assert len(reopened.get_orders()) == 1, "existing rows are untouched"


class TestDailyReconciliationResults:
    """R11: the books-vs-broker outcome is persisted, one row per date."""

    @staticmethod
    def _result(**overrides):
        result = {
            "date": "2026-10-20",
            "timestamp": "2026-10-20T15:05:00",
            "status": "failed",
            "n_positions": 2,
            "n_mismatched": 1,
            "cash_diff": -125.5,
            "total_mismatch_value": 500.0,
            "details": [
                {"asset": "600519", "expected_shares": 100,
                 "actual_shares": 90, "diff": 10, "is_matched": False},
            ],
            "tolerances": {"shares": 0, "cash": 1000.0,
                           "cash_bps": 0, "materiality": 0},
        }
        result.update(overrides)
        return result

    def test_round_trip(self, store):
        store.save_reconciliation_result(self._result())

        row = store.get_reconciliation_result("2026-10-20")

        assert row["status"] == "failed"
        assert row["n_positions"] == 2
        assert row["n_mismatched"] == 1
        assert row["cash_diff"] == pytest.approx(-125.5)
        assert row["total_mismatch_value"] == pytest.approx(500.0)
        assert row["details"][0]["asset"] == "600519"
        assert row["details"][0]["diff"] == 10
        assert row["tolerances"]["cash"] == pytest.approx(1000.0)

    def test_one_row_per_date(self, store):
        store.save_reconciliation_result(self._result())
        store.save_reconciliation_result(self._result(status="passed"))

        import sqlite3
        conn = sqlite3.connect(store._db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM reconciliation_results").fetchone()[0]
        finally:
            conn.close()
        assert count == 1
        assert store.get_reconciliation_result("2026-10-20")["status"] == "passed"

    def test_unknown_keeps_nulls_instead_of_zero(self, store):
        """An unknown comparison must not persist as a zero diff."""
        store.save_reconciliation_result(self._result(
            status="unknown", cash_diff=None,
            total_mismatch_value=None, details=[],
        ))

        row = store.get_reconciliation_result("2026-10-20")
        assert row["status"] == "unknown"
        assert row["cash_diff"] is None
        assert row["total_mismatch_value"] is None

    def test_an_unknown_status_is_rejected(self, store):
        with pytest.raises(ValueError, match="status"):
            store.save_reconciliation_result(self._result(status="maybe"))

    def test_last_reconciliation_is_the_newest_date(self, store):
        assert store.get_last_reconciliation() is None

        store.save_reconciliation_result(self._result(date="2026-10-19"))
        store.save_reconciliation_result(self._result(date="2026-10-20"))

        assert store.get_last_reconciliation()["date"] == "2026-10-20"
        assert store.get_reconciliation_result("2026-10-18") is None

    def test_an_existing_database_gains_the_table(self, tmp_path):
        import sqlite3

        db = str(tmp_path / "legacy_r11.db")
        first = Store(db)
        first.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 1800.0})
        first.close()

        # Simulate a database created before reconciliation_results existed.
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE reconciliation_results")
        conn.commit()
        conn.close()

        reopened = Store(db)
        assert reopened.get_last_reconciliation() is None
        reopened.save_reconciliation_result(self._result())

        assert reopened.get_reconciliation_result(
            "2026-10-20")["status"] == "failed"
        assert len(reopened.get_orders()) == 1, "existing rows are untouched"

    def test_basis_is_persisted(self, store):
        """R12: the expected-side basis travels with the durable result."""
        store.save_reconciliation_result(self._result())
        assert store.get_reconciliation_result(
            "2026-10-20")["basis"] == "trade_ledger"

        store.save_reconciliation_result(
            self._result(date="2026-10-21", basis="direct"),
        )
        assert store.get_reconciliation_result("2026-10-21")["basis"] == "direct"

    def test_an_existing_results_table_gains_the_basis_column(self, tmp_path):
        """CREATE TABLE IF NOT EXISTS cannot add a column, so an R11-era
        database must be migrated in place."""
        import sqlite3

        db = str(tmp_path / "legacy_basis.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE reconciliation_results (
                date TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                n_positions INTEGER DEFAULT 0,
                n_mismatched INTEGER DEFAULT 0,
                cash_diff REAL,
                total_mismatch_value REAL,
                details TEXT DEFAULT '[]',
                tolerances TEXT DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

        reopened = Store(db)
        reopened.save_reconciliation_result(self._result(
            date="2026-10-20", status="passed",
        ))

        assert reopened.get_reconciliation_result(
            "2026-10-20")["basis"] == "trade_ledger"


class TestLedgerBaselines:
    """R12: append-only broker baselines the trade ledger derives from."""

    @staticmethod
    def _baseline(**overrides):
        baseline = {
            "created_at": "2026-10-20T09:00:00",
            "tenant_id": "default",
            "engine_state_key": "live_engine:default",
            "cash": 100_000.0,
            "positions_json": {
                "600519": {"quantity": 100, "avg_cost": 45.0},
            },
            "note": "test",
        }
        baseline.update(overrides)
        return baseline

    def test_round_trip(self, store):
        row_id = store.save_ledger_baseline(self._baseline())

        baseline = store.get_latest_ledger_baseline(
            "default", "live_engine:default",
        )

        assert baseline["id"] == row_id
        assert baseline["cash"] == pytest.approx(100_000.0)
        assert baseline["positions"]["600519"]["quantity"] == 100
        assert baseline["created_at"] == "2026-10-20T09:00:00"
        assert baseline["engine_state_key"] == "live_engine:default"
        assert baseline["note"] == "test"

    def test_restarts_append_and_the_newest_id_wins(self, store):
        first_id = store.save_ledger_baseline(self._baseline())
        # Same created_at on purpose: ordering must be by insert id.
        second_id = store.save_ledger_baseline(self._baseline(
            cash=99_000.0, positions_json={},
        ))

        latest = store.get_latest_ledger_baseline()

        assert second_id > first_id
        assert latest["id"] == second_id
        assert latest["cash"] == pytest.approx(99_000.0)

    def test_filters_by_tenant_and_engine_state_key(self, store):
        store.save_ledger_baseline(self._baseline())
        store.save_ledger_baseline(self._baseline(
            tenant_id="other", engine_state_key="live_engine:other", cash=1.0,
        ))

        assert store.get_latest_ledger_baseline(
            "default", "live_engine:default")["cash"] == pytest.approx(100_000.0)
        assert store.get_latest_ledger_baseline(
            "other", "live_engine:other")["cash"] == pytest.approx(1.0)
        assert store.get_latest_ledger_baseline("missing", "") is None
        # An empty filter leaves that column unscoped.
        assert store.get_latest_ledger_baseline(
            "", "live_engine:other")["cash"] == pytest.approx(1.0)
        assert store.get_latest_ledger_baseline(
            "other", "")["cash"] == pytest.approx(1.0)

    def test_cash_is_required_and_must_be_finite(self, store):
        missing = self._baseline()
        del missing["cash"]
        with pytest.raises(ValueError, match="cash"):
            store.save_ledger_baseline(missing)

        with pytest.raises(ValueError, match="finite"):
            store.save_ledger_baseline(self._baseline(cash=float("nan")))

    def test_positions_may_be_json_text(self, store):
        store.save_ledger_baseline(
            self._baseline(positions_json='{"600519": 100}'),
        )
        assert store.get_latest_ledger_baseline()["positions"] == {
            "600519": 100,
        }

    def test_malformed_positions_are_rejected(self, store):
        with pytest.raises(ValueError, match="JSON"):
            store.save_ledger_baseline(self._baseline(positions_json="{nope"))
        with pytest.raises(ValueError, match="positions"):
            store.save_ledger_baseline(self._baseline(positions_json=[1, 2]))

    def test_an_existing_database_gains_the_table(self, tmp_path):
        import sqlite3

        db = str(tmp_path / "legacy_baseline.db")
        first = Store(db)
        first.save_order({"order_id": "o1", "code": "600519", "side": "buy",
                          "quantity": 100, "price": 1800.0})
        first.close()

        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE ledger_baselines")
        conn.commit()
        conn.close()

        reopened = Store(db)
        assert reopened.get_latest_ledger_baseline() is None
        reopened.save_ledger_baseline(self._baseline())

        assert reopened.get_latest_ledger_baseline()["cash"] == pytest.approx(
            100_000.0,
        )
        assert len(reopened.get_orders()) == 1, "existing rows are untouched"

    def test_unreadable_positions_json_raises(self, store):
        """Degrading to {} could 'pass' a flat broker account."""
        import sqlite3

        conn = sqlite3.connect(store._db_path)
        conn.execute(
            "INSERT INTO ledger_baselines "
            "(created_at, tenant_id, engine_state_key, cash, positions_json) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-10-20T09:00:00", "default", "", 1.0, "{nope"),
        )
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="unreadable"):
            store.get_latest_ledger_baseline()

