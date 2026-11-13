"""Ops wave 2 regressions — retention job and the store timestamp basis.

S10: no table ever had rows archived or deleted; the database grew without
bound. S17: new pnl/session/event rows must use one documented basis
(Asia/Shanghai naive), not the host clock.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import quant_core.store as store_module

from quant_platform.core.store import RetentionPolicy, Store, exchange_now_iso
from quant_platform.utils.metrics import get_metrics

NOW = datetime(2026, 10, 20, 15, 0, 0)
OLD_ISO = "2020-01-01T10:00:00"
RECENT_ISO = "2026-10-19T10:00:00"


def _store(tmp_path) -> Store:
    return Store(str(tmp_path / "retention.db"))


def _seed_old_and_new(store: Store) -> None:
    store.save_pnl_snapshot({
        "timestamp": OLD_ISO, "total_equity": 1.0, "cash": 1.0,
        "market_value": 0.0,
    })
    store.save_pnl_snapshot({
        "timestamp": RECENT_ISO, "total_equity": 2.0, "cash": 2.0,
        "market_value": 0.0,
    })
    store.save_session({
        "session_id": "old-session", "started_at": OLD_ISO, "status": "stopped",
    })
    store.save_session({
        "session_id": "new-session", "started_at": RECENT_ISO, "status": "stopped",
    })
    store.save_signal({
        "signal_id": "old-signal", "code": "600519", "direction": "buy",
        "strength": 1.0, "generated_at": OLD_ISO,
    })
    store.save_signal({
        "signal_id": "new-signal", "code": "600519", "direction": "buy",
        "strength": 1.0, "generated_at": RECENT_ISO,
    })
    store.save_nav({
        "date": "2020-01-01", "nav_total": 1.0, "nav_per_unit": 1.0,
        "total_units": 1.0,
    })
    store.save_nav({
        "date": "2026-10-19", "nav_total": 2.0, "nav_per_unit": 2.0,
        "total_units": 1.0,
    })
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO events (event_id, tenant_id, topic, data, source, "
            "timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old-event", "default", "test", "{}", "test", 1.0, OLD_ISO),
        )
        conn.execute(
            "INSERT INTO events (event_id, tenant_id, topic, data, source, "
            "timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("new-event", "default", "test", "{}", "test", 2.0, RECENT_ISO),
        )
    store.save_order_with_trade(
        {
            "order_id": "old-order", "code": "600519", "side": "buy",
            "quantity": 100, "price": 10.0, "status": "filled",
            "created_at": OLD_ISO, "updated_at": OLD_ISO,
        },
        {
            "trade_id": "old-trade", "order_id": "old-order", "code": "600519",
            "side": "buy", "quantity": 100, "price": 10.0,
            "executed_at": OLD_ISO,
        },
    )
    store.save_order_with_trade(
        {
            "order_id": "new-order", "code": "600519", "side": "buy",
            "quantity": 100, "price": 10.0, "status": "filled",
            "created_at": RECENT_ISO, "updated_at": RECENT_ISO,
        },
        {
            "trade_id": "new-trade", "order_id": "new-order", "code": "600519",
            "side": "buy", "quantity": 100, "price": 10.0,
            "executed_at": RECENT_ISO,
        },
    )


def _policy(days: int = 30) -> RetentionPolicy:
    return RetentionPolicy(
        orders_days=days,
        trades_days=days,
        pnl_days=days,
        signals_days=days,
        sessions_days=days,
        events_days=days,
        nav_days=days,
    )


class TestRetention:
    def test_old_rows_are_archived_and_deleted_new_rows_kept(self, tmp_path):
        store = _store(tmp_path)
        _seed_old_and_new(store)
        archive_dir = tmp_path / "archive"

        result = store.run_retention(
            _policy(), archive_dir=archive_dir, now=NOW,
        )

        tables = result["tables"]
        for table in (
            "pnl_history", "sessions", "signals", "events", "nav_history",
            "trades", "orders",
        ):
            assert tables[table]["archived"] == 1, table
            assert tables[table]["deleted"] == 1, table
            assert Path(tables[table]["archive_path"]).exists(), table

        # Old rows gone, recent rows kept.
        pnl = store.get_pnl_history(start_date="2019-01-01", end_date="2027-01-01")
        assert [row["timestamp"] for row in pnl] == [RECENT_ISO]
        assert [s["session_id"] for s in store.get_sessions()] == ["new-session"]
        assert [s["signal_id"] for s in store.get_signals()] == ["new-signal"]
        nav = store.get_nav_history(days=3650)
        assert [row["date"] for row in nav] == ["2026-10-19"]
        assert [o["order_id"] for o in store.get_orders()] == ["new-order"]
        assert [t["trade_id"] for t in store.get_trades()] == ["new-trade"]
        events = store.get_events(limit=10)
        assert [e["event_id"] for e in events] == ["new-event"]

        archived = Path(tables["pnl_history"]["archive_path"])
        with archived.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["timestamp"] == OLD_ISO

    def test_second_run_is_idempotent(self, tmp_path):
        store = _store(tmp_path)
        _seed_old_and_new(store)
        store.run_retention(_policy(), archive_dir=tmp_path / "archive", now=NOW)

        second = store.run_retention(
            _policy(), archive_dir=tmp_path / "archive", now=NOW,
        )

        assert all(
            entry["archived"] == 0 and entry["deleted"] == 0
            for entry in second["tables"].values()
        )

    def test_an_order_referenced_by_a_retained_trade_is_kept(self, tmp_path):
        store = _store(tmp_path)
        # Old order, but its trade is newer than the trades cutoff: deleting
        # the order would orphan a retained fill.
        store.save_order_with_trade(
            {
                "order_id": "old-order-referenced", "code": "600519",
                "side": "buy", "quantity": 100, "price": 10.0,
                "status": "filled", "created_at": OLD_ISO, "updated_at": OLD_ISO,
            },
            {
                "trade_id": "recent-trade", "order_id": "old-order-referenced",
                "code": "600519", "side": "buy", "quantity": 100,
                "price": 10.0, "executed_at": RECENT_ISO,
            },
        )

        store.run_retention(_policy(), archive_dir=tmp_path / "archive", now=NOW)

        assert [
            order["order_id"] for order in store.get_orders()
        ] == ["old-order-referenced"]

    def test_disabled_table_is_untouched(self, tmp_path):
        store = _store(tmp_path)
        _seed_old_and_new(store)
        policy = RetentionPolicy(
            orders_days=0, trades_days=0, pnl_days=0, signals_days=0,
            sessions_days=0, events_days=0, nav_days=0,
        )

        result = store.run_retention(
            policy, archive_dir=tmp_path / "archive", now=NOW,
        )

        assert all(not entry["enabled"] for entry in result["tables"].values())
        assert len(store.get_pnl_history(start_date="2019-01-01",
                                         end_date="2027-01-01")) == 2

    def test_checkpoint_vacuum_and_size_gauge(self, tmp_path):
        store = _store(tmp_path)
        _seed_old_and_new(store)

        result = store.run_retention(
            _policy(), archive_dir=tmp_path / "archive", now=NOW,
        )

        assert result["checkpointed"] is True
        assert result["vacuumed"] is True
        assert result["db_size_bytes"] > 0
        gauge = get_metrics().get_snapshot().get("store_size_bytes")
        assert gauge is not None
        assert gauge["value"] > 0

    def test_cli_command_runs_retention(self, tmp_path, capsys):
        import quant_platform.main as main_module

        db = str(tmp_path / "cli.db")
        store = Store(db)
        _seed_old_and_new(store)
        store.close()

        args = SimpleNamespace(
            subcommand="retention", db=db,
            archive_dir=str(tmp_path / "cli-archive"),
            orders_days=30, trades_days=30, pnl_days=30, signals_days=30,
            sessions_days=30, events_days=30, nav_days=30, no_vacuum=False,
        )
        code = main_module.cmd_data(args)

        assert code == 0
        out = capsys.readouterr().out
        assert "Retention complete" in out
        assert "pnl_history" in out


    def test_more_rows_than_sqlite_variable_limit_are_deleted(self, tmp_path):
        """40k rows used to fail the single IN-list DELETE after archiving."""
        store = _store(tmp_path)
        with store._conn() as conn:
            conn.executemany(
                "INSERT INTO events (event_id, tenant_id, topic, data, source, "
                "timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (f"evt-{i}", "default", "test", "{}", "test", 1.0, OLD_ISO)
                    for i in range(33_100)
                ),
            )
        policy = RetentionPolicy(
            orders_days=0, trades_days=0, pnl_days=0, signals_days=0,
            sessions_days=0, events_days=30, nav_days=0,
        )

        result = store.run_retention(
            policy, archive_dir=tmp_path / "archive", now=NOW,
        )

        assert result["tables"]["events"]["deleted"] == 33_100
        assert store.get_events(limit=1) == []
        with Path(result["tables"]["events"]["archive_path"]).open(
            newline="", encoding="utf-8",
        ) as fh:
            assert sum(1 for _ in csv.DictReader(fh)) == 33_100


class TestLatestEngineState:
    def test_risk_and_daemon_rows_are_not_engine_books(self, tmp_path):
        store = Store(str(tmp_path / "states.db"))
        store.save_engine_state(
            {
                "cash": 90_000.0, "positions": [
                    {"code": "600519", "market_value": 160_000.0},
                ], "portfolio_value": 250_000.0,
            },
            "live_engine:default",
        )
        # Newer rows that are not engine books.
        store.save_engine_state(
            {"portfolio_value": 999_999.0, "breaches": [], "cash": 1.0},
            "risk_state:default",
        )
        store.save_engine_state(
            {"current_equity": 777_777.0, "last_processed_date": "2026-10-19"},
            "live_engine_alpha_v1",
        )

        key, state = store.get_latest_engine_state()

        assert key == "live_engine:default"
        assert state["portfolio_value"] == 250_000.0

    def test_tenant_scope_is_honoured(self, tmp_path):
        store = Store(str(tmp_path / "states.db"))
        for tenant, equity in (("a", 100.0), ("b", 200.0)):
            store.save_engine_state(
                {"cash": equity, "positions": [], "portfolio_value": equity},
                f"live_engine:{tenant}",
            )

        key, state = store.get_latest_engine_state("a")

        assert key == "live_engine:a"
        assert state["portfolio_value"] == 100.0
        assert store.get_latest_engine_state("c") is None


class TestAtomicEodMarker:
    def test_second_claim_writes_no_duplicate_snapshot(self, tmp_path):
        store = Store(str(tmp_path / "eod.db"))
        snapshot = {
            "timestamp": "2026-10-20T15:00:00", "total_equity": 1.0,
            "cash": 1.0, "market_value": 0.0,
        }

        assert store.save_pnl_and_mark_eod(
            snapshot, "2026-10-20", "2026-10-20T15:00:00",
        ) is True
        assert store.save_pnl_and_mark_eod(
            snapshot, "2026-10-20", "2026-10-20T15:00:00",
        ) is False

        rows = store.get_pnl_history(
            start_date="2026-10-20", end_date="2026-10-20",
        )
        assert len(rows) == 1
        assert store.has_eod_reconciliation("2026-10-20") is True


class TestExchangeTimestampBasis:
    def test_exchange_now_ignores_the_host_clock(self, monkeypatch):
        # 2023-11-15 06:13:20 in Shanghai; a UTC-5 host would say 17:13:20
        # on 2023-11-14.
        monkeypatch.setattr(
            store_module, "time", SimpleNamespace(time=lambda: 1_700_000_000),
        )
        calls = []

        class ProbeDatetime:
            @classmethod
            def now(cls, tz=None):
                calls.append(tz)
                if tz is None:
                    return datetime(2023, 11, 14, 17, 13, 20)
                return datetime.now(tz)

            @classmethod
            def fromtimestamp(cls, timestamp, tz=None):
                calls.append(tz)
                return datetime.fromtimestamp(timestamp, tz)

        monkeypatch.setattr(store_module, "datetime", ProbeDatetime)

        assert exchange_now_iso() == "2023-11-15T06:13:20"
        assert None not in calls, (
            "the store timestamp basis must never consult the naive host clock"
        )

    def test_pnl_snapshot_default_timestamp_uses_shanghai_time(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            store_module, "time", SimpleNamespace(time=lambda: 1_700_000_000),
        )
        store = Store(str(tmp_path / "ts.db"))

        store.save_pnl_snapshot({
            "total_equity": 1.0, "cash": 1.0, "market_value": 0.0,
        })

        rows = store.get_pnl_history(start_date="2023-11-15",
                                     end_date="2023-11-15")
        assert len(rows) == 1
        assert rows[0]["timestamp"] == "2023-11-15T06:13:20"

    def test_archive_csv_neutralizes_formula_injection(self, tmp_path):
        """The retention archiver wrote stored values verbatim (CWE-1236)."""
        store = _store(tmp_path)
        payload = "=cmd|' /C calc'!A0"
        store.save_signal({
            "signal_id": "evil-signal", "code": payload, "direction": "buy",
            "strength": 1.0, "generated_at": OLD_ISO,
        })

        result = store.run_retention(
            _policy(), archive_dir=tmp_path / "archive", now=NOW,
        )

        archive = Path(result["tables"]["signals"]["archive_path"])
        with open(archive, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["code"] == "'" + payload
