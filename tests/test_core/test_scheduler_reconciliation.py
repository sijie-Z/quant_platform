"""R11: daily books-vs-broker reconciliation wired into the EOD path.

expected = Store position ledger + the day's P&L snapshot cash;
actual   = broker positions + broker account cash.

The broker contract is strict: ``get_positions() is None`` means the book is
unknown, never flat. An unknown comparison is persisted as ``status
"unknown"`` and never as a pass.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_platform.core.events import EventBus
from quant_platform.core.scheduler import TradingScheduler
from quant_platform.core.state_machine import PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.trading.broker import Position

TZ = ZoneInfo("Asia/Shanghai")
DAY = "2026-10-20"


def _scheduler(tmp_path, clock, *, broker=None, bus=None, name="recon.db"):
    store = Store(str(tmp_path / name))
    scheduler = TradingScheduler(
        PortfolioStateMachine(), store, bus or EventBus(), broker=broker,
    )
    scheduler._now = lambda: clock[0]
    return scheduler, store


def _seed_store_position(
    store, code="600519", quantity=100, price=50.0, avg_cost=45.0,
):
    store.save_position({
        "code": code, "quantity": quantity, "available": quantity,
        "avg_cost": avg_cost, "current_price": price,
        "market_value": quantity * price,
    })


def _result_row_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM reconciliation_results").fetchone()[0]
    finally:
        conn.close()


class FakeBroker:
    """Broker double: injectable positions/account, like the real contract."""

    def __init__(self, positions=None, cash=100_000.0, connected=True,
                 account=None):
        self._positions = positions
        self._cash = cash
        self._connected = connected
        self._account = account

    def get_positions(self):
        return self._positions

    def get_account(self):
        if self._account is not None:
            return dict(self._account)
        market_value = sum(
            p.quantity * p.current_price for p in (self._positions or [])
        )
        return {
            "cash": self._cash,
            "market_value": market_value,
            "total_equity": self._cash + market_value,
            "connected": self._connected,
        }


class TestPassedReconciliation:
    def test_matching_books_persist_a_pass_and_enrich_the_event(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        _seed_store_position(store)

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row is not None
        assert row["status"] == "passed"
        assert row["n_positions"] == 1
        assert row["n_mismatched"] == 0
        assert row["cash_diff"] == pytest.approx(0.0)
        assert row["total_mismatch_value"] == pytest.approx(0.0)

        assert len(events) == 1
        payload = events[0].data
        # Legacy keys stay; the reconciliation outcome is added.
        for key in (
            "date", "positions", "market_value", "total_equity",
            "equity_source", "catch_up",
        ):
            assert key in payload
        assert payload["status"] == "passed"
        assert payload["passes"] is True
        assert payload["n_mismatched"] == 0
        assert payload["cash_diff"] == pytest.approx(0.0)
        assert payload["total_mismatch_value"] == pytest.approx(0.0)

        # The result table holds exactly one row for the day.
        assert _result_row_count(store._db_path) == 1

    def test_a_second_run_same_day_writes_no_second_result_or_event(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        _seed_store_position(store)

        scheduler._tick()
        scheduler._tick()

        assert _result_row_count(store._db_path) == 1
        assert len(events) == 1
        assert len(store.get_pnl_history(
            start_date=DAY, end_date=DAY)) == 1


class TestFailedReconciliation:
    def test_share_mismatch_fails_records_details_and_logs(self, tmp_path, caplog):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=90,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        _seed_store_position(store, quantity=100)

        with caplog.at_level(logging.ERROR, logger="quant_platform"):
            scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "failed"
        assert row["n_mismatched"] == 1
        detail = {d["asset"]: d for d in row["details"]}[
            "600519"
        ]
        assert detail["expected_shares"] == 100
        assert detail["actual_shares"] == 90
        assert detail["diff"] == 10
        assert detail["is_matched"] is False
        assert any(
            "600519" in record.getMessage()
            and "expected=100" in record.getMessage()
            for record in caplog.records
        ), "the failing position must be logged with asset/expected/actual/diff"
        assert events[-1].data["status"] == "failed"
        assert events[-1].data["passes"] is False

    def test_share_difference_within_the_env_tolerance_passes(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("QUANT_RECON_SHARES_TOLERANCE", "10")
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=95,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_store_position(store, quantity=100)  # diff 5 <= 10

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "passed"
        assert row["tolerances"]["shares"] == pytest.approx(10.0)

    def test_cash_breach_fails_even_when_positions_match(self, tmp_path, caplog):
        """Upgrade-day backfill: the snapshot already exists, so a cash
        breach between the ledger and the broker is observable."""
        clock = [datetime(2026, 10, 20, 15, 10, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(positions=[], cash=80_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        store.save_pnl_snapshot({
            "timestamp": f"{DAY}T15:05:00", "total_equity": 100_000.0,
            "cash": 100_000.0, "market_value": 0.0, "n_positions": 0,
        })
        store.mark_eod_reconciliation(DAY, f"{DAY}T15:05:00")

        with caplog.at_level(logging.ERROR, logger="quant_platform"):
            scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "failed"
        assert row["cash_diff"] == pytest.approx(20_000.0)
        assert any(
            "cash mismatch" in record.getMessage()
            for record in caplog.records
        )
        # The backfill never writes a second snapshot.
        assert len(store.get_pnl_history(
            start_date=DAY, end_date=DAY)) == 1
        assert events[-1].data.get("backfill") is True

    def test_cash_tolerance_env_widens_or_narrows_the_limit(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("QUANT_RECON_CASH_TOLERANCE", "100")
        clock = [datetime(2026, 10, 20, 15, 10, tzinfo=TZ)]
        broker = FakeBroker(positions=[], cash=99_500.0)  # diff 500 > 100
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        store.save_pnl_snapshot({
            "timestamp": f"{DAY}T15:05:00", "total_equity": 100_000.0,
            "cash": 100_000.0, "market_value": 0.0, "n_positions": 0,
        })
        store.mark_eod_reconciliation(DAY, f"{DAY}T15:05:00")

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "failed"
        assert store.get_reconciliation_result(
            DAY)["tolerances"]["cash"] == pytest.approx(100.0)


class TestUnknownBook:
    def test_none_positions_are_unknown_never_a_pass(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(positions=None, cash=100_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        _seed_store_position(store)

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "unknown"
        assert row["cash_diff"] is None, "unknown must not persist as zero"
        assert row["total_mismatch_value"] is None
        assert events[-1].data["status"] == "unknown"
        assert events[-1].data["passes"] is False

    def test_no_broker_attached_is_unknown(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store = _scheduler(tmp_path, clock)
        store.save_engine_state(
            {"portfolio_value": 100_000.0, "cash": 100_000.0, "positions": []},
            "live_engine:test",
        )

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "unknown"

    def test_a_disconnected_broker_is_unknown(self, tmp_path):
        """A disconnected QMT serves placeholder balances; the comparison
        must not run against them."""
        clock = [datetime(2026, 10, 20, 15, 10, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[],
            account={
                "connected": False, "cash": 1_000_000.0,
                "total_equity": 1_000_000.0, "market_value": 0.0,
            },
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        store.save_pnl_snapshot({
            "timestamp": f"{DAY}T15:05:00", "total_equity": 100_000.0,
            "cash": 100_000.0, "market_value": 0.0, "n_positions": 0,
        })
        store.mark_eod_reconciliation(DAY, f"{DAY}T15:05:00")

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "unknown"
        assert row["cash_diff"] is None

    def test_an_unreadable_store_book_is_unknown_not_empty(
        self, tmp_path, monkeypatch,
    ):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_store_position(store)

        def boom(tenant_id=""):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(store, "get_positions", boom)

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "unknown", (
            "an unreadable expected book must not be reconciled as empty"
        )


class TestBackfillAndHistoricalDays:
    def test_a_missing_result_is_backfilled_without_a_new_snapshot(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 10, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(positions=[], cash=100_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        store.save_pnl_snapshot({
            "timestamp": f"{DAY}T15:05:00", "total_equity": 100_000.0,
            "cash": 100_000.0, "market_value": 0.0, "n_positions": 0,
        })
        store.mark_eod_reconciliation(DAY, f"{DAY}T15:05:00")

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "passed"
        assert len(store.get_pnl_history(
            start_date=DAY, end_date=DAY)) == 1, "no duplicate snapshot"
        assert _result_row_count(store._db_path) == 1
        assert events[-1].data.get("backfill") is True

        # A later tick sees the result and does nothing again.
        scheduler._tick()
        assert _result_row_count(store._db_path) == 1
        assert len(events) == 1

    def test_a_historical_day_never_gets_a_reconciliation_result(self, tmp_path):
        clock = [datetime(2026, 10, 21, 10, 0, tzinfo=TZ)]
        broker = FakeBroker(positions=[], cash=100_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        store.mark_eod_reconciliation(DAY, f"{DAY}T15:05:00")

        # Catch-up skips marked days; call directly to exercise the guard.
        assert scheduler._do_eod_reconciliation(DAY, catch_up=True) is False

        assert store.get_reconciliation_result(DAY) is None
        assert _result_row_count(store._db_path) == 0


class TestToleranceConfiguration:
    @pytest.mark.parametrize(("env", "value"), [
        ("QUANT_RECON_SHARES_TOLERANCE", "abc"),
        ("QUANT_RECON_CASH_TOLERANCE", "-1"),
        ("QUANT_RECON_CASH_TOLERANCE_BPS", "nan"),
        ("QUANT_RECON_MATERIALITY", "inf"),
    ])
    def test_invalid_env_values_fail_loud_at_construction(
        self, tmp_path, monkeypatch, env, value,
    ):
        monkeypatch.setenv(env, value)
        store = Store(str(tmp_path / "bad_env.db"))

        with pytest.raises(ValueError, match=env):
            TradingScheduler(PortfolioStateMachine(), store, EventBus())

    def test_defaults_match_the_reconcile_contract(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_store_position(store)

        scheduler._tick()

        tolerances = store.get_reconciliation_result(DAY)["tolerances"]
        assert tolerances == {
            "shares": 0.0, "cash": 1000.0, "cash_bps": 0.0,
            "materiality": 0.0,
        }
