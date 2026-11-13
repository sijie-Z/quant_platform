"""R11/R12: daily books-vs-broker reconciliation wired into the EOD path.

expected = trade ledger derived from the engine-start ledger baseline plus the
fills stored after it (R12); actual = broker positions + broker account cash.

The broker contract is strict: ``get_positions() is None`` means the book is
unknown, never flat. An unknown comparison is persisted as ``status
"unknown"`` and never as a pass. Without a baseline the day is unknown with
reason ``no_ledger_baseline`` -- the Store position book is never used as a
fallback expected leg.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
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


def _scheduler(
    tmp_path, clock, *, broker=None, bus=None, name="recon.db",
    tenant_id="", engine_state_key="",
):
    store = Store(str(tmp_path / name))
    scheduler = TradingScheduler(
        PortfolioStateMachine(), store, bus or EventBus(), broker=broker,
        tenant_id=tenant_id, engine_state_key=engine_state_key,
    )
    scheduler._now = lambda: clock[0]
    return scheduler, store


def _seed_store_position(
    store, code="600519", quantity=100, price=50.0, avg_cost=45.0,
):
    """Seed the persisted position book (R12: NOT the expected side)."""
    store.save_position({
        "code": code, "quantity": quantity, "available": quantity,
        "avg_cost": avg_cost, "current_price": price,
        "market_value": quantity * price,
    })


def _seed_baseline(
    store, *, positions=None, cash=100_000.0, created_at=f"{DAY}T09:00:00",
    tenant="default", engine_state_key="",
):
    return store.save_ledger_baseline({
        "created_at": created_at,
        "tenant_id": tenant,
        "engine_state_key": engine_state_key,
        "cash": cash,
        "positions_json": positions or {},
        "note": "test baseline",
    })


def _save_trade(
    store, *, code="600519", side="buy", quantity=100, price=10.0,
    commission=0.0, tax=0.0, executed_at=f"{DAY}T10:00:00",
    tenant="default",
):
    store.save_trade({
        "trade_id": f"t-{uuid.uuid4().hex[:8]}",
        "tenant_id": tenant,
        "order_id": "ord-test",
        "code": code,
        "side": side,
        "quantity": quantity,
        "price": price,
        "commission": commission,
        "tax": tax,
        "executed_at": executed_at,
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
        # The Store book deliberately disagrees with the ledger: R12's
        # expected side is the baseline, never the persisted book.
        _seed_store_position(store, quantity=500)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row is not None
        assert row["status"] == "passed"
        assert row["basis"] == "trade_ledger"
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
        assert payload["basis"] == "trade_ledger"
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
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )

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
        # Store book agrees with the broker; the ledger (100) does not. The
        # failure must come from the ledger, not from the persisted book.
        _seed_store_position(store, quantity=90)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )

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
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )  # diff 5 <= 10

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
        _seed_baseline(store, positions={}, cash=100_000.0)
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
        _seed_baseline(store, positions={}, cash=100_000.0)
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
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )

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
        _seed_baseline(store, positions={}, cash=100_000.0)
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
        _seed_baseline(store, positions={}, cash=100_000.0)
        store.save_pnl_snapshot({
            "timestamp": f"{DAY}T15:05:00", "total_equity": 100_000.0,
            "cash": 100_000.0, "market_value": 0.0, "n_positions": 0,
        })
        store.mark_eod_reconciliation(DAY, f"{DAY}T15:05:00")

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "unknown"
        assert row["cash_diff"] is None

    def test_an_unreadable_ledger_baseline_is_unknown_not_empty(
        self, tmp_path, monkeypatch,
    ):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(store, cash=95_000.0)

        def boom(tenant_id="", engine_state_key=""):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(store, "get_latest_ledger_baseline", boom)

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "unknown", (
            "an unreadable expected source must not be reconciled as empty"
        )

    def test_an_unreadable_trade_ledger_is_unknown(self, tmp_path, monkeypatch):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(positions=[], cash=100_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(store, positions={}, cash=100_000.0)

        def boom(*_args, **_kwargs):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(store, "get_trades_for_export", boom)

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "unknown"

    def test_a_broker_position_without_a_code_is_unknown(self, tmp_path):
        """A malformed position must not be dropped (that would look flat)."""
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(quantity=100, current_price=50.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(store, cash=95_000.0)

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "unknown"


class TestTradeLedgerBasis:
    """R12: expected = startup baseline + stored fills, actual = broker."""

    def test_fills_since_the_baseline_move_the_expected_book(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        # Trade at 50, broker mark at 60: expected is valued at the broker's
        # price, so the price difference alone cannot fail the comparison.
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=120,
                                current_price=60.0, avg_cost=48.0)],
            cash=93_998.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )
        _save_trade(
            store, quantity=20, price=50.0, commission=2.0,
            executed_at=f"{DAY}T10:00:00",
        )

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "passed"
        detail = row["details"][0]
        assert detail["expected_shares"] == 120
        assert detail["actual_shares"] == 120
        assert detail["expected_price"] == pytest.approx(60.0)
        assert row["cash_diff"] == pytest.approx(0.0)
        assert events[-1].data["basis"] == "trade_ledger"

    def test_a_two_share_mismatch_fails_with_asset_details(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=102,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "failed"
        detail = {d["asset"]: d for d in row["details"]}["600519"]
        assert detail["expected_shares"] == 100
        assert detail["actual_shares"] == 102

    def test_a_one_yuan_cash_difference_is_within_tolerance(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(positions=[], cash=99_999.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(store, positions={}, cash=100_000.0)

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "passed"
        assert row["cash_diff"] == pytest.approx(1.0)

    def test_a_fill_before_the_baseline_is_excluded(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )
        _save_trade(
            store, quantity=500, price=10.0,
            executed_at=f"{DAY}T08:59:59",
        )

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "passed"

    def test_a_fill_at_the_baseline_instant_is_applied(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=110,
                                current_price=50.0, avg_cost=45.0)],
            cash=99_900.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=100_000.0,
            created_at=f"{DAY}T09:00:00",
        )
        _save_trade(
            store, quantity=10, price=10.0,
            executed_at=f"{DAY}T09:00:00",
        )

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "passed"

    def test_fills_from_another_tenant_are_excluded(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=100,
                                current_price=50.0, avg_cost=45.0)],
            cash=95_000.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
            tenant="default",
        )
        _save_trade(
            store, quantity=999, price=10.0, tenant="other",
            executed_at=f"{DAY}T10:00:00",
        )

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "passed"

    def test_an_unpriced_mismatch_cannot_be_absorbed_by_materiality(
        self, tmp_path, monkeypatch,
    ):
        """A vanished expected position with no broker mark and no baseline
        cost is worth 0; materiality must not turn that into a pass."""
        monkeypatch.setenv("QUANT_RECON_MATERIALITY", "1000000")
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(positions=[], cash=100_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        # Bare quantity: no avg_cost to value the missing leg.
        _seed_baseline(store, positions={"600519": 100}, cash=100_000.0)

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "failed"
        assert events[-1].data["reason"].startswith("unpriced mismatch")

    def test_the_engine_scoped_tenant_and_key_select_only_their_baseline(
        self, tmp_path,
    ):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=120,
                                current_price=50.0, avg_cost=45.0)],
            cash=94_000.0,
        )
        scheduler, store = _scheduler(
            tmp_path, clock, broker=broker,
            tenant_id="default", engine_state_key="live_engine:default",
        )
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0, tenant="default",
            engine_state_key="live_engine:default",
        )
        _save_trade(
            store, quantity=20, price=50.0,
            executed_at=f"{DAY}T10:00:00", tenant="default",
        )
        # Newer, mismatching row under another scope must stay invisible.
        _seed_baseline(
            store, positions={}, cash=1.0,
            tenant="other", engine_state_key="live_engine:other",
        )

        scheduler._tick()

        assert store.get_reconciliation_result(DAY)["status"] == "passed"

    def test_the_newest_baseline_is_used_and_older_fills_are_not_reapplied(
        self, tmp_path,
    ):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        broker = FakeBroker(
            positions=[Position(code="600519", quantity=150,
                                current_price=50.0, avg_cost=45.0)],
            cash=99_500.0,
        )
        scheduler, store = _scheduler(tmp_path, clock, broker=broker)
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=100_000.0, created_at=f"{DAY}T09:00:00",
        )
        # Recorded before the restart baseline and already baked into it: if
        # the older baseline were used, expected would be 600 shares and
        # 95,000 cash and the comparison would fail.
        _save_trade(
            store, quantity=500, price=10.0,
            executed_at=f"{DAY}T09:30:00",
        )
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 150, "avg_cost": 45.0}},
            cash=99_500.0, created_at=f"{DAY}T10:00:00",
        )

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "passed"
        assert row["details"][0]["expected_shares"] == 150


class TestNoLedgerBaseline:
    def test_no_baseline_is_unknown_never_a_pass(self, tmp_path):
        """Without a baseline the comparison cannot honestly run, even when
        the Store book happens to match the broker."""
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
        _seed_store_position(store)  # matches the broker; must not be used

        scheduler._tick()

        row = store.get_reconciliation_result(DAY)
        assert row["status"] == "unknown"
        assert row["cash_diff"] is None
        assert row["basis"] == "trade_ledger"
        assert events[-1].data["status"] == "unknown"
        assert events[-1].data["passes"] is False
        assert events[-1].data["reason"] == "no_ledger_baseline"


class TestBackfillAndHistoricalDays:
    def test_a_missing_result_is_backfilled_without_a_new_snapshot(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 10, tzinfo=TZ)]
        bus = EventBus()
        events = []
        bus.subscribe("eod.reconciliation", events.append)
        broker = FakeBroker(positions=[], cash=100_000.0)
        scheduler, store = _scheduler(tmp_path, clock, broker=broker, bus=bus)
        _seed_baseline(store, positions={}, cash=100_000.0)
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
        _seed_baseline(
            store,
            positions={"600519": {"quantity": 100, "avg_cost": 45.0}},
            cash=95_000.0,
        )

        scheduler._tick()

        tolerances = store.get_reconciliation_result(DAY)["tolerances"]
        assert tolerances == {
            "shares": 0.0, "cash": 1000.0, "cash_bps": 0.0,
            "materiality": 0.0,
        }
