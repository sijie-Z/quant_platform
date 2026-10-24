"""Ops wave 2 regressions — scheduler EOD timing, equity, catch-up, interval.

S1: EOD ran at 09:00 because `closed + TRADING` looked like a market close.
S2: EOD fabricated a 1,000,000 equity when no P&L snapshot existed.
S3: a stand-alone scheduler stayed in INIT and never backfilled missed days.
S20: rebalance state was dead and the interval was unvalidated.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from quant_platform.core.events import EventBus
from quant_platform.core.scheduler import TradingScheduler
from quant_platform.core.state_machine import PortfolioState, PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.utils.metrics import get_metrics

TZ = ZoneInfo("Asia/Shanghai")


def _scheduler(tmp_path, clock, *, name="sched.db", broker=None,
               engine_state_key="", bus=None):
    store = Store(str(tmp_path / name))
    sm = PortfolioStateMachine()
    scheduler = TradingScheduler(
        sm, store, bus or EventBus(), broker=broker,
        engine_state_key=engine_state_key,
    )
    scheduler._now = lambda: clock[0]
    return scheduler, store, sm


def _seed_engine_state(store, *, equity=250_000.0, cash=90_000.0,
                       positions=None, key="live_engine:default"):
    store.save_engine_state(
        {
            "portfolio_value": equity,
            "cash": cash,
            "positions": positions or [],
        },
        key,
    )


def _capture(bus, topic):
    events = []
    bus.subscribe(topic, lambda event: events.append(event))
    return events


def _trade_to(sm):
    sm.transition(PortfolioState.READY, "test")
    sm.transition(PortfolioState.TRADING, "test")


class TestPreMarketEod:
    """S1: a pre-market tick must not reconcile the day that has not begun."""

    def test_0900_trading_state_is_not_a_market_close(self, tmp_path):
        clock = [datetime(2026, 10, 20, 9, 0, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)
        _seed_engine_state(store)
        _trade_to(sm)

        scheduler._tick()

        assert sm.state == PortfolioState.TRADING
        assert store.has_eod_reconciliation("2026-10-20") is False
        assert store.get_pnl_history(days=1) == []

    def test_1501_tick_writes_exactly_one_snapshot(self, tmp_path):
        clock = [datetime(2026, 10, 20, 9, 0, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)
        _seed_engine_state(store)
        _trade_to(sm)

        scheduler._tick()  # 09:00 — nothing
        assert store.get_pnl_history(start_date="2026-10-20",
                                     end_date="2026-10-20") == []

        clock[0] = datetime(2026, 10, 20, 15, 1, tzinfo=TZ)
        scheduler._tick()
        scheduler._tick()  # idempotent marker

        rows = store.get_pnl_history(start_date="2026-10-20",
                                     end_date="2026-10-20")
        assert len(rows) == 1
        assert rows[0]["total_equity"] == pytest.approx(250_000.0)
        assert store.has_eod_reconciliation("2026-10-20") is True

    def test_a_weekend_never_reconciles(self, tmp_path):
        clock = [datetime(2026, 10, 17, 16, 0, tzinfo=TZ)]  # Saturday
        scheduler, store, sm = _scheduler(tmp_path, clock)
        _seed_engine_state(store)
        _trade_to(sm)

        scheduler._tick()

        assert store.has_eod_reconciliation("2026-10-17") is False


class TestEodEquity:
    """S2: never fabricate equity; use broker/engine state or skip."""

    def test_no_equity_source_skips_instead_of_inventing_one(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        alerts = _capture(bus, "system.alert")
        scheduler, store, sm = _scheduler(tmp_path, clock, bus=bus)
        # A 160,000 position alone used to yield total_equity=1,000,000.
        store.save_position({
            "code": "600519", "quantity": 16000, "available": 16000,
            "avg_cost": 10.0, "current_price": 10.0, "market_value": 160_000.0,
        })
        _trade_to(sm)
        before = get_metrics().get_snapshot().get(
            "scheduler_eod_skipped_total", {},
        ).get("value", 0)

        scheduler._tick()

        assert store.get_pnl_history(start_date="2026-10-20",
                                     end_date="2026-10-20") == []
        assert store.has_eod_reconciliation("2026-10-20") is False
        after = get_metrics().get_snapshot().get(
            "scheduler_eod_skipped_total", {},
        ).get("value", 0)
        assert after >= before + 1
        assert any(
            "no equity source" in event.data.get("message", "")
            for event in alerts
        )

    def test_persisted_engine_state_supplies_equity_and_cash(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(
            tmp_path, clock, engine_state_key="live_engine:default",
        )
        _seed_engine_state(
            store, equity=250_000.0, cash=90_000.0,
            positions=[{"code": "600519", "market_value": 160_000.0}],
        )
        _trade_to(sm)

        scheduler._tick()

        rows = store.get_pnl_history(start_date="2026-10-20",
                                     end_date="2026-10-20")
        assert len(rows) == 1
        assert rows[0]["total_equity"] == pytest.approx(250_000.0)
        assert rows[0]["cash"] == pytest.approx(90_000.0)
        assert rows[0]["market_value"] == pytest.approx(160_000.0)

    def test_broker_account_is_preferred_when_attached(self, tmp_path):
        class Broker:
            def get_account(self):
                return {"total_equity": 777_000.0, "cash": 777_000.0}

            def get_positions(self):
                return []

        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock, broker=Broker())
        _seed_engine_state(store, equity=1.0)
        _trade_to(sm)

        scheduler._tick()

        rows = store.get_pnl_history(start_date="2026-10-20",
                                     end_date="2026-10-20")
        assert len(rows) == 1
        assert rows[0]["total_equity"] == pytest.approx(777_000.0)

    def test_a_disconnected_broker_placeholder_is_not_equity(self, tmp_path):
        """QMT returns initial_cash as a placeholder when disconnected."""
        class DisconnectedBroker:
            def get_account(self):
                return {
                    "connected": False, "cash": 1_000_000.0,
                    "total_equity": 1_000_000.0,
                }

            def get_positions(self):
                return []

        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(
            tmp_path, clock, broker=DisconnectedBroker(),
        )
        _trade_to(sm)

        scheduler._tick()

        assert store.get_pnl_history(
            start_date="2026-10-20", end_date="2026-10-20",
        ) == []
        assert store.has_eod_reconciliation("2026-10-20") is False

    def test_a_risk_state_row_is_never_used_as_engine_equity(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)
        _seed_engine_state(
            store, equity=250_000.0, cash=90_000.0,
            positions=[{"code": "600519", "market_value": 160_000.0}],
            key="live_engine:default",
        )
        # Newer row from the risk monitor: not an engine book.
        store.save_engine_state(
            {
                "portfolio_value": 999_999.0, "kill_switch_active": False,
                "breaches": [],
            },
            "risk_state:default",
        )
        _trade_to(sm)

        scheduler._tick()

        rows = store.get_pnl_history(
            start_date="2026-10-20", end_date="2026-10-20",
        )
        assert len(rows) == 1
        assert rows[0]["total_equity"] == pytest.approx(250_000.0)

    def test_tenant_scoped_lookup_ignores_other_tenants(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)
        scheduler._tenant_id = "acme"
        store.save_engine_state(
            {
                "tenant_id": "other", "portfolio_value": 111_111.0,
                "cash": 111_111.0, "positions": [],
            },
            "live_engine:other",
        )
        _trade_to(sm)

        scheduler._tick()

        # No engine book for 'acme': never borrow another tenant's equity.
        assert store.get_pnl_history(
            start_date="2026-10-20", end_date="2026-10-20",
        ) == []

    def test_no_prior_pnl_does_not_mean_one_million(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)
        store.save_position({
            "code": "600519", "quantity": 16000, "available": 16000,
            "avg_cost": 10.0, "current_price": 10.0, "market_value": 160_000.0,
        })
        _trade_to(sm)

        scheduler._tick()

        rows = store.get_pnl_history(start_date="2026-10-20",
                                     end_date="2026-10-20")
        assert rows == []
        assert not any(row["total_equity"] == 1_000_000 for row in rows)


class TestStartupCatchUp:
    """S3: bootstrap INIT -> READY and reconcile days missed while down."""

    def test_fresh_scheduler_at_1000_transitions_to_a_live_state(self, tmp_path):
        clock = [datetime(2026, 10, 20, 10, 0, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)

        scheduler._tick()

        assert sm.state == PortfolioState.TRADING
        assert scheduler.is_market_open() is True

    def test_start_after_the_close_backfills_the_missing_day(self, tmp_path):
        clock = [datetime(2026, 10, 20, 16, 0, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)
        # Friday 2026-10-16 was reconciled; Monday 2026-10-19 and Tuesday
        # 2026-10-20 are missing.
        store.mark_eod_reconciliation("2026-10-16", "2026-10-16T15:05:00")
        _seed_engine_state(store, equity=300_000.0, cash=300_000.0)

        scheduler.start()
        try:
            assert store.has_eod_reconciliation("2026-10-20") is True
            assert store.has_eod_reconciliation("2026-10-19") is True
            rows = store.get_pnl_history(start_date="2026-10-19",
                                         end_date="2026-10-20")
            # Today's close is a real observation and gets a snapshot.
            # Monday is historical: its marker is written without inventing
            # an equity value for a day we did not observe.
            assert [row["timestamp"] for row in rows] == [
                "2026-10-20T15:00:00",
            ]
            assert rows[0]["total_equity"] == pytest.approx(300_000.0)
        finally:
            scheduler.stop()

    def test_historical_catch_up_never_invents_equity(self, tmp_path):
        """A missed past day gets a marker and an alert, not today's equity."""
        clock = [datetime(2026, 10, 21, 10, 0, tzinfo=TZ)]
        bus = EventBus()
        alerts = _capture(bus, "system.alert")
        scheduler, store, sm = _scheduler(tmp_path, clock, bus=bus)
        store.mark_eod_reconciliation("2026-10-16", "2026-10-16T15:05:00")
        _seed_engine_state(store, equity=300_000.0, cash=300_000.0)

        scheduler.start()
        try:
            assert store.has_eod_reconciliation("2026-10-19") is True
            assert store.has_eod_reconciliation("2026-10-20") is True
            assert store.get_pnl_history(
                start_date="2026-10-19", end_date="2026-10-20",
            ) == []
            assert any(
                "without inventing equity" in event.data.get("message", "")
                for event in alerts
            )
        finally:
            scheduler.stop()

    def test_periodic_catch_up_backfills_after_midnight(self, tmp_path):
        """A day missed while the process stayed up is picked up by the
        hourly scan, not only at startup."""
        clock = [datetime(2026, 10, 22, 0, 30, tzinfo=TZ)]  # Thursday 00:30
        scheduler, store, sm = _scheduler(tmp_path, clock)
        store.mark_eod_reconciliation("2026-10-20", "2026-10-20T15:05:00")

        scheduler._tick()  # outside the EOD window; periodic catch-up runs

        assert store.has_eod_reconciliation("2026-10-21") is True
        assert store.get_pnl_history(
            start_date="2026-10-21", end_date="2026-10-21",
        ) == []

    def test_halted_state_still_reconciles_and_alerts(self, tmp_path):
        clock = [datetime(2026, 10, 20, 15, 5, tzinfo=TZ)]
        bus = EventBus()
        alerts = _capture(bus, "system.alert")
        scheduler, store, sm = _scheduler(tmp_path, clock, bus=bus)
        _seed_engine_state(store)
        sm.transition(PortfolioState.READY, "test")
        sm.transition(PortfolioState.TRADING, "test")
        sm.transition(PortfolioState.HALTED, "test halt")

        scheduler._tick()

        assert store.has_eod_reconciliation("2026-10-20") is True
        assert any(
            "halted" in event.data.get("message", "").lower()
            for event in alerts
        )

    def test_a_fresh_database_has_nothing_to_backfill(self, tmp_path):
        clock = [datetime(2026, 10, 20, 16, 0, tzinfo=TZ)]
        scheduler, store, sm = _scheduler(tmp_path, clock)

        assert scheduler._catch_up_missed_eod() == 0
        assert store.get_pnl_history(start_date="2026-01-01") == []


class TestIntervalAndRebalance:
    """S20: interval validation, real rebalance state, session reuse."""

    @pytest.mark.parametrize("bad", [0, -1, 1, 4, "x", None])
    def test_constructor_rejects_an_unsafe_interval(self, tmp_path, bad):
        store = Store(str(tmp_path / "bad.db"))
        with pytest.raises(ValueError, match="rebalance_interval"):
            TradingScheduler(PortfolioStateMachine(), store, EventBus(),
                             rebalance_interval=bad)

    def test_five_seconds_is_accepted(self, tmp_path):
        store = Store(str(tmp_path / "ok.db"))
        scheduler = TradingScheduler(
            PortfolioStateMachine(), store, EventBus(), rebalance_interval=5,
        )
        assert scheduler.get_state()["rebalance_interval"] == 5

    def test_rebalance_state_is_wired_and_throttled(self, tmp_path):
        clock = [datetime(2026, 10, 20, 10, 0, tzinfo=TZ)]
        bus = EventBus()
        events = _capture(bus, "scheduler.rebalance")
        scheduler, _, sm = _scheduler(tmp_path, clock, bus=bus)

        scheduler._tick()
        scheduler._tick()

        state = scheduler.get_state()
        assert state["rebalance_count"] == 1
        assert state["last_rebalance"] > 0
        assert state["last_rebalance_at"]
        assert len(events) == 1

    def test_the_first_alert_is_not_swallowed_on_a_fresh_boot(self, tmp_path, monkeypatch):
        """time.monotonic() is time-since-boot; 0.0 as the "never sent"
        sentinel suppressed the first alert on a fresh runner/container."""
        clock = [datetime(2026, 10, 20, 10, 0, tzinfo=TZ)]
        bus = EventBus()
        events = _capture(bus, "system.alert")
        scheduler, _, _ = _scheduler(tmp_path, clock, bus=bus)
        monkeypatch.setattr(
            "quant_platform.core.scheduler.time.monotonic", lambda: 5.0,
        )

        assert scheduler._publish_alert("critical", "fresh boot alert") is True
        assert len(events) == 1

    def test_the_engine_session_id_is_reused(self, tmp_path):
        clock = [datetime(2026, 10, 20, 10, 0, tzinfo=TZ)]
        scheduler, store, _ = _scheduler(tmp_path, clock)
        store.save_session({
            "session_id": "engine-session-1",
            "broker": "SimulatedBroker",
            "status": "active",
            "started_at": "2026-10-20T09:55:00",
        })

        scheduler.start(session_id="engine-session-1")
        try:
            sessions = store.get_sessions()
            assert [s["session_id"] for s in sessions] == ["engine-session-1"]
            assert sessions[0]["broker"] == "SimulatedBroker"
            assert scheduler.get_state()["session_id"] == "engine-session-1"
        finally:
            scheduler.stop()
        assert len(store.get_sessions()) == 1

    def test_start_refuses_while_the_previous_loop_is_still_alive(self, tmp_path):
        """C4: stop() timing out used to let start() resurrect the old thread
        (shared Event.clear()), producing two scheduler loops."""
        clock = [datetime(2026, 10, 20, 10, 0, tzinfo=TZ)]
        scheduler, _, _ = _scheduler(tmp_path, clock)
        scheduler._stop_join_timeout = 0.1
        release = threading.Event()
        scheduler._tick = lambda: release.wait(timeout=5)

        scheduler.start()
        try:
            scheduler.stop()  # times out while the tick is blocked
            with pytest.raises(RuntimeError, match="previous loop thread"):
                scheduler.start()
        finally:
            release.set()
            deadline = time.monotonic() + 5
            while (
                scheduler._thread is not None
                and scheduler._thread.is_alive()
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            scheduler.stop()

        assert scheduler._thread is None or not scheduler._thread.is_alive()

    def test_tick_errors_are_counted_and_alerted_once(self, tmp_path):
        clock = [datetime(2026, 10, 20, 10, 0, tzinfo=TZ)]
        bus = EventBus()
        alerts = _capture(bus, "system.alert")
        scheduler, _, _ = _scheduler(tmp_path, clock, bus=bus)
        metrics = get_metrics()
        before = metrics.get_snapshot().get(
            "scheduler_tick_errors_total", {},
        ).get("value", 0)

        scheduler._handle_tick_error(RuntimeError("boom"))
        scheduler._handle_tick_error(RuntimeError("boom"))

        after = metrics.get_snapshot().get(
            "scheduler_tick_errors_total", {},
        ).get("value", 0)
        assert after >= before + 2
        tick_alerts = [
            event for event in alerts
            if "scheduler tick failed" in event.data.get("message", "")
        ]
        assert len(tick_alerts) == 1, "identical tick alerts must be rate-limited"
