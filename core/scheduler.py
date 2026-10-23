"""Trading session scheduler — orchestrates the trading day.

Manages:
- Market hours detection (A-share: 9:30-11:30, 13:00-15:00)
- Pre-market preparation (load data, compute signals)
- Trading session lifecycle
- Post-market reconciliation (save P&L, reconcile positions)
- Rebalance timing
- Startup catch-up for EOD reconciliation missed during downtime (S3)

Integrates with EventBus and StateMachine.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.core.exchange_calendar import is_trading_day
from quant_platform.core.state_machine import PortfolioState, PortfolioStateMachine
from quant_platform.core.store import RetentionPolicy, Store
from quant_platform.operations.reconciliation import (
    expected_book_from_ledger,
    ledger_from_trades,
    positions_from_broker,
    reconcile,
)
from quant_platform.utils.logging import get_logger
from quant_platform.utils.metrics import get_metrics

logger = get_logger(__name__)

DEFAULT_EXCHANGE_TZ = "Asia/Shanghai"

#: A zero/negative interval made ``Event.wait(timeout=...)`` return
#: immediately, turning the scheduler/engine loops into a tight loop that
#: hammered the broker (S20). Five seconds is the documented floor.
MIN_REBALANCE_INTERVAL = 5

#: Exchange wall-clock close. EOD reconciliation must never run before this
#: on a trading day (S1): a 09:00 engine start used to look like "market
#: closed while TRADING" and wrote the day's P&L snapshot before the open.
EOD_HOUR = 15
EOD_MINUTE = 0

#: Bound on startup EOD backfill (S3).
MAX_EOD_CATCHUP_DAYS = 10

#: Minimum spacing between identical scheduler alerts (seconds). The alert
#: sink deduplicates for five minutes as well; this only keeps the EventBus
#: history from filling with one event per tick.
ALERT_THROTTLE_SECONDS = 300.0

#: R11: tolerances for the daily books-vs-broker reconciliation, as
#: ``(environment variable, default)``. The defaults mirror ``reconcile()``.
RECON_TOLERANCE_ENV = {
    "shares": ("QUANT_RECON_SHARES_TOLERANCE", 0.0),
    "cash": ("QUANT_RECON_CASH_TOLERANCE", 1000.0),
    "cash_bps": ("QUANT_RECON_CASH_TOLERANCE_BPS", 0.0),
    "materiality": ("QUANT_RECON_MATERIALITY", 0.0),
}


def _reconciliation_tolerances_from_env() -> dict:
    """Read the ``QUANT_RECON_*`` tolerances; invalid values raise ValueError.

    An unset or empty variable takes the default. Anything that is not a
    finite, non-negative number is rejected (fail loud): a typo must not
    silently widen or narrow the daily reconciliation.
    """
    values: dict[str, float] = {}
    for key, (env, default) in RECON_TOLERANCE_ENV.items():
        raw = os.environ.get(env)
        if raw is None or not raw.strip():
            values[key] = default
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{env} must be a number, got {raw!r}"
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{env} must be a finite non-negative number, got {raw!r}"
            )
        values[key] = value
    return values


@dataclass
class MarketSession:
    """A-share market session info."""
    date: str = ""
    is_trading_day: bool = True
    market_open: str = "09:30"
    market_close: str = "15:00"
    lunch_start: str = "11:30"
    lunch_end: str = "13:00"
    pre_market: str = "09:15"
    post_market: str = "15:30"
    status: str = "closed"  # pre_market, trading_am, lunch, trading_pm, post_market, closed


class TradingScheduler:
    """Orchestrates trading sessions with market-aware timing.

    Responsibilities:
    - Detect market hours and trading days
    - Trigger pre-market / trading / post-market state transitions
    - Schedule rebalance cycles at configured intervals
    - Handle weekends and holidays
    - Publish timing events on the bus
    """

    def __init__(
        self,
        state_machine: PortfolioStateMachine,
        store: Store,
        bus: EventBus | None = None,
        rebalance_interval: int = 300,  # seconds
        exchange_tz: str = DEFAULT_EXCHANGE_TZ,
        broker: object | None = None,
        engine_state_key: str = "",
        tenant_id: str = "",
        retention_policy: RetentionPolicy | None = None,
        retention_archive_dir: str = "data/archive",
    ):
        # S20: an interval below the floor used to make the wait return
        # immediately; validate in the constructor so every entry point
        # (engine, API, CLI, embedded) inherits the guard.
        try:
            interval = int(rebalance_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"rebalance_interval must be an integer number of seconds, "
                f"got {rebalance_interval!r}"
            ) from exc
        if interval < MIN_REBALANCE_INTERVAL:
            raise ValueError(
                f"rebalance_interval must be >= {MIN_REBALANCE_INTERVAL}s, "
                f"got {interval}"
            )
        self._sm = state_machine
        self._store = store
        self._bus = bus or get_event_bus()
        self._rebalance_interval = interval
        # Market hours are Shanghai wall-clock times. `datetime.now()` on a
        # UTC container (the shipped Docker image) put the session seven
        # hours off: "closed" during the real session and "trading_am" at
        # 17:30 Beijing.
        self._tz = ZoneInfo(exchange_tz)

        # S2: where the EOD reconciliation may read equity from. The engine
        # passes its broker and its tenant-scoped engine_state key; a
        # standalone scheduler falls back to the latest persisted snapshot.
        self._broker = broker
        self._engine_state_key = engine_state_key or ""
        self._tenant_id = tenant_id or ""

        # S10: daily retention hook after EOD.
        self._retention_policy = (
            retention_policy if retention_policy is not None
            else RetentionPolicy.from_env()
        )
        self._retention_archive_dir = retention_archive_dir

        # R11: tolerances for the daily books-vs-broker reconciliation.
        # Parsed at construction so a typo in QUANT_RECON_* fails the
        # scheduler loudly instead of silently reconciling with defaults.
        self._recon_tolerances = _reconciliation_tolerances_from_env()

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._session_id = ""
        self._owns_session = True
        self._last_rebalance = 0.0
        self._last_rebalance_at = ""
        self._rebalance_count = 0
        self._cycle_count = 0
        self._last_tick_at = 0.0
        # -inf, not 0.0: a fresh container's monotonic clock can be below the
        # 60 s rate limit, which would swallow the first tick-failure alert.
        self._last_tick_alert = float("-inf")
        self._alert_throttle: dict[str, float] = {}
        # How long stop() waits for the tick loop before reporting it stuck.
        self._stop_join_timeout = 10.0
        # Periodic (hourly) scan for missed EOD markers; startup catch-up
        # handles restarts, this handles a process that stays up across a
        # missed window (host sleep, no equity source all evening).
        # -inf, not 0.0: time.monotonic() is time-since-boot on Windows and
        # can legitimately be below 3600.
        self._catchup_interval_seconds = 3600.0
        self._last_catchup_at = float("-inf")

    def _now(self) -> datetime:
        return datetime.now(self._tz)

    @staticmethod
    def _naive_local(now: datetime) -> str:
        """ISO timestamp in exchange local time, naive like the rest of the store."""
        return now.replace(tzinfo=None).isoformat()

    # ── Heartbeat / alerts ──

    def _record_tick_heartbeat(self) -> None:
        self._last_tick_at = time.time()
        try:
            get_metrics().gauge(
                "last_scheduler_tick_timestamp",
                help="Epoch seconds of the last completed scheduler tick",
            ).set(self._last_tick_at)
        except Exception:  # metrics must never break scheduling
            pass

    def _publish_alert(self, severity: str, message: str,
                       details: dict | None = None) -> bool:
        """Publish a throttled operational alert for the sink handler.

        Returns True when the alert was emitted (callers can keep noisy log
        lines at the same rate instead of once per tick).
        """
        key = f"{severity}|{message}"
        now = time.monotonic()
        # Missing key means "never emitted". Using 0.0 as the sentinel
        # suppressed the first alert on a freshly booted host or container,
        # where time.monotonic() can legitimately be below the throttle.
        last = self._alert_throttle.get(key)
        if last is not None and now - last < ALERT_THROTTLE_SECONDS:
            return False
        self._alert_throttle[key] = now
        if severity in ("critical", "error"):
            logger.error("Scheduler alert [%s]: %s", severity, message)
        self._bus.publish("system.alert", {
            "severity": severity,
            "message": message,
            "source": "scheduler",
            **(details or {}),
        }, source="scheduler")
        return True

    # ── Lifecycle ──

    def start(self, session_id: str | None = None):
        """Start the scheduler.

        ``session_id`` lets the live engine reuse its own session record
        instead of creating a second row for the same run (S20). A stand-alone
        scheduler creates and owns its session.
        """
        if self._running:
            return

        # D14 for the scheduler: never clear an Event a live thread may be
        # waiting on. A tick can exceed the stop join timeout (EOD retention
        # runs inside it), and the old `clear()` resurrected that thread,
        # producing two scheduler loops after a restart.
        previous = self._thread
        if previous is not None and previous.is_alive():
            previous.join(timeout=1.0)
            if previous.is_alive():
                message = (
                    "Scheduler start refused: previous loop thread is still "
                    "alive; stop it and wait for the tick to quiesce"
                )
                logger.critical(message)
                self._bus.publish("system.alert", {
                    "severity": "critical", "message": message,
                    "source": "scheduler",
                }, source="scheduler")
                raise RuntimeError(message)
        self._thread = None
        self._stop_event = threading.Event()

        self._running = True
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._owns_session = not session_id

        if self._owns_session:
            try:
                self._store.save_session({
                    "session_id": self._session_id,
                    "status": "active",
                    "started_at": self._naive_local(self._now()),
                })
            except Exception as exc:
                # A session row is observability, not a precondition for
                # scheduling; the engine's own session write is separate.
                logger.error("Scheduler session record failed: %s", exc)

        # S3: bootstrap INIT -> READY and reconcile EOD days missed while the
        # process was down, before the tick loop starts.
        try:
            self._bootstrap_and_catch_up()
        except Exception as exc:
            logger.error("Scheduler startup catch-up failed: %s", exc, exc_info=True)
            self._publish_alert(
                "error", f"scheduler startup catch-up failed: {exc}",
            )
        self._last_catchup_at = time.monotonic()

        self._record_tick_heartbeat()
        try:
            self._thread = threading.Thread(
                target=self._run_loop, args=(self._stop_event,), daemon=True,
            )
            self._thread.start()
        except Exception as exc:
            # Do not leave `_running=True` without a loop: retries would
            # return early and get_state would claim a live scheduler.
            self._running = False
            self._thread = None
            if self._owns_session and self._session_id:
                try:
                    self._store.save_session({
                        "session_id": self._session_id,
                        "status": "failed",
                        "stopped_at": self._naive_local(self._now()),
                    })
                except Exception:
                    pass
            self._publish_alert(
                "critical", f"scheduler thread start failed: {exc}",
            )
            raise

        self._bus.publish("scheduler.started", {
            "session_id": self._session_id,
        }, source="scheduler")

        logger.info("Scheduler started. Session: %s", self._session_id)

    def stop(self):
        """Stop the scheduler (best-effort; never raises)."""
        self._running = False
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        thread = self._thread
        if thread is not None:
            try:
                thread.join(timeout=self._stop_join_timeout)
            except Exception as exc:  # extremely defensive: join can raise
                logger.error("Scheduler thread join failed: %s", exc)
            if thread.is_alive():
                # Keep `_thread` so a later start() refuses instead of
                # running two loops against one store/state machine.
                message = (
                    f"Scheduler loop thread still alive "
                    f"{self._stop_join_timeout:.0f}s after stop request"
                )
                logger.critical(message)
                self._publish_alert(
                    "critical", message,
                    {"session_id": self._session_id},
                )

        if self._owns_session and self._session_id:
            try:
                self._store.save_session({
                    "session_id": self._session_id,
                    "status": "stopped",
                    "stopped_at": self._naive_local(self._now()),
                })
            except Exception as exc:
                logger.error("Scheduler session stop record failed: %s", exc)

        self._bus.publish("scheduler.stopped", {
            "session_id": self._session_id,
            "cycles": self._cycle_count,
        }, source="scheduler")

        logger.info("Scheduler stopped. Cycles: %d", self._cycle_count)

    def _bootstrap_and_catch_up(self) -> int:
        """INIT -> READY and backfill EOD days missed during downtime.

        A standalone scheduler used to stay in INIT forever (no tick branch
        handled INIT), and a scheduler started after the close never ran EOD
        for the day it missed (S3).
        """
        if self._sm.state == PortfolioState.INIT:
            self._sm.transition(PortfolioState.READY, "scheduler bootstrap")

        return self._catch_up_missed_eod()

    def get_market_session(self, now: datetime | None = None) -> MarketSession:
        """Get current market session info.

        `now` is an injection point for tests; production uses the exchange
        timezone, not the host clock.
        """
        now = now or self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=self._tz)
        else:
            now = now.astimezone(self._tz)

        t = now.hour * 100 + now.minute

        session = MarketSession(date=now.strftime("%Y-%m-%d"))

        # Weekends and exchange holidays are both non-trading days. The
        # holiday table lives in core/exchange_calendar.py (extendable via
        # QUANT_EXCHANGE_HOLIDAYS).
        if not is_trading_day(now.date()):
            session.is_trading_day = False
            session.status = (
                "closed_weekend" if now.weekday() >= 5 else "closed_holiday"
            )
            return session

        if t < 915:
            session.status = "closed"
        elif 915 <= t < 930:
            session.status = "pre_market"
        elif 930 <= t < 1130:
            session.status = "trading_am"
        elif 1130 <= t < 1300:
            session.status = "lunch"
        elif 1300 <= t < 1500:
            # The closing call auction (14:57-15:00) is part of the trading
            # session. The old 14:57 cutoff declared the market closed three
            # minutes before the close and ran EOD reconciliation there.
            session.status = "trading_pm"
        elif 1500 <= t < 1530:
            # Post-market (and therefore EOD reconciliation) starts only
            # after the 15:00 close.
            session.status = "post_market"
        else:
            session.status = "closed"

        return session

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        s = self.get_market_session()
        return s.status in ("trading_am", "trading_pm")

    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance."""
        if not self.is_market_open():
            return False
        return time.time() - self._last_rebalance >= self._rebalance_interval

    def get_state(self) -> dict:
        """Get scheduler state."""
        ms = self.get_market_session()
        return {
            "running": self._running,
            "session_id": self._session_id,
            "market_status": ms.status,
            "is_trading_day": ms.is_trading_day,
            "is_market_open": self.is_market_open(),
            "cycle_count": self._cycle_count,
            "last_rebalance": self._last_rebalance,
            "last_rebalance_at": self._last_rebalance_at,
            "rebalance_count": self._rebalance_count,
            "rebalance_interval": self._rebalance_interval,
            "last_tick_at": self._last_tick_at,
        }

    def _run_loop(self, stop_event: threading.Event | None = None):
        """Main scheduler loop (per-run event; D14-style)."""
        event = stop_event if stop_event is not None else self._stop_event
        if event is None:
            event = threading.Event()
        while not event.is_set():
            try:
                self._tick()
            except Exception as e:
                self._handle_tick_error(e)

            event.wait(timeout=30)  # Check every 30 seconds

    def _handle_tick_error(self, exc: Exception) -> None:
        """Count and (rate-limited) alert a failed tick (S20).

        A failing tick used to be a log line only, invisible to monitoring.
        """
        logger.error("Scheduler tick error: %s", exc, exc_info=True)
        try:
            get_metrics().counter(
                "scheduler_tick_errors_total",
                help="Scheduler ticks that raised an exception",
            ).inc()
        except Exception:
            pass
        now = time.monotonic()
        if now - self._last_tick_alert >= 60.0:
            self._last_tick_alert = now
            # Dedupe by exception type: varying text (ids/addresses) must not
            # defeat the sink's five-minute dedupe and POST every tick.
            self._publish_alert(
                "error", f"scheduler tick failed: {type(exc).__name__}",
                {"error": str(exc)},
            )

    def _tick(self):
        """Single scheduler tick — check market state and trigger transitions."""
        self._cycle_count += 1
        self._record_tick_heartbeat()
        now = self._now()
        ms = self.get_market_session(now)

        # S3: a stand-alone scheduler (and one embedded in an engine whose
        # start() was aborted before its own transition) must leave INIT.
        if self._sm.state == PortfolioState.INIT:
            self._sm.transition(PortfolioState.READY, "scheduler bootstrap")

        # State transitions based on market hours
        if ms.status == "pre_market" and self._sm.can_transition(PortfolioState.PRE_MARKET):
            self._sm.transition(PortfolioState.PRE_MARKET, "market pre-open")
            self._bus.publish("market.pre_open", {"date": ms.date}, source="scheduler")

        elif ms.status in ("trading_am", "trading_pm"):
            if self._sm.state != PortfolioState.TRADING and self._sm.state != PortfolioState.REBALANCING:
                if self._sm.can_transition(PortfolioState.TRADING):
                    self._sm.transition(PortfolioState.TRADING, "market open")
            self._maybe_rebalance()

        elif ms.status in ("post_market", "closed") and self._in_eod_window(ms, now):
            self._handle_eod_window()

        # Once an hour, look for past trading days whose EOD marker is missing
        # (today is left to the EOD window above).
        self._maybe_catch_up()

    def _maybe_catch_up(self) -> None:
        now = time.monotonic()
        if now - self._last_catchup_at < self._catchup_interval_seconds:
            return
        self._last_catchup_at = now
        try:
            self._catch_up_missed_eod(include_today=False)
        except Exception as exc:
            logger.error("Periodic EOD catch-up failed: %s", exc, exc_info=True)

    @staticmethod
    def _in_eod_window(ms: MarketSession, now: datetime) -> bool:
        """EOD runs only at/after 15:00 on an exchange trading day (S1).

        The old condition keyed on status in ("post_market", "closed"), which
        includes every minute before 09:15: a pre-market engine start looked
        like a close and reconciled the day before it began.
        """
        if not ms.is_trading_day:
            return False
        return (now.hour, now.minute) >= (EOD_HOUR, EOD_MINUTE)

    def _handle_eod_window(self) -> None:
        state = self._sm.state
        if state in (PortfolioState.TRADING, PortfolioState.REBALANCING):
            if self._sm.transition(PortfolioState.POST_MARKET, "market closed"):
                self._do_eod_reconciliation()
        elif state == PortfolioState.POST_MARKET:
            # Idempotent: the marker suppresses a second snapshot.
            self._do_eod_reconciliation()
            self._sm.transition(PortfolioState.READY, "post-market complete")
        elif state in (PortfolioState.HALTED, PortfolioState.ERROR):
            # S3: HALTED/ERROR used to skip reconciliation silently. Run it
            # (the day still happened) and alert so the missed state is
            # visible to an operator.
            self._do_eod_reconciliation()
            self._publish_alert(
                "error",
                f"EOD reconciliation ran while portfolio state is {state.value}",
                {"state": state.value},
            )
        else:
            # READY/PRE_MARKET: a stand-alone scheduler, or a scheduler whose
            # engine was started after the close. There is no TRADING
            # transition to hang EOD from, so reconcile directly.
            self._do_eod_reconciliation()

    def _maybe_rebalance(self) -> None:
        """Drive `should_rebalance()` and expose a real last-rebalance stamp.

        The method existed but had no caller and `_last_rebalance` was never
        updated, so get_state() reported zero forever (S20).
        """
        if not self.should_rebalance():
            return
        self._last_rebalance = time.time()
        self._last_rebalance_at = self._naive_local(self._now())
        self._rebalance_count += 1
        self._bus.publish("scheduler.rebalance", {
            "session_id": self._session_id,
            "interval": self._rebalance_interval,
            "count": self._rebalance_count,
        }, source="scheduler")

    # ── EOD reconciliation (S1/S2/S3) ──

    def _do_eod_reconciliation(
        self, date_str: str | None = None, catch_up: bool = False,
    ) -> bool:
        """End-of-day reconciliation: save a P&L snapshot, log events.

        Runs at most once per exchange-local date. Without a persisted
        marker a scheduler restarted after the close wrote a second snapshot
        for the same day, double-counting it in P&L history.

        Equity comes from the broker account or the engine state persisted by
        the live engine (S2). The old hard-coded 1,000,000 fabricated a
        snapshot for any book with no prior P&L; when neither source is
        available the day is skipped, alerted and counted instead.

        R11/R12: once the snapshot is durable, the expected book derived from
        the engine-start baseline plus stored fills (trade ledger) is
        reconciled against the broker's positions and account cash (actual).
        The outcome is persisted to ``reconciliation_results`` and added to
        the ``eod.reconciliation`` event. A repeated trigger never writes a
        second result; a day whose marker exists but whose result is missing
        (an upgraded database) has only the result backfilled — never a
        second snapshot.
        """
        target = date_str or self._now().strftime("%Y-%m-%d")
        try:
            marked = self._store.has_eod_reconciliation(target)
        except Exception as exc:
            logger.error("EOD marker read failed for %s: %s", target, exc)
            return False

        if marked:
            if not catch_up:
                logger.info("EOD reconciliation already done for %s; skipping", target)
            return self._backfill_missing_reconciliation(target)

        # S2/S3: a missed *past* day has no observation we can honestly
        # reconstruct. Writing today's equity under a historical date would
        # fabricate the missed day's P&L (the failure S2 removed). Mark the
        # day reconciled (so it is not retried forever) without a snapshot,
        # and alert the operator about the history gap.
        if catch_up and target != self._now().strftime("%Y-%m-%d"):
            return self._backfill_historical_eod(target)

        equity, cash, positions, source = self._resolve_eod_values()
        if equity is None:
            self._skip_eod(target)
            return False

        market_value = sum(self._position_market_value(p) for p in positions)
        if cash is None:
            cash = max(0.0, equity - market_value)

        timestamp = (
            f"{target}T15:00:00" if catch_up
            else self._naive_local(self._now())
        )
        snapshot = {
            "timestamp": timestamp,
            "total_equity": equity,
            "cash": cash,
            "market_value": market_value,
            "n_positions": len(positions),
        }
        try:
            # Snapshot + marker share one transaction: a crash between two
            # writes would otherwise leave an unmarked snapshot and double-
            # count the day on the next run. A racing process that already
            # claimed the day makes this return False (no snapshot).
            written = self._store.save_pnl_and_mark_eod(
                snapshot, target, timestamp,
            )
        except Exception as exc:
            logger.critical(
                "EOD reconciliation failed for %s; no marker written: %s",
                target, exc,
            )
            self._publish_alert(
                "critical", f"EOD reconciliation failed for {target}: {exc}",
            )
            return False

        if not written:
            logger.info(
                "EOD reconciliation for %s was already claimed by another "
                "run; no duplicate snapshot written", target,
            )
            return False

        # R11: reconcile the persisted book against the broker counter-book
        # (persisted to reconciliation_results), then publish one enriched
        # eod.reconciliation event. The snapshot stays durable even when the
        # reconciliation cannot run.
        recon = self._run_eod_reconciliation(target, snapshot=snapshot)
        self._publish_eod_reconciliation(
            target, len(positions), market_value, equity, source, catch_up,
            recon,
        )

        logger.info(
            "EOD reconciliation %s: %d positions, equity=%.2f (source=%s), "
            "books-vs-broker=%s",
            target, len(positions), equity, source, recon["status"],
        )

        if target == self._now().strftime("%Y-%m-%d"):
            # Daily retention after today's EOD, whether it was written by
            # the normal 15:00 tick or by the startup catch-up.
            self._run_retention_hook()
        return True

    def _backfill_historical_eod(self, target: str) -> bool:
        """Reconcile a missed historical day without inventing its equity.

        If the engine wrote snapshots for that date (it was running but the
        scheduler missed the window), the existing rows are the observation:
        only the marker is written. Otherwise the day is marked as reconciled
        with no P&L row and an alert records the gap, so drawdown/VaR never
        see a fabricated flat day.
        """
        rows: list[dict] = []
        try:
            rows = self._store.get_pnl_history(
                start_date=target, end_date=target,
            )
        except Exception as exc:
            logger.error("EOD catch-up snapshot read failed for %s: %s", target, exc)
            rows = []

        marker_ts = (
            str(rows[-1].get("timestamp") or f"{target}T15:00:00")
            if rows else f"{target}T15:00:00"
        )
        try:
            self._store.mark_eod_reconciliation(target, marker_ts)
        except Exception as exc:
            logger.critical(
                "EOD catch-up marker write failed for %s: %s", target, exc,
            )
            self._publish_alert(
                "critical", f"EOD catch-up marker write failed for {target}: {exc}",
                {"date": target},
            )
            return False

        if rows:
            logger.warning(
                "EOD catch-up %s: marker written using %d existing snapshot(s)",
                target, len(rows),
            )
        else:
            message = (
                f"EOD catch-up for {target}: no snapshot exists for that day; "
                "marker written without inventing equity (P&L history gap)"
            )
            logger.critical(message)
            self._publish_alert(
                "warning", message, {"date": target, "marker_only": True},
            )

        self._bus.publish("eod.reconciliation", {
            "date": target,
            "positions": 0,
            "market_value": 0.0,
            "total_equity": None,
            "equity_source": "backfill-marker-only" if not rows else "existing-snapshot",
            "catch_up": True,
            "marker_only": not rows,
            # R11: keep the topic's payload shape uniform. A historical day
            # is never compared against today's broker book (that would
            # fabricate an observation), so its reconciliation is unknown.
            "status": "unknown",
            "basis": "trade_ledger",
            "passes": False,
            "n_mismatched": 0,
            "cash_diff": None,
            "total_mismatch_value": None,
            "reason": "historical day: no broker/book observation to reconcile",
        }, source="scheduler")
        return True

    # ── Daily books-vs-broker reconciliation (R11) ──

    def _backfill_missing_reconciliation(self, target: str) -> bool:
        """Write today's missing reconciliation result without a snapshot.

        An upgraded database can already hold the day's EOD marker (and its
        P&L snapshot) from before the daily books-vs-broker step existed.
        Only *today's* observation is backfilled: a historical day's counter-
        book no longer exists, and reconciling it against today's broker
        would fabricate an observation — the same failure class as S2.
        Never returns True: this call writes no snapshot and no marker.
        """
        if target != self._now().strftime("%Y-%m-%d"):
            return False
        try:
            existing = self._store.get_reconciliation_result(target)
        except Exception as exc:
            logger.error(
                "EOD reconciliation result read failed for %s: %s", target, exc,
            )
            return False
        if existing is not None:
            return False

        snapshot = self._latest_pnl_snapshot(target)
        result = self._run_eod_reconciliation(target, snapshot=snapshot)
        positions = int(snapshot.get("n_positions") or 0) if snapshot else 0
        market_value = float(snapshot.get("market_value") or 0.0) if snapshot else 0.0
        equity = snapshot.get("total_equity") if snapshot else None
        self._publish_eod_reconciliation(
            target, positions, market_value, equity,
            "existing-snapshot" if snapshot else "marker-only",
            catch_up=False, result=result, backfill=True,
        )
        logger.warning(
            "EOD reconciliation %s: backfilled the missing books-vs-broker "
            "result (%s) from the existing snapshot; no new snapshot written",
            target, result["status"],
        )
        return False

    def _run_eod_reconciliation(
        self, target: str, snapshot: dict | None = None,
    ) -> dict:
        """Compute and persist the day's books-vs-broker outcome (R11/R12).

        Never raises: the EOD snapshot is already durable, and a broken
        reconciliation must neither undo it nor stop the event. A failure
        to compute is recorded as ``unknown`` (fail loud: never a pass).
        """
        try:
            result = self._compute_reconciliation(target, snapshot)
        except Exception as exc:
            logger.critical(
                "EOD books-vs-broker reconciliation failed for %s: %s",
                target, exc, exc_info=True,
            )
            result = self._unknown_reconciliation(
                target, f"reconciliation failed: {exc}",
            )
        try:
            self._store.save_reconciliation_result(result)
        except Exception as exc:
            # Keep the computed outcome in the event/log; the next EOD tick
            # sees the missing row and backfills it.
            logger.critical(
                "EOD reconciliation result write failed for %s: %s",
                result.get("date"), exc,
            )
        return result

    def _compute_reconciliation(
        self, target: str, snapshot: dict | None,
    ) -> dict:
        """expected = trade ledger (startup baseline + stored fills), actual = broker.

        R12: the expected leg is derived from the newest ``ledger_baselines``
        row -- the broker's cash/positions captured when the engine started --
        plus the fills stored after it (``ledger_from_trades``). The R11
        expected side (Store positions + the day's P&L snapshot cash) is gone:
        on the normal EOD path that snapshot's cash came from the very broker
        account being compared, so the cash arm was a self-comparison.

        Without a baseline the comparison cannot honestly run: the result is
        ``status="unknown"`` with reason ``no_ledger_baseline``. The Store
        position book is never used as a fallback expected leg.

        ``snapshot`` is kept for the caller's signature (its counts feed the
        event) but no longer feeds the comparison.

        Price basis: expected positions are valued at the actual (broker)
        price for the same asset (baseline ``avg_cost`` only when the asset is
        missing from the broker book), so only quantities and cash are
        compared -- a price-basis difference cannot manufacture a mismatch.
        """
        timestamp = self._naive_local(self._now())
        tolerances = dict(self._recon_tolerances)

        try:
            baseline = self._store.get_latest_ledger_baseline(
                self._tenant_id, self._engine_state_key,
            )
        except Exception as exc:
            logger.error(
                "EOD reconciliation ledger baseline read failed: %s", exc,
            )
            return self._unknown_reconciliation(
                target, f"ledger baseline query failed: {exc}", timestamp,
            )
        if baseline is None:
            return self._unknown_reconciliation(
                target, "no_ledger_baseline", timestamp,
            )
        try:
            trades = self._store.get_trades_for_export(
                tenant_id=self._tenant_id,
                start_date=str(baseline.get("created_at") or "")[:10],
            )
        except Exception as exc:
            logger.error(
                "EOD reconciliation trade ledger read failed: %s", exc,
            )
            return self._unknown_reconciliation(
                target, f"trade ledger query failed: {exc}", timestamp,
            )
        try:
            ledger = ledger_from_trades(baseline, trades)
        except Exception as exc:
            logger.error(
                "EOD reconciliation ledger derivation failed: %s", exc,
            )
            return self._unknown_reconciliation(
                target, f"trade ledger derivation failed: {exc}", timestamp,
            )

        broker = self._broker
        if broker is None:
            return self._unknown_reconciliation(
                target, "no broker attached; counter-book unknown", timestamp,
            )
        try:
            broker_positions = broker.get_positions()
        except Exception as exc:
            logger.error(
                "EOD reconciliation broker position query failed: %s", exc,
            )
            return self._unknown_reconciliation(
                target, f"broker position query failed: {exc}", timestamp,
            )
        if broker_positions is None:
            # The broker contract: None means the query failed. Treating it
            # as a flat account would "pass" a book we never observed.
            return self._unknown_reconciliation(
                target, "broker positions unknown (None)", timestamp,
            )
        try:
            account = broker.get_account() or {}
        except Exception as exc:
            logger.error(
                "EOD reconciliation broker account query failed: %s", exc,
            )
            return self._unknown_reconciliation(
                target, f"broker account query failed: {exc}", timestamp,
            )
        if not isinstance(account, dict):
            return self._unknown_reconciliation(
                target, "broker account payload is not a dict", timestamp,
            )
        if account.get("connected") is False:
            # A disconnected QMT/XTP adapter serves placeholder balances;
            # the snapshot path already refuses them (S2).
            return self._unknown_reconciliation(
                target, "broker reports disconnected", timestamp,
            )
        actual_cash = account.get("cash")
        if not self._is_number(actual_cash):
            return self._unknown_reconciliation(
                target, "broker account has no usable cash", timestamp,
            )

        actual_book = positions_from_broker(list(broker_positions))
        # Baseline cost per asset values an expected-only leg (a position the
        # broker book no longer lists); it never prices an asset the broker
        # holds -- that leg uses the broker's own mark.
        avg_costs: dict[str, float] = {}
        for code, entry in (baseline.get("positions") or {}).items():
            if not isinstance(entry, dict):
                continue
            try:
                avg_costs[str(code)] = float(entry.get("avg_cost") or 0.0)
            except (TypeError, ValueError):
                continue
        expected_book = expected_book_from_ledger(
            ledger["positions"], actual_book, avg_costs,
        )

        report = reconcile(
            expected_book,
            actual_book,
            float(ledger["cash"]),
            float(actual_cash),
            tolerance=tolerances["shares"],
            cash_tolerance=tolerances["cash"],
            cash_tolerance_bps=tolerances["cash_bps"],
            mismatch_materiality=tolerances["materiality"],
        )
        status = "passed" if report.passes else "failed"
        if status == "failed":
            self._log_reconciliation_failure(
                report, tolerances, float(ledger["cash"]), float(actual_cash),
            )
        return {
            "date": target,
            "timestamp": timestamp,
            "status": status,
            "basis": "trade_ledger",
            "n_positions": report.total_positions,
            "n_mismatched": report.mismatched,
            "cash_diff": report.cash_diff,
            "total_mismatch_value": report.total_mismatch_value,
            "details": [
                {
                    "asset": row.asset,
                    "expected_shares": row.expected_shares,
                    "actual_shares": row.actual_shares,
                    "expected_price": row.expected_price,
                    "actual_price": row.actual_price,
                    "diff": row.diff,
                    "tolerance": row.tolerance,
                    "is_matched": row.is_matched,
                }
                for row in report.details
            ],
            "tolerances": tolerances,
        }

    @staticmethod
    def _log_reconciliation_failure(
        report, tolerances: dict,
        expected_cash: float, actual_cash: float,
    ) -> None:
        """Log the first few breaching positions and any cash breach."""
        for row in [row for row in report.details if not row.is_matched][:5]:
            logger.error(
                "EOD reconciliation mismatch %s: expected=%s actual=%s diff=%s",
                row.asset, row.expected_shares, row.actual_shares, row.diff,
            )
        cash_limit = tolerances["cash"] + (
            tolerances["cash_bps"] / 10_000.0
            * max(abs(expected_cash), abs(actual_cash))
        )
        if abs(report.cash_diff) > cash_limit:
            logger.error(
                "EOD reconciliation cash mismatch: expected=%.2f actual=%.2f "
                "diff=%.2f limit=%.2f",
                expected_cash, actual_cash, report.cash_diff, cash_limit,
            )

    def _unknown_reconciliation(
        self, target: str, reason: str, timestamp: str | None = None,
    ) -> dict:
        """A comparison that could not run: never a pass, never a zero diff."""
        return {
            "date": target,
            "timestamp": timestamp or self._naive_local(self._now()),
            "status": "unknown",
            "basis": "trade_ledger",
            "n_positions": 0,
            "n_mismatched": 0,
            "cash_diff": None,
            "total_mismatch_value": None,
            "details": [],
            "tolerances": dict(self._recon_tolerances),
            "reason": reason,
        }

    def _latest_pnl_snapshot(self, target: str) -> dict | None:
        """Newest P&L snapshot for `target`, or None.

        The EOD snapshot is written under the store's default tenant (the
        snapshot dict carries no tenant_id), so this read is deliberately
        unscoped — the same basis ``save_pnl_and_mark_eod`` writes with. In
        a multi-tenant deployment every EOD snapshot shares that default
        tenant; tenant-scoped EOD snapshots are a separate change.
        """
        try:
            rows = self._store.get_pnl_history(
                start_date=target, end_date=target,
            )
        except Exception as exc:
            logger.error("EOD P&L snapshot read failed for %s: %s", target, exc)
            return None
        return rows[-1] if rows else None

    def _publish_eod_reconciliation(
        self, target: str, positions: int, market_value: float,
        equity: float | None, source: str, catch_up: bool, result: dict,
        backfill: bool = False,
    ) -> None:
        """Publish the enriched ``eod.reconciliation`` event (R11).

        The legacy keys are kept verbatim; the reconciliation outcome is
        added as ``status``/``passes``/``n_mismatched``/``cash_diff``/
        ``total_mismatch_value``.
        """
        payload = {
            "date": target,
            "positions": positions,
            "market_value": market_value,
            "total_equity": equity,
            "equity_source": source,
            "catch_up": catch_up,
            "status": result["status"],
            # R12: the expected leg is the derived trade ledger
            # (startup baseline + stored fills).
            "basis": result.get("basis", "trade_ledger"),
            # An unknown comparison is not a pass (fail loud).
            "passes": result["status"] == "passed",
            "n_mismatched": result.get("n_mismatched", 0),
            "cash_diff": result.get("cash_diff"),
            "total_mismatch_value": result.get("total_mismatch_value"),
        }
        if result.get("reason"):
            payload["reason"] = result["reason"]
        if backfill:
            payload["backfill"] = True
        self._bus.publish("eod.reconciliation", payload, source="scheduler")

    def _resolve_eod_values(
        self,
    ) -> tuple[float | None, float | None, list, str]:
        """Equity/cash/positions from the broker or persisted engine state.

        Returns ``(equity, cash, positions, source)``; equity is None when no
        trustworthy source exists.
        """
        broker = self._broker
        if broker is not None:
            try:
                account = broker.get_account() or {}
            except Exception as exc:
                logger.error("EOD broker account query failed: %s", exc)
                account = {}
            if not isinstance(account, dict):
                account = {}
            # A disconnected QMT/XTP adapter returns its `initial_cash` as a
            # placeholder balance (trading/broker.py). Treating that as equity
            # would fabricate a snapshot exactly like the old 1,000,000
            # constant; fall through to persisted state instead.
            if account.get("connected") is False:
                logger.error(
                    "EOD broker reports disconnected; ignoring its placeholder "
                    "account values",
                )
                account = {}
            equity = account.get("total_equity")
            if not self._valid_equity(equity):
                market_value = account.get("market_value")
                cash_value = account.get("cash")
                if self._is_number(cash_value) and self._is_number(market_value):
                    equity = float(cash_value) + float(market_value)
            if self._valid_equity(equity):
                cash = account.get("cash")
                cash_value = float(cash) if self._is_number(cash) else None
                try:
                    positions = list(broker.get_positions() or [])
                except Exception:
                    positions = []
                if not positions:
                    positions = self._store_positions(self._tenant_id)
                return float(equity), cash_value, positions, (
                    f"broker:{type(broker).__name__}"
                )

        state: dict | None = None
        state_key = ""
        if self._engine_state_key:
            try:
                candidate = self._store.get_engine_state(self._engine_state_key)
            except Exception as exc:
                logger.error("EOD engine-state read failed: %s", exc)
                candidate = None
            if isinstance(candidate, dict):
                state, state_key = candidate, self._engine_state_key
        if state is None:
            try:
                latest = self._store.get_latest_engine_state(self._tenant_id)
            except Exception as exc:
                logger.error("EOD latest engine-state read failed: %s", exc)
                latest = None
            if latest is not None:
                state_key, state = latest

        if isinstance(state, dict):
            equity = state.get("portfolio_value")
            if not self._valid_equity(equity):
                equity = state.get("total_equity")
            if self._valid_equity(equity):
                cash_raw = state.get("cash")
                cash = float(cash_raw) if self._is_number(cash_raw) else None
                positions = [
                    p for p in (state.get("positions") or [])
                    if isinstance(p, dict)
                ]
                if not positions:
                    positions = self._store_positions(self._tenant_id)
                return float(equity), cash, positions, (
                    f"engine_state:{state_key or 'latest'}"
                )

        return None, None, [], ""

    def _store_positions(self, tenant_id: str = "") -> list[dict]:
        try:
            return list(self._store.get_positions(tenant_id) or [])
        except Exception as exc:
            logger.error("EOD position read failed: %s", exc)
            return []

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    @classmethod
    def _valid_equity(cls, value) -> bool:
        return cls._is_number(value) and float(value) > 0

    @staticmethod
    def _position_market_value(position) -> float:
        if isinstance(position, dict):
            value = position.get("market_value", 0)
        else:
            value = getattr(position, "market_value", 0)
        try:
            number = float(value or 0)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    def _skip_eod(self, target: str) -> None:
        """No trustworthy equity: never invent one (S2)."""
        message = (
            f"EOD reconciliation skipped for {target}: no equity source "
            "(no broker account and no persisted engine state)"
        )
        try:
            get_metrics().counter(
                "scheduler_eod_skipped_total",
                help="EOD reconciliations skipped because equity was unavailable",
            ).inc()
        except Exception:
            pass
        # `_publish_alert` throttles the log/event to once per five minutes;
        # logging CRITICAL on every 30s tick filled the log with ~1,000 lines
        # per affected evening.
        self._publish_alert("critical", message, {"date": target})

    def _run_retention_hook(self) -> None:
        """Daily retention after a successful EOD (S10). Failure is non-fatal."""
        if self._retention_policy is None or not self._retention_policy.is_enabled():
            return
        try:
            result = self._store.run_retention(
                policy=self._retention_policy,
                archive_dir=self._retention_archive_dir,
            )
        except Exception as exc:
            logger.error("Retention job failed: %s", exc, exc_info=True)
            self._publish_alert("error", f"retention job failed: {exc}")
            return
        self._bus.publish("store.retention", {
            "tables": {
                table: {
                    "archived": entry.get("archived", 0),
                    "deleted": entry.get("deleted", 0),
                }
                for table, entry in (result.get("tables") or {}).items()
            },
            "db_size_bytes": result.get("db_size_bytes", 0),
        }, source="scheduler")

    def _catch_up_missed_eod(self, include_today: bool = True) -> int:
        """Backfill EOD reconciliation for trading days missed while down.

        Compares the newest EOD marker (or P&L snapshot) with the exchange
        calendar and reconciles at most ``MAX_EOD_CATCHUP_DAYS`` sessions
        (S3). A fresh database has no baseline and therefore nothing to
        backfill. ``include_today=False`` is used by the periodic scan: today
        belongs to the normal 15:00 EOD window.
        """
        now = self._now()
        try:
            baseline = self._store.get_last_eod_reconciliation()
            if not baseline:
                last_pnl = self._store.get_last_pnl_timestamp()
                baseline = last_pnl[:10] if last_pnl else ""
        except Exception as exc:
            logger.error("EOD catch-up baseline read failed: %s", exc)
            return 0
        if not baseline:
            return 0
        try:
            baseline_date = date.fromisoformat(baseline[:10])
        except ValueError:
            logger.warning("Ignoring unparsable EOD baseline %r", baseline)
            return 0

        end = now.date()
        if not include_today or (
            is_trading_day(end) and (now.hour, now.minute) < (EOD_HOUR, EOD_MINUTE)
        ):
            # Today only counts once the exchange has closed; the periodic
            # scan never includes today at all.
            end = end - timedelta(days=1)

        scan_start = baseline_date + timedelta(days=1)
        earliest = end - timedelta(days=MAX_EOD_CATCHUP_DAYS * 3 + 14)
        if scan_start < earliest:
            self._publish_alert(
                "warning",
                f"EOD catch-up bounded to the most recent "
                f"{MAX_EOD_CATCHUP_DAYS} trading day(s); baseline is {baseline}",
                {"baseline": baseline},
            )
            scan_start = earliest

        missed: list[date] = []
        cursor = scan_start
        while cursor <= end:
            if is_trading_day(cursor):
                missed.append(cursor)
            cursor += timedelta(days=1)
        missed = missed[-MAX_EOD_CATCHUP_DAYS:]

        reconciled = 0
        for day in missed:
            day_str = day.isoformat()
            if self._store.has_eod_reconciliation(day_str):
                continue
            if self._do_eod_reconciliation(day_str, catch_up=True):
                reconciled += 1

        if reconciled:
            logger.warning(
                "EOD catch-up reconciled %d missed trading day(s)", reconciled,
            )
            self._publish_alert(
                "warning",
                f"EOD catch-up reconciled {reconciled} missed trading day(s)",
                {"days": [str(day) for day in missed], "reconciled": reconciled},
            )
        return reconciled
