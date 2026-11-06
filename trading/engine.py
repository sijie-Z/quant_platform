"""Event-driven live trading engine.

Architecture:
    Scheduler → Engine → Broker
        ↓          ↓        ↓
    EventBus    Store    Market Data
        ↓          ↓
    AuditLog   StateMachine

Every decision flows through the event bus.
Every state is persisted in SQLite.
Every transition is logged for compliance.

Signal generation uses the same alpha pipeline as backtest research.
See trading/signal_generator.py for details.
"""

from __future__ import annotations

import inspect
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quant_platform.core.audit import AuditAction, AuditLog
from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.core.exchange_calendar import is_trading_day
from quant_platform.core.scheduler import TradingScheduler
from quant_platform.core.state_machine import PortfolioState, PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.execution.models import OrderType
from quant_platform.risk.circuit_breaker import RiskMonitor, get_risk_monitor
from quant_platform.risk.healthcheck import (
    HealthCheck,
    SystemBlockError,
    emit_alert,
    install_alert_handler,
)
from quant_platform.risk.realtime_engine import RealTimeRiskEngine
from quant_platform.shared.symbols import normalize_symbol
from quant_platform.trading.broker import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    SimulatedBroker,
)
from quant_platform.trading.live_engine import (
    normalize_corporate_action_feed,
    summarize_corporate_action_events,
    working_order_codes,
    working_order_quantities,
)
from quant_platform.trading.signal_generator import LiveSignalGenerator
from quant_platform.trading.signals import (
    DEFAULT_MIN_SIGNAL_SCORE,
    SIGNAL_HISTORY_SESSIONS,
    composite_signal,
    rank_symbols,
)
from quant_platform.utils.logging import get_logger
from quant_platform.utils.metrics import get_metrics

try:
    from quant_platform.core.context import TenantContext
except ImportError:
    TenantContext = None  # type: ignore[assignment,misc]

try:
    from quant_platform.operations.nav import NAVCalculator
except ImportError:
    NAVCalculator = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)

#: T+1 rollover and the daily P&L baseline must follow the exchange's
#: calendar, not the host clock (D12): on a UTC/UTC-3 container a local
#: midnight used to unlock same-session buys and reset the loss baseline.
EXCHANGE_TZ = ZoneInfo("Asia/Shanghai")


def exchange_now() -> datetime:
    """Current time in exchange local time (Asia/Shanghai).

    Derived from the epoch clock rather than ``datetime.now(tz)`` so a frozen
    or monkeypatched host clock (embedded runners, tests) cannot move the
    exchange calendar under the engine. The exchange calendar, not the host
    calendar, decides T+1 rollover and the daily P&L baseline (D12).
    """
    return datetime.fromtimestamp(time.time(), EXCHANGE_TZ)


@dataclass
class TradeSignal:
    code: str
    side: OrderSide
    target_weight: float
    current_weight: float
    signal_strength: float
    reason: str = ""


@dataclass
class CycleResult:
    cycle_id: int = 0
    timestamp: str = ""
    signals: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    portfolio_value: float = 0
    cash: float = 0
    n_positions: int = 0

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id, "timestamp": self.timestamp,
            "signals": self.signals, "orders": self.orders,
            "portfolio_value": round(self.portfolio_value, 2),
            "cash": round(self.cash, 2), "n_positions": self.n_positions,
        }


class LiveTradingEngine:
    """Event-driven live trading engine.

    Integrates with core architecture:
    - EventBus: publishes market.tick, signal.generated, order.*, portfolio.* events
    - Store: persists orders, positions, P&L, signals to SQLite
    - StateMachine: manages INIT→READY→TRADING→REBALANCING→POST_MARKET lifecycle
    - AuditLog: compliance-grade logging of every decision
    - Scheduler: market-aware timing and session management

    Usage:
        store = Store('data/trading.db')
        bus = get_event_bus()
        sm = PortfolioStateMachine()
        audit = AuditLog(store, bus)
        broker = SimulatedBroker(1_000_000)

        engine = LiveTradingEngine(broker, store, bus, sm, audit)
        engine.set_universe(['600519', '000001', ...])
        engine.start()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        store: Store | None = None,
        bus: EventBus | None = None,
        state_machine: PortfolioStateMachine | None = None,
        audit: AuditLog | None = None,
        risk_monitor: RiskMonitor | None = None,
        signal_generator: LiveSignalGenerator | None = None,
        rebalance_interval: int = 300,
        n_stocks: int = 50,
        min_trade_value: float = 1000,
        min_signal_score: float = DEFAULT_MIN_SIGNAL_SCORE,
        tenant_id: str = "default",
        corporate_actions=None,
        data_provider=None,
    ):
        self._broker = broker
        self._store = store or Store()
        self._bus = bus or get_event_bus()
        self._sm = state_machine or PortfolioStateMachine()
        self._audit = audit or AuditLog(
            self._store, self._bus, tenant_id=tenant_id,
        )
        # State changes were never audited: the machine's callback is
        # optional and no production caller registered one.
        if getattr(self._sm, "_on_transition", None) is None:
            self._sm.set_transition_callback(
                lambda record: self._audit.log_state_change(
                    record.from_state, record.to_state, record.reason,
                )
            )
        # D13: default to the process-wide RiskMonitor -- the same singleton
        # the API kill switch mutates. RealTimeRiskEngine's compatibility
        # `update_portfolio_state` ignored daily_pnl/positions/sector_weights
        # and its pre_trade_check only inspected the new order, so an engine
        # constructed without an explicit monitor could hold 4% of a name and
        # still be approved for another 4% (8% > 5% limit), and a 20%
        # drawdown never armed the kill switch. Callers may still inject any
        # object implementing check_pre_trade/update_portfolio_state/
        # get_status (tests inject a fresh RiskMonitor for isolation).
        self._risk = risk_monitor or get_risk_monitor()
        # S21: risk level, breaches and the kill switch are durable. Attaching
        # the store here restores a manually armed kill switch from a previous
        # process before any health check or order can run.
        attach_risk_state = getattr(self._risk, "attach_persistence", None)
        if callable(attach_risk_state):
            try:
                attach_risk_state(
                    self._store, tenant_id=tenant_id, audit=self._audit,
                    restore=True,
                )
            except Exception as exc:
                logger.error("Risk-state persistence setup failed: %s", exc)
        # The default signal path is the shared `trading.signals.composite_signal`
        # -- the same causal 7-factor equal-weight score the CLI LiveRunner
        # uses.  An injected generator (tests or an alternative alpha model
        # exposing update_prices()/generate()) still takes precedence.
        self._signal_generator = signal_generator
        self._rebalance_interval = rebalance_interval
        self._n_stocks = n_stocks
        self._min_trade_value = min_trade_value
        self._min_signal_score = min_signal_score
        self._tenant_id = tenant_id

        # Set tenant context for this engine instance
        if TenantContext is not None:
            TenantContext.set_current(TenantContext(tenant_id=tenant_id))

        self._running = False
        # Distinguishes an intentional stop from a rolled-back failed start,
        # so stop() does not rewrite a "failed" session as "stopped".
        self._start_failed = False
        self._thread: threading.Thread | None = None
        # D14: created fresh in `start()` for every run; the loop receives its
        # own Event by argument. The old shared event was cleared by start(),
        # which resurrected a loop that had outlived stop().
        self._stop_event: threading.Event | None = None
        self._universe: list[str] = []
        self._current_prices: dict[str, float] = {}
        self._price_history: dict[str, list[float]] = {}
        # D11: per-symbol freshness, monotonic clock so a host clock jump
        # cannot make dead quotes look fresh (or vice versa). `_last_fetch_codes`
        # is the set refreshed by the most recent fetch; a symbol omitted from
        # the latest fetch is not eligible for sizing even inside the age
        # window (its feed is dead while another symbol keeps updating).
        self._price_last_seen: dict[str, float] = {}
        self._last_fetch_codes: set[str] = set()
        self._cycle_count = 0
        self._trade_count = 0
        self._signal_count = 0
        self._started_at = ""
        self._last_update = ""
        self._last_prices_update = ""
        # Monotonic companion used for the aggregate staleness guard so a
        # host clock jump cannot make quotes look fresh (D11/D12).
        self._last_prices_update_monotonic = 0.0
        # One provider per engine: constructing a fresh RealTimeMarket every
        # cycle threw away its 10-second quote cache and forced a full market
        # scrape on every fetch.
        self._rt_market = None
        self._cycle_log: list[CycleResult] = []
        self._session_id = ""
        # Serialises the daemon loop against manual run-once triggers.
        self._cycle_lock = threading.Lock()
        # R13: a manual order whose broker execution is being persisted by
        # the synchronous path right now (`manual_order` runs outside the
        # cycle lock). While non-zero the fill sync stands down: a broker may
        # replace the tracked order with a recovered instance under a new
        # local id inside `place_order`, so no id-based skip can see the
        # whole window.
        self._fills_in_flight = 0
        self._fills_in_flight_lock = threading.Lock()
        # One sync at a time: the idle loop after the close, a run-once
        # cycle and a manual trigger can call the sync concurrently, and two
        # delta reads racing the same broker snapshot would each book the
        # same fill.
        self._fill_sync_lock = threading.Lock()
        # R13 startup-sync outcome: `confirmed` means the broker order book
        # was actually readable; `blocked` means a filled order could not be
        # booked (query/write failure) and a later cycle might retry it. Both
        # gate the R12 baseline: writing it while fills are unbooked would
        # make the EOD ledger count them twice.
        self._fill_sync_confirmed = False
        self._fill_sync_blocked = False
        # Refuse to size orders from quotes older than this.
        self._max_quote_age_seconds = 120
        # D6: the cycle's validated position snapshot, reused by signal
        # generation and order sizing so one query cannot disagree with the
        # next within the same cycle.
        self._cycle_positions: list | None = None
        # T+1 rollover and daily P&L baseline.
        self._trading_day = ""
        self._day_start_equity: float | None = None
        # D4/D5: engine-state snapshot key (tenant-scoped so two tenants on
        # one Store cannot clobber each other's cash/kill-switch row).
        self._engine_state_key = f"live_engine:{tenant_id}"
        # Corporate actions (paper books only).  A feed can be supplied
        # directly (DataFrame, or a callable returning one) and/or loaded
        # lazily from an injected ``data_provider`` on the first trading-day
        # advance.  Real brokers (QMT/XTP) settle actions on their side:
        # ``supports_corporate_actions`` is False for them and the engine
        # never hands them a local feed (no double accounting).
        self._corporate_action_feed = corporate_actions
        self._corporate_action_provider = data_provider
        # True only once a paper book accepted a feed (a real broker never
        # does).  A callable/provider feed stays dynamic and is re-queried on
        # every day advance.
        self._corporate_actions_loaded = False
        self._corporate_actions_source = ""
        self._corporate_action_feed_rows = 0
        self._corporate_action_events: list[dict] = []
        if self._broker_supports_corporate_actions():
            # Applied (asset, ex_date) markers must survive a same-day
            # restart: a fresh SimulatedBroker re-loads them from the Store.
            configure = getattr(
                self._broker, "configure_corporate_action_persistence", None,
            )
            if callable(configure):
                try:
                    configure(
                        self._store, f"paper_corporate_actions:{tenant_id}",
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not wire corporate-action marker persistence: %s",
                        exc,
                    )
        if corporate_actions is not None and not callable(corporate_actions):
            self.set_corporate_actions(corporate_actions, source="constructor")
        # D14: how long stop() waits for the loop thread and for an in-flight
        # cycle to release the cycle lock before giving up loudly.
        self._stop_join_timeout = 30.0

        # NAV calculator
        self._nav_calc = None
        if NAVCalculator is not None:
            self._nav_calc = NAVCalculator(self._store)

        # Scheduler. It receives the broker and the tenant-scoped engine-state
        # key so EOD reconciliation can read real equity instead of inventing
        # one (S2), and reuses this engine's session id (S20).
        self._scheduler = TradingScheduler(
            self._sm, self._store, self._bus, rebalance_interval,
            broker=broker,
            engine_state_key=self._engine_state_key,
            tenant_id=tenant_id,
        )
        # Use the scheduler's validated/coerced integer. A numeric string
        # passed here would otherwise reach `event.wait(timeout=...)` and
        # raise outside the loop's try block.
        self._rebalance_interval = self._scheduler._rebalance_interval

    def set_universe(self, codes: list[str]):
        self._universe = codes
        self._bus.publish("engine.universe_set", {"codes": codes, "count": len(codes)}, source="engine")

    # ── Corporate actions (paper books only) ──

    def _broker_supports_corporate_actions(self) -> bool:
        """Whether this engine owns the book (paper) or the broker does."""
        return bool(
            getattr(self._broker, "supports_corporate_actions", False)
        )

    def set_corporate_actions(self, actions, source: str = "manual") -> int:
        """Load the paper book's cash/split feed.

        ``actions`` is a provider-contract frame
        ``(ex_date, asset, kind, cash_per_share, split_ratio)`` or a callable
        returning one (a zero-argument callable, or one taking
        ``(start_date, end_date)``).  A callable is resolved lazily on the
        next trading-day advance over a lookback window ending on that day.

        Nothing is applied here: actions are applied exactly once as the
        exchange day advances (before marking/sizing), so an ex-date that
        already passed is caught up on the next cycle.  Real brokers settle
        corporate actions remotely; for them the feed is refused (returns 0)
        and nothing is applied locally.

        Returns the number of feed rows loaded (0 for a deferred callable or
        an unsupported/real broker).
        """
        if callable(actions):
            self._corporate_action_feed = actions
            self._corporate_actions_loaded = False
            self._corporate_actions_source = (
                f"callable:{getattr(actions, '__name__', 'callable')}"
            )
            return 0
        if not self._broker_supports_corporate_actions():
            logger.warning(
                "Broker %s settles corporate actions remotely; the feed is "
                "NOT applied to the local book",
                type(self._broker).__name__,
            )
            return 0
        setter = getattr(self._broker, "set_corporate_actions", None)
        if not callable(setter):
            return 0
        # Pin tz-aware ex-dates to naive exchange days before the broker
        # compares them with the engine's YYYY-MM-DD advance date.
        frame = normalize_corporate_action_feed(actions)
        count = int(setter(frame) or 0)
        self._corporate_actions_loaded = True
        self._corporate_actions_source = str(source or "manual")
        self._corporate_action_feed_rows = count
        return count

    @staticmethod
    def _invoke_action_loader(loader, start_date: str, end_date: str):
        """Call a feed callable with ``(start, end)`` or no arguments."""
        try:
            parameters = inspect.signature(loader).parameters.values()
        except (TypeError, ValueError):  # builtins / exotic callables
            parameters = ()
        required = [
            parameter for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(required) >= 2:
            return loader(start_date, end_date)
        return loader()

    def _refresh_corporate_action_feed(self, date: str, since: str = "") -> None:
        """(Re)load a dynamic feed for the advance window ending on ``date``.

        A callable ``corporate_actions`` feed or an injected
        ``data_provider`` is queried on *every* exchange-day advance over
        ``[last processed day, date]`` (400-day lookback when there is no
        previous day), so a long-running engine still sees dividends/splits
        announced after startup.  Applied markers make replacing the feed
        idempotent.  A static DataFrame is loaded once and never re-queried.
        A failing or malformed feed is logged and never aborts the cycle.
        """
        feed = self._corporate_action_feed
        provider = self._corporate_action_provider
        feed_is_callable = callable(feed)
        if feed is not None and not feed_is_callable:
            return  # an explicit static feed is authoritative
        if not feed_is_callable and provider is None:
            return
        if not self._broker_supports_corporate_actions():
            return
        try:
            end = pd.Timestamp(date).normalize()
        except (TypeError, ValueError):
            end = pd.Timestamp(self._exchange_trading_date())
        first = None
        if since:
            try:
                first = pd.Timestamp(since).normalize()
            except (TypeError, ValueError):
                first = None
        if first is None or first > end:
            first = end - pd.Timedelta(days=400)
        start = first.date().isoformat()
        end_text = end.date().isoformat()

        if feed_is_callable:
            try:
                frame = self._invoke_action_loader(feed, start, end_text)
            except Exception as exc:
                logger.error("Corporate-action feed callable failed: %s", exc)
                return
            source = f"callable:{getattr(feed, '__name__', 'callable')}"
        else:
            hook = getattr(provider, "get_corporate_actions", None)
            if not callable(hook):
                return
            try:
                # Provider accessors and pipelines differ in arity
                # (``(start, end)`` vs ``()``); invoke whichever fits.
                frame = self._invoke_action_loader(hook, start, end_text)
            except Exception as exc:
                logger.error("Corporate-action provider failed: %s", exc)
                return
            source = f"provider:{type(provider).__name__}"
        try:
            self.set_corporate_actions(frame, source=source)
        except Exception as exc:
            logger.error(
                "Corporate-action feed returned an invalid frame: %s", exc,
            )

    def _advance_trading_day(self, date: str, since: str = "") -> list[dict]:
        """Release T+1 locks and apply the day's corporate actions.

        Replaces the bare ``new_trading_day()`` call: the paper broker is
        given the exchange date so every pending ``ex_date <= date`` is
        applied exactly once, before the cycle sizes orders or marks the
        book.  ``since`` is the previously processed exchange day; a lazily
        loaded provider feed only asks for actions after it.  Applied events
        are kept for the run report.  Real brokers only get the legacy no-op
        rollover.
        """
        self._refresh_corporate_action_feed(date, since=since)
        new_day = getattr(self._broker, "new_trading_day", None)
        if not callable(new_day):
            return []
        applied: list[dict] = []
        if self._broker_supports_corporate_actions():
            try:
                applied = new_day(date=date) or []
            except TypeError:  # compatible custom broker signature
                new_day()
                applied = []
        else:
            try:
                new_day(date=date)
            except TypeError:
                new_day()
        if not isinstance(applied, (list, tuple)):
            applied = []
        applied = [event for event in applied if isinstance(event, dict)]
        if applied:
            self._corporate_action_events.extend(applied)
            logger.info(
                "Applied %d corporate-action event(s) on %s",
                len(applied), date,
            )
        return applied

    def get_corporate_actions_applied(self) -> dict:
        """Run-report provenance: count, latest ex-date and feed source."""
        return summarize_corporate_action_events(
            self._corporate_action_events,
            source=self._corporate_actions_source,
        )

    def start(self):
        if self._running:
            return

        # D14: refuse to resurrect a previous run. `stop()` used to return
        # after a 10-second join with the cycle thread still alive; `start()`
        # then cleared the shared Event and spawned a second loop, leaving
        # two trading threads. A per-run Event plus this liveness check makes
        # that impossible.
        previous_thread = self._thread
        if previous_thread is not None and previous_thread.is_alive():
            previous_thread.join(timeout=1.0)
            if previous_thread.is_alive():
                message = (
                    "Engine start refused: previous run loop is still alive; "
                    "stop it and wait for the cycle to quiesce"
                )
                logger.critical(message)
                self._bus.publish(
                    "engine.start_blocked", {"error": message}, source="engine",
                )
                raise RuntimeError(message)
        self._thread = None
        self._stop_event = threading.Event()
        # True until the loop thread is actually running: failures before the
        # rollback block (broker connect, state restore, health check) must
        # still leave a "failed" session when the API rolls back with stop().
        self._start_failed = True

        self._session_id = uuid.uuid4().hex[:12]
        # A failed QMT/XTP connection used to be ignored: the engine reported
        # "started" and every order was silently rejected.
        if not self._broker.connect():
            message = f"Broker connection failed: {type(self._broker).__name__}"
            logger.critical(message)
            self._bus.publish("engine.start_blocked", {"error": message}, source="engine")
            raise RuntimeError(message)

        # D4/D5: restore the persisted engine snapshot and reconcile the
        # store's book with the broker BEFORE the health check compares them.
        # A fresh process with a fresh SimulatedBroker used to be blocked by
        # `position_sync` on every restart with open positions.
        try:
            self._restore_state()
        except SystemBlockError as e:
            logger.critical("Engine start blocked by state reconciliation: %s", e)
            self._bus.publish("engine.start_blocked", {"error": str(e)}, source="engine")
            raise

        # R13: absorb fills the broker executed while this process was down
        # (or that its asynchronous callback never handed to the engine)
        # BEFORE the ledger baseline is stamped. The baseline's broker book
        # already contains those fills, so trades written before its
        # ``created_at`` are baked into the baseline instead of being
        # replayed by the EOD ledger (which would double count them). The
        # explicit stamp guarantees that ordering even when the host clock
        # has coarse granularity (Windows ticks ~15 ms) or the broker's
        # ``updated_at`` is observation time rather than execution time.
        startup_sync_at = datetime.now()
        self._sync_broker_fills(executed_at=startup_sync_at.isoformat())
        if self._fill_sync_confirmed and not self._fill_sync_blocked:
            # R12: broker and Store are reconciled, so the broker state is
            # now authoritative. Record it once as the trade ledger's
            # baseline; the EOD expected book is derived from this row plus
            # later fills.
            self._write_ledger_baseline(not_before=startup_sync_at)
        else:
            # Writing a baseline whose broker book may include fills we could
            # not book would make the EOD ledger count them twice (once via
            # the baseline, once when a later sync retries). Withhold it: EOD
            # then reports `no_ledger_baseline` (unknown, loud) instead.
            logger.critical(
                "Broker fills could not be fully reconciled at startup "
                "(order book readable=%s, unbooked-fill failure=%s); the "
                "trade-ledger baseline is withheld so EOD reports "
                "'no_ledger_baseline' instead of double counting",
                self._fill_sync_confirmed, self._fill_sync_blocked,
            )

        # Pre-flight health check
        health = HealthCheck(
            event_bus=self._bus,
            broker=self._broker,
            risk_monitor=self._risk,
            local_positions=lambda: self._store.get_positions(self._tenant_id),
        )
        try:
            health.run_all_sync()
        except SystemBlockError as e:
            logger.critical("Engine start blocked by health check: %s", e)
            self._bus.publish("engine.start_blocked", {"error": str(e)}, source="engine")
            raise

        # Set initial equity / restore the drawdown peak for the risk engine.
        account = self._broker.get_account()
        equity = float(account.get("total_equity", account.get("initial_cash", 1_000_000)) or 0)
        if isinstance(self._risk, RealTimeRiskEngine):
            self._risk.set_initial_equity(equity)
        elif isinstance(self._risk, RiskMonitor):
            self._risk.peak_value = max(self._risk.peak_value, equity)
            if self._risk.portfolio_value <= 0:
                self._risk.portfolio_value = equity
        self._started_at = self._exchange_now().replace(tzinfo=None).isoformat()

        # S1: only enter TRADING when the exchange is actually open. The old
        # unconditional READY -> TRADING meant a 09:00 start looked like a
        # closed market with a trading portfolio, which made the scheduler run
        # EOD reconciliation before the session existed.
        self._sm.transition(PortfolioState.READY, "engine started")
        market = self._scheduler.get_market_session()
        if self._scheduler.is_market_open():
            self._sm.transition(PortfolioState.TRADING, "market open")
        elif market.status == "pre_market":
            self._sm.transition(PortfolioState.PRE_MARKET, "market pre-open")
        # Otherwise the engine stays READY and the scheduler performs the
        # PRE_MARKET/TRADING transition when the exchange session begins.

        # S16/C5: every fallible setup step is inside one rollback. Setting
        # `_running = True` before a session/scheduler/thread failure left a
        # zombie engine: get_state() said "running" but no loop existed and
        # the `if self._running: return` guard blocked every retry.
        started_scheduler = False
        try:
            # Record session
            self._store.save_session({
                "session_id": self._session_id,
                "broker": type(self._broker).__name__,
                "status": "active",
                "started_at": self._started_at,
            })

            # S4: make sure a sink handler exists for health-check/engine
            # alerts even when embedded without the API lifespan.
            try:
                install_alert_handler(self._bus)
            except Exception as exc:
                logger.error("Alert handler install failed: %s", exc)

            # Start scheduler (reusing this engine's session id, S20). Mark
            # it before the call so the rollback below always stops it, even
            # if start() raises after partially entering.
            started_scheduler = True
            self._scheduler.start(session_id=self._session_id)

            # S16: write ENGINE_START *before* the loop thread exists.
            try:
                self._audit.log(
                    AuditAction.ENGINE_START, "engine",
                    reason=f"session={self._session_id}",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ENGINE_START audit failed ({exc})"
                ) from exc

            # Start engine loop with its own stop Event (D14). Embedded/test
            # overrides of `_run_loop` may still take zero arguments, so only
            # pass the per-run event when the callable accepts it.
            loop_target = getattr(self._run_loop, "__func__", self._run_loop)
            accepts_event = False
            try:
                accepts_event = "stop_event" in inspect.signature(
                    loop_target
                ).parameters
            except (TypeError, ValueError):
                accepts_event = False
            if accepts_event:
                target_thread = threading.Thread(
                    target=self._run_loop, args=(self._stop_event,), daemon=True,
                )
            else:
                target_thread = threading.Thread(
                    target=self._run_loop, daemon=True,
                )

            self._running = True
            self._thread = target_thread
            self._thread.start()
            self._start_failed = False
        except Exception as exc:
            self._running = False
            self._start_failed = True
            self._thread = None
            # Reset the lifecycle so a retry starts from READY rather than a
            # half-entered TRADING/PRE_MARKET state.
            try:
                self._sm.force_state(PortfolioState.READY, "start rollback")
            except Exception:
                pass
            if started_scheduler:
                try:
                    self._scheduler.stop()
                except Exception as stop_exc:
                    logger.error("Rollback scheduler stop failed: %s", stop_exc)
            try:
                self._store.save_session({
                    "session_id": self._session_id,
                    "status": "failed",
                    "stopped_at": self._exchange_now().replace(
                        tzinfo=None,
                    ).isoformat(),
                })
            except Exception as session_exc:
                logger.error("Rollback session write failed: %s", session_exc)
            try:
                self._bus.publish("engine.start_blocked", {
                    "error": str(exc),
                }, source="engine")
            except Exception:
                pass
            logger.critical("Engine start aborted: %s", exc)
            raise

        # Start the dead-man heartbeat immediately: readiness allows 2x the
        # cycle interval before it expects the first loop beat.
        self._record_cycle_heartbeat(None)

        self._bus.publish("engine.started", {
            "session_id": self._session_id,
            "n_stocks": self._n_stocks,
            "universe_size": len(self._universe),
        }, source="engine")

        logger.info("Engine started. Session: %s, Universe: %d", self._session_id, len(self._universe))

    def stop(self):
        """Stop the engine. Every step is best-effort and independent.

        S16: `scheduler.stop()` used to run before the final snapshot and
        could raise, skipping the engine_state write, the session completion
        and the store flush entirely. The final snapshot and WAL checkpoint
        now always run.
        """
        self._running = False
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()

        try:
            self._scheduler.stop()
        except Exception as exc:
            logger.error("Scheduler stop failed during engine stop: %s", exc)

        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=self._stop_join_timeout)
            except Exception as exc:
                logger.error("Engine thread join failed: %s", exc)
            if thread.is_alive():
                logger.critical(
                    "Engine loop thread still alive %.0fs after stop request",
                    self._stop_join_timeout,
                )
                try:
                    self._bus.publish("system.error", {
                        "error": "engine loop thread did not exit before stop completed",
                    }, source="engine")
                except Exception:
                    pass
                emit_alert(
                    "critical",
                    f"engine loop thread still alive after "
                    f"{self._stop_join_timeout:.0f}s stop timeout",
                    source="engine",
                )

        # D14: wait for an in-flight manually triggered cycle to release the
        # lock before writing the "final" session/NAV snapshot. Previously the
        # loop thread join did not cover `POST /trading/run-once`, so stop
        # could save a final state while that cycle was still trading.
        try:
            quiesced = self._cycle_lock.acquire(timeout=self._stop_join_timeout)
        except Exception as exc:
            quiesced = False
            logger.error("Cycle-lock wait failed during stop: %s", exc)
        if quiesced:
            self._cycle_lock.release()
        else:
            logger.critical(
                "A trading cycle held the cycle lock for %.0fs during stop; "
                "final state may not reflect it",
                self._stop_join_timeout,
            )
            try:
                self._bus.publish("system.error", {
                    "error": "active cycle did not quiesce before stop completed",
                }, source="engine")
            except Exception:
                pass
            emit_alert(
                "critical",
                "active trading cycle did not quiesce before engine stop",
                source="engine",
            )

        # R13: a live loop stops cycling after the close, so fills that
        # landed after the final cycle would never reach the trade ledger.
        # Best effort: a broken broker link only warns (and EOD fails loud).
        self._sync_broker_fills()

        try:
            if self._sm.state in (
                PortfolioState.TRADING, PortfolioState.REBALANCING,
            ):
                self._sm.transition(PortfolioState.POST_MARKET, "engine stopped")
            if self._sm.state == PortfolioState.POST_MARKET:
                self._sm.transition(PortfolioState.READY, "post-market complete")
        except Exception as exc:
            logger.error("State transition during stop failed: %s", exc)

        # D5/S16: the final snapshot and the session row always run, even
        # when scheduler.stop()/the thread join failed above.
        try:
            self._persist_engine_state()
        except Exception as exc:
            logger.error("Final engine-state persist failed: %s", exc)

        try:
            self._store.save_session({
                "session_id": self._session_id,
                "status": "failed" if self._start_failed else "stopped",
                "stopped_at": self._exchange_now().replace(tzinfo=None).isoformat(),
                "total_trades": self._trade_count,
            })
        except Exception as exc:
            logger.error("Final session write failed: %s", exc)

        # Day-end NAV calculation
        if self._nav_calc is not None:
            try:
                account = self._broker.get_account()
                cash = account.get("cash", 0)
                market_value = account.get("total_equity", 0) - cash
                nav = self._nav_calc.update_daily_nav(
                    date=self._exchange_now().strftime("%Y-%m-%d"),
                    cash=cash,
                    market_value=market_value,
                )
                self._bus.publish("nav.updated", nav.to_dict(), source="engine")
            except Exception as e:
                logger.warning("NAV calculation failed: %s", e)

        try:
            self._audit.log(AuditAction.ENGINE_STOP, "engine",
                            details={"trades": self._trade_count, "cycles": self._cycle_count})
        except Exception as exc:
            logger.error("ENGINE_STOP audit failed: %s", exc)

        try:
            self._bus.publish("engine.stopped", {
                "session_id": self._session_id,
                "trades": self._trade_count,
                "cycles": self._cycle_count,
            }, source="engine")
        except Exception:
            pass

        try:
            self._sm.transition(PortfolioState.READY, "post-market complete")
        except Exception as exc:
            logger.error("State transition back to READY failed: %s", exc)

        # S16: flush the WAL after the final writes so a subsequent stop/start
        # (or a container restart) sees a consolidated database file.
        try:
            self._store.checkpoint()
        except Exception as exc:
            logger.error("Store checkpoint during engine stop failed: %s", exc)

        logger.info("Engine stopped. Trades: %d, Cycles: %d", self._trade_count, self._cycle_count)

    def run_once(self) -> CycleResult:
        return self._execute_cycle()

    def cycles_outside_session(self) -> bool:
        """Whether the loop trades outside exchange hours (paper mode).

        Live brokers skip cycles when the market is closed; readiness must not
        treat that deliberate idle time as a dead engine (S4).
        """
        return isinstance(self._broker, SimulatedBroker)

    def get_state(self) -> dict:
        account = self._broker.get_account()
        return {
            "status": "running" if self._running else "idle",
            "session_id": self._session_id,
            "started_at": self._started_at,
            "last_update": self._last_update,
            "n_cycles": self._cycle_count,
            "total_trades": self._trade_count,
            "total_signals": self._signal_count,
            "portfolio_value": round(account.get("total_equity", 0), 2),
            "cash": round(account.get("cash", 0), 2),
            "n_positions": account.get("n_positions", 0),
            "total_pnl": round(account.get("total_pnl", 0), 2),
            "total_pnl_pct": round(account.get("total_pnl_pct", 0), 4),
            "last_prices_update": self._last_prices_update,
            "universe_size": len(self._universe),
            "state_machine": self._sm.state_str,
            "market_status": self._scheduler.get_market_session().status,
            "scheduler": self._scheduler.get_state(),
            "risk": self._risk.get_status(),
            # Run-report provenance: did this engine apply a corporate-action
            # feed, from where, and how many events (with the latest ex-date)?
            "corporate_actions_applied": self.get_corporate_actions_applied(),
            "corporate_actions_feed": {
                "loaded": bool(self._corporate_actions_loaded),
                "rows": int(self._corporate_action_feed_rows),
                "source": self._corporate_actions_source,
                "broker_supports_local": self._broker_supports_corporate_actions(),
            },
        }

    def get_positions(self) -> list[dict]:
        return [p.to_dict() for p in self._broker.get_positions()]

    def get_account(self) -> dict:
        return self._broker.get_account()

    def get_recent_cycles(self, n: int = 10) -> list[dict]:
        return [c.to_dict() for c in self._cycle_log[-n:]]

    def manual_order(self, code: str, side: str, quantity: int, price: float) -> dict:
        if price <= 0:
            return {"order_id": "", "status": OrderStatus.REJECTED.value,
                    "error_msg": "Price must be positive"}
        # D5: a persisted kill switch must block manual orders too. Only the
        # API route used to guard manual orders, so a direct/embedded caller
        # could trade straight through an active kill.
        if getattr(self._risk, "kill_switch_active", False):
            return {"order_id": "", "status": OrderStatus.REJECTED.value,
                    "error_msg": "Kill switch active; orders blocked until deactivated"}
        order = Order(code=code, side=OrderSide(side), quantity=quantity, price=price)
        order_dict = {"ticker": order.code, "side": order.side.value,
                      "quantity": order.quantity, "price": order.price}
        approved, breaches = self._risk.check_pre_trade(order_dict)
        if not approved:
            message = "; ".join(b.message for b in breaches)
            self._audit.log_risk_breach(
                "pre_trade_block",
                {"code": order.code, "side": order.side.value,
                 "breaches": message},
                severity="warning",
            )
            self._bus.publish("risk.order_blocked", {
                "code": order.code, "breaches": message,
            }, source="risk")
            return {"order_id": order.order_id, "status": OrderStatus.REJECTED.value,
                    "error_msg": f"Risk check failed: {message}"}
        # R13: keep the order marked in flight until its synchronous fill is
        # persisted; the cycle's fill sync runs in another thread and must not
        # book the same execution as a delta in the meantime.
        with self._fills_in_flight_lock:
            self._fills_in_flight += 1
        try:
            result = self._broker.place_order(order)
            # A manual fill is a real broker execution: persist order+trade
            # and count only actual fills. The old code logged MANUAL_ORDER
            # (no trade row) and incremented `_trade_count` even for
            # rejections.
            filled = (
                result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
                and self._filled_quantity(result) > 0
            )
            if filled:
                self._trade_count += 1
                self._persist_fill_result(result, reason="manual")
                try:
                    self._audit.log(AuditAction.MANUAL_ORDER, "engine",
                                    details=result.to_dict(), reason="manual")
                except Exception as exc:
                    logger.error("Manual-order audit write failed: %s", exc)
            elif result.status == OrderStatus.REJECTED:
                self._persist_order_event(
                    result, AuditAction.ORDER_REJECTED, result.error_msg,
                )
            else:
                self._persist_order_event(
                    result, AuditAction.ORDER_SUBMITTED, "manual",
                )
        finally:
            with self._fills_in_flight_lock:
                self._fills_in_flight -= 1
        # Off-cycle fills must still reach the persisted snapshot. A filled
        # sell may legitimately leave the broker flat while the store still
        # has the (about to be pruned) row.
        self._persist_engine_state(
            trust_flat=filled and order.side == OrderSide.SELL,
        )
        return result.to_dict()

    def _run_loop(self, stop_event: threading.Event | None = None):
        # `stop_event` is the event of *this* run; a zero-arg override (or an
        # internal caller) falls back to the current run's event. start()
        # refuses to begin a new run while an old loop is alive, so the
        # fallback cannot resurrect a stale loop (D14).
        event = stop_event if stop_event is not None else self._stop_event
        if event is None:
            event = threading.Event()
        while not event.is_set():
            # S4 dead-man switch: beat once per loop iteration, before the
            # market gate. Live brokers correctly idle outside the session,
            # but a hung/dead loop must still be visible to /api/readiness.
            self._record_cycle_heartbeat(None)
            try:
                # Paper trading runs continuously; live brokers only trade in market hours
                if self._scheduler.is_market_open() or isinstance(self._broker, SimulatedBroker):
                    cycle = self._execute_cycle()
                    self._cycle_log.append(cycle)
                    if len(self._cycle_log) > 100:
                        self._cycle_log = self._cycle_log[-50:]
                else:
                    # R13: actual fills (closing auction, late callbacks) can
                    # land after the last cycle; the loop keeps idling after
                    # the close, so book them here instead of waiting for
                    # stop() -- otherwise the day's EOD could run with a
                    # stale ledger.
                    self._sync_broker_fills()
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
                try:
                    self._bus.publish("system.error", {"error": str(e)}, source="engine")
                except Exception:
                    pass
                # Dedupe by exception type: an error whose message varies
                # (addresses, ids) would otherwise bypass the sink's
                # five-minute dedupe and POST a webhook every cycle.
                emit_alert(
                    "error",
                    f"engine cycle error: {type(e).__name__}",
                    source="engine", details={"error": str(e)},
                )

            event.wait(timeout=self._rebalance_interval)

    def _deleverage(self, cycle: CycleResult) -> None:
        """Sell every available share while the kill switch is active.

        Only `pos.available` is sold, so T+1-locked shares bought today are
        left for the next cycle. De-risking sells are exempt from the kill
        switch by design (`RiskMonitor.check_pre_trade`), which is exactly
        the behaviour this method implements.
        """
        try:
            positions = self._broker.get_positions()
        except Exception as exc:
            logger.error("De-risking skipped: position query failed: %s", exc)
            return
        if positions is None:
            logger.error("De-risking skipped: position book unknown")
            return
        orders = self._orders_for_signal_generation()
        if orders is None:
            logger.error("De-risking skipped: order book unknown")
            return
        working_sells = working_order_quantities(
            orders, side=OrderSide.SELL,
            include_partial=not isinstance(self._broker, SimulatedBroker),
        )
        for pos in positions:
            quantity = max(
                0, int(pos.available)
                - working_sells.get(normalize_symbol(pos.code), 0),
            )
            if quantity <= 0:
                continue
            # Only price from a fresh quote; otherwise send a market order and
            # let the broker use its last *fresh* reference (D11).
            price = self._current_prices.get(pos.code, 0.0)
            if not self._quote_is_fresh(pos.code):
                price = 0.0
            order = Order(
                code=pos.code,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=price if price > 0 else 0.0,
            )
            try:
                result = self._broker.place_order(order)
            except Exception as exc:
                logger.error("De-risking sell failed for %s: %s", pos.code, exc)
                continue

            cycle.orders.append(result.to_dict())
            if (result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
                    and self._filled_quantity(result) > 0):
                self._trade_count += 1
                self._persist_fill_result(result, reason="kill switch de-risking")
                self._bus.publish("order.filled", result.to_dict(), source="broker")
                if isinstance(self._risk, RealTimeRiskEngine):
                    self._risk.on_fill({
                        "symbol": order.code,
                        "side": order.side.value,
                        "price": result.filled_price,
                        "quantity": self._filled_quantity(result),
                    })
            elif result.status == OrderStatus.REJECTED:
                # A rejected de-risking sell used to be audited as
                # ORDER_SUBMITTED, hiding a failed exit from compliance (D2).
                self._persist_order_event(
                    result, AuditAction.ORDER_REJECTED,
                    reason=f"kill switch de-risking: {result.error_msg}",
                )
            else:
                self._persist_order_event(
                    result, AuditAction.ORDER_SUBMITTED,
                    reason="kill switch de-risking",
                )

    @staticmethod
    def _has_filled_sell(orders: list[dict]) -> bool:
        """Whether this cycle executed any sell quantity (explains flat)."""
        for order in orders or []:
            if str(order.get("side", "")).lower() not in ("sell", "s"):
                continue
            try:
                if int(order.get("filled_quantity", 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _halt(self, reason: str) -> None:
        """Move the state machine to HALTED.

        HALTED was unreachable from production code -- the only transitions to
        it lived in tests. The cycle moves the machine to REBALANCING on entry,
        and the kill-switch early return then skipped the transition back at
        the end of the cycle, so the state the engine reported stayed at
        REBALANCING for the life of the process while the kill switch was on.
        """
        if self._sm.state != PortfolioState.HALTED:
            if self._sm.transition(PortfolioState.HALTED, reason):
                # S4: a halted engine is an operational emergency; route it
                # through the alert sink (deduplicated for five minutes).
                emit_alert(
                    "critical", f"engine halted: {reason}", source="engine",
                    details={"session_id": self._session_id},
                )

    # ── Exchange clock and quote freshness (D11/D12) ──

    def _exchange_now(self) -> datetime:
        """Now in exchange local time; overridable in tests for D12."""
        return exchange_now()

    def _exchange_trading_date(self) -> str:
        return self._exchange_now().strftime("%Y-%m-%d")

    def _aggregate_quote_age(self) -> float | None:
        """Age in seconds of the last successful price fetch, or None.

        Prefers the monotonic stamp (immune to host clock jumps); falls back
        to the ISO field for embedded callers/tests that set it directly.
        """
        if self._last_prices_update_monotonic:
            return time.monotonic() - self._last_prices_update_monotonic
        if not self._last_prices_update:
            return None
        try:
            return (
                datetime.now() - datetime.fromisoformat(self._last_prices_update)
            ).total_seconds()
        except ValueError:
            return None

    def _quote_is_fresh(self, code: str, now: float | None = None) -> bool:
        """Whether `code` has a quote younger than the staleness limit.

        D11: the old aggregate `_last_prices_update` treated a mix of fresh
        and dead feeds as fresh; an order for a symbol whose feed died hours
        ago was still sized and executed at the last (stale) price.

        A symbol that was not refreshed by the latest fetch is stale even if
        its last quote is technically young: the feed that carries it is
        dead. Symbols with no per-symbol data at all (prices injected by an
        embedded caller without any fetch) fall back to the aggregate check.
        """
        if self._last_fetch_codes and code not in self._last_fetch_codes:
            return False
        stamp = self._price_last_seen.get(code)
        if stamp is not None:
            age = (now if now is not None else time.monotonic()) - stamp
            return age <= self._max_quote_age_seconds
        aggregate_age = self._aggregate_quote_age()
        if aggregate_age is None:
            return False
        return aggregate_age <= self._max_quote_age_seconds

    # ── Position book (D4/D6) ──

    def _query_positions(self, trust_flat: bool = False) -> tuple[list | None, bool]:
        """Query the broker book, distinguishing "flat" from "unknown".

        Returns ``(positions, known)``. `known=False` means the broker could
        not tell us: the call raised, the adapter returned None (QMT reports
        None when its own position query fails), or it reported an empty book
        while the Store still records holdings. Callers must skip sizing and
        pruning in that case.

        `trust_flat=True` is only for callers that *know* the book can be
        empty now -- e.g. the cycle just executed a filled sell, or a fresh
        process is being reconciled against the store. Even then the empty
        answer is re-confirmed once. A plain transient `[]` (QMT converts a
        failed query into an empty list) used to make the engine size a
        duplicate stake and erase the persisted book (D6).
        """
        try:
            positions = self._broker.get_positions()
        except Exception as exc:
            logger.error("Position query failed: %s", exc)
            return None, False
        if positions is None:
            logger.error("Position book unknown: broker returned None")
            return None, False
        positions = list(positions)
        if positions:
            return positions, True

        try:
            held_rows = self._store.get_positions(self._tenant_id)
        except Exception:
            held_rows = []
        if not held_rows:
            return [], True

        if not trust_flat:
            logger.error(
                "Broker reports a flat book while the store records %d held "
                "code(s); treating the book as UNKNOWN for this cycle "
                "(no sizing, no pruning)", len(held_rows),
            )
            return None, False

        try:
            confirmed = self._broker.get_positions()
        except Exception as exc:
            logger.error("Position confirmation query failed: %s", exc)
            return None, False
        if confirmed is None:
            return None, False
        confirmed = list(confirmed)
        if not confirmed:
            logger.warning(
                "Broker reports a flat book while the store records %d held "
                "code(s); confirmed flat after a filled liquidation",
                len(held_rows),
            )
        return confirmed, True

    def _sync_store_book(self, positions: list) -> None:
        """Upsert the broker book into the Store and prune closed rows.

        Only ever called with a *known* book. Deleting rows from an unknown
        query was how one transient empty answer erased the persisted
        positions (D6).

        The whole book is written with one bulk transaction (and the stale
        rows pruned in the same transaction). This used to commit once per
        position, and `_persist_engine_state` plus the end-of-cycle sync means
        the book is persisted at least twice per cycle.
        """
        rows = []
        for pos in positions or []:
            row = self._position_row(pos)
            if not str(row.get("code") or ""):
                continue
            row["tenant_id"] = self._tenant_id
            rows.append(row)
        self._store.save_positions(
            rows, tenant_id=self._tenant_id, prune_missing=True,
        )

    @staticmethod
    def _position_row(pos) -> dict:
        """Normalize a broker Position / dict / test stub into a store row."""
        if hasattr(pos, "to_dict"):
            return pos.to_dict()
        if isinstance(pos, dict):
            return dict(pos)
        return {
            "code": str(getattr(pos, "code", "") or ""),
            "name": str(getattr(pos, "name", "") or ""),
            "quantity": int(getattr(pos, "quantity", 0) or 0),
            "available": int(getattr(pos, "available", 0) or 0),
            "avg_cost": float(getattr(pos, "avg_cost", 0.0) or 0.0),
            "current_price": float(getattr(pos, "current_price", 0.0) or 0.0),
            "market_value": float(getattr(pos, "market_value", 0.0) or 0.0),
            "unrealized_pnl": float(getattr(pos, "unrealized_pnl", 0.0) or 0.0),
            "realized_pnl": float(getattr(pos, "realized_pnl", 0.0) or 0.0),
        }

    @staticmethod
    def _filled_quantity(result) -> int:
        """Filled quantity of a broker result, tolerant of minimal DTOs."""
        value = getattr(result, "filled_quantity", None)
        if value is None:
            try:
                value = result.to_dict().get("filled_quantity", 0)
            except Exception:
                value = 0
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    # ── Persistence helpers (D2/D3/D5) ──

    def _audit_event(self, action: AuditAction, payload: dict, reason: str = "") -> None:
        """Write an audit event, retrying once; never raises.

        The broker has already executed by the time this runs, so a broken
        audit sink must not abort the cycle.
        """
        for attempt in (1, 2):
            try:
                self._audit.log(action, "broker", details=payload, reason=reason)
                return
            except Exception as exc:
                logger.error(
                    "Audit write failed (attempt %d, %s): %s",
                    attempt, getattr(action, "value", action), exc,
                )

    def _persist_fill_result(self, result, reason: str = "") -> None:
        """Persist a fill as an order + trade row, then audit it (D2/D3).

        The broker execution is the source of truth. The order and trade are
        written atomically *before* the best-effort audit event, and the
        trade gets a deterministic id so a retry cannot double-book. The old
        path audited first: one `log_order` exception lost the fill
        permanently (store orders=0, trades=0 after a real broker fill).
        """
        payload = result.to_dict()
        # BrokerOrder.to_dict() omits the wire id; persist it so the R13 fill
        # sync can recognise this order after a restart regenerates the local
        # order_id (QMT rebuilds recovered orders with a fresh uuid).
        broker_order_id = str(getattr(result, "broker_order_id", "") or "")
        if broker_order_id:
            payload["broker_order_id"] = broker_order_id
        filled_quantity = int(payload.get("filled_quantity") or 0)
        trade = {
            "trade_id": f"{payload['order_id']}:fill",
            "tenant_id": self._tenant_id,
            "order_id": payload["order_id"],
            "code": payload["code"],
            "side": payload["side"],
            "quantity": filled_quantity,
            "price": payload.get("filled_price") or payload.get("price") or 0.0,
            "commission": payload.get("commission", 0),
            "tax": payload.get("tax", 0),
            "executed_at": (
                payload.get("updated_at")
                or self._exchange_now().replace(tzinfo=None).isoformat()
            ),
        }
        for attempt in (1, 2):
            try:
                self._store.save_order_with_trade(payload, trade)
                break
            except Exception as exc:
                logger.error(
                    "Failed to persist fill %s (attempt %d): %s",
                    payload.get("order_id"), attempt, exc,
                )
        else:
            logger.critical(
                "Fill %s could not be persisted; ledger will drift from broker",
                payload.get("order_id"),
            )
            return
        self._audit_event(AuditAction.ORDER_FILLED, payload, reason=reason)

    def _persist_order_event(self, result, action: AuditAction, reason: str = "") -> None:
        """Persist a non-fill order outcome (rejected/submitted) and audit it."""
        payload = result.to_dict()
        broker_order_id = str(getattr(result, "broker_order_id", "") or "")
        if broker_order_id:
            payload["broker_order_id"] = broker_order_id
        for attempt in (1, 2):
            try:
                self._store.save_order(payload)
                break
            except Exception as exc:
                logger.error(
                    "Failed to persist order %s (attempt %d): %s",
                    payload.get("order_id"), attempt, exc,
                )
        self._audit_event(action, payload, reason=reason)

    def _sync_broker_fills(self, executed_at: str | None = None) -> int:
        """Book asynchronous broker fills into the trade ledger (R13).

        ``place_order`` only persists fills it observes synchronously
        (``FILLED``/``PARTIAL``). QMT/XTP report most executions through
        callbacks/order resyncs after returning ``SUBMITTED``, so without
        this pass those fills never reach the ``trades`` table and the R12
        EOD trade ledger fails loudly against a stale baseline.

        Each order's *delta* is ``filled_quantity`` from the broker minus the
        quantity already persisted for that ``order_id``. The persisted side
        is summed from ``trades`` (``get_trade_quantity_for_order``), so a
        replay of the same broker snapshot -- or a fill already written by
        the synchronous ``{order_id}:fill`` path -- yields ``delta <= 0`` and
        writes nothing. A partial order that grows from 40 to 100 produces
        ``{order_id}:sync:40`` then ``{order_id}:sync:100`` (60 shares).

        Fees: a synced fill records ``commission = tax = 0``. The broker
        order object carries at most an order-level total that cannot be
        attributed to one incremental print, and a pro-rata guess would
        fabricate cash movements. The EOD cash reconciliation will surface
        the difference as a mismatch until the broker's fee report is booked;
        the fee gap is loud, never silent.

        Price basis: ``filled_price`` is the broker's *cumulative* average
        fill price. Booking a delta at that average can drift from the actual
        per-print cash beyond fees (40 shares at 10.0 followed by 60 more at
        a cumulative 10.2 books the second print at 10.2, not its true
        print); EOD cash shows it as a mismatch, never silently absorbed.

        Events: ``order.filled`` is deliberately *not* published -- its
        consumers treat the payload as the newly executed quantity, so
        republishing a cumulative order state would double count. The new
        ``order.synced`` topic carries the delta instead.

        A missing/None/erroring broker book is a no-op (debug/warning): a
        fill-sync failure must not take down the trading cycle. Returns the
        number of trade rows written.

        ``executed_at`` overrides the per-order stamp for every trade this
        call writes. The startup catch-up passes the process's pre-baseline
        clock so those fills sort strictly before the R12 ``ledger_baselines``
        row that bakes them in (see ``start``); per-cycle syncs leave it
        unset and use the broker's ``updated_at``.

        ``_fill_sync_confirmed`` reports whether the broker order book was
        actually readable and ``_fill_sync_blocked`` whether a filled order
        could not be booked (and might be retried later); ``start`` uses both
        to decide whether a baseline may be written.
        """
        with self._fill_sync_lock:
            return self._sync_broker_fills_locked(executed_at=executed_at)

    def _sync_broker_fills_locked(self, executed_at: str | None = None) -> int:
        self._fill_sync_confirmed = False
        self._fill_sync_blocked = False
        with self._fills_in_flight_lock:
            manual_fill_in_flight = self._fills_in_flight > 0
        if manual_fill_in_flight:
            logger.debug(
                "Broker fill sync deferred: a manual fill is being persisted",
            )
            return 0

        # QMT's get_orders() degrades to its in-memory cache -- an empty list
        # on a fresh process -- when the broker query fails. Ask the adapter's
        # raw resync path for the real outcome when it exposes one; adapters
        # without it (paper/test) are trusted through get_orders(). Only the
        # startup call needs the confirmation, so cycles skip the extra query.
        resync_probe = getattr(self._broker, "_resync_from_broker", None)
        probe_ok = True
        if executed_at is not None and callable(resync_probe):
            try:
                probe_ok = bool(resync_probe())
            except Exception as exc:
                logger.warning(
                    "Broker fill sync: order-book resync probe failed: %s", exc,
                )
                probe_ok = False
        try:
            orders = self._broker.get_orders()
        except Exception as exc:
            logger.warning("Broker fill sync skipped: get_orders failed: %s", exc)
            return 0
        if orders is None:
            logger.debug("Broker fill sync skipped: broker order book unknown")
            return 0
        try:
            order_list = list(orders)
        except Exception as exc:
            logger.warning(
                "Broker fill sync skipped: get_orders returned %s (%s)",
                type(orders).__name__, exc,
            )
            return 0

        self._fill_sync_confirmed = probe_ok
        if not probe_ok:
            logger.warning(
                "Broker fill sync: the broker order query failed; treating "
                "the snapshot as unconfirmed",
            )
        written = 0
        for order in order_list:
            try:
                written += self._sync_one_broker_order(
                    order, executed_at=executed_at,
                )
            except Exception as exc:
                logger.warning(
                    "Broker fill sync skipped order %r: %s",
                    getattr(order, "order_id", ""), exc,
                )
                # A retried success after the baseline would double count.
                self._fill_sync_blocked = True
        return written

    def _sync_one_broker_order(self, order, executed_at: str | None = None) -> int:
        """Book one broker order's fill delta; returns trades written (0/1)."""
        try:
            filled = int(getattr(order, "filled_quantity", 0) or 0)
        except (TypeError, ValueError):
            logger.warning(
                "Broker fill sync: unreadable filled_quantity on order %r",
                getattr(order, "order_id", ""),
            )
            # An unknown fill may be retried (or corrected) later; at startup
            # that means the baseline must be withheld.
            self._fill_sync_blocked = True
            return 0
        if filled <= 0:
            return 0

        order_id = str(getattr(order, "order_id", "") or "")
        code = str(getattr(order, "code", "") or "")
        if not order_id or not code:
            logger.warning(
                "Broker fill sync: order without order_id/code skipped (%r)",
                getattr(order, "order_id", ""),
            )
            self._fill_sync_blocked = True
            return 0

        booked = self._booked_quantity_for_order(order, order_id)
        if booked is None:
            # Never guess a delta from a failed query. Skipping is safe for
            # the next sync, but at startup an unbooked fill could still land
            # after the baseline, so the caller must withhold the baseline.
            self._fill_sync_blocked = True
            return 0
        delta = filled - int(booked or 0)
        if delta <= 0:
            return 0

        side = getattr(order, "side", "")
        side_value = str(getattr(side, "value", side) or "")
        if side_value not in ("buy", "sell"):
            logger.warning(
                "Broker fill sync: order %s has unsupported side %r; skipped",
                order_id, side_value,
            )
            self._fill_sync_blocked = True
            return 0
        status = getattr(order, "status", "")
        status_value = str(getattr(status, "value", status) or "")
        try:
            filled_price = float(getattr(order, "filled_price", 0) or 0)
        except (TypeError, ValueError):
            filled_price = 0.0
        try:
            limit_price = float(getattr(order, "price", 0) or 0)
        except (TypeError, ValueError):
            limit_price = 0.0
        price = filled_price if filled_price > 0 else limit_price
        if price <= 0:
            logger.warning(
                "Broker fill sync: order %s has no usable fill price; "
                "booking quantity at 0 (cash reconciliation will flag it)",
                order_id,
            )
        try:
            order_quantity = int(getattr(order, "quantity", 0) or 0)
        except (TypeError, ValueError):
            order_quantity = 0
        updated_at = str(getattr(order, "updated_at", "") or "")
        trade_executed_at = (
            executed_at
            or updated_at
            or self._exchange_now().replace(tzinfo=None).isoformat()
        )

        trade = {
            "trade_id": f"{order_id}:sync:{filled}",
            "tenant_id": self._tenant_id,
            "order_id": order_id,
            "code": code,
            "side": side_value,
            "quantity": delta,
            "price": price,
            # Fees cannot be split per incremental print; see the method
            # docstring. Never fabricate a commission/tax here.
            "commission": 0,
            "tax": 0,
            "executed_at": trade_executed_at,
        }
        order_row = {
            "order_id": order_id,
            "tenant_id": self._tenant_id,
            "code": code,
            "side": side_value,
            "quantity": order_quantity,
            "price": limit_price,
            "filled_quantity": filled,
            "filled_price": price,
            "commission": float(getattr(order, "commission", 0) or 0),
            "tax": float(getattr(order, "tax", 0) or 0),
            "slippage": float(getattr(order, "slippage", 0) or 0),
            "broker_order_id": str(getattr(order, "broker_order_id", "") or ""),
            "error_msg": str(getattr(order, "error_msg", "") or ""),
            "updated_at": trade_executed_at,
        }
        # Never overwrite an existing row's creation time with an empty
        # value just because a minimal DTO omitted it.
        created_at = str(getattr(order, "created_at", "") or "")
        if created_at:
            order_row["created_at"] = created_at
        if status_value:
            order_row["status"] = status_value
        for attempt in (1, 2):
            try:
                # One transaction: the order row and its trade cannot
                # disagree, and a retry after a failed write re-uses the
                # same deterministic trade id (a committed write makes the
                # duplicate impossible; see _persist_fill_result).
                self._store.save_order_with_trade(order_row, trade)
                break
            except Exception as exc:
                logger.error(
                    "Failed to persist synced fill %s (attempt %d): %s",
                    order_id, attempt, exc,
                )
        else:
            # The store may have committed the write and then raised (e.g. a
            # commit error): re-read before declaring drift. A present trade
            # means the fill is booked, and the retry's duplicate-key failure
            # was the wrong signal. Re-read through the cross-restart-aware
            # helper: the previous partial may sit under an older local id.
            try:
                present = self._booked_quantity_for_order(order, order_id)
            except Exception:
                present = None
            if present is None or int(present) < filled:
                logger.critical(
                    "Synced fill %s (%d shares) could not be persisted; the "
                    "ledger will drift from the broker", order_id, delta,
                )
                self._fill_sync_blocked = True
                return 0
            logger.warning(
                "Synced fill %s reported a write error but is present in the "
                "ledger; treating it as booked", order_id,
            )

        # Compliance: a synced fill is a real execution and belongs in the
        # audit trail. `order.filled` stays unpublished; this is the audit
        # sink, which consumers read as one fill of `quantity` shares.
        self._audit_event(AuditAction.ORDER_FILLED, {
            "order_id": order_id,
            "code": code,
            "side": side_value,
            "quantity": delta,
            "price": price,
            "filled_quantity": filled,
            "trade_id": trade["trade_id"],
            "source": "broker_fill_sync",
        }, reason="broker fill sync")

        try:
            self._bus.publish("order.synced", {
                "order_id": order_id,
                "code": code,
                "side": side_value,
                "delta": delta,
                "price": price,
            }, source="broker")
            if isinstance(self._risk, RealTimeRiskEngine):
                self._risk.on_fill({
                    "symbol": code,
                    "side": side_value,
                    "price": price,
                    "quantity": delta,
                })
        except Exception as exc:
            # The ledger write already succeeded; a broken subscriber must
            # not abort the trading cycle (the position book is re-synced
            # from the broker every cycle regardless).
            logger.warning(
                "Synced fill %s notification failed: %s", order_id, exc,
            )
        return 1

    def _booked_quantity_for_order(self, order, order_id: str) -> int | None:
        """Quantity already booked for this broker order, across restarts.

        The local ``order_id`` is authoritative within one process. QMT's
        resync rebuilds orders it did not watch being submitted (i.e. orders
        placed by a previous process) with a fresh uuid, so a restart would
        otherwise re-book an already persisted fill under the new id. Order
        rows written by the sync carry ``broker_order_id``; summing the
        trades of every row with the same broker id keeps the delta
        idempotent across restarts without touching the broker adapter.

        Returns ``None`` when a query fails -- the caller must not guess a
        delta from a half-known ledger.
        """
        try:
            booked = self._store.get_trade_quantity_for_order(
                order_id, self._tenant_id,
            )
        except Exception as exc:
            logger.warning(
                "Broker fill sync: persisted quantity query failed for %s: %s",
                order_id, exc,
            )
            return None
        broker_order_id = str(getattr(order, "broker_order_id", "") or "")
        if not broker_order_id:
            return booked
        code = str(getattr(order, "code", "") or "")
        side = getattr(order, "side", "")
        side_value = str(getattr(side, "value", side) or "")
        try:
            # A broker's numeric wire ids are reused across days; scope the
            # match to the same asset/side and to order rows created since
            # the *current exchange trading day began* (converted to the
            # host-local clock the adapters stamp `created_at` with -- a
            # plain host-date cutoff would drop orders placed before the
            # host's midnight on a non-Shanghai host).
            booked += self._store.get_trade_quantity_for_broker_order(
                broker_order_id,
                tenant_id=self._tenant_id,
                code=code,
                side=side_value,
                since=self._exchange_day_start_host_iso(),
                exclude_order_id=order_id,
            )
        except Exception as exc:
            logger.warning(
                "Broker fill sync: broker-id dedupe query failed for %s "
                "(broker order %s): %s", order_id, broker_order_id, exc,
            )
            return None
        return booked

    def _exchange_day_start_host_iso(self) -> str:
        """Start of the current exchange trading day on the host clock.

        Order rows are stamped with host-local ``created_at``; the current
        exchange day starts at 00:00 Shanghai, which can fall on the
        previous host-local date (US/Europe). Converting the instant -- not
        the date string -- keeps the cross-restart dedupe correct on hosts
        outside Asia/Shanghai.
        """
        try:
            day = datetime.strptime(
                self._exchange_trading_date(), "%Y-%m-%d",
            ).replace(tzinfo=EXCHANGE_TZ)
            return day.astimezone().replace(tzinfo=None).isoformat()
        except Exception:
            # Conservative fallback: include the previous host day.
            return (
                datetime.now() - timedelta(days=1)
            ).date().isoformat()

    def _unbooked_broker_fill_deltas(self) -> dict[str, int] | None:
        """Net signed shares the broker reports but the Store has not booked.

        Startup uses this to decide whether a broker/store position
        difference is *fully explained* by fills that happened while the
        engine was down (R13). ``None`` means the broker order book cannot be
        read; callers keep the old fail-closed behaviour rather than guess.
        """
        try:
            orders = self._broker.get_orders()
        except Exception as exc:
            logger.warning(
                "Startup fill explanation skipped: get_orders failed: %s", exc,
            )
            return None
        if orders is None:
            logger.debug(
                "Startup fill explanation skipped: broker order book unknown",
            )
            return None
        try:
            order_list = list(orders)
        except Exception as exc:
            logger.warning(
                "Startup fill explanation skipped: broker order book could "
                "not be iterated: %s", exc,
            )
            return None
        deltas: dict[str, int] = {}
        for order in order_list:
            try:
                filled = int(getattr(order, "filled_quantity", 0) or 0)
            except (TypeError, ValueError):
                continue
            if filled <= 0:
                continue
            order_id = str(getattr(order, "order_id", "") or "")
            code = str(getattr(order, "code", "") or "")
            side = getattr(order, "side", "")
            side_value = str(getattr(side, "value", side) or "")
            if not order_id or not code or side_value not in ("buy", "sell"):
                continue
            booked = self._booked_quantity_for_order(order, order_id)
            if booked is None:
                return None
            delta = filled - int(booked or 0)
            if delta <= 0:
                continue
            sign = 1 if side_value == "buy" else -1
            deltas[code] = deltas.get(code, 0) + sign * delta
        return deltas

    def _persist_engine_state(self, trust_flat: bool = False) -> None:
        """Persist the restart-critical snapshot (D5).

        kill switch, peak equity, day baseline, exchange trading day, cash and
        the position book. Without this a restarted engine silently resumed
        trading with the kill switch off and the account reset to the
        broker's initial notional.

        `trust_flat` is forwarded to the position query: pass True only when
        the caller knows a filled sell explains an empty broker book.
        """
        try:
            account = self._broker.get_account() or {}
        except Exception as exc:
            logger.error("Engine-state persist: account query failed: %s", exc)
            account = {}
        if not isinstance(account, dict):
            account = {}
        if account.get("connected") is False:
            # QMT/XTP return an initial-cash placeholder when disconnected
            # (trading/broker.py). Persisting it would overwrite the last real
            # cash/equity and poison the scheduler's EOD equity source (S2).
            logger.error(
                "Engine-state persist: broker reports disconnected; keeping "
                "the prior persisted cash/equity",
            )
            account = {}

        try:
            prior_state = self._store.get_engine_state(self._engine_state_key) or {}
        except Exception:
            prior_state = {}
        if not isinstance(prior_state, dict):
            prior_state = {}

        positions, known = self._query_positions(trust_flat=trust_flat)
        if known:
            try:
                self._sync_store_book(positions or [])
            except Exception as exc:
                logger.error("Engine-state persist: position sync failed: %s", exc)
        # else: keep the last known book from `prior_state` rather than
        # replacing it with a list derived from an unknown query.

        risk = self._risk
        if isinstance(risk, RiskMonitor):
            peak = float(risk.peak_value or 0.0)
        elif isinstance(risk, RealTimeRiskEngine):
            peak = float(getattr(risk, "_peak_equity", 0.0) or 0.0)
        else:
            peak = 0.0
        state = {
            "version": 1,
            "tenant_id": self._tenant_id,
            "session_id": self._session_id,
            "kill_switch_active": bool(getattr(risk, "kill_switch_active", False)),
            "peak_equity": peak,
            "day_start_equity": self._day_start_equity,
            "trading_day": self._trading_day,
            "cash": account.get("cash", prior_state.get("cash", 0.0)),
            "portfolio_value": account.get(
                "total_equity", prior_state.get("portfolio_value", 0.0),
            ),
            "positions": (
                [dict(self._position_row(p), tenant_id=self._tenant_id)
                 for p in (positions or [])]
                if known else prior_state.get("positions", [])
            ),
            "trade_count": self._trade_count,
            "cycle_count": self._cycle_count,
            "updated_at": self._exchange_now().replace(tzinfo=None).isoformat(),
        }
        try:
            self._store.save_engine_state(state, self._engine_state_key)
        except Exception as exc:
            logger.error("Failed to persist engine state: %s", exc)

    def _restore_state(self) -> None:
        """Restore the persisted snapshot and reconcile store <-> broker.

        D4: a fresh process used to be blocked forever by the health check's
        position-sync comparison whenever a position was open at shutdown.
        The broker is reconciled explicitly here, before that check runs:

        * broker fresh + store book -> restore the store book into the broker
          (only adapters that implement `restore_positions`, i.e. paper);
        * broker book + empty store -> adopt the broker book as truth;
        * both non-empty and disagreeing -> refuse to start (SystemBlockError),
          unless the difference is *exactly* the broker's unbooked fills
          (R13: an order that filled while the engine was down). In that case
          the broker book is adopted and the startup fill sync books the
          fills immediately afterwards. A flat broker book explained by
          unbooked sells is adopted the same way;
        * broker unknown -> leave it to the health check to block.

        D5: kill switch, peak, day baseline, trading day and cash are restored
        from `engine_state`; a persisted kill keeps blocking orders until
        explicitly deactivated.
        """
        try:
            state = self._store.get_engine_state(self._engine_state_key)
        except Exception as exc:
            logger.error("Failed to load persisted engine state: %s", exc)
            state = None
        state = state if isinstance(state, dict) else {}

        broker_positions, broker_known = self._query_positions(trust_flat=True)
        try:
            store_rows = self._store.get_positions(self._tenant_id)
        except Exception:
            store_rows = []
        # engine_state carries positions too; fall back to it when the
        # positions table was not written (e.g. manual fills before a crash).
        local_rows = store_rows or state.get("positions") or []

        def _quantities(rows) -> dict[str, int]:
            out: dict[str, int] = {}
            for row in rows or []:
                code = str(row.get("code") or "")
                try:
                    qty = int(row.get("quantity", 0) or 0)
                except (TypeError, ValueError):
                    qty = 0
                if code and qty > 0:
                    out[code] = out.get(code, 0) + qty
            return out

        local_map = _quantities(local_rows)
        broker_map: dict[str, int] = {}
        for pos in broker_positions or []:
            code = str(getattr(pos, "code", "") or "")
            try:
                qty = int(getattr(pos, "quantity", 0) or 0)
            except (TypeError, ValueError):
                qty = 0
            if code and qty > 0:
                broker_map[code] = broker_map.get(code, 0) + qty

        cash = state.get("cash")
        cash = float(cash) if isinstance(cash, (int, float)) else None

        explained_by_fills: bool | None = None

        def _explained_by_fills() -> bool:
            """Whether the book difference is exactly the unbooked fills (R13).

            Fails closed on an unreadable order book: only a complete
            broker-side delta set may explain away a position difference.
            """
            nonlocal explained_by_fills
            if explained_by_fills is None:
                deltas = self._unbooked_broker_fill_deltas()
                if deltas is None:
                    explained_by_fills = False
                else:
                    expected = dict(local_map)
                    for code, delta in deltas.items():
                        expected[code] = expected.get(code, 0) + delta
                    expected = {c: q for c, q in expected.items() if q}
                    explained_by_fills = expected == broker_map
                    if explained_by_fills:
                        logger.warning(
                            "Position book differs from the Store only by "
                            "unbooked broker fills (%s); adopting the broker "
                            "book", deltas,
                        )
            return explained_by_fills

        if not broker_known:
            logger.error(
                "Position book unknown at start; the health check will block "
                "trading until the broker answers",
            )
        elif broker_map and local_map and broker_map != local_map:
            if _explained_by_fills():
                # A fill while the engine was down moved the broker book;
                # the R13 startup sync (which runs right after this
                # reconciler) writes those fills to the trade ledger.
                self._sync_store_book(broker_positions or [])
                cash = None  # never overwrite a live broker's balance
            else:
                raise SystemBlockError(
                    "Position reconciliation failed: broker and store disagree "
                    f"(broker={broker_map}, local={local_map}); refusing to trade"
                )
        elif not broker_map and local_map:
            if _explained_by_fills():
                # Downtime sells (or a full liquidation) explain the flat
                # broker book; adopt it and let the startup sync book them.
                self._sync_store_book([])
                cash = None  # never overwrite a live broker's balance
            else:
                restore = getattr(self._broker, "restore_positions", None)
                if not callable(restore):
                    raise SystemBlockError(
                        "Position reconciliation failed: broker book is empty "
                        f"but the store records {len(local_map)} position(s) "
                        f"and {type(self._broker).__name__} cannot restore them"
                    )
                restore(local_rows, cash)
                logger.warning(
                    "Restored %d position(s) from the store into %s",
                    len(local_map), type(self._broker).__name__,
                )
        elif broker_map and not local_map:
            # First run against an existing broker account: the broker is the
            # source of truth; refusing to start would strand every user who
            # attaches a fresh database to a funded account.
            logger.warning(
                "Store book empty but the broker holds %d position(s); "
                "adopting the broker book", len(broker_map),
            )
            self._sync_store_book(broker_positions or [])
            cash = None  # never overwrite a live broker's balance
        elif not broker_map and not local_map:
            restore = getattr(self._broker, "restore_positions", None)
            if callable(restore) and cash is not None:
                # Flat restart: restore cash so equity/P&L do not silently
                # reset to the initial notional (D5).
                restore([], cash)

        # Restore engine/risk state.
        self._trading_day = str(state.get("trading_day") or "")
        if not self._trading_day and local_map:
            # D7 edge case: a crash before the first engine_state write
            # leaves only the positions table. Infer the exchange day from
            # the last position snapshot so a same-day restart does not
            # release today's T+1 lock (the row's `available` already
            # encodes that lock).
            snapshot_day = max(
                (str(row.get("updated_at") or "")[:10] for row in local_rows),
                default="",
            )
            if snapshot_day and snapshot_day == self._exchange_trading_date():
                self._trading_day = snapshot_day
                logger.warning(
                    "No persisted trading day; inferred %s from the position "
                    "snapshot to keep today's T+1 lock", snapshot_day,
                )
        day_start = state.get("day_start_equity")
        self._day_start_equity = (
            float(day_start) if isinstance(day_start, (int, float)) else None
        )
        self._trade_count = int(state.get("trade_count") or 0)
        self._cycle_count = int(state.get("cycle_count") or 0)
        if isinstance(self._risk, RiskMonitor):
            self._risk.peak_value = max(
                self._risk.peak_value, float(state.get("peak_equity") or 0.0),
            )
            self._risk.portfolio_value = max(
                self._risk.portfolio_value,
                float(state.get("portfolio_value") or 0.0),
            )
        # S21 follow-up: the risk monitor's own persisted state is
        # authoritative for the kill switch. `engine_state` records the flag
        # on every cycle, so a dashboard deactivation performed after the
        # engine stopped would otherwise be undone by the stale engine row on
        # the next start. Only fall back to the engine row when no risk_state
        # record was restored and the switch is not already armed.
        risk_state_authoritative = False
        risk_updated = ""
        has_persisted = getattr(self._risk, "has_persisted_state", None)
        if callable(has_persisted):
            try:
                risk_state_authoritative = bool(has_persisted())
            except Exception:
                risk_state_authoritative = False
        try:
            risk_updated = str(
                self._risk.get_status().get("persisted_at") or ""
            )
        except Exception:
            risk_updated = ""
        engine_updated = str(state.get("updated_at") or "")
        # Fail safe in both directions: if the engine snapshot is newer than
        # the risk row (e.g. the arm's risk_state write failed but a later
        # cycle persisted engine_state), honour the engine flag.
        engine_flag_is_newer = (
            bool(engine_updated) and engine_updated > risk_updated
        )
        if (
            state.get("kill_switch_active")
            and (not risk_state_authoritative or engine_flag_is_newer)
            and not getattr(self._risk, "kill_switch_active", False)
        ):
            activate = getattr(self._risk, "activate_kill_switch", None)
            if callable(activate):
                activate("restored persisted kill switch", actor="engine-restore")
            logger.critical(
                "Persisted kill switch restored; orders remain blocked until "
                "explicitly deactivated",
            )

        # Persist the reconciled snapshot so a crash right after start does
        # not lose the reconciliation.
        self._persist_engine_state()

    def _write_ledger_baseline(self, not_before: datetime | None = None) -> None:
        """Record the broker's book as the trade ledger's baseline (R12).

        Called exactly once per process start, after ``_restore_state`` has
        reconciled broker and Store, so the broker state is trustworthy. The
        EOD expected book is derived from this row plus the fills stored after
        its ``created_at``; a restart appends a new row (fills from before it
        are already baked into the restored broker book, so history is never
        applied twice).

        An unknown broker book (``get_positions() is None``) writes nothing:
        the health check blocks trading, and a baseline built from a failed
        query would be wrong. A missing/non-finite account cash raises --
        recording 0 would make every later cash comparison meaningless.

        ``created_at`` is stamped *after* both broker queries: a fill absorbed
        before the stamp is presumed part of the baseline and excluded from
        the ledger by the ``>=`` boundary, so the only ambiguous window is a
        same-tick fill (~1 ms) rather than the whole broker query latency
        (stamping first would make every fill during the queries eligible for
        double counting).

        ``not_before`` is the startup fill sync's clock (R13): the stamp is
        forced strictly later than it so trades absorbed before the baseline
        sort before the boundary even on a coarse host clock. Without this a
        same-tick startup fill would be counted twice at EOD -- once by the
        baseline book, once by the ledger's ``executed_at >= created_at``
        rule.
        """
        try:
            account = self._broker.get_account()
        except Exception as exc:
            raise RuntimeError(
                f"ledger baseline: broker account query failed: {exc}"
            ) from exc
        if not isinstance(account, dict):
            raise RuntimeError(
                "ledger baseline: broker account payload is not a dict "
                f"({type(account).__name__})"
            )
        if account.get("connected") is False:
            # QMT/XTP serve an initial-cash placeholder when disconnected.
            raise RuntimeError(
                "ledger baseline: broker reports disconnected; refusing to "
                "record placeholder balances"
            )
        cash = account.get("cash")
        try:
            cash_value = float(cash)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ledger baseline: broker account has no usable cash "
                f"({cash!r})"
            ) from exc
        if not math.isfinite(cash_value):
            raise RuntimeError(
                f"ledger baseline: broker cash is not finite ({cash!r})"
            )

        positions, known = self._query_positions(trust_flat=True)
        if not known:
            logger.warning(
                "Ledger baseline not written: broker position book is unknown",
            )
            return

        book: dict[str, dict] = {}
        for pos in positions or []:
            row = self._position_row(pos)
            code = str(row.get("code") or "")
            if not code:
                raise RuntimeError(
                    f"ledger baseline: position row has no asset code: {row!r}"
                )
            try:
                quantity = int(row.get("quantity") or 0)
                avg_cost = float(row.get("avg_cost") or 0.0)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"ledger baseline: malformed position row for {code}: "
                    f"{row!r}"
                ) from exc
            if code in book:
                # Two rows for one asset: net the quantity and keep the first
                # known cost (the ledger only needs the net book).
                book[code]["quantity"] += quantity
                if not book[code]["avg_cost"] and avg_cost:
                    book[code]["avg_cost"] = avg_cost
            else:
                book[code] = {"quantity": quantity, "avg_cost": avg_cost}

        created_at = datetime.now()
        if not_before is not None and created_at <= not_before:
            created_at = not_before + timedelta(microseconds=1)
        self._store.save_ledger_baseline({
            # Same clock as broker fill stamps (`Order.updated_at` is host
            # wall-clock in every adapter): the ledger's `executed_at >=
            # created_at` boundary must compare timestamps from one clock.
            # A Shanghai-labelled baseline on a non-Shanghai host would
            # exclude every later fill and fail the EOD comparison.
            "created_at": created_at.isoformat(),
            "tenant_id": self._tenant_id,
            "engine_state_key": self._engine_state_key,
            "cash": cash_value,
            "positions_json": book,
            "note": f"live engine start (session {self._session_id})",
        })
        logger.info(
            "Ledger baseline written: cash=%.2f, %d position(s)",
            cash_value, len(book),
        )

    def _execute_cycle(self) -> CycleResult:
        """Run one cycle, serialised against the background loop.

        `POST /trading/run-once` used to execute this from the FastAPI thread
        while `_run_loop` executed it independently. Both read the same
        position book and could place the same order twice -- duplicate real
        orders with a live broker, corrupted positions in paper mode.
        """
        if not self._cycle_lock.acquire(blocking=False):
            logger.warning("Cycle already running; skipping duplicate trigger")
            account = self._broker.get_account()
            cycle = CycleResult(
                cycle_id=self._cycle_count,
                timestamp=self._exchange_now().replace(tzinfo=None).isoformat(),
                portfolio_value=account.get("total_equity", 0),
                cash=account.get("cash", 0),
                n_positions=account.get("n_positions", 0),
            )
            self._record_cycle_heartbeat(cycle)
            return cycle
        try:
            cycle = self._execute_cycle_body()
        except Exception:
            self._record_cycle_heartbeat(None)
            raise
        finally:
            self._cycle_lock.release()
        self._record_cycle_heartbeat(cycle)
        return cycle

    def _record_cycle_heartbeat(self, cycle: CycleResult | None) -> None:
        """Export the dead-man-switch gauges consumed by /api/readiness (S4).

        Cheap attribute writes into the in-process metrics collector; no
        threads, no I/O.
        """
        try:
            metrics = get_metrics()
            metrics.gauge(
                "last_engine_cycle_timestamp",
                help="Epoch seconds of the last engine cycle (including skipped)",
            ).set(time.time())
            if cycle is not None:
                metrics.gauge(
                    "trading_equity",
                    help="Last cycle portfolio equity",
                ).set(float(cycle.portfolio_value or 0.0))
                metrics.gauge(
                    "trading_positions",
                    help="Last cycle open position count",
                ).set(float(cycle.n_positions or 0))
            metrics.gauge(
                "risk_kill_switch_active",
                help="1 when the risk kill switch is armed",
            ).set(1.0 if getattr(self._risk, "kill_switch_active", False) else 0.0)
        except Exception:
            pass

    def _execute_cycle_body(self) -> CycleResult:
        self._cycle_count += 1
        # S17: P&L/session/event rows use the exchange-local naive basis
        # shared with the scheduler, not the host clock.
        now = self._exchange_now().replace(tzinfo=None).isoformat()
        self._last_update = now

        # D7/D12: release T+1 locks and reset the daily P&L baseline only
        # when the *exchange* session advances: the date must move and the
        # exchange must actually be open. This used to run on the host-local
        # calendar date, so a UTC/UTC-3 deployment unlocked same-session buys
        # at local midnight and reset the daily loss baseline mid-session;
        # rolling on a weekend/holiday would let paper mode (which trades
        # continuously) sell Friday's buys on Saturday. `_trading_day` is
        # persisted (D7) so a new engine on the same day does not release
        # today's buys.
        exchange_now = self._exchange_now()
        today = exchange_now.strftime("%Y-%m-%d")
        if today != self._trading_day and is_trading_day(exchange_now.date()):
            previous_trading_day = self._trading_day
            self._trading_day = today
            self._day_start_equity = None
            self._advance_trading_day(today, since=previous_trading_day)
            logger.info("Exchange trading day advanced to %s; T+1 released", today)

        # Recover once the kill switch has been cleared. HALTED only reaches
        # READY and INIT (core/state_machine.py:48), so resuming is two hops.
        if self._sm.state == PortfolioState.HALTED and not self._risk.kill_switch_active:
            self._sm.transition(PortfolioState.READY, "kill switch cleared")
            self._sm.transition(PortfolioState.TRADING, "resumed after halt")

        if self._sm.state == PortfolioState.TRADING:
            self._sm.transition(PortfolioState.REBALANCING, f"cycle {self._cycle_count}")

        cycle = CycleResult(cycle_id=self._cycle_count, timestamp=now)

        if self._risk.kill_switch_active:
            logger.critical("Kill switch active: de-risking, no new exposure")
            self._bus.publish("risk.kill_switch", {"active": True}, source="risk")
            self._audit.log_risk_breach("kill_switch", {"active": True}, severity="critical")
            # The early return used to happen before signal generation, so
            # the reducing sells that the pre-trade checks deliberately allow
            # were never created: the switch halted the engine with the whole
            # book still exposed. Sell every available share instead, and
            # still report the account in the cycle result.
            account = self._broker.get_account()
            cycle.portfolio_value = account.get("total_equity", 0)
            cycle.cash = account.get("cash", 0)
            cycle.n_positions = account.get("n_positions", 0)
            self._deleverage(cycle)
            self._halt("kill switch active")
            # R13: a sustained kill switch returns before the normal
            # end-of-cycle sync; book fills that landed asynchronously
            # while the engine sat halted so the ledger still closes.
            self._sync_broker_fills()
            # D5: the kill state itself must survive a restart.
            self._persist_engine_state(
                trust_flat=self._has_filled_sell(cycle.orders),
            )
            return cycle

        # Step 1: Fetch prices → publish market.tick events
        self._fetch_prices_with_tracking()

        # Refuse to place orders from stale quotes. A failed feed used to be
        # swallowed, leaving the engine to size orders from the last good (or
        # NaN) prices indefinitely.
        if self._last_prices_update:
            age = self._aggregate_quote_age()
            if age is not None and age > self._max_quote_age_seconds:
                logger.error(
                    "Quotes are %.0fs old (limit %ss); skipping order placement",
                    age, self._max_quote_age_seconds,
                )
                self._bus.publish("system.error", {
                    "error": f"stale market data: last update {self._last_prices_update}",
                    "age_seconds": age,
                }, source="engine")
                account = self._broker.get_account()
                cycle.portfolio_value = account.get("total_equity", 0)
                cycle.cash = account.get("cash", 0)
                cycle.n_positions = account.get("n_positions", 0)
                if self._sm.state == PortfolioState.REBALANCING:
                    self._sm.transition(PortfolioState.TRADING, "cycle skipped: stale quotes")
                # R13: even a skipped cycle books fills the broker reported
                # asynchronously; otherwise a dead feed could starve the
                # ledger until the next healthy cycle (or the close).
                self._sync_broker_fills()
                self._persist_engine_state()
                return cycle

        # D6: take one position snapshot per cycle and distinguish "flat"
        # from "unknown". An unknown book means sizing and pruning are both
        # skipped: the old code read a transient empty query as flat, bought
        # a duplicate stake and deleted every persisted position row.
        self._cycle_positions = None
        positions, book_known = self._query_positions()
        if not book_known:
            logger.error(
                "Position book unavailable; skipping sizing and pruning this cycle",
            )
            self._bus.publish("system.error", {
                "error": "position book unavailable; cycle skipped without "
                         "sizing or pruning",
            }, source="engine")
            account = self._broker.get_account()
            cycle.portfolio_value = account.get("total_equity", 0)
            cycle.cash = account.get("cash", 0)
            cycle.n_positions = account.get("n_positions", 0)
            if self._sm.state == PortfolioState.REBALANCING:
                self._sm.transition(
                    PortfolioState.TRADING, "cycle skipped: position book unknown",
                )
            # R13: the position query failed, but the order book is a
            # separate endpoint; keep booking broker fills even while the
            # cycle is otherwise skipped.
            self._sync_broker_fills()
            self._persist_engine_state()
            return cycle

        # Step 2: Generate signals → publish signal.generated events. The
        # validated snapshot is stashed for zero-arg signal providers and for
        # sell sizing, so every decision in the cycle sees the same book.
        self._cycle_positions = positions
        signals = self._generate_signals()
        cycle.signals = [s.__dict__ for s in signals]
        self._signal_count += len(signals)

        for sig in signals:
            self._audit.log_signal(sig.code, sig.side.value, sig.signal_strength,
                                   strategy="momentum", factors={"reason": sig.reason})

        # Step 3: Execute trades → publish order.* events (with risk checks)
        account = self._broker.get_account()
        total_equity = account.get("total_equity", account.get("market_value", 0) + account.get("cash", 0))

        # Update risk monitor portfolio state from the same snapshot the
        # signals were sized against.
        positions_for_risk = {}
        sector_weights = {}
        for pos in positions:
            weight = pos.market_value / max(total_equity, 1)
            sector = getattr(pos, "sector", "") or "Unknown"
            positions_for_risk[pos.code] = {
                "value": pos.market_value,
                "weight": weight,
                "sector": sector,
            }
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        # Daily P&L, not cumulative-since-inception. Feeding total_pnl here
        # meant a profitable run never tripped the daily-loss breaker and a
        # cumulative loss beyond 3% blocked every order forever.
        if self._day_start_equity is None:
            self._day_start_equity = total_equity
        daily_pnl = total_equity - self._day_start_equity

        self._risk.update_portfolio_state(
            portfolio_value=total_equity,
            daily_pnl=daily_pnl,
            positions=positions_for_risk,
            sector_weights=sector_weights,
        )

        # Check kill switch
        if self._risk.kill_switch_active:
            logger.critical("Kill switch active — blocking all orders this cycle")
            self._bus.publish("risk.kill_switch", {"active": True}, source="risk")
            self._audit.log_risk_breach("kill_switch", {"active": True}, severity="critical")

        for sig in sorted(
            signals, key=lambda s: 0 if s.side == OrderSide.SELL else 1,
        ):
            order = self._signal_to_order(sig, total_equity)
            if not order:
                continue

            # Pre-trade risk check
            order_dict = {"ticker": order.code, "side": order.side.value,
                          "quantity": order.quantity, "price": order.price}
            approved, breaches = self._risk.check_pre_trade(order_dict)

            if not approved:
                logger.warning("Order blocked by risk: %s %s x%d @ %.2f",
                               order.side.value, order.code, order.quantity, order.price)
                self._audit.log_risk_breach(
                    "pre_trade_block",
                    {"code": order.code, "side": order.side.value,
                     "breaches": [b.message for b in breaches]},
                    severity="warning",
                )
                self._bus.publish("risk.order_blocked", {
                    "code": order.code, "breaches": [b.message for b in breaches],
                }, source="risk")
                continue

            # The approved order has not filled yet. Include its exposure in
            # the checks for the rest of this cycle, otherwise a batch of
            # buys all pass a sector limit none of them would pass after the
            # previous ones filled.
            if hasattr(self._risk, "project_order"):
                self._risk.project_order(order_dict)

            result = self._broker.place_order(order)
            cycle.orders.append(result.to_dict())
            # D2: PARTIAL used to fall through both branches: the broker had
            # filled shares but no order row, trade row, audit event or trade
            # count was written, so reconciliation and compliance missed it.
            if (result.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
                    and self._filled_quantity(result) > 0):
                self._trade_count += 1
                self._persist_fill_result(result, reason=sig.reason)
                self._bus.publish("order.filled", result.to_dict(), source="broker")
                # Update real-time risk engine with fill
                if isinstance(self._risk, RealTimeRiskEngine):
                    self._risk.on_fill({
                        "symbol": order.code,
                        "side": order.side.value,
                        "price": result.filled_price,
                        "quantity": self._filled_quantity(result),
                    })
            elif result.status == OrderStatus.REJECTED:
                self._persist_order_event(
                    result, AuditAction.ORDER_REJECTED, reason=result.error_msg,
                )
            else:
                self._persist_order_event(
                    result, AuditAction.ORDER_SUBMITTED, reason=sig.reason,
                )

        # Publish risk status
        self._bus.publish("risk.status", self._risk.get_status(), source="risk")

        # R13: book fills the broker executed asynchronously (QMT/XTP report
        # most executions after place_order returned SUBMITTED). Runs before
        # the P&L snapshot and the position-book write so the trade ledger is
        # complete before the scheduler reconciles at EOD.
        self._sync_broker_fills()

        # Step 4: Save P&L snapshot
        account = self._broker.get_account()
        cycle.portfolio_value = account.get("total_equity", 0)
        cycle.cash = account.get("cash", 0)
        cycle.n_positions = account.get("n_positions", 0)

        self._store.save_pnl_snapshot({
            "timestamp": now,
            "total_equity": cycle.portfolio_value,
            "cash": cycle.cash,
            "market_value": cycle.portfolio_value - cycle.cash,
            "n_positions": cycle.n_positions,
        })

        # Save positions to store, using a fresh query so fills from this
        # cycle are included. An empty answer is trusted only when this cycle
        # executed a filled sell (which explains a flat broker); otherwise an
        # empty answer against a non-empty store is unknown and the persisted
        # book is left untouched (D6). The tenant has to travel with the row:
        # `Position.to_dict()` from the broker has no tenant_id, so the store
        # fell back to 'default' for every engine and the rows of two tenants
        # collided on `code`.
        final_trust_flat = self._has_filled_sell(cycle.orders)
        final_positions, final_known = self._query_positions(
            trust_flat=final_trust_flat,
        )
        if final_known:
            self._sync_store_book(final_positions or [])
        else:
            logger.error(
                "Final position query unavailable; leaving the persisted book "
                "unchanged for this cycle",
            )

        # Publish portfolio event
        self._bus.publish("portfolio.snapshot", {
            "equity": cycle.portfolio_value,
            "cash": cycle.cash,
            "positions": cycle.n_positions,
        }, source="engine")

        # Transition back to trading -- unless the kill switch fired during
        # this cycle, in which case the machine goes to HALTED instead of
        # reporting a normal cycle completion.
        if self._sm.state == PortfolioState.REBALANCING:
            if self._risk.kill_switch_active:
                self._halt("kill switch active")
            else:
                self._sm.transition(PortfolioState.TRADING, f"cycle {self._cycle_count} complete")

        # D5: persist the restart-critical snapshot at the end of every cycle.
        self._persist_engine_state(trust_flat=final_trust_flat)

        logger.info("Cycle %d: signals=%d, orders=%d, equity=%.2f",
                     self._cycle_count, len(signals), len(cycle.orders), cycle.portfolio_value)

        return cycle

    def _fetch_prices_with_tracking(self) -> None:
        """Run the (possibly overridden) fetch and record the refreshed set.

        Embedded runners/tests can replace `_fetch_prices` wholesale. The
        real method records `_last_fetch_codes` itself; for overrides this
        wrapper observes which symbols gained history or changed price during
        the call, so per-symbol staleness works for them too (D11).
        """
        previous_codes = self._last_fetch_codes
        history_before = {
            code: len(history) for code, history in self._price_history.items()
        }
        prices_before = dict(self._current_prices)
        self._fetch_prices()
        if self._last_fetch_codes is not previous_codes:
            return  # the real fetch recorded the refreshed set itself
        refreshed = {
            code for code, history in self._price_history.items()
            if len(history) > history_before.get(code, 0)
        }
        refreshed.update(
            code for code, price in self._current_prices.items()
            if code not in prices_before or prices_before[code] != price
        )
        if refreshed:
            stamp = time.monotonic()
            for code in refreshed:
                self._price_last_seen[code] = stamp
            self._last_fetch_codes = refreshed
            if self._last_prices_update and not self._last_prices_update_monotonic:
                self._last_prices_update_monotonic = stamp

    def _fetch_prices(self):
        try:
            from quant_platform.trading.realtime import RealTimeMarket
            if self._rt_market is None:
                self._rt_market = RealTimeMarket()
            rt = self._rt_market
            quotes = rt.get_quotes(self._universe)
            if getattr(rt, "snapshot_is_stale", False):
                # RealTimeMarket serves its last good frame after a failed
                # refresh. Reusing it is fine for display, but stamping those
                # quotes now would re-certify them as fresh on every cycle.
                logger.warning("Skipping stale market snapshot for trading")
                self._bus.publish(
                    "market.snapshot", {"count": 0, "stale": True},
                    source="realtime",
                )
                return
            updated = 0
            fresh_prices: dict[str, float] = {}
            for q in quotes:
                # A NaN survives `float(value or 0)` in the feed and then
                # poisons equity, sizing and the broker's marks. Drop it here.
                price = float(q.price)
                if not math.isfinite(price) or price <= 0:
                    logger.warning("Dropping invalid quote for %s: %r", q.code, q.price)
                    continue
                source_time = getattr(q, "timestamp", None)
                if source_time:
                    try:
                        quoted_at = datetime.fromisoformat(
                            str(source_time).replace("Z", "+00:00")
                        )
                        now = (datetime.now(quoted_at.tzinfo)
                               if quoted_at.tzinfo is not None else datetime.now())
                        age = (now - quoted_at).total_seconds()
                        if age > self._max_quote_age_seconds or age < -60:
                            logger.warning(
                                "Dropping out-of-window quote for %s: %s",
                                q.code, source_time,
                            )
                            continue
                    except (TypeError, ValueError):
                        logger.warning(
                            "Dropping quote with invalid timestamp for %s: %r",
                            q.code, source_time,
                        )
                        continue
                self._current_prices[q.code] = price
                # D11: freshness is recorded per symbol. Only these symbols
                # may be sized/marked from this cycle's data.
                self._price_last_seen[q.code] = time.monotonic()
                fresh_prices[q.code] = price
                updated += 1
                if q.code not in self._price_history:
                    self._price_history[q.code] = []
                self._price_history[q.code].append(price)
                # One shared signal lookback window.  Each entry is one quote
                # fetch (the web engine's input is a quote buffer, not a
                # daily-close history); with a daily close feed this is the
                # same 127 sessions the CLI runner preloads.  The window must
                # reach 126 returns for six-month momentum.
                if len(self._price_history[q.code]) > SIGNAL_HISTORY_SESSIONS:
                    self._price_history[q.code] = self._price_history[q.code][
                        -SIGNAL_HISTORY_SESSIONS:
                    ]

                # Publish tick event
                self._bus.publish("market.tick", {
                    "code": q.code, "price": price,
                    "change_pct": q.change_pct, "volume": q.volume,
                }, source="realtime")

            if isinstance(self._broker, SimulatedBroker):
                # D11: pass only quotes fetched this cycle. Marking positions
                # (and re-seeding the book) from stale cached prices let a
                # dead symbol trade at an hours-old price.
                self._broker.update_market_prices(fresh_prices)

            if updated:
                # D11: the refreshed set drives per-symbol eligibility; a
                # symbol omitted from this fetch is treated as a dead feed
                # until it quotes again.
                self._last_fetch_codes = set(fresh_prices)
                self._last_prices_update = datetime.now().isoformat()
                self._last_prices_update_monotonic = time.monotonic()
                # S4: dead-man-switch heartbeat for /api/readiness. Only a
                # successful quote fetch counts as "data success".
                try:
                    get_metrics().gauge(
                        "last_data_success_timestamp",
                        help="Epoch seconds of the last successful market-data fetch",
                    ).set(time.time())
                except Exception:
                    pass
            self._bus.publish("market.snapshot", {"count": updated}, source="realtime")
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)

    def _score_prices(self, price_df) -> pd.Series:
        """Score a ``(dates x symbols)`` price panel with the shared composite.

        This is the same causal 7-factor equal-weight function the CLI
        ``LiveRunner`` calls (``trading/signals.py``), so a paper run and the
        live engine produce identical scores for identical price history.
        An injected ``signal_generator`` (tests, alternative alpha models)
        still takes precedence.

        Symbols without enough history come back as NaN.
        """
        if self._signal_generator is not None:
            self._signal_generator.update_prices(price_df)
            return pd.Series(
                self._signal_generator.generate() or {}, dtype=float,
            )
        return composite_signal(price_df)

    def _orders_for_signal_generation(self) -> list | None:
        """Authoritative broker orders, or None when the book is unknown."""
        try:
            authoritative = getattr(self._broker, "get_orders_authoritative", None)
            return (authoritative() if callable(authoritative)
                    else self._broker.get_orders())
        except Exception as exc:
            logger.error("Order book unavailable; refusing new signals: %s", exc)
            return None

    def _working_order_codes(self) -> set[str] | None:
        """Symbols with open broker intents, or None when the book is unknown."""
        orders = self._orders_for_signal_generation()
        if orders is None:
            logger.error("Order book unavailable; refusing new signals")
            return None
        return working_order_codes(
            orders, include_partial=not isinstance(self._broker, SimulatedBroker),
        )

    def _generate_signals(self, positions: list | None = None) -> list[TradeSignal]:
        """Generate signals from the shared causal multi-factor composite.

        Scores come from :func:`quant_platform.trading.signals.composite_signal`
        -- the same 1/3/6-month momentum + 20/60-day volatility + RSI + MACD
        equal-weight composite the CLI ``LiveRunner`` uses, on a [0, 1]
        percentile scale.  An injected ``signal_generator`` overrides it.

        `positions` is the cycle's position snapshot (D6). Passing it in keeps
        signal sizing on the same book the engine validated as *known*, and
        only symbols with a fresh quote may generate a signal (D11).
        """
        signals = []
        if not self._price_history:
            return signals

        if positions is None:
            positions = self._cycle_positions
        if positions is None:
            positions, known = self._query_positions()
            if not known:
                logger.error("Signal generation skipped: position book unknown")
                return signals

        # Build rolling price DataFrame from the price history buffer
        codes_with_data = [c for c, p in self._price_history.items() if len(p) >= 20]
        if not codes_with_data:
            return signals

        # Align all price histories onto one row axis, left-padded with NaN so
        # each symbol keeps its full trailing history (the same shape the
        # runner's history mapping produces).
        max_len = max(len(self._price_history[c]) for c in codes_with_data)
        price_df = pd.DataFrame({
            c: [np.nan] * (max_len - len(self._price_history[c]))
            + list(self._price_history[c])
            for c in codes_with_data
        })

        # Score with the shared causal composite (same function as the CLI
        # runner); symbols without enough history are NaN and excluded.
        signal_scores = self._score_prices(price_df).dropna().to_dict()

        if not signal_scores:
            return signals

        # Rank by score descending, keep top N above the entry threshold.
        # Ties break on the symbol inside rank_symbols, deterministically.
        ranked = rank_symbols(
            signal_scores,
            limit=self._n_stocks,
            min_score=self._min_signal_score,
        )
        target_codes = {code for code, _score in ranked}

        if not target_codes:
            return signals

        working_codes = self._working_order_codes()
        if working_codes is None:
            return signals

        target_weight = 1.0 / max(len(target_codes), 1)
        position_map = {p.code: p for p in positions}
        current_codes = set(position_map)
        equity = max(self._broker.get_account().get("total_equity", 1), 1)

        # Buy new targets (ranked order is deterministic)
        for code, score in ranked:
            if code in current_codes or normalize_symbol(code) in working_codes:
                continue
            if self._quote_is_fresh(code):
                signals.append(TradeSignal(
                    code=code, side=OrderSide.BUY,
                    target_weight=target_weight, current_weight=0,
                    signal_strength=float(score),
                    reason=f"signal_score={score:.3f}",
                ))

        # Sell removed positions
        for code in sorted(current_codes - target_codes):
            if normalize_symbol(code) in working_codes:
                continue
            if self._quote_is_fresh(code):
                pos = position_map[code]
                signals.append(TradeSignal(
                    code=code, side=OrderSide.SELL,
                    target_weight=0,
                    current_weight=pos.market_value / equity,
                    signal_strength=0, reason="removed_from_universe",
                ))

        # Rebalance existing positions (drift > 2%)
        for code in sorted(target_codes & current_codes):
            if normalize_symbol(code) in working_codes:
                continue
            if self._quote_is_fresh(code):
                pos = position_map[code]
                current_weight = pos.market_value / equity
                drift = abs(current_weight - target_weight)
                if drift > 0.02:
                    side = OrderSide.BUY if current_weight < target_weight else OrderSide.SELL
                    signals.append(TradeSignal(
                        code=code, side=side,
                        target_weight=target_weight, current_weight=current_weight,
                        signal_strength=drift, reason=f"rebalance_drift={drift:.3f}",
                    ))

        return signals

    def _signal_to_order(self, signal: TradeSignal, total_equity: float,
                         positions: list | None = None) -> Order | None:
        # D11: never size an order from a stale or missing quote; the cached
        # price may be hours old while another symbol keeps the aggregate
        # freshness flag alive.
        if not self._quote_is_fresh(signal.code):
            logger.warning(
                "Refusing to size %s %s: quote is stale or missing",
                signal.side.value, signal.code,
            )
            return None
        price = self._current_prices.get(signal.code)
        if price is None or not math.isfinite(price) or price <= 0:
            return None

        if signal.side == OrderSide.BUY:
            target_value = signal.target_weight * total_equity
            current_value = signal.current_weight * total_equity
            trade_value = target_value - current_value
            if trade_value < self._min_trade_value:
                return None
            quantity = int(trade_value / price / 100) * 100
            if quantity <= 0:
                return None
        else:
            if positions is None:
                positions = self._cycle_positions
            if positions is None:
                positions, known = self._query_positions()
                if not known:
                    return None
            position_map = {p.code: p for p in positions}
            pos = position_map.get(signal.code)
            if not pos or pos.available <= 0:
                return None
            quantity = pos.available if signal.target_weight <= 0 else min(
                int((pos.market_value - signal.target_weight * total_equity) / price / 100) * 100,
                pos.available)
            if quantity <= 0:
                return None

        return Order(code=signal.code, side=signal.side, quantity=quantity, price=price)
