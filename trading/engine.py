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
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from quant_platform.core.audit import AuditAction, AuditLog
from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.core.exchange_calendar import is_trading_day
from quant_platform.core.scheduler import TradingScheduler
from quant_platform.core.state_machine import PortfolioState, PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.execution.models import OrderType
from quant_platform.risk.circuit_breaker import RiskMonitor, get_risk_monitor
from quant_platform.risk.healthcheck import HealthCheck, SystemBlockError
from quant_platform.risk.realtime_engine import RealTimeRiskEngine
from quant_platform.trading.broker import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    SimulatedBroker,
)
from quant_platform.trading.signal_generator import LiveSignalGenerator
from quant_platform.utils.logging import get_logger

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
        min_signal_score: float = 0.3,
        tenant_id: str = "default",
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
        self._signal_generator = signal_generator or LiveSignalGenerator()
        self._rebalance_interval = rebalance_interval
        self._n_stocks = n_stocks
        self._min_trade_value = min_trade_value
        self._min_signal_score = min_signal_score
        self._tenant_id = tenant_id

        # Set tenant context for this engine instance
        if TenantContext is not None:
            TenantContext.set_current(TenantContext(tenant_id=tenant_id))

        self._running = False
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
        # D14: how long stop() waits for the loop thread and for an in-flight
        # cycle to release the cycle lock before giving up loudly.
        self._stop_join_timeout = 30.0

        # NAV calculator
        self._nav_calc = None
        if NAVCalculator is not None:
            self._nav_calc = NAVCalculator(self._store)

        # Scheduler
        self._scheduler = TradingScheduler(self._sm, self._store, self._bus, rebalance_interval)

    def set_universe(self, codes: list[str]):
        self._universe = codes
        self._bus.publish("engine.universe_set", {"codes": codes, "count": len(codes)}, source="engine")

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
        self._running = True
        self._started_at = datetime.now().isoformat()

        # Transition state machine
        self._sm.transition(PortfolioState.READY, "engine started")
        self._sm.transition(PortfolioState.TRADING, "auto-transition to trading")

        # Record session
        self._store.save_session({
            "session_id": self._session_id,
            "broker": type(self._broker).__name__,
            "status": "active",
            "started_at": self._started_at,
        })

        # Start scheduler
        self._scheduler.start()

        # Start engine loop with its own stop Event (D14). Embedded/test
        # overrides of `_run_loop` may still take zero arguments, so only pass
        # the per-run event when the callable accepts it.
        loop_target = getattr(self._run_loop, "__func__", self._run_loop)
        accepts_event = False
        try:
            accepts_event = "stop_event" in inspect.signature(loop_target).parameters
        except (TypeError, ValueError):
            accepts_event = False
        if accepts_event:
            self._thread = threading.Thread(
                target=self._run_loop, args=(self._stop_event,), daemon=True,
            )
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Audit + Event
        self._audit.log(AuditAction.ENGINE_START, "engine", reason=f"session={self._session_id}")
        self._bus.publish("engine.started", {
            "session_id": self._session_id,
            "n_stocks": self._n_stocks,
            "universe_size": len(self._universe),
        }, source="engine")

        logger.info("Engine started. Session: %s, Universe: %d", self._session_id, len(self._universe))

    def stop(self):
        self._running = False
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        self._scheduler.stop()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._stop_join_timeout)
            if thread.is_alive():
                logger.critical(
                    "Engine loop thread still alive %.0fs after stop request",
                    self._stop_join_timeout,
                )
                self._bus.publish("system.error", {
                    "error": "engine loop thread did not exit before stop completed",
                }, source="engine")

        # D14: wait for an in-flight manually triggered cycle to release the
        # lock before writing the "final" session/NAV snapshot. Previously the
        # loop thread join did not cover `POST /trading/run-once`, so stop
        # could save a final state while that cycle was still trading.
        quiesced = self._cycle_lock.acquire(timeout=self._stop_join_timeout)
        if quiesced:
            self._cycle_lock.release()
        else:
            logger.critical(
                "A trading cycle held the cycle lock for %.0fs during stop; "
                "final state may not reflect it",
                self._stop_join_timeout,
            )
            self._bus.publish("system.error", {
                "error": "active cycle did not quiesce before stop completed",
            }, source="engine")

        self._sm.transition(PortfolioState.POST_MARKET, "engine stopped")

        # D5: persist the final snapshot (kill switch, peak equity, day
        # baseline, trading day, cash/positions) so a restart resumes.
        self._persist_engine_state()

        # Save final state
        self._store.save_session({
            "session_id": self._session_id,
            "status": "stopped",
            "stopped_at": datetime.now().isoformat(),
            "total_trades": self._trade_count,
        })

        # Day-end NAV calculation
        if self._nav_calc is not None:
            try:
                account = self._broker.get_account()
                cash = account.get("cash", 0)
                market_value = account.get("total_equity", 0) - cash
                nav = self._nav_calc.update_daily_nav(
                    date=datetime.now().strftime("%Y-%m-%d"),
                    cash=cash,
                    market_value=market_value,
                )
                self._bus.publish("nav.updated", nav.to_dict(), source="engine")
            except Exception as e:
                logger.warning("NAV calculation failed: %s", e)

        self._audit.log(AuditAction.ENGINE_STOP, "engine",
                        details={"trades": self._trade_count, "cycles": self._cycle_count})
        self._bus.publish("engine.stopped", {
            "session_id": self._session_id,
            "trades": self._trade_count,
            "cycles": self._cycle_count,
        }, source="engine")

        self._sm.transition(PortfolioState.READY, "post-market complete")
        logger.info("Engine stopped. Trades: %d, Cycles: %d", self._trade_count, self._cycle_count)

    def run_once(self) -> CycleResult:
        return self._execute_cycle()

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
        result = self._broker.place_order(order)
        # A manual fill is a real broker execution: persist order+trade and
        # count only actual fills. The old code logged MANUAL_ORDER (no trade
        # row) and incremented `_trade_count` even for rejections.
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
            try:
                # Paper trading runs continuously; live brokers only trade in market hours
                if self._scheduler.is_market_open() or isinstance(self._broker, SimulatedBroker):
                    cycle = self._execute_cycle()
                    self._cycle_log.append(cycle)
                    if len(self._cycle_log) > 100:
                        self._cycle_log = self._cycle_log[-50:]
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
                self._bus.publish("system.error", {"error": str(e)}, source="engine")

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
        for pos in positions:
            quantity = int(pos.available)
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
            self._sm.transition(PortfolioState.HALTED, reason)

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
        """
        held = set()
        for pos in positions or []:
            row = self._position_row(pos)
            code = str(row.get("code") or "")
            if not code:
                continue
            row["tenant_id"] = self._tenant_id
            self._store.save_position(row)
            held.add(code)
        for row in self._store.get_positions(self._tenant_id):
            code = row.get("code")
            if code and code not in held:
                self._store.delete_position(code, tenant_id=self._tenant_id)

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
            "executed_at": payload.get("updated_at") or datetime.now().isoformat(),
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
            "updated_at": self._exchange_now().isoformat(),
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
        * both non-empty and disagreeing -> refuse to start (SystemBlockError);
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

        if not broker_known:
            logger.error(
                "Position book unknown at start; the health check will block "
                "trading until the broker answers",
            )
        elif broker_map and local_map and broker_map != local_map:
            raise SystemBlockError(
                "Position reconciliation failed: broker and store disagree "
                f"(broker={broker_map}, local={local_map}); refusing to trade"
            )
        elif not broker_map and local_map:
            restore = getattr(self._broker, "restore_positions", None)
            if not callable(restore):
                raise SystemBlockError(
                    "Position reconciliation failed: broker book is empty but "
                    f"the store records {len(local_map)} position(s) and "
                    f"{type(self._broker).__name__} cannot restore them"
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
        if state.get("kill_switch_active"):
            activate = getattr(self._risk, "activate_kill_switch", None)
            if callable(activate):
                activate("restored persisted kill switch")
            logger.critical(
                "Persisted kill switch restored; orders remain blocked until "
                "explicitly deactivated",
            )

        # Persist the reconciled snapshot so a crash right after start does
        # not lose the reconciliation.
        self._persist_engine_state()

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
            return CycleResult(
                cycle_id=self._cycle_count,
                timestamp=datetime.now().isoformat(),
                portfolio_value=account.get("total_equity", 0),
                cash=account.get("cash", 0),
                n_positions=account.get("n_positions", 0),
            )
        try:
            return self._execute_cycle_body()
        finally:
            self._cycle_lock.release()

    def _execute_cycle_body(self) -> CycleResult:
        self._cycle_count += 1
        now = datetime.now().isoformat()
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
            self._trading_day = today
            self._day_start_equity = None
            self._broker.new_trading_day()
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
            updated = 0
            fresh_prices: dict[str, float] = {}
            for q in quotes:
                # A NaN survives `float(value or 0)` in the feed and then
                # poisons equity, sizing and the broker's marks. Drop it here.
                price = float(q.price)
                if not math.isfinite(price) or price <= 0:
                    logger.warning("Dropping invalid quote for %s: %r", q.code, q.price)
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
                if len(self._price_history[q.code]) > 60:
                    self._price_history[q.code] = self._price_history[q.code][-60:]

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
            self._bus.publish("market.snapshot", {"count": updated}, source="realtime")
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)

    def _generate_signals(self, positions: list | None = None) -> list[TradeSignal]:
        """Generate signals using the same alpha pipeline as backtest research.

        Uses LiveSignalGenerator which reuses factor classes from factors/technical.py
        and AlphaPipeline from alpha/pipeline.py — identical to the research pipeline.

        `positions` is the cycle's position snapshot (D6). Passing it in keeps
        signal sizing on the same book the engine validated as *known*, and
        only symbols with a fresh quote may generate a signal (D11).
        """
        signals = []
        if len(self._price_history) < 20:
            return signals

        if positions is None:
            positions = self._cycle_positions
        if positions is None:
            positions, known = self._query_positions()
            if not known:
                logger.error("Signal generation skipped: position book unknown")
                return signals

        # Build rolling price DataFrame from the price history buffer
        import pandas as pd

        codes_with_data = [c for c, p in self._price_history.items() if len(p) >= 20]
        if not codes_with_data:
            return signals

        # Align all price histories to the same length
        min_len = min(len(self._price_history[c]) for c in codes_with_data)
        price_df = pd.DataFrame(
            {c: self._price_history[c][-min_len:] for c in codes_with_data}
        )

        # Feed into the shared signal generator
        self._signal_generator.update_prices(price_df)
        signal_scores = self._signal_generator.generate()

        if not signal_scores:
            return signals

        # Sort by signal score descending, take top N
        sorted_codes = sorted(signal_scores, key=signal_scores.get, reverse=True)
        target_codes = set(
            c for c in sorted_codes[:self._n_stocks]
            if signal_scores[c] >= self._min_signal_score
        )

        if not target_codes:
            return signals

        target_weight = 1.0 / max(len(target_codes), 1)
        position_map = {p.code: p for p in positions}
        current_codes = set(position_map)
        equity = max(self._broker.get_account().get("total_equity", 1), 1)

        # Buy new targets
        for code in target_codes - current_codes:
            if self._quote_is_fresh(code):
                signals.append(TradeSignal(
                    code=code, side=OrderSide.BUY,
                    target_weight=target_weight, current_weight=0,
                    signal_strength=float(signal_scores.get(code, 0)),
                    reason=f"signal_score={signal_scores.get(code, 0):.3f}",
                ))

        # Sell removed positions
        for code in current_codes - target_codes:
            if self._quote_is_fresh(code):
                pos = position_map[code]
                signals.append(TradeSignal(
                    code=code, side=OrderSide.SELL,
                    target_weight=0,
                    current_weight=pos.market_value / equity,
                    signal_strength=0, reason="removed_from_universe",
                ))

        # Rebalance existing positions (drift > 2%)
        for code in target_codes & current_codes:
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
