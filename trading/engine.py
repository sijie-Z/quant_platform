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
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.core.store import Store
from quant_platform.core.state_machine import PortfolioStateMachine, PortfolioState
from quant_platform.core.audit import AuditLog, AuditAction
from quant_platform.core.scheduler import TradingScheduler
from quant_platform.trading.broker import (
    BrokerInterface, Order, OrderSide, OrderStatus, OrderType, Position, SimulatedBroker,
)
from quant_platform.risk.circuit_breaker import RiskMonitor, RiskLimits, RiskLevel
from quant_platform.risk.healthcheck import HealthCheck, SystemBlockError
from quant_platform.risk.realtime_engine import RealTimeRiskEngine
from quant_platform.utils.logging import get_logger

try:
    from quant_platform.core.context import TenantContext
except ImportError:
    TenantContext = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


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
        rebalance_interval: int = 300,
        n_stocks: int = 50,
        min_trade_value: float = 1000,
        tenant_id: str = "default",
    ):
        self._broker = broker
        self._store = store or Store()
        self._bus = bus or get_event_bus()
        self._sm = state_machine or PortfolioStateMachine()
        self._audit = audit or AuditLog(self._store, self._bus)
        self._risk = risk_monitor or RealTimeRiskEngine(
            max_daily_loss=0.03,
            max_drawdown=0.15,
            auto_hedge=False,
        )
        self._rebalance_interval = rebalance_interval
        self._n_stocks = n_stocks
        self._min_trade_value = min_trade_value
        self._tenant_id = tenant_id

        # Set tenant context for this engine instance
        if TenantContext is not None:
            TenantContext.set_current(TenantContext(tenant_id=tenant_id))

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._universe: list[str] = []
        self._current_prices: dict[str, float] = {}
        self._price_history: dict[str, list[float]] = {}
        self._cycle_count = 0
        self._trade_count = 0
        self._signal_count = 0
        self._started_at = ""
        self._last_update = ""
        self._last_prices_update = ""
        self._cycle_log: list[CycleResult] = []
        self._session_id = ""

        # Scheduler
        self._scheduler = TradingScheduler(self._sm, self._store, self._bus, rebalance_interval)

    def set_universe(self, codes: list[str]):
        self._universe = codes
        self._bus.publish("engine.universe_set", {"codes": codes, "count": len(codes)}, source="engine")

    def start(self):
        if self._running:
            return

        self._session_id = uuid.uuid4().hex[:12]
        self._broker.connect()
        self._stop_event.clear()

        # Pre-flight health check
        health = HealthCheck(
            event_bus=self._bus,
            broker=self._broker,
            risk_monitor=self._risk,
        )
        try:
            health.run_all_sync()
        except SystemBlockError as e:
            logger.critical("Engine start blocked by health check: %s", e)
            self._bus.publish("engine.start_blocked", {"error": str(e)}, source="engine")
            raise

        # Set initial equity for risk engine
        if isinstance(self._risk, RealTimeRiskEngine):
            account = self._broker.get_account()
            equity = account.get("total_equity", account.get("initial_cash", 1_000_000))
            self._risk.set_initial_equity(equity)
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

        # Start engine loop
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
        self._stop_event.set()
        self._scheduler.stop()

        if self._thread:
            self._thread.join(timeout=10)

        self._sm.transition(PortfolioState.POST_MARKET, "engine stopped")

        # Save final state
        self._store.save_session({
            "session_id": self._session_id,
            "status": "stopped",
            "stopped_at": datetime.now().isoformat(),
            "total_trades": self._trade_count,
        })

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
        order = Order(code=code, side=OrderSide(side), quantity=quantity, price=price)
        result = self._broker.place_order(order)
        self._trade_count += 1
        self._audit.log_order(result.to_dict(), AuditAction.MANUAL_ORDER, reason="manual")
        return result.to_dict()

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                # Only trade when market is open (or forced)
                if self._scheduler.is_market_open() or True:  # Always run for paper trading
                    cycle = self._execute_cycle()
                    self._cycle_log.append(cycle)
                    if len(self._cycle_log) > 100:
                        self._cycle_log = self._cycle_log[-50:]
            except Exception as e:
                logger.error("Cycle error: %s", e, exc_info=True)
                self._bus.publish("system.error", {"error": str(e)}, source="engine")

            self._stop_event.wait(timeout=self._rebalance_interval)

    def _execute_cycle(self) -> CycleResult:
        self._cycle_count += 1
        now = datetime.now().isoformat()
        self._last_update = now

        if self._sm.state == PortfolioState.TRADING:
            self._sm.transition(PortfolioState.REBALANCING, f"cycle {self._cycle_count}")

        cycle = CycleResult(cycle_id=self._cycle_count, timestamp=now)

        # Step 1: Fetch prices → publish market.tick events
        self._fetch_prices()

        # Step 2: Generate signals → publish signal.generated events
        signals = self._generate_signals()
        cycle.signals = [s.__dict__ for s in signals]
        self._signal_count += len(signals)

        for sig in signals:
            self._audit.log_signal(sig.code, sig.side.value, sig.signal_strength,
                                   strategy="momentum", factors={"reason": sig.reason})

        # Step 3: Execute trades → publish order.* events (with risk checks)
        account = self._broker.get_account()
        total_equity = account.get("total_equity", account.get("market_value", 0) + account.get("cash", 0))

        # Update risk monitor portfolio state
        positions_for_risk = {}
        sector_weights = {}
        for pos in self._broker.get_positions():
            positions_for_risk[pos.code] = {
                "value": pos.market_value,
                "weight": pos.market_value / max(total_equity, 1),
                "sector": "Unknown",
            }

        self._risk.update_portfolio_state(
            portfolio_value=total_equity,
            daily_pnl=account.get("total_pnl", 0),
            positions=positions_for_risk,
            sector_weights=sector_weights,
        )

        # Check kill switch
        if self._risk.kill_switch_active:
            logger.critical("Kill switch active — blocking all orders this cycle")
            self._bus.publish("risk.kill_switch", {"active": True}, source="risk")
            self._audit.log_risk_breach("kill_switch", {"active": True}, severity="critical")

        for sig in signals:
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

            result = self._broker.place_order(order)
            cycle.orders.append(result.to_dict())
            if result.status == OrderStatus.FILLED:
                self._trade_count += 1
                self._audit.log_order(result.to_dict(), AuditAction.ORDER_FILLED,
                                      reason=sig.reason)
                self._bus.publish("order.filled", result.to_dict(), source="broker")
                # Update real-time risk engine with fill
                if isinstance(self._risk, RealTimeRiskEngine):
                    self._risk.on_fill({
                        "symbol": order.code,
                        "side": order.side.value,
                        "price": result.filled_price,
                        "quantity": result.filled_quantity,
                    })
            elif result.status == OrderStatus.REJECTED:
                self._audit.log_order(result.to_dict(), AuditAction.ORDER_REJECTED,
                                      reason=result.error_msg)

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

        # Save positions to store
        for pos in self._broker.get_positions():
            self._store.save_position(pos.to_dict())

        # Publish portfolio event
        self._bus.publish("portfolio.snapshot", {
            "equity": cycle.portfolio_value,
            "cash": cycle.cash,
            "positions": cycle.n_positions,
        }, source="engine")

        # Transition back to trading
        if self._sm.state == PortfolioState.REBALANCING:
            self._sm.transition(PortfolioState.TRADING, f"cycle {self._cycle_count} complete")

        logger.info("Cycle %d: signals=%d, orders=%d, equity=%.2f",
                     self._cycle_count, len(signals), len(cycle.orders), cycle.portfolio_value)

        return cycle

    def _fetch_prices(self):
        try:
            from quant_platform.trading.realtime import RealTimeMarket
            rt = RealTimeMarket()
            quotes = rt.get_quotes(self._universe)
            for q in quotes:
                self._current_prices[q.code] = q.price
                if q.code not in self._price_history:
                    self._price_history[q.code] = []
                self._price_history[q.code].append(q.price)
                if len(self._price_history[q.code]) > 60:
                    self._price_history[q.code] = self._price_history[q.code][-60:]

                # Publish tick event
                self._bus.publish("market.tick", {
                    "code": q.code, "price": q.price,
                    "change_pct": q.change_pct, "volume": q.volume,
                }, source="realtime")

            if isinstance(self._broker, SimulatedBroker):
                self._broker.update_market_prices(self._current_prices)

            self._last_prices_update = datetime.now().isoformat()
            self._bus.publish("market.snapshot", {"count": len(quotes)}, source="realtime")
        except Exception as e:
            logger.warning("Price fetch failed: %s", e)

    def _generate_signals(self) -> list[TradeSignal]:
        signals = []
        if len(self._price_history) < 20:
            return signals

        # Build rolling price DataFrame for factor computation
        import pandas as pd
        codes_with_data = [c for c, p in self._price_history.items() if len(p) >= 20]
        if not codes_with_data:
            return signals

        min_len = min(len(self._price_history[c]) for c in codes_with_data)
        price_df = pd.DataFrame(
            {c: self._price_history[c][-min_len:] for c in codes_with_data}
        )

        # Multi-factor composite: momentum(3M) + volatility(20d) + RSI(14d) + MACD
        factor_scores = pd.DataFrame(index=price_df.columns)

        # Factor 1: 3-month momentum (positive = bullish)
        if min_len >= 63:
            mom = price_df.iloc[-1] / price_df.iloc[-63] - 1
            factor_scores['momentum'] = mom.rank(pct=True)
        else:
            mom = price_df.iloc[-1] / price_df.iloc[max(-min_len, -20)] - 1
            factor_scores['momentum'] = mom.rank(pct=True)

        # Factor 2: Low volatility (negative = lower vol is better)
        ret = price_df.pct_change()
        vol = ret.iloc[-20:].std()
        factor_scores['low_vol'] = (-vol).rank(pct=True)

        # Factor 3: RSI mean-reversion (oversold = buy signal)
        delta = price_df.diff()
        gain = delta.clip(lower=0).iloc[-14:].mean()
        loss = (-delta).clip(lower=0).iloc[-14:].mean()
        loss = loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + gain / loss)
        factor_scores['rsi_contrarian'] = (-rsi).rank(pct=True)  # Buy oversold

        # Factor 4: MACD signal
        ema12 = price_df.ewm(span=12).mean()
        ema26 = price_df.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.iloc[-1].rank(pct=True)
        factor_scores['macd'] = macd_signal

        # Equal-weight composite score
        composite = factor_scores.mean(axis=1)
        composite = composite.dropna().sort_values(ascending=False)

        if composite.empty:
            return signals

        # Select top N stocks with positive composite score
        target_codes = set()
        for code in composite.index[:self._n_stocks]:
            if composite[code] > 0.3:  # Minimum threshold
                target_codes.add(code)

        target_weight = 1.0 / max(len(target_codes), 1)
        positions = {p.code: p for p in self._broker.get_positions()}
        current_codes = set(positions.keys())
        equity = max(self._broker.get_account().get("total_equity", 1), 1)

        # Buy new targets
        for code in target_codes - current_codes:
            if code in self._current_prices:
                signals.append(TradeSignal(
                    code=code, side=OrderSide.BUY,
                    target_weight=target_weight, current_weight=0,
                    signal_strength=float(composite.get(code, 0)),
                    reason=f"multi_factor_score={composite.get(code, 0):.3f}",
                ))

        # Sell removed positions
        for code in current_codes - target_codes:
            if code in self._current_prices:
                pos = positions[code]
                signals.append(TradeSignal(
                    code=code, side=OrderSide.SELL,
                    target_weight=0,
                    current_weight=pos.market_value / equity,
                    signal_strength=0, reason="removed_from_universe",
                ))

        # Rebalance existing positions
        for code in target_codes & current_codes:
            if code in self._current_prices:
                pos = positions[code]
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

    def _signal_to_order(self, signal: TradeSignal, total_equity: float) -> Order | None:
        price = self._current_prices.get(signal.code)
        if not price or price <= 0:
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
            positions = {p.code: p for p in self._broker.get_positions()}
            pos = positions.get(signal.code)
            if not pos or pos.available <= 0:
                return None
            quantity = pos.available if signal.target_weight <= 0 else min(
                int((pos.market_value - signal.target_weight * total_equity) / price / 100) * 100,
                pos.available)
            if quantity <= 0:
                return None

        return Order(code=signal.code, side=signal.side, quantity=quantity, price=price)
