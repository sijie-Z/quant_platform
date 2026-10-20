"""Production Live Trading Runner — end-to-end execution with dual tracking.

Orchestrates the complete daily trading workflow:
1. Morning health check (data, cash, positions, risk limits)
2. Multi-factor signal generation (shared 7-factor equal-weight composite:
   1/3/6-month momentum + 20/60-day volatility + RSI + MACD)
3. Pre-trade risk validation (RiskMonitor)
4. Order execution via broker (QMT sim or Paper)
5. Parallel Paper Broker execution for TCA calibration
6. Real-time P&L tracking and NAV update
7. End-of-day report generation (TCA comparison, drawdown, VaR)

Usage:
    # Paper-only run
    runner = LiveRunner(broker_type="simulated", initial_cash=10_000_000)
    runner.set_universe(["600519", "000858", ...])
    runner.run_once()       # single cycle
    runner.run(days=30)     # multi-day simulation

    # Dual-mode run (QMT sim + Paper parallel for TCA comparison)
    runner = LiveRunner(broker_type="qmt_sim", initial_cash=10_000_000, dual_track=True)
    runner.run(days=30)
    report = runner.generate_report()
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from quant_platform.core.audit import AuditAction, AuditLog
from quant_platform.core.events import get_event_bus
from quant_platform.core.instrument import normalize_symbol
from quant_platform.core.state_machine import PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.execution.paper_broker import (
    LatencyModel,
    PaperBroker,
    PaperBrokerMetrics,
)
from quant_platform.risk.circuit_breaker import RiskMonitor, get_risk_monitor
from quant_platform.trading.broker import (
    Order,
    OrderSide,
    OrderType,
    SimulatedBroker,
    create_broker,
)
from quant_platform.trading.live_engine import ensure_raw_price_basis
from quant_platform.trading.signals import (
    DEFAULT_MIN_SIGNAL_SCORE,
    SIGNAL_HISTORY_SESSIONS,
    composite_signal,
    rank_symbols,
)
from quant_platform.utils.exchange_calendar import is_trading_day, trading_days
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

_EXCHANGE_TZ = ZoneInfo("Asia/Shanghai")

#: RiskLevel enum -> numeric value used by the Grafana "Risk Level" panel.
_RISK_LEVEL_VALUES = {
    "green": 0.0,
    "yellow": 1.0,
    "orange": 2.0,
    "red": 3.0,
    "kill": 4.0,
}


# ── Data Models ──


@dataclass
class DailyReport:
    """End-of-day trading report."""
    date: str = ""
    portfolio_value: float = 0.0
    cash: float = 0.0
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    n_positions: int = 0
    n_orders: int = 0
    n_fills: int = 0
    total_commission: float = 0.0
    total_tax: float = 0.0
    total_slippage: float = 0.0
    risk_level: str = "GREEN"
    drawdown_pct: float = 0.0
    paper_metrics: dict | None = None
    price_basis: str = "raw"

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "portfolio_value": round(self.portfolio_value, 2),
            "cash": round(self.cash, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_return_pct": round(self.daily_return_pct, 4),
            "n_positions": self.n_positions,
            "n_orders": self.n_orders,
            "n_fills": self.n_fills,
            "total_commission": round(self.total_commission, 4),
            "total_tax": round(self.total_tax, 4),
            "total_slippage": round(self.total_slippage, 4),
            "risk_level": self.risk_level,
            "drawdown_pct": round(self.drawdown_pct, 4),
            # Every runner price is a broker/exchange quote (traded CNY), so
            # the equity/P&L figures are on a raw basis (D15/D41).
            "price_basis": self.price_basis,
            # The dual-track PaperBroker metrics were computed every cycle
            # and then dropped here, so the TCA comparison never surfaced.
            "paper_metrics": self.paper_metrics,
        }


@dataclass
class SessionReport:
    """Multi-day session report with TCA analysis."""
    session_id: str
    start_date: str
    end_date: str
    days_traded: int
    total_orders: int
    total_fills: int
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_daily_volume: float
    paper_metrics: PaperBrokerMetrics | None = None
    broker_type: str = ""
    status: str = "completed"
    failed_cycles: int = 0
    daily_reports: list[dict] = field(default_factory=list)
    price_basis: str = "raw"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "days_traded": self.days_traded,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "initial_capital": round(self.initial_capital, 2),
            "final_value": round(self.final_value, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "annualized_return_pct": round(self.annualized_return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "avg_daily_volume": round(self.avg_daily_volume, 2),
            "broker_type": self.broker_type,
            "price_basis": self.price_basis,
            "status": self.status,
            "failed_cycles": self.failed_cycles,
        }


# ── Main Runner ──


class LiveRunner:
    """Production trading runner for QMT sim and paper trading.

    Features:
    - Single or dual-broker execution
    - Daily multi-factor signal generation
    - Pre-trade risk checks with kill switch
    - Real-time P&L and position tracking
    - Parallel Paper Broker for TCA comparison
    - End-of-day NAV calculation
    - Session report generation
    """

    def __init__(
        self,
        broker_type: str = "simulated",
        initial_cash: float = 10_000_000,
        dual_track: bool = True,
        risk_monitor: RiskMonitor | None = None,
        store: Store | None = None,
        broker: Any | None = None,
        min_signal_score: float = DEFAULT_MIN_SIGNAL_SCORE,
        max_positions: int = 50,
        **broker_kwargs,
    ):
        self._initial_cash = initial_cash
        self._dual_track = dual_track
        self._session_id = ""
        self._broker_type_name = broker_type
        # Entry threshold on the shared composite's [0, 1] percentile scale
        # (the same default the web LiveTradingEngine uses).
        self._min_signal_score = float(min_signal_score)
        # Cap on how many top-ranked names are targeted (the engine's
        # `n_stocks`). Selection count is configuration; the scores and their
        # order come from the shared `rank_symbols`.
        self._max_positions = max(int(max_positions), 0)
        # A real broker never falls back to the simulated book: a QMT outage
        # that silently becomes paper trading is indistinguishable from a
        # successful live run. connect() returning False is a hard failure.
        if broker is not None:
            self._broker = broker
            if not self._broker.connect():
                raise RuntimeError(
                    f"Broker '{type(broker).__name__}' failed to connect. "
                    "Refusing to continue (fail closed)."
                )
        else:
            try:
                self._broker = create_broker(
                    broker_type, initial_cash=initial_cash, **broker_kwargs,
                )
            except Exception as e:
                logger.error(
                    "Failed to create broker '%s': %s", broker_type, e,
                )
                raise
            if not self._broker.connect():
                raise RuntimeError(
                    f"Broker '{broker_type}' failed to connect. "
                    "Refusing to continue (fail closed)."
                )
        self._is_simulated = isinstance(self._broker, SimulatedBroker)
        # Market-data feed for real brokers. It must return fresh quotes;
        # synthetic random-walk prices are only allowed for the paper broker.
        self._price_provider: Callable[[list[str]], dict[str, float]] | None = None
        # Historical daily closes for real brokers, used to warm up the
        # shared composite (>= 127 sessions so 3/6-month momentum is live).
        self._history_provider: (
            Callable[[list[str], int], dict[str, list[float]]] | None
        ) = None

        # Paper broker (for TCA comparison)
        self._paper_broker: PaperBroker | None = None
        if dual_track:
            self._paper_broker = PaperBroker(
                initial_cash=initial_cash,
                latency=LatencyModel.LOW,
                partial_fill_rate=0.10,
            )

        # Core infrastructure. `store` is injectable so tests (and callers
        # wanting an isolated ledger) do not have to write to the default
        # data/trading.db.
        self._store = store or Store()
        self._bus = get_event_bus()
        self._audit = AuditLog(self._store, self._bus)
        self._sm = PortfolioStateMachine()
        if getattr(self._sm, "_on_transition", None) is None:
            self._sm.set_transition_callback(
                lambda record: self._audit.log_state_change(
                    record.from_state, record.to_state, record.reason,
                )
            )
        # Same singleton the API kill switch mutates. A private instance made
        # the dashboard's kill switch invisible to this runner. Tests inject a
        # fresh monitor so one module's order-rate window cannot leak into the
        # next.
        self._risk = risk_monitor or get_risk_monitor()

        # Universe
        self._universe: list[str] = []

        # State
        self._running = False
        self._positions: dict[str, dict] = {}
        self._current_prices: dict[str, float] = {}
        self._price_history: dict[str, list[float]] = {}
        self._cycle_count = 0
        self._trade_count = 0
        self._peak_equity = initial_cash
        self._current_date = ""
        self._day_start_equity: float | None = None
        self._session_status = "active"
        self._failed_cycles = 0

        # Reports
        self._daily_reports: list[DailyReport] = []
        self._session_report: SessionReport | None = None
        # Last observed cumulative totals for delta-based counters.
        self._metric_state: dict[str, float] = {}
        # Basis of every quote this runner accepts; live/paper marks are raw
        # traded CNY prices only (D15).
        self._price_basis: str = "raw"

    # ── Setup ──

    def set_universe(self, codes: list[str]):
        """Set the trading universe in canonical ``CODE.EXCHANGE`` form.

        Normalizing at the boundary means ``600519`` and ``600519.SH`` are
        one position, not two, and Beijing/board rules see the right
        exchange.  Quote keys fed through :meth:`set_prices` are normalized
        the same way, so lookups always agree.
        """
        canonical = []
        for code in codes:
            if not isinstance(code, str) or not code.strip():
                continue
            normalized = normalize_symbol(code)
            if normalized not in canonical:
                canonical.append(normalized)
        self._universe = canonical
        logger.info("Universe set: %d stocks", len(self._universe))

    def set_price_provider(self, provider: Callable[[list[str]], dict[str, float]]):
        """Set the fresh-quote feed for a real broker.

        ``provider(codes)`` must return ``{code: price}`` with current market
        prices. The runner refuses to run a real broker without one: feeding
        synthetic random-walk prices into live orders would send real orders
        at invented prices.
        """
        self._price_provider = provider

    def set_history_provider(
        self,
        provider: Callable[[list[str], int], dict[str, list[float]]],
    ):
        """Set the historical daily-close feed used to warm up the signal.

        ``provider(codes, sessions)`` must return ``{code: [close, ...]}``
        with trailing daily closes (oldest first), ending at the session
        before the one being traded: the current cycle's quote is appended by
        :meth:`set_prices`, so including it here would count it twice. The
        runner requests ``SIGNAL_HISTORY_SESSIONS`` (127) sessions so the full
        seven-factor composite -- including 3/6-month momentum -- is
        computable on the first live cycle, matching the web engine's window.
        A live broker without this feed falls back to whatever quotes its
        cycles accumulate and logs a warning that the long-horizon factors
        are unavailable.
        """
        self._history_provider = provider

    def set_price_history(
        self,
        history: dict[str, list[float]],
        price_basis: str = "raw",
    ):
        """Preload trailing daily closes used by the shared composite signal.

        Replaces the stored history for each supplied symbol (later
        :meth:`set_prices` calls keep appending to it). Only the trailing
        ``SIGNAL_HISTORY_SESSIONS`` rows are kept, the same window the web
        engine scores. Values must be raw traded prices. Do not include the
        session that :meth:`set_prices` will also append, or that close is
        counted twice.
        """
        basis = ensure_raw_price_basis(
            price_basis, context="LiveRunner signal history",
        )
        self._price_basis = basis if basis != "unknown" else "raw"
        for code, values in history.items():
            cleaned: list[float] = []
            for value in values:
                try:
                    price = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(price) and price > 0:
                    cleaned.append(price)
            if not cleaned:
                continue
            normalized = normalize_symbol(code)
            self._price_history[normalized] = cleaned[-SIGNAL_HISTORY_SESSIONS:]
            if len(cleaned) < SIGNAL_HISTORY_SESSIONS:
                logger.info(
                    "Signal history for %s has %d sessions (< %d); the "
                    "3/6-month factors will be NaN for it",
                    normalized, len(cleaned), SIGNAL_HISTORY_SESSIONS,
                )

    def set_prices(
        self, prices: dict[str, float], price_basis: str = "raw",
    ):
        """Feed market prices for the current cycle.

        Prices must be raw (traded CNY) quotes; an explicitly adjusted basis
        is refused.  Keys are normalized to canonical form so a provider that
        returns ``600519`` and a universe set as ``600519.SH`` still agree.
        """
        basis = ensure_raw_price_basis(
            price_basis, context="LiveRunner marks/fills",
        )
        self._price_basis = basis if basis != "unknown" else "raw"
        normalized: dict[str, float] = {}
        for code, raw_price in prices.items():
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                logger.warning("Dropping non-numeric price for %s: %r", code, raw_price)
                continue
            # Same gate as the web engine's quote fetch: a NaN/inf/zero price
            # would poison equity and every trailing factor window.
            if not np.isfinite(price) or price <= 0:
                logger.warning("Dropping invalid price for %s: %r", code, raw_price)
                continue
            normalized[normalize_symbol(code)] = price
        self._current_prices.update(normalized)
        for code, price in normalized.items():
            history = self._price_history.setdefault(code, [])
            history.append(price)
            # Keep the same trailing window the web engine scores.
            if len(history) > SIGNAL_HISTORY_SESSIONS:
                del history[:-SIGNAL_HISTORY_SESSIONS]

    # ── Signal Generation ──

    def _score_prices(self, price_history=None):
        """Score the price history with the shared causal composite signal.

        Defaults to the runner's accumulated ``_price_history`` for the
        current universe; a ``(dates x symbols)`` DataFrame or
        ``{symbol: prices}`` mapping may be passed explicitly (tests and
        callers sharing one history with the web engine).

        Returns a ``pd.Series`` on the [0, 1] percentile scale, NaN for
        symbols without enough history.
        """
        if price_history is None:
            price_history = {
                code: self._price_history.get(code, [])
                for code in self._universe
            }
        return composite_signal(price_history)

    def _generate_signals(self) -> list[dict]:
        """Generate multi-factor trading signals.

        Scores come from :func:`quant_platform.trading.signals.composite_signal`
        -- the same causal 7-factor equal-weight composite the web
        ``LiveTradingEngine`` uses (1/3/6-month momentum + 20/60-day
        volatility + RSI + MACD), on a [0, 1] cross-sectional percentile
        scale. Symbols without enough history are excluded (no synthetic
        fallback: an invented score is not a signal).

        Returns list of {code, side, target_value|quantity, strength, reason}.
        """
        signals = []
        if not self._universe or not self._current_prices:
            return signals

        scores = self._score_prices()
        ranked = rank_symbols(
            scores,
            limit=self._max_positions,
            min_score=self._min_signal_score,
        )
        if not ranked:
            return signals

        current_equity = self._get_equity()
        target_value_per_stock = current_equity * 0.04  # 4% per position

        # Rebalance against what is actually held.
        #
        # This used to emit `buy` for the whole top-N on every cycle without
        # ever looking at the book, so a 4%-per-name target drained ~4% x N of
        # cash per cycle and no position could ever be exited. Cash went
        # 1M -> 927k -> 873k -> 839k over three cycles with three positions
        # held throughout.
        held = {
            normalize_symbol(p.code): p for p in self._broker.get_positions()
        }
        wanted = {code for code, _score in ranked}

        # Exit anything that has dropped out of the target set.
        for ticker, pos in held.items():
            if ticker in wanted or pos.quantity <= 0:
                continue
            signals.append({
                "code": ticker,
                "side": "sell",
                "quantity": pos.quantity,
                "strength": 0.0,
                "reason": "dropped out of target set",
            })

        # Buy only the shortfall on names we want, not the whole target again.
        for code, score in ranked:
            price = self._current_prices.get(code, 0)
            if price <= 0:
                continue
            existing = held.get(code)
            held_value = existing.market_value if existing is not None else 0.0
            shortfall = target_value_per_stock - held_value
            if shortfall <= 0:
                continue
            signals.append({
                "code": code,
                "side": "buy",
                "target_value": shortfall,
                "strength": round(score, 4),
                "reason": f"composite={score:.3f}",
            })

        return signals

    # ── Execution ──

    def _execute_signal(self, sig: dict) -> dict | None:
        """Execute a single signal through the broker."""
        code = sig["code"]
        price = self._current_prices.get(code, 0)
        if price <= 0:
            return None

        # Calculate quantity (round to lot)
        lot_size = 100
        if isinstance(self._broker, SimulatedBroker):
            lot_size = self._broker._get_lot_size(code)

        if sig.get("quantity") is not None:
            # Exit signal: an explicit share count, already held, so it is
            # whole lots by construction.
            qty = int(sig["quantity"])
        else:
            target_value = sig["target_value"]
            qty = int(target_value / price / lot_size) * lot_size
        if qty <= 0:
            return None

        # Risk pre-trade check
        side = OrderSide.BUY if sig["side"] == "buy" else OrderSide.SELL
        approved, breaches = self._risk.check_pre_trade({
            "ticker": code, "side": sig["side"],
            "quantity": qty, "price": price,
        })
        if not approved:
            logger.warning("Order blocked: %s %s", code, [b.message for b in breaches])
            return None

        # Include the approved-but-unfilled exposure in the checks for the
        # remaining signals of this cycle.
        if hasattr(self._risk, "project_order"):
            self._risk.project_order({
                "ticker": code, "side": sig["side"],
                "quantity": qty, "price": price,
            })

        order = Order(
            code=code, side=side,
            order_type=OrderType.LIMIT,
            quantity=qty, price=price,
        )
        result = self._broker.place_order(order)
        return result.to_dict()

    def _execute_signal_paper(self, sig: dict, primary: dict | None = None):
        """Execute same signal through PaperBroker for TCA comparison.

        D10: the PaperBroker's book started empty for every symbol, so no
        paper order could ever fill and the TCA comparison was always empty.
        Before placing, seed a deterministic ladder from this cycle's quote,
        sized by the primary broker's fill so the comparison is meaningful.
        """
        if not self._paper_broker:
            return None
        code = sig["code"]
        price = self._current_prices.get(code, 0)
        if price <= 0:
            return None
        if sig.get("quantity") is not None:
            # Exit signal: explicit share count, already held.
            qty = int(sig["quantity"])
        else:
            target_value = sig["target_value"]
            qty = int(target_value / price / 100) * 100
        if qty <= 0:
            return None

        reference_quantity = qty
        if primary:
            try:
                primary_filled = int(primary.get("filled_quantity") or 0)
            except (TypeError, ValueError):
                primary_filled = 0
            if primary_filled > 0:
                reference_quantity = max(qty, primary_filled)
        self._seed_paper_liquidity(code, price, reference_quantity)

        order = Order(
            code=code,
            side=OrderSide.BUY if sig["side"] == "buy" else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=qty,
            price=price,
        )
        return self._paper_broker.place_order(order)

    def _seed_paper_liquidity(self, symbol: str, price: float,
                              reference_quantity: int = 0) -> None:
        """Seed the dual-track PaperBroker's book around the cycle quote.

        `seed_book_from_snapshot` is the public, deterministic entry point of
        PaperBroker; the seeded levels are passive, so only our order trades
        against them. Depth scales with the primary fill's quantity.
        """
        if self._paper_broker is None or price <= 0:
            return
        tick = 0.01
        base_quantity = max(100, int(reference_quantity or 10_000))
        bids: list[list] = []
        asks: list[list] = []
        for level in range(5):
            level_quantity = max(100, base_quantity // (level + 1))
            step = tick * level
            bids.append([round(max(tick, price - step), 2), level_quantity])
            asks.append([round(price + step, 2), level_quantity])
        try:
            self._paper_broker.seed_book_from_snapshot(
                symbol, {"bids": bids, "asks": asks},
            )
        except Exception as exc:
            logger.error("Failed to seed paper liquidity for %s: %s", symbol, exc)

    def _record_order_result(self, result: dict, reason: str = "") -> None:
        """Persist an executed order (and its fill) plus an audit event (D8).

        The runner used to write only P&L and session rows, so a run that
        executed real broker orders left zero order/trade rows for compliance
        and reconciliation. A fill is persisted as order+trade atomically
        with a deterministic trade id before the best-effort audit event.
        """
        status = str(result.get("status", "")).lower()
        filled_quantity = int(result.get("filled_quantity", 0) or 0)
        filled = status in ("filled", "partial") and filled_quantity > 0
        try:
            if filled:
                self._store.save_order_with_trade(result, {
                    "trade_id": f"{result['order_id']}:fill",
                    "order_id": result["order_id"],
                    "code": result["code"],
                    "side": result["side"],
                    "quantity": filled_quantity,
                    "price": result.get("filled_price") or result.get("price") or 0.0,
                    "commission": result.get("commission", 0),
                    "tax": result.get("tax", 0),
                })
            else:
                self._store.save_order(result)
        except Exception as exc:
            logger.error("Failed to persist order %s: %s", result.get("order_id"), exc)
        if filled:
            action = AuditAction.ORDER_FILLED
        elif status == "rejected":
            action = AuditAction.ORDER_REJECTED
        else:
            action = AuditAction.ORDER_SUBMITTED
        try:
            self._audit.log(action, "broker", details=result, reason=reason)
        except Exception as exc:
            logger.error("Audit write failed for order %s: %s", result.get("order_id"), exc)

    # ── Main Cycle ──

    def run_once(self, date: str = "", prices: dict[str, float] | None = None) -> DailyReport:
        """Execute a single trading cycle."""
        if not self._running:
            self._start_session()

        self._cycle_count += 1
        previous_date = self._current_date
        self._current_date = date or datetime.now().strftime("%Y-%m-%d")
        if previous_date and self._current_date != previous_date:
            # A new trading day releases T+1-locked shares.
            self._broker.new_trading_day()
            self._day_start_equity = None

        if prices:
            self.set_prices(prices)

        # Mark to market before anything reads equity.
        self._mark_to_market()
        self._update_risk_state()
        self._publish_runtime_metrics()

        # Generate and execute signals
        signals = self._generate_signals()
        orders = []
        paper_records = []
        total_commission = 0.0
        total_tax = 0.0
        total_slippage = 0.0

        # D8: every generated signal is persisted (signal row + audit event)
        # before execution; nothing used to be written.
        for sig in signals:
            # Signals may arrive from strategy code using a native symbol
            # form; normalize once so order/position keys stay canonical.
            if isinstance(sig.get("code"), str):
                sig["code"] = normalize_symbol(sig["code"])
            try:
                self._audit.log_signal(
                    sig["code"], sig["side"], float(sig.get("strength", 0) or 0),
                    strategy="live_runner",
                    factors={"reason": sig.get("reason", "")},
                )
            except Exception as exc:
                logger.error("Failed to persist signal %s: %s", sig.get("code"), exc)

        for sig in sorted(
            signals, key=lambda s: 0 if s["side"] == "sell" else 1,
        ):
            result = self._execute_signal(sig)
            if result:
                orders.append(result)
                # D8: order/trade rows + audit for every execution.
                self._record_order_result(result, reason=sig.get("reason", ""))
                total_commission += result.get("commission", 0)
                total_tax += result.get("tax", 0)
                total_slippage += result.get("slippage", 0)
                if result.get("status") in ("filled", "partial"):
                    self._trade_count += 1

            if self._dual_track:
                paper_record = self._execute_signal_paper(sig, primary=result)
                if paper_record:
                    paper_records.append(paper_record)

        # Update risk monitor state
        risk_status = self._risk.get_status()
        risk_status.get("risk_level", "GREEN")

        # Calculate P&L. Mark to market again first: a position opened during
        # this cycle is still valued at zero until its price is refreshed.
        self._mark_to_market()
        equity = self._get_equity()
        prev_equity = self._peak_equity
        if self._daily_reports:
            prev_equity = self._daily_reports[-1].portfolio_value or self._initial_cash

        daily_pnl = equity - prev_equity
        daily_return = daily_pnl / max(prev_equity, 1)
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / max(self._peak_equity, 1)

        # Paper metrics
        paper_metrics = None
        if self._paper_broker:
            pm = self._paper_broker.get_metrics()
            paper_metrics = {
                "total_orders": pm.total_orders,
                "total_trades": pm.total_trades,
                "partial_fills": pm.partial_fills,
                "cancel_failures": pm.cancel_failures,
                "avg_latency_ms": pm.avg_latency_ms,
                "avg_fill_pct": pm.avg_fill_pct,
            }

        report = DailyReport(
            date=self._current_date,
            portfolio_value=equity,
            cash=self._get_cash(),
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return,
            n_positions=self._get_n_positions(),
            # `n_orders` used to be the signal count and `n_fills` every
            # returned order including rejected/unfilled ones.
            n_orders=len(orders),
            n_fills=sum(
                1 for order in orders
                if order.get("status") in ("filled", "partial")
            ),
            total_commission=total_commission,
            total_tax=total_tax,
            total_slippage=total_slippage,
            risk_level=self._risk.get_status().get("risk_level", "GREEN"),
            drawdown_pct=drawdown,
            paper_metrics=paper_metrics,
            price_basis=self._price_basis,
        )
        self._daily_reports.append(report)

        # Save P&L
        self._store.save_pnl_snapshot({
            "timestamp": self._pnl_timestamp(),
            "total_equity": equity,
            "cash": report.cash,
            "market_value": equity - report.cash,
            "n_positions": report.n_positions,
        })

        return report

    def _pnl_timestamp(self) -> str:
        """P&L timestamp: close-of-day for simulated history, now for live.

        The old code always wrote ``<date>T15:00:00`` with a back-dated
        calendar date, so a run at 10:00 created a P&L row for a session that
        had not happened yet.
        """
        if self._is_simulated:
            return f"{self._current_date}T15:00:00"
        return datetime.now(_EXCHANGE_TZ).strftime("%Y-%m-%dT%H:%M:%S")

    def run(self, days: int = 30, seed: int = 42):
        """Run the trading loop over exchange sessions.

        Simulated/paper brokers replay the last ``days`` *exchange* sessions
        (weekends and closures skipped) with a reproducible random walk.

        Real brokers (QMT/XTP) execute a single live exchange day and require
        a fresh quote feed (``set_price_provider`` or ``broker.get_quotes``):
        synthetic prices are never fed into real orders.
        """
        if not self._universe:
            raise ValueError("Universe not set. Call set_universe() first.")
        days = int(days)
        if days < 1:
            raise ValueError(f"days must be >= 1, got {days}")

        if not self._is_simulated:
            today = datetime.now(_EXCHANGE_TZ).date()
            if not is_trading_day(today):
                raise RuntimeError(
                    f"Exchange is closed on {today}; refusing to run a live cycle."
                )
            if days != 1:
                raise ValueError(
                    "Real brokers support one exchange day per run; use --days 1."
                )
            if self._price_provider is None and not callable(
                getattr(self._broker, "get_quotes", None)
            ):
                raise RuntimeError(
                    f"Real broker '{self._broker_type_name}' requires a market-data "
                    "feed (set_price_provider() or broker.get_quotes()). Refusing to "
                    "feed synthetic prices into live orders."
                )
            dates = [str(today)]
        else:
            dates = self._run_dates(days)

        self._start_session()
        rng = np.random.default_rng(seed)

        # Synthetic base prices (simulated broker only)
        base_prices = {}
        for code in self._universe:
            base_prices[code] = 10.0 + rng.uniform(5, 195)

        # Warm up the signal window before the first traded session: the
        # shared composite needs 126 daily returns for six-month momentum, so
        # a run that starts cold never scores the long-horizon factors.
        if self._is_simulated:
            self._warmup_simulated_history(base_prices, rng)
        else:
            self._load_signal_history()

        failed_cycles = 0
        # D9: one broker/execution failure used to abort the entire multi-day
        # run and leave the session `active` with `_running=True` forever.
        # Failures are now isolated to the affected day and `_end_session()`
        # always runs.
        try:
            for date in dates:
                try:
                    if self._is_simulated:
                        prices = {}
                        for code in self._universe:
                            # Random walk with 15% annual vol
                            daily_ret = rng.normal(0.0003, 0.01)
                            base_prices[code] *= (1 + daily_ret)
                            base_prices[code] = max(base_prices[code], 0.5)
                            prices[code] = round(base_prices[code], 2)
                    else:
                        prices = self._fresh_prices()
                    self.run_once(date=date, prices=prices)
                except Exception as e:
                    failed_cycles += 1
                    logger.error(
                        "Trading day %s aborted: %s", date, e, exc_info=True,
                    )
                    self._bus.publish("system.error", {
                        "error": str(e), "date": date,
                    }, source="live_runner")
                    continue
        finally:
            fills = sum(report.n_fills for report in self._daily_reports)
            # Never claim a session succeeded when nothing filled.
            self._end_session(status="completed" if fills > 0 else "failed")

        self._failed_cycles = failed_cycles
        return self.generate_report()

    def _run_dates(self, days: int) -> list[str]:
        """The last ``days`` exchange sessions ending today (Shanghai time).

        Calendar days included weekends and exchange holidays and shifted the
        P&L rows by the run length; sessions now match the scheduler's
        calendar.
        """
        today = datetime.now(_EXCHANGE_TZ).date()
        start = today - timedelta(days=max(days * 2 + 10, 30))
        sessions = trading_days(start, today)
        return [str(ts.date()) for ts in sessions[-days:]]

    def _warmup_simulated_history(
        self, base_prices: dict[str, float], rng: np.random.Generator,
    ) -> None:
        """Replay synthetic sessions before the first traded date.

        The shared composite requires ``SIGNAL_HISTORY_SESSIONS`` (127) closes
        for six-month momentum. These rows are strictly past data -- they are
        generated before the first ``run_once`` and never include a traded
        date -- so the simulated run exercises the same 7-factor score the
        live engine uses instead of a short-horizon subset. Skipped when the
        caller already supplied a history buffer.
        """
        if self._price_history:
            return
        for _ in range(SIGNAL_HISTORY_SESSIONS):
            for code in self._universe:
                daily_ret = rng.normal(0.0003, 0.01)
                base_prices[code] = max(base_prices[code] * (1 + daily_ret), 0.5)
            self.set_prices({
                code: round(base_prices[code], 2) for code in self._universe
            })

    def _load_signal_history(self) -> None:
        """Preload daily closes for a real-broker run via the history feed.

        The web engine keeps a rolling quote buffer; a live runner only sees
        one quote per cycle, so without a history preload its 3/6-month
        factors would stay NaN indefinitely. The provider is asked for the
        same window the engine scores.
        """
        if self._history_provider is not None:
            try:
                history = self._history_provider(
                    list(self._universe), SIGNAL_HISTORY_SESSIONS,
                ) or {}
            except Exception as exc:
                logger.error("Signal history provider failed: %s", exc)
                history = {}
            if history:
                self.set_price_history(history)
                return
        shortest = min(
            (len(self._price_history.get(code, [])) for code in self._universe),
            default=0,
        )
        logger.warning(
            "No signal history preloaded (shortest history %d < %d sessions); "
            "3/6-month factors stay NaN until enough daily closes accumulate",
            shortest, SIGNAL_HISTORY_SESSIONS,
        )

    def _fresh_prices(self) -> dict[str, float]:
        """Fetch fresh quotes for a real-broker cycle, or raise."""
        raw: dict[str, Any] = {}
        if self._price_provider is not None:
            raw = self._price_provider(list(self._universe)) or {}
        else:
            getter = getattr(self._broker, "get_quotes", None)
            if callable(getter):
                raw = getter(list(self._universe)) or {}
        prices: dict[str, float] = {}
        for code, value in (raw or {}).items():
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0 and np.isfinite(price):
                prices[normalize_symbol(code)] = price
        if not prices:
            raise RuntimeError(
                "market data feed returned no fresh quotes; cycle skipped"
            )
        return prices

    # ── Session Management ──

    def _start_session(self):
        self._running = True
        self._session_status = "active"
        self._session_id = uuid.uuid4().hex[:12]
        self._store.save_session({
            "session_id": self._session_id,
            "broker": type(self._broker).__name__,
            "status": "active",
            "started_at": datetime.now().isoformat(),
        })

    def _end_session(self, status: str = "completed"):
        self._running = False
        self._session_status = status
        self._store.save_session({
            "session_id": self._session_id,
            "status": status,
            "stopped_at": datetime.now().isoformat(),
            "total_trades": self._trade_count,
        })

    # ── Account Helpers ──

    def _get_equity(self) -> float:
        acct = self._broker.get_account()
        return acct.get("total_equity", self._initial_cash)

    def _mark_to_market(self) -> None:
        """Refresh broker position valuations from the latest prices.

        `Position.market_value` is written by exactly one method --
        `SimulatedBroker.update_price()`, reached through
        `update_market_prices()` -- and this runner never called it. Every
        position therefore kept `market_value == 0` and `get_account()`'s
        `total_equity` was just cash: a cycle that spent 160k on stock reported
        a 160k *loss*, and that wrong figure fed back into sizing
        (`target_value_per_stock = equity * 0.04`), so the book shrank every
        cycle. `LiveTradingEngine` does this at its own step 1.

        Called twice per cycle on purpose. Once before sizing, so the equity
        behind the target values includes the book; and once after execution,
        because a position opened this cycle is still valued at zero until the
        next price update -- the same trap as the P&L snapshot in
        `LiveTradingEngine._execute_cycle`.
        """
        if isinstance(self._broker, SimulatedBroker) and self._current_prices:
            self._broker.update_market_prices(self._current_prices)

    def _update_risk_state(self) -> None:
        """Feed the shared risk monitor the current book.

        The runner used to create a private RiskMonitor and never call
        `update_portfolio_state`, so `portfolio_value` stayed 0 and every
        position, daily-loss and drawdown check early-returned None: a 50%
        single-name position sailed through.
        """
        account = self._broker.get_account()
        equity = float(account.get("total_equity", 0) or 0)
        if self._day_start_equity is None:
            self._day_start_equity = equity

        positions = {}
        sector_weights = {}
        for pos in self._broker.get_positions():
            weight = pos.market_value / equity if equity > 0 else 0.0
            sector = getattr(pos, "sector", "") or "Unknown"
            positions[normalize_symbol(pos.code)] = {
                "value": pos.market_value,
                "weight": weight,
                "sector": sector,
            }
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        self._risk.update_portfolio_state(
            portfolio_value=equity,
            daily_pnl=equity - self._day_start_equity,
            positions=positions,
            sector_weights=sector_weights,
        )

    # ── Metrics (Grafana producers) ──

    def _publish_runtime_metrics(self) -> None:
        """Emit the trading/risk/EventBus/store series the dashboard queries.

        Every gauge is read from the live objects (broker account, risk
        monitor status, EventBus counters, Store stats) -- no fabricated
        values. Failures are swallowed: monitoring must not abort a cycle.
        """
        try:
            from quant_platform.utils.metrics import get_metrics

            m = get_metrics()
            broker_kind = type(self._broker).__name__

            account = self._broker.get_account() or {}
            n_positions = account.get("n_positions", 0) or 0
            pnl = account.get("total_pnl")
            if pnl is None:
                pnl = float(account.get("total_equity", self._initial_cash) or 0) - self._initial_cash
            m.gauge(
                "trading_active_positions",
                labels={"broker": broker_kind},
                help="Open positions reported by the broker",
            ).set(float(n_positions))
            m.gauge(
                "trading_portfolio_pnl",
                labels={"broker": broker_kind},
                help="Portfolio P&L reported by the broker (CNY)",
            ).set(float(pnl or 0.0))

            status = self._risk.get_status() or {}
            level = str(status.get("risk_level", "green")).lower()
            m.gauge(
                "risk_level",
                labels={"source": "risk_monitor"},
                help="Risk level: 0=GREEN 1=YELLOW 2=ORANGE 3=RED 4=KILL",
            ).set(_RISK_LEVEL_VALUES.get(level, 0.0))
            m.gauge(
                "risk_kill_switch_active",
                labels={"source": "risk_monitor"},
                help="1 when the kill switch is active",
            ).set(1.0 if status.get("kill_switch") else 0.0)

            bus_metrics = self._bus.get_metrics() or {}
            dlq = bus_metrics.get("dead_letter_queue") or {}
            self._emit_delta_counter(
                m, "eventbus_events_published_total",
                float(bus_metrics.get("published", 0) or 0),
            )
            self._emit_delta_counter(
                m, "eventbus_events_delivered_total",
                float(bus_metrics.get("delivered", 0) or 0),
            )
            self._emit_delta_counter(
                m, "eventbus_dead_letters_total",
                float(dlq.get("total_enqueued", bus_metrics.get("dead_letters", 0)) or 0),
            )

            stats = self._store.get_stats() or {}
            self._emit_delta_counter(
                m, "store_orders_total", float(stats.get("orders", 0) or 0),
            )
            self._emit_delta_counter(
                m, "store_trades_total", float(stats.get("trades", 0) or 0),
            )
        except Exception as exc:  # monitoring must never break trading
            logger.debug("Runtime metric publish failed: %s", exc)

    def _emit_delta_counter(self, metrics, name: str, total: float) -> None:
        """Increment a Prometheus counter by the delta of a cumulative total."""
        previous = self._metric_state.get(name, 0.0)
        delta = total - previous
        counter = metrics.counter(name)
        if delta > 0:
            counter.inc(delta)
        self._metric_state[name] = max(previous, total)

    def _get_cash(self) -> float:
        acct = self._broker.get_account()
        return acct.get("cash", 0)

    def _get_n_positions(self) -> int:
        acct = self._broker.get_account()
        return acct.get("n_positions", 0)

    # ── Report Generation ──

    def generate_report(self) -> SessionReport:
        """Generate a session report with full metrics."""
        if not self._daily_reports:
            return SessionReport(
                session_id=self._session_id,
                start_date="", end_date="",
                days_traded=0, total_orders=0, total_fills=0,
                initial_capital=self._initial_cash,
                final_value=self._initial_cash,
                total_return_pct=0, annualized_return_pct=0,
                sharpe_ratio=0, max_drawdown_pct=0,
                avg_daily_volume=0, broker_type=self._broker_type_name,
                status=self._session_status,
                failed_cycles=self._failed_cycles,
                price_basis=self._price_basis,
            )

        # Calculate metrics
        values = [r.portfolio_value for r in self._daily_reports]
        returns = np.diff(values) / values[:-1]

        annual_factor = 252 / max(len(self._daily_reports), 1)

        total_return = (values[-1] - self._initial_cash) / self._initial_cash
        ann_return = (1 + total_return) ** annual_factor - 1

        if len(returns) > 1:
            sharpe = float(np.mean(returns) / max(np.std(returns), 1e-8) * np.sqrt(252))
        else:
            sharpe = 0.0

        peak = self._initial_cash
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / max(peak, 1)
            max_dd = max(max_dd, dd)

        total_orders = sum(r.n_orders for r in self._daily_reports)
        total_fills = sum(r.n_fills for r in self._daily_reports)
        avg_volume = np.mean(values[-20:]) if len(values) >= 20 else np.mean(values)

        if self._paper_broker:
            self._paper_broker.reset_metrics()

        return SessionReport(
            session_id=self._session_id,
            start_date=self._daily_reports[0].date,
            end_date=self._daily_reports[-1].date,
            days_traded=len(self._daily_reports),
            total_orders=total_orders,
            total_fills=total_fills,
            initial_capital=self._initial_cash,
            final_value=values[-1],
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            avg_daily_volume=round(avg_volume, 2),
            broker_type=self._broker_type_name,
            status=self._session_status,
            failed_cycles=self._failed_cycles,
            daily_reports=[r.to_dict() for r in self._daily_reports],
            price_basis=self._price_basis,
        )

    def get_state(self) -> dict:
        return {
            "running": self._running,
            "session_id": self._session_id,
            "cycles": self._cycle_count,
            "trades": self._trade_count,
            "universe_size": len(self._universe),
            "dual_track": self._dual_track,
            "broker": type(self._broker).__name__,
            "risk_level": self._risk.get_status().get("risk_level", "GREEN"),
            "price_basis": self._price_basis,
        }
