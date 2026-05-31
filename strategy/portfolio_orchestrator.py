"""Portfolio orchestrator: connects signal generation → order execution.

Bridges the gap between MultiStrategyManager (capital allocation, P&L)
and ExecutionEngine (order lifecycle, positions). Handles:

- Converting alpha signals to executable orders
- Per-strategy order routing
- Rebalance event management
- Position reconciliation across strategies
- Cash management (T+1 settlement for A-share)

Usage:
    orchestrator = PortfolioOrchestrator(multi_strat_mgr, exec_engine)
    orchestrator.on_signal(date, signal_df)
    orchestrator.rebalance()
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.core.events import get_event_bus
from quant_platform.execution.engine import (
    ExecutionEngine,
    OrderSide,
    OrderType,
)
from quant_platform.execution.models import OrderStatus
from quant_platform.strategy.multi_strategy import (
    MultiStrategyManager,
    StrategyConfig,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioOrchestrator:
    """Orchestrates signal → order → fill → position lifecycle.

    Owns one ExecutionEngine that all strategies share, ensuring that
    orders from different strategies are consolidated, position limits
    are checked, and cash is properly managed.
    """

    def __init__(
        self,
        multi_strategy: MultiStrategyManager,
        exec_engine: ExecutionEngine | None = None,
        lot_size: int = 100,
        t_plus: int = 1,
    ):
        self.multi_strategy = multi_strategy
        self.exec_engine = exec_engine or ExecutionEngine()
        self.lot_size = lot_size
        self.t_plus = t_plus  # A-share T+1 settlement

        # Track positions per strategy (strategy_id -> {ticker: target_qty})
        self._targets: dict[str, dict[str, int]] = {}
        self._last_prices: dict[str, float] = {}
        self._cash_locked: float = 0.0  # Cash reserved for pending buys
        self._date: str = ""

    @property
    def cash_available(self) -> float:
        """Cash not locked by pending orders."""
        total = self.multi_strategy.total_capital
        locked = self._cash_locked
        # Subtract positions value
        pos_value = sum(
            p.market_value for p in self.exec_engine.positions
        )
        return total - locked - pos_value

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def on_signal(
        self,
        date: str,
        signal: pd.DataFrame,
        strategy_id: str = "default",
        max_positions: int = 50,
        cash_per_position: float | None = None,
    ) -> None:
        """Process an alpha signal and compute target positions.

        This is the main integration point with the factor pipeline.
        Called after AlphaPipeline generates a signal.

        Args:
            date: Current trading date.
            signal: (asset × signal_value) Series or 1-row DataFrame.
            strategy_id: Which strategy this signal belongs to.
            max_positions: Maximum number of positions to hold.
            cash_per_position: Cash per position. If None, computed from
                strategy allocation.
        """
        self._date = date

        # Extract signal as Series
        if isinstance(signal, pd.DataFrame):
            sig_series = signal.iloc[-1].dropna().sort_values(ascending=False)
        else:
            sig_series = signal.dropna().sort_values(ascending=False)

        # Determine capital for this strategy
        strat_state = self.multi_strategy.states.get(strategy_id)
        if strat_state is None:
            logger.warning("Strategy %s not found — using default allocation", strategy_id)
            strategy_capital = self.multi_strategy.total_capital * 0.1
        else:
            strategy_capital = strat_state.capital_allocated

        # Top N assets by signal strength
        top_assets = sig_series.head(max_positions)

        if cash_per_position is None:
            cash_per_position = strategy_capital / max_positions

        # Convert to target quantities
        targets = {}
        for asset, signal_val in top_assets.items():
            price = self._last_prices.get(asset, 0.0)
            if price <= 0:
                continue
            target_cash = cash_per_position * (1 + signal_val)
            qty = int(target_cash / price / self.lot_size) * self.lot_size
            if qty >= self.lot_size:
                targets[asset] = qty

        self._targets[strategy_id] = targets
        logger.info(
            "Signal processed for %s: %d targets, capital=%.0f",
            strategy_id, len(targets), strategy_capital,
        )

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def rebalance(self) -> list[dict]:
        """Execute rebalance: compare targets vs current positions → create orders.

        Returns:
            List of order dicts for audit/logging.
        """
        orders_created = []

        for strategy_id, targets in self._targets.items():
            current_positions = {
                p.ticker: p.quantity
                for p in self.exec_engine.positions
            }

            # Determine buys and sells
            all_tickers = set(targets.keys()) | set(current_positions.keys())

            for ticker in sorted(all_tickers):
                target_qty = targets.get(ticker, 0)
                current_qty = current_positions.get(ticker, 0)

                if target_qty == current_qty:
                    continue

                if target_qty > current_qty:
                    # Need to buy
                    buy_qty = target_qty - current_qty
                    price = self._last_prices.get(ticker, 0.0)
                    if price <= 0:
                        continue

                    order = self.exec_engine.create_order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=self._round_lot(buy_qty),
                        strategy=strategy_id,
                    )
                    self.exec_engine.submit_order(order)
                    self._cash_locked += order.quantity * price

                    orders_created.append({
                        "order_id": order.order_id,
                        "strategy": strategy_id,
                        "ticker": ticker,
                        "side": "buy",
                        "quantity": order.quantity,
                        "price": price,
                    })
                    logger.debug("  BUY  %s %d shares", ticker, order.quantity)

                else:
                    # Need to sell
                    sell_qty = current_qty - target_qty
                    order = self.exec_engine.create_order(
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=self._round_lot(sell_qty),
                        strategy=strategy_id,
                    )
                    self.exec_engine.submit_order(order)

                    orders_created.append({
                        "order_id": order.order_id,
                        "strategy": strategy_id,
                        "ticker": ticker,
                        "side": "sell",
                        "quantity": order.quantity,
                    })
                    logger.debug("  SELL %s %d shares", ticker, order.quantity)

        logger.info(
            "Rebalance complete: %d orders created across %d strategies",
            len(orders_created), len(self._targets),
        )
        return orders_created

    # ------------------------------------------------------------------
    # Fill simulation (for backtest mode)
    # ------------------------------------------------------------------

    def process_fills(
        self,
        prices: dict[str, float],
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        slippage_rate: float = 0.001,
    ) -> None:
        """Simulate fills for all active orders at current prices.

        Call this during backtest after rebalance.

        Args:
            prices: {ticker: current_price}
            commission_rate: 0.03% per trade
            stamp_tax_rate: 0.1% on sell only
            slippage_rate: 0.1% market impact
        """
        self._last_prices.update(prices)

        for order in list(self.exec_engine._orders.values()):  # noqa: SLF001
            if order.status != OrderStatus.SUBMITTED:
                continue

            price = prices.get(order.ticker, 0.0)
            if price <= 0:
                continue

            # Apply slippage
            fill_price = price * (1 + slippage_rate) if order.side == OrderSide.BUY else price * (1 - slippage_rate)

            # Commission (both sides)
            commission = fill_price * order.quantity * commission_rate
            commission = max(commission, 5.0)  # Min 5 RMB

            # Stamp tax (sell only, A-share)
            tax = fill_price * order.quantity * stamp_tax_rate if order.side == OrderSide.SELL else 0.0

            self.exec_engine.process_fill(
                order,
                price=fill_price,
                quantity=order.quantity,
                commission=commission,
                tax=tax,
                slippage=abs(fill_price - price) * order.quantity,
            )

            if order.side == OrderSide.BUY:
                self._cash_locked -= fill_price * order.quantity

        # Update strategy P&L
        self._update_strategy_pnl()

    def _update_strategy_pnl(self) -> None:
        """Push current P&L to MultiStrategyManager."""
        for sid in self._targets:
            if sid not in self.multi_strategy.states:
                continue
            state = self.multi_strategy.states[sid]
            total_pnl = sum(
                p.unrealized_pnl + p.realized_pnl
                for p in self.exec_engine.positions
            )
            if state.current_value > 0:
                daily_return = total_pnl / state.current_value
                self.multi_strategy.update_strategy_pnl(sid, daily_return)

    # ------------------------------------------------------------------
    # Position inquiry
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> dict:
        """Get consolidated portfolio summary."""
        engine_summary = self.exec_engine.portfolio_snapshot(self._last_prices)
        agg = self.multi_strategy.get_aggregate_metrics()
        alerts = self.multi_strategy.get_risk_alerts()

        return {
            "date": self._date,
            "cash_available": round(self.cash_available, 2),
            "positions_value": round(engine_summary["positions_value"], 2),
            "n_positions": engine_summary["n_positions"],
            "unrealized_pnl": engine_summary["total_unrealized_pnl"],
            "realized_pnl": engine_summary["total_realized_pnl"],
            "total_pnl": engine_summary["total_pnl"],
            "n_strategies": agg.get("n_strategies", 0),
            "alerts": len(alerts),
            "positions": [
                {
                    "ticker": p.ticker,
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 3),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                }
                for p in self.exec_engine.positions
                if p.quantity > 0
            ],
        }

    @staticmethod
    def _round_lot(qty: int, lot_size: int = 100) -> int:
        """Round quantity down to nearest lot size."""
        return max((qty // lot_size) * lot_size, lot_size) if qty >= lot_size else 0
