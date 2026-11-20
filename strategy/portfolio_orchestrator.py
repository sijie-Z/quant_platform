"""Portfolio orchestrator: connects signal generation → order execution.

Bridges the gap between MultiStrategyManager (capital allocation, P&L)
and ExecutionEngine (order lifecycle, positions). Handles:

- Converting alpha signals to executable orders
- Per-strategy order routing and pending-order netting
- Rebalance event management (global sells-first, then buys)
- Position reconciliation and strategy removal
- Cash management (real cash ledger + T+1 settlement for A-share)

Usage:
    orchestrator = PortfolioOrchestrator(multi_strat_mgr, exec_engine)
    orchestrator.on_signal(date, signal_df)
    orchestrator.rebalance()
    orchestrator.process_fills(prices)
"""

from __future__ import annotations

import pandas as pd
from quant_platform.execution.engine import (
    ExecutionEngine,
    OrderSide,
)
from quant_platform.execution.models import (
    Order,
    OrderStatus,
)
from quant_platform.strategy.multi_strategy import MultiStrategyManager
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Default A-share cost model used by the simulated fill pass and by forced
# liquidation on strategy removal.
DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_STAMP_TAX_RATE = 0.001
DEFAULT_SLIPPAGE_RATE = 0.001
MIN_COMMISSION = 5.0


class PortfolioOrchestrator:
    """Orchestrates signal → order → fill → position lifecycle.

    Owns one ExecutionEngine that all strategies share, ensuring that
    orders from different strategies are consolidated, position limits
    are checked, and cash is properly managed.

    Rebalance invariants:

    - Open orders are netted against the new target, so calling
      ``rebalance()`` twice before ``process_fills()`` does not duplicate
      orders. If a new target supersedes a pending order in the opposite
      direction, that order is cancelled and its cash lock released.
    - All strategies' sells are created (and later filled) before any buy, so
      a purchase is funded by the same pass's sales.
    - Sells are capped by the strategy's book, the engine's real holding and
      today's T+1-available quantity; buys are capped by consolidated cash.
    """

    def __init__(
        self,
        multi_strategy: MultiStrategyManager,
        exec_engine: ExecutionEngine | None = None,
        lot_size: int = 100,
        t_plus: int = 1,
        risk_monitor=None,
    ):
        self.multi_strategy = multi_strategy
        self.exec_engine = exec_engine or ExecutionEngine(
            initial_cash=multi_strategy.total_capital,
        )
        # The shared engine must own the cash ledger; if a caller supplied an
        # engine without one, fund it from the manager's capital so cash
        # checks and `equity = cash + marks` are meaningful.
        if not self.exec_engine.cash_tracking_enabled:
            self.exec_engine.enable_cash_tracking(multi_strategy.total_capital)
        self.exec_engine.t_plus = t_plus
        self.lot_size = lot_size
        self.t_plus = t_plus
        self.risk_monitor = risk_monitor

        # Track positions per strategy (strategy_id -> {ticker: target_qty})
        self._targets: dict[str, dict[str, int]] = {}
        # Per-strategy position ledger. `ExecutionEngine.positions` is one
        # shared book with no strategy attribution, so comparing each
        # strategy's target against it made every strategy act on the whole
        # book: with two strategies each targeting 100 shares, the first fill
        # satisfied both, and the next rebalance made both sell the other's
        # shares. The ledger is updated from fills, which carry `order.strategy`.
        self._books: dict[str, dict[str, dict[str, float]]] = {}
        self._last_strategy_pnl: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        # Cash reserved by open buy orders: order_id -> (locked price, qty).
        # Released on fill, cancel, reject or expiry so an unfilled order never
        # holds capital hostage.
        self._order_locks: dict[str, tuple[float, int]] = {}
        self._date: str = ""
        self._exchange_date: str = ""
        self._day_start_equity: float | None = None

        # The API's strategy-remove endpoint calls `remove_strategy()` on the
        # manager; forward that to the orchestrator so orders/positions are
        # cleaned up before the manager drops the state.
        multi_strategy.set_removal_handler(self._handle_strategy_removal)

    @property
    def cash_available(self) -> float:
        """Real cash minus what open buy orders have reserved."""
        return self.exec_engine.cash - self._locked_cash()

    def _locked_cash(self) -> float:
        return sum(price * qty for price, qty in self._order_locks.values())

    def _lock_cash(self, order: Order, price: float) -> None:
        self._order_locks[order.order_id] = (float(price), int(order.quantity))

    def _release_lock(self, order_id: str, filled_quantity: int | None = None) -> None:
        """Release an order's cash lock (fully on cancel/reject/expiry, or by
        the filled quantity on each fill)."""
        entry = self._order_locks.get(order_id)
        if entry is None:
            return
        price, quantity = entry
        if filled_quantity is None:
            del self._order_locks[order_id]
            return
        remaining = quantity - filled_quantity
        if remaining <= 0:
            del self._order_locks[order_id]
        else:
            self._order_locks[order_id] = (price, remaining)

    # ------------------------------------------------------------------
    # Trading-day bookkeeping (A-share T+1)
    # ------------------------------------------------------------------

    def new_trading_day(self) -> None:
        """Roll the exchange session: prior-day buys become sellable.

        Call this when the exchange date advances (``on_signal`` does it
        automatically for a later date). Also rebases the risk monitor's
        day-start equity, so daily P&L is measured from the new day's open.
        """
        self.exec_engine.new_trading_day()
        self._day_start_equity = None

    def _advance_to(self, date: str) -> None:
        if date and self._exchange_date and date > self._exchange_date:
            self.new_trading_day()
        if date:
            self._exchange_date = date
        self._date = date

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
            date: Current trading date. A later date rolls the T+1 book.
            signal: (asset × signal_value) Series or 1-row DataFrame.
            strategy_id: Which strategy this signal belongs to. Unknown ids
                raise: capital may only be traded inside a registered and
                allocated strategy.
            max_positions: Maximum number of positions to hold.
            cash_per_position: Cash per position. If None, computed from
                strategy allocation. In both cases the total target notional
                is capped at the strategy's allocated capital, so a signal
                vector biased positive cannot over-allocate the strategy.

        Raises:
            ValueError: if ``strategy_id`` is not registered.
        """
        self._advance_to(date)

        if strategy_id not in self.multi_strategy.states:
            raise ValueError(
                f"Unknown strategy_id {strategy_id!r}: register the strategy "
                "with MultiStrategyManager.add_strategy() before sending signals"
            )

        # Extract signal as Series
        if isinstance(signal, pd.DataFrame):
            sig_series = signal.iloc[-1].dropna().sort_values(ascending=False)
        else:
            sig_series = signal.dropna().sort_values(ascending=False)

        strat_state = self.multi_strategy.states[strategy_id]
        strategy_capital = strat_state.capital_allocated
        max_positions = max(int(max_positions), 1)

        if strategy_capital <= 0:
            self._targets[strategy_id] = {}
            logger.info(
                "Signal for %s ignored: no capital allocated", strategy_id,
            )
            return

        # Top N assets by signal strength
        top_assets = sig_series.head(max_positions)
        if cash_per_position is None:
            cash_per_position = strategy_capital / max_positions

        # Raw sizing tilts each position by (1 + signal). Summed over the
        # selected assets this can exceed the strategy's capital when signals
        # are positive-biased (the old code invested up to 150% of allocation),
        # so scale the whole vector down into the budget when it overflows.
        raw_targets: list[tuple[str, float]] = []
        raw_total = 0.0
        for asset, signal_val in top_assets.items():
            price = self._last_prices.get(asset, 0.0)
            if price <= 0:
                continue
            target_cash = cash_per_position * max(0.0, 1.0 + float(signal_val))
            if target_cash <= 0:
                continue
            raw_targets.append((asset, target_cash))
            raw_total += target_cash

        scale = min(1.0, strategy_capital / raw_total) if raw_total > 0 else 1.0

        targets = {}
        for asset, target_cash in raw_targets:
            price = self._last_prices[asset]
            qty = self._round_lot(int(target_cash * scale / price), self.lot_size)
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
        """Execute rebalance: compare targets vs projected positions → orders.

        Projected positions include open orders, so the call is idempotent
        while orders are pending. Pending orders that oppose the new target
        are cancelled first (their cash locks released). Sells for every
        strategy are created before any buy (two-phase), and buys are gated on
        consolidated cash plus expected same-pass sell proceeds.

        Returns:
            List of order dicts for audit/logging.
        """
        orders_created: list[dict] = []

        self._update_risk_monitor()

        self._reconcile_pending_orders()

        sell_intents: list[tuple[str, str, int]] = []
        buy_intents: list[tuple[str, str, int]] = []

        for strategy_id, targets in list(self._targets.items()):
            if strategy_id not in self.multi_strategy.states:
                continue
            # A strategy with no allocated capital must not take on new
            # exposure; any existing book is unwound toward zero.
            if self.multi_strategy.states[strategy_id].capital_allocated <= 0:
                targets = {}
            book = self._books.get(strategy_id, {})
            tickers = set(targets) | set(book)
            tickers |= {
                order.ticker for order in self._open_orders()
                if (order.strategy or "default") == strategy_id
            }
            for ticker in sorted(tickers):
                target_qty = int(targets.get(ticker, 0))
                current_qty = int(book.get(ticker, {}).get("qty", 0))
                pending_buys = self._pending_quantity(
                    strategy_id, ticker, OrderSide.BUY,
                )
                pending_sells = self._pending_quantity(
                    strategy_id, ticker, OrderSide.SELL,
                )
                # Open orders already move the book toward (or away from) the
                # target; only the residual exposure needs a new order.
                delta = target_qty - (current_qty + pending_buys - pending_sells)

                if delta > 0 and pending_sells:
                    # Target supersedes the pending sells: cancel them so the
                    # book does not first sell and then buy back.
                    self._cancel_pending(strategy_id, ticker, OrderSide.SELL)
                    pending_sells = 0
                    delta = target_qty - (current_qty + pending_buys)
                elif delta < 0 and pending_buys:
                    self._cancel_pending(strategy_id, ticker, OrderSide.BUY)
                    pending_buys = 0
                    delta = target_qty - (current_qty - pending_sells)

                if delta > 0:
                    buy_intents.append((strategy_id, ticker, delta))
                elif delta < 0:
                    sell_intents.append((strategy_id, ticker, -delta))

        # ----- Phase 1: sells for every strategy, before any buy -----
        availability: dict[str, int] = {}
        for position in self.exec_engine.positions:
            availability[position.ticker] = (
                position.available if self.exec_engine.t_plus > 0
                else position.quantity
            )
        # Sells already pending from earlier rebalances have first claim on
        # today's sellable quantity.
        for order in self._open_orders():
            if order.side == OrderSide.SELL:
                availability[order.ticker] = (
                    availability.get(order.ticker, 0)
                    - order.remaining_quantity
                )

        for strategy_id, ticker, wanted in sell_intents:
            price = self._last_prices.get(ticker, 0.0)
            if price <= 0:
                continue
            book_qty = int(
                self._books.get(strategy_id, {}).get(ticker, {}).get("qty", 0)
            )
            # Never sell more than this strategy owns, what the engine really
            # holds, or what is T+1-available today.
            sell_qty = min(wanted, book_qty, max(availability.get(ticker, 0), 0))
            sell_qty = self._round_lot(sell_qty, self.lot_size)
            if sell_qty <= 0:
                continue
            order = self.exec_engine.create_order(
                ticker=ticker,
                side=OrderSide.SELL,
                quantity=sell_qty,
                strategy=strategy_id,
            )
            self.exec_engine.submit_order(order)
            availability[ticker] -= sell_qty
            if hasattr(self.risk_monitor, "project_order"):
                self.risk_monitor.project_order({
                    "ticker": ticker,
                    "side": "sell",
                    "quantity": order.quantity,
                    "price": price,
                })
            orders_created.append({
                "order_id": order.order_id,
                "strategy": strategy_id,
                "ticker": ticker,
                "side": "sell",
                "quantity": order.quantity,
            })
            logger.debug("  SELL %s %d shares", ticker, order.quantity)

        # ----- Phase 2: buys, gated on consolidated cash -----
        # Expected proceeds from sells open in this pass fund the buys; the
        # engine still enforces the hard cash limit at fill time.
        sell_proceeds = 0.0
        for order in self._open_orders():
            if order.side != OrderSide.SELL:
                continue
            price = self._last_prices.get(order.ticker, 0.0)
            if price > 0:
                sell_proceeds += order.remaining_quantity * price * (
                    1 - DEFAULT_COMMISSION_RATE - DEFAULT_STAMP_TAX_RATE
                )
        cash_budget = max(
            self.exec_engine.cash + sell_proceeds - self._locked_cash(), 0.0,
        )

        for strategy_id, ticker, wanted in buy_intents:
            price = self._last_prices.get(ticker, 0.0)
            if price <= 0:
                continue
            # Buffer for slippage/commission so the fill-time check does not
            # reject an order the sizing pass thought was affordable.
            affordable = int(cash_budget / (price * 1.002))
            buy_qty = self._round_lot(min(wanted, affordable), self.lot_size)
            if buy_qty <= 0:
                continue
            order = self.exec_engine.create_order(
                ticker=ticker,
                side=OrderSide.BUY,
                quantity=buy_qty,
                strategy=strategy_id,
            )
            if self.risk_monitor is not None:
                approved, breaches = self.risk_monitor.check_pre_trade({
                    "ticker": ticker,
                    "quantity": order.quantity,
                    "price": price,
                    "side": "buy",
                })
                if not approved:
                    reason = "; ".join(b.message for b in breaches)
                    logger.warning("Order blocked by risk: %s", reason)
                    order.notes = f"Risk blocked: {reason}"
                    self.exec_engine.reject_order(order)
                    continue
            self.exec_engine.submit_order(order)
            self._lock_cash(order, price)
            cash_budget -= order.quantity * price
            if hasattr(self.risk_monitor, "project_order"):
                self.risk_monitor.project_order({
                    "ticker": ticker,
                    "side": "buy",
                    "quantity": order.quantity,
                    "price": price,
                })
            orders_created.append({
                "order_id": order.order_id,
                "strategy": strategy_id,
                "ticker": ticker,
                "side": "buy",
                "quantity": order.quantity,
                "price": price,
            })
            logger.debug("  BUY  %s %d shares", ticker, order.quantity)

        logger.info(
            "Rebalance complete: %d orders created across %d strategies",
            len(orders_created), len(self._targets),
        )
        return orders_created

    def _open_orders(self) -> list[Order]:
        return [
            order for order in self.exec_engine.orders
            if order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
        ]

    def _pending_quantity(
        self, strategy_id: str, ticker: str, side: OrderSide,
    ) -> int:
        """Open order quantity for one (strategy, ticker, side) bucket."""
        return sum(
            order.remaining_quantity
            for order in self._open_orders()
            if (order.strategy or "default") == strategy_id
            and order.ticker == ticker
            and order.side == side
        )

    def _cancel_pending(
        self, strategy_id: str, ticker: str, side: OrderSide,
    ) -> None:
        """Cancel open orders in one bucket and release their cash locks."""
        for order in self._open_orders():
            if (
                (order.strategy or "default") == strategy_id
                and order.ticker == ticker
                and order.side == side
            ):
                self.exec_engine.cancel_order(order)
                self._release_lock(order.order_id)
                logger.info(
                    "Cancelled superseded %s order %s (%s x%d)",
                    side.value, order.order_id[:8], ticker,
                    order.remaining_quantity,
                )

    def _reconcile_pending_orders(self) -> None:
        """Drop locks for orders that reached a terminal state outside the fill
        loop (e.g. cancelled/rejected through the engine directly)."""
        for order_id in list(self._order_locks):
            order = self.exec_engine.get_order(order_id)
            if order is None or order.status in (
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.EXPIRED,
                OrderStatus.FILLED,
            ):
                self._release_lock(order_id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order and release its cash lock.

        Returns True when the order was open and is now cancelled.
        """
        order = self.exec_engine.get_order(order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            self.exec_engine.cancel_order(order)
            self._release_lock(order.order_id)
            return True
        return False

    def expire_order(self, order_id: str) -> bool:
        """Expire an unfilled day order and release its cash lock."""
        order = self.exec_engine.get_order(order_id)
        if order is None:
            return False
        if order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            self.exec_engine.expire_order(order)
            self._release_lock(order.order_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Fill simulation (for backtest mode)
    # ------------------------------------------------------------------

    def process_fills(
        self,
        prices: dict[str, float],
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ) -> None:
        """Simulate fills for all active orders at current prices.

        Call this during backtest after rebalance. Sells are filled before
        buys regardless of creation order, so the same pass's sales fund its
        purchases. Positions are marked and the risk monitor refreshed at the
        end.

        Args:
            prices: {ticker: current_price}
            commission_rate: 0.03% per trade (0 disables the fee entirely,
                including the 5 RMB minimum)
            stamp_tax_rate: 0.1% on sell only
            slippage_rate: 0.1% market impact
        """
        self._last_prices.update(prices)
        self.exec_engine.mark_to_market(prices)

        submitted = [
            order for order in self.exec_engine.orders
            if order.status == OrderStatus.SUBMITTED
        ]
        # Deterministic global sells-first ordering.
        submitted.sort(key=lambda order: 0 if order.side == OrderSide.SELL else 1)

        for order in submitted:
            price = prices.get(order.ticker, 0.0)
            if price <= 0:
                logger.debug(
                    "No fill price for %s; order %s stays open",
                    order.ticker, order.order_id[:8],
                )
                continue

            quantity = order.remaining_quantity
            if quantity <= 0:
                continue

            # A sell must not consume another strategy's shares: the order can
            # only settle against the lots its own book owns.
            if order.side == OrderSide.SELL:
                book_qty = int(
                    self._books
                    .get(order.strategy or "default", {})
                    .get(order.ticker, {})
                    .get("qty", 0)
                )
                if book_qty < quantity:
                    logger.warning(
                        "Rejecting sell %s for strategy %s: book holds %d, "
                        "order sells %d",
                        order.ticker, order.strategy or "default",
                        book_qty, quantity,
                    )
                    self.exec_engine.reject_order(order)
                    self._release_lock(order.order_id)
                    continue

            fill_price = (
                price * (1 + slippage_rate) if order.side == OrderSide.BUY
                else price * (1 - slippage_rate)
            )
            commission = (
                max(fill_price * quantity * commission_rate, MIN_COMMISSION)
                if commission_rate > 0 else 0.0
            )
            tax = (
                fill_price * quantity * stamp_tax_rate
                if order.side == OrderSide.SELL else 0.0
            )

            try:
                self.exec_engine.process_fill(
                    order,
                    price=fill_price,
                    quantity=quantity,
                    commission=commission,
                    tax=tax,
                    slippage=abs(fill_price - price) * quantity,
                )
            except ValueError as exc:
                # Cash shortfall (price gapped) or a T+1/naked-sell violation:
                # reject the order and release its lock instead of letting one
                # bad order abort the whole fill pass.
                logger.warning(
                    "Fill rejected for order %s (%s %s x%d): %s",
                    order.order_id[:8], order.side.value, order.ticker,
                    quantity, exc,
                )
                self.exec_engine.reject_order(order)
                self._release_lock(order.order_id)
                continue

            self._record_strategy_fill(
                order, fill_price, quantity, commission, tax,
            )
            self._release_lock(order.order_id, filled_quantity=quantity)

        # Update strategy P&L and risk marks (positions were marked above).
        self._update_strategy_pnl(prices)
        self._update_risk_monitor()

    def _record_strategy_fill(
        self,
        order,
        fill_price: float,
        quantity: int,
        commission: float = 0.0,
        tax: float = 0.0,
    ) -> None:
        """Attribute a fill to the order's strategy.

        The shared book nets the positions; this ledger records who owns what.
        Buy commission is capitalised into the average cost, sell commission
        and stamp tax reduce realised P&L, and the fill price already carries
        slippage. This mirrors `ExecutionEngine.update_position`, so strategy
        books and the engine summary reconcile to the cent.
        """
        sid = order.strategy or "default"
        book = self._books.setdefault(sid, {})
        pos = book.setdefault(
            order.ticker, {"qty": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0},
        )

        if order.side == OrderSide.BUY:
            new_qty = pos["qty"] + quantity
            pos["avg_cost"] = (
                pos["qty"] * pos["avg_cost"] + quantity * fill_price
                + commission + tax
            ) / new_qty
            pos["qty"] = new_qty
        else:
            if pos["qty"] + 1e-9 < quantity:
                # The fill loop validates this before calling the engine; a
                # violation here would mean the shared book was about to
                # realise P&L against lots this strategy does not own.
                raise ValueError(
                    f"Strategy {sid} book holds {pos['qty']} of {order.ticker}, "
                    f"cannot attribute a sell of {quantity}"
                )
            pos["realized_pnl"] += (
                quantity * (fill_price - pos["avg_cost"]) - commission - tax
            )
            pos["qty"] -= quantity

    def _update_strategy_pnl(self, prices: dict[str, float] | None = None) -> None:
        """Push each strategy's own P&L into MultiStrategyManager.

        Each strategy's P&L is realised P&L plus the mark-to-market of only
        its own lots; the daily return handed to `update_strategy_pnl` is the
        change in that P&L over its allocated capital (the manager adds
        daily_pnl to current_value rather than compounding). Zero-capital
        strategies keep their baseline updated but are not fed a return.
        """
        marks = prices or self._last_prices

        for sid, book in self._books.items():
            state = self.multi_strategy.states.get(sid)
            total_pnl = 0.0
            for ticker, pos in book.items():
                price = marks.get(ticker, pos["avg_cost"])
                total_pnl += pos["realized_pnl"] + pos["qty"] * (
                    price - pos["avg_cost"]
                )

            if state is None:
                continue
            previous = self._last_strategy_pnl.get(sid, 0.0)
            self._last_strategy_pnl[sid] = total_pnl
            if state.capital_allocated <= 0:
                continue
            self.multi_strategy.update_strategy_pnl(
                sid, (total_pnl - previous) / state.capital_allocated,
            )

    def _update_risk_monitor(self) -> None:
        """Feed the shared risk monitor the consolidated, marked book.

        Equity is the real cash ledger plus marked positions, so a price drop
        shows up in portfolio value, daily P&L and drawdown. `_day_start_equity`
        is rebased when the exchange date advances.
        """
        if self.risk_monitor is None or not hasattr(
            self.risk_monitor, "update_portfolio_state"
        ):
            return

        self.exec_engine.mark_to_market(self._last_prices)
        positions_value = sum(
            position.market_value for position in self.exec_engine.positions
        )
        equity = self.exec_engine.cash + positions_value
        if equity <= 0:
            return
        if self._day_start_equity is None:
            self._day_start_equity = equity

        state = {}
        sector_weights: dict[str, float] = {}
        for pos in self.exec_engine.positions:
            if pos.quantity <= 0:
                continue
            weight = pos.market_value / equity
            sector = getattr(pos, "sector", "") or "Unknown"
            state[pos.ticker] = {
                "value": pos.market_value,
                "weight": weight,
                "sector": sector,
            }
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        self.risk_monitor.update_portfolio_state(
            portfolio_value=equity,
            daily_pnl=equity - self._day_start_equity,
            positions=state,
            sector_weights=sector_weights,
        )

    # ------------------------------------------------------------------
    # Strategy removal
    # ------------------------------------------------------------------

    def remove_strategy(self, strategy_id: str) -> dict:
        """Remove a strategy, cleaning up before the manager drops it.

        Cancels its open orders, liquidates T+1-available holdings at the last
        known prices (proceeds stay in the shared cash pool) and purges its
        targets, books and P&L baseline. Shares bought today cannot be sold
        under T+1; they are transferred to the shared (unattributed) pool and
        logged. Signals for the removed id are rejected afterwards.

        Raises:
            ValueError: if the strategy is not registered.
        """
        if strategy_id not in self.multi_strategy.states:
            raise ValueError(f"Unknown strategy_id {strategy_id!r}")
        return self.multi_strategy.remove_strategy(strategy_id)

    def _handle_strategy_removal(self, strategy_id: str) -> dict:
        """Cleanup hook installed on the manager (idempotent)."""
        canceled: list[str] = []
        for order in self._open_orders():
            if (order.strategy or "default") == strategy_id:
                self.exec_engine.cancel_order(order)
                self._release_lock(order.order_id)
                canceled.append(order.order_id)

        liquidated: list[dict] = []
        transferred: list[dict] = []
        for ticker, pos in list(self._books.get(strategy_id, {}).items()):
            qty = int(pos.get("qty", 0))
            if qty <= 0:
                continue
            price = self._last_prices.get(ticker, 0.0)
            engine_pos = self.exec_engine.get_position(ticker)
            if price <= 0 or engine_pos is None:
                transferred.append({"ticker": ticker, "quantity": qty})
                logger.warning(
                    "Removed strategy %s: no price/holding to liquidate %s "
                    "x%d; position transferred to the shared pool",
                    strategy_id, ticker, qty,
                )
                continue
            available = (
                engine_pos.available if self.exec_engine.t_plus > 0
                else engine_pos.quantity
            )
            # Other strategies' pending sells already have a claim on the
            # shared sellable quantity; do not let the forced liquidation
            # consume shares they are counting on.
            for other in self._open_orders():
                if (
                    other.side == OrderSide.SELL
                    and (other.strategy or "default") != strategy_id
                ):
                    available -= other.remaining_quantity
            sell_qty = min(qty, int(available))
            if sell_qty <= 0:
                transferred.append({"ticker": ticker, "quantity": qty})
                logger.warning(
                    "Removed strategy %s: %s x%d bought today is T+1 locked "
                    "and transferred to the shared pool",
                    strategy_id, ticker, qty,
                )
                continue
            order = self.exec_engine.create_order(
                ticker=ticker,
                side=OrderSide.SELL,
                quantity=sell_qty,
                strategy=strategy_id,
            )
            self.exec_engine.submit_order(order)
            commission = max(
                price * sell_qty * DEFAULT_COMMISSION_RATE, MIN_COMMISSION,
            )
            tax = price * sell_qty * DEFAULT_STAMP_TAX_RATE
            self.exec_engine.process_fill(
                order, price=price, quantity=sell_qty,
                commission=commission, tax=tax,
            )
            self._record_strategy_fill(
                order, price, sell_qty, commission, tax,
            )
            liquidated.append({
                "order_id": order.order_id,
                "ticker": ticker,
                "quantity": sell_qty,
                "price": price,
            })
            if sell_qty < qty:
                transferred.append(
                    {"ticker": ticker, "quantity": qty - sell_qty},
                )

        self._targets.pop(strategy_id, None)
        self._books.pop(strategy_id, None)
        self._last_strategy_pnl.pop(strategy_id, None)

        logger.info(
            "Strategy %s removed: %d orders cancelled, %d lots liquidated",
            strategy_id, len(canceled), len(liquidated),
        )
        return {
            "strategy_id": strategy_id,
            "canceled_orders": canceled,
            "liquidated": liquidated,
            "transferred": transferred,
        }

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
            "equity": round(
                self.exec_engine.cash + engine_summary["positions_value"], 2,
            ),
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
        """Round quantity down to the nearest lot; 0 when below one lot."""
        qty = int(qty)
        if lot_size <= 0:
            return max(qty, 0)
        if qty < lot_size:
            return 0
        return (qty // lot_size) * lot_size
