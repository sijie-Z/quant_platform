"""Unified execution engine with Order state machine.

Inspired by NautilusTrader's execution architecture: shared order
semantics across research (backtest) and production (live trading).

Key components:
- OrderStateMachine: validates and enforces FSM transitions
- ExecutionEngine: reconciles orders, positions, fills across modes
- EventBus integration for order lifecycle events
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from quant_platform.core.events import get_event_bus
from quant_platform.execution.models import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from quant_platform.execution.oms import require_raw_price_basis
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Order state machine
# ---------------------------------------------------------------------------

# Valid transitions: {from_status: set(to_status)}
ORDER_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING:    {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {OrderStatus.PARTIAL, OrderStatus.FILLED,
                            OrderStatus.CANCELLED, OrderStatus.REJECTED,
                            OrderStatus.EXPIRED},
    OrderStatus.PARTIAL:   {OrderStatus.PARTIAL, OrderStatus.FILLED,
                            OrderStatus.CANCELLED, OrderStatus.EXPIRED},
    OrderStatus.FILLED:    set(),  # Terminal
    OrderStatus.CANCELLED: set(),  # Terminal
    OrderStatus.REJECTED:  set(),  # Terminal
    OrderStatus.EXPIRED:   set(),  # Terminal
}


def validate_order_transition(
    order: Order,
    new_status: OrderStatus,
) -> tuple[bool, str]:
    """Validate order state transition.

    Returns:
        (is_valid, reason) tuple.
    """
    allowed = ORDER_TRANSITIONS.get(order.status)
    if allowed is None:
        return False, f"Unknown current status: {order.status}"
    if new_status not in allowed:
        return False, (
            f"Invalid transition: {order.status} → {new_status}"
        )
    return True, ""


def transition_order(
    order: Order,
    new_status: OrderStatus,
    timestamp: str | None = None,
) -> None:
    """Transition an order to a new state with validation and event emission.

    Args:
        order: The order to transition.
        new_status: Target status.
        timestamp: Event timestamp. Defaults to now.

    Raises:
        ValueError: If the transition is invalid.
    """
    valid, reason = validate_order_transition(order, new_status)
    if not valid:
        raise ValueError(reason)

    old_status = order.status
    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    order.status = new_status
    order.updated_at = ts

    # Set mode-specific timestamps
    if new_status == OrderStatus.SUBMITTED:
        order.submitted_at = ts
    elif new_status == OrderStatus.FILLED:
        order.filled_at = ts
    elif new_status == OrderStatus.REJECTED:
        if not order.notes:
            order.notes = f"Rejected at {ts}"

    # Emit EventBus event
    bus = get_event_bus()
    bus.publish("order.status", {
        "order_id": order.order_id,
        "ticker": order.ticker,
        "old_status": old_status.value,
        "new_status": new_status.value,
        "timestamp": ts,
        "side": order.side.value,
    })

    logger.debug("Order %s: %s → %s", order.order_id[:8], old_status.value, new_status.value)


# ---------------------------------------------------------------------------
# Fill matching
# ---------------------------------------------------------------------------


def apply_fill(order: Order, fill: Fill) -> None:
    """Apply a fill to an order and transition state appropriately.

    Args:
        order: The order being filled.
        fill: The fill to apply.

    Raises:
        ValueError: if the order is not in a fillable state
            (SUBMITTED/PARTIAL) or the fill quantity is not in
            ``(0, order.remaining_quantity]``. The check runs before the
            fill is appended or any state changes, so a rejected fill leaves
            the order untouched.
    """
    if order.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
        raise ValueError(
            f"Cannot fill order in {order.status} state "
            f"(expected SUBMITTED or PARTIAL)"
        )
    if not (0 < fill.quantity <= order.remaining_quantity):
        raise ValueError(
            f"Invalid fill quantity {fill.quantity}: order {order.order_id} "
            f"has {order.remaining_quantity} remaining"
        )

    order.fills.append(fill)
    order.updated_at = fill.timestamp or datetime.now().isoformat(timespec="seconds")

    remaining = order.remaining_quantity
    if remaining <= 0:
        transition_order(order, OrderStatus.FILLED, fill.timestamp)
    else:
        transition_order(order, OrderStatus.PARTIAL, fill.timestamp)

    # Emit fill event
    bus = get_event_bus()
    bus.publish("order.fill", {
        "order_id": order.order_id,
        "ticker": order.ticker,
        "fill_id": fill.fill_id,
        "price": fill.price,
        "quantity": fill.quantity,
        "commission": fill.commission,
        "tax": fill.tax,
        "slippage": fill.slippage,
        "timestamp": fill.timestamp or order.updated_at,
    })


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


def update_position(
    position: Position,
    order: Order,
    fill: Fill,
) -> None:
    """Update a position after a fill using weighted average cost.

    Costs are part of the P&L: buy commission (and any buy-side tax) is
    capitalised into ``avg_cost`` and sell commission/stamp tax are subtracted
    from realised P&L. Without this the strategy ledger (which already
    capitalised costs) and the execution engine (which did not) disagreed by
    exactly the fee amount on every round trip.

    Args:
        position: The position to update.
        order: The source order.
        fill: The fill to apply.
    """
    if order.side == OrderSide.BUY:
        total_qty = position.quantity + fill.quantity
        total_cost = (position.avg_cost * position.quantity
                      + fill.price * fill.quantity
                      + fill.commission + fill.tax)
        position.avg_cost = total_cost / total_qty if total_qty > 0 else 0.0
        position.quantity = total_qty
    else:  # SELL
        # Realized PnL = (sell_price - avg_cost) * qty - fees
        realized = (
            (fill.price - position.avg_cost) * fill.quantity
            - fill.commission - fill.tax
        )
        position.realized_pnl += realized
        position.quantity -= fill.quantity
        if position.quantity <= 0:
            position.quantity = 0
            position.avg_cost = 0.0
            position.market_value = 0.0
            position.unrealized_pnl = 0.0
            position.unrealized_pnl_pct = 0.0

    position.last_updated = fill.timestamp or datetime.now().isoformat(timespec="seconds")

    # Emit position event
    bus = get_event_bus()
    bus.publish("position.update", {
        "ticker": position.ticker,
        "quantity": position.quantity,
        "avg_cost": position.avg_cost,
        "realized_pnl": position.realized_pnl,
    })


# ---------------------------------------------------------------------------
# ExecutionEngine: ties orders, positions, fills together
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Unified execution engine for backtest and live trading.

    Manages the complete order lifecycle:
    Order → Submit → Fill/Cancel → Position update → Portfolio

    Same semantics regardless of mode (backtest or live).

    A-share safeguards live here rather than only in the orchestrator:

    - **T+1**: buys increase ``Position.quantity`` but not
      ``Position.available``; sells are validated against ``available``.
      ``new_trading_day()`` releases the prior day's purchases.
    - **Naked sells**: a sell larger than the real holding raises instead of
      clamping to zero and realising P&L against a zero cost basis.
    - **Cash ledger**: when ``initial_cash`` is provided (or
      ``enable_cash_tracking`` is called), every fill debits/credits cash
      including commission and tax; a buy that would overdraw raises.
    - **Fee-aware P&L**: cost basis includes buy costs and realised P&L nets
      sell costs, so the engine summary reconciles with strategy books.

    Usage:
        engine = ExecutionEngine(initial_cash=1_000_000)
        order = engine.create_order(ticker="600519", side=OrderSide.BUY, quantity=100)
        engine.submit_order(order)
        engine.process_fill(order, price=150.0, quantity=100)
    """

    def __init__(self, initial_cash: float | None = None):
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._executed_orders: list[Order] = []
        #: Basis of the most recent fill/mark prices (cash is CNY, so raw).
        self.price_basis: str = "raw"

        #: Real cash ledger. Only enforced/mutated when tracking is enabled;
        #: engines constructed without `initial_cash` (and never enabled) keep
        #: their historical position-only behaviour.
        self._cash_tracked = initial_cash is not None
        self._initial_cash = float(initial_cash) if initial_cash is not None else 0.0
        self.cash: float = self._initial_cash

        #: A-share T+1: 1 means shares bought today are not sellable until
        #: `new_trading_day()`. 0 disables the lock (buy fills immediately
        #: become available).
        self.t_plus: int = 1

    @property
    def cash_tracking_enabled(self) -> bool:
        return self._cash_tracked

    def enable_cash_tracking(self, initial_cash: float) -> None:
        """Start tracking cash from ``initial_cash``.

        Calling this on an engine that already traded positions will not
        retroactively account for their cost; the normal path is to enable it
        before the first order (the orchestrator does this in ``__init__``).
        """
        self._initial_cash = float(initial_cash)
        self.cash = self._initial_cash
        self._cash_tracked = True

    @property
    def orders(self) -> list[Order]:
        return list(self._orders.values())

    @property
    def positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_position(self, ticker: str) -> Position | None:
        return self._positions.get(ticker)

    def create_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        order_type: str = "market",
        limit_price: float | None = None,
        strategy: str = "",
    ) -> Order:
        """Create and register a new order."""
        order = Order(
            ticker=ticker,
            side=side,
            order_type=order_type,  # type: ignore[arg-type]
            quantity=quantity,
            limit_price=limit_price,
            strategy=strategy,
            status=OrderStatus.PENDING,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._orders[order.order_id] = order
        return order

    def submit_order(self, order: Order) -> None:
        """Submit a pending order to the execution engine."""
        transition_order(order, OrderStatus.SUBMITTED)

    def process_fill(
        self,
        order: Order,
        price: float,
        quantity: int,
        commission: float = 0.0,
        tax: float = 0.0,
        slippage: float = 0.0,
        price_basis: str = "raw",
    ) -> Fill:
        """Process a fill for an order.

        Validation happens before any mutation, so a rejected fill leaves the
        order, position and cash untouched.  ``price`` must be a raw traded
        CNY price; an adjusted basis raises before the book is touched (D15).

        Args:
            order: The order being filled.
            price: Fill price.
            quantity: Fill quantity.
            commission: Commission cost.
            tax: Tax (stamp tax for A-share sells).
            slippage: Slippage cost.
            price_basis: Basis of ``price``; must not be ``qfq``/``hfq``.

        Returns:
            The Fill object.

        Raises:
            ValueError: on an adjusted price basis, a naked sell (quantity >
                holding), a same-day sell of T+1-locked shares, or a buy that
                would drive tracked cash below zero.
        """
        basis = require_raw_price_basis(
            price_basis, context="ExecutionEngine fills",
        )
        if order.side == OrderSide.SELL:
            position = self._positions.get(order.ticker)
            held = position.quantity if position else 0
            if quantity > held:
                raise ValueError(
                    f"Naked sell rejected: {order.ticker} holds {held} shares, "
                    f"order {order.order_id} sells {quantity}"
                )
            available = position.available if position else 0
            if self.t_plus > 0 and quantity > available:
                raise ValueError(
                    f"T+1 locked: {order.ticker} has {available} sellable "
                    f"shares (holding {held}); {held - available} were bought "
                    f"today"
                )
        elif self._cash_tracked:
            required = price * quantity + commission + tax
            if required > self.cash + 1e-9:
                raise ValueError(
                    f"Insufficient cash: buy of {order.ticker} x{quantity} "
                    f"requires {required:.2f}, cash balance {self.cash:.2f}"
                )

        fill = Fill(
            price=price,
            quantity=quantity,
            commission=commission,
            tax=tax,
            slippage=slippage,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        apply_fill(order, fill)

        # Update position
        if order.ticker not in self._positions:
            self._positions[order.ticker] = Position(ticker=order.ticker)
        position = self._positions[order.ticker]
        update_position(position, order, fill)

        # Update the cash ledger and T+1 availability. Buy cost includes
        # commission (and any buy-side tax); sell proceeds net commission and
        # stamp tax.
        if order.side == OrderSide.BUY:
            if self._cash_tracked:
                self.cash -= price * quantity + commission + tax
            if self.t_plus <= 0:
                position.available += quantity
        else:
            if self._cash_tracked:
                self.cash += price * quantity - commission - tax
            position.available = max(position.available - quantity, 0)

        # Mark the position at the fill price so risk consumers see a real
        # market value even before the next snapshot.
        position.update_price(price)
        self.price_basis = basis

        return fill

    def cancel_order(self, order: Order) -> None:
        """Cancel an active order."""
        if order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            transition_order(order, OrderStatus.CANCELLED)

    def reject_order(self, order: Order) -> None:
        """Reject a pending or submitted order."""
        if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            transition_order(order, OrderStatus.REJECTED)

    def expire_order(self, order: Order) -> None:
        """Expire an active day order that did not fill."""
        if order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            transition_order(order, OrderStatus.EXPIRED)

    def new_trading_day(self) -> None:
        """Roll T+1: everything held overnight becomes sellable today.

        Must be called once per trading session before orders are placed
        (the orchestrator does this when the exchange date advances).
        """
        for position in self._positions.values():
            position.available = position.quantity

    def mark_to_market(
        self, prices: dict[str, float], price_basis: str = "raw",
    ) -> None:
        """Mark every position at the given raw CNY prices (missing skipped)."""
        basis = require_raw_price_basis(
            price_basis, context="ExecutionEngine marks",
        )
        self.price_basis = basis
        for position in self._positions.values():
            price = prices.get(position.ticker, 0.0)
            if price > 0:
                position.update_price(price)

    def reset(self) -> None:
        """Clear all orders and positions, restoring the initial cash."""
        self._orders.clear()
        self._positions.clear()
        self._executed_orders.clear()
        self.cash = self._initial_cash

    def portfolio_snapshot(
        self, prices: dict[str, float], price_basis: str = "raw",
    ) -> dict:
        """Compute current portfolio snapshot.

        Args:
            prices: {ticker: current_price} in raw traded CNY.
            price_basis: Basis of ``prices``; adjusted levels are refused.

        Returns:
            Dict with positions_value, n_positions, total_unrealized_pnl,
            total_realized_pnl, total_pnl, cash and equity. Realised P&L is
            reported for closed positions too, so a round trip that ended flat
            still shows its (negative) fees. ``cash``/``equity`` are only
            meaningful when cash tracking is enabled; untracked engines report
            the raw ledger (0.0) and callers should keep their own balance.
        """
        self.price_basis = require_raw_price_basis(
            price_basis, context="ExecutionEngine snapshot marks",
        )
        total_upnl = 0.0
        total_rpnl = 0.0
        positions_value = 0.0
        n_positions = 0

        for pos in self._positions.values():
            price = prices.get(pos.ticker, 0.0)
            if price > 0:
                pos.update_price(price)
            total_rpnl += pos.realized_pnl
            if price > 0 and pos.quantity > 0:
                positions_value += pos.market_value
                total_upnl += pos.unrealized_pnl
                n_positions += 1

        return {
            "positions_value": positions_value,
            "n_positions": n_positions,
            "total_unrealized_pnl": round(total_upnl, 2),
            "total_realized_pnl": round(total_rpnl, 2),
            "total_pnl": round(total_upnl + total_rpnl, 2),
            "cash": round(self.cash, 2),
            "equity": round(self.cash + positions_value, 2),
        }
