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
    """
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

    Args:
        position: The position to update.
        order: The source order.
        fill: The fill to apply.
    """
    if order.side == OrderSide.BUY:
        total_qty = position.quantity + fill.quantity
        total_cost = (position.avg_cost * position.quantity
                      + fill.price * fill.quantity)
        position.avg_cost = total_cost / total_qty if total_qty > 0 else 0.0
        position.quantity = total_qty
    else:  # SELL
        # Realized PnL = (sell_price - avg_cost) * qty
        realized = (fill.price - position.avg_cost) * fill.quantity
        position.realized_pnl += realized
        position.quantity -= fill.quantity
        if position.quantity <= 0:
            position.quantity = 0
            position.avg_cost = 0.0

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

    Usage:
        engine = ExecutionEngine()
        order = engine.create_order(ticker="600519", side=OrderSide.BUY, quantity=100)
        engine.submit_order(order)
        engine.process_fill(order, Fill(price=150.0, quantity=100))
    """

    def __init__(self):
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._executed_orders: list[Order] = []

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
    ) -> Fill:
        """Process a fill for an order.

        Args:
            order: The order being filled.
            price: Fill price.
            quantity: Fill quantity.
            commission: Commission cost.
            tax: Tax (stamp tax for A-share sells).
            slippage: Slippage cost.

        Returns:
            The Fill object.
        """
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
        update_position(self._positions[order.ticker], order, fill)

        return fill

    def cancel_order(self, order: Order) -> None:
        """Cancel an active order."""
        if order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            transition_order(order, OrderStatus.CANCELLED)

    def reject_order(self, order: Order) -> None:
        """Reject a pending or submitted order."""
        if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            transition_order(order, OrderStatus.REJECTED)

    def reset(self) -> None:
        """Clear all orders and positions."""
        self._orders.clear()
        self._positions.clear()
        self._executed_orders.clear()

    def portfolio_snapshot(self, prices: dict[str, float]) -> dict:
        """Compute current portfolio snapshot.

        Args:
            prices: {ticker: current_price}

        Returns:
            Dict with cash, positions_value, unrealized_pnl, etc.
        """
        total_value = 0.0
        total_upnl = 0.0
        total_rpnl = 0.0
        positions_value = 0.0
        n_positions = 0

        for pos in self._positions.values():
            price = prices.get(pos.ticker, 0.0)
            if price > 0 and pos.quantity > 0:
                pos.update_price(price)
                positions_value += pos.market_value
                total_upnl += pos.unrealized_pnl
                total_rpnl += pos.realized_pnl
                n_positions += 1

        return {
            "positions_value": positions_value,
            "n_positions": n_positions,
            "total_unrealized_pnl": round(total_upnl, 2),
            "total_realized_pnl": round(total_rpnl, 2),
            "total_pnl": round(total_upnl + total_rpnl, 2),
        }
