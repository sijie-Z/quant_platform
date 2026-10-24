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

import pandas as pd

from quant_platform.core.events import get_event_bus
from quant_platform.data.providers.base import normalize_corporate_actions
from quant_platform.execution.models import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from quant_platform.execution.oms import require_raw_price_basis
from quant_platform.shared.symbols import normalize_symbol
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

    def __init__(
        self,
        initial_cash: float | None = None,
        cash_dividend_tax_rate: float = 0.0,
    ):
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

        #: Corporate-action state (paper/backtest book only).  Markers are
        #: keyed on ``(canonical_ticker, ex_date)``; call
        #: :meth:`restore_corporate_action_markers` after a restart.
        self._cash_dividend_tax_rate = max(0.0, float(cash_dividend_tax_rate))
        self._applied_corporate_actions: set[tuple[str, str]] = set()
        self._corporate_action_events: list[dict[str, Any]] = []
        self._cash_ledger: list[dict[str, Any]] = []
        #: Optional feed applied by dated :meth:`new_trading_day` calls.
        self._corporate_actions: pd.DataFrame | None = None

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

    def new_trading_day(self, date: str | None = None) -> list[dict[str, Any]]:
        """Roll T+1 and apply corporate actions on an exchange-day advance.

        Everything held overnight becomes sellable today.  When ``date`` is
        given and a feed was loaded with :meth:`set_corporate_actions`, every
        action with ``ex_date <= date`` that has not been applied yet is
        applied exactly once.  Omitting ``date`` preserves the historical
        T+1-only behaviour.

        Must be called once per trading session before orders are placed
        (the orchestrator does this when the exchange date advances).
        """
        for position in self._positions.values():
            position.available = position.quantity
        if date is None or self._corporate_actions is None:
            return []
        return self.apply_corporate_actions(
            self._corporate_actions, ex_date=date,
        )

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

    # ── Corporate actions (paper/backtest book only) ──

    @property
    def corporate_action_markers(self) -> list[tuple[str, str]]:
        """Applied ``(ticker, ex_date)`` markers for external persistence."""
        return sorted(self._applied_corporate_actions)

    def restore_corporate_action_markers(self, markers) -> None:
        """Rehydrate applied markers after a restart (idempotent).

        Accepts any iterable of ``(ticker, ex_date)`` pairs (or a dict with
        an ``"applied"`` list, the shape SimulatedBroker persists).
        """
        if isinstance(markers, dict):
            markers = markers.get("applied", [])
        for item in markers or []:
            try:
                ticker, date_text = item
                self._applied_corporate_actions.add(
                    (normalize_symbol(ticker), str(date_text))
                )
            except (TypeError, ValueError):
                continue

    def get_corporate_action_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._corporate_action_events]

    def get_cash_ledger(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._cash_ledger]

    def set_corporate_actions(self, actions) -> int:
        """Load the pending cash/split feed applied on exchange-day advances."""
        self._corporate_actions = normalize_corporate_actions(actions)
        return len(self._corporate_actions)

    def apply_corporate_actions(
        self,
        actions,
        ex_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply split/cash actions to positions and cash exactly once.

        Same contract and conventions as
        :meth:`quant_platform.trading.broker.SimulatedBroker.apply_corporate_actions`:
        a split scales ``quantity``/``available`` and divides ``avg_cost`` and
        the mark (market value unchanged); a cash dividend credits
        ``quantity * cash_per_share * (1 - tax)``, adds it to the position's
        realised P&L and lowers the cached mark by the per-share amount so
        equity stays continuous.  Applied ``(ticker, ex_date)`` markers make
        replays no-ops; persist them with
        :attr:`corporate_action_markers` / :meth:`restore_corporate_action_markers`
        to survive a restart.
        """
        frame = normalize_corporate_actions(actions)
        if frame.empty:
            return []
        cutoff = (
            pd.Timestamp(ex_date).normalize() if ex_date is not None else None
        )
        rows = frame.copy()
        rows["_order"] = rows["kind"].map(
            lambda kind: 0 if kind == "cash" else 1
        )
        rows = rows.sort_values(["ex_date", "asset", "_order"], kind="stable")
        applied: list[dict[str, Any]] = []
        # One marker per (asset, ex_date) event: a combined cash+split pair
        # shares the key and both components are applied before it is
        # recorded, so the second row is never skipped as a replay.
        for (group_date, group_asset), group in rows.groupby(
            ["ex_date", "asset"], sort=False,
        ):
            action_date = pd.Timestamp(group_date).normalize()
            if cutoff is not None and action_date > cutoff:
                continue
            marker = (
                normalize_symbol(group_asset),
                action_date.date().isoformat(),
            )
            if marker in self._applied_corporate_actions:
                continue
            self._applied_corporate_actions.add(marker)
            event = self._apply_action_group(marker, group)
            if event is not None:
                self._corporate_action_events.append(event)
                applied.append(event)
        return applied

    def _position_for_ticker(self, ticker: str) -> Position | None:
        for key, position in self._positions.items():
            if normalize_symbol(key) == ticker:
                return position
        return None

    def _apply_action_group(
        self, marker: tuple[str, str], group: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Apply all components sharing one ``(ticker, ex_date)`` marker."""
        ticker = marker[0]
        position = self._position_for_ticker(ticker)
        quantity_before = int(position.quantity) if position is not None else 0
        quantity_after = quantity_before
        avg_cost_after = position.avg_cost if position is not None else 0.0
        cash_total = 0.0
        tax_total = 0.0
        cash_per_share_total = 0.0
        split_ratio_total = 1.0
        components: list[dict[str, Any]] = []
        timestamp = datetime.now().isoformat(timespec="seconds")

        # Cash first: a 送转+派息 dividend is declared per pre-split share.
        for _, row in group.iterrows():
            if str(row["kind"]) != "cash":
                continue
            cash_per_share = float(row["cash_per_share"])
            if not (cash_per_share > 0):
                continue
            gross = quantity_before * cash_per_share
            tax = gross * self._cash_dividend_tax_rate
            net = gross - tax
            cash_total += net
            tax_total += tax
            cash_per_share_total += cash_per_share
            components.append({
                "kind": "cash",
                "cash_per_share": cash_per_share,
                "cash_delta": net,
                "tax": tax,
            })
            if quantity_before > 0:
                if position is not None:
                    position.realized_pnl += net
                    if position.current_price > 0:
                        position.update_price(
                            max(position.current_price - cash_per_share, 0.0)
                        )
                self._cash_ledger.append({
                    "date": marker[1],
                    "ticker": ticker,
                    "kind": "cash_dividend",
                    "quantity": quantity_before,
                    "cash_per_share": cash_per_share,
                    "tax": round(tax, 6),
                    "cash_delta": round(net, 6),
                    "description": (
                        f"{quantity_before} shares x {cash_per_share}/share"
                    ),
                    "timestamp": timestamp,
                })
        if quantity_before > 0 and cash_total:
            # Cash is only tracked when the engine was given an initial
            # balance or `enable_cash_tracking` was called; an untracked
            # engine's `cash` must not silently become a dividend-only
            # balance while its trades are not debited.
            if self._cash_tracked:
                self.cash += cash_total

        for _, row in group.iterrows():
            if str(row["kind"]) != "split":
                continue
            ratio = float(row["split_ratio"])
            if not (ratio > 0) or ratio == 1.0:
                continue
            split_ratio_total *= ratio
            components.append({"kind": "split", "split_ratio": ratio})
            if position is not None:
                new_quantity = int(round(position.quantity * ratio))
                position.quantity = new_quantity
                position.available = min(
                    new_quantity, int(round(position.available * ratio)),
                )
                if position.avg_cost > 0:
                    position.avg_cost = position.avg_cost / ratio
                if position.current_price > 0:
                    position.update_price(position.current_price / ratio)
                quantity_after = new_quantity
                avg_cost_after = position.avg_cost

        if not components:
            return None
        kinds = sorted({component["kind"] for component in components})
        event: dict[str, Any] = {
            "date": marker[1],
            "ticker": ticker,
            "kind": "+".join(kinds),
            "quantity_before": quantity_before,
            "quantity_after": quantity_after,
            "cash_delta": cash_total,
            "tax": tax_total,
            "cash_per_share": cash_per_share_total,
            "split_ratio": split_ratio_total,
            "components": components,
            "timestamp": timestamp,
        }
        if position is not None:
            event["avg_cost_after"] = avg_cost_after
        self._publish_corporate_action(event)
        return event

    @staticmethod
    def _publish_corporate_action(event: dict[str, Any]) -> None:
        try:
            get_event_bus().publish(
                "corporate_action", event, source="execution_engine",
            )
        except Exception:  # monitoring must never break the book
            pass

    def reset(self) -> None:
        """Clear all orders and positions, restoring the initial cash.

        Corporate-action markers are deliberately kept: they record external
        events already reflected in the book, and replaying them after a reset
        would double-apply the action.
        """
        self._orders.clear()
        self._positions.clear()
        self._executed_orders.clear()
        self.cash = self._initial_cash
        self._cash_ledger.clear()
        self._corporate_action_events.clear()

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
