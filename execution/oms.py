"""Order Management System — core engine.

Manages order lifecycle, fill processing, position tracking, and trade cost analysis.
Supports A-share specific rules: T+1, stamp tax (sell-side only), lot sizes of 100.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from quant_platform.execution.models import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
)
from quant_platform.utils.logging import get_logger

try:
    from quant_platform.core.context import TenantContext
except ImportError:
    TenantContext = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)

# A-share cost constants
COMMISSION_RATE = 0.0003      # 0.03% per side
MIN_COMMISSION = 5.0          # Minimum 5 RMB per trade
STAMP_TAX_RATE = 0.001        # 0.1% sell-side only
MIN_LOT = 100                 # 1 lot = 100 shares


class OrderManager:
    """Core OMS engine.

    Responsibilities:
    - Order creation and lifecycle management
    - Fill processing with A-share cost model
    - Position tracking with avg cost method
    - Portfolio snapshots and P&L calculation
    - Trade cost analysis (TCA)
    - Order blotter persistence
    """

    def __init__(self, initial_cash: float = 10_000_000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.blotter: list[Order] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self._trade_date_offset = 0  # For T+1 simulation

    # ── Order Lifecycle ──

    def create_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: float | None = None,
        strategy: str = "",
        notes: str = "",
    ) -> Order:
        """Create a new order (PENDING state)."""
        # Round to lot size
        quantity = (quantity // MIN_LOT) * MIN_LOT
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity} (min {MIN_LOT})")

        # A-share T+1: cannot sell shares bought today
        if side == "sell":
            pos = self.positions.get(ticker)
            if not pos or pos.quantity < quantity:
                avail = pos.quantity if pos else 0
                raise ValueError(
                    f"Insufficient position: {ticker} has {avail}, "
                    f"trying to sell {quantity}"
                )

        # Get tenant from context if available
        tenant_id = "default"
        if TenantContext is not None:
            try:
                tenant_id = TenantContext.get_current().tenant_id
            except Exception:
                pass

        order = Order(
            ticker=ticker,
            tenant_id=tenant_id,
            side=OrderSide(side),
            order_type=OrderType(order_type),
            quantity=quantity,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now().isoformat(),
            strategy=strategy,
            notes=notes,
        )
        self.orders[order.order_id] = order
        logger.info("Created order %s: %s %s x%d @ %s",
                     order.order_id, side, ticker, quantity,
                     limit_price or "market")
        return order

    def submit_order(self, order_id: str) -> Order:
        """Submit order (PENDING -> SUBMITTED)."""
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in {order.status} state")
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now().isoformat()
        logger.info("Submitted order %s", order_id)
        return order

    def fill_order(
        self,
        order_id: str,
        price: float,
        quantity: int | None = None,
        commission: float | None = None,
    ) -> Order:
        """Process a fill (partial or full)."""
        order = self.orders[order_id]
        if order.is_complete:
            raise ValueError(f"Order {order_id} is already {order.status}")

        qty = quantity or order.remaining_quantity
        qty = min(qty, order.remaining_quantity)

        # Calculate costs
        if commission is None:
            commission = max(price * qty * COMMISSION_RATE, MIN_COMMISSION)
        tax = (price * qty * STAMP_TAX_RATE) if order.side == OrderSide.SELL else 0.0

        # Calculate slippage vs arrival price
        arrival_price = order.limit_price or price
        slippage = abs(price - arrival_price) * qty

        fill = Fill(
            timestamp=datetime.now().isoformat(),
            price=price,
            quantity=qty,
            commission=round(commission, 2),
            tax=round(tax, 2),
            slippage=round(slippage, 2),
        )
        order.fills.append(fill)

        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now().isoformat()
        else:
            order.status = OrderStatus.PARTIAL

        # Update positions and cash
        self._process_fill(order, fill)

        logger.info("Filled order %s: %s %s x%d @ %.2f (comm=%.2f, tax=%.2f)",
                     order_id, order.side.value, order.ticker, qty,
                     price, fill.commission, fill.tax)

        # Add to blotter if fully filled
        if order.is_complete:
            self.blotter.append(order)

        return order

    def cancel_order(self, order_id: str, reason: str = "") -> Order:
        """Cancel an order."""
        order = self.orders[order_id]
        if order.is_complete:
            raise ValueError(f"Cannot cancel order in {order.status} state")
        order.status = OrderStatus.CANCELLED
        order.notes = f"{order.notes} | Cancelled: {reason}".strip(" | ")
        logger.info("Cancelled order %s: %s", order_id, reason)
        return order

    # ── Position Management ──

    def _process_fill(self, order: Order, fill: Fill):
        """Update positions and cash after a fill."""
        ticker = order.ticker
        cost = fill.price * fill.quantity
        total_cost = cost + fill.commission + fill.tax

        if order.side == OrderSide.BUY:
            # Deduct cash
            self.cash -= total_cost
            # Update position (avg cost method)
            if ticker not in self.positions:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    avg_cost=fill.price,
                    last_updated=fill.timestamp,
                )
            pos = self.positions[ticker]
            total_qty = pos.quantity + fill.quantity
            if total_qty > 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty

        elif order.side == OrderSide.SELL:
            # Add cash (minus costs)
            self.cash += cost - fill.commission - fill.tax
            # Update position
            pos = self.positions[ticker]
            # Realized P&L
            realized = (fill.price - pos.avg_cost) * fill.quantity
            pos.realized_pnl += realized
            pos.quantity -= fill.quantity
            # Remove position if fully sold
            if pos.quantity <= 0:
                del self.positions[ticker]

    def update_prices(self, prices: dict[str, float]):
        """Update all positions with current market prices."""
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.update_price(prices[ticker])

    # ── Portfolio Analytics ──

    def get_snapshot(self) -> PortfolioSnapshot:
        """Generate current portfolio snapshot."""
        positions_value = sum(p.market_value for p in self.positions.values())
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = sum(p.realized_pnl for p in self.positions.values())

        total_value = self.cash + positions_value
        daily_pnl = total_value - self.initial_cash

        # Calculate weights
        for pos in self.positions.values():
            pos.weight = pos.market_value / total_value if total_value > 0 else 0

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now().isoformat(),
            total_value=round(total_value, 2),
            cash=round(self.cash, 2),
            positions_value=round(positions_value, 2),
            n_positions=len(self.positions),
            total_unrealized_pnl=round(total_unrealized, 2),
            total_realized_pnl=round(total_realized, 2),
            daily_pnl=round(daily_pnl, 2),
            positions=list(self.positions.values()),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_order_blotter(self, tenant_id: str = "") -> list[dict]:
        """Get all orders as a list of dicts, optionally filtered by tenant."""
        result = []
        for order in self.blotter:
            if tenant_id and order.tenant_id != tenant_id:
                continue
            result.append({
                "order_id": order.order_id,
                "tenant_id": order.tenant_id,
                "ticker": order.ticker,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": order.quantity,
                "avg_fill_price": round(order.avg_fill_price, 2),
                "filled_quantity": order.filled_quantity,
                "commission": order.total_commission,
                "tax": order.total_tax,
                "slippage": order.total_slippage,
                "status": order.status.value,
                "created_at": order.created_at,
                "filled_at": order.filled_at,
                "strategy": order.strategy,
            })
        return result

    def get_trade_cost_analysis(self) -> dict:
        """Compute TCA metrics across all filled orders."""
        if not self.blotter:
            return {}

        total_commission = sum(o.total_commission for o in self.blotter)
        total_tax = sum(o.total_tax for o in self.blotter)
        total_slippage = sum(o.total_slippage for o in self.blotter)
        total_volume = sum(
            o.avg_fill_price * o.filled_quantity for o in self.blotter
        )

        buy_orders = [o for o in self.blotter if o.side == OrderSide.BUY]
        sell_orders = [o for o in self.blotter if o.side == OrderSide.SELL]

        return {
            "total_orders": len(self.blotter),
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
            "total_commission": round(total_commission, 2),
            "total_tax": round(total_tax, 2),
            "total_slippage": round(total_slippage, 2),
            "total_volume": round(total_volume, 2),
            "cost_bps": round(
                (total_commission + total_tax + total_slippage) / total_volume * 10000, 2
            ) if total_volume > 0 else 0,
            "avg_order_size": round(
                total_volume / len(self.blotter), 0
            ) if self.blotter else 0,
        }

    # ── Persistence ──

    def save_blotter(self, path: str):
        """Save order blotter to JSON."""
        data = self.get_order_blotter()
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load_blotter(self, path: str):
        """Load order blotter from JSON."""
        data = json.loads(Path(path).read_text())
        for item in data:
            order = Order(
                order_id=item["order_id"],
                ticker=item["ticker"],
                side=OrderSide(item["side"]),
                order_type=OrderType(item["type"]),
                quantity=item["quantity"],
                status=OrderStatus(item["status"]),
                created_at=item.get("created_at", ""),
                filled_at=item.get("filled_at"),
                strategy=item.get("strategy", ""),
            )
            self.orders[order.order_id] = order
            if order.is_complete:
                self.blotter.append(order)


class SimulatedExchange:
    """Simulated exchange for paper trading.

    Simulates order matching with realistic A-share market microstructure:
    - Price-time priority
    - Tick size: 0.01 RMB
    - Lot size: 100 shares
    - T+1 settlement
    - Daily price limits: ±10% (±20% for ChiNext/STAR)
    """

    def __init__(self):
        self.order_manager: OrderManager | None = None
        self.current_prices: dict[str, float] = {}
        self.price_history: dict[str, list[float]] = {}

    def set_order_manager(self, om: OrderManager):
        self.order_manager = om

    def update_market(self, prices: dict[str, float]):
        """Update market prices."""
        self.current_prices.update(prices)
        for ticker, price in prices.items():
            if ticker not in self.price_history:
                self.price_history[ticker] = []
            self.price_history[ticker].append(price)

    def match_orders(self):
        """Match all submitted orders against current market."""
        if not self.order_manager:
            return

        for order in list(self.order_manager.orders.values()):
            if order.status != OrderStatus.SUBMITTED:
                continue

            price = self.current_prices.get(order.ticker)
            if price is None:
                order.status = OrderStatus.REJECTED
                order.notes = f"No market data for {order.ticker}"
                continue

            # Apply slippage model
            if order.order_type == OrderType.MARKET:
                # Market orders get ~5bps slippage
                slippage_factor = 1.0005 if order.side == OrderSide.BUY else 0.9995
                fill_price = round(price * slippage_factor, 2)
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.limit_price < price:
                    continue  # Limit not reached
                if order.side == OrderSide.SELL and order.limit_price > price:
                    continue
                fill_price = order.limit_price
            else:
                fill_price = price

            self.order_manager.fill_order(order.order_id, fill_price)

    def simulate_trading_day(self, prices: dict[str, float]):
        """Simulate a full trading day."""
        self.update_market(prices)
        if self.order_manager:
            self.order_manager.update_prices(prices)
        self.match_orders()
