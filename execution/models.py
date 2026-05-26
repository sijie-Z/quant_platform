"""Order Management System — models for orders, fills, and positions.

Modeled after institutional OMS: order blotter, execution tracking,
realized/unrealized P&L, and trade cost analysis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"          # Time-Weighted Average Price
    VWAP = "vwap"          # Volume-Weighted Average Price
    ICEBERG = "iceberg"    # Hidden quantity


class OrderStatus(StrEnum):
    PENDING = "pending"        # Created, not yet sent
    SUBMITTED = "submitted"    # Sent to broker/exchange
    PARTIAL = "partial"        # Partially filled
    FILLED = "filled"          # Fully filled
    CANCELLED = "cancelled"    # Cancelled before fill
    REJECTED = "rejected"      # Rejected by broker
    EXPIRED = "expired"        # Expired (e.g., day order end)


@dataclass
class Fill:
    """Single execution fill."""
    fill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = ""
    price: float = 0.0
    quantity: int = 0
    commission: float = 0.0
    tax: float = 0.0          # Stamp tax (sell-side only in A-share)
    slippage: float = 0.0


@dataclass
class Order:
    """Single order in the OMS blotter."""
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tenant_id: str = "default"
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: str = ""
    submitted_at: str | None = None
    filled_at: str | None = None
    fills: list[Fill] = field(default_factory=list)
    strategy: str = ""        # Which strategy generated this order
    parent_order_id: str | None = None  # For child orders (TWAP/VWAP)
    notes: str = ""

    @property
    def filled_quantity(self) -> int:
        return sum(f.quantity for f in self.fills)

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def avg_fill_price(self) -> float:
        if not self.fills:
            return 0.0
        total_cost = sum(f.price * f.quantity for f in self.fills)
        total_qty = sum(f.quantity for f in self.fills)
        return total_cost / total_qty if total_qty > 0 else 0.0

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def total_tax(self) -> float:
        return sum(f.tax for f in self.fills)

    @property
    def total_slippage(self) -> float:
        return sum(f.slippage for f in self.fills)

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED, OrderStatus.EXPIRED)


@dataclass
class Position:
    """Current position for a single stock."""
    ticker: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    sector: str = ""
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    weight: float = 0.0
    last_updated: str = ""

    def update_price(self, price: float):
        self.current_price = price
        self.market_value = self.quantity * price
        if self.avg_cost > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity
            self.unrealized_pnl_pct = (price / self.avg_cost - 1) * 100
        self.total_pnl = self.realized_pnl + self.unrealized_pnl


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: str = ""
    total_value: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    n_positions: int = 0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    positions: list[Position] = field(default_factory=list)
