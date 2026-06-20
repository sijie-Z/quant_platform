"""Unified protocol layer — JSON-safe data interchange objects.

Every module in the platform should communicate through these protocol
objects rather than raw dicts. This ensures:
- Stable interfaces between modules (research ↔ trading ↔ API ↔ MCP)
- Validation at boundary crossings (no bad data propagates silently)
- JSON-serializable for API/Agent/MCP export

Protocol objects support:
- to_dict() / from_dict() for serialization
- validate() for boundary checks
- Immutable-style frozen dataclasses

Inspired by quawn's protocol layer design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


# ── Enums ──


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class RecommendationAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    INCREASE = "INCREASE"


class RiskLevel(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    KILL = "kill"


# ── Protocol Objects ──


@dataclass(frozen=True)
class AccountState:
    """Account-level financial state."""
    cash: float = 0.0
    equity: float = 0.0
    market_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    initial_cash: float = 0.0
    n_positions: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "equity": round(self.equity, 2),
            "market_value": round(self.market_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "initial_cash": round(self.initial_cash, 2),
            "n_positions": self.n_positions,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AccountState:
        return cls(
            cash=d.get("cash", 0.0),
            equity=d.get("equity", 0.0),
            market_value=d.get("market_value", 0.0),
            total_pnl=d.get("total_pnl", 0.0),
            total_pnl_pct=d.get("total_pnl_pct", 0.0),
            initial_cash=d.get("initial_cash", 0.0),
            n_positions=d.get("n_positions", 0),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            metadata=d.get("metadata", {}),
        )

    def validate(self) -> list[str]:
        errors = []
        if self.cash < -0.01:
            errors.append(f"cash is negative: {self.cash}")
        if self.equity < -0.01:
            errors.append(f"equity is negative: {self.equity}")
        if abs(self.cash + self.market_value - self.equity) > 0.02 * max(self.equity, 1):
            errors.append("account reconciliation: cash + market_value != equity")
        return errors


@dataclass(frozen=True)
class Position:
    """A single stock/asset position."""
    code: str = ""
    name: str = ""
    quantity: int = 0
    available: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    weight: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "quantity": self.quantity,
            "available": self.available,
            "avg_cost": round(self.avg_cost, 3),
            "current_price": round(self.current_price, 3),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "weight": round(self.weight, 4),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Position:
        return cls(**d)

    def validate(self) -> list[str]:
        errors = []
        if self.quantity < 0:
            errors.append(f"negative quantity for {self.code}: {self.quantity}")
        if self.available < 0:
            errors.append(f"negative available for {self.code}: {self.available}")
        if self.available > self.quantity:
            errors.append(f"available > quantity for {self.code}")
        return errors


@dataclass(frozen=True)
class Order:
    """A trade order."""
    order_id: str = ""
    code: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0
    price: float = 0.0
    filled_quantity: int = 0
    filled_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    commission: float = 0.0
    tax: float = 0.0
    slippage: float = 0.0
    strategy_id: str = ""
    signal_id: str = ""
    error_msg: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "code": self.code,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": round(self.price, 3),
            "filled_quantity": self.filled_quantity,
            "filled_price": round(self.filled_price, 3),
            "status": self.status.value,
            "commission": round(self.commission, 4),
            "tax": round(self.tax, 4),
            "slippage": round(self.slippage, 4),
            "strategy_id": self.strategy_id,
            "signal_id": self.signal_id,
            "error_msg": self.error_msg,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Order:
        kwargs = dict(d)
        if "side" in kwargs and isinstance(kwargs["side"], str):
            kwargs["side"] = OrderSide(kwargs["side"])
        if "order_type" in kwargs and isinstance(kwargs["order_type"], str):
            kwargs["order_type"] = OrderType(kwargs["order_type"])
        if "status" in kwargs and isinstance(kwargs["status"], str):
            kwargs["status"] = OrderStatus(kwargs["status"])
        return cls(**kwargs)

    def validate(self) -> list[str]:
        errors = []
        if self.quantity <= 0:
            errors.append(f"quantity must be positive: {self.quantity}")
        if self.price <= 0:
            errors.append(f"price must be positive: {self.price}")
        if self.filled_quantity > self.quantity:
            errors.append("filled_quantity > quantity")
        return errors


@dataclass(frozen=True)
class Fill:
    """An executed trade fill."""
    fill_id: str = ""
    order_id: str = ""
    code: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    signal_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "code": self.code,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": round(self.price, 3),
            "commission": round(self.commission, 4),
            "tax": round(self.tax, 4),
            "executed_at": self.executed_at,
            "signal_date": self.signal_date,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Fill:
        kwargs = dict(d)
        if "side" in kwargs and isinstance(kwargs["side"], str):
            kwargs["side"] = OrderSide(kwargs["side"])
        return cls(**kwargs)

    def validate(self) -> list[str]:
        errors = []
        if self.quantity <= 0:
            errors.append(f"fill quantity must be positive: {self.quantity}")
        if self.price <= 0:
            errors.append(f"fill price must be positive: {self.price}")
        return errors


@dataclass(frozen=True)
class Signal:
    """An alpha signal for one stock at one point in time."""
    code: str = ""
    direction: str = ""  # long / short / neutral
    strength: float = 0.0
    signal_date: str = ""
    factor_values: dict[str, float] = field(default_factory=dict)
    strategy_id: str = ""
    signal_id: str = ""
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "signal_date": self.signal_date,
            "factor_values": {k: round(v, 6) for k, v in self.factor_values.items()},
            "strategy_id": self.strategy_id,
            "signal_id": self.signal_id,
            "confidence": round(self.confidence, 4),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Signal:
        return cls(**d)

    def validate(self) -> list[str]:
        errors = []
        if not self.code:
            errors.append("signal missing code")
        if not (-1 <= self.strength <= 1):
            errors.append(f"signal strength out of range [-1, 1]: {self.strength}")
        return errors


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    timestamp: str = ""
    total_equity: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    cumulative_pnl: float = 0.0
    n_positions: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_equity": round(self.total_equity, 2),
            "cash": round(self.cash, 2),
            "market_value": round(self.market_value, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "n_positions": self.n_positions,
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PortfolioSnapshot:
        return cls(**d)

    def validate(self) -> list[str]:
        errors = []
        if self.cash < -0.01:
            errors.append(f"negative cash: {self.cash}")
        if abs(self.cash + self.market_value - self.total_equity) > 0.02 * max(self.total_equity, 1):
            errors.append("reconciliation: cash + market_value != total_equity")
        return errors


@dataclass(frozen=True)
class Recommendation:
    """Deterministic BUY/SELL/HOLD recommendation."""
    code: str = ""
    action: RecommendationAction = RecommendationAction.HOLD
    quantity: int = 0
    price: float = 0.0
    reason: str = ""
    confidence: float = 0.0
    signal_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "action": self.action.value,
            "quantity": self.quantity,
            "price": round(self.price, 3),
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "signal_date": self.signal_date,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Recommendation:
        kwargs = dict(d)
        if "action" in kwargs and isinstance(kwargs["action"], str):
            kwargs["action"] = RecommendationAction(kwargs["action"])
        return cls(**kwargs)

    def validate(self) -> list[str]:
        errors = []
        if not self.code:
            errors.append("recommendation missing code")
        if self.quantity < 0:
            errors.append(f"negative quantity: {self.quantity}")
        return errors


# ── Batch Validation ──


def validate_protocol(name: str, obj: Any) -> list[str]:
    """Validate any protocol object safely."""
    validate = getattr(obj, "validate", None)
    if callable(validate):
        return list(validate())
    return [f"{name} does not expose validate()"]


def validate_weights(weights: dict[str, float], tolerance: float = 0.02) -> list[str]:
    """Check that allocation weights are valid."""
    if not weights:
        return []
    total = sum(weights.values())
    errors = []
    if abs(total - 1.0) > tolerance:
        errors.append(f"weights sum to {total:.4f}, expected 1.0 ± {tolerance}")
    if any(v < -tolerance for v in weights.values()):
        errors.append("weights contain negative values")
    return errors
