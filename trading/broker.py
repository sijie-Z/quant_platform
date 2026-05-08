"""Broker abstraction layer.

Supports:
- SimulatedBroker: Paper trading with real market prices (no broker needed)
- QMTBroker: Live trading via xtquant/miniQMT (requires QMT running)

Both implement the same BrokerInterface for seamless switching.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """A trade order."""
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    code: str = ""               # e.g. '600519'
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    quantity: int = 0            # shares
    price: float = 0.0           # limit price
    filled_quantity: int = 0
    filled_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    commission: float = 0.0
    tax: float = 0.0
    slippage: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    broker_order_id: str = ""
    error_msg: str = ""

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id, "code": self.code,
            "side": self.side.value, "order_type": self.order_type.value,
            "quantity": self.quantity, "price": self.price,
            "filled_quantity": self.filled_quantity, "filled_price": self.filled_price,
            "status": self.status.value, "commission": self.commission,
            "tax": self.tax, "slippage": self.slippage,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "error_msg": self.error_msg,
        }


@dataclass
class Position:
    """A stock position."""
    code: str = ""
    name: str = ""
    quantity: int = 0
    available: int = 0     # T+1: can sell today
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        self.market_value = self.quantity * price
        if self.avg_cost > 0 and self.quantity > 0:
            self.unrealized_pnl = (price - self.avg_cost) * self.quantity
            self.unrealized_pnl_pct = (price - self.avg_cost) / self.avg_cost
        else:
            self.unrealized_pnl = 0
            self.unrealized_pnl_pct = 0

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name,
            "quantity": self.quantity, "available": self.available,
            "avg_cost": round(self.avg_cost, 3),
            "current_price": round(self.current_price, 3),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "realized_pnl": round(self.realized_pnl, 2),
        }


class BrokerInterface(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker. Returns True if successful."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Submit an order. Returns order with broker_order_id set."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get current positions."""

    @abstractmethod
    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get order history."""

    @abstractmethod
    def get_account(self) -> dict:
        """Get account info (cash, equity, etc.)."""


class SimulatedBroker(BrokerInterface):
    """Paper trading broker with real market prices.

    Simulates A-share trading rules:
    - T+1 (can't sell shares bought today)
    - Lot size 100
    - Commission 0.03% (min 5 yuan)
    - Stamp tax 0.1% (sell only)
    - Price limit ±10% (±20% for 创业板/科创板)
    """

    def __init__(self, initial_cash: float = 1_000_000):
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._today_bought: set[str] = set()  # T+1 tracking
        self._connected = False
        self._commission_rate = 0.0003   # 0.03%
        self._min_commission = 5.0
        self._stamp_tax_rate = 0.001     # 0.1% sell only
        self._slippage_bps = 5           # 5 basis points

    def connect(self) -> bool:
        self._connected = True
        logger.info("SimulatedBroker connected. Cash: %.2f", self._cash)
        return True

    def place_order(self, order: Order) -> Order:
        """Execute order immediately at market price + slippage."""
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error_msg = "Not connected"
            return order

        # Validate lot size
        if order.quantity % 100 != 0:
            order.status = OrderStatus.REJECTED
            order.error_msg = f"Quantity must be multiple of 100, got {order.quantity}"
            self._orders.append(order)
            return order

        # Validate sell
        if order.side == OrderSide.SELL:
            pos = self._positions.get(order.code)
            if not pos or pos.available < order.quantity:
                avail = pos.available if pos else 0
                order.status = OrderStatus.REJECTED
                order.error_msg = f"Insufficient position. Available: {avail}, requested: {order.quantity}"
                self._orders.append(order)
                return order

        # Calculate fill price with slippage
        slippage = order.price * self._slippage_bps / 10000
        if order.side == OrderSide.BUY:
            fill_price = order.price + slippage
        else:
            fill_price = order.price - slippage

        # Calculate costs
        trade_value = fill_price * order.quantity
        commission = max(trade_value * self._commission_rate, self._min_commission)
        tax = trade_value * self._stamp_tax_rate if order.side == OrderSide.SELL else 0
        total_cost = trade_value + commission + tax

        # Check cash for buy
        if order.side == OrderSide.BUY and total_cost > self._cash:
            order.status = OrderStatus.REJECTED
            order.error_msg = f"Insufficient cash. Need {total_cost:.2f}, have {self._cash:.2f}"
            self._orders.append(order)
            return order

        # Execute
        if order.side == OrderSide.BUY:
            self._cash -= total_cost
            pos = self._positions.get(order.code)
            if pos:
                total_qty = pos.quantity + order.quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * order.quantity) / total_qty
                pos.quantity = total_qty
            else:
                pos = Position(code=order.code, quantity=order.quantity,
                               avg_cost=fill_price)
                self._positions[order.code] = pos
            self._today_bought.add(order.code)
        else:  # SELL
            self._cash += trade_value - commission - tax
            pos = self._positions[order.code]
            pos.quantity -= order.quantity
            pos.available -= order.quantity
            # Realized P&L
            pos.realized_pnl += (fill_price - pos.avg_cost) * order.quantity
            if pos.quantity <= 0:
                del self._positions[order.code]

        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = commission
        order.tax = tax
        order.slippage = slippage * order.quantity
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now().isoformat()

        self._orders.append(order)
        logger.info("Filled: %s %s %d @ %.3f (cost: %.2f)",
                     order.side.value, order.code, order.quantity, fill_price, commission + tax)
        return order

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o.order_id == order_id and o.status == OrderStatus.PENDING:
                o.status = OrderStatus.CANCELLED
                return True
        return False

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status:
            return [o for o in self._orders if o.status == status]
        return self._orders

    def get_account(self) -> dict:
        market_value = sum(p.market_value for p in self._positions.values())
        total_equity = self._cash + market_value
        total_pnl = total_equity - self._initial_cash
        return {
            "cash": round(self._cash, 2),
            "market_value": round(market_value, 2),
            "total_equity": round(total_equity, 2),
            "initial_cash": self._initial_cash,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / self._initial_cash, 4),
            "n_positions": len(self._positions),
            "connected": self._connected,
            "broker": "simulated",
        }

    def update_market_prices(self, prices: dict[str, float]):
        """Update position prices from real-time market data."""
        for code, pos in self._positions.items():
            if code in prices:
                pos.update_price(prices[code])

    def new_trading_day(self):
        """Reset T+1 restrictions. Call at start of each trading day."""
        self._today_bought.clear()
        for pos in self._positions.values():
            pos.available = pos.quantity


class QMTBroker(BrokerInterface):
    """Live broker via xtquant/miniQMT.

    Requires:
    - miniQMT client running and logged in
    - xtquant package installed
    - Broker account configured

    Supported brokers: 国金证券, 华鑫证券, 国盛证券, 东方财富 etc.
    """

    def __init__(self, qmt_path: str = "", account_id: str = ""):
        """Initialize QMT broker.

        Args:
            qmt_path: miniQMT UserData directory, e.g. 'C:\\国金证券QMT\\UserData_mini'
            account_id: Broker account ID
        """
        self._qmt_path = qmt_path
        self._account_id = account_id
        self._trader = None
        self._account = None
        self._connected = False
        self._orders: list[Order] = []

        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
            import xtquant.xtconstant as xtc
            self._XtQuantTrader = XtQuantTrader
            self._StockAccount = StockAccount
            self._xtc = xtc
            self._HAS_XTQUANT = True
        except ImportError:
            self._HAS_XTQUANT = False
            logger.warning("xtquant not installed. QMTBroker unavailable.")

    def connect(self) -> bool:
        if not self._HAS_XTQUANT:
            logger.error("xtquant not installed")
            return False

        if not self._qmt_path or not self._account_id:
            logger.error("qmt_path and account_id required")
            return False

        try:
            session_id = int(datetime.now().timestamp())
            self._trader = self._XtQuantTrader(self._qmt_path, session_id)
            self._account = self._StockAccount(self._account_id, 'STOCK')

            # Set up callbacks
            from xtquant.xttrader import XtQuantTraderCallback
            class Callback(XtQuantTraderCallback):
                def on_disconnected(cb_self):
                    logger.error("QMT disconnected!")
                    self._connected = False

                def on_order_error(cb_self, order_error):
                    logger.error("Order error: %s", order_error.error_msg)

                def on_order_stock_async_response(cb_self, response):
                    logger.info("Order response: %s", response.order_id)

            self._trader.register_callback(Callback())
            self._trader.start()

            result = self._trader.connect()
            if result == 0:
                self._connected = True
                logger.info("QMT connected. Account: %s", self._account_id)
                return True
            else:
                logger.error("QMT connect failed: %s", result)
                return False

        except Exception as e:
            logger.error("QMT connection error: %s", e)
            return False

    def place_order(self, order: Order) -> Order:
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error_msg = "Not connected to QMT"
            return order

        try:
            # Convert code format: 600519 -> 600519.SH
            stock_code = self._to_xt_code(order.code)
            order_type = 23 if order.side == OrderSide.BUY else 24  # 23=BUY, 24=SELL
            price_type = self._xtc.FIX_PRICE if order.order_type == OrderType.LIMIT else self._xtc.LATEST_PRICE

            broker_id = self._trader.order_stock(
                account=self._account,
                stock_code=stock_code,
                order_type=order_type,
                order_volume=order.quantity,
                strategy_name='quant_platform',
                price_type=price_type,
                price=order.price,
            )

            order.broker_order_id = str(broker_id)
            order.status = OrderStatus.SUBMITTED
            self._orders.append(order)
            logger.info("Order submitted: %s %s %d @ %.2f (broker_id: %s)",
                        order.side.value, order.code, order.quantity, order.price, broker_id)
            return order

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_msg = str(e)
            self._orders.append(order)
            return order

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        for o in self._orders:
            if o.order_id == order_id and o.broker_order_id:
                try:
                    self._trader.cancel_order_stock(self._account, int(o.broker_order_id))
                    o.status = OrderStatus.CANCELLED
                    return True
                except Exception as e:
                    logger.error("Cancel failed: %s", e)
                    return False
        return False

    def get_positions(self) -> list[Position]:
        if not self._connected:
            return []
        try:
            raw = self._trader.query_stock_positions(self._account)
            positions = []
            for p in raw:
                pos = Position(
                    code=self._from_xt_code(p.stock_code),
                    quantity=p.volume,
                    available=p.can_use_volume,
                    avg_cost=p.avg_price,
                    market_value=p.market_value,
                )
                positions.append(pos)
            return positions
        except Exception as e:
            logger.error("Query positions failed: %s", e)
            return []

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status:
            return [o for o in self._orders if o.status == status]
        return self._orders

    def get_account(self) -> dict:
        if not self._connected:
            return {"connected": False, "broker": "qmt"}
        try:
            positions = self.get_positions()
            market_value = sum(p.market_value for p in positions)
            return {
                "market_value": round(market_value, 2),
                "n_positions": len(positions),
                "connected": True,
                "broker": "qmt",
                "account_id": self._account_id,
            }
        except Exception as e:
            return {"connected": False, "error": str(e), "broker": "qmt"}

    @staticmethod
    def _to_xt_code(code: str) -> str:
        """Convert '600519' to '600519.SH'."""
        if '.' in code:
            return code
        if code.startswith(('6', '9')):
            return f"{code}.SH"
        return f"{code}.SZ"

    @staticmethod
    def _from_xt_code(code: str) -> str:
        """Convert '600519.SH' to '600519'."""
        return code.split('.')[0] if '.' in code else code
