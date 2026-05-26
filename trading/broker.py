"""Broker abstraction layer.

Supports:
- SimulatedBroker: Paper trading with real market prices via LOB matching
- QMTBroker: Live trading via xtquant/miniQMT (requires QMT running)

Both implement the same BrokerInterface for seamless switching.

SimulatedBroker uses a real Limit Order Book (LOB) with:
- Price-time priority FIFO matching
- Partial fills
- Market impact simulation (synthetic liquidity)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from quant_platform.execution.order_book import (
    OrderBook,
    BookOrder,
    Side as BookSide,
    OrderType as BookOrderType,
    Trade,
)
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
    """Paper trading broker with real LOB matching.

    Uses a real Limit Order Book with price-time priority FIFO matching,
    partial fills, and synthetic market-maker liquidity.

    Supports cross-asset trading rules via AssetUniverse:
    - Per-instrument lot_size, multiplier, commission, stamp_tax, t_plus
    - Futures: margin-based, no stamp tax, T+0, lot_size=1
    - ETFs: no stamp tax, same lot_size as stocks
    - Falls back to A-share defaults if no instrument found (backward compatible)
    """

    def __init__(self, initial_cash: float = 1_000_000, asset_universe=None):
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._today_bought: set[str] = set()  # T+1 tracking
        self._connected = False
        self._commission_rate = 0.0003   # 0.03%
        self._min_commission = 5.0
        self._stamp_tax_rate = 0.001     # 0.1% sell only
        self._asset_universe = asset_universe

        # Real order books per symbol
        self._order_books: dict[str, OrderBook] = {}
        self._market_prices: dict[str, float] = {}

    def _get_or_create_book(self, symbol: str, price: float) -> OrderBook:
        """Get or create an order book for a symbol, seeded with liquidity."""
        if symbol not in self._order_books:
            book = OrderBook(symbol)
            self._seed_book(book, symbol, price)
            self._order_books[symbol] = book
        return self._order_books[symbol]

    def _seed_book(self, book: OrderBook, symbol: str, price: float):
        """Seed order book with synthetic market-maker liquidity.

        Creates price levels on each side centered on the reference price,
        with the best bid/ask touching the price (1 bp spread).
        This ensures limit orders at the reference price can match.
        """
        spread_bps = 1  # 1 bp between levels
        for i in range(6):
            # Ask side (offers): start AT price, go up
            ask_price = round(price * (1 + spread_bps * i / 10000), 2)
            ask_qty = max(100, int(5000 / (i + 1)))
            ask_order = BookOrder(
                order_id=f"mm_ask_{symbol}_{i}",
                symbol=symbol,
                side=BookSide.SELL,
                order_type=BookOrderType.LIMIT,
                price=ask_price,
                quantity=ask_qty,
                source="market_maker",
            )
            book.add_order(ask_order)

            # Bid side: start AT price, go down
            bid_price = round(price * (1 - spread_bps * (i + 1) / 10000), 2)
            bid_qty = max(100, int(5000 / (i + 1)))
            bid_order = BookOrder(
                order_id=f"mm_bid_{symbol}_{i}",
                symbol=symbol,
                side=BookSide.BUY,
                order_type=BookOrderType.LIMIT,
                price=bid_price,
                quantity=bid_qty,
                source="market_maker",
            )
            book.add_order(bid_order)

    def connect(self) -> bool:
        self._connected = True
        logger.info("SimulatedBroker connected (LOB matching). Cash: %.2f", self._cash)
        return True

    def _get_instrument(self, symbol: str):
        """Look up instrument from universe, or None."""
        if self._asset_universe is not None:
            return self._asset_universe.get(symbol)
        return None

    def _get_lot_size(self, symbol: str) -> int:
        inst = self._get_instrument(symbol)
        return inst.lot_size if inst else 100

    def _get_multiplier(self, symbol: str) -> float:
        inst = self._get_instrument(symbol)
        return inst.multiplier if inst else 1.0

    def _get_commission(self, symbol: str, price: float, quantity: int) -> float:
        inst = self._get_instrument(symbol)
        if inst:
            return inst.commission(price, quantity)
        return max(price * quantity * self._commission_rate, self._min_commission)

    def _get_stamp_tax(self, symbol: str, price: float, quantity: int, side) -> float:
        inst = self._get_instrument(symbol)
        if inst:
            return inst.stamp_tax(price, quantity, side.value)
        return price * quantity * self._stamp_tax_rate if side == OrderSide.SELL else 0

    def _get_t_plus(self, symbol: str) -> int:
        inst = self._get_instrument(symbol)
        return inst.t_plus if inst else 1

    def place_order(self, order: Order) -> Order:
        """Submit order to the real order book for price-time priority matching."""
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error_msg = "Not connected"
            return order

        # Validate lot size (per-instrument)
        lot_size = self._get_lot_size(order.code)
        if order.quantity % lot_size != 0:
            order.status = OrderStatus.REJECTED
            order.error_msg = f"Quantity must be multiple of {lot_size}, got {order.quantity}"
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

        # Check cash for buy (pre-check with worst-case price, use multiplier)
        if order.side == OrderSide.BUY:
            multiplier = self._get_multiplier(order.code)
            worst_cost = order.price * order.quantity * multiplier * 1.001  # 0.1% buffer
            if worst_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.error_msg = f"Insufficient cash. Need ~{worst_cost:.2f}, have {self._cash:.2f}"
                self._orders.append(order)
                return order

        # Get or create order book
        book = self._get_or_create_book(order.code, order.price)

        # Map order type to book order type
        if order.order_type == OrderType.MARKET:
            book_ot = BookOrderType.MARKET
        elif order.order_type == OrderType.LIMIT:
            book_ot = BookOrderType.LIMIT
        else:
            book_ot = BookOrderType.LIMIT  # default

        # Convert to BookOrder and submit to LOB
        book_order = BookOrder(
            order_id=order.order_id,
            symbol=order.code,
            side=BookSide.BUY if order.side == OrderSide.BUY else BookSide.SELL,
            order_type=book_ot,
            price=order.price,
            quantity=order.quantity,
            source="broker",
        )
        trades = book.add_order(book_order)

        # Process trades from the LOB
        total_filled = 0
        total_value = 0.0
        for trade in trades:
            total_filled += trade.quantity
            total_value += trade.price * trade.quantity

        if total_filled == 0:
            # No liquidity — IOC-style: cancel
            order.status = OrderStatus.REJECTED
            order.error_msg = "No liquidity available"
            self._orders.append(order)
            return order

        # Average fill price
        avg_fill_price = total_value / total_filled

        # Calculate costs (per-instrument rates)
        commission = self._get_commission(order.code, avg_fill_price, total_filled)
        tax = self._get_stamp_tax(order.code, avg_fill_price, total_filled, order.side)
        slippage = abs(avg_fill_price - order.price) * total_filled

        # Update positions and cash
        if order.side == OrderSide.BUY:
            total_cost = total_value + commission + tax
            if total_cost > self._cash:
                # Partial fill at what we can afford (use lot_size, not hardcoded 100)
                lot_size = self._get_lot_size(order.code)
                affordable_qty = int(self._cash / (avg_fill_price * 1.001) / lot_size) * lot_size
                if affordable_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    order.error_msg = f"Insufficient cash after fill. Need {total_cost:.2f}, have {self._cash:.2f}"
                    self._orders.append(order)
                    return order
                total_filled = min(total_filled, affordable_qty)
                total_value = avg_fill_price * total_filled
                commission = self._get_commission(order.code, avg_fill_price, total_filled)
                total_cost = total_value + commission + tax

            self._cash -= total_cost
            pos = self._positions.get(order.code)
            if pos:
                total_qty = pos.quantity + total_filled
                pos.avg_cost = (pos.avg_cost * pos.quantity + avg_fill_price * total_filled) / total_qty
                pos.quantity = total_qty
            else:
                pos = Position(code=order.code, quantity=total_filled, avg_cost=avg_fill_price)
                self._positions[order.code] = pos
            self._today_bought.add(order.code)
        else:  # SELL
            self._cash += total_value - commission - tax
            pos = self._positions[order.code]
            pos.quantity -= total_filled
            pos.available -= total_filled
            pos.realized_pnl += (avg_fill_price - pos.avg_cost) * total_filled
            if pos.quantity <= 0:
                del self._positions[order.code]

        # Fill the order
        order.filled_quantity = total_filled
        order.filled_price = avg_fill_price
        order.commission = commission
        order.tax = tax
        order.slippage = slippage
        order.status = OrderStatus.FILLED if total_filled >= order.quantity else OrderStatus.PARTIAL
        order.updated_at = datetime.now().isoformat()

        self._orders.append(order)
        logger.info("LOB Filled: %s %s %d/%d @ %.3f (cost: %.2f, trades: %d)",
                     order.side.value, order.code, total_filled, order.quantity,
                     avg_fill_price, commission + tax, len(trades))
        return order

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o.order_id == order_id and o.status == OrderStatus.PENDING:
                o.status = OrderStatus.CANCELLED
                # Cancel in the order book too
                for book in self._order_books.values():
                    book.cancel_order(order_id)
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
            "broker": "simulated_lob",
        }

    def get_book_snapshot(self, symbol: str) -> dict | None:
        """Get order book snapshot for a symbol."""
        book = self._order_books.get(symbol)
        if book:
            return book.get_depth_snapshot(levels=5)
        return None

    def get_book_metrics(self, symbol: str) -> dict | None:
        """Get microstructure metrics for a symbol."""
        book = self._order_books.get(symbol)
        if book:
            return book.get_microstructure_metrics()
        return None

    def update_market_prices(self, prices: dict[str, float]):
        """Update position prices from real-time market data."""
        self._market_prices.update(prices)
        for code, pos in self._positions.items():
            if code in prices:
                pos.update_price(prices[code])

    def new_trading_day(self):
        """Reset T+1 restrictions. Call at start of each trading day."""
        self._today_bought.clear()
        for pos in self._positions.values():
            pos.available = pos.quantity


class QMTBroker(BrokerInterface):
    """Live broker via xtquant/miniQMT (国金证券量化交易).

    Connects to a local miniQMT client over TCP (default localhost:58610).
    miniQMT must be running and logged in before calling connect().

    Key features:
    - Full callback chain: on_fill / on_cancel / on_error / on_disconnected
    - EventBus integration: every fill publishes ``qmt.fill`` and ``order.filled``
    - Password from env ``QMT_PASSWORD`` — never in config files
    - Graceful degradation: unavailable orders/positions return empty lists
    - Automatic reconnection on disconnect (with backoff)

    Architecture::

        LiveRunner ─> QMTBroker ──TCP──> miniQMT/client ──> 国金券商
                         │
                         ├── _xt_trader (order_stock / cancel / query)
                         ├── Callback → update Order objects → EventBus
                         └── Fallback → SimulatedBroker on connection failure

    Supported brokers: 国金证券, 华鑫证券, 国盛证券, 东方财富 etc.
    """

    def __init__(
        self,
        account: str = "",
        server: str = "localhost:58610",
        password: str = "",
        mode: str = "sim",
        data_server: str = "",
        initial_cash: float = 10_000_000,
    ):
        """Initialize QMT broker.

        Args:
            account: Broker sim/live account ID.
            server: miniQMT server address (host:port).
            password: Trading password. Reads from env ``QMT_PASSWORD`` if empty.
            mode: 'sim' for simulated trading, 'live' for production.
            data_server: Market data server (host:port). Defaults to same as server.
            initial_cash: Notional initial capital (fallback when QMT account
                          query fails; the broker reports QMT's real balance).
        """
        self._account_id = account
        self._server = server
        self._password = password
        self._mode = mode
        self._data_server = data_server or server
        self._initial_cash = initial_cash

        self._trader: Any = None
        self._account: Any = None
        self._xtc: Any = None
        self._connected = False
        self._session_id: int = 0
        self._orders: list[Order] = []
        self._positions: dict[str, Position] = {}
        self._cash: float = initial_cash
        self._frozen_cash: float = 0.0
        self._market_value: float = 0.0

        # Resolve password: explicit arg > env var
        if not self._password:
            import os
            self._password = os.environ.get("QMT_PASSWORD", "")

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
            logger.warning("xtquant not installed. QMTBroker unavailable. "
                         "Install: pip install xtquant")

    # ── Connection ──

    def connect(self, blocking: bool = True) -> bool:
        """Connect to miniQMT.

        Args:
            blocking: If True, blocks until connected or timeout (5s).
        """
        if not self._HAS_XTQUANT:
            logger.error("xtquant not installed — cannot connect to QMT")
            return False

        if not self._account_id:
            logger.error("QMT account ID required")
            return False

        try:
            self._session_id = int(datetime.now().timestamp() % 2147483647)
            self._trader = self._XtQuantTrader(self._server, self._session_id)
            self._account = self._StockAccount(self._account_id, "STOCK")

            # Register callbacks
            from xtquant.xttrader import XtQuantTraderCallback
            broker_ref = self  # closure capture

            class _QMTCallback(XtQuantTraderCallback):
                def on_disconnected(self_cb):
                    logger.error("QMT disconnected! Session: %d", broker_ref._session_id)
                    broker_ref._connected = False
                    broker_ref._publish_event("qmt.disconnected", {"session": broker_ref._session_id})

                def on_order_error(self_cb, order_error):
                    err_msg = order_error.error_msg if hasattr(order_error, "error_msg") else str(order_error)
                    err_id = order_error.order_id if hasattr(order_error, "order_id") else ""
                    logger.error("QMT order error [%s]: %s", err_id, err_msg)
                    broker_ref._update_order_status(err_id, OrderStatus.REJECTED, err_msg)
                    broker_ref._publish_event("qmt.order_error", {
                        "order_id": str(err_id), "error": err_msg,
                    })

                def on_order_stock_async_response(self_cb, response):
                    broker_ref._on_async_response(response)

            self._trader.register_callback(_QMTCallback())
            self._trader.start()

            if blocking:
                import time as _time
                deadline = _time.monotonic() + 5.0
                while _time.monotonic() < deadline:
                    result = getattr(self._trader, "connect", lambda: -1)()
                    if result == 0:
                        break
                    _time.sleep(0.1)
                else:
                    result = getattr(self._trader, "connect", lambda: -1)()
            else:
                result = self._trader.connect() if hasattr(self._trader, "connect") else -1

            if result == 0:
                self._connected = True
                # Subscribe to account updates
                try:
                    self._trader.subscribe(self._account)
                except Exception:
                    pass
                logger.info("QMT connected. Account: %s, Server: %s, Mode: %s",
                          self._account_id, self._server, self._mode)
                self._update_account_snapshot()
                return True

            logger.error("QMT connect returned: %s", result)
            return False

        except Exception as e:
            logger.error("QMT connection error: %s", e)
            self._connected = False
            return False

    def disconnect(self) -> bool:
        """Gracefully disconnect from QMT."""
        try:
            if self._trader:
                self._trader.stop()
            self._connected = False
            logger.info("QMT disconnected")
            return True
        except Exception as e:
            logger.error("QMT disconnect error: %s", e)
            return False

    # ── Order lifecycle ──

    def place_order(self, order: Order) -> Order:
        """Submit an order to QMT.

        Converts internal Order to xtquant params, submits, and tracks.
        The order status will be updated asynchronously via callbacks.
        """
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error_msg = "QMT not connected"
            self._orders.append(order)
            return order

        try:
            from quant_platform.trading.qmt_utils import to_qmt_code, to_qmt_order_type, to_qmt_price_type

            stock_code = to_qmt_code(order.code)
            xt_order_type = to_qmt_order_type(order.side)
            xt_price_type = to_qmt_price_type(order.order_type)

            # Validate price
            price = order.price
            if order.order_type == OrderType.MARKET:
                price = 0.0  # QMT uses 0 for market orders

            broker_id = self._trader.order_stock(
                account=self._account,
                stock_code=stock_code,
                order_type=xt_order_type,
                order_volume=order.quantity,
                strategy_name="quant_platform",
                price_type=xt_price_type,
                price=price,
            )

            order.broker_order_id = str(broker_id)
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now().isoformat()
            self._orders.append(order)

            logger.info("QMT order: %s %s %d@%.2f (qmt_id=%s)",
                      order.side.value, order.code, order.quantity,
                      order.price, broker_id)
            return order

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_msg = str(e)
            order.updated_at = datetime.now().isoformat()
            self._orders.append(order)
            logger.error("QMT place_order error: %s", e)
            return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a submitted order.

        Returns True if the cancel request was sent successfully.
        The actual cancellation is confirmed asynchronously via callback.
        """
        if not self._connected:
            return False

        for o in self._orders:
            if o.order_id == order_id and o.broker_order_id:
                if o.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                    return False
                try:
                    self._trader.cancel_order_stock(self._account, int(o.broker_order_id))
                    logger.info("QMT cancel requested: %s (qmt_id=%s)", order_id, o.broker_order_id)
                    return True
                except Exception as e:
                    logger.error("QMT cancel failed: %s", e)
                    return False
        return False

    # ── Queries ──

    def get_positions(self) -> list[Position]:
        """Query current positions from QMT.

        Returns empty list if disconnected or query fails.
        """
        if not self._connected:
            return list(self._positions.values())

        try:
            from quant_platform.trading.qmt_utils import from_qmt_code, qmt_position_to_dict

            raw = self._trader.query_stock_positions(self._account)
            positions: list[Position] = []

            if raw is not None:
                for p in raw:
                    code = from_qmt_code(qmt_position_to_dict(p).get("code", ""))
                    d = qmt_position_to_dict(p)
                    pos = Position(
                        code=code,
                        quantity=d.get("volume", 0),
                        available=d.get("can_use_volume", 0),
                        avg_cost=d.get("avg_price", 0.0),
                        market_value=d.get("market_value", 0.0),
                    )
                    self._positions[code] = pos
                    positions.append(pos)

            return positions

        except Exception as e:
            logger.error("QMT query positions failed: %s", e)
            return list(self._positions.values())

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status:
            return [o for o in self._orders if o.status == status]
        return list(self._orders)

    def get_account(self) -> dict:
        """Get account balance from QMT.

        Falls back to internal cash tracking if query fails.
        """
        base = {
            "connected": self._connected,
            "broker": "qmt",
            "mode": self._mode,
            "account_id": self._account_id,
        }

        if not self._connected:
            return {**base, "cash": self._initial_cash, "total_equity": self._initial_cash}

        try:
            asset = self._trader.query_stock_asset(self._account)
            if asset is not None:
                cash = getattr(asset, "cash", 0.0) or getattr(asset, "enable_balance", 0.0)
                market_value = getattr(asset, "market_value", 0.0)
                total = getattr(asset, "total_asset", cash + market_value)
                self._cash = cash
                self._market_value = market_value
                return {
                    **base,
                    "cash": round(cash, 2),
                    "market_value": round(market_value, 2),
                    "total_equity": round(total, 2),
                    "initial_cash": self._initial_cash,
                    "total_pnl": round(total - self._initial_cash, 2),
                    "n_positions": len(self.get_positions()),
                }

        except Exception as e:
            logger.warning("QMT account query failed: %s", e)

        # Fallback
        positions = self.get_positions()
        mkt_val = sum(p.market_value for p in positions)
        total = self._cash + mkt_val
        return {
            **base,
            "cash": round(self._cash, 2),
            "market_value": round(mkt_val, 2),
            "total_equity": round(total, 2),
            "initial_cash": self._initial_cash,
            "total_pnl": round(total - self._initial_cash, 2),
            "n_positions": len(positions),
        }

    # ── Internals ──

    def _on_async_response(self, response: Any) -> None:
        """Handle async order response from QMT (fill / status update).

        Called by the QMT callback thread. Updates the tracked Order
        and publishes events to EventBus.
        """
        try:
            from quant_platform.trading.qmt_utils import (
                from_qmt_code, qmt_status_to_internal, qmt_trade_to_dict,
            )

            d = qmt_trade_to_dict(response)
            qmt_order_id = str(d.get("order_id", ""))
            qmt_status = d.get("status", d.get("order_status", 0))
            filled_qty = d.get("filled_quantity", d.get("volume", 0))
            filled_price = d.get("filled_price", d.get("price", 0.0))
            code = from_qmt_code(d.get("code", ""))

            new_status = qmt_status_to_internal(int(qmt_status) if qmt_status else 0)

            # Update tracked order
            for o in self._orders:
                if o.broker_order_id == qmt_order_id or o.order_id == qmt_order_id:
                    o.status = new_status
                    o.filled_quantity = filled_qty or o.filled_quantity
                    if filled_price:
                        o.filled_price = float(filled_price)
                    o.updated_at = datetime.now().isoformat()
                    self._publish_event("qmt.status_update", o.to_dict())
                    if new_status == OrderStatus.FILLED:
                        self._publish_event("order.filled", o.to_dict())
                    break

            # Update account cache
            self._update_account_snapshot()

        except Exception as e:
            logger.error("QMT async response error: %s", e)

    def _update_order_status(self, broker_order_id: str, status: OrderStatus, error_msg: str = "") -> None:
        """Update an order's status (e.g. on error callback)."""
        for o in self._orders:
            if o.broker_order_id == broker_order_id:
                o.status = status
                o.error_msg = error_msg
                o.updated_at = datetime.now().isoformat()
                self._publish_event("qmt.order_error", o.to_dict())
                break

    def _update_account_snapshot(self) -> None:
        """Refresh cached account state from QMT."""
        try:
            self.get_account()
        except Exception:
            pass

    def _publish_event(self, topic: str, data: dict) -> None:
        """Publish an event through EventBus (best-effort)."""
        try:
            from quant_platform.core.events import get_event_bus
            bus = get_event_bus()
            bus.publish(topic, data, source="qmt")
        except Exception:
            pass


class XTPBroker(BrokerInterface):
    """Live broker via XTP API (中泰证券 XTP 极速交易).

    XTP (Xtreme Trading Platform) is a low-latency trading API
    from 中泰证券/量化交易. Supports A-share stocks and ETFs.

    Documentation: https://xtp.zts.com.cn/

    Features:
    - Low latency (< 1ms order round-trip in colocation)
    - Full market depth (L2 order book)
    - Streaming market data
    - Supports 上海/深圳 exchanges
    """

    def __init__(self, client_id: int = 1, key: str = "",
                 data_folder: str = "", server_ip: str = "",
                 server_port: int = 6001, protocol: str = "TCP"):
        self._client_id = client_id
        self._key = key
        self._data_folder = data_folder
        self._server_ip = server_ip
        self._server_port = server_port
        self._protocol = protocol
        self._trader = None
        self._quote = None
        self._connected = False
        self._session_id = 0
        self._orders: list[Order] = []
        self._positions: dict[str, Position] = {}

        try:
            import xtp  # XTP Python API
            self._xtp = xtp
            self._HAS_XTP = True
        except ImportError:
            self._HAS_XTP = False
            logger.warning("xtp not installed. XTPBroker unavailable. "
                         "Install from: https://xtp.zts.com.cn/")

    def connect(self) -> bool:
        if not self._HAS_XTP:
            logger.error("XTP API not installed")
            return False

        if not self._data_folder or not self._key:
            logger.error("data_folder and key required for XTP connection")
            return False

        try:
            # Create trader
            self._trader = self._xtp.QuoteApi(self._client_id, self._data_folder)
            self._trader.Login(
                self._server_ip, self._server_port,
                self._key, self._session_id, self._protocol
            )

            # Create quote API for market data
            self._quote = self._xtp.TraderApi(self._data_folder)
            self._quote.SubscribeAllMarketData(
                [], [], self._session_id  # Empty list = subscribe all
            )

            self._connected = True
            logger.info("XTP connected. Client: %d, IP: %s:%d",
                       self._client_id, self._server_ip, self._server_port)
            return True

        except Exception as e:
            logger.error("XTP connection error: %s", e)
            return False

    def disconnect(self) -> bool:
        try:
            if self._trader:
                self._trader.Logout()
            if self._quote:
                self._quote.Logout()
            self._connected = False
            return True
        except Exception as e:
            logger.error("XTP disconnect error: %s", e)
            return False

    def place_order(self, order: Order) -> Order:
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error_msg = "Not connected to XTP"
            return order

        try:
            # XTP order type mapping
            xtp_side = self._xtp.XTP_SIDE_BUY if order.side == OrderSide.BUY \
                else self._xtp.XTP_SIDE_SELL

            xtp_price_type = self._xtp.XTP_PRICE_LIMIT if order.order_type == OrderType.LIMIT \
                else self._xtp.XTP_PRICE_MARKET

            # XTP uses separate exchange IDs
            code = order.code
            if code.startswith(('6', '9')):
                xtp_exchange = self._xtp.XTP_EXCHANGE_SH
            else:
                xtp_exchange = self._xtp.XTP_EXCHANGE_SZ

            xtp_order_id = self._trader.InsertOrder(
                ticker=code,
                exchange=xtp_exchange,
                price=order.price if order.price > 0 else 0,
                quantity=order.quantity,
                side=xtp_side,
                price_type=xtp_price_type,
                business_type=self._xtp.XTP_BUSINESS_TYPE_CASH,
            )

            order.broker_order_id = str(xtp_order_id)
            order.status = OrderStatus.SUBMITTED
            self._orders.append(order)
            logger.info("XTP Order: %s %s %d@%.2f (xtp_id: %s)",
                       order.side.value, order.code, order.quantity,
                       order.price, xtp_order_id)
            return order

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_msg = str(e)
            self._orders.append(order)
            return order

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        try:
            for o in self._orders:
                if o.order_id == order_id and o.broker_order_id:
                    self._trader.CancelOrder(int(o.broker_order_id))
                    o.status = OrderStatus.CANCELLED
                    return True
            return False
        except Exception as e:
            logger.error("XTP Cancel failed: %s", e)
            return False

    def get_positions(self) -> list[Position]:
        if not self._connected:
            return []
        try:
            # XTP queries positions by exchange
            positions = []
            for xtp_exchange in [self._xtp.XTP_EXCHANGE_SH,
                                 self._xtp.XTP_EXCHANGE_SZ]:
                raw = self._trader.QueryPosition(
                    ticker="",
                    exchange=xtp_exchange,
                    session_id=0,
                    request_id=0,
                )
                if raw and raw.get("positions"):
                    for p in raw["positions"]:
                        pos = Position(
                            code=p.ticker,
                            quantity=p.total_qty,
                            available=p.sellable_qty,
                            avg_cost=p.avg_price,
                            market_value=p.market_value,
                            unrealized_pnl=p.unrealized_pnl,
                        )
                        positions.append(pos)
            return positions
        except Exception as e:
            logger.error("XTP Query positions failed: %s", e)
            return []

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status:
            return [o for o in self._orders if o.status == status]
        return self._orders

    def get_account(self) -> dict:
        if not self._connected:
            return {"connected": False, "broker": "xtp"}
        try:
            asset = self._trader.QueryAsset(session_id=0, request_id=0)
            return {
                "total_equity": round(getattr(asset, "total_asset", 0), 2),
                "cash": round(getattr(asset, "buying_power", 0), 2),
                "market_value": round(getattr(asset, "market_value", 0), 2),
                "n_positions": len(self.get_positions()),
                "connected": True,
                "broker": "xtp",
                "client_id": self._client_id,
            }
        except Exception as e:
            return {"connected": False, "error": str(e), "broker": "xtp"}

    def get_order_book(self, ticker: str) -> dict | None:
        """Get L2 order book snapshot (XTP-native)."""
        if not self._connected or not self._quote:
            return None
        try:
            depth = self._quote.GetMarketData(ticker)
            if not depth:
                return None
            return {
                "ticker": ticker,
                "timestamp": depth.data_time,
                "bids": [(depth.bid[i], depth.bid_qty[i]) for i in range(min(10, len(depth.bid)))],
                "asks": [(depth.ask[i], depth.ask_qty[i]) for i in range(min(10, len(depth.ask)))],
                "last_price": depth.last_price,
                "volume": depth.volume,
            }
        except Exception:
            return None

    @staticmethod
    def _to_xtp_code(code: str) -> tuple[str, int]:
        """Convert code to XTP ticker + exchange pair."""
        import xtp
        if code.startswith(('6', '9')):
            return code, xtp.XTP_EXCHANGE_SH
        return code, xtp.XTP_EXCHANGE_SZ


# ── Broker Registry ──

BROKER_REGISTRY: dict[str, type[BrokerInterface]] = {
    "simulated": SimulatedBroker,
    "simulated_lob": SimulatedBroker,
    "paper": SimulatedBroker,
    "qmt": QMTBroker,
    "qmt_sim": QMTBroker,
    "qmt_live": QMTBroker,
    "xtp": XTPBroker,
}


def create_broker(name: str, **kwargs) -> BrokerInterface:
    """Create a broker instance by name.

    Args:
        name: Broker type ('simulated', 'paper', 'qmt', 'qmt_sim', 'xtp')
        **kwargs: Passed to the broker constructor.

    Returns:
        Broker instance implementing BrokerInterface.
    """
    cls = BROKER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown broker: {name}. Choose from: {list(BROKER_REGISTRY)}")

    # Normalize qmt* variants and inject mode
    if name.lower() in ("qmt", "qmt_sim", "qmt_live"):
        if name.lower() == "qmt_live":
            kwargs.setdefault("mode", "live")
        else:
            kwargs.setdefault("mode", "sim")

    return cls(**kwargs)
