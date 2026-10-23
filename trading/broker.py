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

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from quant_platform.data.providers.base import (
    empty_corporate_actions,
    normalize_corporate_actions,
)
from quant_platform.execution.models import (
    Fill as ExecutionFill,
)
from quant_platform.execution.models import (
    Order as ExecutionOrder,
)
from quant_platform.execution.models import (
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_platform.execution.order_book import (
    BookOrder,
    OrderBook,
)
from quant_platform.execution.order_book import (
    OrderType as BookOrderType,
)
from quant_platform.execution.order_book import (
    Side as BookSide,
)
from quant_platform.shared.symbols import normalize_symbol, price_limit_threshold
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BrokerOrder:
    """Broker-layer order DTO.

    This is the wire shape used by SimulatedBroker/QMTBroker. The canonical
    domain model is `quant_platform.execution.models.Order`. Both used to be
    called `Order`, so a function signature could not say which contract it
    expected; use the conversion methods at the layer boundary (issue #74).
    """
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

    def to_execution_order(self) -> ExecutionOrder:
        """Convert this broker DTO to the canonical OMS model."""
        order = ExecutionOrder(
            order_id=self.order_id,
            ticker=self.code,
            side=OrderSide(self.side.value),
            order_type=OrderType(self.order_type.value),
            quantity=self.quantity,
            limit_price=self.price if self.price > 0 else None,
            status=OrderStatus(self.status.value),
            created_at=self.created_at,
            notes=self.error_msg,
        )
        if self.filled_quantity > 0:
            order.fills.append(ExecutionFill(
                timestamp=self.updated_at,
                price=self.filled_price,
                quantity=self.filled_quantity,
                commission=self.commission,
                tax=self.tax,
                slippage=self.slippage,
            ))
        return order

    @classmethod
    def from_execution_order(
        cls, order: ExecutionOrder, **overrides: Any,
    ) -> BrokerOrder:
        """Build a broker DTO from the canonical OMS model."""
        broker_order = cls(
            order_id=order.order_id,
            code=order.ticker,
            side=OrderSide(order.side.value),
            order_type=OrderType(order.order_type.value),
            quantity=order.quantity,
            price=order.limit_price or 0.0,
            filled_quantity=order.filled_quantity,
            filled_price=order.avg_fill_price,
            status=OrderStatus(order.status.value),
            commission=order.total_commission,
            tax=order.total_tax,
            slippage=order.total_slippage,
            created_at=order.created_at or datetime.now().isoformat(),
            updated_at=(
                order.filled_at or order.submitted_at or datetime.now().isoformat()
            ),
            error_msg=order.notes,
        )
        for key, value in overrides.items():
            setattr(broker_order, key, value)
        return broker_order

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


# Backward-compatible name. New code should use `BrokerOrder` so the layer is
# explicit and `execution.models.Order` stays the single domain model.
Order = BrokerOrder


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

    #: True only for brokers that maintain the paper book locally.  Real
    #: brokers (QMT/XTP) perform corporate actions on the broker side; a
    #: local share/dividend adjustment would double-count them.
    supports_corporate_actions: bool = False

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
    def get_positions(self) -> list[Position] | None:
        """Get current positions.

        ``None`` means the query failed and the book state is unknown; an
        empty list means a confirmed flat account. Callers must not treat an
        unknown result as "no positions" (D6).
        """

    @abstractmethod
    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get order history."""

    @abstractmethod
    def get_account(self) -> dict:
        """Get account info (cash, equity, etc.)."""

    def new_trading_day(self, date: str | None = None) -> None:
        """Release T+1 locks for a new trading day.

        Default no-op: brokers that settle remotely (QMT/XTP) need no local
        rollover -- and must not apply corporate actions locally either.
        SimulatedBroker overrides it, and the live engine calls it on every
        calendar-date change.
        """
        return None


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

    #: Paper book: corporate actions are applied locally on the ex-date.
    supports_corporate_actions = True

    def __init__(
        self,
        initial_cash: float = 1_000_000,
        asset_universe=None,
        cash_dividend_tax_rate: float = 0.0,
    ):
        # D1: `place_order` is a check-then-act sequence over `_cash` and
        # `_positions` (validate available shares/cash, then mutate). FastAPI
        # serves sync endpoints on a thread pool and `engine.manual_order`
        # bypasses the engine's `_cycle_lock`, so two concurrent sells both
        # passed the availability check and credited cash twice, and two
        # concurrent first buys of a new symbol overwrote each other's
        # position (get()-then-assign). Holding an RLock across the whole
        # check-and-mutate makes fills serialized and atomic. RLock (not
        # Lock) because account getters may be called from code already
        # inside a locked section.
        self._lock = threading.RLock()
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

        # Corporate actions (paper only).  `cash_dividend_tax_rate` is the
        # documented withholding rate applied to gross cash dividends; the
        # default 0 credits the full declared amount (the contract's
        # `cash_per_share` is pre-tax).  Markers are keyed on
        # `(canonical_asset, ex_date)` so a replayed day never double-applies,
        # and can be persisted through a Store (see
        # `configure_corporate_action_persistence`) to survive a restart.
        self._cash_dividend_tax_rate = max(0.0, float(cash_dividend_tax_rate))
        self._corporate_actions: pd.DataFrame = empty_corporate_actions()
        self._applied_corporate_actions: set[tuple[str, str]] = set()
        self._corporate_action_events: list[dict[str, Any]] = []
        self._cash_ledger: list[dict[str, Any]] = []
        self._ca_store: Any = None
        self._ca_state_key: str = ""
        self._exchange_date: str = ""
        #: Post-split reference prices used to reject a stale pre-split quote
        #: (a suspended ex-date feeds the previous raw close, which would
        #: value doubled shares at the old level).  See update_market_prices.
        self._split_price_guards: dict[str, dict[str, Any]] = {}

    def _get_or_create_book(self, symbol: str, price: float) -> OrderBook:
        """Get or create an order book for a symbol, seeded with liquidity."""
        if symbol not in self._order_books:
            book = OrderBook(symbol)
            self._seed_book(book, symbol, price)
            self._order_books[symbol] = book
        return self._order_books[symbol]

    def _seed_book(self, book: OrderBook, symbol: str, price: float):
        """Seed order book with synthetic market-maker liquidity.

        Both sides quote outward from the reference price, so the best bid
        and the best ask both sit AT `price`: a limit order at the reference
        crosses on either side, and a seller is not forced to undercut the
        market to get a fill.

        The quotes are submitted as *passive* orders. They rest without
        matching, which is what makes quoting the same price on both sides
        safe -- an aggressive bid at the ask would otherwise trade against
        the ladder as it is built, emptying the book. A user order still
        fills against them.

        The step is at least one tick, so the levels stay distinct when the
        spread is finer than the tick (1 bp of ¥10 is ¥0.001).
        """
        spread_bps = 1
        step = max(book.tick_size, price * spread_bps / 10000)
        for i in range(6):
            level_qty = max(100, int(5000 / (i + 1)))

            ask_price = round(price + step * i, 2)
            ask_order = BookOrder(
                order_id=f"mm_ask_{symbol}_{i}",
                symbol=symbol,
                side=BookSide.SELL,
                order_type=BookOrderType.LIMIT,
                price=ask_price,
                quantity=level_qty,
                source="market_maker",
                passive=True,
            )
            book.add_order(ask_order)

            bid_price = round(price - step * i, 2)
            bid_order = BookOrder(
                order_id=f"mm_bid_{symbol}_{i}",
                symbol=symbol,
                side=BookSide.BUY,
                order_type=BookOrderType.LIMIT,
                price=bid_price,
                quantity=level_qty,
                source="market_maker",
                passive=True,
            )
            book.add_order(bid_order)

    def connect(self) -> bool:
        with self._lock:
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

    def _max_affordable(self, symbol: str, price: float, cash: float) -> int:
        """Largest lot-multiple quantity whose value *plus fees* fits in `cash`.

        A percentage buffer cannot stand in for the fee schedule: commission
        has a ¥5 floor, so 100 shares at ¥10 costs ¥1005 while a 0.1% buffer
        allows only ¥1001. Any order in `[1.001 * value, value + fees)` passed
        the old pre-check and then drove `self._cash` negative.
        """
        lot = self._get_lot_size(symbol)
        multiplier = self._get_multiplier(symbol)
        if price <= 0 or lot <= 0 or multiplier <= 0:
            return 0

        qty = int(cash / (price * multiplier) / lot) * lot
        while qty > 0:
            value = price * qty * multiplier
            fees = (
                self._get_commission(symbol, price, qty)
                + self._get_stamp_tax(symbol, price, qty, OrderSide.BUY)
            )
            if value + fees <= cash:
                return qty
            qty -= lot
        return 0

    def place_order(self, order: Order) -> Order:
        """Submit order to the real order book for price-time priority matching.

        The entire method runs under `self._lock`: the availability checks,
        the LOB matching and the cash/position mutation must be one atomic
        step, or concurrent orders act on the same snapshot of the book
        (D1). Callers therefore never observe a fill whose cash and shares
        disagree.
        """
        with self._lock:
            return self._place_order_locked(order)

    def _place_order_locked(self, order: Order) -> Order:
        if not self._connected:
            order.status = OrderStatus.REJECTED
            order.error_msg = "Not connected"
            return order

        # Market orders arrive with price 0. Without a reference price the
        # affordability check asked for 0 shares and rejected every market
        # buy; the book and slippage also need a reference.
        if order.order_type == OrderType.MARKET and order.price <= 0:
            reference = self._market_prices.get(order.code, 0.0)
            book = self._order_books.get(order.code)
            if reference <= 0 and book is not None:
                depth = book.get_depth_snapshot(levels=1)
                if depth.get("asks"):
                    first = depth["asks"][0]
                    reference = float(
                        first["price"] if isinstance(first, dict) else first[0]
                    )
                elif depth.get("bids"):
                    first = depth["bids"][0]
                    reference = float(
                        first["price"] if isinstance(first, dict) else first[0]
                    )
            if reference <= 0:
                order.status = OrderStatus.REJECTED
                order.error_msg = "No reference price for market order"
                self._orders.append(order)
                return order
            order.price = reference

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

        # Check cash for buy (worst-case cost, including the fee schedule)
        if order.side == OrderSide.BUY:
            if self._max_affordable(order.code, order.price, self._cash) < order.quantity:
                worst_value = order.price * order.quantity * self._get_multiplier(order.code)
                worst_cost = (
                    worst_value
                    + self._get_commission(order.code, order.price, order.quantity)
                    + self._get_stamp_tax(order.code, order.price, order.quantity, order.side)
                )
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

        # IOC semantics: the broker reports a rejection or partial fill, so
        # the unfilled remainder must not rest in the book. Leaving it there
        # let a later order trade against a sell the broker had already
        # reported as rejected, minting shares.
        if total_filled < order.quantity:
            book.cancel_order(order.order_id)

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
                # Partial fill at what we can afford, fees included. The old
                # bound divided by 1.001 and ignored the commission floor, so
                # the recomputed `total_cost` was still larger than cash and
                # the subtraction below drove the balance negative.
                affordable_qty = self._max_affordable(order.code, avg_fill_price, self._cash)
                if affordable_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    order.error_msg = f"Insufficient cash after fill. Need {total_cost:.2f}, have {self._cash:.2f}"
                    self._orders.append(order)
                    return order
                total_filled = min(total_filled, affordable_qty)
                total_value = avg_fill_price * total_filled
                commission = self._get_commission(order.code, avg_fill_price, total_filled)
                tax = self._get_stamp_tax(order.code, avg_fill_price, total_filled, order.side)
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
            # Mark to market at the fill price.
            #
            # A new position otherwise keeps market_value = 0 until the next
            # update_market_prices() call, which the live engine makes *before*
            # it executes orders (engine.py, step 1 vs step 3). For the rest of
            # the cycle the position contributed nothing to equity while its
            # cost had already left cash, so the step-4 P&L snapshot reported
            # the whole purchase as a loss.
            pos.update_price(avg_fill_price)
            self._today_bought.add(order.code)
        else:  # SELL
            # The availability check above and this mutation are inside the
            # same `_lock`, so the position cannot have been removed by a
            # concurrent fill in between (the old unlocked code raised
            # KeyError *after* crediting cash here).
            self._cash += total_value - commission - tax
            pos = self._positions[order.code]
            pos.quantity -= total_filled
            pos.available -= total_filled
            pos.realized_pnl += (avg_fill_price - pos.avg_cost) * total_filled
            pos.update_price(avg_fill_price)
            if pos.quantity <= 0:
                del self._positions[order.code]
                self._split_price_guards.pop(
                    normalize_symbol(order.code), None,
                )

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
        with self._lock:
            for o in self._orders:
                if o.order_id == order_id and o.status == OrderStatus.PENDING:
                    o.status = OrderStatus.CANCELLED
                    # Cancel in the order book too
                    for book in self._order_books.values():
                        book.cancel_order(order_id)
                    return True
            return False

    def get_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        with self._lock:
            if status:
                return [o for o in self._orders if o.status == status]
            return list(self._orders)

    def get_account(self) -> dict:
        with self._lock:
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
        """Update position prices and refresh synthetic book liquidity.

        A quote for a symbol whose split was just applied is validated
        against the post-split reference: a move larger than the board's
        daily limit is impossible as a real one-day return and is, in
        practice, the *pre-split* stale close a suspended stock keeps
        feeding.  Such a quote is ignored (the action-adjusted mark and the
        reseeded ladder stay in place) until a plausible post-split quote
        arrives, so a split on a suspended day cannot double the position's
        market value.
        """
        with self._lock:
            accepted: dict[str, float] = {}
            for code, price in prices.items():
                guard = self._split_price_guards.get(normalize_symbol(code))
                if guard is not None and price and price > 0:
                    limit = price_limit_threshold(normalize_symbol(code))
                    expected = float(guard["expected"])
                    if expected > 0 and price > expected * (1 + limit + 1e-9):
                        logger.warning(
                            "Ignoring implausible post-split quote for %s: "
                            "%.4f vs post-split reference %.4f (ratio %.2f, "
                            "ex-date %s); keeping the adjusted mark",
                            code, price, expected, guard["ratio"],
                            guard["ex_date"],
                        )
                        continue
                    self._split_price_guards.pop(normalize_symbol(code), None)
                accepted[code] = price

            self._market_prices.update(accepted)
            for code, pos in self._positions.items():
                if code in accepted:
                    pos.update_price(accepted[code])

            # Re-seed the market-maker quotes around the new reference price.
            # The book was seeded once, so a limit order at the *new* market price
            # could fill against stale quotes from the old one (phantom profit).
            for code, price in accepted.items():
                book = self._order_books.get(code)
                if book is None or not price or price <= 0:
                    continue
                for order_id, resting in list(getattr(book, "_orders", {}).items()):
                    if getattr(resting, "source", "") == "market_maker":
                        book.cancel_order(order_id)
                self._seed_book(book, code, float(price))

    def new_trading_day(self, date: str | None = None):
        """Reset T+1 restrictions and apply the day's corporate actions.

        ``date`` (``YYYY-MM-DD``) advances the paper exchange day: every
        action with ``ex_date <= date`` that has not been applied yet is
        applied exactly once, before the session starts.  Callers that only
        want the legacy T+1 rollover may omit it.
        """
        with self._lock:
            self._today_bought.clear()
            for pos in self._positions.values():
                pos.available = pos.quantity
        if date is None:
            return []
        self._exchange_date = pd.Timestamp(date).normalize().date().isoformat()
        return self.apply_corporate_actions(ex_date=self._exchange_date)

    # ── Corporate actions (paper book only) ──

    def set_corporate_actions(self, actions) -> int:
        """Replace the paper book's pending corporate-action feed.

        ``actions`` follows the provider contract
        ``(ex_date, asset, kind, cash_per_share, split_ratio)``.  Nothing is
        applied until :meth:`apply_corporate_actions` (or a dated
        :meth:`new_trading_day`) runs, and already-applied ``(asset, ex_date)``
        markers are honoured.  Returns the number of actions loaded.
        """
        frame = normalize_corporate_actions(actions)
        with self._lock:
            self._corporate_actions = frame
        return len(frame)

    def configure_corporate_action_persistence(
        self, store, strategy_id: str = "paper_corporate_actions",
    ) -> None:
        """Persist applied ``(asset, ex_date)`` markers in a Store.

        The store is any object exposing ``get_engine_state(key)`` /
        ``save_engine_state(state, key)`` (``core.store.Store``).  Markers are
        unioned with any already-loaded state, so wiring several times (or
        reconnecting after a restart) can never un-apply an action.
        """
        self._ca_store = store
        self._ca_state_key = str(strategy_id)
        self._load_corporate_action_markers()

    def _load_corporate_action_markers(self) -> None:
        if self._ca_store is None or not self._ca_state_key:
            return
        try:
            state = self._ca_store.get_engine_state(self._ca_state_key) or {}
        except Exception as exc:  # persistence is best effort
            logger.warning("Could not load corporate-action markers: %s", exc)
            return
        applied = state.get("applied", []) if isinstance(state, dict) else []
        for item in applied:
            try:
                asset, date_text = item
                self._applied_corporate_actions.add(
                    (normalize_symbol(asset), str(date_text))
                )
            except (TypeError, ValueError):
                continue

    def _save_corporate_action_markers(self) -> None:
        if self._ca_store is None or not self._ca_state_key:
            return
        payload = {
            "version": 1,
            "applied": sorted(
                [asset, date_text]
                for asset, date_text in self._applied_corporate_actions
            ),
        }
        try:
            self._ca_store.save_engine_state(payload, self._ca_state_key)
        except Exception as exc:  # never let bookkeeping block trading
            logger.warning("Could not persist corporate-action markers: %s", exc)

    def get_applied_corporate_actions(self) -> list[tuple[str, str]]:
        """Applied ``(asset, ex_date)`` markers, sorted."""
        with self._lock:
            return sorted(self._applied_corporate_actions)

    def get_corporate_action_events(self) -> list[dict[str, Any]]:
        """Audit records of every action applied to this book."""
        with self._lock:
            return [dict(event) for event in self._corporate_action_events]

    def get_cash_ledger(self) -> list[dict[str, Any]]:
        """Cash entries generated by corporate actions (dividends)."""
        with self._lock:
            return [dict(entry) for entry in self._cash_ledger]

    @staticmethod
    def _action_order(kind: str) -> int:
        # Cash before split on a combined ex-date: the dividend is declared
        # per pre-split share, so it must use the pre-split quantity.
        return 0 if kind == "cash" else 1

    def apply_corporate_actions(
        self,
        actions=None,
        ex_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply pending split/cash actions exactly once.

        Args:
            actions: Optional frame overriding the stored feed.
            ex_date: Apply only actions with ``ex_date <= ex_date`` (the
                exchange day being advanced to).  ``None`` applies every
                loaded action.

        Returns:
            One audit dict per ``(asset, ex_date)`` event applied to a held
            position (a combined cash+split event is one dict with two
            components).
            Replays are no-ops: markers are checked under the book lock and
            persisted after each batch.  A split scales quantity *and*
            availability by the ratio, divides the average cost and the
            current mark (market value unchanged); a cash dividend credits
            ``quantity * cash_per_share * (1 - tax)`` to cash, adds it to the
            position's realised P&L, and reduces the cached mark by the
            per-share amount so NAV stays continuous if no fresh quote has
            arrived yet.  T+1 convention: split-added shares inherit the
            availability of the shares they came from (on the ex-date the
            engine has already rolled the day, so an overnight holding's
            whole post-split quantity is sellable).
        """
        with self._lock:
            frame = (
                self._corporate_actions if actions is None
                else normalize_corporate_actions(actions)
            )
            if frame is None or frame.empty:
                return []
            cutoff = (
                pd.Timestamp(ex_date).normalize() if ex_date is not None else None
            )
            rows = frame.copy()
            rows["_order"] = rows["kind"].map(self._action_order).fillna(2)
            rows = rows.sort_values(
                ["ex_date", "asset", "_order"], kind="stable",
            )
            applied: list[dict[str, Any]] = []
            markers_changed = False
            # One marker per (asset, ex_date) *event*: a combined 送转+派息
            # arrives as two rows sharing the key and both components must be
            # applied before the marker is recorded.
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
                event = self._apply_action_group(marker, group)
                # The marker covers "evaluated on this ex-date", including a
                # zero holding: entitlement is fixed on the ex-date, so a
                # position restored later must not receive the action again.
                self._applied_corporate_actions.add(marker)
                markers_changed = True
                if event is not None:
                    self._corporate_action_events.append(event)
                    applied.append(event)
            if markers_changed:
                self._save_corporate_action_markers()
            return applied

    def _find_position(self, asset: str) -> Position | None:
        for key, pos in self._positions.items():
            if normalize_symbol(key) == asset:
                return pos
        return None

    def _find_market_price_key(self, asset: str) -> str | None:
        for key in self._market_prices:
            if normalize_symbol(key) == asset:
                return key
        return None

    def _find_book_key(self, asset: str) -> str | None:
        for key in self._order_books:
            if normalize_symbol(key) == asset:
                return key
        return None

    def _remark(self, asset: str, pos: Position | None, new_price: float) -> None:
        """Update the cached mark/order book after an action-adjusted price."""
        new_price = float(new_price)
        if pos is not None:
            pos.update_price(new_price)
        price_key = self._find_market_price_key(asset)
        if price_key is not None:
            self._market_prices[price_key] = new_price
        book_key = self._find_book_key(asset)
        if book_key is not None and new_price > 0:
            # The resting market-maker ladder was seeded at the pre-action
            # price.  A sell at the new (post-split) level would otherwise
            # trade against a stale bid at the old level and mint cash.
            book = self._order_books[book_key]
            for order_id, resting in list(getattr(book, "_orders", {}).items()):
                if getattr(resting, "source", "") == "market_maker":
                    book.cancel_order(order_id)
            self._seed_book(book, book_key, new_price)

    def _apply_action_group(
        self, marker: tuple[str, str], group: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Apply every component sharing one ``(asset, ex_date)`` marker.

        Cash components run first on the pre-split quantity (a 送转+派息
        dividend is declared per pre-split share), then splits scale the
        position.  One audit event covers the whole group.
        """
        asset = marker[0]
        pos = self._find_position(asset)
        quantity_before = int(pos.quantity) if pos is not None else 0
        quantity_after = quantity_before
        avg_cost_after = pos.avg_cost if pos is not None else 0.0
        cash_total = 0.0
        tax_total = 0.0
        cash_per_share_total = 0.0
        split_ratio_total = 1.0
        components: list[dict[str, Any]] = []
        timestamp = datetime.now().isoformat()

        # ── cash components first (pre-split entitlement) ──
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
                if pos is not None:
                    pos.realized_pnl += net
                    if pos.current_price > 0:
                        # Estimate of the ex-dividend mark; the next quote
                        # (`update_market_prices`) overwrites it with the real
                        # raw close, which already carries the drop.
                        self._remark(
                            asset, pos,
                            max(pos.current_price - cash_per_share, 0.0),
                        )
                self._cash_ledger.append({
                    "date": marker[1],
                    "asset": asset,
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
            self._cash += cash_total

        # ── split components ──
        for _, row in group.iterrows():
            if str(row["kind"]) != "split":
                continue
            ratio = float(row["split_ratio"])
            if not (ratio > 0) or ratio == 1.0:
                continue
            split_ratio_total *= ratio
            components.append({"kind": "split", "split_ratio": ratio})
            if pos is not None:
                new_quantity = int(round(pos.quantity * ratio))
                pos.quantity = new_quantity
                # Split-added shares inherit the original shares' T+1 lock.
                pos.available = min(
                    new_quantity,
                    int(round(pos.available * ratio)),
                )
                pos.avg_cost = (
                    pos.avg_cost / ratio if pos.avg_cost > 0 else 0.0
                )
                if pos.current_price > 0:
                    pre_price = pos.current_price
                    self._remark(asset, pos, pre_price / ratio)
                    self._split_price_guards[asset] = {
                        "expected": pre_price / ratio,
                        "pre": pre_price,
                        "ratio": ratio,
                        "ex_date": marker[1],
                    }
                quantity_after = new_quantity
                avg_cost_after = pos.avg_cost

        if not components:
            return None
        kinds = sorted({component["kind"] for component in components})
        event: dict[str, Any] = {
            "date": marker[1],
            "asset": asset,
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
        if pos is not None:
            event["avg_cost_after"] = avg_cost_after
        self._publish_event("corporate_action", event)
        return event

    def _publish_event(self, topic: str, data: dict[str, Any]) -> None:
        try:
            from quant_platform.core.events import get_event_bus

            get_event_bus().publish(topic, data, source="simulated_broker")
        except Exception:  # monitoring must never break the book
            pass

    def restore_positions(self, rows: list[dict], cash: float | None = None) -> None:
        """Rebuild the in-memory account from persisted rows (D4 restart).

        A paper/SimulatedBroker process restart starts from a fresh book, but
        the Store still records the open positions and (via engine_state) the
        cash balance. Without this method the engine's pre-flight position-sync
        check refused to start on every restart with open positions. Real
        brokers (QMT/XTP) do not implement this: their account is the source of
        truth and must never be overwritten from local records.

        `available` is restored as of the last snapshot so a same-day restart
        keeps today's T+1 lock (the engine separately persists the exchange
        trading date for D7).
        """
        with self._lock:
            self._positions.clear()
            self._split_price_guards.clear()
            for row in rows or []:
                code = str(row.get("code") or row.get("ticker") or "")
                quantity = int(row.get("quantity", 0) or 0)
                if not code or quantity <= 0:
                    continue
                available = row.get("available", quantity)
                try:
                    available = int(available)
                except (TypeError, ValueError):
                    available = quantity
                pos = Position(
                    code=code,
                    name=str(row.get("name", "") or ""),
                    quantity=quantity,
                    available=max(0, min(available, quantity)),
                    avg_cost=float(row.get("avg_cost", 0.0) or 0.0),
                    realized_pnl=float(row.get("realized_pnl", 0.0) or 0.0),
                )
                pos.update_price(float(row.get("current_price", 0.0) or 0.0))
                self._positions[code] = pos
            if cash is not None:
                self._cash = float(cash)


class QMTBroker(BrokerInterface):
    """Live broker via xtquant/miniQMT (国金证券量化交易).

    Connects to a local miniQMT client over TCP (default localhost:58610).
    miniQMT must be running and logged in before calling connect().

    Key features:
    - Full callback chain: on_fill / on_cancel / on_error / on_disconnected
    - EventBus integration: every fill publishes ``qmt.fill`` and ``order.filled``
    - Password from env ``QMT_PASSWORD`` — never in config files
    - Lazy reconnection: ``on_disconnected`` only flips state and logs; the
      next order/query call reconnects with exponential backoff
      (``max_reconnect_attempts`` connect attempts, waiting 1s, 2s, 4s, ...
      between retries). If every attempt fails, ``BrokerConnectionError`` is
      raised — order submission fails closed and ``order_stock`` is never
      called while the broker is unreachable.
    - Broker reconciliation: after a successful lazy reconnect, and on
      ``get_orders`` at most once per ``resync_ttl`` seconds (an explicit
      ``connect()`` invalidates the TTL), the broker's ``query_stock_orders``
      / ``query_stock_trades`` are merged into the local order book
      (idempotent, keyed by broker order id). Read paths fail loud when the
      broker cannot be reached, and degrade to the in-memory book (with a
      warning) when only the resync query fails.

    Architecture::

        LiveRunner ─> QMTBroker ──TCP──> miniQMT/client ──> 国金券商
                         │
                         ├── _xt_trader (order_stock / cancel / query)
                         ├── xtdata.get_full_tick (fresh quotes for live runs)
                         ├── Callback → update Order objects → EventBus
                         ├── _ensure_connected → lazy reconnect + resync
                         └── connect() == False → hard failure (no simulated fallback)

    Supported brokers: 国金证券, 华鑫证券, 国盛证券, 东方财富 etc.
    """

    class BrokerConnectionError(RuntimeError):
        """QMT is unreachable after all lazy-reconnect attempts failed.

        Raised from ``place_order`` / ``cancel_order`` / the query entry
        points. Order submission fails closed: no ``order_stock`` call is
        made when this error is raised.
        """

    def __init__(
        self,
        account: str = "",
        server: str = "localhost:58610",
        password: str = "",
        mode: str = "sim",
        data_server: str = "",
        initial_cash: float = 10_000_000,
        max_reconnect_attempts: int = 3,
        reconnect_backoff: float = 1.0,
        resync_ttl: float = 2.0,
        sleep_fn: Any = None,
        clock_fn: Any = None,
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
            max_reconnect_attempts: Connect attempts per lazy reconnect
                          (>= 1). Waits double between retries.
            reconnect_backoff: Base backoff in seconds; the wait before retry
                          *k* is ``reconnect_backoff * 2 ** (k - 1)``
                          (1s, 2s, 4s, ... by default).
            resync_ttl: Minimum seconds between broker order/trade re-queries
                          from ``get_orders`` (avoids one broker round-trip
                          per poll).
            sleep_fn: Injectable sleep callable (tests pass a no-op so retry
                          backoff does not really wait).
            clock_fn: Injectable monotonic clock for the TTL (tests control it).
        """
        self._account_id = account
        self._server = server
        self._password = password
        self._mode = mode
        self._data_server = data_server or server
        self._initial_cash = initial_cash
        self._max_reconnect_attempts = max(1, int(max_reconnect_attempts))
        self._reconnect_backoff = max(0.0, float(reconnect_backoff))
        self._resync_ttl = max(0.0, float(resync_ttl))
        if sleep_fn is None or clock_fn is None:
            import time as _time
            sleep_fn = sleep_fn or _time.sleep
            clock_fn = clock_fn or _time.monotonic
        self._sleep_fn = sleep_fn
        self._clock_fn = clock_fn

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
        #: Monotonic time of the last successful broker order/trade resync.
        self._last_resync_at: float | None = None
        #: Serializes resync so concurrent get_orders calls cannot both
        #: append the same unknown broker order (check-then-act on _orders).
        self._resync_lock = threading.Lock()
        #: Serializes lazy reconnects: two threads polling a dead link must
        #: not each create a trader session.
        self._reconnect_lock = threading.Lock()
        #: Thread-local re-entrancy guard. connect() calls
        #: _update_account_snapshot() -> get_account() -> _ensure_connected(),
        #: and an on_disconnected callback racing that window flips
        #: `_connected` back to False before the inner check. A plain Lock
        #: would then deadlock on the thread that already owns it; the inner
        #: call raises instead and lets the outer attempt retry.
        self._reconnect_state = threading.local()

        # Resolve password: explicit arg > env var
        if not self._password:
            import os
            self._password = os.environ.get("QMT_PASSWORD", "")

        try:
            import xtquant.xtconstant as xtc
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
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
            # A retry (or manual reconnect) must not leak the previous trader
            # session into miniQMT: stop it before opening a new one.
            if self._trader is not None:
                try:
                    self._trader.stop()
                except Exception:
                    pass
                self._trader = None
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
                    # Placement ACK only; it carries no lifecycle status.
                    broker_ref._on_async_ack(response)

                def on_stock_order(self_cb, xt_order):
                    broker_ref._on_order_update(xt_order)

                def on_stock_trade(self_cb, xt_trade):
                    broker_ref._on_trade(xt_trade)

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
                # A fresh session invalidates the resync TTL: the next
                # get_orders must reconcile against this connection (also
                # covers explicit connect() calls made outside the lazy path).
                self._last_resync_at = None
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

    # ── Lazy reconnect & broker reconciliation ──

    def _ensure_connected(self) -> None:
        """Reconnect on demand, or raise ``BrokerConnectionError``.

        ``on_disconnected`` only flips ``_connected`` and logs; every order
        and query entry point calls this first. A healthy link returns
        immediately. Otherwise it makes at most ``max_reconnect_attempts``
        calls to ``connect(blocking=True)``, waiting
        ``reconnect_backoff * 2 ** (k - 1)`` seconds before retry *k*
        (1s, 2s, 4s, ... by default). The waits go through the injected
        ``sleep_fn`` so tests never actually wait.

        After a successful reconnect the broker book is reconciled via
        ``_resync_from_broker`` (a failed resync only warns — the connection
        to the order gateway is what matters for the next operation). When
        every attempt fails this raises ``BrokerConnectionError``; callers
        must not place, cancel or report state as if the broker answered.
        """
        if self._connected:
            return

        if getattr(self._reconnect_state, "active", False):
            # Re-entered from connect()'s own callback chain (post-connect
            # account snapshot) after on_disconnected flipped the flag.
            # Blocking on the reconnect lock this thread already owns would
            # deadlock; report failure and let the outer attempt retry.
            raise self.BrokerConnectionError(
                "QMT reconnect re-entered from its own connect() callback chain"
            )

        with self._reconnect_lock:
            if self._connected:
                return
            self._reconnect_state.active = True
            try:
                self._reconnect_locked()
            finally:
                self._reconnect_state.active = False

    def _reconnect_locked(self) -> None:
        """Body of ``_ensure_connected``; caller holds ``_reconnect_lock``."""
        logger.warning(
            "QMT link is down — attempting lazy reconnect (up to %d attempts)",
            self._max_reconnect_attempts,
        )
        last_error: Exception | None = None
        for attempt in range(self._max_reconnect_attempts):
            if attempt > 0:
                delay = self._reconnect_backoff * (2 ** (attempt - 1))
                if delay > 0:
                    logger.warning(
                        "QMT reconnect backoff %.2fs before attempt %d/%d",
                        delay, attempt + 1, self._max_reconnect_attempts,
                    )
                    self._sleep_fn(delay)
            try:
                # connect() catches its own errors, but a mock/subclass may
                # raise; either way the attempt failed.
                connected = bool(self.connect(blocking=True))
            except Exception as e:
                last_error = e
                connected = False
            if connected and self._connected:
                logger.info("QMT reconnected after %d attempt(s)", attempt + 1)
                # Reserve the TTL slot before reconciling: if the resync
                # query fails here, get_orders must not immediately retry the
                # same broken endpoint on entry (one attempt per TTL).
                self._last_resync_at = self._clock_fn()
                try:
                    self._resync_from_broker()
                except Exception as e:
                    logger.warning("QMT post-reconnect resync failed: %s", e)
                return

        raise self.BrokerConnectionError(
            "QMT reconnect failed after "
            f"{self._max_reconnect_attempts} attempt(s); refusing to trade"
            + (f" (last error: {last_error})" if last_error is not None else "")
        )

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _created_at_from_broker(raw_time: Any) -> str | None:
        """Normalize a broker ``order_time`` (epoch seconds or string)."""
        if isinstance(raw_time, (int, float)) and raw_time > 0:
            try:
                return datetime.fromtimestamp(float(raw_time)).isoformat()
            except (OverflowError, OSError, ValueError):
                return None
        if isinstance(raw_time, str) and raw_time.strip():
            return raw_time
        return None

    def _resync_from_broker(self) -> bool:
        """Rebuild today's order book from the broker; cross-check fills.

        ``query_stock_orders(account)`` is authoritative for the current
        trading day: tracked orders are updated in place and orders this
        process has never seen (submitted before a restart, or missed by
        callbacks) are appended. Matching is by broker order id, so running
        it twice never duplicates an order.

        ``query_stock_trades(account)`` is then used only to cross-check
        cumulative filled volume: it can raise a stale ``filled_quantity``
        (never above the order quantity) but never invents a fill or flips
        an order's status. When the trade endpoint or its fields are
        unavailable a warning is logged and the order-query values stand.

        Any order-query failure leaves the in-memory book untouched. A
        trade-query failure only skips the cross-check; the order merge from
        the same call is kept. Returns True only when the order query itself
        succeeded.
        """
        from quant_platform.trading.qmt_utils import qmt_order_to_dict

        with self._resync_lock:
            try:
                raw_orders = self._trader.query_stock_orders(self._account)
            except Exception as e:
                logger.warning(
                    "QMT resync: query_stock_orders failed: %s — serving cached orders",
                    e,
                )
                return False
            if raw_orders is None:
                logger.warning(
                    "QMT resync: query_stock_orders returned no result — "
                    "serving cached orders",
                )
                return False
            try:
                broker_orders = list(raw_orders)
            except TypeError as e:
                logger.warning(
                    "QMT resync: query_stock_orders returned a non-iterable "
                    "(%s) — serving cached orders", e,
                )
                return False

            seen: set[str] = set()
            for raw in broker_orders:
                d = qmt_order_to_dict(raw)
                broker_id = str(d.get("order_id", "") or "")
                if not broker_id or broker_id in seen:
                    continue
                seen.add(broker_id)
                self._merge_broker_order(d, broker_id)

            self._cross_check_filled_with_trades()
            return True

    def _merge_broker_order(self, d: dict, broker_id: str) -> None:
        """Update one tracked order or append an unknown broker order.

        Caller holds ``_resync_lock``. Updates follow the callback path's
        monotonic rules (no terminal downgrade, no shrinking fills) because
        the same snapshot can race a callback.
        """
        from quant_platform.trading.qmt_utils import (
            from_qmt_code,
            from_qmt_order_type,
            from_qmt_price_type,
            qmt_status_to_internal,
        )

        status = OrderStatus.PENDING
        raw_status = d.get("order_status")
        if raw_status is not None:
            try:
                status = qmt_status_to_internal(int(raw_status))
            except (TypeError, ValueError):
                status = OrderStatus.PENDING
        traded = self._as_int(d.get("traded_volume"), 0)
        traded_price = self._as_float(d.get("traded_price"), 0.0)

        for order in self._orders:
            if str(order.broker_order_id) != broker_id:
                continue
            if order.status not in self._TERMINAL_STATUSES:
                if order.status == OrderStatus.PARTIAL and status in (
                        OrderStatus.PENDING, OrderStatus.SUBMITTED):
                    pass  # never demote a partially filled order
                elif (order.status == OrderStatus.SUBMITTED
                        and status == OrderStatus.PENDING):
                    # Stale pre-report snapshot (UNREPORTED/WAIT_REPORTING)
                    # from a query that raced the callback; demoting would
                    # make cancel_order refuse a live order.
                    pass
                else:
                    order.status = status
            if traded > int(order.filled_quantity or 0):
                order.filled_quantity = (
                    min(traded, int(order.quantity)) if order.quantity > 0 else traded
                )
            if traded_price > 0:
                order.filled_price = traded_price
            order.updated_at = datetime.now().isoformat()
            return

        # Unknown order: submitted before this process started (or missed by
        # the callback chain). Rebuild it from the broker snapshot.
        quantity = self._as_int(d.get("order_volume"), 0)
        order = Order(
            code=from_qmt_code(str(d.get("code", "") or "")),
            side=from_qmt_order_type(d.get("order_type")),
            order_type=from_qmt_price_type(d.get("price_type")),
            quantity=quantity,
            price=self._as_float(d.get("price"), 0.0),
            filled_quantity=min(traded, quantity) if quantity > 0 else traded,
            filled_price=traded_price,
            status=status,
            broker_order_id=broker_id,
        )
        status_msg = str(d.get("status_msg", "") or "")
        if status_msg:
            order.error_msg = status_msg
        created_at = self._created_at_from_broker(d.get("order_time"))
        if created_at:
            order.created_at = created_at
        self._orders.append(order)
        logger.info(
            "QMT resync: recovered broker order %s (%s %s x%d, status=%s)",
            broker_id, order.side.value, order.code, order.quantity,
            order.status.value,
        )

    def _cross_check_filled_with_trades(self) -> None:
        """Raise stale cumulative fills from ``query_stock_trades`` (best effort).

        Trades only ever increase ``filled_quantity`` up to the order
        quantity; they never create orders, lower fills, or change status.
        Any failure warns; R10 does not fabricate trade data.
        """
        from quant_platform.trading.qmt_utils import qmt_trade_to_dict

        try:
            raw_trades = self._trader.query_stock_trades(self._account)
        except Exception as e:
            logger.warning(
                "QMT resync: query_stock_trades failed: %s — "
                "filled volume not cross-checked", e,
            )
            return
        if raw_trades is None:
            logger.warning(
                "QMT resync: query_stock_trades returned no result — "
                "filled volume not cross-checked",
            )
            return
        try:
            trades = list(raw_trades)
        except TypeError as e:
            logger.warning(
                "QMT resync: query_stock_trades returned a non-iterable "
                "(%s) — filled volume not cross-checked", e,
            )
            return

        traded_by_order: dict[str, int] = {}
        for raw in trades:
            d = qmt_trade_to_dict(raw)
            broker_id = str(d.get("order_id", "") or "")
            if not broker_id:
                continue
            volume = self._as_int(d.get("quantity", d.get("volume")), 0)
            if volume > 0:
                traded_by_order[broker_id] = (
                    traded_by_order.get(broker_id, 0) + volume
                )

        for order in self._orders:
            traded = traded_by_order.get(str(order.broker_order_id or ""))
            if traded is None:
                continue
            current = int(order.filled_quantity or 0)
            if traded <= current:
                continue
            if order.quantity > 0:
                traded = min(traded, int(order.quantity))
            if traded > current:
                logger.info(
                    "QMT resync: trades report %d filled for order %s (had %d)",
                    traded, order.order_id, current,
                )
                order.filled_quantity = traded
                order.updated_at = datetime.now().isoformat()

    def _maybe_resync(self) -> None:
        """Refresh the order book from the broker at most once per TTL.

        The TTL also rate-limits failed attempts: a broken query endpoint is
        retried at most once per ``resync_ttl`` seconds rather than on every
        poll. Failures degrade to the in-memory book (warning only).
        """
        now = self._clock_fn()
        last = self._last_resync_at
        if last is not None and (now - last) < self._resync_ttl:
            return
        self._last_resync_at = now
        try:
            self._resync_from_broker()
        except Exception as e:
            logger.warning(
                "QMT resync failed unexpectedly: %s — serving cached orders", e,
            )

    # ── Order lifecycle ──

    #: Statuses after which no further broker callback may change the order.
    _TERMINAL_STATUSES = frozenset({
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    })

    def place_order(self, order: Order) -> Order:
        """Submit an order to QMT.

        Lazy reconnect runs first (``_ensure_connected``). When the broker is
        unreachable this raises ``BrokerConnectionError`` and
        ``order_stock`` is never called — order submission fails closed.
        A resync failure after a *successful* reconnect does not block the
        order: the order gateway is reachable, only the reconciliation query
        failed.

        Converts internal Order to xtquant params, submits, and tracks.
        The order status will be updated asynchronously via callbacks.

        ``order_stock`` returns a broker order id, or an error code <= 0 on
        failure. A non-positive result is a rejection: storing it as
        ``broker_order_id`` marked the order SUBMITTED even though QMT never
        accepted it, and no callback could ever arrive for it.
        """
        self._ensure_connected()

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

            try:
                accepted = broker_id is not None and int(broker_id) > 0
            except (TypeError, ValueError):
                accepted = False

            if not accepted:
                order.status = OrderStatus.REJECTED
                order.error_msg = f"QMT order_stock rejected (broker_id={broker_id!r})"
                order.updated_at = datetime.now().isoformat()
                with self._resync_lock:
                    self._orders.append(order)
                logger.error("QMT order_stock rejected: %s %s %d@%.2f (broker_id=%r)",
                             order.side.value, order.code, order.quantity,
                             order.price, broker_id)
                return order

            order.broker_order_id = str(broker_id)
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now().isoformat()
            with self._resync_lock:
                # A concurrent resync can recover this broker order between
                # order_stock() returning and this append (query_stock_orders
                # already lists it). Appending would create two entries for
                # one broker id, and callbacks/resync only update the first,
                # leaving the caller's order permanently stale. Keep the
                # single tracked entry and return it instead.
                tracked = next(
                    (o for o in self._orders
                     if str(o.broker_order_id) == order.broker_order_id),
                    None,
                )
                if tracked is not None and tracked is not order:
                    logger.info(
                        "QMT order %s already recovered by a concurrent "
                        "resync; not duplicating", broker_id,
                    )
                    order = tracked
                else:
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

        Reconnects first (``BrokerConnectionError`` when the broker cannot be
        reached — cancelling blind would report success without a broker
        acknowledgement). Returns True if the cancel request was sent
        successfully. The actual cancellation is confirmed asynchronously via
        callback.
        """
        self._ensure_connected()

        for o in self._orders:
            if o.order_id == order_id and o.broker_order_id:
                if o.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                    return False
                try:
                    result = self._trader.cancel_order_stock(
                        self._account, int(o.broker_order_id),
                    )
                    # xtquant returns 0 on success and a non-zero error code
                    # when the exchange/broker refuses the cancellation. The
                    # old code never looked at the return value and reported
                    # every request as success.
                    if result != 0:
                        logger.error(
                            "QMT cancel rejected: %s (qmt_id=%s, result=%s)",
                            order_id, o.broker_order_id, result,
                        )
                        return False
                    logger.info("QMT cancel requested: %s (qmt_id=%s)", order_id, o.broker_order_id)
                    return True
                except Exception as e:
                    logger.error("QMT cancel failed: %s", e)
                    return False
        return False

    # ── Queries ──

    def get_positions(self) -> list[Position] | None:
        """Query current positions from QMT.

        Reconnects first; when the broker stays unreachable this raises
        ``BrokerConnectionError`` (fail loud: a fabricated flat book is more
        dangerous than an error).

        Returns the cached book when the query raises, and **None** when the
        terminal answers with no result at all. None means "unknown":
        `query_stock_positions` returning None used to be converted to `[]`,
        which the engine and health check read as a flat account -- one
        transient failure made the engine buy a duplicate stake and prune the
        persisted book (D6).
        """
        self._ensure_connected()

        try:
            from quant_platform.trading.qmt_utils import from_qmt_code, qmt_position_to_dict

            raw = self._trader.query_stock_positions(self._account)
            if raw is None:
                logger.error("QMT query positions returned no result; reporting UNKNOWN")
                return None

            positions: list[Position] = []
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
        """Return tracked orders, refreshed from the broker at most per TTL.

        Entry point for lazy reconnect: a down link is re-established first
        (``BrokerConnectionError`` when that fails — a read path must not
        silently serve a stale book as if it were the broker's answer).

        Once connected, the broker's order/trade books are reconciled into
        memory at most once per ``resync_ttl`` seconds (default 2s), so a
        polling caller does not hit QMT on every call. Within the TTL the
        in-memory book can be up to ``resync_ttl`` seconds old — that bound
        is the documented freshness contract. If the resync query itself
        fails, a warning is logged and the current in-memory list is
        returned unchanged (read-path degradation).
        """
        self._ensure_connected()
        self._maybe_resync()
        if status:
            return [o for o in self._orders if o.status == status]
        return list(self._orders)

    def get_account(self) -> dict:
        """Get account balance from QMT.

        Reconnects first; when the broker stays unreachable this raises
        ``BrokerConnectionError`` (fail loud rather than serving a synthetic
        balance). Falls back to internal cash tracking when the asset query
        itself fails.
        """
        self._ensure_connected()

        base = {
            "connected": self._connected,
            "broker": "qmt",
            "mode": self._mode,
            "account_id": self._account_id,
        }

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
                    "n_positions": len(self.get_positions() or []),
                }

        except Exception as e:
            logger.warning("QMT account query failed: %s", e)

        # Fallback
        positions = self.get_positions() or []
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

    def get_quotes(self, codes: list[str]) -> dict[str, float]:
        """Fetch fresh last prices for ``codes`` from QMT market data.

        This is the quote feed a live run must use. Returns an empty mapping
        when xtdata is unavailable or the query fails -- the caller must treat
        that as "no fresh quote" and skip the cycle rather than inventing a
        price.
        """
        if not codes:
            return {}
        try:
            from xtquant import xtdata

            from quant_platform.trading.qmt_utils import to_qmt_code

            qmt_codes = [to_qmt_code(code) for code in codes]
            ticks = xtdata.get_full_tick(qmt_codes) or {}
            prices: dict[str, float] = {}
            for original, qmt_code in zip(codes, qmt_codes, strict=False):
                tick = ticks.get(qmt_code)
                if not tick:
                    continue
                raw_price = (
                    tick.get("lastPrice") if isinstance(tick, dict)
                    else getattr(tick, "lastPrice", None)
                )
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    prices[original] = price
            return prices
        except Exception as e:
            logger.warning("QMT quote query failed: %s", e)
            return {}

    def _on_async_ack(self, response: Any) -> None:
        """Handle the placement ACK.

        `on_order_stock_async_response` reports the broker id, not a fill.
        The old handler parsed it as a status update, mapped the missing
        status to 0/PENDING and downgraded a fresh SUBMITTED order.
        """
        d = response if isinstance(response, dict) else {}
        order_id = d.get("order_id", getattr(response, "order_id", ""))
        if order_id:
            logger.debug("QMT placement ACK: %s", order_id)
            self._publish_event("qmt.order_ack", {"order_id": str(order_id)})

    def _on_order_update(self, xt_order: Any) -> None:
        """Apply an XtOrder status update to the tracked order.

        Callbacks can arrive out of order: a late ``ORDER_PART_SUCC`` snapshot
        after ``ORDER_SUCCEEDED``, or a stale ``ORDER_REPORTED`` after a
        partial fill. Terminal states must never be downgraded, partial fills
        must never fall back to PENDING/SUBMITTED, and the cumulative traded
        volume must not shrink.
        """
        from quant_platform.trading.qmt_utils import qmt_status_to_internal

        broker_order_id = str(getattr(xt_order, "order_id", ""))
        status_code = getattr(xt_order, "order_status", None)
        if status_code is None:
            return
        status = qmt_status_to_internal(int(status_code))
        traded = int(getattr(xt_order, "traded_volume", 0) or 0)
        traded_price = float(getattr(xt_order, "traded_price", 0) or 0)

        for order in self._orders:
            if str(order.broker_order_id) == broker_order_id:
                if order.status in self._TERMINAL_STATUSES:
                    logger.debug(
                        "Ignoring stale QMT update %s for terminal order %s (%s)",
                        status.value, order.order_id, order.status.value,
                    )
                    break
                if (order.status == OrderStatus.PARTIAL
                        and status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)):
                    logger.debug(
                        "Ignoring stale QMT update %s for partially filled order %s",
                        status.value, order.order_id,
                    )
                    break
                order.status = status
                if traded > 0:
                    order.filled_quantity = max(
                        int(order.filled_quantity or 0), traded,
                    )
                    if traded_price > 0:
                        order.filled_price = traded_price
                order.updated_at = datetime.now().isoformat()
                self._publish_event("qmt.order_update", order.to_dict())
                break

    def _on_trade(self, xt_trade: Any) -> None:
        """Apply an XtTrade fill report to the tracked order."""
        broker_order_id = str(getattr(xt_trade, "order_id", ""))
        volume = int(
            getattr(xt_trade, "traded_volume", 0)
            or getattr(xt_trade, "volume", 0)
            or 0
        )
        price = float(
            getattr(xt_trade, "traded_price", 0)
            or getattr(xt_trade, "price", 0)
            or 0
        )
        if volume <= 0:
            return

        for order in self._orders:
            if str(order.broker_order_id) == broker_order_id:
                order.filled_quantity = min(
                    order.quantity, int(order.filled_quantity or 0) + volume,
                )
                if price > 0:
                    order.filled_price = price
                order.status = (
                    OrderStatus.FILLED
                    if order.filled_quantity >= order.quantity
                    else OrderStatus.PARTIAL
                )
                order.updated_at = datetime.now().isoformat()
                self._publish_event("qmt.trade", order.to_dict())
                break

    def _on_async_response(self, response: Any) -> None:
        """Handle an async order response carrying a status (legacy hook)."""
        try:
            from quant_platform.trading.qmt_utils import (
                qmt_status_to_internal,
                qmt_trade_to_dict,
            )

            d = qmt_trade_to_dict(response)
            if "status" not in d and "order_status" not in d:
                # Placement ACK without lifecycle information: do not touch
                # the order's status.
                return
            qmt_order_id = str(d.get("order_id", ""))
            qmt_status = d.get("status", d.get("order_status", 0))
            filled_qty = d.get("filled_quantity", d.get("volume", 0))
            filled_price = d.get("filled_price", d.get("price", 0.0))

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
        target = str(broker_order_id)
        for o in self._orders:
            # xtquant reports integer ids while broker_order_id is stored as
            # a string; comparing raw values never matched.
            if str(o.broker_order_id) == target or str(o.order_id) == target:
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
            # Create trader API for orders and quote API for market data
            self._trader = self._xtp.TraderApi(self._client_id, self._data_folder)
            self._trader.Login(
                self._server_ip, self._server_port,
                self._key, self._session_id, self._protocol
            )

            self._quote = self._xtp.QuoteApi(self._data_folder)
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

    def get_positions(self) -> list[Position] | None:
        """Query positions. Returns None when the query itself failed.

        An exception used to be reported as `[]` ("flat"), which could make a
        caller size duplicate exposure (D6). Unknown and empty are now
        distinguishable.
        """
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
            return None

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
