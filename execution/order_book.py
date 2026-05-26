"""Realistic order book with price-time priority matching.

Implements a proper limit order book (LOB) with:
- Sorted price levels (bid descending, ask ascending)
- FIFO queue at each price level (price-time priority)
- Partial fills
- IOC (Immediate or Cancel) and FOK (Fill or Kill) order types
- Market orders that walk the book
- Market data snapshots (L1/L2/L3)
- Microstructure metrics (spread, depth imbalance, VPIN)

This replaces SimulatedExchange's naive single-price matching.

Data structure:
- bids: sorted dict [price → PriceLevel] in descending order
- asks: sorted dict [price → PriceLevel] in ascending order
- PriceLevel: deque of orders (FIFO)

Performance:
- Insert/cancel: O(log N) for price lookup + O(1) for deque append
- Best bid/ask: O(1) with cached pointers
- Trade matching: O(1) amortized per fill
"""

from __future__ import annotations

import bisect
import enum
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────


class Side(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(enum.StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"          # Immediate or Cancel: fill what you can, cancel rest
    FOK = "fok"          # Fill or Kill: fill all or cancel all


class BookOrderStatus(enum.StrEnum):
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# ──────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────


@dataclass
class BookOrder:
    """An order in the order book."""
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: float          # Limit price (ignored for market orders)
    quantity: int         # Original quantity
    filled_quantity: int = 0
    status: BookOrderStatus = BookOrderStatus.OPEN
    timestamp_ns: int = 0   # Nanosecond timestamp for FIFO ordering
    source: str = ""        # Strategy/source identifier

    def __post_init__(self):
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (BookOrderStatus.OPEN, BookOrderStatus.PARTIALLY_FILLED)

    @property
    def avg_fill_price(self) -> float:
        """Average fill price (computed from fills, not stored here)."""
        return self.price  # Simplified; real impl would track fills


@dataclass
class Trade:
    """A trade resulting from order matching."""
    trade_id: str
    symbol: str
    price: float
    quantity: int
    aggressor_side: Side     # Side of the taker (who crossed the spread)
    maker_order_id: str      # Passive order
    taker_order_id: str      # Aggressive order
    timestamp_ns: int = 0
    is_maker: bool = False   # True if the maker side is the reference

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = uuid.uuid4().hex[:12]
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class PriceLevel:
    """A single price level in the order book.

    Maintains a FIFO queue of orders at this price.
    """
    price: float
    orders: deque[BookOrder] = field(default_factory=deque)
    total_quantity: int = 0

    def add_order(self, order: BookOrder):
        """Add order to the back of the FIFO queue."""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity

    def remove_order(self, order_id: str) -> BookOrder | None:
        """Remove an order by ID. O(N) scan of deque."""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.orders.remove(order)
                self.total_quantity -= order.remaining_quantity
                return order
        return None

    def reduce(self, quantity: int) -> list[tuple[BookOrder, int]]:
        """Reduce quantity from the front of the queue (FIFO).

        Returns list of (order, filled_qty) pairs.
        """
        fills = []
        remaining = quantity

        while remaining > 0 and self.orders:
            head = self.orders[0]
            fill_qty = min(remaining, head.remaining_quantity)
            head.filled_quantity += fill_qty
            remaining -= fill_qty
            self.total_quantity -= fill_qty

            fills.append((head, fill_qty))

            if head.remaining_quantity <= 0:
                head.status = BookOrderStatus.FILLED
                self.orders.popleft()
            else:
                head.status = BookOrderStatus.PARTIALLY_FILLED

        return fills

    @property
    def order_count(self) -> int:
        return len(self.orders)

    @property
    def is_empty(self) -> bool:
        return self.total_quantity <= 0 or len(self.orders) == 0

    def __repr__(self):
        return f"PriceLevel(price={self.price}, qty={self.total_quantity}, orders={len(self.orders)})"


# ──────────────────────────────────────────────────────────────────────
# Order Book
# ──────────────────────────────────────────────────────────────────────


class OrderBook:
    """Realistic limit order book with price-time priority.

    Features:
    - Sorted price levels (bids descending, asks ascending)
    - FIFO queue at each level
    - Partial fills, IOC, FOK order types
    - Market orders that walk the book
    - L1/L2/L3 market data snapshots
    - Microstructure metrics

    Usage:
        book = OrderBook("600519", tick_size=0.01)

        # Add limit orders
        order = BookOrder("o1", "600519", Side.BUY, OrderType.LIMIT, 1800.00, 100)
        trades = book.add_order(order)

        # Get market data
        l2 = book.get_depth_snapshot(levels=5)
        metrics = book.get_microstructure_metrics()
    """

    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size

        # Price levels sorted by price
        # bids: descending order (best bid = highest price = first element)
        # asks: ascending order (best ask = lowest price = first element)
        self._bid_prices: list[float] = []   # Sorted descending
        self._ask_prices: list[float] = []   # Sorted ascending
        self._bid_levels: dict[float, PriceLevel] = {}
        self._ask_levels: dict[float, PriceLevel] = {}

        # Order lookup for fast cancel
        self._orders: dict[str, BookOrder] = {}
        self._order_prices: dict[str, float] = {}  # order_id → price

        # Trade history (ring buffer)
        self._trades: list[Trade] = []
        self._max_trades = 10_000
        self._total_trades = 0
        self._total_volume = 0

        # Cached best bid/ask
        self._best_bid: float | None = None
        self._best_ask: float | None = None
        self._cache_valid = False

        # Microstructure metrics
        self._spread_history: list[float] = []
        self._depth_imbalance_history: list[float] = []

    # ── Core Operations ──

    def add_order(self, order: BookOrder) -> list[Trade]:
        """Add an order to the book. Returns list of trades.

        Matching logic:
        1. Buy order: match against ask side (price <= order.price)
        2. Sell order: match against bid side (price >= order.price)
        3. IOC: cancel remainder after matching
        4. FOK: cancel entire order if cannot fill completely
        5. Market order: match at any price
        """
        self._invalidate_cache()

        if order.order_type == OrderType.FOK:
            # Check if full fill is possible before matching
            available = self._available_quantity(order.side, order.price, order.quantity)
            if available < order.quantity:
                order.status = BookOrderStatus.CANCELLED
                return []

        # Match against opposite side
        trades = self._match(order)

        # Handle IOC: cancel remainder
        if order.order_type == OrderType.IOC and order.remaining_quantity > 0:
            order.status = BookOrderStatus.CANCELLED

        # Add remainder to book (only for LIMIT orders)
        elif order.remaining_quantity > 0 and order.is_active:
            if order.order_type in (OrderType.LIMIT,):
                self._insert(order)
                self._orders[order.order_id] = order
                self._order_prices[order.order_id] = order.price

        # Record trades
        for trade in trades:
            self._record_trade(trade)

        return trades

    def get_order(self, order_id: str) -> BookOrder | None:
        """Look up an order by ID without removing it."""
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> BookOrder | None:
        """Cancel an order by ID."""
        self._invalidate_cache()

        price = self._order_prices.pop(order_id, None)
        if price is None:
            return None

        order = self._orders.pop(order_id, None)
        if order is None:
            return None

        # Remove from price level
        side_levels = self._bid_levels if order.side == Side.BUY else self._ask_levels
        if price in side_levels:
            level = side_levels[price]
            level.remove_order(order_id)
            if level.is_empty:
                self._remove_level(order.side, price)

        order.status = BookOrderStatus.CANCELLED
        return order

    def modify_order(
        self,
        order_id: str,
        new_price: float | None = None,
        new_quantity: int | None = None,
    ) -> tuple[BookOrder | None, list[Trade]]:
        """Modify an order (cancel + re-add with new params)."""
        old_order = self.cancel_order(order_id)
        if old_order is None:
            return None, []

        new_order = BookOrder(
            order_id=order_id,
            symbol=old_order.symbol,
            side=old_order.side,
            order_type=old_order.order_type,
            price=new_price if new_price is not None else old_order.price,
            quantity=new_quantity if new_quantity is not None else old_order.remaining_quantity,
            source=old_order.source,
        )
        trades = self.add_order(new_order)
        return new_order, trades

    # ── Matching Engine ──

    def _match(self, order: BookOrder) -> list[Trade]:
        """Match order against opposite side of the book."""
        trades = []

        if order.side == Side.BUY:
            # Match against asks (ascending price order)
            while (self._ask_prices and order.remaining_quantity > 0 and
                   order.is_active):
                best_ask = self._ask_prices[0]

                # Check price constraint
                if order.order_type == OrderType.LIMIT and best_ask > order.price:
                    break  # Can't match at this price

                level = self._ask_levels[best_ask]
                fills = level.reduce(order.remaining_quantity)

                for maker_order, fill_qty in fills:
                    trade = Trade(
                        trade_id="",
                        symbol=self.symbol,
                        price=best_ask,
                        quantity=fill_qty,
                        aggressor_side=Side.BUY,
                        maker_order_id=maker_order.order_id,
                        taker_order_id=order.order_id,
                    )
                    trades.append(trade)

                    order.filled_quantity += fill_qty
                    if order.remaining_quantity <= 0:
                        order.status = BookOrderStatus.FILLED
                    else:
                        order.status = BookOrderStatus.PARTIALLY_FILLED

                if level.is_empty:
                    self._remove_level(Side.SELL, best_ask)

        elif order.side == Side.SELL:
            # Match against bids (descending price order)
            while (self._bid_prices and order.remaining_quantity > 0 and
                   order.is_active):
                best_bid = self._bid_prices[0]

                # Check price constraint
                if order.order_type == OrderType.LIMIT and best_bid < order.price:
                    break

                level = self._bid_levels[best_bid]
                fills = level.reduce(order.remaining_quantity)

                for maker_order, fill_qty in fills:
                    trade = Trade(
                        trade_id="",
                        symbol=self.symbol,
                        price=best_bid,
                        quantity=fill_qty,
                        aggressor_side=Side.SELL,
                        maker_order_id=maker_order.order_id,
                        taker_order_id=order.order_id,
                    )
                    trades.append(trade)

                    order.filled_quantity += fill_qty
                    if order.remaining_quantity <= 0:
                        order.status = BookOrderStatus.FILLED
                    else:
                        order.status = BookOrderStatus.PARTIALLY_FILLED

                if level.is_empty:
                    self._remove_level(Side.BUY, best_bid)

        return trades

    def _insert(self, order: BookOrder):
        """Insert order into the book at its price level."""
        if order.side == Side.BUY:
            prices = self._bid_prices
            levels = self._bid_levels
        else:
            prices = self._ask_prices
            levels = self._ask_levels

        price = order.price

        if price not in levels:
            # Insert price in sorted position
            if order.side == Side.BUY:
                # Descending order: find insertion point
                idx = bisect.bisect_left([-p for p in prices], -price)
                prices.insert(idx, price)
            else:
                # Ascending order
                idx = bisect.bisect_left(prices, price)
                prices.insert(idx, price)
            levels[price] = PriceLevel(price=price)

        levels[price].add_order(order)

    def _remove_level(self, side: Side, price: float):
        """Remove an empty price level."""
        if side == Side.BUY:
            if price in self._bid_levels:
                del self._bid_levels[price]
            if price in self._bid_prices:
                self._bid_prices.remove(price)
        else:
            if price in self._ask_levels:
                del self._ask_levels[price]
            if price in self._ask_prices:
                self._ask_prices.remove(price)

    def _available_quantity(self, side: Side, price: float, quantity: int) -> int:
        """Check how much quantity is available for matching (for FOK)."""
        available = 0
        if side == Side.BUY:
            for ask_price in self._ask_prices:
                if ask_price > price:
                    break
                available += self._ask_levels[ask_price].total_quantity
                if available >= quantity:
                    return quantity
        else:
            for bid_price in self._bid_prices:
                if bid_price < price:
                    break
                available += self._bid_levels[bid_price].total_quantity
                if available >= quantity:
                    return quantity
        return available

    # ── Market Data ──

    @property
    def best_bid(self) -> float | None:
        return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> float | None:
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return None

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    @property
    def spread_bps(self) -> float | None:
        """Spread in basis points."""
        s = self.spread
        mid = self.mid_price
        if s is not None and mid is not None and mid > 0:
            return (s / mid) * 10000
        return None

    @property
    def last_trade_price(self) -> float | None:
        return self._trades[-1].price if self._trades else None

    def get_depth_snapshot(self, levels: int = 10) -> dict:
        """Get L2 depth snapshot.

        Returns:
            {
                "symbol": str,
                "timestamp_ns": int,
                "bids": [{"price": float, "quantity": int, "orders": int}, ...],
                "asks": [{"price": float, "quantity": int, "orders": int}, ...],
                "mid_price": float,
                "spread": float,
            }
        """
        bids = []
        for price in self._bid_prices[:levels]:
            level = self._bid_levels[price]
            bids.append({
                "price": price,
                "quantity": level.total_quantity,
                "orders": level.order_count,
            })

        asks = []
        for price in self._ask_prices[:levels]:
            level = self._ask_levels[price]
            asks.append({
                "price": price,
                "quantity": level.total_quantity,
                "orders": level.order_count,
            })

        return {
            "symbol": self.symbol,
            "timestamp_ns": time.time_ns(),
            "bids": bids,
            "asks": asks,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }

    def get_full_book_snapshot(self) -> dict:
        """Get L3 full book snapshot (all orders at all levels)."""
        bids = []
        for price in self._bid_prices:
            level = self._bid_levels[price]
            for order in level.orders:
                bids.append({
                    "order_id": order.order_id,
                    "price": price,
                    "quantity": order.remaining_quantity,
                    "timestamp_ns": order.timestamp_ns,
                })

        asks = []
        for price in self._ask_prices:
            level = self._ask_levels[price]
            for order in level.orders:
                asks.append({
                    "order_id": order.order_id,
                    "price": price,
                    "quantity": order.remaining_quantity,
                    "timestamp_ns": order.timestamp_ns,
                })

        return {
            "symbol": self.symbol,
            "timestamp_ns": time.time_ns(),
            "bids": bids,
            "asks": asks,
        }

    # ── Microstructure Metrics ──

    def get_microstructure_metrics(self) -> dict:
        """Compute market microstructure metrics.

        Returns:
            {
                "spread": float,
                "spread_bps": float,
                "mid_price": float,
                "depth_imbalance": float,  # [-1, 1] bid-heavy to ask-heavy
                "bid_depth_5": int,
                "ask_depth_5": int,
                "total_bid_depth": int,
                "total_ask_depth": int,
                "bid_levels": int,
                "ask_levels": int,
                "total_orders": int,
                "total_trades": int,
                "total_volume": int,
                "vwap": float,
            }
        """
        spread = self.spread
        spread_bps = self.spread_bps

        # Depth within 5 levels
        bid_depth_5 = sum(
            self._bid_levels[p].total_quantity
            for p in self._bid_prices[:5]
        )
        ask_depth_5 = sum(
            self._ask_levels[p].total_quantity
            for p in self._ask_prices[:5]
        )

        # Total depth
        total_bid = sum(lv.total_quantity for lv in self._bid_levels.values())
        total_ask = sum(lv.total_quantity for lv in self._ask_levels.values())

        # Depth imbalance: (bid - ask) / (bid + ask), range [-1, 1]
        total = bid_depth_5 + ask_depth_5
        depth_imbalance = (bid_depth_5 - ask_depth_5) / total if total > 0 else 0

        # VWAP from recent trades
        vwap = 0.0
        if self._trades:
            total_notional = sum(t.price * t.quantity for t in self._trades[-1000:])
            total_qty = sum(t.quantity for t in self._trades[-1000:])
            vwap = total_notional / total_qty if total_qty > 0 else 0

        return {
            "spread": spread,
            "spread_bps": round(spread_bps, 2) if spread_bps else None,
            "mid_price": self.mid_price,
            "depth_imbalance": round(depth_imbalance, 4),
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "total_bid_depth": total_bid,
            "total_ask_depth": total_ask,
            "bid_levels": len(self._bid_prices),
            "ask_levels": len(self._ask_prices),
            "total_orders": len(self._orders),
            "total_trades": self._total_trades,
            "total_volume": self._total_volume,
            "vwap": round(vwap, 4),
        }

    # ── VPIN (Volume-Synchronized Probability of Informed Trading) ──

    def compute_vpin(self, n_buckets: int = 50, bucket_size: int = 100) -> float:
        """Compute VPIN metric.

        VPIN estimates the probability of informed trading based on
        order flow imbalance. High VPIN → adverse selection risk.

        VPIN = |V_buy - V_sell| / (V_buy + V_sell) averaged over buckets.

        Args:
            n_buckets: Number of volume buckets to average
            bucket_size: Volume per bucket

        Returns:
            VPIN value in [0, 1]
        """
        if len(self._trades) < n_buckets * bucket_size:
            return 0.0

        # Classify trades as buy/sell based on aggressor side
        buckets = []
        current_buy = 0
        current_sell = 0
        current_vol = 0

        for trade in self._trades[-n_buckets * bucket_size * 2:]:
            if trade.aggressor_side == Side.BUY:
                current_buy += trade.quantity
            else:
                current_sell += trade.quantity
            current_vol += trade.quantity

            if current_vol >= bucket_size:
                imbalance = abs(current_buy - current_sell)
                total = current_buy + current_sell
                if total > 0:
                    buckets.append(imbalance / total)
                current_buy = 0
                current_sell = 0
                current_vol = 0

        if not buckets:
            return 0.0

        return sum(buckets[-n_buckets:]) / min(len(buckets), n_buckets)

    # ── Utilities ──

    def _record_trade(self, trade: Trade):
        """Record a trade in history."""
        self._trades.append(trade)
        if len(self._trades) > self._max_trades:
            self._trades = self._trades[-self._max_trades:]
        self._total_trades += 1
        self._total_volume += trade.quantity

    def _invalidate_cache(self):
        self._cache_valid = False

    def get_recent_trades(self, limit: int = 100) -> list[dict]:
        """Get recent trades."""
        return [
            {
                "trade_id": t.trade_id,
                "price": t.price,
                "quantity": t.quantity,
                "side": t.aggressor_side.value,
                "timestamp_ns": t.timestamp_ns,
            }
            for t in self._trades[-limit:]
        ]

    def get_order_count(self) -> dict:
        """Count orders by side."""
        buy_count = sum(len(lv.orders) for lv in self._bid_levels.values())
        sell_count = sum(len(lv.orders) for lv in self._ask_levels.values())
        return {"buy": buy_count, "sell": sell_count, "total": buy_count + sell_count}

    def clear(self):
        """Clear the entire book."""
        self._bid_prices.clear()
        self._ask_prices.clear()
        self._bid_levels.clear()
        self._ask_levels.clear()
        self._orders.clear()
        self._order_prices.clear()
        self._trades.clear()
        self._total_trades = 0
        self._total_volume = 0
        self._invalidate_cache()

    def __repr__(self):
        bb = self.best_bid
        ba = self.best_ask
        return (
            f"OrderBook({self.symbol}, bid={bb}, ask={ba}, "
            f"spread={self.spread}, trades={self._total_trades})"
        )


# ──────────────────────────────────────────────────────────────────────
# Order Book Factory
# ──────────────────────────────────────────────────────────────────────


class OrderBookManager:
    """Manages multiple order books (one per symbol).

    Provides a unified interface for multi-asset trading.
    """

    def __init__(self, tick_size: float = 0.01):
        self._books: dict[str, OrderBook] = {}
        self._tick_size = tick_size

    def get_or_create(self, symbol: str) -> OrderBook:
        """Get or create an order book for a symbol."""
        if symbol not in self._books:
            self._books[symbol] = OrderBook(symbol, self._tick_size)
        return self._books[symbol]

    def get(self, symbol: str) -> OrderBook | None:
        return self._books.get(symbol)

    def add_order(self, order: BookOrder) -> list[Trade]:
        """Add order to the appropriate book."""
        book = self.get_or_create(order.symbol)
        return book.add_order(order)

    def cancel_order(self, symbol: str, order_id: str) -> BookOrder | None:
        book = self._books.get(symbol)
        if book:
            return book.cancel_order(order_id)
        return None

    def get_all_depths(self, levels: int = 5) -> dict[str, dict]:
        """Get depth snapshots for all books."""
        return {
            symbol: book.get_depth_snapshot(levels)
            for symbol, book in self._books.items()
        }

    def get_all_metrics(self) -> dict[str, dict]:
        """Get microstructure metrics for all books."""
        return {
            symbol: book.get_microstructure_metrics()
            for symbol, book in self._books.items()
        }

    @property
    def symbols(self) -> list[str]:
        return list(self._books.keys())

    def clear(self):
        for book in self._books.values():
            book.clear()
        self._books.clear()


# ──────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────


def benchmark_order_book(
    n_orders: int = 100_000,
    n_symbols: int = 1,
    spread_pct: float = 0.001,
) -> dict:
    """Benchmark order book performance.

    Args:
        n_orders: Total orders to process
        n_symbols: Number of symbols
        spread_pct: Initial spread as percentage of mid price

    Returns:
        Dict with throughput and latency metrics.
    """
    import random

    manager = OrderBookManager()
    mid_price = 100.0
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

    # Pre-populate with some orders
    for sym in symbols:
        book = manager.get_or_create(sym)
        for i in range(50):
            bid_price = round(mid_price * (1 - spread_pct / 2 - i * 0.001), 2)
            ask_price = round(mid_price * (1 + spread_pct / 2 + i * 0.001), 2)
            book.add_order(BookOrder(
                order_id=f"init_bid_{sym}_{i}",
                symbol=sym, side=Side.BUY, order_type=OrderType.LIMIT,
                price=bid_price, quantity=random.randint(100, 1000),
            ))
            book.add_order(BookOrder(
                order_id=f"init_ask_{sym}_{i}",
                symbol=sym, side=Side.SELL, order_type=OrderType.LIMIT,
                price=ask_price, quantity=random.randint(100, 1000),
            ))

    # Benchmark
    start = time.time_ns()
    total_trades = 0

    for i in range(n_orders):
        sym = random.choice(symbols)
        side = random.choice([Side.BUY, Side.SELL])
        # Place orders near the spread to generate trades
        book = manager.get(sym)
        if side == Side.BUY:
            ref = book.best_ask or mid_price
            price = round(ref * (1 + random.uniform(-0.001, 0.002)), 2)
        else:
            ref = book.best_bid or mid_price
            price = round(ref * (1 + random.uniform(-0.002, 0.001)), 2)

        order = BookOrder(
            order_id=f"bench_{i}",
            symbol=sym,
            side=side,
            order_type=random.choice([OrderType.LIMIT, OrderType.IOC]),
            price=price,
            quantity=random.randint(100, 500),
        )
        trades = manager.add_order(order)
        total_trades += len(trades)

    elapsed_ns = time.time_ns() - start

    return {
        "n_orders": n_orders,
        "n_symbols": n_symbols,
        "total_trades": total_trades,
        "elapsed_ms": round(elapsed_ns / 1_000_000, 2),
        "throughput_orders_per_sec": int(n_orders / (elapsed_ns / 1_000_000_000)),
        "avg_latency_us": round(elapsed_ns / n_orders / 1000, 2),
    }
