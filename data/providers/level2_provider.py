"""Level 2 market data provider.

Provides order book snapshots and tick-by-tick trade data:
- Level 2 order book: 10-level bid/ask with queue sizes
- Tick data: every trade with price/volume/direction
- VWAP/TWAP computed from tick data

Usage:
    l2 = Level2DataProvider()
    book = l2.get_order_book("600519")
    ticks = l2.get_ticks("600519", start="09:30:00", end="10:00:00")
    vwap = l2.compute_vwap("600519")
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    volume: float
    order_count: int = 0

    def to_dict(self) -> dict:
        return {"price": self.price, "volume": self.volume, "order_count": self.order_count}


@dataclass
class OrderBookSnapshot:
    """Full order book snapshot (10-level bid/ask)."""
    code: str
    timestamp: str
    bids: list[OrderBookLevel] = field(default_factory=list)  # sorted desc by price
    asks: list[OrderBookLevel] = field(default_factory=list)  # sorted asc by price

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid > 0:
            return self.spread / mid * 10000
        return 0.0

    @property
    def bid_depth(self) -> float:
        """Total bid volume across all levels."""
        return sum(level.volume for level in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total ask volume across all levels."""
        return sum(level.volume for level in self.asks)

    @property
    def depth_imbalance(self) -> float:
        """Order book imbalance: positive = more bid pressure."""
        total = self.bid_depth + self.ask_depth
        if total > 0:
            return (self.bid_depth - self.ask_depth) / total
        return 0.0

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "timestamp": self.timestamp,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": round(self.mid_price, 4),
            "spread": round(self.spread, 4),
            "spread_bps": round(self.spread_bps, 2),
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "depth_imbalance": round(self.depth_imbalance, 4),
            "bids": [lv.to_dict() for lv in self.bids],
            "asks": [lv.to_dict() for lv in self.asks],
        }


@dataclass
class TickData:
    """Single tick/trade record."""
    code: str
    timestamp: str
    price: float
    volume: float
    amount: float
    direction: str = ""  # "B" = buy, "S" = sell, "" = unknown
    trade_type: str = ""  # "regular", "block", "odd_lot"

    def to_dict(self) -> dict:
        return {
            "code": self.code, "timestamp": self.timestamp,
            "price": self.price, "volume": self.volume,
            "amount": self.amount, "direction": self.direction,
        }


class Level2DataProvider:
    """Level 2 market data provider with simulated data.

    In production, this would connect to a real L2 data feed.
    The simulated version generates realistic order book and tick data
    for testing and development.

    Args:
        codes: Stock codes to track
        n_levels: Order book depth (default 10)
        tick_buffer_size: Max ticks to buffer per stock
    """

    def __init__(
        self,
        codes: list[str] | None = None,
        n_levels: int = 10,
        tick_buffer_size: int = 10000,
    ):
        self._codes = codes or ["600519", "000001", "300750"]
        self._n_levels = n_levels
        self._tick_buffer_size = tick_buffer_size

        self._books: dict[str, OrderBookSnapshot] = {}
        self._ticks: dict[str, deque[TickData]] = {}
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(42)

        # Base prices for simulation
        self._base_prices = {code: 100 + self._rng.random() * 200 for code in self._codes}
        self._running = False
        self._thread: threading.Thread | None = None

    def get_order_book(self, code: str) -> OrderBookSnapshot | None:
        """Get current order book snapshot."""
        with self._lock:
            return self._books.get(code)

    def get_ticks(
        self,
        code: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[TickData]:
        """Get tick data, optionally filtered by time range."""
        with self._lock:
            ticks = list(self._ticks.get(code, []))

        if start:
            ticks = [t for t in ticks if t.timestamp >= start]
        if end:
            ticks = [t for t in ticks if t.timestamp <= end]

        return ticks[-limit:]

    def compute_vwap(self, code: str, n_ticks: int = 100) -> float:
        """Compute VWAP from recent ticks."""
        ticks = self.get_ticks(code, limit=n_ticks)
        if not ticks:
            return 0.0

        total_amount = sum(t.amount for t in ticks)
        total_volume = sum(t.volume for t in ticks)
        if total_volume > 0:
            return total_amount / total_volume
        return 0.0

    def compute_trade_flow(self, code: str, n_ticks: int = 100) -> dict:
        """Compute buy/sell trade flow imbalance."""
        ticks = self.get_ticks(code, limit=n_ticks)
        if not ticks:
            return {"buy_volume": 0, "sell_volume": 0, "net_flow": 0, "buy_pct": 0}

        buy_volume = sum(t.volume for t in ticks if t.direction == "B")
        sell_volume = sum(t.volume for t in ticks if t.direction == "S")
        total = buy_volume + sell_volume

        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_flow": buy_volume - sell_volume,
            "buy_pct": round(buy_volume / total * 100, 2) if total > 0 else 50,
        }

    def get_all_books(self) -> dict[str, OrderBookSnapshot]:
        """Get all order book snapshots."""
        with self._lock:
            return dict(self._books)

    def start(self):
        """Start generating simulated data."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._simulate_loop, daemon=True)
        self._thread.start()
        logger.info("Level2 provider started for %d stocks", len(self._codes))

    def stop(self):
        """Stop data generation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Level2 provider stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "codes": len(self._codes),
                "books_cached": len(self._books),
                "total_ticks": sum(len(t) for t in self._ticks.values()),
                "running": self._running,
            }

    # ── Simulation ──

    def _simulate_loop(self):
        """Generate simulated L2 data at ~100ms intervals."""
        while self._running:
            for code in self._codes:
                self._generate_book(code)
                self._generate_tick(code)
            time.sleep(0.1)

    def _generate_book(self, code: str):
        """Generate a simulated order book snapshot."""
        base = self._base_prices.get(code, 100)
        tick_size = 0.01

        mid = base * (1 + self._rng.normal(0, 0.001))
        self._base_prices[code] = mid

        bids = []
        asks = []
        for i in range(self._n_levels):
            bid_price = round(mid - (i + 1) * tick_size * (1 + self._rng.random()), 2)
            ask_price = round(mid + (i + 1) * tick_size * (1 + self._rng.random()), 2)
            bid_vol = round(self._rng.exponential(5000) + 100)
            ask_vol = round(self._rng.exponential(5000) + 100)
            bid_count = max(1, int(self._rng.poisson(5)))
            ask_count = max(1, int(self._rng.poisson(5)))

            bids.append(OrderBookLevel(price=bid_price, volume=bid_vol, order_count=bid_count))
            asks.append(OrderBookLevel(price=ask_price, volume=ask_vol, order_count=ask_count))

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        snapshot = OrderBookSnapshot(
            code=code,
            timestamp=datetime.now().isoformat(),
            bids=bids,
            asks=asks,
        )

        with self._lock:
            self._books[code] = snapshot

    def _generate_tick(self, code: str):
        """Generate a simulated tick."""
        book = self._books.get(code)
        if not book:
            return

        # Randomly pick between bid/ask price
        if self._rng.random() > 0.5:
            price = book.best_ask
            direction = "B"
        else:
            price = book.best_bid
            direction = "S"

        volume = max(1, round(self._rng.exponential(200)))
        amount = price * volume

        tick = TickData(
            code=code,
            timestamp=datetime.now().isoformat(),
            price=price,
            volume=volume,
            amount=amount,
            direction=direction,
        )

        with self._lock:
            if code not in self._ticks:
                self._ticks[code] = deque(maxlen=self._tick_buffer_size)
            self._ticks[code].append(tick)


class OrderBookAnalytics:
    """Analytics computed from order book data.

    Useful for microstructure factors and execution optimization.
    """

    @staticmethod
    def effective_spread(book: OrderBookSnapshot, trade_price: float) -> float:
        """Effective spread: 2 * |trade_price - mid_price|."""
        mid = book.mid_price
        if mid > 0:
            return 2 * abs(trade_price - mid)
        return 0.0

    @staticmethod
    def weighted_mid_price(book: OrderBookSnapshot) -> float:
        """Volume-weighted mid price."""
        if not book.bids or not book.asks:
            return 0.0

        bid_vol = book.bids[0].volume
        ask_vol = book.asks[0].volume
        total = bid_vol + ask_vol

        if total > 0:
            return (book.best_bid * ask_vol + book.best_ask * bid_vol) / total
        return book.mid_price

    @staticmethod
    def book_pressure(book: OrderBookSnapshot, n_levels: int = 5) -> dict:
        """Order book pressure analysis."""
        bid_vol = sum(lv.volume for lv in book.bids[:n_levels])
        ask_vol = sum(lv.volume for lv in book.asks[:n_levels])
        total = bid_vol + ask_vol

        return {
            "bid_pressure": bid_vol,
            "ask_pressure": ask_vol,
            "net_pressure": bid_vol - ask_vol,
            "pressure_ratio": round(bid_vol / total, 4) if total > 0 else 0.5,
        }

    @staticmethod
    def book_slope(book: OrderBookSnapshot, n_levels: int = 5) -> dict:
        """Order book slope: how quickly depth increases away from mid.

        Steeper slope = more confidence in current price level.
        """
        if len(book.bids) < 2 or len(book.asks) < 2:
            return {"bid_slope": 0, "ask_slope": 0}

        # Compute slope as volume change per price level
        bid_vols = [lv.volume for lv in book.bids[:n_levels]]
        ask_vols = [lv.volume for lv in book.asks[:n_levels]]

        bid_slope = np.polyfit(range(len(bid_vols)), bid_vols, 1)[0] if len(bid_vols) > 1 else 0
        ask_slope = np.polyfit(range(len(ask_vols)), ask_vols, 1)[0] if len(ask_vols) > 1 else 0

        return {
            "bid_slope": round(float(bid_slope), 2),
            "ask_slope": round(float(ask_slope), 2),
            "slope_asymmetry": round(float(bid_slope - ask_slope), 2),
        }
