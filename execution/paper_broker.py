"""Enhanced Paper Trading Broker — realistic simulation for backtesting.

Extends the existing order book with institutional-grade simulation features:

1. Latency Simulation:
   - Network latency (round-trip, configurable per exchange)
   - Exchange processing time (matching engine delay)
   - Jitter (random variation)
   - Geographic latency model (SSE/SZSE/CFFEX have different latencies)

2. Partial Fills:
   - Probability-driven: larger orders more likely to get partial fills
   - Queue position degradation: orders that sit longer lose priority
   - Volume-based: fill rate depends on available volume at price level

3. Cancel Failures:
   - Stochastic: probability of cancel rejection based on queue position
   - Race condition: if order is about to be matched, cancel can fail
   - Market state: harder to cancel in fast-moving markets

4. L2 Replay:
   - Replay historical Level 2 order book snapshots
   - Reconstruct bid/ask depth from saved market data
   - Feed into SimulatedBroker for realistic backtest fills

5. Latency Arbitrage Detection:
   - Monitor for stale-price arbitrage opportunities
   - Flag patterns that would be caught in live trading
   - Log warning when backtest results are unrealistic

Usage:
    broker = PaperBroker(
        initial_cash=10_000_000,
        asset_universe=universe,
        latency_model=LatencyModel.LOW,
        partial_fill_rate=0.15,
        cancel_fail_rate=0.02,
    )
    order = broker.place_order(Order(...))
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import (
    empty_corporate_actions,
    normalize_corporate_actions,
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
from quant_platform.shared.symbols import normalize_symbol
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Default latency by exchange (round-trip in milliseconds)
EXCHANGE_LATENCY_MS: dict[str, float] = {
    "SSE": 2.0,     # Shanghai Stock Exchange
    "SZSE": 2.0,    # Shenzhen Stock Exchange
    "CFFEX": 1.5,   # China Financial Futures Exchange
    "SHFE": 3.0,    # Shanghai Futures Exchange
    "DCE": 3.5,     # Dalian Commodity Exchange
    "CZCE": 3.5,    # Zhengzhou Commodity Exchange
    "DEFAULT": 5.0,
}


class LatencyModel(StrEnum):
    """Pre-configured latency profiles."""
    ZERO = "zero"       # No latency (instant)
    LOW = "low"         # Co-located (1-3ms)
    MEDIUM = "medium"   # Nearby data center (5-15ms)
    HIGH = "high"       # Remote connection (20-50ms)
    REALISTIC = "realistic"  # Retail broker (50-200ms)


LATENCY_PROFILES: dict[LatencyModel, dict[str, float]] = {
    LatencyModel.ZERO: {"base_ms": 0.0, "jitter_ms": 0.0, "processing_ms": 0.0, "exchange_multiplier": 0.0},
    LatencyModel.LOW: {"base_ms": 3.0, "jitter_ms": 1.0, "processing_ms": 1.0, "exchange_multiplier": 1.0},
    LatencyModel.MEDIUM: {"base_ms": 10.0, "jitter_ms": 5.0, "processing_ms": 3.0, "exchange_multiplier": 1.0},
    LatencyModel.HIGH: {"base_ms": 35.0, "jitter_ms": 15.0, "processing_ms": 5.0, "exchange_multiplier": 1.0},
    LatencyModel.REALISTIC: {"base_ms": 100.0, "jitter_ms": 50.0, "processing_ms": 10.0, "exchange_multiplier": 1.0},
}


@dataclass
class LatencyConfig:
    """Fine-grained latency configuration."""
    base_ms: float = 3.0
    jitter_ms: float = 1.0
    processing_ms: float = 1.0
    exchange_multiplier: float = 1.0  # Multiply exchange-specific latency

    def sample(self, exchange: str = "DEFAULT") -> float:
        """Sample a latency value (ms) with jitter."""
        exchange_base = EXCHANGE_LATENCY_MS.get(exchange, EXCHANGE_LATENCY_MS["DEFAULT"])
        base = (exchange_base + self.base_ms) * self.exchange_multiplier
        jitter = np.random.default_rng().uniform(-self.jitter_ms, self.jitter_ms)
        return max(0.0, base + jitter + self.processing_ms)


@dataclass
class FillRecord:
    """Detailed fill record for audit."""
    order_id: str
    symbol: str
    side: str
    requested_qty: int
    filled_qty: int
    fill_price: float
    latency_ms: float
    partial: bool = False
    cancel_attempted: bool = False
    cancel_succeeded: bool = False
    timestamp_ns: int = 0


@dataclass
class PaperBrokerMetrics:
    """Aggregated paper trading metrics."""
    total_orders: int = 0
    total_trades: int = 0
    partial_fills: int = 0
    cancel_attempts: int = 0
    cancel_failures: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_fill_pct: float = 0.0
    total_slippage_bps: float = 0.0


class PaperBroker:
    """Enhanced paper trading broker with realistic market simulation.

    Combines the OrderBook matching engine with latency simulation,
    stochastic partial fills, cancel failures, and L2 replay capabilities.
    """

    #: The dual-track TCA shadow keeps its own paper book, so splits and
    #: cash dividends must be applied locally to that book on their ex-date.
    supports_corporate_actions = True

    def __init__(
        self,
        initial_cash: float = 10_000_000,
        asset_universe=None,
        latency: LatencyModel | LatencyConfig = LatencyModel.LOW,
        partial_fill_rate: float = 0.10,
        cancel_fail_rate: float = 0.02,
        cash_dividend_tax_rate: float = 0.0,
    ):
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._asset_universe = asset_universe

        # Latency
        if isinstance(latency, LatencyModel):
            prof = LATENCY_PROFILES[latency]
            self._latency = LatencyConfig(**prof)
        else:
            self._latency = latency

        # Simulation parameters
        self._partial_fill_rate = partial_fill_rate
        self._cancel_fail_rate = cancel_fail_rate

        # Order books per symbol
        self._books: dict[str, OrderBook] = {}
        self._pending_orders: dict[str, BookOrder] = {}

        # Comparison book: canonical symbol -> {quantity, avg_cost,
        # current_price}.  Kept from this broker's own fills so the TCA
        # shadow survives corporate actions (a split scales quantity and
        # divides cost; a dividend credits cash).  It is a shadow ledger, not
        # an accounting-grade book: fees are not modelled here.
        self._positions: dict[str, dict] = {}
        self._market_prices: dict[str, float] = {}
        self._today_bought: set[str] = set()

        # Fill history
        self._fill_records: list[FillRecord] = []

        # Metrics
        self._metrics = PaperBrokerMetrics()
        self._latency_samples: list[float] = []

        # L2 replay
        self._l2_snapshots: dict[str, list[dict]] = {}
        self._replay_mode = False
        self._replay_index: dict[str, int] = {}

        # RNG
        self._rng = np.random.default_rng()

        # Corporate actions (comparison book only).  Markers are keyed on
        # ``(canonical_asset, ex_date)`` so a replayed day never double-applies.
        self._cash_dividend_tax_rate = max(0.0, float(cash_dividend_tax_rate))
        self._corporate_actions: pd.DataFrame = empty_corporate_actions()
        self._applied_corporate_actions: set[tuple[str, str]] = set()
        self._corporate_action_events: list[dict] = []
        self._cash_ledger: list[dict] = []

    # ── Order Book Access ──

    def _get_book(self, symbol: str, tick_size: float = 0.01) -> OrderBook:
        if symbol not in self._books:
            self._books[symbol] = OrderBook(symbol, tick_size=tick_size)
        return self._books[symbol]

    def _get_tick_size(self, symbol: str) -> float:
        if self._asset_universe:
            inst = self._asset_universe.get(symbol)
            if inst:
                return inst.tick_size
        return 0.01

    # ── Latency ──

    def _sample_latency_ms(self, symbol: str) -> float:
        exchange = "DEFAULT"
        if self._asset_universe:
            inst = self._asset_universe.get(symbol)
            if inst:
                exchange = inst.exchange
        latency = self._latency.sample(exchange)
        self._latency_samples.append(latency)
        if len(self._latency_samples) > 5_000:
            self._latency_samples = self._latency_samples[-2_500:]
        return latency

    def _record_fill(self, record: FillRecord) -> FillRecord:
        """Append a fill record, keeping the in-memory list bounded."""
        self._fill_records.append(record)
        if len(self._fill_records) > 5_000:
            self._fill_records = self._fill_records[-2_500:]
        return record

    def _wait_latency(self, symbol: str, actual_wait: bool = False):
        """Simulate network/exchange latency.

        If actual_wait=True, sleep for the sampled duration (for live paper trading).
        For backtesting, just record the latency without sleeping.
        """
        latency_ms = self._sample_latency_ms(symbol)
        if actual_wait:
            time.sleep(latency_ms / 1000.0)
        return latency_ms

    # ── Partial Fill Logic ──

    def _should_partial_fill(self, order_qty: int, available_qty: int) -> bool:
        """Determine if an order should be partially filled.

        Larger orders relative to available volume are more likely
        to get partial fills.
        """
        if available_qty <= 0:
            return True

        size_ratio = order_qty / max(available_qty, 1)
        # Probability increases with size relative to available volume
        prob = self._partial_fill_rate * min(size_ratio, 5.0)
        return self._rng.random() < prob

    def _available_liquidity(self, book: OrderBook, order: BookOrder) -> int:
        """Quantity the book could immediately match against ``order``."""
        depth = book.get_depth_snapshot(levels=10_000)
        levels = depth["asks"] if order.side == BookSide.BUY else depth["bids"]

        available = 0
        for level in levels:
            price = level["price"]
            if order.order_type == BookOrderType.LIMIT:
                if order.side == BookSide.BUY and price > order.price:
                    break
                if order.side == BookSide.SELL and price < order.price:
                    break
            available += level["quantity"]
            if available >= order.quantity:
                return order.quantity
        return available

    def _apply_partial_fill(self, book: OrderBook, order: BookOrder) -> bool:
        """Size a partial fill *before* the order reaches the matching engine.

        The old `_simulate_partial_fill` let `OrderBook.add_order` match the
        whole aggressor and then rewrote ``order.filled_quantity`` downwards.
        The resting remainder kept the pre-rewrite quantity, so a later order
        could trade against shares that had already been matched away, and
        the book's trade volume disagreed with the broker's fill records.

        Shrinking the aggressor first keeps the order, the book and the
        reported fills consistent. Returns True if the order was shrunk.
        """
        available = self._available_liquidity(book, order)
        if available <= 0:
            return False
        if not self._should_partial_fill(order.quantity, available):
            return False

        # Fill 30-80% of what the LOB would have filled.
        fill_pct = self._rng.uniform(0.3, 0.8)
        target = max(1, int(min(order.quantity, available) * fill_pct))
        order.quantity = target

        self._metrics.partial_fills += 1
        return True

    # ── Cancel Failure Logic ──

    def _should_cancel_fail(self, order: BookOrder) -> bool:
        """Determine if a cancel request should fail.

        Cancel failures happen when:
        1. Order is at the front of the queue and about to be matched
        2. Market is moving fast (high volatility)
        3. Pure random exchange glitch
        """
        if order.filled_quantity > 0 and order.filled_quantity >= order.quantity * 0.5:
            # Order already >50% filled, harder to cancel
            return self._rng.random() < self._cancel_fail_rate * 3

        base_prob = self._cancel_fail_rate
        # Active orders at the best price level are harder to cancel
        if order.remaining_quantity > 0:
            base_prob *= 1.5

        return self._rng.random() < base_prob

    # ── L2 Replay ──

    def load_l2_snapshots(self, symbol: str, snapshots: list[dict]):
        """Load historical L2 snapshots for replay.

        Each snapshot dict:
            timestamp: str or int (ns)
            bids: [[price, quantity], ...]   descending
            asks: [[price, quantity], ...]   ascending
        """
        self._l2_snapshots[symbol] = snapshots
        self._replay_index[symbol] = 0

    def replay_l2(self, symbol: str, current_time_ns: int) -> dict | None:
        """Get the L2 snapshot at or before current_time_ns.

        Advances the replay index to maintain temporal ordering.
        Returns None if no more snapshots available.
        """
        snapshots = self._l2_snapshots.get(symbol)
        if not snapshots:
            return None

        idx = self._replay_index.get(symbol, 0)
        # Advance to the most recent snapshot before current_time
        while idx < len(snapshots):
            snap = snapshots[idx]
            ts = snap.get("timestamp", 0)
            if isinstance(ts, str):
                ts = int(ts)
            if ts > current_time_ns:
                break
            idx += 1

        self._replay_index[symbol] = idx
        if idx > 0:
            return snapshots[idx - 1]
        return None

    def seed_book_from_snapshot(self, symbol: str, snapshot: dict):
        """Seed an order book from an L2 snapshot.

        Seeded levels are passive: they rest and never aggress. Without this,
        the ask ladder is added *into* the bid ladder through the matching
        engine, so a locked snapshot (best bid == best ask) trades against
        itself before any participant order arrives -- destroying depth and
        fabricating a print. Incoming broker orders still fill against the
        seeded levels.
        """
        book = OrderBook(symbol)
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])

        mm_id = f"l2_replay_{uuid.uuid4().hex[:8]}"
        for i, (price, qty) in enumerate(bids[:10]):
            order = BookOrder(
                order_id=f"{mm_id}_b{i}",
                symbol=symbol,
                side=BookSide.BUY,
                order_type=BookOrderType.LIMIT,
                price=float(price),
                quantity=int(qty),
                source="l2_replay",
                passive=True,
            )
            book.add_order(order)

        for i, (price, qty) in enumerate(asks[:10]):
            order = BookOrder(
                order_id=f"{mm_id}_a{i}",
                symbol=symbol,
                side=BookSide.SELL,
                order_type=BookOrderType.LIMIT,
                price=float(price),
                quantity=int(qty),
                source="l2_replay",
                passive=True,
            )
            book.add_order(order)

        self._books[symbol] = book

    # ── Order Placement ──

    def place_order(self, order) -> FillRecord:
        """Place an order with realistic simulation.

        Flow:
        1. Simulate network latency (order acceptance)
        2. Match against order book
        3. Apply partial fill simulation
        4. Simulate exchange processing latency
        5. Calculate costs
        6. Record fill

        Args:
            order: An Order object from execution.models or trading.broker

        Returns:
            FillRecord with fill details
        """
        from quant_platform.trading.broker import Order as BrokerOrder

        # Extract order fields (support both Order types)
        if isinstance(order, BrokerOrder):
            order_id = order.order_id
            symbol = order.code
            side_str = order.side.value
            qty = order.quantity
            price = order.price or 0
            is_limit = (order.order_type.value if hasattr(order.order_type, 'value')
                        else str(order.order_type)) == "limit"
        else:
            order_id = getattr(order, "order_id", uuid.uuid4().hex[:12])
            symbol = getattr(order, "ticker", getattr(order, "code", ""))
            side_str = getattr(order, "side", "buy")
            if hasattr(side_str, 'value'):
                side_str = side_str.value
            qty = getattr(order, "quantity", 0)
            price = getattr(order, "limit_price", 0) or 0
            is_limit = hasattr(order, "order_type") and str(order.order_type) in ("limit", "LIMIT")

        latency_ms = self._wait_latency(symbol)

        # Validate lot size
        lot_size = 100
        if self._asset_universe:
            inst = self._asset_universe.get(symbol)
            if inst:
                lot_size = inst.lot_size
        if qty % lot_size != 0:
            record = FillRecord(
                order_id=order_id, symbol=symbol, side=side_str,
                requested_qty=qty, filled_qty=0, fill_price=0,
                latency_ms=latency_ms,
            )
            self._record_fill(record)
            self._metrics.total_orders += 1
            return record

        # Convert to BookOrder and match
        book = self._get_book(symbol, tick_size=self._get_tick_size(symbol))
        book_side = BookSide.BUY if side_str == "buy" else BookSide.SELL
        book_type = BookOrderType.LIMIT if is_limit else BookOrderType.MARKET

        book_order = BookOrder(
            order_id=order_id,
            symbol=symbol,
            side=book_side,
            order_type=book_type,
            price=price,
            quantity=qty,
            source="paper_broker",
        )

        # Decide the partial fill before the order enters the matching engine.
        # Rewriting filled_quantity after matching left the resting remainder
        # larger than the quantity the book had actually matched away.
        self._apply_partial_fill(book, book_order)

        trades = book.add_order(book_order)

        # Calculate results
        total_filled = sum(t.quantity for t in trades)
        if total_filled > 0:
            avg_price = sum(t.price * t.quantity for t in trades) / total_filled
        else:
            avg_price = price

        # Keep the comparison book in step with this broker's own fills so a
        # later split/dividend can be applied to the book the TCA shadow
        # represents.
        if total_filled > 0:
            self._update_comparison_book(
                symbol, side_str, total_filled, avg_price,
            )

        # Exchange processing latency
        self._wait_latency(symbol)

        # Update metrics
        self._metrics.total_orders += 1
        self._metrics.total_trades += len(trades)
        if len(self._latency_samples) > 0:
            self._metrics.avg_latency_ms = float(np.mean(self._latency_samples[-100:]))
            self._metrics.max_latency_ms = max(self._metrics.max_latency_ms, latency_ms)
        if qty > 0:
            self._metrics.avg_fill_pct = (
                self._metrics.avg_fill_pct * (self._metrics.total_orders - 1)
                + total_filled / qty * 100
            ) / self._metrics.total_orders

        record = FillRecord(
            order_id=order_id,
            symbol=symbol,
            side=side_str,
            requested_qty=qty,
            filled_qty=total_filled,
            fill_price=round(avg_price, 4) if total_filled > 0 else 0.0,
            latency_ms=round(latency_ms, 2),
            partial=total_filled < qty and total_filled > 0,
            timestamp_ns=time.time_ns(),
        )
        self._record_fill(record)
        return record

    # ── Cancel Order ──

    def cancel_order(self, order_id: str) -> FillRecord:
        """Attempt to cancel an order. May fail stochastically."""
        self._metrics.cancel_attempts += 1

        book_order = None
        book = None
        for b in self._books.values():
            bo = b.get_order(order_id)
            if bo:
                book_order = bo
                book = b
                break

        if book_order is None:
            return FillRecord(
                order_id=order_id, symbol="", side="",
                requested_qty=0, filled_qty=0, fill_price=0,
                latency_ms=0, cancel_attempted=True, cancel_succeeded=False,
            )

        # Check if cancel fails
        if self._should_cancel_fail(book_order):
            self._metrics.cancel_failures += 1
            record = FillRecord(
                order_id=order_id, symbol=book_order.symbol,
                side=book_order.side.value,
                requested_qty=book_order.quantity,
                filled_qty=book_order.filled_quantity,
                fill_price=book_order.price,
                latency_ms=0,
                cancel_attempted=True,
                cancel_succeeded=False,
                timestamp_ns=time.time_ns(),
            )
            self._record_fill(record)
            return record

        # Perform cancel
        book.cancel_order(order_id)

        record = FillRecord(
            order_id=order_id, symbol=book_order.symbol,
            side=book_order.side.value,
            requested_qty=book_order.quantity,
            filled_qty=book_order.filled_quantity,
            fill_price=book_order.price,
            latency_ms=0,
            cancel_attempted=True,
            cancel_succeeded=True,
            timestamp_ns=time.time_ns(),
        )
        self._record_fill(record)
        return record

    # ── Comparison book / corporate actions ──

    def _update_comparison_book(
        self, symbol: str, side: str, quantity: int, price: float,
    ) -> None:
        """Record a fill in the shadow position ledger (best effort)."""
        asset = normalize_symbol(symbol)
        position = self._positions.setdefault(
            asset, {"quantity": 0, "avg_cost": 0.0, "current_price": 0.0},
        )
        if side == "buy":
            previous_cost = position["avg_cost"] * position["quantity"]
            position["quantity"] += int(quantity)
            position["avg_cost"] = (
                (previous_cost + price * quantity) / position["quantity"]
                if position["quantity"] > 0 else 0.0
            )
            self._cash -= price * quantity
        else:
            # The TCA shadow mirrors the primary book's sell sizes; a sell
            # larger than what this shadow actually holds must not credit
            # cash for shares it never bought (that minted fictitious cash
            # and skewed the shadow's dividend entitlement).  Credit only
            # the held quantity; the excess fill is a shadow artifact.
            held = max(0, int(position["quantity"]))
            sold = min(int(quantity), held)
            position["quantity"] = held - sold
            self._cash += price * sold
        position["current_price"] = float(price)

    def get_positions(self) -> dict[str, dict]:
        """The comparison book: ``{canonical_symbol: {quantity, avg_cost, ...}}``."""
        return {
            asset: dict(position)
            for asset, position in self._positions.items()
        }

    def get_cash(self) -> float:
        return float(self._cash)

    def set_corporate_actions(self, actions) -> int:
        """Replace the comparison book's pending corporate-action feed.

        ``actions`` follows the provider contract
        ``(ex_date, asset, kind, cash_per_share, split_ratio)``.  Nothing is
        applied until :meth:`apply_corporate_actions` (or a dated
        :meth:`new_trading_day`) runs.  Returns the number of feed rows.
        """
        frame = normalize_corporate_actions(actions)
        self._corporate_actions = frame
        return len(frame)

    def new_trading_day(self, date: str | None = None) -> list[dict]:
        """Advance the comparison book's exchange day.

        ``date`` applies every action with ``ex_date <= date`` exactly once;
        omitting it is a no-op (the shadow has no T+1 lock bookkeeping here).
        """
        if date is None:
            return []
        return self.apply_corporate_actions(ex_date=date)

    @staticmethod
    def _naive_action_date(value) -> pd.Timestamp:
        """Pin a feed/advance date to a naive Asia/Shanghai exchange day.

        Providers may stamp ex-dates with a UTC or exchange-local timezone;
        the broker compares them with ``YYYY-MM-DD`` strings.  Converting to
        the exchange zone and dropping the tz keeps the comparison valid
        (and prevents a naive/aware ``TypeError``).
        """
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
        return timestamp.normalize()

    def apply_corporate_actions(
        self, actions=None, ex_date: str | None = None,
    ) -> list[dict]:
        """Apply split/cash actions to the comparison book exactly once.

        A split scales quantity and divides the average cost and cached mark
        (market value unchanged); a cash dividend credits
        ``quantity * cash_per_share * (1 - tax)`` to cash and lowers the
        cached mark so the shadow NAV stays continuous.  Cash components run
        before splits (a 送转+派息 dividend is declared per pre-split share).
        """
        frame = (
            self._corporate_actions if actions is None
            else normalize_corporate_actions(actions)
        )
        if frame is None or frame.empty:
            return []
        cutoff = (
            self._naive_action_date(ex_date) if ex_date is not None else None
        )
        rows = frame.copy()
        rows["_order"] = rows["kind"].map(
            lambda kind: 0 if kind == "cash" else 1
        )
        rows = rows.sort_values(["ex_date", "asset", "_order"], kind="stable")
        applied: list[dict] = []
        for (group_date, group_asset), group in rows.groupby(
            ["ex_date", "asset"], sort=False,
        ):
            action_date = self._naive_action_date(group_date)
            if cutoff is not None and action_date > cutoff:
                continue
            asset = normalize_symbol(group_asset)
            marker = (asset, action_date.date().isoformat())
            if marker in self._applied_corporate_actions:
                continue
            event = self._apply_action_group(asset, marker, group)
            self._applied_corporate_actions.add(marker)
            if event is not None:
                self._corporate_action_events.append(event)
                applied.append(event)
        return applied

    def _apply_action_group(
        self, asset: str, marker: tuple[str, str], group: pd.DataFrame,
    ) -> dict | None:
        position = self._positions.get(asset)
        quantity_before = int(position["quantity"]) if position else 0
        quantity_after = quantity_before
        cash_total = 0.0
        tax_total = 0.0
        cash_per_share_total = 0.0
        split_ratio_total = 1.0
        components: list[dict] = []
        timestamp = time.time_ns()

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
                self._cash += net
                if position and position["current_price"] > 0:
                    position["current_price"] = max(
                        position["current_price"] - cash_per_share, 0.0,
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

        for _, row in group.iterrows():
            if str(row["kind"]) != "split":
                continue
            ratio = float(row["split_ratio"])
            if not (ratio > 0) or ratio == 1.0:
                continue
            split_ratio_total *= ratio
            components.append({"kind": "split", "split_ratio": ratio})
            if position is not None:
                position["quantity"] = int(
                    round(int(position["quantity"]) * ratio)
                )
                if position["avg_cost"] > 0:
                    position["avg_cost"] = position["avg_cost"] / ratio
                if position["current_price"] > 0:
                    position["current_price"] = (
                        position["current_price"] / ratio
                    )
                quantity_after = position["quantity"]

        if not components:
            return None
        # The resting LOB ladder was seeded at the pre-action price; a sell at
        # the new level must not trade against a stale bid.  The next cycle
        # re-seeds it (LiveRunner seeds liquidity before every paper order).
        for key in list(self._books):
            if normalize_symbol(key) == asset:
                del self._books[key]

        return {
            "date": marker[1],
            "asset": asset,
            "kind": "+".join(sorted({c["kind"] for c in components})),
            "quantity_before": quantity_before,
            "quantity_after": quantity_after,
            "cash_delta": cash_total,
            "tax": tax_total,
            "cash_per_share": cash_per_share_total,
            "split_ratio": split_ratio_total,
            "components": components,
            "timestamp": timestamp,
        }

    def get_applied_corporate_actions(self) -> list[tuple[str, str]]:
        return sorted(self._applied_corporate_actions)

    def get_corporate_action_events(self) -> list[dict]:
        return [dict(event) for event in self._corporate_action_events]

    def get_cash_ledger(self) -> list[dict]:
        return [dict(entry) for entry in self._cash_ledger]

    # ── Metrics ──

    def get_metrics(self) -> PaperBrokerMetrics:
        return self._metrics

    def get_fill_records(self, limit: int = 100) -> list[FillRecord]:
        return self._fill_records[-limit:]

    def reset_metrics(self):
        self._metrics = PaperBrokerMetrics()
        self._latency_samples.clear()
        self._fill_records.clear()

    # ── Latency Arbitrage Detection ──

    def detect_latency_arbitrage(self, symbol: str, price: float,
                                 reference_price: float) -> dict:
        """Check if a trade exploits latency arbitrage.

        In live trading, prices across exchanges are synchronized.
        If a backtest shows a trade at a price that's significantly
        different from the reference (NBBO), it may be unrealistic.

        Returns:
            dict with 'is_arbitrage', 'deviation_bps', 'warning'
        """
        if reference_price <= 0:
            return {"is_arbitrage": False, "deviation_bps": 0, "warning": ""}

        deviation_bps = abs(price - reference_price) / reference_price * 10000
        is_arb = deviation_bps > 50  # 50 bps threshold

        warning = ""
        if is_arb:
            lowest = EXCHANGE_LATENCY_MS.get("DEFAULT", 5.0)
            max_deviation = lowest / 1000 * 0.001 * 10000  # ~0.05 bps
            warning = (
                f"Latency arbitrage detected: {deviation_bps:.1f} bps deviation. "
                f"Max realistic deviation at {lowest}ms latency: {max_deviation:.1f} bps. "
                f"Backtest results may be unrealistic."
            )
            logger.warning(warning)

        return {
            "is_arbitrage": is_arb,
            "deviation_bps": round(deviation_bps, 1),
            "warning": warning,
        }
