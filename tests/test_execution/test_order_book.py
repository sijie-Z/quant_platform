"""Tests for the real order book."""

import pytest

from quant_platform.execution.order_book import (
    BookOrder,
    OrderBook,
    OrderBookManager,
    OrderType,
    PriceLevel,
    Side,
    benchmark_order_book,
)

# ── PriceLevel Tests ──


class TestPriceLevel:
    def test_add_order(self):
        level = PriceLevel(price=100.0)
        order = BookOrder("o1", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 500)
        level.add_order(order)
        assert level.total_quantity == 500
        assert level.order_count == 1

    def test_fifo_ordering(self):
        level = PriceLevel(price=100.0)
        level.add_order(BookOrder("o1", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 100))
        level.add_order(BookOrder("o2", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 200))
        level.add_order(BookOrder("o3", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 300))
        assert level.total_quantity == 600

        # Reduce should fill from front (FIFO)
        fills = level.reduce(150)
        assert len(fills) == 2  # o1 fully filled, o2 partially
        assert fills[0][1] == 100  # o1: 100
        assert fills[1][1] == 50   # o2: 50 of 200

    def test_remove_order(self):
        level = PriceLevel(price=100.0)
        level.add_order(BookOrder("o1", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 100))
        level.add_order(BookOrder("o2", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 200))
        removed = level.remove_order("o1")
        assert removed is not None
        assert removed.order_id == "o1"
        assert level.total_quantity == 200

    def test_is_empty(self):
        level = PriceLevel(price=100.0)
        assert level.is_empty
        level.add_order(BookOrder("o1", "SYM", Side.BUY, OrderType.LIMIT, 100.0, 100))
        assert not level.is_empty


# ── OrderBook Tests ──


class TestOrderBook:
    @pytest.fixture
    def book(self):
        return OrderBook("TEST", tick_size=0.01)

    def test_add_limit_buy(self, book):
        order = BookOrder("o1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 500)
        trades = book.add_order(order)
        assert len(trades) == 0  # No asks to match against
        assert book.best_bid == 100.0
        assert book.best_ask is None

    def test_add_limit_sell(self, book):
        order = BookOrder("o1", "TEST", Side.SELL, OrderType.LIMIT, 105.0, 500)
        trades = book.add_order(order)
        assert len(trades) == 0
        assert book.best_ask == 105.0
        assert book.best_bid is None

    def test_basic_matching(self, book):
        # Add sell order
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 105.0, 500))
        # Add buy order that crosses the spread
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 105.0, 300))
        assert len(trades) == 1
        assert trades[0].price == 105.0
        assert trades[0].quantity == 300
        assert trades[0].aggressor_side == Side.BUY

    def test_full_fill(self, book):
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 100))
        assert len(trades) == 1
        assert trades[0].quantity == 100

    def test_partial_fill(self, book):
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        # Buy 200 but only 100 available
        buy = BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 200)
        trades = book.add_order(buy)
        assert len(trades) == 1
        assert trades[0].quantity == 100
        assert buy.filled_quantity == 100
        assert buy.remaining_quantity == 100

    def test_price_time_priority(self, book):
        # Two sell orders at same price
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100,
                                 timestamp_ns=1000))
        book.add_order(BookOrder("s2", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 200,
                                 timestamp_ns=2000))
        # Buy should fill s1 first (earlier timestamp)
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 150))
        assert len(trades) == 2
        assert trades[0].maker_order_id == "s1"  # First in FIFO
        assert trades[0].quantity == 100
        assert trades[1].maker_order_id == "s2"
        assert trades[1].quantity == 50

    def test_price_priority(self, book):
        # Two sell orders at different prices
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 101.0, 100))
        book.add_order(BookOrder("s2", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        # Buy should fill s2 first (lower price)
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 101.0, 100))
        assert len(trades) == 1
        assert trades[0].maker_order_id == "s2"
        assert trades[0].price == 100.0

    def test_market_order(self, book):
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        book.add_order(BookOrder("s2", "TEST", Side.SELL, OrderType.LIMIT, 101.0, 100))
        # Market buy walks the book
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.MARKET, 0, 150))
        assert len(trades) == 2
        assert trades[0].price == 100.0
        assert trades[1].price == 101.0

    def test_ioc_order(self, book):
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        # IOC buy 200: fill 100, cancel 100
        order = BookOrder("b1", "TEST", Side.BUY, OrderType.IOC, 100.0, 200)
        trades = book.add_order(order)
        assert len(trades) == 1
        assert trades[0].quantity == 100
        assert order.status == "cancelled"

    def test_fok_order_full_fill(self, book):
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 200))
        order = BookOrder("b1", "TEST", Side.BUY, OrderType.FOK, 100.0, 200)
        trades = book.add_order(order)
        assert len(trades) == 1
        assert trades[0].quantity == 200

    def test_fok_order_cancel(self, book):
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        # FOK needs 200 but only 100 available
        order = BookOrder("b1", "TEST", Side.BUY, OrderType.FOK, 100.0, 200)
        trades = book.add_order(order)
        assert len(trades) == 0
        assert order.status == "cancelled"

    def test_cancel_order(self, book):
        book.add_order(BookOrder("o1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 500))
        assert book.best_bid == 100.0
        cancelled = book.cancel_order("o1")
        assert cancelled is not None
        assert cancelled.order_id == "o1"
        assert book.best_bid is None

    def test_cancel_nonexistent(self, book):
        result = book.cancel_order("nonexistent")
        assert result is None

    def test_depth_snapshot(self, book):
        for i in range(5):
            book.add_order(BookOrder(f"b{i}", "TEST", Side.BUY, OrderType.LIMIT,
                                     100.0 - i, 100 * (i + 1)))
            book.add_order(BookOrder(f"a{i}", "TEST", Side.SELL, OrderType.LIMIT,
                                     101.0 + i, 100 * (i + 1)))

        snapshot = book.get_depth_snapshot(levels=3)
        assert len(snapshot["bids"]) == 3
        assert len(snapshot["asks"]) == 3
        assert snapshot["bids"][0]["price"] == 100.0
        assert snapshot["asks"][0]["price"] == 101.0

    def test_microstructure_metrics(self, book):
        book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 500))
        book.add_order(BookOrder("a1", "TEST", Side.SELL, OrderType.LIMIT, 101.0, 300))
        metrics = book.get_microstructure_metrics()
        assert metrics["spread"] == 1.0
        assert metrics["bid_depth_5"] == 500
        assert metrics["ask_depth_5"] == 300

    def test_multiple_price_levels(self, book):
        # Build a book with multiple levels
        for i in range(10):
            book.add_order(BookOrder(f"b{i}", "TEST", Side.BUY, OrderType.LIMIT,
                                     99.0 - i * 0.5, 100))
            book.add_order(BookOrder(f"a{i}", "TEST", Side.SELL, OrderType.LIMIT,
                                     101.0 + i * 0.5, 100))

        assert book.best_bid == 99.0
        assert book.best_ask == 101.0
        assert len(book._bid_prices) == 10
        assert len(book._ask_prices) == 10

    def test_walking_multiple_levels(self, book):
        # Sell side with multiple levels
        book.add_order(BookOrder("s1", "TEST", Side.SELL, OrderType.LIMIT, 100.0, 100))
        book.add_order(BookOrder("s2", "TEST", Side.SELL, OrderType.LIMIT, 101.0, 100))
        book.add_order(BookOrder("s3", "TEST", Side.SELL, OrderType.LIMIT, 102.0, 100))

        # Large market buy walks through all levels
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.MARKET, 0, 250))
        assert len(trades) == 3
        assert sum(t.quantity for t in trades) == 250

    def test_empty_book_no_trades(self, book):
        trades = book.add_order(BookOrder("b1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 100))
        assert len(trades) == 0

    def test_clear(self, book):
        book.add_order(BookOrder("o1", "TEST", Side.BUY, OrderType.LIMIT, 100.0, 100))
        book.clear()
        assert book.best_bid is None
        assert book.best_ask is None
        assert len(book._orders) == 0


# ── OrderBookManager Tests ──


class TestOrderBookManager:
    def test_get_or_create(self):
        manager = OrderBookManager()
        book1 = manager.get_or_create("SYM1")
        book2 = manager.get_or_create("SYM1")
        assert book1 is book2

    def test_multi_symbol(self):
        manager = OrderBookManager()
        manager.add_order(BookOrder("o1", "SYM1", Side.BUY, OrderType.LIMIT, 100.0, 100))
        manager.add_order(BookOrder("o2", "SYM2", Side.BUY, OrderType.LIMIT, 200.0, 100))
        assert len(manager.symbols) == 2

    def test_get_all_depths(self):
        manager = OrderBookManager()
        manager.add_order(BookOrder("o1", "SYM1", Side.BUY, OrderType.LIMIT, 100.0, 100))
        depths = manager.get_all_depths()
        assert "SYM1" in depths


# ── Benchmark ──


class TestBenchmark:
    def test_benchmark_runs(self):
        result = benchmark_order_book(n_orders=1000, n_symbols=1)
        assert result["n_orders"] == 1000
        assert result["total_trades"] >= 0
        assert result["throughput_orders_per_sec"] > 0
