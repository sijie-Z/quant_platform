"""Tests for Level 2 market data provider."""

import time

import pytest
from quant_platform.data.providers.level2_provider import (
    Level2DataProvider,
    OrderBookAnalytics,
    OrderBookLevel,
    OrderBookSnapshot,
    TickData,
)


class TestOrderBookLevel:
    def test_creation(self):
        lv = OrderBookLevel(price=100.5, volume=5000, order_count=10)
        assert lv.price == 100.5
        assert lv.volume == 5000

    def test_to_dict(self):
        lv = OrderBookLevel(price=100.5, volume=5000, order_count=10)
        d = lv.to_dict()
        assert d["price"] == 100.5
        assert d["volume"] == 5000
        assert d["order_count"] == 10


class TestOrderBookSnapshot:
    @pytest.fixture
    def sample_book(self):
        bids = [
            OrderBookLevel(price=100.0, volume=5000),
            OrderBookLevel(price=99.9, volume=3000),
            OrderBookLevel(price=99.8, volume=2000),
        ]
        asks = [
            OrderBookLevel(price=100.1, volume=4000),
            OrderBookLevel(price=100.2, volume=3500),
            OrderBookLevel(price=100.3, volume=2500),
        ]
        return OrderBookSnapshot(code="600519", timestamp="2024-01-01T10:00:00", bids=bids, asks=asks)

    def test_best_bid(self, sample_book):
        assert sample_book.best_bid == 100.0

    def test_best_ask(self, sample_book):
        assert sample_book.best_ask == 100.1

    def test_mid_price(self, sample_book):
        assert sample_book.mid_price == pytest.approx(100.05)

    def test_spread(self, sample_book):
        assert sample_book.spread == pytest.approx(0.1)

    def test_spread_bps(self, sample_book):
        assert sample_book.spread_bps == pytest.approx(9.995, abs=0.01)

    def test_bid_depth(self, sample_book):
        assert sample_book.bid_depth == 10000

    def test_ask_depth(self, sample_book):
        assert sample_book.ask_depth == 10000

    def test_depth_imbalance(self, sample_book):
        assert sample_book.depth_imbalance == pytest.approx(0.0)

    def test_imbalance_positive(self):
        bids = [OrderBookLevel(price=100.0, volume=8000)]
        asks = [OrderBookLevel(price=100.1, volume=2000)]
        book = OrderBookSnapshot(code="600519", timestamp="", bids=bids, asks=asks)
        assert book.depth_imbalance > 0

    def test_imbalance_negative(self):
        bids = [OrderBookLevel(price=100.0, volume=2000)]
        asks = [OrderBookLevel(price=100.1, volume=8000)]
        book = OrderBookSnapshot(code="600519", timestamp="", bids=bids, asks=asks)
        assert book.depth_imbalance < 0

    def test_empty_book(self):
        book = OrderBookSnapshot(code="600519", timestamp="")
        assert book.best_bid == 0.0
        assert book.best_ask == 0.0
        assert book.mid_price == 0.0

    def test_to_dict(self, sample_book):
        d = sample_book.to_dict()
        assert d["code"] == "600519"
        assert "best_bid" in d
        assert "bids" in d
        assert len(d["bids"]) == 3


class TestTickData:
    def test_creation(self):
        tick = TickData(
            code="600519", timestamp="10:00:00",
            price=1800.0, volume=100, amount=180000,
            direction="B",
        )
        assert tick.code == "600519"
        assert tick.direction == "B"

    def test_to_dict(self):
        tick = TickData(code="600519", timestamp="10:00:00", price=1800.0, volume=100, amount=180000)
        d = tick.to_dict()
        assert d["price"] == 1800.0


class TestLevel2DataProvider:
    def test_creation(self):
        l2 = Level2DataProvider(codes=["600519"])
        assert l2._codes == ["600519"]

    def test_start_stop(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        assert l2.is_running
        time.sleep(0.3)
        l2.stop()
        assert not l2.is_running

    def test_generates_order_book(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        book = l2.get_order_book("600519")
        assert book is not None
        assert book.code == "600519"
        assert len(book.bids) == 10
        assert len(book.asks) == 10

    def test_order_book_sorted(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        book = l2.get_order_book("600519")
        # Bids descending
        for i in range(len(book.bids) - 1):
            assert book.bids[i].price >= book.bids[i + 1].price
        # Asks ascending
        for i in range(len(book.asks) - 1):
            assert book.asks[i].price <= book.asks[i + 1].price

    def test_generates_ticks(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        ticks = l2.get_ticks("600519")
        assert len(ticks) > 0
        assert ticks[0].code == "600519"
        assert ticks[0].price > 0

    def test_tick_direction(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        ticks = l2.get_ticks("600519")
        directions = {t.direction for t in ticks}
        assert "B" in directions or "S" in directions

    def test_compute_vwap(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        vwap = l2.compute_vwap("600519")
        assert vwap > 0

    def test_compute_trade_flow(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        flow = l2.compute_trade_flow("600519")
        assert "buy_volume" in flow
        assert "sell_volume" in flow
        assert "net_flow" in flow
        assert "buy_pct" in flow

    def test_stats(self):
        l2 = Level2DataProvider(codes=["600519", "000001"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        stats = l2.stats
        assert stats["codes"] == 2
        assert stats["books_cached"] == 2
        assert stats["total_ticks"] > 0

    def test_get_all_books(self):
        l2 = Level2DataProvider(codes=["600519", "000001"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        books = l2.get_all_books()
        assert len(books) == 2

    def test_vwap_empty(self):
        l2 = Level2DataProvider(codes=["600519"])
        assert l2.compute_vwap("600519") == 0.0

    def test_tick_filter_by_time(self):
        l2 = Level2DataProvider(codes=["600519"])
        l2.start()
        time.sleep(0.3)
        l2.stop()

        ticks = l2.get_ticks("600519", start="00:00:00", end="23:59:59")
        assert isinstance(ticks, list)


class TestOrderBookAnalytics:
    @pytest.fixture
    def sample_book(self):
        bids = [
            OrderBookLevel(price=100.0, volume=5000),
            OrderBookLevel(price=99.9, volume=3000),
        ]
        asks = [
            OrderBookLevel(price=100.1, volume=4000),
            OrderBookLevel(price=100.2, volume=3500),
        ]
        return OrderBookSnapshot(code="600519", timestamp="", bids=bids, asks=asks)

    def test_effective_spread(self, sample_book):
        spread = OrderBookAnalytics.effective_spread(sample_book, 100.08)
        assert spread == pytest.approx(0.06)

    def test_weighted_mid_price(self, sample_book):
        wmid = OrderBookAnalytics.weighted_mid_price(sample_book)
        assert 100.0 < wmid < 100.1

    def test_book_pressure(self, sample_book):
        pressure = OrderBookAnalytics.book_pressure(sample_book)
        assert "bid_pressure" in pressure
        assert "ask_pressure" in pressure
        assert pressure["bid_pressure"] == 8000
        assert pressure["ask_pressure"] == 7500

    def test_book_slope(self, sample_book):
        slope = OrderBookAnalytics.book_slope(sample_book)
        assert "bid_slope" in slope
        assert "ask_slope" in slope

    def test_effective_spread_empty(self):
        book = OrderBookSnapshot(code="600519", timestamp="")
        assert OrderBookAnalytics.effective_spread(book, 100.0) == 0.0
