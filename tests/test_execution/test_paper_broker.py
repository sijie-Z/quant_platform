"""Tests for enhanced paper trading broker."""

import time

import pytest

from quant_platform.execution.paper_broker import (
    EXCHANGE_LATENCY_MS,
    LATENCY_PROFILES,
    FillRecord,
    LatencyConfig,
    LatencyModel,
    PaperBroker,
    PaperBrokerMetrics,
)
from quant_platform.execution.order_book import (
    BookOrder,
    BookOrderStatus,
    OrderBook,
    OrderType as BookOrderType,
    Side as BookSide,
)
from quant_platform.trading.broker import Order, OrderSide, OrderType


# ── LatencyConfig ──


class TestLatencyConfig:
    def test_defaults(self):
        lc = LatencyConfig()
        assert lc.base_ms == 3.0
        assert lc.jitter_ms == 1.0
        assert lc.processing_ms == 1.0

    def test_zero_latency(self):
        lc = LatencyConfig(base_ms=0, jitter_ms=0, processing_ms=0, exchange_multiplier=0)
        assert lc.sample() == 0.0

    def test_latency_model_zero_is_zero(self):
        lc = LatencyConfig(**LATENCY_PROFILES[LatencyModel.ZERO])
        assert lc.sample() == 0.0

    def test_no_negative_latency(self):
        lc = LatencyConfig(base_ms=1, jitter_ms=100, processing_ms=1)
        for _ in range(100):
            assert lc.sample() >= 0.0

    def test_exchange_specific(self):
        lc = LatencyConfig(base_ms=1, jitter_ms=0.5, processing_ms=1)
        sse = lc.sample("SSE")
        default = lc.sample("DEFAULT")
        assert sse >= 0.0
        assert default >= 0.0

    def test_latency_reasonable_range(self):
        lc = LatencyConfig(base_ms=5, jitter_ms=2, processing_ms=1)
        samples = [lc.sample() for _ in range(200)]
        avg = sum(samples) / len(samples)
        # Average should be near (5 + 1 + exchange_base ~5) ≈ 11
        assert 3.0 <= avg <= 20.0


# ── LatencyModel Profiles ──


class TestLatencyProfiles:
    def test_zero_profile(self):
        assert LatencyConfig(**LATENCY_PROFILES[LatencyModel.ZERO]).base_ms == 0

    def test_low_profile(self):
        lc = LatencyConfig(**LATENCY_PROFILES[LatencyModel.LOW])
        assert lc.base_ms == 3

    def test_realistic_profile(self):
        lc = LatencyConfig(**LATENCY_PROFILES[LatencyModel.REALISTIC])
        assert lc.base_ms == 100


# ── EXCHANGE_LATENCY_MS ──


class TestExchangeLatency:
    def test_all_exchanges_present(self):
        for ex in ["SSE", "SZSE", "CFFEX", "SHFE", "DCE", "CZCE", "DEFAULT"]:
            assert ex in EXCHANGE_LATENCY_MS

    def test_sse_latency_reasonable(self):
        assert EXCHANGE_LATENCY_MS["SSE"] < 5.0


# ── PaperBrokerMetrics ──


class TestPaperBrokerMetrics:
    def test_defaults(self):
        m = PaperBrokerMetrics()
        assert m.total_orders == 0
        assert m.partial_fills == 0
        assert m.cancel_failures == 0
        assert m.avg_latency_ms == 0.0


# ── PaperBroker ──


class TestPaperBrokerInit:
    def test_default_init(self):
        pb = PaperBroker()
        assert pb._cash == 10_000_000

    def test_init_with_cash(self):
        pb = PaperBroker(initial_cash=5_000_000)
        assert pb._cash == 5_000_000

    def test_init_with_latency_model(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        assert pb._latency.base_ms == 0

    def test_init_with_custom_latency(self):
        lc = LatencyConfig(base_ms=7, jitter_ms=3, processing_ms=2)
        pb = PaperBroker(latency=lc)
        assert pb._latency.base_ms == 7

    def test_init_with_partial_fill_rate(self):
        pb = PaperBroker(partial_fill_rate=0.25)
        assert pb._partial_fill_rate == 0.25

    def test_init_with_cancel_fail_rate(self):
        pb = PaperBroker(cancel_fail_rate=0.05)
        assert pb._cancel_fail_rate == 0.05


class TestPaperBrokerPlaceOrder:
    def test_place_market_buy(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=100, price=1800)
        record = pb.place_order(order)
        assert record.order_id == order.order_id
        assert record.symbol == "600519"
        assert record.side == "buy"
        assert record.requested_qty == 100

    def test_place_limit_sell(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.SELL, order_type=OrderType.LIMIT,
                      quantity=100, price=1800)
        record = pb.place_order(order)
        assert record.side == "sell"

    def test_place_order_with_latency(self):
        pb = PaperBroker(latency=LatencyConfig(base_ms=10, jitter_ms=0, processing_ms=5))
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=100, price=1800)
        record = pb.place_order(order)
        assert record.latency_ms > 0

    def test_place_order_invalid_lot(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=150, price=1800)  # Not multiple of 100
        record = pb.place_order(order)
        assert record.filled_qty == 0

    def test_place_order_metrics_updated(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=100, price=1800)
        pb.place_order(order)
        metrics = pb.get_metrics()
        assert metrics.total_orders == 1


class TestPaperBrokerCancel:
    def test_cancel_nonexistent(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        record = pb.cancel_order("nonexistent")
        assert record.cancel_attempted
        assert not record.cancel_succeeded

    def test_cancel_after_order(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                      quantity=100, price=1800)
        pb.place_order(order)

        # If not fully filled, cancel should succeed (most of the time)
        record = pb.cancel_order(order.order_id)
        assert record.cancel_attempted
        # May succeed or fail stochastically

    def test_cancel_metrics(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                      quantity=100, price=1800)
        pb.place_order(order)
        pb.cancel_order(order.order_id)
        metrics = pb.get_metrics()
        assert metrics.cancel_attempts >= 1


class TestPaperBrokerMetrics:
    def test_get_metrics_after_orders(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        for i in range(5):
            order = Order(code=f"60051{i}", side=OrderSide.BUY,
                          order_type=OrderType.MARKET, quantity=100, price=1800)
            pb.place_order(order)
        metrics = pb.get_metrics()
        assert metrics.total_orders == 5

    def test_reset_metrics(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=100, price=1800)
        pb.place_order(order)
        pb.reset_metrics()
        metrics = pb.get_metrics()
        assert metrics.total_orders == 0

    def test_fill_records_persist(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        order = Order(code="600519", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=100, price=1800)
        pb.place_order(order)
        records = pb.get_fill_records()
        assert len(records) >= 1

    def test_fill_records_limit(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        for i in range(10):
            order = Order(code=f"60051{i}", side=OrderSide.BUY,
                          order_type=OrderType.MARKET, quantity=100, price=1800)
            pb.place_order(order)
        records = pb.get_fill_records(limit=5)
        assert len(records) == 5


# ── L2 Replay ──


class TestL2Replay:
    def test_load_snapshots(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        snapshots = [
            {
                "timestamp": 1000,
                "bids": [[1800.0, 500], [1799.5, 300]],
                "asks": [[1800.5, 400], [1801.0, 200]],
            },
            {
                "timestamp": 2000,
                "bids": [[1801.0, 600], [1800.5, 300]],
                "asks": [[1801.5, 500], [1802.0, 300]],
            },
        ]
        pb.load_l2_snapshots("600519", snapshots)
        assert "600519" in pb._l2_snapshots
        assert len(pb._l2_snapshots["600519"]) == 2

    def test_replay_l2_returns_correct_snapshot(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        snapshots = [
            {"timestamp": 1000, "bids": [[100.0, 100]], "asks": [[101.0, 100]]},
            {"timestamp": 2000, "bids": [[101.0, 100]], "asks": [[102.0, 100]]},
        ]
        pb.load_l2_snapshots("600519", snapshots)
        snap = pb.replay_l2("600519", 1500)
        assert snap is not None
        assert snap["timestamp"] == 1000

    def test_replay_l2_empty(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        snap = pb.replay_l2("600519", 1000)
        assert snap is None

    def test_replay_l2_advances_index(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        snapshots = [
            {"timestamp": 1000, "bids": [[100.0, 100]], "asks": [[101.0, 100]]},
            {"timestamp": 2000, "bids": [[101.0, 100]], "asks": [[102.0, 100]]},
        ]
        pb.load_l2_snapshots("600519", snapshots)
        pb.replay_l2("600519", 2500)
        # Index should be at 2
        assert pb._replay_index.get("600519") == 2

    def test_seed_book_from_snapshot(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        snap = {
            "timestamp": 1000,
            "bids": [[1800.0, 500], [1799.5, 300]],
            "asks": [[1800.5, 400], [1801.0, 200]],
        }
        pb.seed_book_from_snapshot("600519", snap)
        book = pb._books.get("600519")
        assert book is not None
        depth = book.get_depth_snapshot(levels=5)
        assert len(depth["bids"]) > 0
        assert len(depth["asks"]) > 0


# ── Latency Arbitrage Detection ──


class TestLatencyArbitrage:
    def test_no_arbitrage_normal(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        result = pb.detect_latency_arbitrage("600519", 1800.0, 1800.01)
        assert not result["is_arbitrage"]
        assert result["deviation_bps"] < 50

    def test_detect_arbitrage_large_deviation(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        result = pb.detect_latency_arbitrage("600519", 1810.0, 1800.0)
        assert result["is_arbitrage"]
        assert result["deviation_bps"] > 50

    def test_zero_reference(self):
        pb = PaperBroker(latency=LatencyModel.ZERO)
        result = pb.detect_latency_arbitrage("600519", 1800.0, 0)
        assert not result["is_arbitrage"]


# ── OrderBook.get_order ──


class TestOrderBookGetOrder:
    def test_get_order_returns_order(self):
        book = OrderBook("600519")
        bo = BookOrder("o1", "600519", BookSide.BUY, BookOrderType.LIMIT, 1800.0, 100)
        book.add_order(bo)
        found = book.get_order("o1")
        assert found is not None
        assert found.order_id == "o1"

    def test_get_order_nonexistent(self):
        book = OrderBook("600519")
        assert book.get_order("nonexistent") is None

    def test_get_order_after_cancel(self):
        book = OrderBook("600519")
        bo = BookOrder("o1", "600519", BookSide.BUY, BookOrderType.LIMIT, 1800.0, 100)
        book.add_order(bo)
        book.cancel_order("o1")
        assert book.get_order("o1") is None


# ── FillRecord ──


class TestFillRecord:
    def test_defaults(self):
        fr = FillRecord("o1", "600519", "buy", 100, 100, 1800.0, 5.0)
        assert fr.order_id == "o1"
        assert not fr.partial
        assert not fr.cancel_attempted
        assert fr.latency_ms == 5.0

    def test_partial_fill(self):
        fr = FillRecord("o1", "600519", "buy", 200, 100, 1800.0, 5.0, partial=True)
        assert fr.partial
        assert fr.filled_qty < fr.requested_qty

    def test_cancel_fail(self):
        fr = FillRecord("o1", "600519", "buy", 100, 0, 0, 3.0,
                        cancel_attempted=True, cancel_succeeded=False)
        assert fr.cancel_attempted
        assert not fr.cancel_succeeded
