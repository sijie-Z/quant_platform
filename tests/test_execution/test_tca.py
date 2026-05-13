"""Tests for Transaction Cost Analysis (TCA)."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from quant_platform.execution.tca import (
    TCAEngine,
    TCAResult,
    TCASummary,
    TCABenchmark,
    VWAPCalculator,
)


@pytest.fixture
def sample_fills():
    """Generate sample fills for testing."""
    return [
        {"price": 100.5, "quantity": 5000, "timestamp": "2024-03-01 10:00:00"},
        {"price": 100.8, "quantity": 3000, "timestamp": "2024-03-01 10:15:00"},
        {"price": 101.0, "quantity": 2000, "timestamp": "2024-03-01 10:30:00"},
    ]


@pytest.fixture
def engine():
    """Create a TCA engine."""
    return TCAEngine()


class TestVWAPCalculator:
    """Test VWAP computation."""

    def test_basic_vwap(self):
        """VWAP should be volume-weighted average."""
        prices = pd.Series([100.0, 101.0, 99.0])
        volumes = pd.Series([1000, 2000, 3000])

        vwap = VWAPCalculator.from_bars(prices, volumes)
        expected = (100 * 1000 + 101 * 2000 + 99 * 3000) / 6000
        assert abs(vwap - expected) < 0.01

    def test_single_bar(self):
        """Single bar should return that bar's price."""
        prices = pd.Series([100.0])
        volumes = pd.Series([5000])

        vwap = VWAPCalculator.from_bars(prices, volumes)
        assert abs(vwap - 100.0) < 0.01

    def test_zero_volume_returns_mean(self):
        """Zero volume should fall back to mean price."""
        prices = pd.Series([100.0, 102.0])
        volumes = pd.Series([0, 0])

        vwap = VWAPCalculator.from_bars(prices, volumes)
        assert abs(vwap - 101.0) < 0.01

    def test_tick_vwap(self):
        """Tick VWAP should work the same way."""
        prices = pd.Series([100.0, 100.5, 101.0])
        volumes = pd.Series([100, 200, 300])

        vwap = VWAPCalculator.from_ticks(prices, volumes)
        expected = (100 * 100 + 100.5 * 200 + 101 * 300) / 600
        assert abs(vwap - expected) < 0.01


class TestTCAEngine:
    """Test TCA analysis."""

    def test_returns_tca_result(self, engine, sample_fills):
        """Should return a TCAResult."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        assert isinstance(result, TCAResult)
        assert result.order_id == "ORD001"

    def test_avg_execution_price(self, engine, sample_fills):
        """Average execution price should be quantity-weighted."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        expected_avg = (100.5 * 5000 + 100.8 * 3000 + 101.0 * 2000) / 10000
        assert abs(result.avg_exec_price - expected_avg) < 0.01

    def test_buy_positive_cost_unfavorable(self, engine):
        """For buys, higher execution price = positive cost (unfavorable)."""
        fills = [{"price": 105.0, "quantity": 1000, "timestamp": "2024-03-01 10:00:00"}]

        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=1000,
            fills=fills,
            decision_price=100.0,
            arrival_price=100.0,
            vwap_price=100.0,
            close_price=100.0,
        )

        # Bought at 105 vs decision at 100 → positive cost
        assert result.implementation_shortfall_bps > 0

    def test_sell_positive_cost_unfavorable(self, engine):
        """For sells, lower execution price = positive cost (unfavorable)."""
        fills = [{"price": 95.0, "quantity": 1000, "timestamp": "2024-03-01 10:00:00"}]

        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="sell",
            quantity=1000,
            fills=fills,
            decision_price=100.0,
            arrival_price=100.0,
            vwap_price=100.0,
            close_price=100.0,
        )

        # Sold at 95 vs decision at 100 → positive cost
        assert result.implementation_shortfall_bps > 0

    def test_implementation_shortfall_bps(self, engine, sample_fills):
        """IS should be (exec - decision) / decision * 10000 for buys."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        avg_exec = (100.5 * 5000 + 100.8 * 3000 + 101.0 * 2000) / 10000
        expected_is = (avg_exec - 100.0) / 100.0 * 10000
        assert abs(result.implementation_shortfall_bps - round(expected_is, 2)) < 0.1

    def test_arrival_cost_bps(self, engine, sample_fills):
        """Arrival cost should be (exec - arrival) / arrival * 10000 for buys."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        avg_exec = (100.5 * 5000 + 100.8 * 3000 + 101.0 * 2000) / 10000
        expected_arrival = (avg_exec - 100.2) / 100.2 * 10000
        assert abs(result.arrival_cost_bps - round(expected_arrival, 2)) < 0.1

    def test_delay_cost_bps(self, engine, sample_fills):
        """Delay cost should be (arrival - decision) / decision * 10000 for buys."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        expected_delay = (100.2 - 100.0) / 100.0 * 10000
        assert abs(result.delay_cost_bps - round(expected_delay, 2)) < 0.1

    def test_num_fills(self, engine, sample_fills):
        """Should count number of fills."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        assert result.num_fills == 3

    def test_participation_rate(self, engine, sample_fills):
        """Participation rate = total quantity / market volume."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=sample_fills,
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
            market_volume=1000000,
        )

        assert abs(result.participation_rate - 0.01) < 0.001

    def test_empty_fills(self, engine):
        """Empty fills should return zero costs."""
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=[],
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
        )

        assert result.avg_exec_price == 0.0
        assert result.implementation_shortfall_bps == 0.0

    def test_fill_duration(self, engine):
        """Should compute fill duration from timestamps."""
        fills = [
            {"price": 100.0, "quantity": 5000, "timestamp": "2024-03-01 10:00:00"},
            {"price": 100.5, "quantity": 5000, "timestamp": "2024-03-01 10:30:00"},
        ]

        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=fills,
            decision_price=100.0,
            arrival_price=100.0,
            vwap_price=100.0,
            close_price=100.0,
            first_fill_time=datetime(2024, 3, 1, 10, 0, 0),
            last_fill_time=datetime(2024, 3, 1, 10, 30, 0),
        )

        assert result.fill_duration_seconds == 1800.0


class TestTCASummary:
    """Test TCA summarization."""

    def test_empty_summary(self):
        """Empty results should return empty summary."""
        summary = TCAEngine.summarize([])
        assert summary.n_orders == 0

    def test_summary_fields(self, engine):
        """Summary should have all expected fields."""
        results = []
        for i in range(5):
            fills = [{"price": 100.0 + i * 0.1, "quantity": 1000, "timestamp": ""}]
            r = engine.analyze_order(
                order_id=f"ORD{i:03d}",
                ticker="600519.SH",
                side="buy",
                quantity=1000,
                fills=fills,
                decision_price=100.0,
                arrival_price=100.0,
                vwap_price=100.0,
                close_price=100.0,
            )
            results.append(r)

        summary = TCAEngine.summarize(results)
        assert summary.n_orders == 5
        assert isinstance(summary.mean_is_bps, float)
        assert isinstance(summary.median_is_bps, float)
        assert isinstance(summary.std_is_bps, float)
        assert isinstance(summary.by_ticker, dict)

    def test_buy_sell_separate(self, engine):
        """Should compute separate means for buys and sells."""
        results = []

        # Buy with positive cost
        fills_buy = [{"price": 101.0, "quantity": 1000, "timestamp": ""}]
        results.append(engine.analyze_order(
            order_id="BUY001", ticker="A", side="buy", quantity=1000,
            fills=fills_buy, decision_price=100.0, arrival_price=100.0,
            vwap_price=100.0, close_price=100.0,
        ))

        # Sell with positive cost
        fills_sell = [{"price": 99.0, "quantity": 1000, "timestamp": ""}]
        results.append(engine.analyze_order(
            order_id="SELL001", ticker="B", side="sell", quantity=1000,
            fills=fills_sell, decision_price=100.0, arrival_price=100.0,
            vwap_price=100.0, close_price=100.0,
        ))

        summary = TCAEngine.summarize(results)
        assert summary.buy_mean_bps > 0
        assert summary.sell_mean_bps > 0

    def test_to_dataframe(self, engine):
        """to_dataframe should return a DataFrame."""
        fills = [{"price": 100.5, "quantity": 1000, "timestamp": ""}]
        result = engine.analyze_order(
            order_id="ORD001", ticker="600519.SH", side="buy", quantity=1000,
            fills=fills, decision_price=100.0, arrival_price=100.0,
            vwap_price=100.0, close_price=100.0,
        )

        df = engine.to_dataframe([result])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "is_bps" in df.columns
        assert "arrival_bps" in df.columns
        assert "vwap_bps" in df.columns
        assert "delay_bps" in df.columns
        assert "impact_bps" in df.columns

    def test_per_ticker_breakdown(self, engine):
        """Summary should group costs by ticker."""
        results = []

        fills_a = [{"price": 101.0, "quantity": 1000, "timestamp": ""}]
        results.append(engine.analyze_order(
            order_id="ORD001", ticker="A", side="buy", quantity=1000,
            fills=fills_a, decision_price=100.0, arrival_price=100.0,
            vwap_price=100.0, close_price=100.0,
        ))

        fills_b = [{"price": 99.0, "quantity": 1000, "timestamp": ""}]
        results.append(engine.analyze_order(
            order_id="ORD002", ticker="B", side="buy", quantity=1000,
            fills=fills_b, decision_price=100.0, arrival_price=100.0,
            vwap_price=100.0, close_price=100.0,
        ))

        summary = TCAEngine.summarize(results)
        assert "A" in summary.by_ticker
        assert "B" in summary.by_ticker

    def test_realized_spread(self, engine):
        """Realized spread captures bid-ask bounce."""
        # Buy at 100.5, close at 101.0 → captured spread (negative = good for buys)
        fills = [{"price": 100.5, "quantity": 1000, "timestamp": ""}]
        result = engine.analyze_order(
            order_id="ORD001", ticker="A", side="buy", quantity=1000,
            fills=fills, decision_price=100.0, arrival_price=100.0,
            vwap_price=100.0, close_price=101.0,
        )

        # exec < close for buy → negative realized spread (captured spread)
        assert result.realized_spread_bps < 0
