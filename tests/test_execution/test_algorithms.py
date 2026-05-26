"""Tests for execution.algorithms — TWAP/VWAP/Iceberg/SmartRouter."""

from datetime import datetime

from quant_platform.execution.algorithms import (
    ExecutionPlan,
    IcebergAlgorithm,
    SmartRouter,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from quant_platform.execution.models import Order, OrderSide


def make_order(ticker="600519", side="buy", quantity=10000):
    return Order(
        ticker=ticker,
        side=OrderSide(side),
        quantity=quantity,
    )


class TestTWAP:
    def test_create_plan(self):
        algo = TWAPAlgorithm(num_slices=5)
        order = make_order(quantity=1000)
        plan = algo.create_plan(order, start_time=datetime(2025, 1, 1, 9, 30), duration_minutes=60)
        assert plan.algorithm == "twap"
        assert len(plan.slices) == 5
        total_qty = sum(s.quantity for s in plan.slices)
        assert total_qty == 1000

    def test_slice_quantity_sum(self):
        algo = TWAPAlgorithm(num_slices=7)
        order = make_order(quantity=1000)
        plan = algo.create_plan(order, start_time=datetime.now(), duration_minutes=60)
        assert sum(s.quantity for s in plan.slices) == 1000

    def test_remainder_distribution(self):
        algo = TWAPAlgorithm(num_slices=3)
        order = make_order(quantity=1000)
        plan = algo.create_plan(order, start_time=datetime.now(), duration_minutes=60)
        # 1000 / 3 = 333 remainder 1, so first slice gets 334
        assert plan.slices[0].quantity == 334
        assert plan.slices[1].quantity == 333


class TestVWAP:
    def test_create_plan(self):
        algo = VWAPAlgorithm(num_slices=5)
        order = make_order(quantity=5000)
        plan = algo.create_plan(order, start_time=datetime(2025, 1, 1, 9, 30), duration_minutes=240)
        assert plan.algorithm == "vwap"
        assert len(plan.slices) == 5
        total_qty = sum(s.quantity for s in plan.slices)
        assert total_qty == 5000

    def test_volume_profile(self):
        algo = VWAPAlgorithm(num_slices=5)
        profile = algo.get_volume_profile("600519")
        assert len(profile) == 5
        assert abs(sum(profile) - 1.0) < 0.01


class TestIceberg:
    def test_create_plan(self):
        algo = IcebergAlgorithm(visible_pct=0.1)
        order = make_order(quantity=5000)
        plan = algo.create_plan(order, start_time=datetime.now(), duration_minutes=120)
        assert plan.algorithm == "iceberg"
        total_qty = sum(s.quantity for s in plan.slices)
        assert total_qty == 5000

    def test_visible_qty(self):
        algo = IcebergAlgorithm(visible_pct=0.1)
        order = make_order(quantity=10000)
        plan = algo.create_plan(order, start_time=datetime.now(), duration_minutes=120)
        # Visible should be ~10% of total
        assert plan.slices[0].quantity <= 2000  # with jitter


class TestSmartRouter:
    def test_small_order_direct(self):
        algo, params = SmartRouter.select_algorithm(100, 1_000_000)
        assert algo == "direct"

    def test_medium_order_twap(self):
        algo, params = SmartRouter.select_algorithm(100_000, 1_000_000)
        assert algo == "twap"

    def test_large_order_vwap(self):
        algo, params = SmartRouter.select_algorithm(200_000, 1_000_000)
        assert algo == "vwap"

    def test_very_large_order_iceberg(self):
        algo, params = SmartRouter.select_algorithm(400_000, 1_000_000)
        assert algo == "iceberg"

    def test_zero_volume_fallback(self):
        algo, params = SmartRouter.select_algorithm(100, 0)
        assert algo == "twap"

    def test_execute_creates_plan(self):
        order = make_order(quantity=100_000)
        plan = SmartRouter.execute(order, avg_daily_volume=1_000_000)
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.slices) > 0

    def test_high_urgency_fewer_slices(self):
        order = make_order(quantity=100_000)
        plan_normal = SmartRouter.execute(order, avg_daily_volume=1_000_000, urgency="normal")
        plan_high = SmartRouter.execute(order, avg_daily_volume=1_000_000, urgency="high")
        assert plan_high.num_slices <= plan_normal.num_slices
