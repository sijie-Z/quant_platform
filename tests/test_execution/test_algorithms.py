"""Tests for execution.algorithms — TWAP/VWAP/Iceberg/SmartRouter."""

from datetime import datetime

import pytest

# Fixed timestamp so slice target times are deterministic.
_START = datetime(2026, 1, 5, 9, 30)

from quant_platform.execution.algorithms import (
    ExecutionPlan,
    IcebergAlgorithm,
    PlanExecutor,
    SmartRouter,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from quant_platform.execution.models import Order, OrderSide
from quant_platform.execution.oms import OrderManager


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
        # 10 lots / 3 = 3 lots each; the extra lot goes to the first slice.
        assert [s.quantity for s in plan.slices] == [400, 300, 300]

    @pytest.mark.parametrize("quantity", [100, 250, 1000, 1050, 10_100])
    def test_slices_are_whole_lots_never_exceeding_parent(self, quantity):
        plan = TWAPAlgorithm(num_slices=7).create_plan(
            make_order(quantity=quantity), start_time=_START,
        )
        quantities = [s.quantity for s in plan.slices]
        # An odd-lot remainder cannot be bought, so it is left unallocated.
        assert sum(quantities) == quantity - quantity % 100
        assert all(q > 0 for q in quantities)
        assert all(q % 100 == 0 for q in quantities)

    def test_order_below_one_lot_produces_no_slices(self):
        plan = TWAPAlgorithm(num_slices=5).create_plan(
            make_order(quantity=50), start_time=_START,
        )
        assert plan.slices == []


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

    @pytest.mark.parametrize("quantity", [1, 50, 99])
    def test_below_one_lot_produces_no_slices(self, quantity):
        plan = IcebergAlgorithm().create_plan(
            make_order(quantity=quantity), start_time=_START,
        )
        assert plan.slices == []
        assert plan.num_slices == 0

    @pytest.mark.parametrize("quantity", [100, 150, 250, 500, 1000, 10_000])
    def test_slice_count_and_quantities_are_consistent(self, quantity):
        plan = IcebergAlgorithm().create_plan(
            make_order(quantity=quantity), start_time=_START,
        )
        quantities = [slice_.quantity for slice_ in plan.slices]
        assert plan.num_slices == len(plan.slices)
        assert all(qty > 0 and qty % 100 == 0 for qty in quantities)
        assert sum(quantities) == quantity - quantity % 100


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


class TestVWAPSliceQuantities:
    """Regression for BUG-15.

    The lot-size floor (`max(100, ...)`) can exceed what is left on a small
    order, and the clamp that was meant to catch it compared against
    `allocated` *after* incrementing it, so it never bit. A 200-share order over
    10 slices produced [100, 100, 0, -100, -200, ..., -700] -- a plan summing to
    -2600 for a 200-share parent.
    """

    @pytest.mark.parametrize("quantity", [100, 200, 500, 1000, 10_000])
    def test_slices_are_positive_lots_summing_to_the_parent(self, quantity):
        order = Order(ticker="600519", side=OrderSide.BUY, quantity=quantity)
        plan = VWAPAlgorithm(num_slices=10).create_plan(order, start_time=_START)

        quantities = [s.quantity for s in plan.slices]
        assert quantities, "expected at least one slice"
        assert all(q > 0 for q in quantities), f"non-positive slice in {quantities}"
        assert all(q % 100 == 0 for q in quantities), f"non-lot slice in {quantities}"
        assert sum(quantities) == quantity, (
            f"slices sum to {sum(quantities)}, parent is {quantity}"
        )

    def test_small_order_is_not_over_split(self):
        """200 shares is two lots, so it cannot fill ten lot-sized slices."""
        order = Order(ticker="600519", side=OrderSide.BUY, quantity=200)
        plan = VWAPAlgorithm(num_slices=10).create_plan(order, start_time=_START)
        assert len(plan.slices) == 2
        assert plan.num_slices == 2

    def test_plan_never_exceeds_the_parent_quantity(self):
        order = Order(ticker="600519", side=OrderSide.BUY, quantity=300)
        plan = VWAPAlgorithm(num_slices=10).create_plan(order, start_time=_START)
        assert sum(s.quantity for s in plan.slices) <= order.quantity


class TestPlanExecutor:
    """The algorithms used to produce plans that nothing ever executed."""

    def test_buy_plan_moves_positions_and_cash(self):
        oms = OrderManager(initial_cash=1_000_000.0)
        plan = TWAPAlgorithm(num_slices=5).create_plan(
            make_order(quantity=1000), start_time=_START,
        )

        report = PlanExecutor(oms).execute_plan(plan, price=10.0)

        assert report.status == "completed"
        assert report.filled_quantity == 1000
        assert report.average_price == pytest.approx(10.0)
        assert len(report.order_ids) == 5
        assert report.rejects == []
        assert oms.positions["600519"].quantity == 1000
        # Bought today -> T+1 locked.
        assert oms.positions["600519"].available == 0
        # 1000 * 10 + five minimum commissions of 5.
        assert oms.cash == pytest.approx(1_000_000.0 - 10_000.0 - 25.0)
        assert all(order.status.value == "filled" for order in oms.orders.values())

    def test_sell_plan_obeys_t_plus_one(self):
        oms = OrderManager(initial_cash=1_000_000.0)
        buy_plan = TWAPAlgorithm(num_slices=2).create_plan(
            make_order(quantity=1000), start_time=_START,
        )
        assert PlanExecutor(oms).execute_plan(buy_plan, price=10.0).status == "completed"

        sell_plan = TWAPAlgorithm(num_slices=2).create_plan(
            make_order(side="sell", quantity=1000), start_time=_START,
        )
        same_day = PlanExecutor(oms).execute_plan(sell_plan, price=11.0)
        assert same_day.status == "rejected"
        assert same_day.filled_quantity == 0
        assert all("T+1" in r["reason"] for r in same_day.rejects)
        assert oms.positions["600519"].quantity == 1000

        oms.new_trading_day()
        next_day = PlanExecutor(oms).execute_plan(sell_plan, price=11.0)
        assert next_day.status == "completed"
        assert next_day.filled_quantity == 1000
        assert oms.positions == {}

    def test_missing_price_is_reported_not_raised(self):
        oms = OrderManager(initial_cash=1_000_000.0)
        plan = TWAPAlgorithm(num_slices=3).create_plan(
            make_order(quantity=600), start_time=_START,
        )

        report = PlanExecutor(oms).execute_plan(plan)

        assert report.status == "rejected"
        assert report.filled_quantity == 0
        assert len(report.rejects) == 3
        assert oms.orders == {}
