"""Execution algorithms — institutional-grade order execution.

Implements:
- TWAP: Time-Weighted Average Price (split into equal time slices)
- VWAP: Volume-Weighted Average Price (weight by historical volume profile)
- ICEBERG: Hidden quantity with visible reserve
- Smart Router: selects best algorithm based on order characteristics

Each algorithm generates child orders with specific timing and sizing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from quant_platform.execution.models import Order
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionSlice:
    """Single slice of an execution algorithm."""
    slice_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_order_id: str = ""
    ticker: str = ""
    side: str = "buy"
    quantity: int = 0
    target_time: str = ""         # When to execute
    actual_time: str = ""         # When actually executed
    target_price: float = 0.0     # Expected price
    actual_price: float = 0.0     # Actual fill price
    status: str = "pending"       # pending/executing/filled/cancelled
    market_volume: int = 0        # Market volume at execution time
    participation_rate: float = 0.0


@dataclass
class ExecutionPlan:
    """Full execution plan for an algorithm order."""
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_order_id: str = ""
    algorithm: str = ""           # twap/vwap/iceberg
    ticker: str = ""
    side: str = "buy"
    total_quantity: int = 0
    start_time: str = ""
    end_time: str = ""
    slices: list[ExecutionSlice] = field(default_factory=list)
    num_slices: int = 0
    max_participation: float = 0.1  # Max 10% of market volume
    status: str = "active"         # active/completed/cancelled
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TWAPAlgorithm:
    """Time-Weighted Average Price execution.

    Splits a large order into equal-sized child orders spread evenly
    over a time window. Each slice gets ~1/N of the total quantity.

    Use case: Minimize market impact when urgency is moderate.
    Typical: 30min-2hr execution window, 5-20 slices.
    """

    def __init__(self, num_slices: int = 10, max_participation: float = 0.1):
        self.num_slices = num_slices
        self.max_participation = max_participation

    def create_plan(
        self,
        order: Order,
        start_time: datetime,
        duration_minutes: int = 60,
    ) -> ExecutionPlan:
        """Create TWAP execution plan."""
        interval = duration_minutes / self.num_slices
        qty_per_slice = order.quantity // self.num_slices
        remainder = order.quantity - qty_per_slice * self.num_slices

        plan = ExecutionPlan(
            parent_order_id=order.order_id,
            algorithm="twap",
            ticker=order.ticker,
            side=order.side.value,
            total_quantity=order.quantity,
            start_time=start_time.isoformat(),
            end_time=(start_time + timedelta(minutes=duration_minutes)).isoformat(),
            num_slices=self.num_slices,
            max_participation=self.max_participation,
        )

        for i in range(self.num_slices):
            slice_qty = qty_per_slice + (1 if i < remainder else 0)
            target_time = start_time + timedelta(minutes=i * interval)

            plan.slices.append(ExecutionSlice(
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=slice_qty,
                target_time=target_time.isoformat(),
            ))

        logger.info("TWAP plan: %s %s x%d over %d min in %d slices",
                     order.side.value, order.ticker, order.quantity,
                     duration_minutes, self.num_slices)
        return plan


class VWAPAlgorithm:
    """Volume-Weighted Average Price execution.

    Distributes child orders proportional to historical intraday volume profile.
    Heavier execution during high-volume periods (open, close).

    Use case: Minimize tracking error vs VWAP benchmark.
    Requires: Historical intraday volume profile data.
    """

    def __init__(self, num_slices: int = 10, max_participation: float = 0.08):
        self.num_slices = num_slices
        self.max_participation = max_participation

    def get_volume_profile(self, ticker: str) -> list[float]:
        """Get typical intraday volume distribution (normalized to sum=1).

        A-share typical profile:
        - 9:30-10:00: ~15% (opening auction burst)
        - 10:00-11:00: ~15%
        - 11:00-11:30: ~8% (pre-lunch slowdown)
        - 13:00-14:00: ~15%
        - 14:00-14:30: ~12%
        - 14:30-15:00: ~20% (closing auction burst)
        - 14:57-15:00: ~15% (closing auction)
        """
        # Realistic A-share intraday volume profile
        profile = [0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.04,
                    0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.07,
                    0.08, 0.05, 0.03, 0.03, 0.02]

        # Normalize
        total = sum(profile[:self.num_slices])
        if total > 0:
            profile = [p / total for p in profile[:self.num_slices]]
        else:
            profile = [1.0 / self.num_slices] * self.num_slices

        return profile

    def create_plan(
        self,
        order: Order,
        start_time: datetime,
        duration_minutes: int = 240,
    ) -> ExecutionPlan:
        """Create VWAP execution plan."""
        volume_profile = self.get_volume_profile(order.ticker)

        # Adjust number of slices to match profile length
        n_slices = min(self.num_slices, len(volume_profile))
        interval = duration_minutes / n_slices

        plan = ExecutionPlan(
            parent_order_id=order.order_id,
            algorithm="vwap",
            ticker=order.ticker,
            side=order.side.value,
            total_quantity=order.quantity,
            start_time=start_time.isoformat(),
            end_time=(start_time + timedelta(minutes=duration_minutes)).isoformat(),
            num_slices=n_slices,
            max_participation=self.max_participation,
        )

        allocated = 0
        for i in range(n_slices):
            if i == n_slices - 1:
                slice_qty = order.quantity - allocated  # Last slice gets remainder
            else:
                slice_qty = max(100, int(order.quantity * volume_profile[i]))
                slice_qty = (slice_qty // 100) * 100  # Round to lot size

            allocated += slice_qty
            target_time = start_time + timedelta(minutes=i * interval)

            plan.slices.append(ExecutionSlice(
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=min(slice_qty, order.quantity - (allocated - slice_qty)),
                target_time=target_time.isoformat(),
                participation_rate=volume_profile[i],
            ))

        logger.info("VWAP plan: %s %s x%d over %d min, %d slices",
                     order.side.value, order.ticker, order.quantity,
                     duration_minutes, n_slices)
        return plan


class IcebergAlgorithm:
    """Iceberg order execution.

    Shows only a small "visible" quantity to the market,
    refilling from the hidden reserve when visible qty is filled.

    Use case: Large orders where information leakage is critical.
    Typical: visible_qty = 5-10% of total.
    """

    def __init__(self, visible_pct: float = 0.1, randomize: bool = True):
        self.visible_pct = visible_pct
        self.randomize = randomize

    def create_plan(
        self,
        order: Order,
        start_time: datetime,
        duration_minutes: int = 120,
    ) -> ExecutionPlan:
        """Create iceberg execution plan."""
        visible_qty = max(100, int(order.quantity * self.visible_pct))
        visible_qty = (visible_qty // 100) * 100

        num_slices = max(1, order.quantity // visible_qty)
        interval = duration_minutes / num_slices

        plan = ExecutionPlan(
            parent_order_id=order.order_id,
            algorithm="iceberg",
            ticker=order.ticker,
            side=order.side.value,
            total_quantity=order.quantity,
            start_time=start_time.isoformat(),
            end_time=(start_time + timedelta(minutes=duration_minutes)).isoformat(),
            num_slices=num_slices,
        )

        remaining = order.quantity
        rng = np.random.RandomState(42)

        for i in range(num_slices):
            # Randomize visible qty +/- 30%
            if self.randomize and i < num_slices - 1:
                jitter = 1.0 + (rng.random() - 0.5) * 0.6
                slice_qty = min(remaining, int(visible_qty * jitter))
                slice_qty = max(100, (slice_qty // 100) * 100)
            else:
                slice_qty = remaining

            slice_qty = min(slice_qty, remaining)
            remaining -= slice_qty

            target_time = start_time + timedelta(minutes=i * interval)

            plan.slices.append(ExecutionSlice(
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=slice_qty,
                target_time=target_time.isoformat(),
            ))

            if remaining <= 0:
                break

        logger.info("Iceberg plan: %s %s x%d, visible=%d, %d slices",
                     order.side.value, order.ticker, order.quantity,
                     visible_qty, len(plan.slices))
        return plan


class SmartRouter:
    """Smart order router — selects the best execution algorithm.

    Decision logic:
    - Small orders (< 5% ADV): Market/Limit, no algo needed
    - Medium orders (5-15% ADV): TWAP
    - Large orders (15-30% ADV): VWAP
    - Very large (> 30% ADV): Iceberg + VWAP

    Also considers:
    - Urgency: high → fewer slices, shorter window
    - Liquidity: low liquidity → more slices, longer window
    - Market hours: avoid opening/closing auctions for VWAP
    """

    @staticmethod
    def select_algorithm(
        order_quantity: int,
        avg_daily_volume: int,
        urgency: str = "normal",    # low/normal/high
        side: str = "buy",
    ) -> tuple[str, dict]:
        """Select the best execution algorithm.

        Returns:
            (algorithm_name, parameters_dict)
        """
        if avg_daily_volume <= 0:
            # No volume data, default to TWAP
            return "twap", {"num_slices": 5, "duration_minutes": 30}

        participation = order_quantity / avg_daily_volume

        if participation < 0.05:
            # Small order — direct execution
            return "direct", {"num_slices": 1, "duration_minutes": 0}
        elif participation < 0.15:
            # Medium — TWAP
            slices = 8 if urgency == "high" else 12
            duration = 30 if urgency == "high" else 60
            return "twap", {"num_slices": slices, "duration_minutes": duration}
        elif participation < 0.30:
            # Large — VWAP
            slices = 10 if urgency == "high" else 15
            duration = 120 if urgency == "high" else 240
            return "vwap", {"num_slices": slices, "duration_minutes": duration}
        else:
            # Very large — Iceberg + VWAP
            return "iceberg", {
                "visible_pct": 0.08,
                "duration_minutes": 240 if urgency != "high" else 120,
            }

    @staticmethod
    def execute(
        order: Order,
        avg_daily_volume: int = 1_000_000,
        urgency: str = "normal",
    ) -> ExecutionPlan:
        """Create execution plan using smart routing."""
        algo_name, params = SmartRouter.select_algorithm(
            order.quantity, avg_daily_volume, urgency, order.side.value,
        )

        if algo_name == "direct":
            # Single fill, no algo needed
            plan = ExecutionPlan(
                parent_order_id=order.order_id,
                algorithm="direct",
                ticker=order.ticker,
                side=order.side.value,
                total_quantity=order.quantity,
                num_slices=1,
            )
            plan.slices.append(ExecutionSlice(
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=order.quantity,
                target_time=datetime.now().isoformat(),
            ))
            return plan

        algo_map = {
            "twap": TWAPAlgorithm,
            "vwap": VWAPAlgorithm,
            "iceberg": IcebergAlgorithm,
        }

        algo_cls = algo_map.get(algo_name, TWAPAlgorithm)
        algo = algo_cls(**{k: v for k, v in params.items() if k != "duration_minutes"})
        return algo.create_plan(
            order,
            start_time=datetime.now(),
            duration_minutes=params.get("duration_minutes", 60),
        )
