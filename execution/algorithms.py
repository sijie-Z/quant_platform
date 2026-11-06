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

    def __init__(self, num_slices: int = 10, max_participation: float = 0.1,
                 lot_size: int = 100):
        self.num_slices = num_slices
        self.max_participation = max_participation
        self.lot_size = max(1, int(lot_size))

    def create_plan(
        self,
        order: Order,
        start_time: datetime,
        duration_minutes: int = 60,
    ) -> ExecutionPlan:
        """Create TWAP execution plan.

        Slices are whole A-share lots (default 100 shares). The parent order's
        odd-lot remainder, if any, is left unexecuted and reported in a warning.
        """
        total_lots = order.quantity // self.lot_size
        num_slices = min(self.num_slices, total_lots) if total_lots > 0 else 0
        base_lots = total_lots // num_slices if num_slices else 0
        extra_lots = total_lots - base_lots * num_slices if num_slices else 0
        interval = duration_minutes / num_slices if num_slices else 0.0

        plan = ExecutionPlan(
            parent_order_id=order.order_id,
            algorithm="twap",
            ticker=order.ticker,
            side=order.side.value,
            total_quantity=order.quantity,
            start_time=start_time.isoformat(),
            end_time=(start_time + timedelta(minutes=duration_minutes)).isoformat(),
            num_slices=num_slices,
            max_participation=self.max_participation,
        )

        for i in range(num_slices):
            lots = base_lots + (1 if i < extra_lots else 0)
            slice_qty = lots * self.lot_size
            target_time = start_time + timedelta(minutes=i * interval)

            plan.slices.append(ExecutionSlice(
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=slice_qty,
                target_time=target_time.isoformat(),
            ))

        allocated = sum(s.quantity for s in plan.slices)
        if allocated != order.quantity:
            logger.warning(
                "TWAP plan for %s allocates %d of %d shares; %d share(s) are "
                "not a whole %d-share lot and were left unexecuted",
                order.ticker, allocated, order.quantity,
                order.quantity - allocated, self.lot_size,
            )

        logger.info("TWAP plan: %s %s x%d over %d min in %d slices",
                     order.side.value, order.ticker, order.quantity,
                     duration_minutes, len(plan.slices))
        return plan


class VWAPAlgorithm:
    """Volume-Weighted Average Price execution.

    Distributes child orders proportional to historical intraday volume profile.
    Heavier execution during high-volume periods (open, close).

    Use case: Minimize tracking error vs VWAP benchmark.
    Requires: Historical intraday volume profile data.
    """

    def __init__(self, num_slices: int = 10, max_participation: float = 0.08,
                 lot_size: int = 100):
        self.num_slices = num_slices
        self.max_participation = max_participation
        self.lot_size = max(1, int(lot_size))

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
            remaining = order.quantity - allocated
            if remaining <= 0:
                break

            if i == n_slices - 1:
                # The final slice must be a whole lot too: an odd-lot
                # remainder cannot be bought on the exchange.
                slice_qty = (remaining // self.lot_size) * self.lot_size
            else:
                # The lot floor can exceed what is left on a small order: a flat
                # volume profile with a 200-share order over 10 slices wants
                # max(100, 20) on every slice. The previous clamp compared
                # against `allocated` *after* incrementing it, so it never bit,
                # and the plan came out as
                #     [100, 100, 0, -100, -200, ..., -700]
                # -- negative child quantities for a 200-share parent.
                slice_qty = max(self.lot_size, int(order.quantity * volume_profile[i]))
                slice_qty = (slice_qty // self.lot_size) * self.lot_size  # round down to a lot
                slice_qty = min(slice_qty, remaining)

            if slice_qty <= 0:
                break

            allocated += slice_qty
            target_time = start_time + timedelta(minutes=i * interval)

            plan.slices.append(ExecutionSlice(
                parent_order_id=order.order_id,
                ticker=order.ticker,
                side=order.side.value,
                quantity=slice_qty,
                target_time=target_time.isoformat(),
                participation_rate=volume_profile[i],
            ))

        plan.num_slices = len(plan.slices)
        if allocated != order.quantity:
            logger.warning(
                "VWAP plan for %s allocated %d of %d shares -- the order is too "
                "small to spread across %d lot-sized slices",
                order.ticker, allocated, order.quantity, n_slices,
            )

        logger.info("VWAP plan: %s %s x%d over %d min, %d slices",
                     order.side.value, order.ticker, order.quantity,
                     duration_minutes, len(plan.slices))
        return plan


class IcebergAlgorithm:
    """Iceberg order execution.

    Shows only a small "visible" quantity to the market,
    refilling from the hidden reserve when visible qty is filled.

    Use case: Large orders where information leakage is critical.
    Typical: visible_qty = 5-10% of total.
    """

    def __init__(self, visible_pct: float = 0.1, randomize: bool = True,
                 lot_size: int = 100):
        self.visible_pct = visible_pct
        self.randomize = randomize
        self.lot_size = max(1, int(lot_size))

    def create_plan(
        self,
        order: Order,
        start_time: datetime,
        duration_minutes: int = 120,
    ) -> ExecutionPlan:
        """Create iceberg execution plan."""
        visible_qty = max(self.lot_size, int(order.quantity * self.visible_pct))
        visible_qty = (visible_qty // self.lot_size) * self.lot_size

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
                slice_qty = max(self.lot_size, (slice_qty // self.lot_size) * self.lot_size)
            else:
                # The last refill must still be a whole lot; any odd-lot
                # remainder is left unexecuted.
                slice_qty = (remaining // self.lot_size) * self.lot_size

            slice_qty = min(slice_qty, remaining)
            if slice_qty <= 0:
                break
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

        plan.num_slices = len(plan.slices)
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


@dataclass
class ExecutionReport:
    """Result of submitting one ExecutionPlan to an order manager."""
    plan_id: str
    ticker: str
    side: str
    algorithm: str
    requested_quantity: int
    filled_quantity: int = 0
    notional: float = 0.0
    average_price: float = 0.0
    order_ids: list[str] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    rejects: list[dict] = field(default_factory=list)
    status: str = "pending"        # completed / partial / rejected

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "ticker": self.ticker,
            "side": self.side,
            "algorithm": self.algorithm,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "average_price": round(self.average_price, 6),
            "order_ids": self.order_ids,
            "fills": self.fills,
            "rejects": self.rejects,
            "status": self.status,
        }


class PlanExecutor:
    """Execute an ExecutionPlan slice by slice through an OMS.

    The algorithms above only produced plans. Nothing turned a slice into an
    order, so TWAP/VWAP/Iceberg never moved a position or a unit of cash.
    This class is that missing link: every slice becomes a real order
    (create -> submit -> fill), so the OMS blotter, positions, T+1 and cash
    accounting all apply.

    Timing: this executes the plan immediately. ``target_time`` remains on the
    plan for callers that want to schedule slices themselves.
    """

    def __init__(self, order_manager, price_lookup=None):
        self.order_manager = order_manager
        self.price_lookup = price_lookup

    def execute_plan(
        self,
        plan: ExecutionPlan,
        price: float | None = None,
        price_lookup=None,
    ) -> ExecutionReport:
        """Submit every slice of ``plan`` as an order and fill it.

        Args:
            plan: execution plan from TWAP/VWAP/Iceberg/SmartRouter.
            price: single execution price applied to every slice.
            price_lookup: callable(ticker) -> price, used when ``price`` is
                not supplied. Falls back to the executor's own lookup.
        """
        lookup = price_lookup or self.price_lookup
        report = ExecutionReport(
            plan_id=plan.plan_id,
            ticker=plan.ticker,
            side=plan.side,
            algorithm=plan.algorithm,
            requested_quantity=plan.total_quantity,
        )

        for execution_slice in plan.slices:
            if execution_slice.quantity <= 0:
                report.rejects.append({
                    "slice_id": execution_slice.slice_id,
                    "reason": "non-positive slice quantity",
                })
                continue

            fill_price = price
            if fill_price is None and lookup is not None:
                fill_price = lookup(execution_slice.ticker)
            if fill_price is None or fill_price <= 0:
                report.rejects.append({
                    "slice_id": execution_slice.slice_id,
                    "reason": "no execution price",
                })
                continue

            try:
                child = self.order_manager.create_order(
                    ticker=execution_slice.ticker,
                    side=execution_slice.side,
                    quantity=execution_slice.quantity,
                    order_type="market",
                    strategy=f"algo:{plan.algorithm}",
                    notes=f"plan={plan.plan_id} slice={execution_slice.slice_id}",
                )
                self.order_manager.submit_order(child.order_id)
                self.order_manager.fill_order(
                    child.order_id,
                    price=float(fill_price),
                    quantity=execution_slice.quantity,
                )
            except (ValueError, KeyError) as exc:
                # Business rejections (T+1 lock, invalid quantity, unknown
                # order). Programming errors -- a signature mismatch, a
                # missing attribute -- deliberately propagate: swallowing
                # them is how the ExecutionService turned every order into a
                # false "rejected".
                report.rejects.append({
                    "slice_id": execution_slice.slice_id,
                    "reason": str(exc),
                })
                continue

            filled = child.filled_quantity
            report.order_ids.append(child.order_id)
            report.filled_quantity += filled
            report.notional += float(fill_price) * filled
            report.fills.append({
                "order_id": child.order_id,
                "slice_id": execution_slice.slice_id,
                "quantity": filled,
                "price": round(float(fill_price), 6),
            })

        if report.filled_quantity > 0:
            report.average_price = report.notional / report.filled_quantity
        if report.filled_quantity >= report.requested_quantity and not report.rejects:
            report.status = "completed"
        elif report.filled_quantity > 0:
            report.status = "partial"
        else:
            report.status = "rejected"

        logger.info(
            "Plan %s executed: %s %s %d/%d @ %.4f (%d slices, %d rejected)",
            plan.plan_id, plan.side, plan.ticker, report.filled_quantity,
            report.requested_quantity, report.average_price,
            len(report.fills), len(report.rejects),
        )
        return report
