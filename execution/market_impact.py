"""Market impact models for realistic order execution simulation.

Implements the Almgren-Chriss model and Square-Root model for estimating
price impact of orders. Used by the tick-level backtester to simulate
how large orders move the market.

Models:
1. Almgren-Chriss (2000): Linear temporary + permanent impact
2. Square-Root Model: Impact ∝ σ * √(Q/V) — industry standard
3. Kyle's Lambda: Impact from informed trading

Key insight: Market impact has two components:
- Temporary impact: Price recovers after the trade (bid-ask bounce)
- Permanent impact: Price permanently shifts (information content)

Reference:
- Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2003): "Optimal Trading Strategies"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MarketImpact:
    """Estimated market impact of an order."""
    temporary: float       # Temporary impact (price recovers)
    permanent: float       # Permanent impact (price shifts permanently)
    total: float           # total = temporary + permanent
    participation_rate: float  # Q / V (order size / market volume)
    spread_cost: float     # Half-spread cost
    timing_risk: float     # Risk from price volatility during execution


@dataclass
class ExecutionCost:
    """Full execution cost breakdown."""
    market_impact: MarketImpact
    spread_cost: float
    commission: float
    tax: float
    opportunity_cost: float  # Cost of not executing (for limit orders)
    total_cost: float
    total_cost_bps: float    # In basis points


# ──────────────────────────────────────────────────────────────────────
# Almgren-Chriss Model
# ──────────────────────────────────────────────────────────────────────


class AlmgrenChrissModel:
    """Almgren-Chriss market impact model.

    The model separates impact into:
    - Temporary impact: η * (Q / V) — proportional to participation rate
    - Permanent impact: γ * (Q / V) — permanent price shift
    - Spread cost: 0.5 * spread
    - Timing risk: σ * √(T) * √(Q/V) — volatility during execution

    Parameters:
    - η (eta): Temporary impact coefficient
    - γ (gamma): Permanent impact coefficient
    - These are typically calibrated from historical trade data

    Usage:
        model = AlmgrenChrissModel(eta=0.0001, gamma=0.00005)
        impact = model.estimate(
            order_quantity=10000,
            market_volume=1000000,
            volatility=0.02,
            spread=0.01,
        )
    """

    def __init__(
        self,
        eta: float = 0.0001,      # Temporary impact coefficient
        gamma: float = 0.00005,    # Permanent impact coefficient
        spread_coeff: float = 0.5, # Spread cost multiplier (typically 0.5)
    ):
        self.eta = eta
        self.gamma = gamma
        self.spread_coeff = spread_coeff

    def estimate(
        self,
        order_quantity: int,
        market_volume: int,
        volatility: float,
        spread: float = 0.0,
        price: float = 100.0,
        execution_horizon: float = 1.0,  # In days
    ) -> MarketImpact:
        """Estimate market impact for an order.

        Args:
            order_quantity: Number of shares to trade
            market_volume: Average daily volume
            volatility: Daily volatility (σ)
            spread: Bid-ask spread
            price: Current price
            execution_horizon: Time to execute in days

        Returns:
            MarketImpact breakdown
        """
        if market_volume <= 0:
            market_volume = 1

        # Participation rate
        participation = order_quantity / market_volume

        # Temporary impact: η * (Q / V)
        temporary = self.eta * participation * price

        # Permanent impact: γ * (Q / V)
        permanent = self.gamma * participation * price

        # Spread cost: 0.5 * spread
        spread_cost = self.spread_coeff * spread

        # Timing risk: σ * √T * √(Q/V) * price
        timing_risk = volatility * math.sqrt(execution_horizon) * math.sqrt(participation) * price

        total = temporary + permanent + spread_cost

        return MarketImpact(
            temporary=temporary,
            permanent=permanent,
            total=total,
            participation_rate=participation,
            spread_cost=spread_cost,
            timing_risk=timing_risk,
        )


# ──────────────────────────────────────────────────────────────────────
# Square-Root Model (Industry Standard)
# ──────────────────────────────────────────────────────────────────────


class SquareRootModel:
    """Square-root market impact model.

    Industry standard model used by most institutional brokers.

    Impact = Y * σ * √(Q / V)

    Where:
    - Y = impact coefficient (typically 0.5-2.0)
    - σ = daily volatility
    - Q = order quantity
    - V = average daily volume

    This model captures the empirical observation that impact
    scales with the square root of participation rate.

    The model is calibrated by:
    1. Temporary: impact that reverts within the trading day
    2. Permanent: impact that persists (information content)

    Usage:
        model = SquareRootModel(y_temp=0.5, y_perm=0.3)
        impact = model.estimate(
            order_quantity=10000,
            market_volume=1000000,
            volatility=0.02,
            price=100.0,
        )
    """

    def __init__(
        self,
        y_temp: float = 0.5,    # Temporary impact coefficient
        y_perm: float = 0.3,    # Permanent impact coefficient
    ):
        self.y_temp = y_temp
        self.y_perm = y_perm

    def estimate(
        self,
        order_quantity: int,
        market_volume: int,
        volatility: float,
        spread: float = 0.0,
        price: float = 100.0,
        execution_horizon: float = 1.0,
    ) -> MarketImpact:
        """Estimate market impact using square-root model."""
        if market_volume <= 0:
            market_volume = 1

        participation = order_quantity / market_volume
        sqrt_participation = math.sqrt(participation)

        # Temporary impact
        temporary = self.y_temp * volatility * sqrt_participation * price

        # Permanent impact
        permanent = self.y_perm * volatility * sqrt_participation * price

        # Spread cost
        spread_cost = 0.5 * spread

        # Timing risk
        timing_risk = volatility * math.sqrt(execution_horizon) * price

        return MarketImpact(
            temporary=temporary,
            permanent=permanent,
            total=temporary + permanent + spread_cost,
            participation_rate=participation,
            spread_cost=spread_cost,
            timing_risk=timing_risk,
        )


# ──────────────────────────────────────────────────────────────────────
# Kyle's Lambda Model
# ──────────────────────────────────────────────────────────────────────


class KyleModel:
    """Kyle's Lambda model for informed trading impact.

    From Kyle (1985): "Continuous Auctions and Insider Trading"

    λ = σ / √V

    Impact = λ * Q

    Where:
    - λ (lambda): Price impact per unit of order flow
    - σ = asset volatility
    - V = market volume
    - Q = order quantity

    This model is particularly useful for estimating the information
    content of trades — large trades by informed traders move prices
    more because they carry information.

    Usage:
        model = KyleModel()
        impact = model.estimate(
            order_quantity=10000,
            market_volume=1000000,
            volatility=0.02,
            price=100.0,
        )
    """

    def estimate(
        self,
        order_quantity: int,
        market_volume: int,
        volatility: float,
        spread: float = 0.0,
        price: float = 100.0,
        execution_horizon: float = 1.0,
    ) -> MarketImpact:
        """Estimate impact using Kyle's Lambda."""
        if market_volume <= 0:
            market_volume = 1

        # Kyle's lambda
        lam = volatility * price / math.sqrt(market_volume)

        # Impact = lambda * Q
        total_impact = lam * order_quantity

        # Split: assume 60% temporary, 40% permanent
        temporary = 0.6 * total_impact
        permanent = 0.4 * total_impact

        participation = order_quantity / market_volume
        spread_cost = 0.5 * spread
        timing_risk = volatility * math.sqrt(execution_horizon) * price

        return MarketImpact(
            temporary=temporary,
            permanent=permanent,
            total=temporary + permanent + spread_cost,
            participation_rate=participation,
            spread_cost=spread_cost,
            timing_risk=timing_risk,
        )


# ──────────────────────────────────────────────────────────────────────
# Composite Model (Ensemble)
# ──────────────────────────────────────────────────────────────────────


class CompositeImpactModel:
    """Ensemble of multiple impact models.

    Combines estimates from multiple models using weighted averaging.
    Default weights are calibrated from empirical studies.

    Usage:
        model = CompositeImpactModel()
        impact = model.estimate(order_quantity=10000, ...)
    """

    def __init__(
        self,
        models: list[tuple[Any, float]] | None = None,
    ):
        if models is None:
            # Default: equal weight across all three models
            self.models = [
                (AlmgrenChrissModel(), 0.4),
                (SquareRootModel(), 0.4),
                (KyleModel(), 0.2),
            ]
        else:
            self.models = models

    def estimate(
        self,
        order_quantity: int,
        market_volume: int,
        volatility: float,
        spread: float = 0.0,
        price: float = 100.0,
        execution_horizon: float = 1.0,
    ) -> MarketImpact:
        """Estimate impact using weighted ensemble."""
        total_weight = sum(w for _, w in self.models)
        weighted_temp = 0.0
        weighted_perm = 0.0
        weighted_spread = 0.0
        weighted_timing = 0.0
        participation = 0.0

        for model, weight in self.models:
            impact = model.estimate(
                order_quantity=order_quantity,
                market_volume=market_volume,
                volatility=volatility,
                spread=spread,
                price=price,
                execution_horizon=execution_horizon,
            )
            w = weight / total_weight
            weighted_temp += impact.temporary * w
            weighted_perm += impact.permanent * w
            weighted_spread += impact.spread_cost * w
            weighted_timing += impact.timing_risk * w
            participation = impact.participation_rate

        return MarketImpact(
            temporary=weighted_temp,
            permanent=weighted_perm,
            total=weighted_temp + weighted_perm + weighted_spread,
            participation_rate=participation,
            spread_cost=weighted_spread,
            timing_risk=weighted_timing,
        )


# ──────────────────────────────────────────────────────────────────────
# Execution Cost Calculator
# ──────────────────────────────────────────────────────────────────────


class ExecutionCostCalculator:
    """Full execution cost breakdown calculator.

    Combines market impact with explicit costs (commission, tax)
    and implicit costs (timing risk, opportunity cost).

    Usage:
        calc = ExecutionCostCalculator(
            impact_model=CompositeImpactModel(),
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
        )
        cost = calc.calculate(
            order_quantity=10000,
            price=100.0,
            side="buy",
            market_volume=1000000,
            volatility=0.02,
            spread=0.01,
        )
    """

    def __init__(
        self,
        impact_model: Any | None = None,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,  # Sell-side only
    ):
        self.impact_model = impact_model or CompositeImpactModel()
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate

    def calculate(
        self,
        order_quantity: int,
        price: float,
        side: str,
        market_volume: int,
        volatility: float,
        spread: float = 0.0,
        execution_horizon: float = 1.0,
        is_limit_order: bool = False,
        limit_price: float | None = None,
    ) -> ExecutionCost:
        """Calculate full execution cost.

        Args:
            order_quantity: Number of shares
            price: Current market price
            side: "buy" or "sell"
            market_volume: Average daily volume
            volatility: Daily volatility
            spread: Bid-ask spread
            execution_horizon: Execution time in days
            is_limit_order: Whether this is a limit order
            limit_price: Limit price (for opportunity cost)

        Returns:
            ExecutionCost breakdown
        """
        notional = price * order_quantity

        # Market impact
        impact = self.impact_model.estimate(
            order_quantity=order_quantity,
            market_volume=market_volume,
            volatility=volatility,
            spread=spread,
            price=price,
            execution_horizon=execution_horizon,
        )

        # Commission
        commission = max(notional * self.commission_rate, self.min_commission)

        # Stamp tax (sell-side only in A-shares)
        tax = (notional * self.stamp_tax_rate) if side == "sell" else 0.0

        # Opportunity cost for limit orders
        opportunity_cost = 0.0
        if is_limit_order and limit_price is not None:
            # Probability of not filling * adverse price movement
            price_diff = abs(price - limit_price) / price
            # Simplified: assume probability of non-fill increases with distance
            fill_prob = max(0.1, 1.0 - price_diff * 50)
            opportunity_cost = (1 - fill_prob) * volatility * price * order_quantity

        total_cost = impact.total + commission + tax + opportunity_cost
        total_cost_bps = (total_cost / notional) * 10000 if notional > 0 else 0

        return ExecutionCost(
            market_impact=impact,
            spread_cost=impact.spread_cost,
            commission=commission,
            tax=tax,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            total_cost_bps=round(total_cost_bps, 2),
        )
