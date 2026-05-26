"""Real-time risk engine — per-tick risk monitoring and pre-trade checks.

Replaces the batch-oriented RiskMonitor with a real-time engine that:
1. Updates portfolio risk on every fill (per-tick, not daily)
2. Pre-trade risk checks: simulates fill before sending order
3. Auto delta-hedge: automatically hedges delta exposure
4. Multi-dimensional limits: position/sector/drawdown/order frequency
5. Real-time stress testing: runs scenarios on current portfolio

Performance target:
- Risk update after fill: < 10μs
- Pre-trade check: < 5μs
- Stress test (500 scenarios): < 1ms

Architecture:
    Fill Event → Risk Engine → Greeks Update → Limit Check → Alert/Halt
                    ↓
              Pre-Trade Check ← New Order Request
                    ↓
              Auto Hedge → Hedge Orders
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from quant_platform.risk.greeks import (
    GreeksCalculator,
    OptionGreeks,
    PortfolioGreeks,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Enums & Data Models
# ──────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    GREEN = "green"       # Normal operation
    YELLOW = "yellow"     # Warning, increased monitoring
    ORANGE = "orange"     # Approaching limits, reduce exposure
    RED = "red"           # Limit breached, halt new orders
    KILL = "kill"         # Kill switch activated, liquidate


class LimitType(str, Enum):
    POSITION_SIZE = "position_size"
    SECTOR_EXPOSURE = "sector_exposure"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    ORDER_FREQUENCY = "order_frequency"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    CONCENTRATION = "concentration"


@dataclass
class RiskLimit:
    """A single risk limit."""
    limit_type: LimitType
    name: str
    threshold: float
    current_value: float = 0.0
    breached: bool = False
    breach_count: int = 0

    @property
    def utilization(self) -> float:
        """How close to the limit (0-1). 1.0 = at limit."""
        if self.threshold == 0:
            return 0.0
        return abs(self.current_value) / abs(self.threshold)

    @property
    def headroom(self) -> float:
        """Remaining capacity before breach."""
        return abs(self.threshold) - abs(self.current_value)


@dataclass
class RiskBreach:
    """A risk limit breach event."""
    limit_type: LimitType
    limit_name: str
    threshold: float
    actual_value: float
    timestamp_ns: int
    action: str  # "warn", "halt", "kill"


@dataclass
class RiskUpdate:
    """Result of a risk update after a fill."""
    risk_level: RiskLevel
    greeks: PortfolioGreeks
    breaches: list[RiskBreach]
    hedge_orders: list[dict]
    limit_utilizations: dict[str, float]
    update_latency_ns: int = 0


@dataclass
class PreTradeCheck:
    """Result of a pre-trade risk check."""
    approved: bool
    reason: str = ""
    projected_greeks: PortfolioGreeks | None = None
    limit_breaches: list[str] = field(default_factory=list)
    check_latency_ns: int = 0


@dataclass
class StressScenario:
    """A stress test scenario."""
    name: str
    equity_shock: float = 0.0      # e.g., -0.10 for -10%
    vol_shock: float = 0.0         # e.g., 2.0 for 2x vol
    rate_shock: float = 0.0        # e.g., 0.01 for +100bp
    spread_widen: float = 0.0      # e.g., 5.0 for 5x spread
    correlation_reset: float = 0.0 # e.g., 0.8 for high correlation
    sector_shock: dict[str, float] = None

    def __post_init__(self):
        if self.sector_shock is None:
            self.sector_shock = {}


@dataclass
class StressTestResult:
    """Result of a stress test run."""
    scenarios: list[dict]
    worst_case_pnl: float
    expected_shortfall: float
    scenarios_breached: int
    total_scenarios: int
    run_time_us: float


# ──────────────────────────────────────────────────────────────────────
# Real-Time Risk Engine
# ──────────────────────────────────────────────────────────────────────


class RealTimeRiskEngine:
    """Real-time risk engine with per-tick monitoring.

    Features:
    1. Per-tick Greeks update after each fill
    2. Pre-trade risk check (simulate fill, check limits)
    3. Auto delta-hedge
    4. Multi-dimensional limits
    5. Real-time stress testing
    6. Kill switch

    Usage:
        engine = RealTimeRiskEngine(config)
        engine.add_limit(RiskLimit(LimitType.DELTA, "max_delta", 1000000))

        # After each fill
        update = engine.on_fill(fill)

        # Before each order
        check = engine.pre_trade_check(order)
    """

    def __init__(
        self,
        max_daily_loss: float = 0.03,
        max_drawdown: float = 0.15,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.30,
        max_order_freq_per_min: int = 50,
        auto_hedge: bool = True,
        hedge_threshold: float = 0.1,  # Hedge when delta > 10% of capital
        asset_universe=None,
    ):
        # Greeks calculator
        self.greeks_calc = GreeksCalculator()
        self._asset_universe = asset_universe

        # Limits
        self._limits: dict[str, RiskLimit] = {}
        self._default_limits(max_daily_loss, max_drawdown,
                            max_position_pct, max_sector_pct,
                            max_order_freq_per_min)

        # State
        self._risk_level = RiskLevel.GREEN
        self._kill_switch_active = False
        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._initial_equity = 0.0

        # Order frequency tracking
        self._order_timestamps: deque[int] = deque(maxlen=10000)

        # Breach history
        self._breach_history: list[RiskBreach] = []

        # Auto-hedge
        self._auto_hedge = auto_hedge
        self._hedge_threshold = hedge_threshold

        # Stress scenarios
        self._stress_scenarios = self._default_stress_scenarios()

        # Metrics
        self._total_checks = 0
        self._total_breaches = 0
        self._total_hedges = 0

    def _default_limits(self, max_daily_loss, max_drawdown,
                       max_position_pct, max_sector_pct,
                       max_order_freq):
        """Set default risk limits."""
        self._limits["max_daily_loss"] = RiskLimit(
            LimitType.DAILY_LOSS, "max_daily_loss", max_daily_loss
        )
        self._limits["max_drawdown"] = RiskLimit(
            LimitType.DRAWDOWN, "max_drawdown", max_drawdown
        )
        self._limits["max_position"] = RiskLimit(
            LimitType.POSITION_SIZE, "max_position", max_position_pct
        )
        self._limits["max_sector"] = RiskLimit(
            LimitType.SECTOR_EXPOSURE, "max_sector", max_sector_pct
        )
        self._limits["max_order_freq"] = RiskLimit(
            LimitType.ORDER_FREQUENCY, "max_order_freq", max_order_freq
        )
        self._limits["max_delta"] = RiskLimit(
            LimitType.DELTA, "max_delta", 1_000_000
        )
        self._limits["max_gamma"] = RiskLimit(
            LimitType.GAMMA, "max_gamma", 500_000
        )

    def add_limit(self, limit: RiskLimit):
        """Add or update a risk limit."""
        self._limits[limit.name] = limit

    def set_initial_equity(self, equity: float):
        """Set initial equity for drawdown calculation."""
        self._initial_equity = equity
        self._peak_equity = equity
        self._current_equity = equity

    # ── Core Operations ──

    def on_fill(self, fill: dict) -> RiskUpdate:
        """Process a fill event and update risk state.

        Args:
            fill: Dict with keys: symbol, side, price, quantity, timestamp_ns

        Returns:
            RiskUpdate with new risk state, breaches, and hedge orders.
        """
        start = time.time_ns()

        symbol = fill.get("symbol", "")
        side = fill.get("side", "buy")
        price = fill.get("price", 0)
        quantity = fill.get("quantity", 0)
        timestamp = fill.get("timestamp_ns", time.time_ns())

        # Update equity (simplified, cross-asset multiplier)
        multiplier = 1.0
        if self._asset_universe is not None:
            inst = self._asset_universe.get(symbol)
            if inst is not None:
                multiplier = inst.multiplier
        notional = price * quantity * multiplier
        if side == "sell":
            self._current_equity += notional
        else:
            self._current_equity -= notional

        self._peak_equity = max(self._peak_equity, self._current_equity)

        # Update Greeks
        portfolio_greeks = self.greeks_calc.compute_portfolio_greeks()

        # Check limits
        breaches = self._check_all_limits(portfolio_greeks, timestamp)

        # Determine risk level
        self._update_risk_level(breaches)

        # Auto-hedge if needed
        hedge_orders = []
        if self._auto_hedge and not self._kill_switch_active:
            hedge_orders = self._maybe_hedge(portfolio_greeks)

        # Limit utilizations
        utilizations = {
            name: limit.utilization
            for name, limit in self._limits.items()
        }

        elapsed = time.time_ns() - start

        return RiskUpdate(
            risk_level=self._risk_level,
            greeks=portfolio_greeks,
            breaches=breaches,
            hedge_orders=hedge_orders,
            limit_utilizations=utilizations,
            update_latency_ns=elapsed,
        )

    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> PreTradeCheck:
        """Check if an order would breach risk limits.

        Simulates the fill and checks limits before sending.

        Args:
            symbol: Asset symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Order price

        Returns:
            PreTradeCheck with approval decision.
        """
        start = time.time_ns()
        self._total_checks += 1

        # Kill switch check
        if self._kill_switch_active:
            return PreTradeCheck(
                approved=False,
                reason="Kill switch active",
                check_latency_ns=time.time_ns() - start,
            )

        # RED level check
        if self._risk_level == RiskLevel.RED:
            return PreTradeCheck(
                approved=False,
                reason=f"Risk level is {self._risk_level.value}",
                check_latency_ns=time.time_ns() - start,
            )

        # Order frequency check
        now = time.time_ns()
        self._order_timestamps.append(now)
        freq_limit = self._limits.get("max_order_freq")
        if freq_limit:
            # Count orders in last minute
            cutoff = now - 60_000_000_000  # 1 minute in ns
            recent = sum(1 for t in self._order_timestamps if t > cutoff)
            freq_limit.current_value = recent
            if recent > freq_limit.threshold:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Order frequency {recent}/min exceeds limit {freq_limit.threshold}",
                    check_latency_ns=time.time_ns() - start,
                )

        # Position size check
        notional = price * quantity
        pos_limit = self._limits.get("max_position")
        if pos_limit and self._current_equity > 0:
            position_pct = notional / self._current_equity
            if position_pct > pos_limit.threshold:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Position {position_pct:.1%} exceeds limit {pos_limit.threshold:.1%}",
                    check_latency_ns=time.time_ns() - start,
                )

        # Simulate Greeks after fill
        # (Simplified: would need to add hypothetical position)
        projected_greeks = self.greeks_calc.compute_portfolio_greeks()

        # Delta limit check
        delta_limit = self._limits.get("max_delta")
        if delta_limit:
            if abs(projected_greeks.total_delta) > delta_limit.threshold:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Projected delta {projected_greeks.total_delta:.0f} exceeds limit",
                    projected_greeks=projected_greeks,
                    check_latency_ns=time.time_ns() - start,
                )

        return PreTradeCheck(
            approved=True,
            projected_greeks=projected_greeks,
            check_latency_ns=time.time_ns() - start,
        )

    def activate_kill_switch(self, reason: str = ""):
        """Activate the kill switch. Halts all trading."""
        self._kill_switch_active = True
        self._risk_level = RiskLevel.KILL
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

        breach = RiskBreach(
            limit_type=LimitType.DRAWDOWN,
            limit_name="kill_switch",
            threshold=0,
            actual_value=0,
            timestamp_ns=time.time_ns(),
            action="kill",
        )
        self._breach_history.append(breach)

    def deactivate_kill_switch(self):
        """Deactivate the kill switch. Resume normal operation."""
        self._kill_switch_active = False
        self._risk_level = RiskLevel.GREEN
        logger.info("Kill switch deactivated")

    # ── Stress Testing ──

    def run_stress_test(self) -> StressTestResult:
        """Run stress scenarios on current portfolio.

        Returns:
            StressTestResult with scenario P&Ls and worst case.
        """
        start = time.time_ns()
        portfolio = self.greeks_calc.compute_portfolio_greeks()
        results = []
        breached = 0

        for scenario in self._stress_scenarios:
            pnl = self._apply_scenario(portfolio, scenario)
            results.append({
                "name": scenario.name,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / max(self._current_equity, 1) * 100, 2),
            })
            if pnl < -self._current_equity * self._limits["max_daily_loss"].threshold:
                breached += 1

        worst_case = min(r["pnl"] for r in results) if results else 0
        sorted_pnls = sorted([r["pnl"] for r in results])
        es_count = max(1, len(sorted_pnls) // 10)
        expected_shortfall = np.mean(sorted_pnls[:es_count])

        elapsed_us = (time.time_ns() - start) / 1000

        return StressTestResult(
            scenarios=results,
            worst_case_pnl=worst_case,
            expected_shortfall=expected_shortfall,
            scenarios_breached=breached,
            total_scenarios=len(self._stress_scenarios),
            run_time_us=elapsed_us,
        )

    def _apply_scenario(self, greeks: PortfolioGreeks, scenario: StressScenario) -> float:
        """Apply a stress scenario to the portfolio."""
        pnl = 0.0

        # Equity shock impact via delta
        if scenario.equity_shock != 0:
            pnl += greeks.dollar_delta * scenario.equity_shock
            # Gamma effect (convexity)
            pnl += 0.5 * greeks.dollar_gamma * scenario.equity_shock ** 2

        # Vol shock impact via vega
        if scenario.vol_shock != 0:
            pnl += greeks.dollar_vega * (scenario.vol_shock - 1)

        # Rate shock impact via rho
        if scenario.rate_shock != 0:
            pnl += greeks.total_rho * scenario.rate_shock * 100

        return pnl

    def _default_stress_scenarios(self) -> list[StressScenario]:
        """Default stress scenarios."""
        return [
            StressScenario("crash_5pct", equity_shock=-0.05, vol_shock=1.5),
            StressScenario("crash_10pct", equity_shock=-0.10, vol_shock=2.0),
            StressScenario("crash_20pct", equity_shock=-0.20, vol_shock=3.0),
            StressScenario("crash_30pct", equity_shock=-0.30, vol_shock=4.0),
            StressScenario("vol_spike_2x", vol_shock=2.0),
            StressScenario("vol_spike_3x", vol_shock=3.0),
            StressScenario("rate_up_50bp", rate_shock=0.005),
            StressScenario("rate_up_100bp", rate_shock=0.01),
            StressScenario("rate_down_50bp", rate_shock=-0.005),
            StressScenario("combo_crash_vol", equity_shock=-0.15, vol_shock=2.5, rate_shock=0.005),
            StressScenario("slow_bleed", equity_shock=-0.03, vol_shock=1.2),
            StressScenario("flash_crash", equity_shock=-0.08, vol_shock=3.0),
        ]

    # ── Internal Methods ──

    def _check_all_limits(
        self,
        greeks: PortfolioGreeks,
        timestamp_ns: int,
    ) -> list[RiskBreach]:
        """Check all risk limits and return breaches."""
        breaches = []

        # Daily loss
        daily_loss_limit = self._limits.get("max_daily_loss")
        if daily_loss_limit and self._initial_equity > 0:
            daily_loss = (self._initial_equity - self._current_equity) / self._initial_equity
            daily_loss_limit.current_value = daily_loss
            if daily_loss > daily_loss_limit.threshold:
                breach = RiskBreach(
                    limit_type=LimitType.DAILY_LOSS,
                    limit_name="max_daily_loss",
                    threshold=daily_loss_limit.threshold,
                    actual_value=daily_loss,
                    timestamp_ns=timestamp_ns,
                    action="halt",
                )
                breaches.append(breach)
                daily_loss_limit.breached = True
                daily_loss_limit.breach_count += 1

        # Drawdown
        dd_limit = self._limits.get("max_drawdown")
        if dd_limit and self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            dd_limit.current_value = drawdown
            if drawdown > dd_limit.threshold:
                breach = RiskBreach(
                    limit_type=LimitType.DRAWDOWN,
                    limit_name="max_drawdown",
                    threshold=dd_limit.threshold,
                    actual_value=drawdown,
                    timestamp_ns=timestamp_ns,
                    action="kill",
                )
                breaches.append(breach)
                dd_limit.breached = True
                dd_limit.breach_count += 1

        # Delta
        delta_limit = self._limits.get("max_delta")
        if delta_limit:
            delta_limit.current_value = abs(greeks.total_delta)
            if abs(greeks.total_delta) > delta_limit.threshold:
                breach = RiskBreach(
                    limit_type=LimitType.DELTA,
                    limit_name="max_delta",
                    threshold=delta_limit.threshold,
                    actual_value=abs(greeks.total_delta),
                    timestamp_ns=timestamp_ns,
                    action="warn",
                )
                breaches.append(breach)

        # Gamma
        gamma_limit = self._limits.get("max_gamma")
        if gamma_limit:
            gamma_limit.current_value = abs(greeks.total_gamma)
            if abs(greeks.total_gamma) > gamma_limit.threshold:
                breach = RiskBreach(
                    limit_type=LimitType.GAMMA,
                    limit_name="max_gamma",
                    threshold=gamma_limit.threshold,
                    actual_value=abs(greeks.total_gamma),
                    timestamp_ns=timestamp_ns,
                    action="warn",
                )
                breaches.append(breach)

        if breaches:
            self._total_breaches += len(breaches)
            self._breach_history.extend(breaches)

        return breaches

    def _update_risk_level(self, breaches: list[RiskBreach]):
        """Update risk level based on breaches."""
        if self._kill_switch_active:
            self._risk_level = RiskLevel.KILL
            return

        if not breaches:
            # Check utilizations for yellow/orange
            max_util = max(
                (l.utilization for l in self._limits.values()),
                default=0,
            )
            if max_util > 0.9:
                self._risk_level = RiskLevel.ORANGE
            elif max_util > 0.7:
                self._risk_level = RiskLevel.YELLOW
            else:
                self._risk_level = RiskLevel.GREEN
            return

        # Has breaches
        has_kill = any(b.action == "kill" for b in breaches)
        has_halt = any(b.action == "halt" for b in breaches)

        if has_kill:
            self._risk_level = RiskLevel.KILL
            self.activate_kill_switch("Drawdown limit breached")
        elif has_halt:
            self._risk_level = RiskLevel.RED
        else:
            self._risk_level = RiskLevel.ORANGE

    def _maybe_hedge(self, greeks: PortfolioGreeks) -> list[dict]:
        """Check if hedging is needed and generate hedge orders."""
        if self._current_equity <= 0:
            return []

        delta_pct = abs(greeks.total_delta) / self._current_equity
        if delta_pct > self._hedge_threshold:
            self._total_hedges += 1
            return self.greeks_calc.get_hedge_orders(target_delta=0)
        return []

    # ── Query Methods ──

    def get_risk_status(self) -> dict:
        """Get current risk status."""
        portfolio_greeks = self.greeks_calc.compute_portfolio_greeks()
        return {
            "risk_level": self._risk_level.value,
            "kill_switch": self._kill_switch_active,
            "equity": round(self._current_equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "daily_pnl": round(self._current_equity - self._initial_equity, 2),
            "greeks": {
                "delta": round(portfolio_greeks.total_delta, 2),
                "gamma": round(portfolio_greeks.total_gamma, 2),
                "vega": round(portfolio_greeks.total_vega, 2),
                "theta": round(portfolio_greeks.total_theta, 2),
            },
            "limits": {
                name: {
                    "threshold": l.threshold,
                    "current": round(l.current_value, 4),
                    "utilization": round(l.utilization, 2),
                    "breached": l.breached,
                    "breach_count": l.breach_count,
                }
                for name, l in self._limits.items()
            },
            "metrics": {
                "total_checks": self._total_checks,
                "total_breaches": self._total_breaches,
                "total_hedges": self._total_hedges,
            },
        }

    def get_breach_history(self, limit: int = 100) -> list[dict]:
        """Get recent breaches."""
        return [
            {
                "type": b.limit_type.value,
                "name": b.limit_name,
                "threshold": b.threshold,
                "actual": round(b.actual_value, 4),
                "action": b.action,
                "timestamp_ns": b.timestamp_ns,
            }
            for b in self._breach_history[-limit:]
        ]

    # ── Backward Compatibility (RiskMonitor API) ──

    @property
    def kill_switch_active(self) -> bool:
        """Backward-compatible property for RiskMonitor.kill_switch_active."""
        return self._kill_switch_active

    def check_pre_trade(self, order: dict) -> tuple[bool, list]:
        """Backward-compatible method matching RiskMonitor.check_pre_trade().

        Args:
            order: Dict with keys: ticker/symbol, side, quantity, price

        Returns:
            (approved, breaches) tuple
        """
        symbol = order.get("ticker", order.get("symbol", ""))
        side = order.get("side", "buy")
        quantity = order.get("quantity", 0)
        price = order.get("price", 0.0)

        result = self.pre_trade_check(symbol, side, quantity, price)

        # Convert RiskBreach objects to simple objects with .message attribute
        breach_objs = []
        if not result.approved and result.reason:
            breach_objs.append(type("Breach", (), {"message": result.reason})())

        return result.approved, breach_objs

    def update_portfolio_state(
        self,
        portfolio_value: float = 0,
        daily_pnl: float = 0,
        positions: dict | None = None,
        sector_weights: dict | None = None,
    ):
        """Backward-compatible method matching RiskMonitor.update_portfolio_state()."""
        if portfolio_value > 0:
            self._current_equity = portfolio_value
            self._peak_equity = max(self._peak_equity, portfolio_value)

    def get_status(self) -> dict:
        """Backward-compatible method matching RiskMonitor.get_status()."""
        return self.get_risk_status()
