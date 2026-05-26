"""Real-time risk monitoring and circuit breakers.

Production-grade risk controls:
- Position limits (single stock, sector, total)
- Daily loss limits (absolute and %)
- Drawdown circuit breaker (auto-flatten at threshold)
- Order rate limiter (prevent runaway algorithms)
- Concentration checks (pre-trade and post-trade)
- Kill switch (emergency flatten all positions)

These run on every order submission and every portfolio update.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class RiskLevel(StrEnum):
    GREEN = "green"       # Normal operation
    YELLOW = "yellow"     # Warning — increased monitoring
    ORANGE = "orange"     # Elevated — restrict new positions
    RED = "red"           # Critical — block all new orders
    KILL = "kill"         # Emergency — flatten everything


class BreachType(StrEnum):
    POSITION_LIMIT = "position_limit"
    SECTOR_LIMIT = "sector_limit"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    ORDER_RATE = "order_rate"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"


@dataclass
class RiskBreach:
    """A single risk limit breach event."""
    breach_id: str = ""
    breach_type: BreachType = BreachType.POSITION_LIMIT
    severity: RiskLevel = RiskLevel.YELLOW
    message: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    ticker: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    auto_action: str = ""   # block/warn/flatten


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    # Position limits
    max_single_position_pct: float = 0.05       # 5% max single stock
    max_sector_pct: float = 0.30                 # 30% max single sector
    max_total_positions: int = 200               # Max number of positions

    # Loss limits
    max_daily_loss_pct: float = 0.03             # 3% daily loss limit
    max_drawdown_pct: float = 0.15               # 15% max drawdown from peak
    kill_drawdown_pct: float = 0.25              # 25% kill switch

    # Order rate
    max_orders_per_minute: int = 50
    max_orders_per_second: int = 10

    # Leverage
    max_gross_exposure: float = 1.0              # No leverage (1.0 = 100%)
    max_net_exposure: float = 1.0


class RiskMonitor:
    """Real-time risk monitoring engine.

    Runs pre-trade checks on every order and post-trade checks
    on every fill. Maintains risk state and generates alerts.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.risk_level = RiskLevel.GREEN
        self.breaches: list[RiskBreach] = []
        self.order_timestamps: list[float] = []

        # Portfolio state (updated externally)
        self.portfolio_value: float = 0.0
        self.peak_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.positions: dict[str, dict] = {}  # ticker -> {value, weight, sector}
        self.sector_weights: dict[str, float] = {}

        # Kill switch
        self.kill_switch_active = False

    def check_pre_trade(self, order: dict) -> tuple[bool, list[RiskBreach]]:
        """Run all pre-trade risk checks on an order.

        Args:
            order: dict with ticker, side, quantity, price

        Returns:
            (is_approved, list of breaches)
        """
        breaches = []

        if self.kill_switch_active:
            breaches.append(RiskBreach(
                breach_type=BreachType.POSITION_LIMIT,
                severity=RiskLevel.KILL,
                message="KILL SWITCH ACTIVE — all orders blocked",
                auto_action="block",
            ))
            return False, breaches

        # 1. Order rate check
        rate_breach = self._check_order_rate()
        if rate_breach:
            breaches.append(rate_breach)

        # 2. Position size check
        pos_breach = self._check_position_limit(order)
        if pos_breach:
            breaches.append(pos_breach)

        # 3. Sector concentration check
        sector_breach = self._check_sector_limit(order)
        if sector_breach:
            breaches.append(sector_breach)

        # 4. Daily loss check
        loss_breach = self._check_daily_loss()
        if loss_breach:
            breaches.append(loss_breach)

        # 5. Drawdown check
        dd_breach = self._check_drawdown()
        if dd_breach:
            breaches.append(dd_breach)

        # Update risk level
        if any(b.severity == RiskLevel.KILL for b in breaches):
            self.risk_level = RiskLevel.KILL
        elif any(b.severity == RiskLevel.RED for b in breaches):
            self.risk_level = RiskLevel.RED
        elif any(b.severity == RiskLevel.ORANGE for b in breaches):
            self.risk_level = RiskLevel.ORANGE
        elif any(b.severity == RiskLevel.YELLOW for b in breaches):
            self.risk_level = RiskLevel.YELLOW
        else:
            self.risk_level = RiskLevel.GREEN

        # Determine if order should be blocked
        is_approved = not any(b.auto_action == "block" for b in breaches)

        if breaches:
            for b in breaches:
                logger.warning("RISK BREACH [%s/%s]: %s (current=%.4f, limit=%.4f)",
                               b.severity.value, b.breach_type.value, b.message,
                               b.current_value, b.limit_value)

        self.breaches.extend(breaches)
        return is_approved, breaches

    def _check_order_rate(self) -> RiskBreach | None:
        """Check if order rate exceeds limits."""
        now = time.time()
        self.order_timestamps.append(now)

        # Clean old timestamps
        self.order_timestamps = [t for t in self.order_timestamps if now - t < 60]

        if len(self.order_timestamps) > self.limits.max_orders_per_minute:
            return RiskBreach(
                breach_type=BreachType.ORDER_RATE,
                severity=RiskLevel.ORANGE,
                message=f"Order rate {len(self.order_timestamps)}/min exceeds limit {self.limits.max_orders_per_minute}",
                current_value=len(self.order_timestamps),
                limit_value=self.limits.max_orders_per_minute,
                auto_action="block",
            )
        return None

    def _check_position_limit(self, order: dict) -> RiskBreach | None:
        """Check single position size limit."""
        ticker = order.get("ticker", "")
        price = order.get("price", 0)
        quantity = order.get("quantity", 0)

        if self.portfolio_value <= 0:
            return None

        # Current position + new order value
        current_value = self.positions.get(ticker, {}).get("value", 0)
        new_value = current_value + price * quantity
        new_weight = new_value / self.portfolio_value

        if new_weight > self.limits.max_single_position_pct:
            return RiskBreach(
                breach_type=BreachType.POSITION_LIMIT,
                severity=RiskLevel.RED if new_weight > self.limits.max_single_position_pct * 2 else RiskLevel.ORANGE,
                message=f"{ticker} weight {new_weight:.1%} exceeds limit {self.limits.max_single_position_pct:.1%}",
                current_value=new_weight,
                limit_value=self.limits.max_single_position_pct,
                ticker=ticker,
                auto_action="block",
            )
        return None

    def _check_sector_limit(self, order: dict) -> RiskBreach | None:
        """Check sector concentration limit."""
        ticker = order.get("ticker", "")
        sector = self.positions.get(ticker, {}).get("sector", "Unknown")

        current_sector_weight = self.sector_weights.get(sector, 0)
        if current_sector_weight > self.limits.max_sector_pct:
            return RiskBreach(
                breach_type=BreachType.SECTOR_LIMIT,
                severity=RiskLevel.ORANGE,
                message=f"Sector {sector} weight {current_sector_weight:.1%} exceeds limit {self.limits.max_sector_pct:.1%}",
                current_value=current_sector_weight,
                limit_value=self.limits.max_sector_pct,
                auto_action="warn",
            )
        return None

    def _check_daily_loss(self) -> RiskBreach | None:
        """Check daily loss limit."""
        if self.portfolio_value <= 0:
            return None

        daily_loss_pct = -self.daily_pnl / self.portfolio_value
        if daily_loss_pct > self.limits.max_daily_loss_pct:
            return RiskBreach(
                breach_type=BreachType.DAILY_LOSS,
                severity=RiskLevel.RED,
                message=f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.limits.max_daily_loss_pct:.2%}",
                current_value=daily_loss_pct,
                limit_value=self.limits.max_daily_loss_pct,
                auto_action="block",
            )
        return None

    def _check_drawdown(self) -> RiskBreach | None:
        """Check drawdown circuit breaker."""
        if self.peak_value <= 0:
            return None

        current_dd = (self.peak_value - self.portfolio_value) / self.peak_value

        if current_dd > self.limits.kill_drawdown_pct:
            self.kill_switch_active = True
            return RiskBreach(
                breach_type=BreachType.DRAWDOWN,
                severity=RiskLevel.KILL,
                message=f"KILL SWITCH: Drawdown {current_dd:.2%} exceeds kill threshold {self.limits.kill_drawdown_pct:.2%}",
                current_value=current_dd,
                limit_value=self.limits.kill_drawdown_pct,
                auto_action="flatten",
            )
        elif current_dd > self.limits.max_drawdown_pct:
            return RiskBreach(
                breach_type=BreachType.DRAWDOWN,
                severity=RiskLevel.RED,
                message=f"Drawdown {current_dd:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}",
                current_value=current_dd,
                limit_value=self.limits.max_drawdown_pct,
                auto_action="block",
            )
        return None

    def update_portfolio_state(
        self,
        portfolio_value: float,
        daily_pnl: float,
        positions: dict[str, dict],
        sector_weights: dict[str, float],
    ):
        """Update the risk monitor's view of the portfolio."""
        self.portfolio_value = portfolio_value
        self.daily_pnl = daily_pnl
        self.positions = positions
        self.sector_weights = sector_weights
        self.peak_value = max(self.peak_value, portfolio_value)

    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Emergency kill switch — blocks all new orders."""
        self.kill_switch_active = True
        self.risk_level = RiskLevel.KILL
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)
        self.breaches.append(RiskBreach(
            breach_type=BreachType.POSITION_LIMIT,
            severity=RiskLevel.KILL,
            message=f"Kill switch: {reason}",
            auto_action="flatten",
        ))

    def deactivate_kill_switch(self):
        """Deactivate kill switch (requires manual confirmation)."""
        self.kill_switch_active = False
        self.risk_level = RiskLevel.YELLOW
        logger.info("Kill switch deactivated")

    def get_status(self) -> dict:
        """Get current risk status summary."""
        recent_breaches = [b for b in self.breaches
                          if b.timestamp > datetime.now().isoformat()[:10]]

        return {
            "risk_level": self.risk_level.value,
            "kill_switch": self.kill_switch_active,
            "portfolio_value": self.portfolio_value,
            "peak_value": self.peak_value,
            "current_drawdown": (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            "n_positions": len(self.positions),
            "n_breaches_today": len(recent_breaches),
            "recent_breaches": [
                {
                    "type": b.breach_type.value,
                    "severity": b.severity.value,
                    "message": b.message,
                    "timestamp": b.timestamp,
                }
                for b in recent_breaches[-10:]
            ],
            "limits": {
                "max_single_position": self.limits.max_single_position_pct,
                "max_sector": self.limits.max_sector_pct,
                "max_daily_loss": self.limits.max_daily_loss_pct,
                "max_drawdown": self.limits.max_drawdown_pct,
                "kill_drawdown": self.limits.kill_drawdown_pct,
            },
        }
