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

    # Order rate. A rebalance legitimately submits the whole basket in one
    # burst, so the per-second cap must not be below the per-minute budget.
    max_orders_per_minute: int = 50
    max_orders_per_second: int = 50

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
        # Optional ticker -> sector map for names not yet held. Without it a
        # new order's sector resolved to "Unknown", so the sector cap could
        # not be evaluated before the position existed.
        self.sector_map: dict[str, str] = {}
        self._last_state_breach = ""
        self._state_risk_level = RiskLevel.GREEN

        # Kill switch
        self.kill_switch_active = False

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        """Provide ticker -> sector classifications for pre-trade checks."""
        self.sector_map = dict(sector_map)

    def _resolve_sector(self, ticker: str, order: dict | None = None) -> str:
        if order and order.get("sector"):
            return str(order["sector"])
        held = self.positions.get(ticker, {}).get("sector")
        if held:
            return str(held)
        if ticker in self.sector_map:
            return str(self.sector_map[ticker])
        return "Unknown"

    def check_pre_trade(self, order: dict) -> tuple[bool, list[RiskBreach]]:
        """Run all pre-trade risk checks on an order.

        Args:
            order: dict with ticker, side, quantity, price

        Returns:
            (is_approved, list of breaches)
        """
        breaches = []

        # A missing instrument key used to degrade silently: the limit checks
        # below read `order.get("ticker", "")`, so an order dict keyed on `code`
        # -- the name `Order` and the positions table use -- resolved to an
        # empty ticker, found no existing position, and re-checked the limit
        # from zero. A held position of 49% against a 5% limit passed. Risk
        # checks fail closed, so an order we cannot identify is blocked rather
        # than waved through, and the aliases the codebase actually uses are
        # accepted instead of only the one this function documents.
        instrument = order.get("ticker") or order.get("symbol") or order.get("code")
        if not instrument:
            breaches.append(RiskBreach(
                breach_type=BreachType.POSITION_LIMIT,
                severity=RiskLevel.KILL,
                message=(
                    "Order has no instrument identifier (ticker/symbol/code) -- "
                    "blocked, because position and sector limits cannot be "
                    "evaluated without one"
                ),
                auto_action="block",
            ))
            return False, breaches
        if order.get("ticker") != instrument:
            order = {**order, "ticker": instrument}

        side = str(order.get("side", "buy")).lower()
        is_reducing = (
            side in ("sell", "s")
            and float(self.positions.get(instrument, {}).get("value", 0)) > 0
        )

        if self.kill_switch_active and not is_reducing:
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

        # 4. Position-count and gross/net exposure limits
        breaches.extend(self._check_exposure_limits(order))

        # 5. Daily loss check. A sell that reduces exposure is how the book
        # recovers from a breach; blocking it turns a bad day into a trap.
        loss_breach = self._check_daily_loss()
        if loss_breach:
            if is_reducing:
                loss_breach.auto_action = "warn"
            breaches.append(loss_breach)

        # 6. Drawdown check
        dd_breach = self._check_drawdown()
        if dd_breach:
            if is_reducing:
                dd_breach.auto_action = "warn"
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

        recent_second = [t for t in self.order_timestamps if now - t < 1.0]
        if len(recent_second) > self.limits.max_orders_per_second:
            return RiskBreach(
                breach_type=BreachType.ORDER_RATE,
                severity=RiskLevel.ORANGE,
                message=(
                    f"Order rate {len(recent_second)}/s exceeds limit "
                    f"{self.limits.max_orders_per_second}"
                ),
                current_value=len(recent_second),
                limit_value=self.limits.max_orders_per_second,
                auto_action="block",
            )

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

    def _check_exposure_limits(self, order: dict) -> list[RiskBreach]:
        """Check position count and gross/net exposure.

        `RiskLimits` declared `max_total_positions`, `max_gross_exposure`
        and `max_net_exposure`, but no check read them.
        """
        breaches: list[RiskBreach] = []
        if self.portfolio_value <= 0:
            return breaches

        ticker = str(order.get("ticker", ""))
        side = str(order.get("side", "buy")).lower()
        price = float(order.get("price", 0) or 0)
        quantity = float(order.get("quantity", 0) or 0)
        delta = price * quantity / self.portfolio_value
        if side not in ("buy", "b"):
            delta = -delta

        if side in ("buy", "b") and ticker not in self.positions:
            n_positions = sum(
                1 for p in self.positions.values()
                if float(p.get("value", 0) or 0) > 0
            )
            if n_positions >= self.limits.max_total_positions:
                breaches.append(RiskBreach(
                    breach_type=BreachType.CONCENTRATION,
                    severity=RiskLevel.ORANGE,
                    message=(
                        f"Position count {n_positions} already at limit "
                        f"{self.limits.max_total_positions}"
                    ),
                    current_value=n_positions,
                    limit_value=self.limits.max_total_positions,
                    auto_action="block",
                ))

        weights = [
            float(p.get("weight", 0) or 0) for p in self.positions.values()
        ]
        gross = sum(abs(w) for w in weights)
        net = sum(weights)
        projected_gross = max(0.0, gross + delta)
        projected_net = net + delta

        if projected_gross > self.limits.max_gross_exposure + 1e-9:
            breaches.append(RiskBreach(
                breach_type=BreachType.LEVERAGE,
                severity=RiskLevel.ORANGE,
                message=(
                    f"Gross exposure {projected_gross:.1%} would exceed limit "
                    f"{self.limits.max_gross_exposure:.0%}"
                ),
                current_value=projected_gross,
                limit_value=self.limits.max_gross_exposure,
                auto_action="block",
            ))
        if abs(projected_net) > self.limits.max_net_exposure + 1e-9:
            breaches.append(RiskBreach(
                breach_type=BreachType.LEVERAGE,
                severity=RiskLevel.ORANGE,
                message=(
                    f"Net exposure {projected_net:.1%} would exceed limit "
                    f"{self.limits.max_net_exposure:.0%}"
                ),
                current_value=projected_net,
                limit_value=self.limits.max_net_exposure,
                auto_action="block",
            ))
        return breaches

    def _check_position_limit(self, order: dict) -> RiskBreach | None:
        """Check single position size limit, direction-aware.

        The old formula added `price * quantity` for both sides, so selling a
        position looked like opening one: a full exit of a 10% holding was
        reported as a 110% position and blocked. Sells are risk-reducing and
        pass this check; only an increasing order can breach the cap.
        """
        ticker = order.get("ticker", "")
        side = str(order.get("side", "buy")).lower()
        price = float(order.get("price", 0) or 0)
        quantity = float(order.get("quantity", 0) or 0)

        if self.portfolio_value <= 0:
            return None

        current_value = float(self.positions.get(ticker, {}).get("value", 0))
        signed_notional = price * quantity
        if side in ("sell", "s"):
            signed_notional = -signed_notional
        new_value = max(0.0, current_value + signed_notional)
        new_weight = new_value / self.portfolio_value

        if side in ("buy", "b") and new_weight > self.limits.max_single_position_pct:
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
        """Check sector concentration limit, including this order's delta."""
        ticker = order.get("ticker", "")
        side = str(order.get("side", "buy")).lower()
        sector = self._resolve_sector(ticker, order)

        current_sector_weight = float(self.sector_weights.get(sector, 0))
        if side in ("buy", "b") and self.portfolio_value > 0:
            price = float(order.get("price", 0) or 0)
            quantity = float(order.get("quantity", 0) or 0)
            projected = (current_sector_weight * self.portfolio_value + price * quantity) / self.portfolio_value
        else:
            projected = current_sector_weight

        if side in ("buy", "b") and projected > self.limits.max_sector_pct:
            return RiskBreach(
                breach_type=BreachType.SECTOR_LIMIT,
                severity=RiskLevel.ORANGE,
                message=f"Sector {sector} weight {projected:.1%} exceeds limit {self.limits.max_sector_pct:.1%}",
                current_value=projected,
                limit_value=self.limits.max_sector_pct,
                auto_action="block",
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
        """Update the risk monitor's view of the portfolio.

        The equity-based breakers are evaluated here as well as in
        `check_pre_trade`. A fully invested book can breach the drawdown or
        daily-loss limit without emitting a single order; previously the
        risk level stayed GREEN and the kill switch never armed until someone
        tried to trade.
        """
        self.portfolio_value = portfolio_value
        self.daily_pnl = daily_pnl
        self.positions = positions
        self.sector_weights = sector_weights
        self.peak_value = max(self.peak_value, portfolio_value)

        state_breaches: list[RiskBreach] = []
        if not self.kill_switch_active:
            for check in (self._check_daily_loss, self._check_drawdown):
                breach = check()
                if breach is not None:
                    state_breaches.append(breach)

        # The engine calls this every cycle; append the breach only when the
        # condition changes, or the same warning would fill the breach log.
        signature = ";".join(b.message for b in state_breaches)
        if signature != self._last_state_breach:
            self._last_state_breach = signature
            for breach in state_breaches:
                logger.warning("RISK BREACH [%s/%s]: %s",
                               breach.severity.value, breach.breach_type.value,
                               breach.message)
                self.breaches.append(breach)

        if state_breaches:
            levels = [b.severity for b in state_breaches]
            if RiskLevel.KILL in levels:
                self._state_risk_level = RiskLevel.KILL
            elif RiskLevel.RED in levels:
                self._state_risk_level = RiskLevel.RED
            else:
                self._state_risk_level = RiskLevel.YELLOW
            self.risk_level = self._state_risk_level
        elif self._state_risk_level != RiskLevel.GREEN:
            # Recovered: only undo a level that this method set, so an
            # order-level breach does not get cleared by the next update.
            self._state_risk_level = RiskLevel.GREEN
            if self.risk_level in (RiskLevel.KILL, RiskLevel.RED, RiskLevel.YELLOW):
                self.risk_level = RiskLevel.GREEN

    def project_order(self, order: dict) -> None:
        """Fold an approved, not-yet-filled buy into the exposure view.

        During a rebalance cycle the engine checks orders one after another.
        Each buy used to be compared against the book *before* the cycle, so
        thirty 3%-of-equity orders all passed a 30% sector limit; the next
        cycle then saw the whole sector and blocked every buy. Projecting the
        approved order makes the remaining checks see the pending exposure.
        The next `update_portfolio_state` replaces this view with the actual
        book.
        """
        side = str(order.get("side", "buy")).lower()
        ticker_value = order.get("ticker") or order.get("symbol") or order.get("code")
        if not ticker_value:
            return
        price = float(order.get("price", 0) or 0)
        quantity = float(order.get("quantity", 0) or 0)
        notional = price * quantity
        if notional <= 0 or self.portfolio_value <= 0:
            return
        ticker = str(ticker_value)

        if side in ("sell", "s"):
            # Project a reducing order so a sells-first batch frees exposure
            # before the buys are checked.
            pos = self.positions.get(ticker)
            if pos is None:
                return
            sector = self._resolve_sector(ticker, order)
            reduce_value = min(float(pos.get("value", 0.0)), notional)
            reduce_weight = min(
                float(pos.get("weight", 0.0)), notional / self.portfolio_value,
            )
            pos["value"] = float(pos.get("value", 0.0)) - reduce_value
            pos["weight"] = float(pos.get("weight", 0.0)) - reduce_weight
            if sector in self.sector_weights:
                self.sector_weights[sector] = max(
                    0.0, self.sector_weights[sector] - reduce_weight,
                )
            return

        if side not in ("buy", "b"):
            return

        sector = self._resolve_sector(ticker, order)
        weight = notional / self.portfolio_value
        pos = self.positions.get(ticker)
        if pos is None:
            pos = {"value": 0.0, "weight": 0.0, "sector": sector}
            self.positions[ticker] = pos
        pos["value"] = float(pos.get("value", 0.0)) + notional
        pos["weight"] = float(pos.get("weight", 0.0)) + weight
        pos["sector"] = sector
        self.sector_weights[sector] = self.sector_weights.get(sector, 0.0) + weight

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


# ---------------------------------------------------------------------------
# Process-wide instance.
#
# docs/ARCHITECTURE_V2.md ADR-0003 already makes this a design rule ("OMS、
# 风控、实盘引擎各只有一条主实现"), but it also has to be literally one object:
# the Monitor dashboard's kill switch and the trading engine must act on the
# same state. They previously did not — api/monitor.py and api/routes.py each
# built their own RiskMonitor, so pressing Kill Switch in the UI flipped a
# flag on an instance no engine ever read, and the engine kept trading.
# ---------------------------------------------------------------------------
_default_monitor: RiskMonitor | None = None


def get_risk_monitor() -> RiskMonitor:
    """Return the process-wide RiskMonitor, creating it on first use.

    Use this everywhere instead of constructing RiskMonitor() directly.
    """
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = RiskMonitor()
    return _default_monitor
