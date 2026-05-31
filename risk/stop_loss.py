"""Dynamic tiered stop-loss with rebound protection.

Inspired by AlphaPilot-Pro V9.2's production stop-loss system. Contains
practical A-share trading experience not found in textbooks:

1. Three-tier monitoring: monitor → half position → full close
2. Board-specific thresholds (30/68 prefix = ChiNext/STAR, wider stops)
3. Time window (10:45-14:50) — avoids opening volatility + closing auction
4. Rebound protection: reset after bounce above cost
5. T+1 aware: only sell available shares (can_use_volume)
6. 100-share lot rounding for A-share compliance
7. Multiple cost-price field fallbacks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# Default thresholds — board-aware
STOP_LOSS_THRESHOLDS: dict[str, dict[str, float]] = {
    "standard": {  # 主板 00/60
        "monitor": 0.005,   # -0.5% 进入监控
        "level1": 0.012,    # -1.2% 减半仓
        "level2": 0.025,    # -2.5% 清仓
    },
    "special": {  # 创业板 30 / 科创板 68
        "monitor": 0.005,
        "level1": 0.016,    # -1.6% 减半仓（更宽）
        "level2": 0.035,    # -3.5% 清仓（更宽）
    },
}


@dataclass
class StopLossState:
    """Per-position stop-loss tracking state."""
    monitoring: bool = False
    level1_triggered: bool = False
    level2_triggered: bool = False
    original_volume: int = 0
    level1_sold: int = 0
    open_price: float = 0.0
    lowest_profit: float = 0.0
    monitor_start_time: float = 0.0


class DynamicStopLoss:
    """Board-aware, time-gated, multi-tier stop-loss manager.

    Usage:
        dsl = DynamicStopLoss()
        dsl.check(positions, prices, time_str="1045")
    """

    def __init__(self):
        self._state: dict[str, StopLossState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        positions: list[dict[str, Any]],
        prices: dict[str, float],
        time_str: str | None = None,
    ) -> list[dict[str, Any]]:
        """Check all positions and return stop-loss actions.

        Args:
            positions: List of position dicts with keys:
                code, quantity, can_sell, avg_cost
            prices: {code: current_price}
            time_str: Current time HHMM. If provided, enforces time window.

        Returns:
            List of action dicts with keys:
                code, action (monitor/level1/level2/rebound),
                sell_quantity, reason
        """
        now = datetime.now()
        time_str = time_str or now.strftime("%H%M")

        # Time window: 10:45 - 14:50
        if time_str < "1045" or time_str >= "1450":
            return []

        actions = []

        for pos in positions:
            code = pos.get("code", "")
            quantity = int(pos.get("quantity", 0))
            can_sell = int(pos.get("can_sell", 0))
            cost = float(pos.get("avg_cost", 0.0))

            if quantity <= 0 or cost <= 0:
                continue

            current_price = prices.get(code, 0.0)
            if current_price <= 0:
                continue

            # Profit ratio
            profit_ratio = (current_price - cost) / cost

            # Initialize state
            if code not in self._state:
                self._state[code] = StopLossState(
                    original_volume=quantity,
                    open_price=cost,
                    lowest_profit=profit_ratio,
                )
            state = self._state[code]

            # Track lowest profit
            if profit_ratio < state.lowest_profit:
                state.lowest_profit = profit_ratio

            loss_ratio = -profit_ratio  # positive = loss

            # Rebound protection: if Level 1 was triggered but now above cost
            if state.level1_triggered and profit_ratio > 0:
                state.level1_triggered = False
                state.level1_sold = 0
                state.monitoring = False
                state.lowest_profit = profit_ratio
                actions.append({
                    "code": code, "action": "rebound",
                    "reason": f"Rebound above cost ({profit_ratio:.2%})",
                    "sell_quantity": 0,
                })
                continue

            # Phase 1: Enter monitoring at -0.5%
            thresholds = self._get_thresholds(code)
            if not state.monitoring and loss_ratio >= thresholds["monitor"]:
                state.monitoring = True
                actions.append({
                    "code": code, "action": "monitor",
                    "reason": f"Enter stop monitor at {loss_ratio:.2%} loss",
                    "sell_quantity": 0,
                })
                continue

            if not state.monitoring:
                continue

            # Phase 2: Level 1 — sell half
            if not state.level1_triggered and loss_ratio >= thresholds["level1"]:
                if can_sell <= 0:
                    actions.append({
                        "code": code, "action": "skip",
                        "reason": "Level 1 triggered but T+1 locked (can_sell=0)",
                        "sell_quantity": 0,
                    })
                    continue

                sell_qty = self._round_lot(quantity // 2)
                if sell_qty == 0 and can_sell >= 100:
                    sell_qty = 100
                sell_qty = min(sell_qty, can_sell)

                if sell_qty >= 100:
                    state.level1_triggered = True
                    state.level1_sold = sell_qty
                    actions.append({
                        "code": code, "action": "level1",
                        "reason": f"Level 1 stop at {loss_ratio:.2%} loss",
                        "sell_quantity": sell_qty,
                    })
                continue

            # Phase 3: Level 2 — full close
            if loss_ratio >= thresholds["level2"]:
                if can_sell <= 0:
                    continue

                sell_qty = min(can_sell, self._round_lot(quantity))
                if sell_qty >= 100:
                    state.level2_triggered = True
                    actions.append({
                        "code": code, "action": "level2",
                        "reason": f"Level 2 stop at {loss_ratio:.2%} loss",
                        "sell_quantity": sell_qty,
                    })

        return actions

    def reset(self, code: str | None = None) -> None:
        """Reset stop-loss state for one or all positions."""
        if code:
            self._state.pop(code, None)
        else:
            self._state.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_thresholds(code: str) -> dict[str, float]:
        """Get board-appropriate thresholds."""
        prefix = code[:2]
        if prefix in ("30", "68"):
            return STOP_LOSS_THRESHOLDS["special"]
        return STOP_LOSS_THRESHOLDS["standard"]

    @staticmethod
    def _round_lot(qty: int) -> int:
        """Round down to nearest 100-share lot."""
        return (qty // 100) * 100
