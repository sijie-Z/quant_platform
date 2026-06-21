"""Capital Safety System — 资金安全硬限制.

实盘保护层, 不是优化工具.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class SafetyLimits:
    max_drawdown: float = 0.30
    max_daily_loss: float = 0.05
    max_position_ratio: float = 0.05
    max_single_order_value: float = 500_000
    max_leverage: float = 1.0
    min_cash_reserve: float = 10_000


@dataclass
class SafetyCheck:
    name: str
    passed: bool
    value: float
    limit: float
    message: str = ""


class SafetySystem:
    """资金安全系统. 硬限制, 不可绕过."""

    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self.checks: list[SafetyCheck] = []
        self.kill_switch_triggered = False

    def check_all(self, equity: float, peak_equity: float, cash: float,
                  daily_pnl: float, positions: dict[str, float],
                  order_value: float = 0) -> list[SafetyCheck]:
        """运行所有安全检查."""
        self.checks = []

        # 1. 最大回撤
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        self.checks.append(SafetyCheck(
            name="max_drawdown",
            passed=dd <= self.limits.max_drawdown,
            value=dd, limit=self.limits.max_drawdown,
            message=f"drawdown={dd:.2%} limit={self.limits.max_drawdown:.0%}",
        ))

        # 2. 单日最大亏损
        daily_loss_ratio = abs(daily_pnl) / equity if equity > 0 else 1
        self.checks.append(SafetyCheck(
            name="max_daily_loss",
            passed=daily_loss_ratio <= self.limits.max_daily_loss,
            value=daily_loss_ratio, limit=self.limits.max_daily_loss,
        ))

        # 3. 现金储备
        self.checks.append(SafetyCheck(
            name="min_cash_reserve",
            passed=cash >= self.limits.min_cash_reserve,
            value=cash, limit=self.limits.min_cash_reserve,
        ))

        # 4. 单笔订单价值上限
        if order_value > 0:
            self.checks.append(SafetyCheck(
                name="max_order_value",
                passed=order_value <= self.limits.max_single_order_value,
                value=order_value, limit=self.limits.max_single_order_value,
            ))

        # 触发 kill switch
        failed = [c for c in self.checks if not c.passed]
        if len(failed) >= 2:
            self.kill_switch_triggered = True

        return self.checks

    def can_trade(self) -> bool:
        if self.kill_switch_triggered:
            return False
        failed = [c for c in self.checks if not c.passed]
        return len(failed) == 0
