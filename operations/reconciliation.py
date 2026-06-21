"""Reconciliation — 期望持仓 vs 实际持仓对账.

实盘和模拟最大的区别: 模拟永远不会出现"以为买了但没成交"的情况。
这个模块每天跑一次, 检测偏差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PositionRow:
    asset: str
    expected_shares: int
    actual_shares: int
    expected_price: float
    actual_price: float
    diff: int = 0

    @property
    def is_matched(self) -> bool:
        return self.expected_shares == self.actual_shares


@dataclass
class ReconReport:
    total_positions: int
    matched: int
    mismatched: int
    cash_diff: float
    total_mismatch_value: float
    passes: bool
    details: list[PositionRow]

    def print(self):
        print(f"\n{'=' * 60}")
        print(f"  Reconciliation Report")
        print(f"{'=' * 60}")
        print(f"  Positions: {self.total_positions}")
        print(f"  Matched:   {self.matched}")
        print(f"  Diff:      {self.mismatched}")
        print(f"  Cash diff: {self.cash_diff:>+.2f}")
        print(f"  Status:    {'PASS' if self.passes else 'FAIL'}")
        if self.mismatched > 0:
            print(f"\n  Mismatched positions:")
            for p in self.details:
                if not p.is_matched:
                    print(f"    {p.asset}: expected={p.expected_shares} actual={p.actual_shares}")
        print(f"{'=' * 60}")


def reconcile(
    expected_positions: dict[str, tuple[int, float]],
    actual_positions: dict[str, tuple[int, float]],
    expected_cash: float,
    actual_cash: float,
    tolerance: int = 0,
) -> ReconReport:
    """对账."""

    all_assets = set(expected_positions.keys()) | set(actual_positions.keys())
    details: list[PositionRow] = []
    matched = 0
    mismatched = 0
    total_mismatch_value = 0.0

    for asset in sorted(all_assets):
        exp = expected_positions.get(asset, (0, 0.0))
        act = actual_positions.get(asset, (0, 0.0))

        diff = exp[0] - act[0]
        if abs(diff) <= tolerance:
            matched += 1
        else:
            mismatched += 1
            total_mismatch_value += abs(diff) * act[1]

        details.append(PositionRow(
            asset=asset,
            expected_shares=exp[0],
            actual_shares=act[0],
            expected_price=exp[1],
            actual_price=act[1],
            diff=diff,
        ))

    cash_diff = expected_cash - actual_cash
    passes = mismatched == 0 and abs(cash_diff) < 1000

    return ReconReport(
        total_positions=len(all_assets),
        matched=matched,
        mismatched=mismatched,
        cash_diff=cash_diff,
        total_mismatch_value=total_mismatch_value,
        passes=passes,
        details=details,
    )
