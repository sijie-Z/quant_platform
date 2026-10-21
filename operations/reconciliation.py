"""Reconciliation — 期望持仓 vs 实际持仓对账.

实盘和模拟最大的区别: 模拟永远不会出现"以为买了但没成交"的情况。
这个模块每天跑一次, 检测偏差。

Note: this is an operations API with no runtime caller yet (the scheduler's
EOD path uses its own reconciliation marker). It is kept for the scheduler
wiring tracked as R52 rather than deleted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PositionRow:
    asset: str
    expected_shares: int
    actual_shares: int
    expected_price: float
    actual_price: float
    diff: int = 0
    tolerance: int = 0

    @property
    def is_matched(self) -> bool:
        # Honours the report's per-position tolerance. The old property
        # compared shares exactly, so print() listed within-tolerance
        # positions as mismatched whenever some other position failed.
        return abs(self.diff) <= self.tolerance


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
        print("  Reconciliation Report")
        print(f"{'=' * 60}")
        print(f"  Positions: {self.total_positions}")
        print(f"  Matched:   {self.matched}")
        print(f"  Diff:      {self.mismatched}")
        print(f"  Cash diff: {self.cash_diff:>+.2f}")
        print(f"  Status:    {'PASS' if self.passes else 'FAIL'}")
        if self.mismatched > 0:
            print("\n  Mismatched positions:")
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
    cash_tolerance: float = 1000.0,
    cash_tolerance_bps: float = 0.0,
    mismatch_materiality: float = 0.0,
) -> ReconReport:
    """对账.

    Args:
        expected_positions: asset -> (shares, price) from the strategy book.
        actual_positions: asset -> (shares, price) from the broker book.
        expected_cash: Expected cash balance.
        actual_cash: Actual cash balance.
        tolerance: Share difference allowed per position.
        cash_tolerance: Absolute cash difference allowed (default 1000).
        cash_tolerance_bps: Additional cash tolerance in basis points of the
            larger cash balance, so the limit scales with AUM.
        mismatch_materiality: Aggregate mismatch value at or below which
            position mismatches do not fail the report (0 = any mismatch
            fails).

    A one-sided position (expected but not in the actual book, or vice
    versa) is valued at the available price — the larger of the expected and
    actual price — never at the missing side's default 0.
    """

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
            # Value the missing leg at the expected/actual price, whichever
            # is available. The old code used act[1] only, so "expected
            # position, actual book empty" (act price defaults to 0)
            # reported a 100 x 50 = 5,000 exposure as 0.0.
            price = max(float(exp[1]), float(act[1]))
            total_mismatch_value += abs(diff) * price

        details.append(PositionRow(
            asset=asset,
            expected_shares=exp[0],
            actual_shares=act[0],
            expected_price=exp[1],
            actual_price=act[1],
            diff=diff,
            tolerance=tolerance,
        ))

    cash_diff = expected_cash - actual_cash
    cash_limit = cash_tolerance + (
        cash_tolerance_bps / 10_000.0
        * max(abs(expected_cash), abs(actual_cash))
    )
    # A zero materiality means "any mismatch fails": it must not absorb a
    # mismatch whose value happens to be 0 (e.g. an unexpected broker
    # position with no price), which used to report PASS.
    positions_ok = mismatched == 0 or (
        mismatch_materiality > 0
        and total_mismatch_value <= mismatch_materiality
    )
    # Inclusive: an exact cash match must pass even when the configured
    # tolerance is 0, and a difference exactly at the limit is "allowed".
    passes = positions_ok and abs(cash_diff) <= cash_limit

    return ReconReport(
        total_positions=len(all_assets),
        matched=matched,
        mismatched=mismatched,
        cash_diff=cash_diff,
        total_mismatch_value=total_mismatch_value,
        passes=passes,
        details=details,
    )
