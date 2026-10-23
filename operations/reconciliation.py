"""Reconciliation — 期望持仓 vs 实际持仓对账.

实盘和模拟最大的区别: 模拟永远不会出现"以为买了但没成交"的情况。
这个模块每天跑一次, 检测偏差。

Runtime caller: the scheduler's end-of-day path reconciles the Store ledger
(expected) against the broker counter-book (actual) and persists the result
(see `core.scheduler.TradingScheduler._do_eod_reconciliation`).
"""

from __future__ import annotations

import math
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


# ── Input adapters for the scheduler's EOD books-vs-broker check (R11) ──

def _mark_price(current_price, avg_cost) -> float:
    """Mark price for one leg: the current price when positive, else cost.

    Both adapters use this rule so expected and actual positions are valued
    on the same basis. Only positive finite values are accepted — a zero mark
    must fall back to cost rather than value a mismatch at 0.
    """
    for candidate in (current_price, avg_cost):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return 0.0


def _book_from(items) -> dict[str, tuple[int, float]]:
    """Build the ``reconcile()`` mapping from Position objects or dict rows.

    Accepts both shapes so broker ``Position`` instances and Store row
    dicts share one conversion. Rows without an asset code are skipped;
    missing/invalid share counts become 0.
    """
    book: dict[str, tuple[int, float]] = {}
    for item in items:
        if isinstance(item, dict):
            code = item.get("code")
            shares = item.get("quantity", item.get("shares", 0))
            price = _mark_price(item.get("current_price"), item.get("avg_cost"))
        else:
            code = getattr(item, "code", None)
            shares = getattr(item, "quantity", 0)
            price = _mark_price(
                getattr(item, "current_price", 0.0),
                getattr(item, "avg_cost", 0.0),
            )
        if not code:
            continue
        try:
            share_count = int(shares or 0)
        except (TypeError, ValueError):
            share_count = 0
        asset = str(code)
        if asset in book:
            # Duplicates only occur for a merged (tenant-less) Store read or
            # a broker that reports one asset twice. Summing shares and
            # keeping the higher mark mirrors reconcile()'s one-sided
            # valuation at max(expected, actual) price.
            prior_shares, prior_price = book[asset]
            book[asset] = (prior_shares + share_count, max(prior_price, price))
        else:
            book[asset] = (share_count, price)
    return book


def positions_from_broker(positions: list | None) -> dict[str, tuple[int, float]]:
    """Broker ``Position`` list -> ``reconcile()``'s positions mapping.

    ``None`` is the broker interface's "query failed, book unknown" answer
    (``trading/broker.py: BrokerInterface.get_positions``). It is rejected
    here so an unknown book can never be reconciled as a flat one; callers
    must handle the unknown state explicitly. An empty list is a confirmed
    flat account and maps to ``{}``.

    Price basis: ``current_price`` when positive, else ``avg_cost``.
    """
    if positions is None:
        raise ValueError(
            "broker positions are unknown (None), not an empty book; "
            "handle the unknown state before reconciling"
        )
    return _book_from(positions)


def positions_from_store(rows: list[dict] | None) -> dict[str, tuple[int, float]]:
    """Store ``get_positions()`` rows -> ``reconcile()``'s positions mapping.

    Rows are the dicts returned by ``Store.get_positions`` (``code``,
    ``quantity``, ``avg_cost``, ``current_price``). Price basis:
    ``current_price`` when positive (the ledger's mark), else ``avg_cost`` —
    the same rule as ``positions_from_broker`` so both legs are valued alike.

    ``None`` maps to ``{}``: the Store query has no unknown-book state (it
    raises on failure instead), so there is nothing honest to preserve.
    """
    return _book_from(rows or [])
