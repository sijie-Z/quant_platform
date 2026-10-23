"""Reconciliation — 期望持仓 vs 实际持仓对账.

实盘和模拟最大的区别: 模拟永远不会出现"以为买了但没成交"的情况。
这个模块每天跑一次, 检测偏差。

Runtime caller: the scheduler's end-of-day path derives the expected book
from the engine-start baseline plus the stored fills (a trade ledger,
independent of the broker snapshot it is compared against) and reconciles it
against the broker counter-book (actual); see
`core.scheduler.TradingScheduler._do_eod_reconciliation`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PositionRow:
    asset: str
    expected_shares: int
    actual_shares: int
    expected_price: float
    actual_price: float
    diff: int = 0
    tolerance: float = 0

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
    tolerance: float = 0,
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
    dicts share one conversion. A row without an asset code raises: silently
    dropping it would let a malformed broker answer reconcile as a flat
    book. A missing (None) share count is 0; a non-numeric or non-integral
    one raises for the same reason.
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
            raise ValueError(
                f"position row has no asset code (would reconcile as flat): {item!r}"
            )
        if shares is None:
            share_count = 0
        else:
            try:
                share_value = float(shares)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"position share count is not a number: {shares!r}"
                ) from exc
            if not math.isfinite(share_value) or not share_value.is_integer():
                raise ValueError(
                    f"position share count is not a finite integer: {shares!r}"
                )
            share_count = int(share_value)
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

    Superseded as the scheduler's expected leg in R12 (expected now derives
    from the startup baseline + stored fills); kept for callers that really
    mean the persisted book.
    """
    return _book_from(rows or [])


# ── Trade-ledger derivation (R12) ──

def _parse_iso_timestamp(value) -> datetime | None:
    """Naive ISO timestamp -> datetime, or None when unparsable/empty."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _at_or_after(executed_at, created_at: str) -> bool:
    """Whether a fill's timestamp is at/after the baseline timestamp.

    Parsed datetimes are compared so equal instants written with different
    fractional precision count as equal (`>=` is inclusive: a fill stamped
    exactly at the baseline instant belongs to the ledger). Only when either
    side is unparsable is the comparison done on the raw ISO strings, which
    are lexicographically ordered for the naive basis the Store writes.
    """
    executed = str(executed_at or "")
    baseline_ts = str(created_at or "")
    if not baseline_ts:
        return True
    trade_dt = _parse_iso_timestamp(executed)
    base_dt = _parse_iso_timestamp(baseline_ts)
    if (
        trade_dt is not None
        and base_dt is not None
        and (trade_dt.tzinfo is None) == (base_dt.tzinfo is None)
    ):
        return trade_dt >= base_dt
    return executed >= baseline_ts


def _as_share_count(value, what: str) -> int:
    """Finite integral share count; anything else fails loud."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not a number: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{what} is not a finite integer: {value!r}")
    return int(number)


def _as_float(value, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{what} is not finite: {value!r}")
    return number


def _baseline_book(baseline: dict) -> dict[str, dict]:
    """Decode a baseline's positions into ``code -> {quantity, avg_cost}``.

    Accepts the decoded mapping (``baseline["positions"]``, as returned by
    ``Store.get_latest_ledger_baseline``) or the raw ``positions_json``
    text/mapping. Values may be a bare quantity or a mapping with
    ``quantity``/``qty``/``shares`` and an optional ``avg_cost``. A malformed
    entry raises: silently dropping it would shrink the expected book.
    """
    raw = baseline.get("positions")
    if raw is None:
        raw = baseline.get("positions_json")
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"baseline positions_json is not valid JSON: {exc}"
            ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            "baseline positions must be a mapping, "
            f"got {type(raw).__name__}"
        )

    book: dict[str, dict] = {}
    for code, entry in raw.items():
        asset = str(code or "")
        if not asset:
            raise ValueError(
                f"baseline position row has no asset code: {entry!r}"
            )
        avg_cost = 0.0
        if isinstance(entry, dict):
            quantity = entry.get(
                "quantity", entry.get("qty", entry.get("shares")),
            )
            avg_cost = entry.get("avg_cost", 0.0) or 0.0
        else:
            quantity = entry
        book[asset] = {
            "quantity": _as_share_count(
                quantity, f"baseline {asset} quantity",
            ),
            "avg_cost": _as_float(
                avg_cost, f"baseline {asset} avg_cost",
            ),
        }
    return book


def ledger_from_trades(baseline: dict, trades: list[dict]) -> dict:
    """Derive the expected book from a broker baseline + stored fills.

    Args:
        baseline: a ``ledger_baselines`` row (or equivalent mapping) with
            ``cash``, ``created_at`` and positions (``positions`` decoded or
            ``positions_json`` text). Cash and positions are the broker's
            authoritative state when the engine started.
        trades: ``Store.get_trades``/``get_trades_for_export`` rows.

    Only fills with ``executed_at >= baseline.created_at`` and the baseline's
    tenant are applied (a fill stamped exactly at the baseline instant is
    included). Quantities are net -- buy +qty, sell -qty -- with no T+1
    simulation, because the comparison is against the closing net book.
    Cash moves by the full principal plus fees::

        cash = baseline.cash
             - Σ(buy  quantity*price + commission + tax)
             + Σ(sell quantity*price - commission - tax)

    Negative quantities (short books) and negative cash (short proceeds are
    not modelled; fee/dividend drift) are returned as-is -- they are
    observations for the reconciliation, not errors.

    Price basis: the returned mapping carries quantities only. Callers value
    the expected leg at the broker's (actual) prices -- see
    :func:`expected_book_from_ledger` -- so a price-basis difference cannot
    manufacture a share mismatch; only quantities and cash are compared.
    """
    cash = _as_float(baseline.get("cash"), "baseline cash")
    created_at = str(baseline.get("created_at") or "")
    if not created_at:
        raise ValueError(
            "baseline has no created_at; the ledger cannot tell which fills "
            "predate it"
        )
    tenant = str(baseline.get("tenant_id") or "")
    positions = {
        code: entry["quantity"]
        for code, entry in _baseline_book(baseline).items()
    }

    for trade in trades or []:
        trade_tenant = str(trade.get("tenant_id") or "default")
        if tenant and trade_tenant != tenant:
            continue
        if not _at_or_after(trade.get("executed_at"), created_at):
            continue
        code = str(trade.get("code") or "")
        if not code:
            raise ValueError(f"trade has no asset code: {trade!r}")
        side = str(trade.get("side") or "").strip().lower()
        quantity = _as_share_count(
            trade.get("quantity"), f"trade {trade.get('trade_id')!r} quantity",
        )
        price = _as_float(
            trade.get("price"), f"trade {trade.get('trade_id')!r} price",
        )
        commission = _as_float(
            trade.get("commission") or 0.0,
            f"trade {trade.get('trade_id')!r} commission",
        )
        tax = _as_float(
            trade.get("tax") or 0.0,
            f"trade {trade.get('trade_id')!r} tax",
        )
        if side in ("buy", "b"):
            positions[code] = positions.get(code, 0) + quantity
            cash -= quantity * price + commission + tax
        elif side in ("sell", "s"):
            positions[code] = positions.get(code, 0) - quantity
            cash += quantity * price - commission - tax
        else:
            raise ValueError(
                f"trade {trade.get('trade_id')!r} has unsupported side "
                f"{trade.get('side')!r}"
            )

    # A net-zero position is flat, not a holding: drop it so the expected
    # book mirrors the broker's (which never lists closed rows).
    positions = {
        code: quantity for code, quantity in positions.items() if quantity != 0
    }
    return {"positions": positions, "cash": cash}


def expected_book_from_ledger(
    ledger_positions: dict[str, int],
    actual_positions: dict[str, tuple[int, float]],
    avg_costs: dict[str, float] | None = None,
) -> dict[str, tuple[int, float]]:
    """Value the ledger's quantities at the broker's marks (R12).

    Expected prices come from the actual (broker) book for the same asset, so
    the two legs are valued identically and only quantities and cash are
    compared. An expected-only asset (absent from the broker book) falls back
    to its baseline ``avg_cost`` when known, then to 0.0, so materiality
    accounting does not value a vanished position at zero when a cost exists.
    """
    avg_costs = avg_costs or {}
    book: dict[str, tuple[int, float]] = {}
    for code, quantity in (ledger_positions or {}).items():
        actual = actual_positions.get(code)
        if actual is not None and float(actual[1]) > 0:
            price = float(actual[1])
        else:
            price = float(avg_costs.get(code) or 0.0)
        book[str(code)] = (int(quantity), price)
    return book
