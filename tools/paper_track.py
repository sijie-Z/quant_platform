#!/usr/bin/env python
"""Forward paper tracking: a fixed target book -> an idempotent daily NAV ledger.

The research report's own conclusion is that the EP + low-turnover tilt is a
*style tilt*, not validated alpha. The next step it asks for is forward paper
tracking. This tool does exactly that, and nothing else: it reads a target
weights file and a cached daily panel, applies the platform's execution
conventions, and appends one row per trading session to a ledger. It never
talks to a broker, never moves money, and never writes outside ``--out-dir``.

Execution and accounting contract
---------------------------------
* **Execution day.** The book is established at the close of the first cached
  session *strictly after* the target's ``as_of`` date -- the same
  "next session at the close" convention as the backtest engine. The target's
  ``as_of`` row itself is never traded on.
* **Returns (ex-rights).** Asset and benchmark returns are Baostock
  ``pct_chg / 100``, whose ``preclose`` is adjusted for ex-rights events, so a
  dividend or split is not booked as a loss (raw close ratios would be). Raw
  ``close`` is used only to convert share volume into CNY traded value for the
  cost model, and as the reference price a stale position is liquidated at.
* **Tradability gate (R9/R19a).** A target asset is bought on the execution
  close only when it has ``close > 0``, a finite ``pct_chg``, ``volume > 0``
  (the R9 ``tradable_mask``), and did not close limit-up
  (``pct_chg > 9.8%``, the platform's limit-up approximation). Blocked or
  untradable weight stays in cash and a warning is recorded -- the tool never
  assumes a fill that could not have happened.
* **Costs (R9 basis).** On the execution day the tool buys the target book
  from cash and charges :class:`~quant_platform.backtest.cost_model.CostModel`
  exactly the way the engine does: CNY buy notionals
  (``weight * capital``), ``daily_volume`` in CNY as
  ``volume_shares * raw_close`` (the cached ``close`` is the raw CNY close --
  ``amount`` matches ``close * volume``), the date-dependent statutory stamp
  tax (no sells on a build, so the stamp tax is zero), the per-order CNY 5
  commission floor, the both-sides transfer fee, and -- with
  ``--cost-mode impact`` -- the square-root impact rule on participation.
  Volatility is the causal ``rolling(20).std()`` estimate on the same
  ``pct_chg`` returns with suspension and resume-day gap returns masked,
  matching ``data/pipeline.py``'s return mask (R9 parity; backward rows make
  the unshifted window at the execution close identical to the engine's
  forward-panel ``shift(1)`` window). If the modelled build cost would
  exhaust the capital the run aborts before writing anything.
* **Drift.** Between executions weights drift with the assets' returns; cash
  earns nothing. NAV is close-to-close: the execution row's only return is
  the build cost, and each later row's return is the drifted book's return
  from the previous close.
* **Stale prices.** A session is *stale* for a held asset when its ``close``
  is missing/non-positive, or its ``pct_chg`` or ``volume`` is missing/NaN, or
  the volume is zero (the cache encodes
  intra-series suspensions as a repeated close with NaN/zero volume and
  ``pct_chg``, not as a NaN close). A stale session is held at the last valid
  close with return 0 and counted in ``stale_assets``; it is never silently
  marked to zero. When the price returns, the resume-day ``pct_chg`` carries
  the whole gap return. After ``--max-stale-sessions`` consecutive stale
  sessions (default 20; 0 disables) the position is liquidated at its last
  valid close and the proceeds become cash, with a warning. Liquidation is a
  valuation convention and books no trade cost: there is no tradable price to
  charge it against.
* **Benchmark.** Equal-weight, daily-rebalanced mean of the cached universe's
  ``pct_chg`` returns over the same window (the ``daily_ew`` convention in
  ``tools/real_data_research.py``); an asset is excluded from a session when
  its observation is stale by the same rule. The execution row is 0.0.
* **Scope.** One target per ledger: this version builds the book once and
  then lets it drift -- there is no scheduled rebalancing after the build.
  To track a different target, use a different ``--out-dir``; an existing
  ledger refuses to mix targets, capital, cost mode, or stale policy.
  Positions are weight-level (the platform backtest does not round to
  100-share lots either).

Idempotent accumulation
-----------------------
``nav.csv`` is keyed by date. Re-running the same inputs appends nothing,
leaves ``nav.csv`` byte-for-byte untouched, and rewrites ``summary.json`` with
identical content (it contains no timestamps). When the cache grows, only
dates after the last ledger row are appended; existing rows are preserved
verbatim. Mixing a different target, capital, cost mode, stale policy, or
ledger version into an existing out-dir is refused, and a ``nav.csv`` without
its ``summary.json`` is treated as an unverifiable orphan and refused rather
than appended to. If a revised cache would change already-written NAV values,
the tool keeps the ledger rows but records the discrepancy in
``summary.json`` warnings.

Errors (target beyond the cache, no session after ``as_of``, malformed target
or panel, a build cost that would exhaust capital) exit non-zero before
anything is written, so a failed run never leaves a half-built ledger.

Outputs (in ``--out-dir``)
--------------------------
``nav.csv`` columns:
``date, nav, daily_return, benchmark_daily_return, cum_return, drawdown,
n_positions, stale_assets, costs_paid``.

``summary.json`` carries since-inception return / annualized / Sharpe / max
drawdown, the benchmark on the same window, tracking difference and tracking
error, the execution date, total costs, the return/tradability bases,
warnings, and the mandatory ``disclaimer``.

Usage::

    python tools/paper_track.py \
      --target docs/research/target_portfolio_2025-12-31.json \
      --cache .cache/baostock/daily_2018-01-01_2025-12-31_500.parquet \
      --out-dir docs/research/paper_track \
      --capital 2000000 --cost-mode impact
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (
    str(ROOT.parent),
    str(ROOT),
    str(ROOT / "packages" / "quant_core" / "src"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    tracking_error,
)

DISCLAIMER = "paper tracking only; style-tilt demo, not validated alpha"
LEDGER_VERSION = 2
NAV_COLUMNS = (
    "date",
    "nav",
    "daily_return",
    "benchmark_daily_return",
    "cum_return",
    "drawdown",
    "n_positions",
    "stale_assets",
    "costs_paid",
)
PERIODS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.03
COMMISSION = 0.0003
SLIPPAGE = 0.001
DEFAULT_CAPITAL = 2_000_000.0
DEFAULT_MAX_STALE_SESSIONS = 20
VOL_LOOKBACK = 20
VOL_MIN_PERIODS = 5
VOL_FALLBACK = 0.02
LIMIT_UP_PCT = 9.8
RETURN_BASIS = (
    "pct_chg/100 (Baostock ex-rights preclose); raw close is used only for "
    "volume x price cost conversion and the stale liquidation reference"
)
TRADABLE_GATE = (
    "execution buys require close > 0, finite pct_chg, volume > 0 (R9 "
    "tradable_mask) and pct_chg <= 9.8% (R19a limit-up approximation); "
    "blocked weight stays in cash with a warning"
)
BENCHMARK_DEFINITION = (
    "equal-weight daily-rebalanced mean of the cached universe's pct_chg "
    "close-to-close returns over the same window; stale observations are "
    "excluded from that session"
)
STALE_POLICY = (
    "a session is stale when close, pct_chg or volume is missing/invalid or "
    "the volume is zero; hold at the last valid close (return 0, flagged "
    "stale); when the price returns the resume-day pct_chg carries the gap "
    "return; after max_stale_sessions consecutive stale sessions, liquidate "
    "at the last valid close with no extra trade cost (no tradable price "
    "exists) and hold the proceeds as cash"
)
VOLATILITY_BASIS = (
    "rolling(20, min_periods=5).std() of pct_chg/100 through the execution "
    "close, masking suspension and resume-day gap returns as data/pipeline.py "
    "does; short histories fall back to 2% daily"
)
REVISION_NAV_TOLERANCE = 1e-6


class PaperTrackError(RuntimeError):
    """Expected, user-actionable failure (bad input or insufficient data)."""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _normalize_code(value: Any) -> str:
    """Normalise an A-share code to its 6-digit string form when numeric."""
    text = str(value).strip()
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def _parse_target_date(value: Any) -> pd.Timestamp:
    """Parse the target ``as_of``; integers are YYYYMMDD, never epoch ns."""
    if isinstance(value, bool) or not isinstance(
        value, (str, int, np.integer, dt.date, pd.Timestamp),
    ):
        raise PaperTrackError(
            f"target 'as_of' must be a date string or YYYYMMDD integer; "
            f"got {type(value).__name__}: {value!r}"
        )
    try:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            parsed = pd.to_datetime(
                str(int(value)), format="%Y%m%d", errors="coerce",
            )
        else:
            parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PaperTrackError(
            f"target 'as_of' is not a parseable date: {value!r}"
        ) from exc
    if not isinstance(parsed, pd.Timestamp) or pd.isna(parsed):
        raise PaperTrackError(f"target 'as_of' is not a parseable date: {value!r}")
    if parsed.tzinfo is not None:
        # The cached sessions are naive dates; a tz-aware input keeps its
        # wall-clock date instead of raising on the comparison.
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def load_target(path: Path) -> tuple[pd.Timestamp, pd.Series]:
    """Read ``{as_of, weights}`` and return (as_of, normalised weights)."""
    path = Path(path)
    if not path.is_file():
        raise PaperTrackError(f"target file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperTrackError(f"target file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or "weights" not in payload:
        raise PaperTrackError(
            "target file must be a JSON object with 'as_of' and 'weights'"
        )
    if "as_of" not in payload:
        raise PaperTrackError("target file is missing 'as_of'")
    as_of = _parse_target_date(payload["as_of"])

    raw = payload["weights"]
    if not isinstance(raw, dict) or not raw:
        raise PaperTrackError("target 'weights' must be a non-empty object")
    items: list[tuple[str, float]] = []
    for code, weight in raw.items():
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise PaperTrackError(
                f"target weight for {code!r} is not a number: {weight!r}"
            ) from exc
        if not math.isfinite(value):
            raise PaperTrackError(f"target weight for {code!r} is not finite")
        if value < 0:
            raise PaperTrackError(
                f"target weight for {code!r} is negative ({value}); "
                "long-only paper tracking"
            )
        items.append((_normalize_code(code), value))
    frame = pd.DataFrame(items, columns=["asset", "weight"])
    weights = frame.groupby("asset", sort=True)["weight"].sum()
    total = float(weights.sum())
    if total <= 0:
        raise PaperTrackError("target weights sum to zero")
    if total > 1.0 + 1e-6:
        raise PaperTrackError(
            f"target weights sum to {total:.6f} > 1; leverage is not supported"
        )
    weights = weights[weights > 0].astype(float)
    return as_of, weights


def _parse_date_level(dates: pd.Index) -> pd.DatetimeIndex:
    """Parse the cache date level (datetime / YYYY-MM-DD / YYYYMMDD)."""
    if isinstance(dates, pd.DatetimeIndex):
        return dates
    values = pd.Index(dates)
    if pd.api.types.is_integer_dtype(values.dtype):
        parsed = pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    if pd.isna(parsed).any():
        raise PaperTrackError(
            "cache panel has unparseable values in the 'date' level "
            "(expected datetime, YYYY-MM-DD strings, or YYYYMMDD integers)"
        )
    return pd.DatetimeIndex(parsed)


@dataclass
class Panels:
    """Wide (date x asset) panels needed for execution and accounting."""

    close: pd.DataFrame
    volume: pd.DataFrame
    pct_chg: pd.DataFrame  # percent, Baostock ex-rights basis

    def observation_mask(self) -> pd.DataFrame:
        """Sessions with a usable, traded observation (the stale complement).

        Requires a positive raw close (the stale/liquidation reference price),
        a finite ``pct_chg`` and a positive share volume -- the R9
        ``tradable_mask``. A NaN close with a valid ``pct_chg`` is still
        treated as stale: there is no price to mark or to liquidate at.
        """
        return (
            self.close.notna()
            & (self.close > 0)
            & self.pct_chg.notna()
            & self.volume.notna()
            & (self.volume > 0)
        )

    def returns(self) -> pd.DataFrame:
        """``pct_chg/100`` where the session observation is usable, else NaN."""
        return (self.pct_chg / 100.0).where(self.observation_mask())

    def volatility_returns(self) -> pd.DataFrame:
        """Returns for the cost model's volatility, with engine pipeline masking.

        ``data/pipeline.py`` masks a forward return when the asset is
        suspended on either end of it, which drops both the suspension day
        and the resume-day gap return. On this backward panel that is
        ``obs(d) & obs(d-1)``; the impact model then sees the same
        observations the engine feeds it (R9 parity).
        """
        mask = self.observation_mask()
        both = mask & mask.shift(1, fill_value=False)
        return (self.pct_chg / 100.0).where(both)


def load_cache(path: Path) -> Panels:
    """Load the cached daily panel into close / volume / pct_chg panels."""
    path = Path(path)
    if not path.is_file():
        raise PaperTrackError(f"cache panel not found: {path}")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise PaperTrackError(f"could not read cache panel {path}: {exc}") from exc
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise PaperTrackError(
            "cache panel must have a two-level (date, asset) MultiIndex; "
            f"got {type(frame.index).__name__}"
        )
    names = list(frame.index.names)
    if set(names) != {"date", "asset"}:
        raise PaperTrackError(
            f"cache panel index levels must be named 'date' and 'asset'; got {names}"
        )
    if names != ["date", "asset"]:
        frame = frame.reorder_levels(["date", "asset"]).sort_index()
    for column in ("close", "volume", "pct_chg"):
        if column not in frame.columns:
            raise PaperTrackError(
                f"cache panel is missing required column: {column!r}"
            )
    dates = _parse_date_level(frame.index.get_level_values("date"))
    assets = pd.Index(
        [_normalize_code(asset) for asset in frame.index.get_level_values("asset")],
        name="asset",
    )
    frame = frame.set_axis(
        pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])
    ).sort_index()
    if frame.index.duplicated().any():
        duplicated = int(frame.index.duplicated().sum())
        raise PaperTrackError(
            f"cache panel has {duplicated} duplicated (date, asset) rows"
        )
    close = frame["close"].unstack("asset").sort_index().astype(float)
    volume = frame["volume"].unstack("asset").reindex_like(close).astype(float)
    pct_chg = frame["pct_chg"].unstack("asset").reindex_like(close).astype(float)
    return Panels(close=close, volume=volume, pct_chg=pct_chg)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Simulation:
    """Everything the writer needs from one in-memory paper simulation."""

    rows: list[dict[str, Any]]
    warnings: list[str]
    as_of: pd.Timestamp
    execution_date: pd.Timestamp
    last_cache_date: pd.Timestamp
    n_target_assets: int
    n_purchased_assets: int
    cash_weight_at_build: float
    cost_mode: str
    max_stale_sessions: int


def _build_cost_model(cost_mode: str) -> CostModel:
    """The R9 cost configuration: statutory stamp schedule, impact or fixed."""
    return CostModel(
        commission=COMMISSION,
        stamp_tax=None,  # statutory date schedule (0.1% -> 0.05% on 2023-08-28)
        slippage=SLIPPAGE,
        slippage_model="impact" if cost_mode == "impact" else "fixed",
        impact_model="composite",
    )


def _vol_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """Causal daily volatility on the same basis as the engine.

    ``returns`` here is already backward-looking: row d is the ``pct_chg``
    return realized *into* close d. The engine's forward-return panel uses
    ``rolling(20).std().shift(1)``, which at the execution close e covers
    ``pct_chg`` dates up to and including e; on this backward panel the same
    observation set is the unshifted rolling std at e. Short histories fall
    back to 2% daily.
    """
    return (
        returns.rolling(VOL_LOOKBACK, min_periods=VOL_MIN_PERIODS)
        .std()
        .fillna(VOL_FALLBACK)
    )


def _execution_block_reason(close, volume, pct_chg, asset) -> str | None:
    """Why a target asset cannot be bought at the execution close (or None)."""
    if asset not in close.index:
        return "is not present in the cached panel"
    price = close.get(asset, np.nan)
    shares = volume.get(asset, np.nan)
    change = pct_chg.get(asset, np.nan)
    if not (np.isfinite(price) and price > 0):
        return "has no valid close on the execution date"
    if not (np.isfinite(change) and np.isfinite(shares) and shares > 0):
        return (
            "is not tradable on the execution date (missing pct_chg or "
            "zero/NaN volume)"
        )
    if change > LIMIT_UP_PCT:
        return (
            f"closed limit-up on the execution date (pct_chg {change:.4f}% > "
            f"{LIMIT_UP_PCT}%); the buy is blocked"
        )
    return None


def simulate(
    weights: pd.Series,
    as_of: pd.Timestamp,
    panels: Panels,
    capital: float,
    cost_mode: str,
    max_stale_sessions: int,
) -> Simulation:
    """Run the close-to-close paper book; raise on insufficient data."""
    close = panels.close
    volume = panels.volume
    pct_chg = panels.pct_chg
    dates = close.index
    if len(dates) == 0:
        raise PaperTrackError("cache panel contains no sessions")
    last_cache_date = pd.Timestamp(dates[-1])
    if as_of > last_cache_date:
        raise PaperTrackError(
            f"target as_of {as_of.date()} is beyond the cache end "
            f"({last_cache_date.date()}); no data to track"
        )
    position = int(dates.searchsorted(as_of, side="right"))
    if position >= len(dates):
        raise PaperTrackError(
            f"no trading day after as_of {as_of.date()} in the cache "
            f"(cache ends {last_cache_date.date()}); nothing to execute"
        )
    execution = pd.Timestamp(dates[position])

    warnings: list[str] = []
    if position == len(dates) - 1:
        warnings.append(
            f"{execution.date()}: the execution day is the last cached "
            "session; the ledger has no tracking sessions yet (build row only)"
        )

    exec_close = close.loc[execution]
    exec_volume = volume.loc[execution]
    exec_pct = pct_chg.loc[execution]
    target = weights.copy()

    buyable = pd.Series(False, index=target.index)
    for asset in target.index:
        reason = _execution_block_reason(exec_close, exec_volume, exec_pct, asset)
        if reason is None:
            buyable.loc[asset] = True
        else:
            warnings.append(
                f"{execution.date()}: target asset {asset} {reason}; its "
                "weight is held as cash and not bought"
            )
    buy_weights = target[buyable].astype(float)
    if buy_weights.empty:
        raise PaperTrackError(
            f"no target asset is tradable on the execution date "
            f"{execution.date()}; nothing to buy"
        )
    gross = float(buy_weights.sum())
    cash_weight = max(0.0, 1.0 - gross)
    cash_at_build = cash_weight

    # --- Build cost: CostModel on CNY notional, volume in CNY (R9 basis) ---
    # Buyability required a positive share volume and a positive raw close, so
    # the CNY traded value below is finite and positive by construction.
    notional = buy_weights * capital
    volume_cny: pd.Series | None = None
    vol_row: pd.Series | None = None
    if cost_mode == "impact":
        shares = exec_volume.reindex(buy_weights.index).astype(float)
        volume_cny = shares * exec_close.reindex(buy_weights.index).astype(float)
        if not np.isfinite(volume_cny.to_numpy()).all() or (volume_cny <= 0).any():
            raise PaperTrackError(
                f"{execution.date()}: non-positive CNY volume for a bought "
                "asset despite the tradability gate; refusing to price costs"
            )
        vol_row = _vol_panel(panels.volatility_returns()).loc[execution].reindex(
            buy_weights.index
        )
    cost_model = _build_cost_model(cost_mode)
    costs = cost_model.compute_costs(
        float(notional.sum()),
        buy_turnover=notional,
        sell_turnover=None,
        daily_volume=volume_cny,
        volatility=vol_row,
        date=execution,
        portfolio_value=capital,
    )
    cost_fraction = float(np.sum(np.asarray(costs, dtype=float)))
    if not math.isfinite(cost_fraction) or cost_fraction < 0:
        raise PaperTrackError(
            f"cost model produced an invalid cost fraction {cost_fraction!r}"
        )
    if cost_fraction >= 1.0:
        raise PaperTrackError(
            f"build cost {cost_fraction:.6f} of capital would exhaust the "
            f"{capital:.2f} CNY book; increase --capital or reduce the book"
        )
    cost_cny = cost_fraction * capital
    nav_prev = capital * (1.0 - cost_fraction)
    if not math.isfinite(nav_prev) or nav_prev <= 0:
        raise PaperTrackError("build produced a non-positive NAV; refusing to write")

    held = buy_weights.astype(float).copy()
    last_price = {asset: float(exec_close[asset]) for asset in held.index}
    stale_streak = {asset: 0 for asset in held.index}

    rows: list[dict[str, Any]] = [
        {
            "date": execution,
            "nav": nav_prev,
            "daily_return": -cost_fraction,
            "benchmark_daily_return": 0.0,
            "cum_return": nav_prev / capital - 1.0,
            "drawdown": nav_prev / capital - 1.0,
            "n_positions": int(len(held)),
            "stale_assets": 0,
            "costs_paid": cost_cny,
        }
    ]
    peak = capital
    benchmark = panels.returns().mean(axis=1, skipna=True)

    for date in dates[position + 1:]:
        date = pd.Timestamp(date)
        asset_returns: dict[str, float] = {}
        stale_today: list[str] = []
        for asset in held.index:
            change = pct_chg.at[date, asset]
            shares = volume.at[date, asset]
            price = close.at[date, asset]
            if (
                np.isfinite(change)
                and np.isfinite(shares) and shares > 0
                and np.isfinite(price) and price > 0
            ):
                asset_returns[asset] = float(change) / 100.0
                last_price[asset] = float(price)
                stale_streak[asset] = 0
            else:
                # Held at the last valid close: zero return, explicitly stale.
                asset_returns[asset] = 0.0
                stale_streak[asset] += 1
                stale_today.append(asset)

        r = pd.Series(asset_returns, dtype=float).reindex(held.index)
        portfolio_return = float((held * r).sum())
        if not math.isfinite(portfolio_return) or portfolio_return <= -1.0:
            raise PaperTrackError(
                f"{date.date()}: portfolio return {portfolio_return!r} is "
                "not a valid long-only daily return"
            )
        nav = nav_prev * (1.0 + portfolio_return)
        if not math.isfinite(nav) or nav <= 0:
            raise PaperTrackError(
                f"{date.date()}: NAV would be non-positive ({nav!r}); "
                "refusing to write"
            )
        held = (held * (1.0 + r)) / (1.0 + portfolio_return)
        cash_weight /= 1.0 + portfolio_return

        liquidated: list[str] = []
        if max_stale_sessions > 0:
            for asset in list(held.index):
                if stale_streak.get(asset, 0) >= max_stale_sessions:
                    frozen = last_price.pop(asset)
                    cash_weight += float(held.loc[asset])
                    held = held.drop(asset)
                    stale_streak.pop(asset, None)
                    liquidated.append(asset)
                    warnings.append(
                        f"{date.date()}: {asset} had no valid price for "
                        f"{max_stale_sessions} consecutive sessions; "
                        f"liquidated at its last close {frozen:.6f} and held "
                        "as cash (no trade cost: no tradable price exists)"
                    )

        bench_return = benchmark.get(date, 0.0)
        if not math.isfinite(bench_return):
            bench_return = 0.0
        peak = max(peak, nav)
        rows.append(
            {
                "date": date,
                "nav": nav,
                "daily_return": portfolio_return,
                "benchmark_daily_return": float(bench_return),
                "cum_return": nav / capital - 1.0,
                "drawdown": nav / peak - 1.0,
                "n_positions": int(len(held)),
                "stale_assets": int(len(stale_today)),
                "costs_paid": 0.0,
            }
        )
        nav_prev = nav

    return Simulation(
        rows=rows,
        warnings=warnings,
        as_of=as_of,
        execution_date=execution,
        last_cache_date=last_cache_date,
        n_target_assets=int(len(target)),
        n_purchased_assets=int(len(buy_weights)),
        cash_weight_at_build=float(cash_at_build),
        cost_mode=cost_mode,
        max_stale_sessions=max_stale_sessions,
    )


# ---------------------------------------------------------------------------
# Ledger writing (idempotent merge)
# ---------------------------------------------------------------------------

def _format_row(row: dict[str, Any]) -> str:
    """Format one ledger row with fixed precision for byte-stable output."""
    return ",".join(
        (
            pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            f"{row['nav']:.6f}",
            f"{row['daily_return']:.10f}",
            f"{row['benchmark_daily_return']:.10f}",
            f"{row['cum_return']:.10f}",
            f"{row['drawdown']:.10f}",
            str(int(row["n_positions"])),
            str(int(row["stale_assets"])),
            f"{row['costs_paid']:.6f}",
        )
    )


def _read_existing_nav(nav_path: Path) -> list[str]:
    """Read an existing ledger's data lines, validating its header and dates."""
    text = nav_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    expected = ",".join(NAV_COLUMNS)
    if not lines or lines[0].strip() != expected:
        raise PaperTrackError(
            f"{nav_path} exists but its header is not the paper_track ledger "
            f"header ({expected!r}); refusing to mix ledgers"
        )
    data = [line for line in lines[1:] if line.strip()]
    previous: pd.Timestamp | None = None
    for line in data:
        parts = line.split(",")
        if len(parts) != len(NAV_COLUMNS):
            raise PaperTrackError(
                f"{nav_path} has a malformed row ({len(parts)} fields): {line!r}"
            )
        date = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(date):
            raise PaperTrackError(f"{nav_path} has an unparseable date: {parts[0]!r}")
        date = pd.Timestamp(date)
        if previous is not None and date <= previous:
            raise PaperTrackError(
                f"{nav_path} is not strictly increasing by date at {parts[0]!r}; "
                "refusing to append"
            )
        previous = date
    return data


def _check_existing_summary(
    summary_path: Path, target_sha256: str, capital: float, cost_mode: str,
    max_stale_sessions: int,
) -> None:
    """Refuse to append to a ledger produced by a different configuration."""
    if not summary_path.is_file():
        return
    try:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperTrackError(
            f"{summary_path} exists but is not valid JSON; refusing to append"
        ) from exc
    checks = (
        ("ledger_version", LEDGER_VERSION, "ledger version (return basis)"),
        ("target_sha256", target_sha256, "target file"),
        ("capital", float(capital), "capital"),
        ("cost_mode", cost_mode, "cost mode"),
        ("max_stale_sessions", int(max_stale_sessions), "stale policy"),
    )
    for key, expected, label in checks:
        if key not in existing:
            raise PaperTrackError(
                f"{summary_path} has no {key!r}; cannot verify it is the same "
                f"{label}; refusing to append"
            )
        if existing[key] != expected:
            raise PaperTrackError(
                f"{summary_path} belongs to a different {label} "
                f"({existing[key]!r} != {expected!r}); use a new --out-dir"
            )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _merged_series(lines: list[str]) -> dict[str, Any]:
    """Parse ledger lines into per-column arrays for the summary metrics."""
    dates: list[pd.Timestamp] = []
    nav: list[float] = []
    daily: list[float] = []
    bench: list[float] = []
    costs: list[float] = []
    stale: list[int] = []
    for line in lines:
        parts = line.split(",")
        dates.append(pd.Timestamp(parts[0]))
        nav.append(float(parts[1]))
        daily.append(float(parts[2]))
        bench.append(float(parts[3]))
        stale.append(int(parts[7]))
        costs.append(float(parts[8]))
    index = pd.DatetimeIndex(dates)
    return {
        "dates": index,
        "nav": np.asarray(nav, dtype=float),
        "daily": pd.Series(daily, index=index),
        "bench": pd.Series(bench, index=index),
        "costs": np.asarray(costs, dtype=float),
        "stale": np.asarray(stale, dtype=int),
    }


def _finite(value: Any) -> float | None:
    """JSON-safe float: non-finite numpy/pandas scalars become ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _drawdown_stats(
    nav: np.ndarray, dates: pd.DatetimeIndex, initial: float,
) -> tuple[float, str | None, str | None]:
    """Max drawdown on the NAV path itself (dates are close-of-day labels).

    The platform's ``max_drawdown`` helper treats a return series as labelled
    by period *start*; a ledger row is labelled by its close, so the peak and
    trough dates are derived directly from the wealth path here.
    """
    wealth = np.concatenate(([float(initial)], np.asarray(nav, dtype=float)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    trough_idx = int(np.argmin(drawdowns))
    worst = float(drawdowns[trough_idx])
    if trough_idx == 0:
        return 0.0, None, None
    peak_idx = int(np.argmax(peaks[: trough_idx + 1]))
    peak_date = dates[peak_idx - 1] if peak_idx >= 1 else None
    trough_date = dates[trough_idx - 1]
    return (
        worst,
        peak_date.strftime("%Y-%m-%d") if peak_date is not None else None,
        trough_date.strftime("%Y-%m-%d"),
    )


def _metrics(
    daily: pd.Series, nav: np.ndarray, dates: pd.DatetimeIndex, initial: float,
) -> dict[str, Any]:
    """Annualized return / vol / Sharpe and NAV-path max drawdown."""
    mdd, peak_date, trough_date = _drawdown_stats(nav, dates, initial)
    metrics: dict[str, Any] = {
        "max_drawdown": _finite(mdd),
        "max_drawdown_peak": peak_date,
        "max_drawdown_trough": trough_date,
    }
    if len(daily) < 2:
        metrics.update(
            annualized_return=None, annualized_volatility=None, sharpe_ratio=None,
        )
        return metrics
    metrics.update(
        annualized_return=_finite(annualized_return(daily, PERIODS_PER_YEAR)),
        annualized_volatility=_finite(
            annualized_volatility(daily, PERIODS_PER_YEAR)
        ),
        sharpe_ratio=_finite(sharpe_ratio(daily, RISK_FREE_ANNUAL, PERIODS_PER_YEAR)),
    )
    return metrics


def build_summary(
    simulation: Simulation,
    lines: list[str],
    capital: float,
    target_file: str,
    target_sha256: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Deterministic summary of the merged ledger (no timestamps inside)."""
    series = _merged_series(lines)
    daily = series["daily"]
    bench = series["bench"]
    nav = series["nav"]
    dates = series["dates"]
    total_return = float(nav[-1] / capital - 1.0)
    bench_nav = np.cumprod(1.0 + bench.to_numpy(dtype=float))
    benchmark_total = float(bench_nav[-1] - 1.0)
    strategy_metrics = _metrics(daily, nav, dates, capital)
    benchmark_metrics = _metrics(bench, bench_nav, dates, 1.0)
    if len(daily) >= 2:
        tracking_error_value = _finite(
            tracking_error(daily, bench, PERIODS_PER_YEAR)
        )
    else:
        tracking_error_value = None
    total_costs = float(series["costs"].sum())
    stale_days = int((series["stale"] > 0).sum())
    return {
        "ledger_version": LEDGER_VERSION,
        "as_of": simulation.as_of.strftime("%Y-%m-%d"),
        "execution_date": simulation.execution_date.strftime("%Y-%m-%d"),
        "first_ledger_date": dates[0].strftime("%Y-%m-%d"),
        "last_ledger_date": dates[-1].strftime("%Y-%m-%d"),
        "cache_last_date": simulation.last_cache_date.strftime("%Y-%m-%d"),
        "n_sessions": int(len(daily)),
        "capital": float(capital),
        "cost_mode": simulation.cost_mode,
        "max_stale_sessions": int(simulation.max_stale_sessions),
        "stale_policy": STALE_POLICY,
        "return_basis": RETURN_BASIS,
        "volatility_basis": VOLATILITY_BASIS,
        "tradable_gate": TRADABLE_GATE,
        "benchmark_definition": BENCHMARK_DEFINITION,
        "risk_free_rate": RISK_FREE_ANNUAL,
        "annualization_periods_per_year": PERIODS_PER_YEAR,
        "target_file": target_file,
        "target_sha256": target_sha256,
        "n_target_assets": simulation.n_target_assets,
        "n_purchased_assets": simulation.n_purchased_assets,
        "cash_weight_at_build": _finite(simulation.cash_weight_at_build),
        "since_inception": {
            "final_nav": float(nav[-1]),
            "total_return": total_return,
            "annualized_return": strategy_metrics["annualized_return"],
            "annualized_volatility": strategy_metrics["annualized_volatility"],
            "sharpe_ratio": strategy_metrics["sharpe_ratio"],
            "max_drawdown": strategy_metrics["max_drawdown"],
            "max_drawdown_peak": strategy_metrics["max_drawdown_peak"],
            "max_drawdown_trough": strategy_metrics["max_drawdown_trough"],
            "n_stale_days": stale_days,
        },
        "benchmark": {
            "definition": BENCHMARK_DEFINITION,
            "total_return": benchmark_total,
            "annualized_return": benchmark_metrics["annualized_return"],
            "annualized_volatility": benchmark_metrics["annualized_volatility"],
            "sharpe_ratio": benchmark_metrics["sharpe_ratio"],
            "max_drawdown": benchmark_metrics["max_drawdown"],
        },
        "tracking": {
            "difference": total_return - benchmark_total,
            "tracking_error": tracking_error_value,
        },
        "costs": {
            "total_cny": total_costs,
            "pct_of_capital": total_costs / capital if capital else None,
        },
        "warnings": list(warnings),
        "disclaimer": DISCLAIMER,
    }


def run(
    target_path: Path,
    cache_path: Path,
    out_dir: Path,
    capital: float = DEFAULT_CAPITAL,
    cost_mode: str = "impact",
    max_stale_sessions: int = DEFAULT_MAX_STALE_SESSIONS,
) -> tuple[dict[str, Any], int, Path]:
    """Execute one paper-tracking run; returns (summary, n_new_rows, nav_path)."""
    if not math.isfinite(capital) or capital <= 0:
        raise PaperTrackError(f"capital must be a positive number; got {capital!r}")
    if max_stale_sessions < 0:
        raise PaperTrackError("--max-stale-sessions must be >= 0 (0 disables)")
    target_path = Path(target_path)
    cache_path = Path(cache_path)
    out_dir = Path(out_dir)

    as_of, weights = load_target(target_path)
    panels = load_cache(cache_path)
    target_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    simulation = simulate(
        weights, as_of, panels, capital, cost_mode, max_stale_sessions,
    )

    nav_path = out_dir / "nav.csv"
    summary_path = out_dir / "summary.json"
    if nav_path.is_file() and not summary_path.is_file():
        # The writer always emits summary.json immediately after nav.csv, so
        # this is a broken/partial ledger with unknown provenance.
        raise PaperTrackError(
            f"{nav_path} exists without {summary_path}; cannot verify which "
            "target/config produced it -- refusing to append. Move the orphan "
            "ledger aside or use a new --out-dir"
        )
    existing_lines: list[str] = []
    warnings = list(simulation.warnings)
    if summary_path.is_file():
        _check_existing_summary(
            summary_path, target_sha256, capital, cost_mode, max_stale_sessions,
        )
    if nav_path.is_file():
        existing_lines = _read_existing_nav(nav_path)

    existing_dates = [
        pd.Timestamp(line.split(",", 1)[0]) for line in existing_lines
    ]
    computed_by_date = {row["date"]: row for row in simulation.rows}
    if existing_dates:
        max_diff = 0.0
        for line, date in zip(existing_lines, existing_dates, strict=True):
            row = computed_by_date.get(date)
            if row is None:
                continue
            file_nav = float(line.split(",")[1])
            max_diff = max(max_diff, abs(file_nav - row["nav"]))
        tolerance = max(REVISION_NAV_TOLERANCE, 1e-9 * capital)
        if max_diff > tolerance:
            warnings.append(
                "recomputation differs from the stored ledger on already "
                f"written dates (max |delta NAV| = {max_diff:.6f}); the stored "
                "rows were kept unchanged"
            )

    new_lines: list[str] = []
    last_existing = existing_dates[-1] if existing_dates else None
    existing_set = set(existing_dates)
    for row in simulation.rows:
        date = row["date"]
        if date in existing_set:
            continue
        if last_existing is not None and date <= last_existing:
            continue
        new_lines.append(_format_row(row))

    merged_lines = existing_lines + new_lines
    if not merged_lines:
        raise PaperTrackError("internal error: empty ledger after simulation")

    summary = build_summary(
        simulation,
        merged_lines,
        capital,
        target_path.as_posix(),
        target_sha256,
        warnings,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if new_lines:
        text = ",".join(NAV_COLUMNS) + "\n" + "\n".join(merged_lines) + "\n"
        _atomic_write_text(nav_path, text)
    summary_text = json.dumps(
        summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    _atomic_write_text(summary_path, summary_text)
    return summary, len(new_lines), nav_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward paper tracking: fixed target book -> idempotent daily "
            "NAV ledger + benchmark (read-only; no broker)."
        )
    )
    parser.add_argument("--target", required=True, help="target weights JSON")
    parser.add_argument("--cache", required=True, help="cached daily parquet")
    parser.add_argument("--out-dir", required=True, help="ledger output directory")
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL,
        help=f"initial capital in CNY (default {DEFAULT_CAPITAL:.0f})",
    )
    parser.add_argument(
        "--cost-mode", choices=("impact", "fixed"), default="impact",
        help="CostModel slippage mode (default impact, R9 basis)",
    )
    parser.add_argument(
        "--max-stale-sessions", type=int, default=DEFAULT_MAX_STALE_SESSIONS,
        help=(
            "consecutive stale sessions before a held asset is liquidated to "
            f"cash (default {DEFAULT_MAX_STALE_SESSIONS}; 0 disables)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary, n_new, nav_path = run(
            Path(args.target),
            Path(args.cache),
            Path(args.out_dir),
            capital=float(args.capital),
            cost_mode=args.cost_mode,
            max_stale_sessions=int(args.max_stale_sessions),
        )
    except PaperTrackError as exc:
        print(f"paper_track: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"paper_track: error: {exc}", file=sys.stderr)
        return 2

    since = summary["since_inception"]
    print(
        "paper_track: executed {execution} on {as_of} target; ledger "
        "{first}..{last} ({sessions} sessions, +{new} new rows); "
        "final NAV {nav:.2f} ({ret:+.2%}); costs {costs:.2f} CNY".format(
            execution=summary["execution_date"],
            as_of=summary["as_of"],
            first=summary["first_ledger_date"],
            last=summary["last_ledger_date"],
            sessions=summary["n_sessions"],
            new=n_new,
            nav=since["final_nav"],
            ret=since["total_return"],
            costs=summary["costs"]["total_cny"],
        )
    )
    print(f"paper_track: wrote {nav_path} and {nav_path.parent / 'summary.json'}")
    if summary["warnings"]:
        print(
            f"paper_track: {len(summary['warnings'])} warning(s) recorded in "
            "summary.json",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
