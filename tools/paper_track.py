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
  dividend or split is not booked as a loss (raw close ratios would be).
* **Raw price basis (R32).** Price-level decisions -- the CNY traded value
  fed to the cost model (``volume_shares * raw_price``), the stale/tradability
  gate, and the stale-liquidation reference -- use a *raw CNY* price, never an
  adjusted one. ``--raw-price-mode close`` keeps the legacy assumption that
  the cached ``close`` is already the raw CNY close. ``--raw-price-mode
  amount_over_volume`` uses ``amount / volume`` (NaN when ``amount`` is
  missing or ``volume`` is not positive), the day's CNY VWAP proxy (the
  wide-pool research convention where ``volume * amount/volume == amount``
  exactly). ``--raw-price-mode auto``
  (the default) samples up to 60 of the most recent sessions with usable
  cells and selects ``amount_over_volume`` when the sample's median
  ``|close*volume - amount| / amount`` exceeds 1% -- an adjusted close breaks
  ``amount == close * volume`` -- otherwise it keeps ``close``; the selected
  mode and the sampled statistic are printed. ``auto`` falls back to
  ``close`` when the cache has no ``amount`` column, while an explicit
  ``amount_over_volume`` against such a cache exits non-zero. ``volume``
  remains a required column in every mode (the R9 tradability gate and the
  CNY traded-value conversion depend on it). The basis is panel-level: a
  single cache must use one adjustment convention (a mixed panel would let
  the majority dilute the detection). ``summary.json`` records
  ``raw_price_mode`` only for the ``amount_over_volume`` basis so that
  legacy/raw-close runs keep their pre-R32 bytes. Returns never use the raw
  price: they stay ``pct_chg / 100``.
* **Tradability gate (R9/R19a).** A target asset is bought on the execution
  close only when its raw price is ``> 0``, ``pct_chg`` is finite,
  ``volume > 0``
  (the R9 ``tradable_mask``), and did not close limit-up
  (``pct_chg > 9.8%``, the platform's limit-up approximation). Blocked or
  untradable weight stays in cash and a warning is recorded -- the tool never
  assumes a fill that could not have happened.
* **Costs (R9 basis).** On the execution day the tool buys the target book
  from cash and charges :class:`~quant_platform.backtest.cost_model.CostModel`
  exactly the way the engine does: CNY buy notionals
  (``weight * capital``), ``daily_volume`` in CNY as
  ``volume_shares * raw_price`` (the resolved raw price basis above; on a
  raw-close cache ``amount`` matches ``close * volume``), the date-dependent
  statutory stamp tax (no sells on a build, so the stamp tax is zero), the per-order CNY 5
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
* **Stale prices.** A session is *stale* for a held asset when its raw price
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
* **Scope (single-target mode).** With ``--target`` the tool builds the book
  once and then lets it drift -- there is no scheduled rebalancing after the
  build. To track a different target, use a different ``--out-dir``; an
  existing ledger refuses to mix targets, capital, cost mode, or stale policy.
  Positions are weight-level (the platform backtest does not round to
  100-share lots either).

Rolling mode (``--targets-dir`` / ``--targets``)
------------------------------------------------
Multiple ``daily_signal`` outputs are applied as a rebalance schedule:
targets are sorted by ``as_of``; the first one builds the book at the close
of the next exchange trading session strictly after its ``as_of``, and every
later target rebalances on its own next exchange session. If that expected
session is absent while later cached dates exist, the cache is incomplete and
the run fails rather than executing late. A target whose expected session has
not arrived yet is pending. On a rebalance session the tool marks the old
book to that close first, then trades **sells first, buys second** at the same
close.

Trading is cash-constrained: buys can spend only the cash left after the
sells plus the existing cash, *and* they must leave enough to pay the
transaction costs. A gross-100% target therefore executes with buys scaled
down by the cost fraction and the book still ends fully invested (cash 0);
on a uniform build (nothing kept from a previous book) the scaled buys
reproduce the target weights exactly, while on a rebalance the positions
kept from the old book retain their drifted weights. If the constraint
binds harder -- for example a limit-down position that cannot be sold --
buys are scaled proportionally and a warning records the realised scale.

Tradability at a rebalance uses the R9/R19a gates on the execution close:
an asset is sellable when it has a positive raw price, finite ``pct_chg``,
positive volume and did not close limit-down (``pct_chg < -9.8%``); buyable
when the same observation exists and it did not close limit-up
(``pct_chg > 9.8%``). A blocked buy leaves that weight in cash; a blocked
sell keeps the existing position (the reduction is superseded), with a
warning naming the asset. Held assets with no valid observation are carried
stale under the same policy as the single-target mode (last raw price, zero
return, optional liquidation after ``--max-stale-sessions``).

Costs are charged once per rebalance on the executed CNY notionals, using
the same CostModel configuration as the single-target mode: per-order
commission with the CNY 5 floor, both-sides transfer fee, the date-dependent
sell-side stamp tax, and -- with ``--cost-mode impact`` -- the square-root
impact rule whose daily volume is ``shares * raw_price`` (the R9 basis, on
the raw-price basis selected above).
Costs are deducted from cash at the rebalance close; the row's return is
the drifted book return minus that cost fraction.

A trailing target with no cached session after its ``as_of`` (only possible
when ``as_of`` equals the cache's last date) is *pending*: it is listed in
``summary.json`` and left for the next run once the cache extends; the
ledger still runs to the cache's last session (the pending target is simply
not executed). A target whose ``as_of`` is
beyond the cache end is an error, as are duplicate ``as_of`` values,
duplicate execution sessions, unknown target assets, negative weights and
weight sums above 1.

Rolling ``nav.csv`` columns:
``date, nav, daily_return, benchmark_daily_return, cum_return, drawdown,
n_positions, turnover, costs_paid, stale_assets`` (``turnover`` is the
realised one-sided turnover of an executed rebalance -- half of buys plus
sells over the pre-cost portfolio value -- and 0 on non-rebalance sessions).
The rolling ledger is rebuilt from scratch on every run, so the same targets
and cache produce byte-identical files; an existing non-rolling ledger in
the out-dir is refused rather than mixed. ``summary.json`` persists every
target's file, ``as_of`` and SHA-256. An existing target schedule must remain
an unchanged hash prefix: mutation, removal and backdated insertion are
refused without touching prior outputs, while strictly later targets may be
appended. Different capital, cost mode or stale policy is also refused.

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

    python tools/paper_track.py \
      --targets-dir .scratch/orchestrate/roll_targets \
      --cache .cache/baostock/daily_2018-01-01_2025-12-31_500.parquet \
      --out-dir docs/research/paper_track/roll_2020_2025 \
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
from quant_platform.utils.exchange_calendar import is_trading_day

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
ROLL_MODE = "rolling"
ROLL_LEDGER_VERSION = 1
ROLL_NAV_COLUMNS = (
    "date",
    "nav",
    "daily_return",
    "benchmark_daily_return",
    "cum_return",
    "drawdown",
    "n_positions",
    "turnover",
    "costs_paid",
    "stale_assets",
)
ROLL_REBALANCE_EPS = 1e-9
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
RAW_PRICE_MODE_AUTO = "auto"
RAW_PRICE_MODE_CLOSE = "close"
RAW_PRICE_MODE_AMOUNT = "amount_over_volume"
RAW_PRICE_MODES = (
    RAW_PRICE_MODE_AUTO, RAW_PRICE_MODE_CLOSE, RAW_PRICE_MODE_AMOUNT,
)
RAW_PRICE_RELATIVE_TOLERANCE = 0.01
RAW_PRICE_SAMPLE_SESSIONS = 60
RAW_PRICE_SAMPLE_MAX_CELLS = 20_000
RETURN_BASIS = (
    "pct_chg/100 (Baostock ex-rights preclose); raw close is used only for "
    "volume x price cost conversion and the stale liquidation reference"
)
RETURN_BASIS_AMOUNT = (
    "pct_chg/100 (Baostock ex-rights preclose); raw price = amount/volume "
    "(CNY VWAP proxy, volume > 0) is used only for volume x price cost "
    "conversion and the stale liquidation reference"
)
TRADABLE_GATE = (
    "execution buys require close > 0, finite pct_chg, volume > 0 (R9 "
    "tradable_mask) and pct_chg <= 9.8% (R19a limit-up approximation); "
    "blocked weight stays in cash with a warning"
)
TRADABLE_GATE_AMOUNT = (
    "execution buys require raw price amount/volume > 0, finite pct_chg, "
    "volume > 0 (R9 tradable_mask) and pct_chg <= 9.8% (R19a limit-up "
    "approximation); blocked weight stays in cash with a warning"
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
STALE_POLICY_AMOUNT = (
    "a session is stale when the raw price amount/volume, pct_chg or volume "
    "is missing/invalid or the volume is zero; hold at the last valid raw "
    "price (return 0, flagged stale); when the price returns the resume-day "
    "pct_chg carries the gap return; after max_stale_sessions consecutive "
    "stale sessions, liquidate at the last valid raw price with no extra "
    "trade cost (no tradable price exists) and hold the proceeds as cash"
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
    """Wide (date x asset) panels needed for execution and accounting.

    ``close`` holds the *resolved raw price*: the cached ``close`` on a
    raw-close cache (``--raw-price-mode close``) or ``amount / volume`` on an
    adjusted cache (``--raw-price-mode amount_over_volume``). Every
    price-level decision (CNY cost conversion, tradability, stale marking,
    the liquidation reference) reads it; returns are always ``pct_chg / 100``.
    """

    close: pd.DataFrame
    volume: pd.DataFrame
    pct_chg: pd.DataFrame  # percent, Baostock ex-rights basis
    raw_price_mode: str = RAW_PRICE_MODE_CLOSE
    raw_price_detection: dict[str, Any] | None = None

    def observation_mask(self) -> pd.DataFrame:
        """Sessions with a usable, traded observation (the stale complement).

        Requires a positive raw price (the stale/liquidation reference), a
        finite ``pct_chg`` and a positive share volume -- the R9
        ``tradable_mask``. A NaN raw price with a valid ``pct_chg`` is still
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


def _amount_over_volume_price(
    amount: pd.DataFrame, volume: pd.DataFrame,
) -> pd.DataFrame:
    """Raw CNY price proxy ``amount / volume`` (NaN when volume <= 0).

    ``amount`` is the day's traded value in CNY and ``volume`` the share
    count, so the ratio is the day's VWAP in CNY -- a raw-price estimate that
    does not depend on the cache's adjustment convention. Zero, missing or
    non-finite volume and missing/non-finite amount yield NaN; callers treat
    a NaN raw price as a stale session.
    """
    sane_volume = volume.where(np.isfinite(volume) & (volume > 0))
    price = amount / sane_volume
    return price.where(np.isfinite(price))


def _detect_raw_price_basis(
    close: pd.DataFrame, volume: pd.DataFrame, amount: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    """Choose between the cached ``close`` and ``amount/volume`` (auto mode).

    Samples up to ``RAW_PRICE_SAMPLE_SESSIONS`` of the most recent sessions
    that contain usable cells (walking backward from the cache end), capped
    at ``RAW_PRICE_SAMPLE_MAX_CELLS`` cells. A cell is usable when ``close``,
    ``volume`` and ``amount`` are finite and positive. The statistic is the
    sample median of ``|close*volume - amount| / amount``.

    Above ``RAW_PRICE_RELATIVE_TOLERANCE`` (1%) the cached close cannot be
    the raw CNY close -- an adjusted (hfq) close is a multiple of the raw
    price, so ``amount == close * volume`` breaks -- and
    ``amount_over_volume`` is selected. Otherwise (including when no usable
    cell exists) the legacy ``close`` basis is kept.
    """
    valid = (
        close.notna() & np.isfinite(close) & (close > 0)
        & volume.notna() & np.isfinite(volume) & (volume > 0)
        & amount.notna() & np.isfinite(amount) & (amount > 0)
    )
    counts = valid.sum(axis=1)
    selected: list[pd.Timestamp] = []
    n_cells = 0
    for date in reversed(counts.index):
        n = int(counts.loc[date])
        if n == 0:
            continue
        if selected and n_cells + n > RAW_PRICE_SAMPLE_MAX_CELLS:
            break
        selected.append(pd.Timestamp(date))
        n_cells += n
        if len(selected) >= RAW_PRICE_SAMPLE_SESSIONS:
            break
    detection: dict[str, Any] = {
        "n_cells": int(n_cells),
        "n_sessions": int(len(selected)),
        "median_relative_error": None,
    }
    if not selected:
        return RAW_PRICE_MODE_CLOSE, detection
    relative = ((close * volume - amount).abs() / amount).loc[selected]
    values = relative.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return RAW_PRICE_MODE_CLOSE, detection
    median = float(np.median(values))
    detection["median_relative_error"] = median
    if median > RAW_PRICE_RELATIVE_TOLERANCE:
        return RAW_PRICE_MODE_AMOUNT, detection
    return RAW_PRICE_MODE_CLOSE, detection


def _detection_note(detection: dict[str, Any]) -> str:
    """Human-readable auto-detection evidence for the stdout log line."""
    median = detection.get("median_relative_error")
    if median is None:
        return (
            f"(no usable amount/close*volume cells in the last "
            f"{detection['n_sessions']} session(s); falling back to close)"
        )
    return (
        f"(median |close*volume - amount| / amount = {median:.4%} over "
        f"{detection['n_cells']} cells in {detection['n_sessions']} sessions; "
        f"switch threshold {RAW_PRICE_RELATIVE_TOLERANCE:.2%})"
    )


def load_cache(
    path: Path, raw_price_mode: str = RAW_PRICE_MODE_AUTO,
) -> Panels:
    """Load the cached daily panel and resolve its raw-price basis.

    ``raw_price_mode`` is ``auto`` (sample and decide), ``close`` (legacy:
    the cached close is already the raw CNY price) or ``amount_over_volume``
    (raw price = ``amount / volume``; a cache without an ``amount`` column is
    a non-zero-exit error). ``auto`` falls back to ``close`` when ``amount``
    is absent. ``volume`` is required in every mode: the R9 tradability gate
    and the CNY traded-value conversion depend on it.
    """
    if raw_price_mode not in RAW_PRICE_MODES:
        raise PaperTrackError(
            f"unknown raw price mode {raw_price_mode!r}; expected one of "
            f"{', '.join(RAW_PRICE_MODES)}"
        )
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
    has_amount = "amount" in frame.columns
    if raw_price_mode == RAW_PRICE_MODE_AMOUNT and not has_amount:
        raise PaperTrackError(
            "cache panel is missing required column: 'amount' (required by "
            "--raw-price-mode amount_over_volume)"
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
    try:
        close = frame["close"].unstack("asset").sort_index().astype(float)
        volume = frame["volume"].unstack("asset").reindex_like(close).astype(float)
        pct_chg = frame["pct_chg"].unstack("asset").reindex_like(close).astype(float)
        amount = (
            frame["amount"].unstack("asset").reindex_like(close).astype(float)
            if has_amount else None
        )
    except (TypeError, ValueError) as exc:
        raise PaperTrackError(
            f"cache panel has non-numeric price/volume values: {exc}"
        ) from exc
    detection: dict[str, Any] | None = None
    if raw_price_mode == RAW_PRICE_MODE_CLOSE:
        resolved_mode = RAW_PRICE_MODE_CLOSE
    elif raw_price_mode == RAW_PRICE_MODE_AMOUNT:
        resolved_mode = RAW_PRICE_MODE_AMOUNT
    elif amount is None:
        resolved_mode = RAW_PRICE_MODE_CLOSE
        print(
            "paper_track: raw-price mode auto -> close (the cache has no "
            "'amount' column; assuming close is the raw CNY price)",
        )
    else:
        resolved_mode, detection = _detect_raw_price_basis(close, volume, amount)
        print(
            f"paper_track: raw-price mode auto -> {resolved_mode} "
            f"{_detection_note(detection)}"
        )
    price = (
        _amount_over_volume_price(amount, volume)
        if resolved_mode == RAW_PRICE_MODE_AMOUNT else close
    )
    return Panels(
        close=price,
        volume=volume,
        pct_chg=pct_chg,
        raw_price_mode=resolved_mode,
        raw_price_detection=detection,
    )


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
    raw_price_mode: str = RAW_PRICE_MODE_CLOSE


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


def _execution_block_reason(
    close, volume, pct_chg, asset, price_label: str = "close",
) -> str | None:
    """Why a target asset cannot be bought at the execution close (or None)."""
    if asset not in close.index:
        return "is not present in the cached panel"
    price = close.get(asset, np.nan)
    shares = volume.get(asset, np.nan)
    change = pct_chg.get(asset, np.nan)
    if not (np.isfinite(price) and price > 0):
        return f"has no valid {price_label} on the execution date"
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
    price_label = (
        "close" if panels.raw_price_mode == RAW_PRICE_MODE_CLOSE
        else "raw price"
    )
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
        reason = _execution_block_reason(
            exec_close, exec_volume, exec_pct, asset, price_label,
        )
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
    # Buyability required a positive share volume and a positive raw price, so
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
                        f"liquidated at its last {price_label} {frozen:.6f} "
                        "and held as cash (no trade cost: no tradable price "
                        "exists)"
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
        raw_price_mode=panels.raw_price_mode,
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
    max_stale_sessions: int, raw_price_mode: str = RAW_PRICE_MODE_CLOSE,
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
        # Pre-R32 ledgers have no key and were all on the legacy close basis.
        ("raw_price_mode", raw_price_mode, "raw price mode"),
    )
    for key, expected, label in checks:
        if key == "raw_price_mode":
            existing_value = existing.get(key, RAW_PRICE_MODE_CLOSE)
        elif key not in existing:
            raise PaperTrackError(
                f"{summary_path} has no {key!r}; cannot verify it is the same "
                f"{label}; refusing to append"
            )
        else:
            existing_value = existing[key]
        if existing_value != expected:
            hint = (
                " or pass the matching --raw-price-mode"
                if key == "raw_price_mode" else ""
            )
            raise PaperTrackError(
                f"{summary_path} belongs to a different {label} "
                f"({existing_value!r} != {expected!r}); use a new --out-dir"
                f"{hint}"
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
    amount_basis = simulation.raw_price_mode == RAW_PRICE_MODE_AMOUNT
    summary: dict[str, Any] = {
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
        "stale_policy": STALE_POLICY_AMOUNT if amount_basis else STALE_POLICY,
        "return_basis": RETURN_BASIS_AMOUNT if amount_basis else RETURN_BASIS,
        "volatility_basis": VOLATILITY_BASIS,
        "tradable_gate": (
            TRADABLE_GATE_AMOUNT if amount_basis else TRADABLE_GATE
        ),
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
    if amount_basis:
        # Legacy raw-close ledgers keep their pre-R32 bytes; the explicit
        # basis is only recorded when it is not the legacy assumption.
        summary["raw_price_mode"] = simulation.raw_price_mode
    return summary


def run(
    target_path: Path,
    cache_path: Path,
    out_dir: Path,
    capital: float = DEFAULT_CAPITAL,
    cost_mode: str = "impact",
    max_stale_sessions: int = DEFAULT_MAX_STALE_SESSIONS,
    raw_price_mode: str = RAW_PRICE_MODE_AUTO,
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
    panels = load_cache(cache_path, raw_price_mode)
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
            panels.raw_price_mode,
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
# Rolling mode: a schedule of targets -> one continuous ledger
# ---------------------------------------------------------------------------

@dataclass
class Target:
    """One parsed target file with its provenance path."""

    path: Path
    as_of: pd.Timestamp
    weights: pd.Series


@dataclass
class Rebalance:
    """One executed rebalance (the first one is the build)."""

    as_of: pd.Timestamp
    execution_date: pd.Timestamp
    turnover: float
    costs_cny: float
    buy_cny: float
    sell_cny: float
    buy_scale: float
    n_buys: int
    n_sells: int
    blocked_buys: list[str]
    blocked_sells: list[str]


@dataclass
class RollSimulation:
    """Everything the rolling writer needs from one in-memory simulation."""

    rows: list[dict[str, Any]]
    warnings: list[str]
    rebalances: list[Rebalance]
    pending: list[tuple[Path, pd.Timestamp]]
    targets: list[Target]
    last_cache_date: pd.Timestamp
    cost_mode: str
    max_stale_sessions: int
    raw_price_mode: str = RAW_PRICE_MODE_CLOSE


def load_targets(paths: list[Path]) -> list[Target]:
    """Load and validate a target schedule, sorted by ``as_of``."""
    if not paths:
        raise PaperTrackError("no target files supplied")
    targets: list[Target] = []
    seen: dict[pd.Timestamp, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            as_of, weights = load_target(path)
        except PaperTrackError as exc:
            # Name the offending file: load_target validates structure/weights
            # but does not know the caller's provenance.
            raise PaperTrackError(f"{path}: {exc}") from exc
        if as_of in seen:
            raise PaperTrackError(
                f"duplicate target as_of {as_of.date()} in {seen[as_of]} and "
                f"{path}; one rebalance per as_of"
            )
        seen[as_of] = path
        targets.append(Target(path=path, as_of=as_of, weights=weights))
    targets.sort(key=lambda target: target.as_of)
    return targets


def target_paths_from_dir(directory: Path) -> list[Path]:
    """All ``*.json`` files directly inside ``directory`` (sorted by name)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PaperTrackError(f"targets directory not found: {directory}")
    paths = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not paths:
        raise PaperTrackError(f"no *.json target files in {directory}")
    return paths


def target_paths_from_list(raw: str) -> list[Path]:
    """Comma-separated target files -> list of existing paths."""
    paths = [Path(piece.strip()) for piece in str(raw).split(",") if piece.strip()]
    if not paths:
        raise PaperTrackError("--targets produced an empty file list")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PaperTrackError(f"target file(s) not found: {', '.join(missing)}")
    return paths


def _next_exchange_session(as_of: pd.Timestamp) -> pd.Timestamp:
    """First exchange trading session strictly after ``as_of``."""
    session = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    while not is_trading_day(session.date()):
        session += pd.Timedelta(days=1)
    return session


def _observation_ok(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    pct_chg: pd.DataFrame,
    date: pd.Timestamp,
    asset: str,
) -> bool:
    """A usable traded observation on ``date`` (the R9 ``tradable_mask``)."""
    price = close.at[date, asset]
    shares = volume.at[date, asset]
    change = pct_chg.at[date, asset]
    return bool(
        np.isfinite(price) and price > 0
        and np.isfinite(shares) and shares > 0
        and np.isfinite(change)
    )


def _trade_block_reason(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    pct_chg: pd.DataFrame,
    date: pd.Timestamp,
    asset: str,
    side: str,
    price_label: str = "close",
) -> str | None:
    """Why a rebalance trade cannot be executed at this close (or None)."""
    price = close.at[date, asset]
    shares = volume.at[date, asset]
    change = pct_chg.at[date, asset]
    if not (np.isfinite(price) and price > 0):
        return f"has no valid {price_label} on the rebalance date"
    if not (np.isfinite(change) and np.isfinite(shares) and shares > 0):
        return (
            "is not tradable on the rebalance date (missing pct_chg or "
            "zero/NaN volume)"
        )
    if side == "buy" and change > LIMIT_UP_PCT:
        return (
            f"closed limit-up on the rebalance date (pct_chg {change:.4f}% > "
            f"{LIMIT_UP_PCT}%); the buy is blocked"
        )
    if side == "sell" and change < -LIMIT_UP_PCT:
        return (
            f"closed limit-down on the rebalance date (pct_chg {change:.4f}% < "
            f"-{LIMIT_UP_PCT}%); the sell is blocked"
        )
    return None


def _rebalance_cost_fraction(
    cost_model: CostModel,
    buy_cny: pd.Series,
    sell_cny: pd.Series,
    date: pd.Timestamp,
    portfolio_value: float,
    panels: Panels,
    vol_row: pd.Series | None,
) -> float:
    """CostModel charge on executed buy/sell notionals, as a fraction of P."""
    index = buy_cny.index.union(sell_cny.index)
    if len(index) == 0:
        return 0.0
    # pandas would align the two Series by label and leave NaN on the
    # asset only present on one side; CostModel expects a dense buy/sell
    # split, so reindex both onto the union with explicit zeros.
    buy = buy_cny.reindex(index, fill_value=0.0).astype(float)
    sell = sell_cny.reindex(index, fill_value=0.0).astype(float)
    daily_volume: pd.Series | None = None
    volatility: pd.Series | None = None
    if vol_row is not None:
        # R9 basis: daily traded value in CNY = shares * raw price.
        daily_volume = (
            panels.volume.loc[date].reindex(index).astype(float)
            * panels.close.loc[date].reindex(index).astype(float)
        )
        volatility = vol_row.reindex(index).astype(float)
        if (
            not np.isfinite(daily_volume.to_numpy()).all()
            or (daily_volume <= 0).any()
        ):
            raise PaperTrackError(
                f"{date.date()}: non-positive CNY volume for a traded asset "
                "despite the tradability gate; refusing to price costs"
            )
    costs = cost_model.compute_costs(
        0.0,
        buy_turnover=buy,
        sell_turnover=sell,
        daily_volume=daily_volume,
        volatility=volatility,
        date=date,
        portfolio_value=portfolio_value,
    )
    total = float(np.sum(np.asarray(costs, dtype=float)))
    if not math.isfinite(total) or total < 0:
        raise PaperTrackError(
            f"{date.date()}: cost model produced an invalid cost fraction "
            f"{total!r}"
        )
    return total


def _rebalance_to_target(
    date: pd.Timestamp,
    target: Target,
    holdings: dict[str, float],
    cash: float,
    panels: Panels,
    cost_model: CostModel,
    vol_row: pd.Series | None,
    warnings: list[str],
    price_label: str = "close",
) -> tuple[float, Rebalance]:
    """Trade ``holdings`` toward ``target`` at the close of ``date``.

    Mutates ``holdings`` in place and returns ``(cash_after, record)``.
    Sells execute first and their proceeds (plus existing cash) fund the
    buys; buys are scaled down when they would leave less cash than the
    transaction costs.
    """
    close = panels.close
    volume = panels.volume
    pct_chg = panels.pct_chg
    pre_value = float(sum(holdings.values()) + cash)
    if not math.isfinite(pre_value) or pre_value <= 0:
        raise PaperTrackError(
            f"{date.date()}: non-positive portfolio value at rebalance"
        )
    desired = {
        asset: float(weight) * pre_value
        for asset, weight in target.weights.items()
    }
    blocked_buys: list[str] = []
    blocked_sells: list[str] = []

    sells: dict[str, float] = {}
    for asset in sorted(holdings):
        held = holdings[asset]
        want = desired.get(asset, 0.0)
        if held <= want + ROLL_REBALANCE_EPS:
            continue
        reason = _trade_block_reason(
            close, volume, pct_chg, date, asset, "sell", price_label,
        )
        if reason is None:
            sells[asset] = held - want
            if want > ROLL_REBALANCE_EPS:
                holdings[asset] = want
            else:
                holdings.pop(asset)
        else:
            desired[asset] = held
            blocked_sells.append(asset)
            warnings.append(
                f"{date.date()}: rebalance sell of {asset} {reason}; the "
                "position is held at its current weight"
            )
    sell_cny = float(sum(sells.values()))
    cash += sell_cny

    buys: dict[str, float] = {}
    for asset in sorted(desired):
        want = desired[asset]
        held = holdings.get(asset, 0.0)
        if want <= held + ROLL_REBALANCE_EPS:
            continue
        reason = _trade_block_reason(
            close, volume, pct_chg, date, asset, "buy", price_label,
        )
        if reason is None:
            buys[asset] = want - held
        else:
            blocked_buys.append(asset)
            warnings.append(
                f"{date.date()}: rebalance buy of {asset} {reason}; its "
                "weight stays in cash"
            )

    buy_series = pd.Series(buys, dtype=float).sort_index()
    sell_series = pd.Series(sells, dtype=float).sort_index()
    available = cash
    # Numerical tolerance for the cash budget: 1e-12 relative (a micro-yuan
    # on a CNY 1M book), large enough to absorb float summation noise but
    # not a real overdraft.
    tolerance = max(1e-9, 1e-12 * max(1.0, pre_value))

    def feasible(scale: float) -> bool:
        scaled = buy_series * scale
        cost_fraction = _rebalance_cost_fraction(
            cost_model, scaled, sell_series, date, pre_value, panels, vol_row,
        )
        return (
            float(scaled.sum()) + cost_fraction * pre_value
            <= available + tolerance
        )

    buy_scale = 1.0
    if len(buy_series):
        if feasible(1.0):
            buy_scale = 1.0
        elif not feasible(0.0):
            raise PaperTrackError(
                f"{date.date()}: expected sell proceeds ({sell_cny:.2f} CNY) "
                "cannot fund the rebalance transaction costs; refusing to "
                "write a negative-cash ledger"
            )
        else:
            # Feasibility is monotone in the buy scale (notionals and their
            # costs both grow with it), so bisection finds the largest
            # affordable scale deterministically.
            low, high = 0.0, 1.0
            for _ in range(64):
                middle = 0.5 * (low + high)
                if feasible(middle):
                    low = middle
                else:
                    high = middle
            buy_scale = low
            # A hair below the bisection point so that re-evaluating the
            # costs at the final notionals cannot land a floating-point ulp
            # above the cash budget.
            buy_scale = max(0.0, buy_scale * (1.0 - 1e-12))
            if buy_scale < 0.995:
                warnings.append(
                    f"{date.date()}: cash constraint scaled rebalance buys to "
                    f"{buy_scale:.4%} of the intended notional"
                )
    executed_buys = buy_series * buy_scale
    cost_fraction = _rebalance_cost_fraction(
        cost_model, executed_buys, sell_series, date, pre_value, panels, vol_row,
    )
    cost_cny = cost_fraction * pre_value
    buy_cny = float(executed_buys.sum())
    if buy_cny + cost_cny > available + tolerance:
        raise PaperTrackError(
            f"{date.date()}: internal cash-feasibility check failed after "
            f"scaling (buys {buy_cny:.2f} + costs {cost_cny:.2f} > cash "
            f"{available:.2f})"
        )
    for asset, amount in executed_buys.items():
        amount = float(amount)
        if amount <= 0.0:
            continue
        holdings[asset] = holdings.get(asset, 0.0) + amount
    cash = available - buy_cny - cost_cny
    if cash < -tolerance:
        raise PaperTrackError(
            f"{date.date()}: rebalance left negative cash ({cash:.6f} CNY)"
        )
    if cash < 0:
        cash = 0.0
    turnover = 0.5 * (buy_cny + sell_cny) / pre_value
    record = Rebalance(
        as_of=target.as_of,
        execution_date=date,
        turnover=float(turnover),
        costs_cny=float(cost_cny),
        buy_cny=buy_cny,
        sell_cny=sell_cny,
        buy_scale=float(buy_scale),
        n_buys=int((executed_buys > 0).sum()),
        n_sells=int(len(sell_series)),
        blocked_buys=blocked_buys,
        blocked_sells=blocked_sells,
    )
    return cash, record


def simulate_roll(
    targets: list[Target],
    panels: Panels,
    capital: float,
    cost_mode: str,
    max_stale_sessions: int,
) -> RollSimulation:
    """Run the multi-target paper book; raise on insufficient/invalid data."""
    if not math.isfinite(capital) or capital <= 0:
        raise PaperTrackError(f"capital must be a positive number; got {capital!r}")
    if max_stale_sessions < 0:
        raise PaperTrackError("--max-stale-sessions must be >= 0 (0 disables)")
    close = panels.close
    volume = panels.volume
    pct_chg = panels.pct_chg
    price_label = (
        "close" if panels.raw_price_mode == RAW_PRICE_MODE_CLOSE
        else "raw price"
    )
    dates = close.index
    if len(dates) == 0:
        raise PaperTrackError("cache panel contains no sessions")
    last_cache_date = pd.Timestamp(dates[-1])

    available_assets = set(close.columns)
    for target in targets:
        unknown = sorted(set(target.weights.index) - available_assets)
        if unknown:
            shown = ", ".join(unknown[:10])
            more = "" if len(unknown) <= 10 else f" (+{len(unknown) - 10} more)"
            raise PaperTrackError(
                f"{target.path}: target asset(s) not present in the cached "
                f"panel: {shown}{more}"
            )

    warnings: list[str] = []
    schedule: list[tuple[Target, pd.Timestamp]] = []
    pending: list[tuple[Path, pd.Timestamp]] = []
    for target in targets:
        if target.as_of > last_cache_date:
            raise PaperTrackError(
                f"{target.path}: target as_of {target.as_of.date()} is beyond "
                f"the cache end ({last_cache_date.date()}); no data to track"
            )
        execution = _next_exchange_session(target.as_of)
        position = int(dates.searchsorted(execution))
        if position >= len(dates):
            pending.append((target.path, target.as_of))
            continue
        cached = pd.Timestamp(dates[position])
        if cached != execution:
            raise PaperTrackError(
                f"{target.path}: next exchange trading session after as_of "
                f"{target.as_of.date()} is {execution.date()}, but that session "
                f"is missing from the cache (next cached date {cached.date()}); "
                "refusing to execute on a later session"
            )
        schedule.append((target, execution))
    if not schedule:
        raise PaperTrackError(
            "no target has an execution session after its as_of in the "
            "cache; nothing to track"
        )
    execution_seen: dict[pd.Timestamp, Path] = {}
    for target, execution in schedule:
        previous = execution_seen.get(execution)
        if previous is not None:
            raise PaperTrackError(
                f"targets {previous} and {target.path} both execute on "
                f"{execution.date()}; duplicate execution sessions are not "
                "supported"
            )
        execution_seen[execution] = target.path
    if pending:
        warnings.append(
            "pending target(s) whose next exchange session is not cached yet: "
            + ", ".join(
                f"{path.name} ({as_of.date()})" for path, as_of in pending
            )
        )

    cost_model = _build_cost_model(cost_mode)
    vol_panel = (
        _vol_panel(panels.volatility_returns())
        if cost_mode == "impact" else None
    )
    benchmark = panels.returns().mean(axis=1, skipna=True)
    execution_map = {execution: target for target, execution in schedule}

    first_execution = schedule[0][1]
    start = int(dates.searchsorted(first_execution))
    holdings: dict[str, float] = {}
    cash = float(capital)
    last_price: dict[str, float] = {}
    stale_streak: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    rebalances: list[Rebalance] = []
    nav_prev_row = float(capital)
    peak = float(capital)

    for date in dates[start:]:
        date = pd.Timestamp(date)
        stale_today: list[str] = []
        for asset in sorted(holdings):
            if _observation_ok(close, volume, pct_chg, date, asset):
                change = float(pct_chg.at[date, asset]) / 100.0
                holdings[asset] *= 1.0 + change
                last_price[asset] = float(close.at[date, asset])
                stale_streak[asset] = 0
            else:
                stale_streak[asset] = stale_streak.get(asset, 0) + 1
                stale_today.append(asset)
        if max_stale_sessions > 0:
            for asset in sorted(holdings):
                if stale_streak.get(asset, 0) < max_stale_sessions:
                    continue
                frozen = last_price.pop(asset, np.nan)
                cash += holdings.pop(asset)
                stale_streak.pop(asset, None)
                warnings.append(
                    f"{date.date()}: {asset} had no valid price for "
                    f"{max_stale_sessions} consecutive sessions; liquidated at "
                    f"its last {price_label} {frozen:.6f} and held as cash "
                    "(no trade cost: no tradable price exists)"
                )
        turnover = 0.0
        costs_paid = 0.0
        target = execution_map.get(date)
        if target is not None:
            date_vol_row = (
                vol_panel.loc[date] if vol_panel is not None else None
            )
            cash, record = _rebalance_to_target(
                date, target, holdings, cash, panels, cost_model, date_vol_row,
                warnings, price_label,
            )
            if not rebalances and record.n_buys == 0:
                raise PaperTrackError(
                    f"{date.date()}: initial rebalance has zero executed buys; "
                    "refusing to publish an all-cash inception ledger"
                )
            rebalances.append(record)
            turnover = record.turnover
            costs_paid = record.costs_cny
            for asset in holdings:
                if _observation_ok(close, volume, pct_chg, date, asset):
                    last_price[asset] = float(close.at[date, asset])
            for asset in list(last_price):
                if asset not in holdings:
                    last_price.pop(asset)
            for asset in list(stale_streak):
                if asset not in holdings:
                    stale_streak.pop(asset)
        nav = float(sum(holdings.values()) + cash)
        if not math.isfinite(nav) or nav <= 0:
            raise PaperTrackError(
                f"{date.date()}: NAV would be non-positive ({nav!r}); "
                "refusing to write"
            )
        daily_return = nav / nav_prev_row - 1.0
        # The first ledger row is the build/execution close: there is no
        # tracked session before it, so the benchmark starts flat exactly like
        # the strategy (the single-target mode's execution row is 0.0 too).
        bench_return = 0.0 if not rows else benchmark.get(date, 0.0)
        if not math.isfinite(bench_return):
            bench_return = 0.0
        peak = max(peak, nav)
        rows.append(
            {
                "date": date,
                "nav": nav,
                "daily_return": float(daily_return),
                "benchmark_daily_return": float(bench_return),
                "cum_return": nav / capital - 1.0,
                "drawdown": nav / peak - 1.0,
                "n_positions": int(len(holdings)),
                "turnover": float(turnover),
                "costs_paid": float(costs_paid),
                "stale_assets": int(len(stale_today)),
            }
        )
        nav_prev_row = nav

    return RollSimulation(
        rows=rows,
        warnings=warnings,
        rebalances=rebalances,
        pending=pending,
        targets=targets,
        last_cache_date=last_cache_date,
        cost_mode=cost_mode,
        max_stale_sessions=max_stale_sessions,
        raw_price_mode=panels.raw_price_mode,
    )


def _format_roll_row(row: dict[str, Any]) -> str:
    """Format one rolling ledger row with fixed precision."""
    return ",".join(
        (
            pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            f"{row['nav']:.6f}",
            f"{row['daily_return']:.10f}",
            f"{row['benchmark_daily_return']:.10f}",
            f"{row['cum_return']:.10f}",
            f"{row['drawdown']:.10f}",
            str(int(row["n_positions"])),
            f"{row['turnover']:.10f}",
            f"{row['costs_paid']:.6f}",
            str(int(row["stale_assets"])),
        )
    )


def _roll_merged_series(lines: list[str]) -> dict[str, Any]:
    """Parse rolling ledger lines into per-column arrays for the summary."""
    dates: list[pd.Timestamp] = []
    nav: list[float] = []
    daily: list[float] = []
    bench: list[float] = []
    costs: list[float] = []
    stale: list[int] = []
    for line in lines:
        parts = line.split(",")
        if len(parts) != len(ROLL_NAV_COLUMNS):
            raise PaperTrackError(
                f"internal error: malformed rolling ledger row: {line!r}"
            )
        dates.append(pd.Timestamp(parts[0]))
        nav.append(float(parts[1]))
        daily.append(float(parts[2]))
        bench.append(float(parts[3]))
        costs.append(float(parts[8]))
        stale.append(int(parts[9]))
    index = pd.DatetimeIndex(dates)
    return {
        "dates": index,
        "nav": np.asarray(nav, dtype=float),
        "daily": pd.Series(daily, index=index),
        "bench": pd.Series(bench, index=index),
        "costs": np.asarray(costs, dtype=float),
        "stale": np.asarray(stale, dtype=int),
    }


def build_roll_summary(
    simulation: RollSimulation,
    lines: list[str],
    capital: float,
    target_files: list[str],
    target_identities: list[dict[str, str]],
    targets_sha256: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Deterministic summary of a rolling ledger (no timestamps inside)."""
    series = _roll_merged_series(lines)
    daily = series["daily"]
    bench = series["bench"]
    nav = series["nav"]
    dates = series["dates"]
    total_return = float(nav[-1] / capital - 1.0)
    bench_nav = np.cumprod(1.0 + bench.to_numpy(dtype=float))
    benchmark_total = float(bench_nav[-1] - 1.0)
    strategy_metrics = _metrics(daily, nav, dates, capital)
    benchmark_metrics = _metrics(bench, bench_nav, dates, 1.0)
    tracking_error_value = (
        _finite(tracking_error(daily, bench, PERIODS_PER_YEAR))
        if len(daily) >= 2 else None
    )
    total_costs = float(series["costs"].sum())
    turnovers = [record.turnover for record in simulation.rebalances]
    amount_basis = simulation.raw_price_mode == RAW_PRICE_MODE_AMOUNT
    summary: dict[str, Any] = {
        "mode": ROLL_MODE,
        "roll_ledger_version": ROLL_LEDGER_VERSION,
        "capital": float(capital),
        "cost_mode": simulation.cost_mode,
        "max_stale_sessions": int(simulation.max_stale_sessions),
        "n_targets": int(len(simulation.targets)),
        "n_rebalances": int(len(simulation.rebalances)),
        "n_pending_targets": int(len(simulation.pending)),
        "targets_sha256": targets_sha256,
        "target_files": list(target_files),
        "target_identities": [dict(identity) for identity in target_identities],
        "first_ledger_date": dates[0].strftime("%Y-%m-%d"),
        "last_ledger_date": dates[-1].strftime("%Y-%m-%d"),
        "cache_last_date": simulation.last_cache_date.strftime("%Y-%m-%d"),
        "n_sessions": int(len(daily)),
        "stale_policy": STALE_POLICY_AMOUNT if amount_basis else STALE_POLICY,
        "return_basis": RETURN_BASIS_AMOUNT if amount_basis else RETURN_BASIS,
        "volatility_basis": VOLATILITY_BASIS,
        "tradable_gate": (
            TRADABLE_GATE_AMOUNT if amount_basis else TRADABLE_GATE
        ) + (
            "; at a rebalance, limit-down blocks the sell and limit-up blocks "
            "the buy"
        ),
        "benchmark_definition": BENCHMARK_DEFINITION,
        "risk_free_rate": RISK_FREE_ANNUAL,
        "annualization_periods_per_year": PERIODS_PER_YEAR,
        "since_inception": {
            "final_nav": float(nav[-1]),
            "total_return": total_return,
            "annualized_return": strategy_metrics["annualized_return"],
            "annualized_volatility": strategy_metrics["annualized_volatility"],
            "sharpe_ratio": strategy_metrics["sharpe_ratio"],
            "max_drawdown": strategy_metrics["max_drawdown"],
            "max_drawdown_peak": strategy_metrics["max_drawdown_peak"],
            "max_drawdown_trough": strategy_metrics["max_drawdown_trough"],
            "n_stale_days": int((series["stale"] > 0).sum()),
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
        "turnover": {
            "definition": (
                "realised one-sided turnover = 0.5 * (buys + sells) / "
                "pre-cost portfolio value, per executed rebalance"
            ),
            "total": float(sum(turnovers)),
            "mean_per_rebalance": (
                float(np.mean(turnovers)) if turnovers else None
            ),
        },
        "rebalances": [
            {
                "as_of": record.as_of.strftime("%Y-%m-%d"),
                "execution_date": record.execution_date.strftime("%Y-%m-%d"),
                "turnover": record.turnover,
                "costs_cny": record.costs_cny,
                "buy_cny": record.buy_cny,
                "sell_cny": record.sell_cny,
                "buy_scale": record.buy_scale,
                "n_buys": record.n_buys,
                "n_sells": record.n_sells,
                "blocked_buys": record.blocked_buys,
                "blocked_sells": record.blocked_sells,
            }
            for record in simulation.rebalances
        ],
        "pending_targets": [
            {
                "file": path.as_posix(),
                "as_of": as_of.strftime("%Y-%m-%d"),
                "reason": "next exchange session is not cached yet",
            }
            for path, as_of in simulation.pending
        ],
        "warnings": list(warnings),
        "disclaimer": DISCLAIMER,
    }
    if amount_basis:
        # Legacy raw-close ledgers keep their pre-R32 bytes; the explicit
        # basis is only recorded when it is not the legacy assumption.
        summary["raw_price_mode"] = simulation.raw_price_mode
    return summary


def _assert_roll_target_prefix(
    existing: dict[str, Any],
    current: list[dict[str, str]],
    current_digest: str,
    summary_path: Path,
) -> None:
    """Allow only an unchanged historical target prefix plus later targets."""
    def refuse(reason: str) -> None:
        raise PaperTrackError(
            f"{summary_path}: {reason}; rolling targets must remain an "
            "unchanged hash prefix and only strictly later appends are allowed"
        )

    previous = existing.get("target_identities")
    if previous is None:
        current_files = [identity["file"] for identity in current]
        if (
            existing.get("targets_sha256") != current_digest
            or existing.get("target_files") != current_files
        ):
            refuse("the legacy target schedule is not exactly unchanged")
        # A legacy summary has no per-target prefix to validate. Its exact
        # digest is sufficient for an idempotent rebuild, but not for appends.
        return
    if (
        not isinstance(previous, list)
        or not all(isinstance(identity, dict) for identity in previous)
        or existing.get("n_targets") != len(previous)
    ):
        refuse("stored target identities are malformed")
    if len(current) < len(previous):
        refuse("the target schedule was shortened")
    for index, old in enumerate(previous):
        new = current[index]
        if any(
            old.get(key) != new.get(key)
            for key in ("file", "as_of", "sha256")
        ):
            refuse(
                f"target schedule changed at position {index + 1} "
                f"({old.get('file')}, {old.get('as_of')})"
            )
    if previous and len(current) > len(previous):
        previous_last = previous[-1].get("as_of")
        first_append = current[len(previous)]["as_of"]
        if not isinstance(previous_last, str) or first_append <= previous_last:
            refuse("every appended target must have a strictly later as_of")


def run_roll(
    target_paths: list[Path],
    cache_path: Path,
    out_dir: Path,
    capital: float = DEFAULT_CAPITAL,
    cost_mode: str = "impact",
    max_stale_sessions: int = DEFAULT_MAX_STALE_SESSIONS,
    raw_price_mode: str = RAW_PRICE_MODE_AUTO,
) -> tuple[dict[str, Any], Path]:
    """Rebuild one rolling paper ledger; returns (summary, nav_path)."""
    if not math.isfinite(capital) or capital <= 0:
        raise PaperTrackError(f"capital must be a positive number; got {capital!r}")
    if max_stale_sessions < 0:
        raise PaperTrackError("--max-stale-sessions must be >= 0 (0 disables)")
    targets = load_targets([Path(path) for path in target_paths])
    panels = load_cache(Path(cache_path), raw_price_mode)
    simulation = simulate_roll(
        targets, panels, float(capital), cost_mode, int(max_stale_sessions),
    )
    lines = [_format_roll_row(row) for row in simulation.rows]
    if not lines:
        raise PaperTrackError("internal error: empty rolling ledger after simulation")

    digest = hashlib.sha256()
    target_identities: list[dict[str, str]] = []
    for target in targets:
        file_sha256 = hashlib.sha256(target.path.read_bytes()).hexdigest()
        digest.update(bytes.fromhex(file_sha256))
        target_identities.append(
            {
                "file": target.path.as_posix(),
                "as_of": target.as_of.strftime("%Y-%m-%d"),
                "sha256": file_sha256,
            }
        )
    target_files = [identity["file"] for identity in target_identities]
    summary = build_roll_summary(
        simulation, lines, float(capital), target_files, target_identities,
        digest.hexdigest(), list(simulation.warnings),
    )

    out_dir = Path(out_dir)
    nav_path = out_dir / "nav.csv"
    summary_path = out_dir / "summary.json"
    if nav_path.is_file():
        existing_lines = nav_path.read_text(encoding="utf-8").splitlines()
        header = existing_lines[0].strip() if existing_lines else ""
        if header != ",".join(ROLL_NAV_COLUMNS):
            raise PaperTrackError(
                f"{nav_path} exists but is not a rolling ledger (header "
                f"{header!r}); use a new --out-dir"
            )
    if summary_path.is_file():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PaperTrackError(
                f"{summary_path} exists but is not valid JSON; refusing to "
                "overwrite"
            ) from exc
        if existing.get("mode") != ROLL_MODE:
            raise PaperTrackError(
                f"{summary_path} belongs to a non-rolling paper_track "
                "ledger; use a new --out-dir"
            )
        _assert_roll_target_prefix(
            existing, target_identities, digest.hexdigest(), summary_path,
        )
        # A growing target schedule is the intended rebuild case, but a
        # different capital / cost mode / stale policy / raw price basis is a
        # configuration mistake that would silently rewrite the whole ledger.
        for key, expected, label in (
            ("capital", float(capital), "capital"),
            ("cost_mode", cost_mode, "cost mode"),
            ("max_stale_sessions", int(max_stale_sessions), "stale policy"),
        ):
            if key not in existing or existing[key] != expected:
                current = existing.get(key, "<missing>")
                raise PaperTrackError(
                    f"{summary_path} belongs to a rolling ledger with a "
                    f"different {label} ({current!r} != {expected!r}); use a "
                    "new --out-dir"
                )
        # Pre-R32 ledgers have no key and were all on the legacy close basis.
        existing_mode = existing.get("raw_price_mode", RAW_PRICE_MODE_CLOSE)
        if existing_mode != panels.raw_price_mode:
            raise PaperTrackError(
                f"{summary_path} belongs to a rolling ledger with a different "
                f"raw price mode ({existing_mode!r} != "
                f"{panels.raw_price_mode!r}); use a new --out-dir or pass the "
                "matching --raw-price-mode"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        nav_path,
        ",".join(ROLL_NAV_COLUMNS) + "\n" + "\n".join(lines) + "\n",
    )
    summary_text = json.dumps(
        summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    _atomic_write_text(summary_path, summary_text)
    return summary, nav_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forward paper tracking: fixed target book (or a rolling target "
            "schedule) -> idempotent daily NAV ledger + benchmark (read-only; "
            "no broker)."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--target",
        help="target weights JSON (single-target mode: build once, then drift)",
    )
    mode.add_argument(
        "--targets-dir",
        help=(
            "directory of daily_signal JSON targets (rolling mode; the files "
            "are sorted by as_of)"
        ),
    )
    mode.add_argument(
        "--targets",
        help="comma-separated target JSON paths (rolling mode)",
    )
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
        "--raw-price-mode", choices=RAW_PRICE_MODES, default=RAW_PRICE_MODE_AUTO,
        help=(
            "raw CNY price basis for cost conversion, tradability and stale "
            "liquidation: 'auto' samples amount vs close*volume and switches "
            "to amount/volume on a >1%% median relative error (default), "
            "'close' keeps the legacy assumption that the cached close is "
            "raw, 'amount_over_volume' uses amount/volume unconditionally "
            "(requires the amount column)"
        ),
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
        if args.target is not None:
            summary, n_new, nav_path = run(
                Path(args.target),
                Path(args.cache),
                Path(args.out_dir),
                capital=float(args.capital),
                cost_mode=args.cost_mode,
                max_stale_sessions=int(args.max_stale_sessions),
                raw_price_mode=args.raw_price_mode,
            )
        else:
            if args.targets_dir is not None:
                target_paths = target_paths_from_dir(Path(args.targets_dir))
            else:
                target_paths = target_paths_from_list(args.targets)
            summary, nav_path = run_roll(
                target_paths,
                Path(args.cache),
                Path(args.out_dir),
                capital=float(args.capital),
                cost_mode=args.cost_mode,
                max_stale_sessions=int(args.max_stale_sessions),
                raw_price_mode=args.raw_price_mode,
            )
    except PaperTrackError as exc:
        print(f"paper_track: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"paper_track: error: {exc}", file=sys.stderr)
        return 2

    since = summary["since_inception"]
    if args.target is not None:
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
    else:
        print(
            "paper_track: rolling ledger {first}..{last} ({sessions} sessions, "
            "{rebalances} rebalances, {pending} pending target(s)); final NAV "
            "{nav:.2f} ({ret:+.2%}) vs benchmark {bench:+.2%} (diff "
            "{diff:+.2%}); costs {costs:.2f} CNY".format(
                first=summary["first_ledger_date"],
                last=summary["last_ledger_date"],
                sessions=summary["n_sessions"],
                rebalances=summary["n_rebalances"],
                pending=summary["n_pending_targets"],
                nav=since["final_nav"],
                ret=since["total_return"],
                bench=summary["benchmark"]["total_return"],
                diff=summary["tracking"]["difference"],
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
