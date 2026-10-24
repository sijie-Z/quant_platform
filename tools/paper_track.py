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
* **Costs (R9 basis).** On the execution day the tool buys the target book
  from cash and charges :class:`~quant_platform.backtest.cost_model.CostModel`
  exactly the way the engine does: CNY buy notionals
  (``weight * capital``), ``daily_volume`` in CNY as
  ``volume_shares * raw_close`` (the cached ``close`` is the raw CNY close --
  ``amount`` matches ``close * volume``), the date-dependent statutory stamp
  tax (no sells on a build, so the stamp tax is zero), the per-order CNY 5
  commission floor, the both-sides transfer fee, and -- with
  ``--cost-mode impact`` -- the square-root impact rule on participation.
  Volatility is the causal ``rolling(20).std().shift(1).fillna(0.02)``
  estimate used by the engine.
* **Drift.** Between executions weights drift with the assets' close-to-close
  returns; cash earns nothing. NAV is close-to-close: the execution row's only
  return is the build cost, and each later row's return is the drifted book's
  return from the previous close.
* **Stale prices.** A held asset with no valid close on a session is held at
  its last valid close (return 0) and counted in ``stale_assets`` -- it is
  never silently marked to zero. After ``--max-stale-sessions`` consecutive
  missing closes (default 20; 0 disables) the position is liquidated at its
  last valid close, the proceeds become cash, and a warning is recorded. A
  target asset without a valid close *on the execution date* is not bought at
  all; its weight stays in cash (warning).
* **Benchmark.** Equal-weight, daily-rebalanced mean of the cached universe's
  close-to-close returns over the same window (the ``daily_ew`` convention in
  ``tools/real_data_research.py``). The execution row is 0.0.

Idempotent accumulation
-----------------------
``nav.csv`` is keyed by date. Re-running the same inputs appends nothing,
leaves ``nav.csv`` byte-for-byte untouched, and rewrites ``summary.json`` with
identical content (it contains no timestamps). When the cache grows, only
dates after the last ledger row are appended; existing rows are preserved
verbatim. Mixing a different target, capital, cost mode, or stale policy into
an existing out-dir is refused. If a revised cache would change already-written
NAV values, the tool keeps the ledger rows but records the discrepancy in
``summary.json`` warnings.

Errors (target beyond the cache, no session after ``as_of``, malformed target
or panel) exit non-zero before anything is written, so a failed run never
leaves a half-built ledger.

Outputs (in ``--out-dir``)
--------------------------
``nav.csv`` columns:
``date, nav, daily_return, benchmark_daily_return, cum_return, drawdown,
n_positions, stale_assets, costs_paid``.

``summary.json`` carries since-inception return / annualized / Sharpe / max
drawdown, the benchmark on the same window, tracking difference and tracking
error, the execution date, total costs, warnings, and the mandatory
``disclaimer``.

Usage::

    python tools/paper_track.py \
      --target docs/research/target_portfolio_2025-12-31.json \
      --cache .cache/baostock/daily_2018-01-01_2025-12-31_500.parquet \
      --out-dir docs/research/paper_track \
      --capital 2000000 --cost-mode impact
"""

from __future__ import annotations

import argparse
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
    max_drawdown,
    sharpe_ratio,
    tracking_error,
)

DISCLAIMER = "paper tracking only; style-tilt demo, not validated alpha"
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
BENCHMARK_DEFINITION = (
    "equal-weight daily-rebalanced mean of the cached universe's "
    "close-to-close returns"
)
STALE_POLICY = (
    "hold at the last valid close (return 0, flagged stale); after "
    "max_stale_sessions consecutive missing closes, liquidate at the last "
    "close and hold the proceeds as cash"
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
    as_of = pd.to_datetime(payload["as_of"], errors="coerce")
    if pd.isna(as_of):
        raise PaperTrackError(
            f"target 'as_of' is not a parseable date: {payload['as_of']!r}"
        )
    as_of = pd.Timestamp(as_of).normalize()

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


def load_cache(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cached daily panel and return (close, volume) wide panels."""
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
    for column in ("close", "volume"):
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
    volume = (
        frame["volume"].unstack("asset").reindex_like(close).astype(float)
    )
    return close, volume


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


def _vol_panel(close: pd.DataFrame) -> pd.DataFrame:
    """Causal daily volatility, exactly the engine's fallback estimate.

    At close(d) only returns before d are observable, so the rolling window is
    shifted one session; short histories fall back to 2% daily.
    """
    returns = close.pct_change(fill_method=None)
    return (
        returns.rolling(VOL_LOOKBACK, min_periods=VOL_MIN_PERIODS)
        .std()
        .shift(1)
        .fillna(VOL_FALLBACK)
    )


def simulate(
    weights: pd.Series,
    as_of: pd.Timestamp,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    capital: float,
    cost_mode: str,
    max_stale_sessions: int,
) -> Simulation:
    """Run the close-to-close paper book; raise on insufficient data."""
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
    exec_prices = close.loc[execution]
    target = weights.copy()

    # Assets without a valid close on the execution date cannot be bought at
    # that close; their weight stays in cash for the whole run.
    valid = pd.Series(False, index=target.index)
    for asset in target.index:
        if asset not in close.columns:
            continue
        price = exec_prices.get(asset, np.nan)
        valid.loc[asset] = bool(np.isfinite(price) and price > 0)
    unpurchased = sorted(target.index[~valid])
    for asset in unpurchased:
        warnings.append(
            f"{execution.date()}: target asset {asset} has no valid close on "
            "the execution date; its weight is held as cash and never bought"
        )
    buy_weights = target[valid].astype(float)
    if buy_weights.empty:
        raise PaperTrackError(
            f"no target asset has a valid close on the execution date "
            f"{execution.date()}; nothing to buy"
        )
    gross = float(buy_weights.sum())
    cash_weight = max(0.0, 1.0 - gross)
    cash_at_build = cash_weight

    # --- Build cost: CostModel on CNY notional, volume in CNY (R9 basis) ---
    notional = buy_weights * capital
    volume_cny: pd.Series | None = None
    vol_row: pd.Series | None = None
    if cost_mode == "impact":
        # Engine parity: missing volume cells use the cross-sectional median.
        shares = volume.loc[execution].reindex(buy_weights.index)
        traded_cny = shares * exec_prices.reindex(buy_weights.index)
        if traded_cny.notna().any():
            volume_cny = traded_cny.fillna(float(traded_cny.median())).astype(float)
        else:
            warnings.append(
                f"{execution.date()}: no volume data for the bought assets; "
                "CostModel falls back to fixed slippage"
            )
        vol_row = _vol_panel(close).loc[execution].reindex(buy_weights.index)
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
    cost_cny = cost_fraction * capital
    nav_prev = capital * (1.0 - cost_fraction)

    held = buy_weights.astype(float).copy()
    last_price = {asset: float(exec_prices[asset]) for asset in held.index}
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
    benchmark = close.pct_change(fill_method=None).mean(axis=1, skipna=True)

    for date in dates[position + 1:]:
        date = pd.Timestamp(date)
        asset_returns: dict[str, float] = {}
        stale_today: list[str] = []
        for asset in held.index:
            price = close.at[date, asset]
            if np.isfinite(price) and price > 0:
                base = last_price[asset]
                asset_returns[asset] = float(price) / base - 1.0
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
                        f"{date.date()}: {asset} had no valid close for "
                        f"{max_stale_sessions} consecutive sessions; "
                        f"liquidated at its last close {frozen:.6f} and held "
                        "as cash"
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
    return {
        "dates": pd.DatetimeIndex(dates),
        "nav": np.asarray(nav, dtype=float),
        "daily": pd.Series(daily, index=pd.DatetimeIndex(dates)),
        "bench": pd.Series(bench, index=pd.DatetimeIndex(dates)),
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


def _metrics(series: pd.Series) -> dict[str, Any]:
    """Annualized return / vol / Sharpe / max drawdown on an explicit 252-day year."""
    n = len(series)
    if n == 0:
        return {"annualized_return": None, "annualized_volatility": None,
                "sharpe_ratio": None, "max_drawdown": None,
                "max_drawdown_peak": None, "max_drawdown_trough": None}
    mdd, peak_date, trough_date = max_drawdown(series)
    if n < 2:
        return {
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": _finite(mdd),
            "max_drawdown_peak": (
                str(peak_date.date()) if pd.notna(peak_date) else None
            ),
            "max_drawdown_trough": (
                str(trough_date.date()) if pd.notna(trough_date) else None
            ),
        }
    return {
        "annualized_return": _finite(annualized_return(series, PERIODS_PER_YEAR)),
        "annualized_volatility": _finite(
            annualized_volatility(series, PERIODS_PER_YEAR)
        ),
        "sharpe_ratio": _finite(
            sharpe_ratio(series, RISK_FREE_ANNUAL, PERIODS_PER_YEAR)
        ),
        "max_drawdown": _finite(mdd),
        "max_drawdown_peak": str(peak_date.date()) if pd.notna(peak_date) else None,
        "max_drawdown_trough": (
            str(trough_date.date()) if pd.notna(trough_date) else None
        ),
    }


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
    total_return = float(nav[-1] / capital - 1.0)
    benchmark_total = float((1.0 + bench).prod() - 1.0)
    strategy_metrics = _metrics(daily)
    benchmark_metrics = _metrics(bench)
    if len(daily) >= 2:
        tracking_error_value = _finite(
            tracking_error(daily, bench, PERIODS_PER_YEAR)
        )
    else:
        tracking_error_value = None
    total_costs = float(series["costs"].sum())
    stale_days = int((series["stale"] > 0).sum())
    return {
        "as_of": simulation.as_of.strftime("%Y-%m-%d"),
        "execution_date": simulation.execution_date.strftime("%Y-%m-%d"),
        "first_ledger_date": series["dates"][0].strftime("%Y-%m-%d"),
        "last_ledger_date": series["dates"][-1].strftime("%Y-%m-%d"),
        "cache_last_date": simulation.last_cache_date.strftime("%Y-%m-%d"),
        "n_sessions": int(len(daily)),
        "capital": float(capital),
        "cost_mode": simulation.cost_mode,
        "max_stale_sessions": int(simulation.max_stale_sessions),
        "stale_policy": STALE_POLICY,
        "benchmark_definition": BENCHMARK_DEFINITION,
        "risk_free_rate": RISK_FREE_ANNUAL,
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
    close, volume = load_cache(cache_path)
    target_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    simulation = simulate(
        weights, as_of, close, volume, capital, cost_mode, max_stale_sessions,
    )

    nav_path = out_dir / "nav.csv"
    summary_path = out_dir / "summary.json"
    existing_lines: list[str] = []
    warnings = list(simulation.warnings)
    if summary_path.is_file():
        _check_existing_summary(
            summary_path, target_sha256, capital, cost_mode, max_stale_sessions,
        )
    if nav_path.is_file():
        existing_lines = _read_existing_nav(nav_path)
        if not summary_path.is_file():
            warnings.append(
                "existing nav.csv has no summary.json; appended without a "
                "provenance check"
            )

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
        str(target_path),
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
            "consecutive missing closes before a held asset is liquidated to "
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
