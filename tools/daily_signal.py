#!/usr/bin/env python
"""Daily signal tool: cached daily panel -> point-in-time target weights.

Turns the research candidate (EP + low turnover, industry-neutral, monthly,
5% per-name cap) into an executable target-weight file for paper/QMT use.
The tool only *produces weights*; it never talks to a broker and produces no
performance figures.

Point-in-time contract (the reason this tool exists):

* The parquet panel is sliced to ``date <= as_of`` **before** anything else.
  The forward-return panel is ``pct_chg / 100`` shifted by ``-1`` *within that
  slice*, so the as_of row's label (the next session's return) is NaN by
  construction. That row is then dropped explicitly before Rank IC and the
  covariance estimate, so no post-as_of information can enter selection.
* Rank-IC weights use only IC observations dated strictly before ``as_of``
  (the platform's ``combine_ic_weighted`` convention: last 252 observations,
  at least 20 required, signed mean IC, normalised by ``sum(|w|)``).
* Factor processing at each date uses only that date's cross-section
  (winsorise 1/99 -> z-score -> optional static-industry neutralisation).

Portfolio step: one-period mean-variance optimisation with a long-only
constraint, a per-name cap, a one-sided turnover cap, and the R8/R9 cost
terms (20bp linear + 0.5 quadratic on ``w - w_prev``). ``--prev-weights``
reads the previous targets JSON (either a bare ``{code: weight}`` object or
a ``targets.json`` with a ``weights`` object). With no previous book this is
a first build and the turnover cap is not applied (a 30% cap against a zero
book is infeasible and would silently force the equal-weight fallback).

Outputs: ``targets.json`` (as_of, data_end, factors, weights, signal_date,
provenance, diagnostics) and ``targets.csv``
(``code,weight,target_value_cny``). If the data is insufficient the tool
exits non-zero with an ``insufficient`` reason and writes nothing.

Usage::

    python tools/daily_signal.py \
      --cache .cache/baostock/daily_2018-01-01_2025-12-31_500.parquet \
      --as-of 2025-12-31 --factors ep,turnover20d \
      --industry-map .cache/baostock/industry_map.parquet \
      --max-weight 0.05 --max-turnover 0.30 --capital 2000000 \
      --prev-weights prev.json --out-json targets.json --out-csv targets.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from quant_platform.factors.evaluation import rank_ic
from quant_platform.factors.processing import process_factor
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.covariance import estimate_covariance
from quant_platform.portfolio.optimizers import MeanVarianceOptimizer

DEFAULT_CACHE = (
    ROOT / ".cache" / "baostock" / "daily_2018-01-01_2025-12-31_500.parquet"
)
DEFAULT_FACTORS = ("ep", "turnover20d")
SUPPORTED_FACTORS = ("ep", "bp", "turnover20d")

MIN_HISTORY_SESSIONS = 252
IC_LOOKBACK = 252
IC_MIN_OBS = 20
TURNOVER_WINDOW = 20
# A signal dated more than this many calendar days past the last cached
# session is treated as stale (10 days clears the longest A-share holiday
# closures without a trading calendar).
STALE_CACHE_DAYS = 10

COST_BPS = 20.0
TURNOVER_PENALTY = 0.5
RISK_AVERSION = 1.0
COVARIANCE_METHOD = "ledoit_wolf"

WEIGHT_DECIMALS = 10
# Weights below this (0.01% of the book, 200 CNY at 2M capital) are solver
# dust, not positions. They are pruned and the residual becomes cash.
MIN_REPORTED_WEIGHT = 1e-4


class SignalError(RuntimeError):
    """Expected, user-actionable failure (bad input or insufficient data)."""


# ---------------------------------------------------------------------------
# Panel loading and the PIT slice
# ---------------------------------------------------------------------------

def _normalize_code(value: Any) -> str:
    """Normalise an A-share code to its 6-digit string form when numeric."""
    text = str(value).strip()
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def load_panel_frame(path: Path) -> pd.DataFrame:
    """Read the cached daily panel and normalise its MultiIndex.

    The asset level is normalised to 6-digit strings so JSON inputs
    (``{"1": 0.02}``) and cache codes (``"000001"``) line up.
    """
    if not Path(path).is_file():
        raise SignalError(f"cache panel not found: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise SignalError(
            "cache panel must have a two-level (date, asset) MultiIndex; "
            f"got {type(frame.index).__name__}"
        )
    names = list(frame.index.names)
    if set(names) != {"date", "asset"}:
        raise SignalError(
            f"cache panel index levels must be named 'date' and 'asset'; got {names}"
        )
    if names != ["date", "asset"]:
        frame = frame.reorder_levels(["date", "asset"]).sort_index()
    # Date level: datetime, YYYY-MM-DD strings, or integer YYYYMMDD. Integer
    # levels are read as YYYYMMDD, never as epoch nanoseconds.
    dates = frame.index.get_level_values("date")
    if not isinstance(dates, pd.DatetimeIndex):
        values = pd.Index(dates)
        if pd.api.types.is_integer_dtype(values.dtype):
            parsed = pd.to_datetime(
                values.astype(str), format="%Y%m%d", errors="coerce",
            )
        else:
            parsed = pd.to_datetime(values, errors="coerce")
        if pd.isna(parsed).any():
            raise SignalError(
                "cache panel has unparseable values in the 'date' level "
                "(expected datetime, YYYY-MM-DD strings, or YYYYMMDD integers)"
            )
        dates = parsed
    assets = pd.Index(
        [_normalize_code(asset) for asset in frame.index.get_level_values("asset")],
        name="asset",
    )
    frame = frame.set_axis(
        pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])
    ).sort_index()
    if frame.index.duplicated().any():
        duplicated = int(frame.index.duplicated().sum())
        raise SignalError(
            f"cache panel has {duplicated} duplicated (date, asset) rows"
        )
    for column in ("close", "pct_chg"):
        if column not in frame.columns:
            raise SignalError(f"cache panel is missing required column: {column!r}")
    return frame


def slice_to_as_of(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Keep only rows dated on or before ``as_of`` (the PIT cut, step 1)."""
    dates = frame.index.get_level_values("date")
    sliced = frame.loc[np.asarray(dates <= as_of)]
    if sliced.empty:
        raise SignalError(
            f"insufficient data: no sessions on or before {as_of.date()} "
            "in the cache panel"
        )
    return sliced


@dataclass
class Panels:
    """Wide panels derived from the sliced frame.

    ``forward_returns[t]`` is the return *from session t to t+1*
    (``pct_chg/100`` shifted by -1 **within the slice**), so its last row is
    always NaN: the signal date's forward label does not exist yet.
    """

    close: pd.DataFrame
    turnover: pd.DataFrame | None
    pe: pd.DataFrame | None
    pb: pd.DataFrame | None
    forward_returns: pd.DataFrame


def build_panels(frame: pd.DataFrame) -> Panels:
    """Build (date x asset) panels; shift AFTER slicing so no label leaks."""
    close = frame["close"].unstack("asset").sort_index()

    def _panel(column: str) -> pd.DataFrame | None:
        if column not in frame.columns:
            return None
        return (
            frame[column].unstack("asset").reindex_like(close).astype(float)
        )

    pct = (
        frame["pct_chg"].unstack("asset").reindex_like(close).astype(float)
        / 100.0
    )
    return Panels(
        close=close,
        turnover=_panel("turnover"),
        pe=_panel("pe_ttm"),
        pb=_panel("pb_mrq"),
        forward_returns=pct.shift(-1),
    )


def _required_columns(factor_names: list[str]) -> set[str]:
    columns = {"close", "pct_chg"}
    if "turnover20d" in factor_names:
        columns.add("turnover")
    if "ep" in factor_names:
        columns.add("pe_ttm")
    if "bp" in factor_names:
        columns.add("pb_mrq")
    return columns


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def parse_factor_list(raw: str) -> list[str]:
    names: list[str] = []
    for piece in str(raw).split(","):
        name = piece.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise SignalError("--factors produced an empty list")
    return names


def raw_factor_panel(name: str, panels: Panels) -> pd.DataFrame:
    """Raw (unprocessed) factor values, computed date row by date row."""
    if name == "turnover20d":
        if panels.turnover is None:
            raise SignalError("factor 'turnover20d' requires the 'turnover' column")
        return panels.turnover.rolling(
            TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW,
        ).mean()
    if name == "ep":
        if panels.pe is None:
            raise SignalError("factor 'ep' requires the 'pe_ttm' column")
        pe = panels.pe
        return 1.0 / pe.where(pe > 0)
    if name == "bp":
        if panels.pb is None:
            raise SignalError("factor 'bp' requires the 'pb_mrq' column")
        pb = panels.pb
        return 1.0 / pb.where(pb > 0)
    raise SignalError(
        f"unsupported factor {name!r}; supported: {', '.join(SUPPORTED_FACTORS)}"
    )


def process_factor_panel(
    values: pd.DataFrame, sector_map: pd.Series | None,
) -> pd.DataFrame:
    """Winsorise 1/99 -> z-score -> optional static-industry neutralisation."""
    return process_factor(
        values,
        winsorize_enabled=True,
        winsorize_lower=0.01,
        winsorize_upper=0.99,
        standardize_enabled=True,
        standardize_method="zscore",
        neutralize_enabled=sector_map is not None,
        sector_map=sector_map,
    )


def load_industry_map(path: Path, assets: pd.Index) -> pd.Series:
    """Static asset -> industry map ("6-digit code" index, ``industry`` column)."""
    if not Path(path).is_file():
        raise SignalError(f"industry map not found: {path}")
    frame = pd.read_parquet(path)
    if "industry" not in frame.columns:
        raise SignalError(
            f"industry map {path} must have an 'industry' column; "
            f"got {list(frame.columns)}"
        )
    lookup = {
        _normalize_code(code): str(industry)
        for code, industry in frame["industry"].items()
    }
    codes = [_normalize_code(asset) for asset in assets]
    sectors = pd.Series(
        [lookup.get(code, "Unknown") for code in codes],
        index=assets,
        dtype=object,
    )
    return sectors.astype(str)


def compute_ic_weights(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    as_of: pd.Timestamp,
    warnings: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """PIT Rank-IC factor weights, mirroring ``combine_ic_weighted``.

    For each factor: mean Rank IC over the last ``IC_LOOKBACK`` observations
    dated strictly before ``as_of``. Fewer than ``IC_MIN_OBS`` observations
    means "no history" and a weight of zero. Weights are signed and
    normalised by ``sum(|w|)`` (the platform convention), so a factor whose
    IC is negative contributes an inverted tilt.
    """
    raw_weights: dict[str, float] = {}
    stats: dict[str, dict[str, Any]] = {}

    for name, panel in factors.items():
        try:
            ic = rank_ic(panel, forward_returns, name=name)
        except Exception as exc:  # noqa: BLE001 - reported, never guessed
            raise SignalError(
                f"factor {name!r}: Rank IC computation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        history = ic[ic.index < as_of]
        n_obs = int(history.notna().sum())
        if n_obs < IC_MIN_OBS:
            raw_weights[name] = 0.0
            stats[name] = {
                "mean_ic": None,
                "n_obs": n_obs,
                "weight": 0.0,
                "window_start": None,
                "window_end": None,
            }
            warnings.append(
                f"factor {name!r}: {n_obs} Rank IC observations before "
                f"{as_of.date()} (< {IC_MIN_OBS}); weight set to 0"
            )
            continue
        recent = history.iloc[-IC_LOOKBACK:]
        mean_ic = float(recent.mean())
        raw_weights[name] = mean_ic
        stats[name] = {
            "mean_ic": mean_ic,
            "n_obs": int(recent.notna().sum()),
            "window_start": recent.index[0].date().isoformat(),
            "window_end": recent.index[-1].date().isoformat(),
        }

    has_history = {name: stats[name]["n_obs"] >= IC_MIN_OBS for name in factors}
    if not any(has_history.values()):
        raise SignalError(
            "insufficient factor IC history: no factor has "
            f">= {IC_MIN_OBS} Rank IC observations before {as_of.date()}"
        )
    total_abs = sum(abs(raw_weights[name]) for name in factors)
    if total_abs < 1e-12:
        names = [name for name in factors if has_history[name]]
        weights = {
            name: (1.0 / len(names) if name in names else 0.0)
            for name in factors
        }
        warnings.append(
            "all mean Rank ICs are zero; equal-weighting factors with history"
        )
    else:
        weights = {name: raw_weights[name] / total_abs for name in factors}

    for name in factors:
        stats[name]["weight"] = float(weights[name])
    window = None
    usable_names = [name for name in factors if has_history[name]]
    if usable_names:
        # Report one coherent window instead of a min-start/max-end union that
        # may not belong to any single factor: use the factor with the most
        # IC observations (all of them end before as_of).
        best = max(usable_names, key=lambda name: stats[name]["n_obs"])
        window = {
            "start": stats[best]["window_start"],
            "end": stats[best]["window_end"],
            "n": stats[best]["n_obs"],
            "factor": best,
        }
    diagnostics: dict[str, Any] = {
        "lookback": IC_LOOKBACK,
        "min_obs": IC_MIN_OBS,
        "window": window,
        "per_factor": stats,
    }
    return weights, diagnostics


def build_signal(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
    signal_date: pd.Timestamp,
) -> pd.Series:
    """Composite ``sum(w_f * z_f) / sum(|w_f|)`` per asset.

    Missing factor values are filled with 0 -- the cross-sectional mean of a
    z-scored factor -- and the denominator is the *fixed* sum of |weight|, not
    the weight of the factors that happen to be present for each asset. This
    mirrors ``alpha.combination._build_row`` and the taskbook's
    ``signal = sum(w_f * z_f)``: a missing value must be neutral, not a
    promotion (renormalising per asset would hand an asset that lacks the
    dominant factor the full weight of the remaining factors). Only assets
    whose values are NaN for *every* weighted factor get no weight (NaN).
    """
    rows: dict[str, pd.Series] = {}
    for name, panel in factors.items():
        if signal_date not in panel.index:
            raise SignalError(
                f"factor {name!r} has no row on signal date "
                f"{signal_date.date()}"
            )
        rows[name] = panel.loc[signal_date]
    cross_section = pd.DataFrame(rows)
    total_abs = sum(
        abs(float(weight)) for weight in weights.values()
        if abs(float(weight)) > 0.0
    )
    if total_abs < 1e-12:
        raise SignalError("all factor IC weights are zero; cannot build a signal")
    numerator = pd.Series(0.0, index=cross_section.index)
    present_any = pd.Series(False, index=cross_section.index)
    for name in factors:
        weight = float(weights.get(name, 0.0))
        if abs(weight) < 1e-12:
            continue
        values = cross_section[name]
        present = values.notna()
        numerator.loc[present] += weight * values.loc[present]
        present_any |= present
    composite = (numerator / total_abs).astype(float)
    composite[~present_any] = np.nan
    if not np.isfinite(composite.to_numpy()).any():
        raise SignalError(
            f"insufficient factor coverage on signal date "
            f"{signal_date.date()}: every asset is NaN for all requested factors"
        )
    return composite


# ---------------------------------------------------------------------------
# Previous book and portfolio step
# ---------------------------------------------------------------------------

def load_prev_weights(path: Path) -> pd.Series | None:
    """Read previous target weights from JSON (bare map or ``targets.json``)."""
    if not Path(path).is_file():
        raise SignalError(f"prev weights file not found: {path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SignalError(f"prev weights {path} is not valid JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
        payload = payload["weights"]
    if not isinstance(payload, dict):
        raise SignalError(
            "prev weights must be a JSON object {code: weight} or a "
            "targets.json containing a 'weights' object"
        )
    values: dict[str, float] = {}
    for code, weight in payload.items():
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise SignalError(
                f"prev weight for {code!r} is not numeric: {weight!r}"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise SignalError(
                f"prev weight for {code!r} must be finite and >= 0 "
                f"(long-only tool); got {value!r}"
            )
        values[_normalize_code(code)] = value
    series = pd.Series(values, dtype=float).sort_index()
    total = float(series.sum())
    if total > 1.0 + 1e-6:
        raise SignalError(
            f"prev weights sum to {total:.6f} > 1; refusing to treat a "
            "levered book as long-only"
        )
    return series


def one_sided_turnover(
    weights: pd.Series, prev_weights: pd.Series | None,
) -> float:
    """One-sided turnover ``0.5 * sum(|w - w_prev|)`` over both indices."""
    if prev_weights is None:
        previous = pd.Series(0.0, index=weights.index)
    else:
        previous = prev_weights.reindex(
            weights.index.union(prev_weights.index), fill_value=0.0,
        )
    current = weights.reindex(previous.index, fill_value=0.0)
    return 0.5 * float((current - previous).abs().sum())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_signal(args: argparse.Namespace) -> dict[str, Any]:
    """Compute the target book and return the JSON-ready payload."""
    warnings: list[str] = []
    cache_path = Path(args.cache)
    frame = load_panel_frame(cache_path)

    all_dates = frame.index.get_level_values("date")
    if args.as_of:
        try:
            as_of = pd.Timestamp(args.as_of)
        except (ValueError, TypeError) as exc:
            raise SignalError(
                f"could not parse --as-of {args.as_of!r}: {exc}"
            ) from exc
        if pd.isna(as_of):
            raise SignalError(f"could not parse --as-of {args.as_of!r}")
    else:
        as_of = pd.Timestamp(all_dates.max())

    sliced = slice_to_as_of(frame, as_of)
    sliced_dates = sliced.index.get_level_values("date").unique()
    n_sessions = int(len(sliced_dates))
    if n_sessions < MIN_HISTORY_SESSIONS:
        raise SignalError(
            f"insufficient data: {n_sessions} sessions on or before "
            f"{as_of.date()}, need >= {MIN_HISTORY_SESSIONS}"
        )

    factor_names = parse_factor_list(args.factors)
    unsupported = [
        name for name in factor_names if name not in SUPPORTED_FACTORS
    ]
    if unsupported:
        raise SignalError(
            f"unsupported factor(s) {unsupported}; supported: "
            f"{', '.join(SUPPORTED_FACTORS)}"
        )
    missing = sorted(_required_columns(factor_names).difference(sliced.columns))
    if missing:
        raise SignalError(f"cache panel is missing required column(s): {missing}")

    sector_map = None
    industry_source = None
    if args.industry_map:
        industry_path = Path(args.industry_map)
        sector_map = load_industry_map(
            industry_path, pd.Index(sorted(sliced.index.get_level_values("asset").unique())),
        )
        industry_source = str(industry_path)
    else:
        warnings.append(
            "no industry map supplied (--industry-map); industry "
            "neutralization skipped"
        )

    panels = build_panels(sliced)
    signal_date = pd.Timestamp(panels.close.index[-1])
    data_lag_days = int(
        (as_of.normalize() - signal_date.normalize()).days
    )
    if data_lag_days > STALE_CACHE_DAYS:
        warnings.append(
            f"stale cache: --as-of {as_of.date()} but the last cached session "
            f"is {signal_date.date()} ({data_lag_days} calendar days earlier); "
            "the signal is computed from older data"
        )

    # The signal date's forward label (next session's return) must never be
    # used. It is already NaN after slicing, and we drop the row explicitly.
    known_returns = panels.forward_returns.drop(
        index=signal_date, errors="ignore",
    )
    if signal_date in panels.forward_returns.index:
        leaked = panels.forward_returns.loc[signal_date]
        if leaked.notna().any():
            raise SignalError(
                "internal PIT check failed: the forward label on the signal "
                f"date {signal_date.date()} is not NaN"
            )

    raw_factors: dict[str, pd.DataFrame] = {}
    factor_skipped: list[list[str]] = []
    for name in factor_names:
        values = raw_factor_panel(name, panels)
        covered = int(values.loc[signal_date].notna().sum())
        if covered == 0:
            factor_skipped.append(
                [name, f"no values on signal date {signal_date.date()}"]
            )
            warnings.append(
                f"factor {name!r} has no coverage on {signal_date.date()}; skipped"
            )
            continue
        raw_factors[name] = values
    if not raw_factors:
        raise SignalError(
            f"insufficient factor coverage on {signal_date.date()}: every "
            f"requested factor is empty ({factor_skipped})"
        )

    processed = {
        name: process_factor_panel(values, sector_map)
        for name, values in raw_factors.items()
    }
    ic_weights, ic_diagnostics = compute_ic_weights(
        processed, known_returns, as_of, warnings,
    )
    composite = build_signal(processed, ic_weights, signal_date)
    composite_values = composite.dropna()
    # MeanVarianceOptimizer expects the platform's rank alpha in [-0.5, 0.5]
    # (AlphaPipeline convention); the composite z is the delivered signal.
    alpha = composite.rank(pct=True) - 0.5

    cov_input = known_returns.dropna(how="all")
    try:
        cov_matrix = estimate_covariance(
            cov_input, method=COVARIANCE_METHOD, lookback=IC_LOOKBACK,
        )
    except ValueError as exc:
        raise SignalError(
            f"insufficient return history for covariance: {exc}"
        ) from exc

    constraints = PortfolioConstraints(
        long_only=True,
        max_weight=float(args.max_weight),
        max_sector_exposure=1.0,  # no sector cap; risk control comes via factors
        max_turnover=float(args.max_turnover),
        risk_aversion=RISK_AVERSION,
        transaction_cost_bps=COST_BPS,
        turnover_penalty=TURNOVER_PENALTY,
    )
    optimizer = MeanVarianceOptimizer(constraints)

    prev_weights = None
    if args.prev_weights:
        prev_weights = load_prev_weights(Path(args.prev_weights))
    first_build = prev_weights is None or float(prev_weights.abs().sum()) <= 1e-12
    prev_for_opt = None if first_build else prev_weights

    raw_targets = optimizer.optimize(
        alpha,
        cov_matrix=cov_matrix,
        prices=None,
        prev_weights=prev_for_opt,
        sector_map=None,
    )
    if optimizer.fallback_count:
        warnings.append(
            "optimizer fell back to equal weight: "
            f"{optimizer.last_fallback_reason}"
        )

    values = raw_targets.reindex(alpha.index, fill_value=0.0).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SignalError(
            "optimizer produced non-finite weights; refusing to write targets"
        )
    targets = pd.Series(values, index=alpha.index).clip(lower=0.0)
    total = float(targets.sum())
    if total <= 1e-9:
        raise SignalError(
            "optimizer produced an empty portfolio; refusing to write targets"
        )
    if total > 1.0:
        targets = targets / total
    targets = targets.round(WEIGHT_DECIMALS)
    pruned = targets <= MIN_REPORTED_WEIGHT
    pruned_weight = float(targets[pruned].sum())
    targets = targets[~pruned]
    if len(targets) == 0:
        raise SignalError(
            "optimizer produced no weights above the reporting threshold"
        )
    max_actual = float(targets.max())
    if max_actual > constraints.max_weight + 1e-6:
        over_cap_prev = None
        if prev_weights is not None:
            over_cap_prev = prev_weights[
                prev_weights > constraints.max_weight + 1e-6
            ]
        if over_cap_prev is not None and len(over_cap_prev):
            top = ", ".join(
                f"{code}={weight:.4f}"
                for code, weight in over_cap_prev.sort_values(
                    ascending=False,
                ).head(5).items()
            )
            raise SignalError(
                "prev weights exceed --max-weight "
                f"{constraints.max_weight:.4f} ({top}); the optimiser could "
                "not produce a compliant book within the "
                f"{constraints.max_turnover:.2f} one-sided turnover cap (it "
                "fell back to equal weight and kept the over-cap holdings); "
                "provide a compliant prev book or raise --max-turnover"
            )
        raise SignalError(
            "internal consistency check failed: max weight "
            f"{max_actual:.8f} exceeds --max-weight "
            f"{constraints.max_weight:.8f}"
        )

    turnover = one_sided_turnover(targets, prev_weights)
    if first_build and turnover > constraints.max_turnover + 1e-6:
        warnings.append(
            f"first build: one-sided turnover {turnover:.4f} against a zero "
            f"book exceeds the {constraints.max_turnover:.4f} cap; the cap is "
            "not applied to a first build (nothing to trade from; the "
            "platform engine also passes prev_weights=None on its first "
            "rebalance)"
        )
    if not first_build and turnover > constraints.max_turnover + 1e-6:
        warnings.append(
            f"turnover {turnover:.4f} exceeds the cap "
            f"{constraints.max_turnover:.4f} after rounding"
        )

    gross = float(targets.sum())
    cash = max(0.0, 1.0 - gross)
    weights_map = {
        str(code): float(weight) for code, weight in targets.sort_index().items()
    }
    capital = float(args.capital)

    payload: dict[str, Any] = {
        "as_of": as_of.date().isoformat(),
        "data_end": signal_date.date().isoformat(),
        "factors": list(factor_names),
        "weights": weights_map,
        "signal_date": signal_date.date().isoformat(),
        "provenance": {
            "tool": "tools/daily_signal.py",
            "cache": str(cache_path),
            "industry_map": industry_source,
            "industry_neutralization": sector_map is not None,
            "prev_weights": str(args.prev_weights) if args.prev_weights else None,
            "capital_cny": capital,
            "max_weight": float(args.max_weight),
            "max_turnover": float(args.max_turnover),
            "risk_aversion": RISK_AVERSION,
            "transaction_cost_bps": COST_BPS,
            "turnover_penalty": TURNOVER_PENALTY,
            "covariance": COVARIANCE_METHOD,
            "covariance_lookback": IC_LOOKBACK,
            "ic_lookback": IC_LOOKBACK,
            "ic_min_obs": IC_MIN_OBS,
            "winsorize": [0.01, 0.99],
            "standardize": "zscore",
            "returns_convention": (
                "pct_chg/100 shift(-1) computed within the panel sliced to "
                "date <= as_of; the signal date's forward label is NaN and "
                "excluded from IC and covariance"
            ),
        },
        "diagnostics": {
            "history_sessions": n_sessions,
            "data_lag_days": data_lag_days,
            "universe_assets": int(alpha.size),
            "candidates": int(alpha.notna().sum()),
            "n_holdings": int(len(targets)),
            "max_weight_actual": max_actual,
            "gross_weight": gross,
            "cash_weight": cash,
            "turnover": turnover,
            "turnover_definition": "one_sided = 0.5 * sum(|w - w_prev|)",
            "first_build": bool(first_build),
            "min_reported_weight": MIN_REPORTED_WEIGHT,
            "pruned_weight": pruned_weight,
            "factor_skipped": factor_skipped,
            "factor_ic": ic_diagnostics,
            "signal": {
                "composite_min": float(composite_values.min()),
                "composite_max": float(composite_values.max()),
                "composite_mean": float(composite_values.mean()),
                "alpha_convention": (
                    "cross-sectional percentile rank - 0.5 "
                    "(MeanVarianceOptimizer input; composite z is the "
                    "delivered signal)"
                ),
            },
            "covariance_assets": int(cov_matrix.shape[0]),
            "covariance_sessions": int(min(len(cov_input), IC_LOOKBACK)),
            "covariance_input_sessions": int(len(cov_input)),
            "optimizer": {
                "name": "MeanVarianceOptimizer",
                "fallback_count": int(optimizer.fallback_count),
                "fallback_reason": optimizer.last_fallback_reason,
            },
            "warnings": warnings,
        },
    }
    return payload


def format_console(payload: dict[str, Any]) -> str:
    diagnostics = payload["diagnostics"]
    provenance = payload["provenance"]
    capital = float(provenance["capital_cny"])
    ic_per_factor = diagnostics["factor_ic"]["per_factor"]
    ic_text = ", ".join(
        f"{name}={values['weight']:+.4f} (IC {values['mean_ic']:+.4f})"
        if values["mean_ic"] is not None else f"{name}=0.0000 (no history)"
        for name, values in ic_per_factor.items()
    )
    window = diagnostics["factor_ic"]["window"] or {}
    lines = [
        f"Daily signal as of {payload['as_of']} "
        f"(signal date {payload['signal_date']}, data end {payload['data_end']})",
        f"Factors: {', '.join(payload['factors'])} | IC weights: {ic_text}",
        f"Rank IC window: {window.get('start')} .. {window.get('end')} "
        f"({window.get('n', 0)} obs, all strictly before as_of)",
        f"Sessions <= as_of: {diagnostics['history_sessions']} | universe: "
        f"{diagnostics['universe_assets']} assets | candidates: "
        f"{diagnostics['candidates']}",
        f"Covariance: {diagnostics['covariance_assets']} assets x "
        f"{diagnostics['covariance_sessions']} sessions "
        f"({provenance['covariance']}, last {provenance['covariance_lookback']})",
        f"Holdings: {diagnostics['n_holdings']} | max weight: "
        f"{diagnostics['max_weight_actual']:.2%} | gross: "
        f"{diagnostics['gross_weight']:.2%} | cash: "
        f"{diagnostics['cash_weight']:.2%}",
        f"Estimated turnover (one-sided): {diagnostics['turnover']:.2%} "
        f"(cap {provenance['max_turnover']:.0%}, first build: "
        f"{'yes' if diagnostics['first_build'] else 'no'})",
        f"Optimizer fallbacks: {diagnostics['optimizer']['fallback_count']}"
        + (
            f" ({diagnostics['optimizer']['fallback_reason']})"
            if diagnostics["optimizer"]["fallback_reason"] else ""
        ),
    ]
    if diagnostics["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in diagnostics["warnings"])
    lines.append("Top 10 weights:")
    ranked = sorted(payload["weights"].items(), key=lambda item: (-item[1], item[0]))
    for code, weight in ranked[:10]:
        lines.append(
            f"  {code}  {weight:8.4%}  {weight * capital:14,.2f} CNY"
        )
    return "\n".join(lines)


def write_outputs(
    payload: dict[str, Any], json_path: Path, csv_path: Path,
) -> None:
    """Write ``targets.json`` and ``targets.csv`` (only after success)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    json_path.write_text(text + "\n", encoding="utf-8")
    capital = float(payload["provenance"]["capital_cny"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "weight", "target_value_cny"])
        for code, weight in sorted(payload["weights"].items()):
            writer.writerow([code, f"{weight:.10f}", f"{weight * capital:.2f}"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Produce point-in-time target portfolio weights from a cached "
            "daily panel (no broker, no performance reporting)."
        ),
    )
    parser.add_argument(
        "--cache", default=str(DEFAULT_CACHE),
        help="cached (date, asset) daily parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--as-of", default=None,
        help="last session whose data may be used (YYYY-MM-DD or YYYYMMDD); "
             "default: the last date in the cache",
    )
    parser.add_argument(
        "--factors", default=",".join(DEFAULT_FACTORS),
        help=f"comma-separated factors; supported: {', '.join(SUPPORTED_FACTORS)} "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--industry-map", default=None,
        help="static industry map parquet (6-digit code index, 'industry' "
             "column) for industry-neutralised factors; omitted = neutralization "
             "skipped with a warning",
    )
    parser.add_argument(
        "--max-weight", type=float, default=0.05,
        help="per-asset weight cap (default: %(default)s)",
    )
    parser.add_argument(
        "--max-turnover", type=float, default=0.30,
        help="one-sided turnover cap vs --prev-weights (default: %(default)s)",
    )
    parser.add_argument(
        "--capital", type=float, default=2_000_000.0,
        help="capital in CNY used for target_value_cny (default: %(default)s)",
    )
    parser.add_argument(
        "--prev-weights", default=None,
        help="previous target weights JSON (bare {code: weight} or a "
             "targets.json with 'weights'); omitted = first build",
    )
    parser.add_argument(
        "--out-json", default="targets.json",
        help="targets JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--out-csv", default="targets.csv",
        help="targets CSV path (default: %(default)s)",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 < float(args.max_weight) <= 1.0:
        parser.error("--max-weight must be in (0, 1]")
    if not 0.0 < float(args.max_turnover) <= 1.0:
        parser.error("--max-turnover must be in (0, 1]")
    if not math.isfinite(float(args.capital)) or float(args.capital) <= 0.0:
        parser.error("--capital must be a finite number > 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        payload = run_signal(args)
        json_path = Path(args.out_json)
        csv_path = Path(args.out_csv)
        write_outputs(payload, json_path, csv_path)
    except SignalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_console(payload))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
