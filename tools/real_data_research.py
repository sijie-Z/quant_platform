#!/usr/bin/env python
"""Real-data research tool: cached panel -> factors -> walk-forward -> report.

Repo-native, re-runnable form of the validated one-off research scripts in
``.scratch/orchestrate/`` (real_run.py / real_variants2.py / real_variants4.py).
Only platform modules are imported; nothing comes from ``.scratch``.

Pipeline semantics (deliberately identical to the validated scratch runs):

* Cache: parquet with a ``(date, asset)`` MultiIndex and columns including
  ``open/high/low/close/preclose/volume/amount/turnover/pct_chg/pe_ttm/pb_mrq``.
* Returns: ``returns[t] = pct_chg[t + 1] / 100`` via ``shift(-1)`` -- row ``t``
  is the next session's return (the platform's forward-return convention).
* Adjusted price path: ``cumprod(1 + pct_chg/100) * 100`` (Baostock ``pct_chg``
  already uses an ex-rights preclose, so compounding it avoids dividend jumps),
  blanked where ``close`` is missing.
* Factors: registered technical factors, winsorized + z-scored, optionally
  industry-neutralized with a static industry map parquet.
* Walk-forward: expanding windows. Every fold selects its factors from that
  fold's training window only (Rank-IC t-test + BH-FDR, top-k by |ICIR|);
  purge and embargo are applied by ``WalkForwardValidator``.
* Costs: impact model (composite) or fixed slippage; commission 0.03%; stamp
  tax follows the statutory date schedule (``stamp_tax=None``).
* Portfolio: MVO with max-weight, turnover cap and cost-aware penalty
  (R8/R9 wiring); monthly rebalance.
* Gates (R9, fixed in R19a): ``tradable_mask`` = close present and volume > 0;
  limit moves use the *same-day* ``pct_chg`` (``|pct_chg| > 9.8%``), i.e. the
  move already realized when the engine trades at that close, matching
  ``data/pipeline.py`` ``is_limit_up``. Using the forward return panel here
  (``returns[t] = pct_chg[t+1]``) let day t+1's limit move decide day t's
  fill -- a one-day look-ahead. ``daily_volume``/``raw_prices`` travel through
  the gates adapter to the engine.
* Benchmarks (R19a): daily equal-weight (rebalanced every session) and
  equal-weight buy & hold, both computed over the *same* OOS window as the
  strategy. Benchmark annual figures are total returns; excess is reported
  separately and never substituted for a total.
* DSR (R19a): deflated Sharpe is computed on excess daily returns
  (``oos - risk_free/252``, 3%/yr); ``--num-trials`` overrides the default
  fold x factor trial count.
* Industry map (when supplied): a *static latest* classification with no
  historical effective dates -- mild look-ahead, disclosed in the report.

Outputs: console table plus JSON and Markdown reports. By default they are
written to ``real_data_research_<date>.{json,md}`` in the current directory.
The JSON excludes wall-clock runtime so identical inputs produce byte-identical
reports (the runtime is still printed and written to the Markdown).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date as _date
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
from quant_platform.backtest.metrics import all_metrics
from quant_platform.backtest.walkforward import WalkForwardValidator
from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.factors.processing import process_factor
from quant_platform.factors.registry import get_registry
from quant_platform.factors.technical import register_all as register_technical
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.research.validation import FactorValidator
from quant_platform.strategy.research_validation import select_walk_forward_factors

DEFAULT_FACTORS = (
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "volatility_20d",
    "volatility_60d",
    "turnover_20d",
    "rsi_14d",
    "amplitude_20d",
    "efficiency_ratio",
    "macd",
    "pure_volatility",
)
DEFAULT_CACHE = ROOT / ".cache" / "baostock" / "daily_2018-01-01_2025-12-31_500.parquet"
DEFAULT_MAX_SECTOR_EXPOSURE = 0.60  # R8/R9 parity with the scratch runs
MIN_FACTOR_COVERAGE = 0.20

ENGINE_LOGGER_NAME = "quant_platform.backtest.engine"
OPTIMIZER_LOGGER_NAME = "quant_platform.portfolio.optimizers"
LIMIT_MOVE_THRESHOLD = 0.098  # 10% board approximation (cache is all 000xxx)
RISK_FREE_ANNUAL = 0.03
TRADING_DAYS_PER_YEAR = 252.0
INDUSTRY_MAP_NOTE = (
    "static latest industry classification (no historical effective dates); "
    "mild look-ahead when used for neutralization"
)


class ResearchError(RuntimeError):
    """Expected, user-actionable failure (bad input, insufficient data)."""


def _finite(value: Any) -> float | None:
    """Return ``value`` as a float when finite, else ``None`` (JSON-safe)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Panel loading and conventions
# ---------------------------------------------------------------------------

def load_panel_frame(path: Path) -> pd.DataFrame:
    """Read the cached daily panel and normalise its MultiIndex order."""
    if not Path(path).is_file():
        raise ResearchError(f"cache panel not found: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ResearchError(
            "cache panel must have a two-level (date, asset) MultiIndex; "
            f"got {type(frame.index).__name__}"
        )
    names = list(frame.index.names)
    if set(names) != {"date", "asset"}:
        raise ResearchError(
            f"cache panel index levels must be named 'date' and 'asset'; got {names}"
        )
    if names != ["date", "asset"]:
        frame = frame.reorder_levels(["date", "asset"]).sort_index()
    # Some parquet writers store the date level as strings; coerce it here so a
    # string-indexed cache fails nowhere deep inside the engine (whose fold and
    # metric code calls .year on the index). Integer levels are read as
    # YYYYMMDD, never as epoch nanoseconds: pd.to_datetime would otherwise
    # silently turn 20240102 into 1970-01-01T00:00:00.020240102.
    dates = frame.index.get_level_values("date")
    if not isinstance(dates, pd.DatetimeIndex):
        values = pd.Index(dates)
        if pd.api.types.is_integer_dtype(values.dtype):
            parsed = pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")
        else:
            parsed = pd.to_datetime(values, errors="coerce")
        if pd.isna(parsed).any():
            raise ResearchError(
                "cache panel has unparseable values in the 'date' level "
                "(expected datetime, YYYY-MM-DD strings, or YYYYMMDD integers)"
            )
        frame = frame.set_axis(
            pd.MultiIndex.from_arrays(
                [parsed, frame.index.get_level_values("asset")],
                names=["date", "asset"],
            )
        ).sort_index()
    required = {"close", "volume", "turnover", "pct_chg"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ResearchError(f"cache panel is missing required column(s): {missing}")
    return frame


def build_panels(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """(date x asset) close / volume / turnover / returns / adjusted prices / pct.

    ``returns`` is the *next-session* return (``pct_chg/100`` shifted by -1),
    matching the platform's forward-return convention used by IC and the
    backtest. ``pct`` is the same-day move (``pct_chg/100`` on the row's own
    date) and is what execution gates must use: the engine trades at day d's
    close and may only look at the move already realized on day d. ``adj``
    compounds the same ``pct_chg`` into a dividend-jump-free price path and is
    blanked wherever the raw close is missing.
    """
    close = frame["close"].unstack("asset").sort_index()
    volume = frame["volume"].unstack("asset").reindex_like(close).astype(float)
    turnover = frame["turnover"].unstack("asset").reindex_like(close).astype(float)
    pct = frame["pct_chg"].unstack("asset").reindex_like(close).astype(float) / 100.0
    returns = pct.shift(-1)
    adj = (1.0 + pct.fillna(0.0)).cumprod() * 100.0
    adj = adj.where(close.notna())
    return close, volume, turnover, returns, adj, pct


def slice_panels(
    panels: tuple[pd.DataFrame, ...], start: str | None, end: str | None,
) -> tuple[pd.DataFrame, ...]:
    """Date-slice every panel on the shared trading-day axis."""
    close = panels[0]
    mask = np.ones(len(close), dtype=bool)
    index = close.index
    if start:
        mask &= index >= pd.Timestamp(start)
    if end:
        mask &= index <= pd.Timestamp(end)
    if not mask.any():
        raise ResearchError(
            f"no sessions in requested range {start or 'start'} .. {end or 'end'}"
        )
    return tuple(panel.loc[mask] for panel in panels)


def build_execution_gates(
    close: pd.DataFrame, volume: pd.DataFrame, pct: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Execution-day gate panels for the engine.

    ``limit_up``/``limit_down`` are indexed by the *execution* day and hold
    that day's own limit state (``pct_chg/100`` on the same row), never the
    forward return panel: the engine trades at day d's close and must decide
    with information available then. This mirrors ``data/pipeline.py``'s
    ``is_limit_up`` (``close.pct_change()`` on the same date) for 10%-limit
    boards. The 9.8% threshold is deliberately the taskbook's flat
    approximation: the current cache is entirely 000xxx main-board names. A
    cache with ChiNext/STAR (20%) or Beijing (30%) names would need the
    board-specific thresholds from ``shared/symbols.py``.
    """
    return {
        "tradable_mask": close.notna() & (volume.fillna(0) > 0),
        "limit_up": pct > LIMIT_MOVE_THRESHOLD,
        "limit_down": pct < -LIMIT_MOVE_THRESHOLD,
        "daily_volume": volume,
        "raw_prices": close,
    }


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def load_industry_map(path: Path, assets: pd.Index) -> pd.Series:
    """Static asset -> industry map ("6-digit code" index, ``industry`` column).

    The map has no historical effective dates: it is the latest classification
    applied to every backtest date, which is a mild look-ahead when used for
    neutralization. Disclosed in the report (``INDUSTRY_MAP_NOTE``); the
    platform has no historical sector data to do better offline.
    """
    if not Path(path).is_file():
        raise ResearchError(f"industry map not found: {path}")
    frame = pd.read_parquet(path)
    if "industry" not in frame.columns:
        raise ResearchError(
            f"industry map {path} must have an 'industry' column; got {list(frame.columns)}"
        )
    codes = frame.index.astype(str).str.zfill(6)
    lookup = dict(zip(codes, frame["industry"].astype(str), strict=True))
    sectors = pd.Series(
        {str(asset).zfill(6): lookup.get(str(asset).zfill(6), "Unknown") for asset in assets},
        index=assets,
        dtype=object,
    )
    return sectors.fillna("Unknown").astype(str)


def build_factors(
    adj: pd.DataFrame,
    volume: pd.DataFrame,
    turnover: pd.DataFrame,
    names: list[str],
    *,
    sector_map: pd.Series | None = None,
    strict: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[list[str]]]:
    """Compute + process the requested technical factors.

    A factor is skipped (recorded in the returned list) when it is unknown,
    raises, or covers less than 20% of the panel. ``strict`` (explicit
    ``--factors``) turns unknown names and exceptions into errors instead of
    silent skips; this is the difference between a default set degrading
    gracefully on a short panel and a user typo going unnoticed.
    """
    register_technical()
    registry = get_registry()
    available = set(registry.list_all())
    raw: dict[str, pd.DataFrame] = {}
    skipped: list[list[str]] = []
    for name in names:
        if name not in available:
            message = f"factor {name!r} is not registered (available: {sorted(available)})"
            if strict:
                raise ResearchError(message)
            skipped.append([name, "not registered"])
            continue
        try:
            instance = registry.get(name)()
            result = instance.run(adj, volume=volume, turnover=turnover)
            values = result.values if hasattr(result, "values") else result
            values = pd.DataFrame(values).reindex_like(adj)
        except Exception as exc:  # noqa: BLE001 - reported, never fabricated
            if strict:
                raise ResearchError(
                    f"factor {name!r} failed: {type(exc).__name__}: {exc}"
                ) from exc
            skipped.append([name, f"{type(exc).__name__}: {exc}"])
            continue
        coverage = float(values.notna().to_numpy().mean())
        if coverage < MIN_FACTOR_COVERAGE:
            skipped.append([name, f"coverage {coverage:.1%} < {MIN_FACTOR_COVERAGE:.0%}"])
            continue
        raw[name] = values

    processed: dict[str, pd.DataFrame] = {}
    for name, values in raw.items():
        processed[name] = process_factor(
            values,
            winsorize_enabled=True,
            standardize_enabled=True,
            neutralize_enabled=sector_map is not None,
            sector_map=sector_map,
        )
    return processed, skipped


# ---------------------------------------------------------------------------
# Walk-forward selection and run
# ---------------------------------------------------------------------------

class _OptimizerFallbackCounter(logging.Handler):
    """Count engine optimizer fallbacks across all folds.

    The engine logs ``Rebalance ...: optimizer fallback #N - reason`` on
    ``quant_platform.backtest.engine``; the optimizer logs
    ``Optimizer fallback to equal weight: reason`` on the *sibling* logger
    ``quant_platform.portfolio.optimizers``. The old handler listened only on
    the engine logger, so a fallback seen only through the optimizer logger
    was invisible. Both loggers are attached now. One fallback emits one line
    per logger, so the engine tally (the line carrying ``#N``) and the
    optimizer tally are kept separately and ``count`` reports the larger --
    never the sum, which would double-count, and never zero when either
    logger is filtered.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.engine_count = 0
        self.optimizer_count = 0

    @property
    def count(self) -> int:
        return max(self.engine_count, self.optimizer_count)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not break the run
            return
        lowered = message.lower()
        if "optimizer fallback #" in lowered:
            self.engine_count += 1
        elif (
            record.name == OPTIMIZER_LOGGER_NAME
            and "optimizer fallback" in lowered
        ):
            self.optimizer_count += 1


def make_factor_selector(state: dict[str, dict], max_factors: int):
    """Build the per-fold selector: train-window IC -> t-test + BH-FDR -> top-k.

    ``fold_info["train_positions"]`` already excludes embargo days and the test
    window is never part of the training slice, so no test-period information
    enters the selection.
    """

    def _select(fold_info, factors_panel, forward_returns):
        positions = fold_info["train_positions"]
        fold_id = int(fold_info["fold"])
        train_factors = {name: panel.iloc[positions] for name, panel in factors_panel.items()}
        train_returns = forward_returns.iloc[positions]
        train_ic: dict[str, dict[str, float]] = {}
        for name, panel in train_factors.items():
            try:
                summary = ic_summary(rank_ic(panel, train_returns))
            except Exception:  # noqa: BLE001 - a broken factor is simply not testable
                continue
            train_ic[name] = {
                "mean_ic": summary["mean_ic"],
                "std_ic": summary["std_ic"],
                "icir": summary["icir"],
            }
        selection = select_walk_forward_factors(
            train_ic, train_factors, train_returns, max_factors=max_factors,
        )
        chosen = list(selection["selected"])
        state["per_fold"][fold_id] = {
            "selected": chosen,
            "survivors": list(selection["survivors"]),
            "fallback": bool(selection["fallback"]),
            "selection_warning": selection.get("selection_warning"),
            "n_obs": {name: int(n) for name, n in selection["n_obs"].items()},
            "p_values": {name: float(p) for name, p in selection["p_values"].items()},
        }
        return chosen

    return _select


def _build_cost_model(cost_mode: str) -> CostModel:
    slippage_model = "impact" if cost_mode == "impact" else "fixed"
    return CostModel(
        commission=0.0003,
        stamp_tax=None,  # statutory date schedule: 0.1% before 2023-08-28, 0.05% after
        slippage=0.001,
        slippage_model=slippage_model,
        impact_model="composite",
    )


def _build_constraints(args: argparse.Namespace) -> PortfolioConstraints:
    return PortfolioConstraints(
        long_only=True,
        max_weight=args.max_weight,
        max_sector_exposure=DEFAULT_MAX_SECTOR_EXPOSURE,
        max_turnover=args.max_turnover,
        risk_aversion=1.0,
        transaction_cost_bps=args.cost_bps,
        turnover_penalty=args.turnover_penalty,
        max_participation=args.max_participation,
    )


def compute_benchmark_returns(
    returns: pd.DataFrame, adj: pd.DataFrame, oos_index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """Both benchmark return series on the strategy's exact OOS clock.

    Row ``t`` is the return realized from session ``t`` to the next session,
    matching ``oos_returns``. ``daily_ew`` is the cross-sectional mean of the
    forward returns (equal weight, rebalanced every session). ``buyhold`` is an
    equal-weight basket formed at the first OOS session and never rebalanced:
    its value is the cross-sectional mean of ``adj(t)/adj(t0)`` and row ``t``
    is ``V(t+1)/V(t) - 1``. The old report mixed a full-sample benchmark with
    the OOS window and quoted an excess as the benchmark's total annual.
    """
    oos_index = pd.DatetimeIndex(oos_index)
    daily_ew = returns.reindex(index=oos_index).mean(axis=1)
    if len(oos_index) == 0:
        buyhold = pd.Series(dtype=float, index=oos_index)
    else:
        window = adj.loc[oos_index[0]:]
        base = window.iloc[0]
        rel = window.divide(base.where(base > 0), axis=1)
        level = rel.mean(axis=1, skipna=True)
        buyhold = (level.shift(-1) / level - 1.0).reindex(oos_index)
    return {"daily_ew": daily_ew, "buyhold": buyhold}


def _series_metrics(
    series: pd.Series, risk_free: float = RISK_FREE_ANNUAL,
) -> dict[str, float | None]:
    """Sharpe / total annual / max drawdown for a daily return series.

    ``risk_free`` is passed straight to ``all_metrics``. Benchmark Sharpe
    uses the platform's 3% convention like the strategy; an *excess* series
    already subtracts the benchmark, so its Sharpe is an information ratio
    and must be computed with ``risk_free=0`` (otherwise the risk-free rate
    is subtracted a second time).
    """
    metrics = all_metrics(series.dropna(), risk_free=risk_free)
    return {
        "sharpe": _finite(metrics.get("sharpe_ratio")),
        "annual": _finite(metrics.get("annual_return")),
        "mdd": _finite(metrics.get("max_drawdown")),
    }


def deflated_sharpe_excess(
    oos_returns: pd.Series,
    num_trials: int,
    risk_free_annual: float = RISK_FREE_ANNUAL,
):
    """Deflated Sharpe on **excess** daily returns (``oos - rf/252``).

    Bailey & Lopez de Prado's SR is defined on excess returns; feeding gross
    returns understates the multiple-testing burden and overstated
    significance. The risk-free rate is converted with the same 252-day basis
    the validator uses.
    """
    excess = np.asarray(oos_returns.to_numpy(), dtype=float) - (
        risk_free_annual / TRADING_DAYS_PER_YEAR
    )
    return FactorValidator().deflated_sharpe_test(excess, num_trials=num_trials)


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full pipeline and return the JSON-ready summary."""
    started = time.time()
    frame = load_panel_frame(Path(args.cache))
    panels = build_panels(frame)
    # ``returns`` was shifted before slicing, so the last included session's
    # label already uses the next session from the full cache (documented in
    # --start/--end help). The buy&hold benchmark needs the same one-session
    # lookahead for its last forward return; hand it the unsliced adj path
    # (build_panels order: close, volume, turnover, returns, adj, pct).
    full_adj = panels[4]
    close, volume, _turnover, returns, adj, pct = slice_panels(
        panels, args.start, args.end,
    )
    n_sessions = len(close)
    step = int(args.step) if args.step else int(args.test)
    if n_sessions < args.train + args.test:
        raise ResearchError(
            "not enough data: "
            f"{n_sessions} sessions < train {args.train} + test {args.test}; "
            "refusing to fabricate a result"
        )

    sector_map = None
    if args.industry_map:
        sector_map = load_industry_map(Path(args.industry_map), close.columns)

    factor_names = parse_factor_list(
        args.factors if getattr(args, "factors", None) is not None
        else ",".join(DEFAULT_FACTORS)
    )
    factors, factors_skipped = build_factors(
        adj, volume, _turnover, factor_names,
        sector_map=sector_map,
        # Explicit --factors turns unknown names into errors; the default set
        # degrades gracefully (short panels can legitimately lose factors).
        strict=getattr(args, "factors", None) is not None,
    )
    if not factors:
        raise ResearchError(
            "no usable factors: every requested factor was skipped "
            f"({factors_skipped}); refusing to fabricate a result"
        )

    # R9/R19a gates: the engine reads daily_volume/raw_prices at run() time.
    # The limit panels are keyed by the execution day and hold that day's own
    # move (``pct``), because the engine trades at day d's close. Building them
    # from the forward panel (``returns[t] = pct_chg[t+1]``) let tomorrow's
    # limit move decide today's fill -- a one-day look-ahead. The same-day
    # convention matches data/pipeline.py's ``is_limit_up``.
    gates = build_execution_gates(close, volume, pct)
    cost_model = _build_cost_model(args.cost_mode)
    constraints = _build_constraints(args)

    state: dict[str, dict] = {"per_fold": {}}
    validator = WalkForwardValidator(
        train_period=int(args.train),
        test_period=int(args.test),
        step_size=step,
        mode="expanding",
        purge_gap=int(args.purge),
        embargo=int(args.embargo),
    )
    fallback_counter = _OptimizerFallbackCounter()
    engine_logger = logging.getLogger(ENGINE_LOGGER_NAME)
    optimizer_logger = logging.getLogger(OPTIMIZER_LOGGER_NAME)
    engine_logger.addHandler(fallback_counter)
    optimizer_logger.addHandler(fallback_counter)
    try:
        result = validator.run(
            signal=returns,
            prices=adj,
            returns=returns,
            benchmark_returns=returns.mean(axis=1),
            sector_map=pd.Series("A", index=returns.columns),
            engine_kwargs={
                "rebalance_frequency": "monthly",
                "initial_capital": float(args.capital),
            },
            cost_model=cost_model,
            constraints=constraints,
            execution_gates=gates,
            factors=factors,
            alpha_kwargs={"method": args.alpha},
            factor_selector=make_factor_selector(state, int(args.top_k)),
        )
    except ValueError as exc:
        raise ResearchError(f"walk-forward run failed: {exc}") from exc
    finally:
        engine_logger.removeHandler(fallback_counter)
        optimizer_logger.removeHandler(fallback_counter)

    oos = result["oos_returns"]
    aggregate = result.get("aggregate_metrics", {}) or {}
    metrics = {
        "sharpe_ratio": _finite(aggregate.get("sharpe_ratio")),
        "annual_return": _finite(aggregate.get("annual_return")),
        "annual_volatility": _finite(aggregate.get("annual_volatility")),
        "max_drawdown": _finite(aggregate.get("max_drawdown")),
        "total_return": _finite(aggregate.get("total_return")),
        "calmar_ratio": _finite(aggregate.get("calmar_ratio")),
        "win_rate": _finite(aggregate.get("win_rate")),
    }
    yearly_returns: dict[str, float] = {}
    for year, group in oos.groupby(oos.index.year):
        values = np.asarray(group.dropna(), dtype=float)
        if len(values):
            yearly_returns[str(int(year))] = float(np.prod(1.0 + values) - 1.0)

    n_folds = int(result.get("n_folds", 0))
    default_trials = max(1, n_folds * max(1, len(factors)))
    num_trials = (
        int(args.num_trials)
        if getattr(args, "num_trials", None)
        else default_trials
    )
    num_trials_source = "cli" if getattr(args, "num_trials", None) else "fold_x_factors"
    # DSR theory defines SR on excess returns; the old gross-input p-value
    # overstated significance.
    dsr = deflated_sharpe_excess(oos, num_trials)

    benchmark_series = compute_benchmark_returns(returns, full_adj, oos.index)
    # Align strategy and both benchmarks on the common valid dates so all
    # three rows cover the same observations (a mid-panel --end, a skipped
    # fold, or a delisted asset can otherwise shorten one leg).
    aligned = pd.concat(
        [
            oos.rename("strategy"),
            *(series.rename(name) for name, series in benchmark_series.items()),
        ],
        axis=1,
    ).dropna()
    benchmark = {
        name: _series_metrics(aligned[name]) for name in benchmark_series
    }
    # An excess series is already over the benchmark: its Sharpe is an
    # information ratio and must NOT subtract the risk-free rate again.
    excess_vs_daily_ew = _series_metrics(
        aligned["strategy"] - aligned["daily_ew"], risk_free=0.0,
    )

    skipped_folds = [
        detail for detail in result.get("fold_details", []) if "error" in detail
    ]
    selection_fallbacks = sum(
        1 for info in state["per_fold"].values() if info["fallback"]
    )
    summary: dict[str, Any] = {
        "panel": {
            "dates": int(n_sessions),
            "assets": int(close.shape[1]),
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
        },
        "config": {
            "cache": str(Path(args.cache)),
            "factors_requested": factor_names,
            "top_k": int(args.top_k),
            "alpha": args.alpha,
            "cost_mode": args.cost_mode,
            "train": int(args.train),
            "test": int(args.test),
            "step": step,
            "purge": int(args.purge),
            "embargo": int(args.embargo),
            "max_weight": float(args.max_weight),
            "max_turnover": float(args.max_turnover),
            "cost_bps": float(args.cost_bps),
            "turnover_penalty": float(args.turnover_penalty),
            "max_participation": float(args.max_participation),
            "capital": float(args.capital),
            "industry_neutralization": sector_map is not None,
            "industry_map": str(Path(args.industry_map)) if args.industry_map else None,
            "industry_map_note": INDUSTRY_MAP_NOTE,
            "num_trials": int(num_trials),
            "num_trials_source": num_trials_source,
        },
        "factors_used": sorted(factors),
        "factors_skipped": factors_skipped,
        "n_folds": n_folds,
        "skipped_folds": skipped_folds,
        "oos_days": int(len(oos)),
        "oos_metrics": metrics,
        "yearly_returns": yearly_returns,
        "per_fold_selection": {str(k): v for k, v in sorted(state["per_fold"].items())},
        "selection_fallback_count": int(selection_fallbacks),
        "optimizer_fallback_count": int(fallback_counter.count),
        "benchmark": benchmark,
        "excess_vs_daily_ew": excess_vs_daily_ew,
        "dsr": {
            "num_trials": int(num_trials),
            "num_trials_source": num_trials_source,
            "return_basis": "excess",
            "risk_free_annual": RISK_FREE_ANNUAL,
            "observed_sharpe": _finite(dsr.observed_sharpe),
            "expected_max_sharpe": _finite(dsr.expected_max_sharpe),
            "deflated_p_value": _finite(dsr.deflated_p_value),
            "significant": bool(dsr.significant),
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_console(summary: dict[str, Any]) -> str:
    panel = summary["panel"]
    config = summary["config"]
    lines = [
        "=" * 78,
        f"Panel: {panel['assets']} assets x {panel['dates']} sessions "
        f"({panel['start']} .. {panel['end']})",
        f"Factors used ({len(summary['factors_used'])}): "
        f"{', '.join(summary['factors_used'])}",
    ]
    if summary["factors_skipped"]:
        lines.append(f"Factors skipped: {summary['factors_skipped']}")
    lines += [
        f"Walk-forward: expanding, train={config['train']}, test={config['test']}, "
        f"step={config['step']}, purge={config['purge']}, embargo={config['embargo']}",
        f"Signal: {config['alpha']} (top-k={config['top_k']}); "
        f"cost mode: {config['cost_mode']}; max weight: {config['max_weight']:.0%}; "
        f"industry neutralization: {'on' if config['industry_neutralization'] else 'off'}",
        "-" * 78,
        f"{'fold':>4}  {'selected':<54} {'fallback':>8}",
    ]
    for fold, info in sorted(summary["per_fold_selection"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"{fold:>4}  {', '.join(info['selected']):<54} "
            f"{'yes' if info['fallback'] else 'no':>8}"
        )
    lines.append("-" * 78)
    metrics = summary["oos_metrics"]
    lines += [
        f"OOS sessions: {summary['oos_days']} over {summary['n_folds']} folds "
        f"(skipped folds: {len(summary['skipped_folds'])})",
        f"OOS Sharpe: {metrics['sharpe_ratio']:.3f}"
        if metrics["sharpe_ratio"] is not None else "OOS Sharpe: n/a",
        f"OOS annual return (total): {_pct(metrics['annual_return'])}",
        f"OOS max drawdown: {_pct(metrics['max_drawdown'])}",
        f"Benchmark daily EW (same window): Sharpe "
        f"{_num(summary['benchmark']['daily_ew']['sharpe'])}, annual (total) "
        f"{_pct(summary['benchmark']['daily_ew']['annual'])}; buy & hold: Sharpe "
        f"{_num(summary['benchmark']['buyhold']['sharpe'])}, annual (total) "
        f"{_pct(summary['benchmark']['buyhold']['annual'])}",
        f"Strategy excess vs daily EW (annualized, aligned daily): "
        f"{_pct(summary['excess_vs_daily_ew']['annual'])}",
        f"Yearly returns: "
        f"{', '.join(f'{year}:{value:+.1%}' for year, value in summary['yearly_returns'].items())}",
        f"Deflated Sharpe (excess returns): p={summary['dsr']['deflated_p_value']} over "
        f"{summary['dsr']['num_trials']} trials "
        f"(significant={summary['dsr']['significant']})",
        f"Fallbacks: selection={summary['selection_fallback_count']}, "
        f"optimizer={summary['optimizer_fallback_count']}",
        f"Runtime: {summary['runtime_seconds']}s",
        "=" * 78,
    ]
    return "\n".join(lines)


def format_markdown(summary: dict[str, Any]) -> str:
    panel = summary["panel"]
    config = summary["config"]
    metrics = summary["oos_metrics"]
    lines = [
        "# Real-data research run (cached panel)",
        "",
        f"- Panel: {panel['assets']} assets x {panel['dates']} sessions "
        f"({panel['start']} .. {panel['end']})",
        f"- Factors: {', '.join(summary['factors_used']) or 'none'}",
        f"- Walk-forward: expanding, train={config['train']}, test={config['test']}, "
        f"step={config['step']}, purge={config['purge']}, embargo={config['embargo']}",
        f"- Signal: {config['alpha']} (top-k={config['top_k']})",
        f"- Costs: {config['cost_mode']} model, stamp tax by date, commission 0.03%; "
        f"cost-aware MVO (max weight {config['max_weight']:.0%}, "
        f"{config['cost_bps']:.0f}bp, turnover penalty {config['turnover_penalty']})",
        f"- Industry neutralization: {'on' if config['industry_neutralization'] else 'off'}"
        + (
            f" (industry map: {config['industry_map_note']})"
            if config["industry_neutralization"] else ""
        ),
        "",
        "## Out-of-sample result (net of costs)",
        "",
        f"- Folds: {summary['n_folds']}; OOS sessions: {summary['oos_days']} "
        f"(skipped folds: {len(summary['skipped_folds'])})",
        f"- OOS Sharpe: {metrics['sharpe_ratio']:.3f}"
        if metrics["sharpe_ratio"] is not None else "- OOS Sharpe: n/a",
        f"- OOS annualized return (total): {_pct(metrics['annual_return'])}",
        f"- OOS max drawdown: {_pct(metrics['max_drawdown'])}",
        f"- Deflated Sharpe (excess returns, rf "
        f"{summary['dsr']['risk_free_annual']:.0%}): p="
        f"{summary['dsr']['deflated_p_value']:.4f} "
        f"over {summary['dsr']['num_trials']} trials "
        f"(significant={summary['dsr']['significant']})",
        f"- Fallbacks: selection={summary['selection_fallback_count']}, "
        f"optimizer={summary['optimizer_fallback_count']}",
        "",
        "## Benchmarks (same OOS window)",
        "",
        "| benchmark | Sharpe | annual (total) | max DD |",
        "|---|---:|---:|---:|",
        f"| daily equal-weight (rebalanced daily) | "
        f"{_num(summary['benchmark']['daily_ew']['sharpe'])} | "
        f"{_pct(summary['benchmark']['daily_ew']['annual'])} | "
        f"{_pct(summary['benchmark']['daily_ew']['mdd'])} |",
        f"| equal-weight buy & hold | "
        f"{_num(summary['benchmark']['buyhold']['sharpe'])} | "
        f"{_pct(summary['benchmark']['buyhold']['annual'])} | "
        f"{_pct(summary['benchmark']['buyhold']['mdd'])} |",
        "",
        f"- Strategy annualized excess vs daily EW (aligned daily): "
        f"{_pct(summary['excess_vs_daily_ew']['annual'])} "
        f"(Sharpe {_num(summary['excess_vs_daily_ew']['sharpe'])}, "
        f"max DD {_pct(summary['excess_vs_daily_ew']['mdd'])}). "
        "This is an excess figure; the benchmark rows above are total returns.",
        "",
        "## Yearly OOS returns",
        "",
        "| year | return |",
        "|---|---:|",
    ]
    for year, value in summary["yearly_returns"].items():
        lines.append(f"| {year} | {value:+.2%} |")
    lines += [
        "",
        "## Per-fold factor selection",
        "",
        "| fold | selected | survivors | fallback |",
        "|---:|---|---|---|",
    ]
    for fold, info in summary["per_fold_selection"].items():
        lines.append(
            f"| {fold} | {', '.join(info['selected']) or '-'} | "
            f"{', '.join(info['survivors']) or '-'} | "
            f"{'yes' if info['fallback'] else 'no'} |"
        )
    if summary["skipped_folds"]:
        lines += ["", "## Skipped folds", ""]
        for detail in summary["skipped_folds"]:
            lines.append(f"- Fold {detail['fold']}: {detail['error']}")
    if summary["factors_skipped"]:
        lines += ["", "## Factors skipped", ""]
        for name, reason in summary["factors_skipped"]:
            lines.append(f"- {name}: {reason}")
    lines += [
        "",
        f"_Runtime: {summary['runtime_seconds']}s. "
        "Generated by `tools/real_data_research.py`._",
    ]
    return "\n".join(lines)


def _write_outputs(summary: dict[str, Any], json_out: Path | None, md_out: Path | None) -> tuple[Path, Path]:
    if json_out is None:
        json_out = Path.cwd() / f"real_data_research_{_date.today().isoformat()}.json"
    if md_out is None:
        md_out = Path.cwd() / f"real_data_research_{_date.today().isoformat()}.md"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    # Wall-clock runtime is the one non-reproducible field; dropping it from
    # the JSON keeps identical inputs byte-identical (it remains in the MD).
    json_payload = {
        key: value for key, value in summary.items() if key != "runtime_seconds"
    }
    json_out.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_out.write_text(format_markdown(summary), encoding="utf-8")
    return json_out, md_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_factor_list(raw: str) -> list[str]:
    names: list[str] = []
    for piece in str(raw).split(","):
        name = piece.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ResearchError("--factors produced an empty list")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-data research pipeline on a cached daily panel: "
            "technical factors -> walk-forward factor selection -> cost-aware MVO."
        ),
    )
    parser.add_argument("--cache", default=str(DEFAULT_CACHE),
                        help="cached (date, asset) daily parquet (default: %(default)s)")
    parser.add_argument("--start", default=None,
                        help="first session (YYYY-MM-DD); returns are forward-labelled, "
                             "so the last included session's label uses the next session "
                             "from the full cache")
    parser.add_argument("--end", default=None,
                        help="last session (YYYY-MM-DD); same forward-label note as --start")
    parser.add_argument("--factors", default=None,
                        help="comma-separated technical factor names; explicit names are "
                             "strict (unknown factors error). Default: the 12-factor set "
                             f"({','.join(DEFAULT_FACTORS)})")
    parser.add_argument("--top-k", type=int, default=2,
                        help="factors kept per fold after BH-FDR (default: %(default)s)")
    parser.add_argument("--train", type=int, default=504, help="train window in sessions")
    parser.add_argument("--test", type=int, default=126, help="test window in sessions")
    parser.add_argument("--step", type=int, default=None,
                        help="fold step in sessions (default: --test)")
    parser.add_argument("--purge", type=int, default=5, help="purge gap in sessions")
    parser.add_argument("--embargo", type=int, default=5, help="embargo days after prior test windows")
    parser.add_argument("--alpha", choices=("equal_weight", "ic_weighted"),
                        default="ic_weighted", help="signal combination method")
    parser.add_argument("--cost-mode", choices=("impact", "fixed"), default="impact",
                        help="cost model: composite impact or fixed slippage")
    parser.add_argument("--max-weight", type=float, default=0.08, help="per-asset weight cap")
    parser.add_argument("--max-turnover", type=float, default=0.30, help="one-sided turnover cap")
    parser.add_argument("--cost-bps", type=float, default=20.0,
                        help="turnover cost term in the MVO objective (bp)")
    parser.add_argument("--turnover-penalty", type=float, default=0.5,
                        help="quadratic turnover penalty in the MVO objective")
    parser.add_argument("--max-participation", type=float, default=0.0,
                        help="ADV participation cap (0 disables)")
    parser.add_argument("--capital", type=float, default=2_000_000.0,
                        help="initial capital in CNY")
    parser.add_argument("--num-trials", type=int, default=None,
                        help="multiple-testing trial count for the deflated Sharpe; "
                             "default: n_folds x n_factors. Pass the real search count "
                             "when the run is one of many variants")
    parser.add_argument("--industry-map", default=None,
                        help="optional industry map parquet (6-digit code index, 'industry' column)")
    parser.add_argument("--json-out", default=None,
                        help="JSON report path (default: ./real_data_research_<date>.json)")
    parser.add_argument("--md-out", default=None,
                        help="Markdown report path (default: ./real_data_research_<date>.md)")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("train", "test", "purge", "embargo"):
        if getattr(args, name) < 1 and name in ("train", "test"):
            parser.error(f"--{name} must be >= 1")
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")
    if args.step is not None and args.step < 1:
        parser.error("--step must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if not 0.0 < args.max_weight <= 1.0:
        parser.error("--max-weight must be in (0, 1]")
    if not 0.0 < args.max_turnover <= 1.0:
        parser.error("--max-turnover must be in (0, 1]")
    if args.cost_bps < 0 or args.turnover_penalty < 0:
        parser.error("--cost-bps and --turnover-penalty must be >= 0")
    if not 0.0 <= args.max_participation <= 1.0:
        parser.error("--max-participation must be 0 or in (0, 1]")
    if args.capital <= 0:
        parser.error("--capital must be > 0")
    if args.num_trials is not None and args.num_trials < 1:
        parser.error("--num-trials must be >= 1")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        summary = run_research(args)
        json_out = Path(args.json_out) if args.json_out else None
        md_out = Path(args.md_out) if args.md_out else None
        json_path, md_path = _write_outputs(summary, json_out, md_out)
    except ResearchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_console(summary))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
