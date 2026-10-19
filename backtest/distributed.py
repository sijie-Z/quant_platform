"""Parallel backtest engine using concurrent.futures.

Runs multiple backtest configurations in parallel for:
- Parameter sweep acceleration (10-100x on multi-core)
- Multi-strategy comparison
- Walk-forward validation parallelism

Uses ProcessPoolExecutor for CPU-bound backtesting.
Handles serialization, error isolation, and result aggregation.

Usage:
    from quant_platform.backtest.distributed import ParallelBacktester

    bt = ParallelBacktester(max_workers=4)
    sweep = bt.run_sweep(
        # Flat aliases ("optimizer", "n_stocks") or dotted config paths
        # ("portfolio.covariance.method") are both accepted.
        param_grid={"optimizer": ["equal_weight", "mean_variance"], "n_stocks": [200, 300]},
    )
    best_params = sweep.best_params
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_platform.backtest.metrics import (
    all_metrics,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# Flat parameter names accepted by `run_sweep`, mapped to their real dotted
# config paths. `param_grid={"optimizer": [...], "n_stocks": [...]}` is the
# documented usage; applying those keys with `setattr` on the root Config used
# to create dead attributes, so every sweep cell silently ran the same
# configuration.
PARAM_ALIASES: dict[str, str] = {
    "optimizer": "portfolio.optimizer",
    "n_stocks": "universe.n_stocks",
    "alpha_method": "alpha.method",
    "rebalance_frequency": "backtest.rebalance_frequency",
    "benchmark": "backtest.benchmark",
    "initial_capital": "backtest.initial_capital",
    "covariance_method": "portfolio.covariance.method",
    "max_weight": "portfolio.constraints.max_weight",
    "max_turnover": "portfolio.constraints.max_turnover",
    "commission": "costs.commission",
    "stamp_tax": "costs.stamp_tax",
    "slippage": "costs.slippage",
    "slippage_model": "costs.slippage_model",
}


def _resolve_param_path(config, key: str) -> str:
    """Resolve a sweep parameter name to a dotted path that exists on `config`.

    Raises:
        ValueError: if `key` is neither a known flat alias nor a dotted path
            whose every segment exists on the config dataclasses.
    """
    path = PARAM_ALIASES.get(key, key)
    if "." not in path:
        raise ValueError(
            f"Unknown sweep parameter {key!r}: use a flat alias "
            f"({', '.join(sorted(PARAM_ALIASES))}) or a dotted config path "
            "such as 'portfolio.optimizer'."
        )
    target = config
    for part in path.split("."):
        if not hasattr(target, part):
            raise ValueError(
                f"Unknown sweep parameter {key!r}: {path!r} does not exist "
                f"on the configuration"
            )
        target = getattr(target, part)
    return path


def _apply_config_overrides(config, overrides: dict[str, Any]):
    """Apply sweep overrides to a Config, resolving flat aliases first."""
    for key, value in overrides.items():
        path = _resolve_param_path(config, key)
        target = config
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return config


def _compute_metrics(daily_returns: pd.Series) -> dict[str, float]:
    """Summarise a worker's daily returns with the shared metric functions.

    The worker used to divide annual return by annual volatility itself (no
    risk-free rate, no zero-volatility guard) and to measure drawdown from the
    first *return* rather than from inception, so sweep results were not
    comparable with the main backtest. Everything now comes from
    `quant_platform.backtest.metrics`; a series too short for `all_metrics`
    (< 10 observations) still gets the five headline figures.
    """
    series = pd.Series(daily_returns, copy=True).dropna()
    if series.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    if len(series) < 2:
        # Not enough observations for a volatility; avoid NaN propagating into
        # sweep results and JSON responses.
        return {
            "total_return": round(float((1 + series).prod() - 1), 4),
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
    if not isinstance(series.index, pd.DatetimeIndex):
        # `max_drawdown` labels its peak/trough with index values as dates;
        # real backtest output always has a DatetimeIndex, but a plain Series
        # would otherwise crash inside `all_metrics`.
        series.index = pd.bdate_range("2000-01-03", periods=len(series))

    summary = all_metrics(series)
    if not summary:
        mdd, _, _ = max_drawdown(series)
        summary = {
            "total_return": float((1 + series).prod() - 1),
            "annual_return": annualized_return(series),
            "annual_volatility": annualized_volatility(series),
            "sharpe_ratio": sharpe_ratio(series),
            "max_drawdown": mdd,
        }

    return {
        "total_return": round(float(summary["total_return"]), 4),
        "annual_return": round(float(summary["annual_return"]), 4),
        "annual_volatility": round(float(summary["annual_volatility"]), 4),
        "sharpe_ratio": round(float(summary["sharpe_ratio"]), 4),
        "max_drawdown": round(float(summary["max_drawdown"]), 4),
    }


@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    params: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0
    run_id: str = ""

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class SweepResult:
    """Aggregated result of a parameter sweep."""
    results: list[BacktestResult] = field(default_factory=list)
    total_duration: float = 0.0
    n_success: int = 0
    n_failed: int = 0
    best_params: dict[str, Any] | None = None
    best_metric: float | None = None

    def summary(self) -> list[dict]:
        """Return summary table sorted by performance."""
        rows = []
        for r in self.results:
            row = {"params": r.params, "duration": round(r.duration_seconds, 2)}
            if r.success:
                row.update(r.metrics)
            else:
                row["error"] = r.error
            rows.append(row)
        return sorted(rows, key=lambda x: x.get("sharpe_ratio", -999), reverse=True)


def _run_single_backtest(args: tuple) -> BacktestResult:
    """Worker function for parallel backtest execution.

    Runs in a separate process — must be picklable.
    """
    params, config_overrides, run_id = args

    start = time.perf_counter()

    try:
        # Import inside worker to avoid pickling issues
        from quant_platform.main import _compute_factors, _generate_signal, _load_data, _run_backtest
        from quant_platform.utils.config import load_config

        # Merge config overrides. `load_config()` returns dataclasses, not
        # dicts. Flat aliases ("optimizer", "n_stocks") are mapped to their
        # real config paths; every other key must be a dotted path that exists
        # on the config (a typo used to become a dead attribute).
        config = load_config()
        _apply_config_overrides(config, config_overrides)

        # Run pipeline with the real helper signatures.
        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config,
        )
        signal = _generate_signal(config, processed_factors, returns)
        backtest_result = _run_backtest(
            config, signal, prices, returns, benchmark, sector_map,
            fin_unstacked if fin_unstacked is not None else financials,
        )

        # Extract key metrics with the shared metric definitions, so sweep
        # results use the same Sharpe/reporting conventions as the main
        # backtest (risk-free rate, zero-vol guard, drawdown from inception).
        daily_returns = backtest_result.get("daily_returns")
        if daily_returns is not None:
            metrics = _compute_metrics(daily_returns)
        else:
            metrics = _compute_metrics(pd.Series(dtype=float))

        duration = time.perf_counter() - start
        return BacktestResult(
            params=params,
            metrics=metrics,
            duration_seconds=duration,
            run_id=run_id,
        )

    except Exception as e:
        duration = time.perf_counter() - start
        return BacktestResult(
            params=params,
            error=f"{type(e).__name__}: {e}",
            duration_seconds=duration,
            run_id=run_id,
        )


class ParallelBacktester:
    """Parallel backtest engine for parameter sweeps and comparisons.

    Args:
        max_workers: Maximum parallel processes (default: CPU count)
        timeout: Per-run timeout in seconds (default: 300)
    """

    def __init__(self, max_workers: int | None = None, timeout: int = 300):
        self.max_workers = max_workers
        self.timeout = timeout

    def run_sweep(
        self,
        param_grid: dict[str, list],
        base_config_overrides: dict[str, Any] | None = None,
        metric: str = "sharpe_ratio",
    ) -> SweepResult:
        """Run parameter sweep in parallel.

        Args:
            param_grid: dict of param_name -> list of values. Names may be
                flat aliases from `PARAM_ALIASES` (e.g. ``"optimizer"`` →
                ``portfolio.optimizer``, ``"n_stocks"`` →
                ``universe.n_stocks``) or dotted config paths such as
                ``"portfolio.covariance.method"``. Unknown names raise
                ``ValueError`` before any process is started.
            base_config_overrides: base config overrides applied to all runs
                (same naming rules as `param_grid`).
            metric: metric to optimize (default: sharpe_ratio)

        Returns:
            SweepResult with all results and best params
        """
        import itertools

        base_overrides = base_config_overrides or {}

        # Fail fast on unknown parameter names. Inside the worker pool the
        # ValueError would only surface as one failed cell per combination.
        from quant_platform.utils.config import load_config

        validation_config = load_config()
        for key in {*param_grid, *base_overrides}:
            _resolve_param_path(validation_config, key)

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info("Starting parallel sweep: %d combinations, %d workers",
                     len(combinations), self.max_workers or 4)

        # Build task args
        tasks = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo, strict=False))
            overrides = {**base_overrides}
            for k, v in params.items():
                overrides[k] = v
            tasks.append((params, overrides, f"sweep_{i}"))

        # Execute in parallel
        sweep_start = time.perf_counter()
        results = []

        if len(tasks) == 1:
            # Single task — skip process pool overhead
            results.append(_run_single_backtest(tasks[0]))
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_run_single_backtest, task): task for task in tasks}
                for future in as_completed(futures, timeout=self.timeout * len(tasks)):
                    try:
                        result = future.result(timeout=self.timeout)
                        results.append(result)
                        if result.success:
                            logger.info("  Run %s: Sharpe=%.2f (%.1fs)",
                                        result.run_id, result.metrics.get("sharpe_ratio", 0), result.duration_seconds)
                        else:
                            logger.warning("  Run %s: FAILED - %s", result.run_id, result.error)
                    except Exception as e:
                        task = futures[future]
                        results.append(BacktestResult(
                            params=task[0],
                            error=f"Future error: {e}",
                        ))

        total_duration = time.perf_counter() - sweep_start

        # Find best
        successful = [r for r in results if r.success]
        best_params = None
        best_metric_val = None
        if successful:
            best = max(successful, key=lambda r: r.metrics.get(metric, -999))
            best_params = best.params
            best_metric_val = best.metrics.get(metric)

        sweep = SweepResult(
            results=results,
            total_duration=total_duration,
            n_success=len(successful),
            n_failed=len(results) - len(successful),
            best_params=best_params,
            best_metric=best_metric_val,
        )

        logger.info("Sweep complete: %d/%d succeeded in %.1fs. Best %s=%.4f",
                     sweep.n_success, len(results), total_duration,
                     metric, best_metric_val or 0)

        return sweep

    def run_comparison(
        self,
        configs: list[dict[str, Any]],
        labels: list[str] | None = None,
    ) -> list[BacktestResult]:
        """Run multiple configurations in parallel for comparison.

        Args:
            configs: list of config override dicts
            labels: optional labels for each config

        Returns:
            list of BacktestResult
        """
        tasks = []
        for i, cfg in enumerate(configs):
            label = labels[i] if labels and i < len(labels) else f"config_{i}"
            tasks.append(({"label": label}, cfg, label))

        logger.info("Running %d configurations in parallel", len(tasks))

        results = []
        if len(tasks) == 1:
            results.append(_run_single_backtest(tasks[0]))
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(_run_single_backtest, task) for task in tasks]
                for future in as_completed(futures, timeout=self.timeout * len(tasks)):
                    try:
                        results.append(future.result(timeout=self.timeout))
                    except Exception as e:
                        results.append(BacktestResult(params={}, error=str(e)))

        return sorted(results, key=lambda r: r.metrics.get("sharpe_ratio", -999), reverse=True)

    @staticmethod
    def find_best(results: list[BacktestResult], metric: str = "sharpe_ratio") -> BacktestResult | None:
        """Find the best result by a given metric."""
        successful = [r for r in results if r.success]
        if not successful:
            return None
        return max(successful, key=lambda r: r.metrics.get(metric, -999))

    @staticmethod
    def format_comparison(results: list[BacktestResult]) -> str:
        """Format comparison results as a text table."""
        lines = []
        lines.append("=" * 80)
        lines.append("  PARALLEL BACKTEST COMPARISON")
        lines.append("=" * 80)
        lines.append(f"  {'Config':<25} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Vol':>8} {'Time':>8}")
        lines.append("-" * 80)

        for r in results:
            label = r.params.get("label", r.run_id)
            if r.success:
                m = r.metrics
                lines.append(f"  {label:<25} {m.get('sharpe_ratio', 0):>8.2f} "
                             f"{m.get('total_return', 0):>7.1%} {m.get('max_drawdown', 0):>7.1%} "
                             f"{m.get('annual_volatility', 0):>7.1%} {r.duration_seconds:>7.1f}s")
            else:
                lines.append(f"  {label:<25} {'FAILED':>8} {r.error[:35]}")

        lines.append("=" * 80)
        return "\n".join(lines)
