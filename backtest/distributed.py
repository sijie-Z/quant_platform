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
    results = bt.run_sweep(
        param_grid={"optimizer": ["equal_weight", "mean_variance"], "n_stocks": [200, 300]},
        base_config=config,
    )
    best = bt.find_best(results, metric="sharpe_ratio")
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


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
        from quant_platform.utils.config import load_config
        from quant_platform.main import _load_data, _compute_factors, _generate_signal, _run_backtest
        from quant_platform.backtest.metrics import compute_metrics

        # Merge config overrides
        config = load_config()
        for key, value in config_overrides.items():
            keys = key.split(".")
            cfg_section = config
            for k in keys[:-1]:
                cfg_section = cfg_section.setdefault(k, {})
            cfg_section[keys[-1]] = value

        # Run pipeline
        data_result = _load_data(config)
        factors_result = _compute_factors(data_result, config)
        signal = _generate_signal(factors_result, config)
        backtest_result = _run_backtest(signal, data_result, config)

        # Extract key metrics
        daily_returns = backtest_result.get("daily_returns")
        if daily_returns is not None:
            import numpy as np
            import pandas as pd

            total_return = float((1 + daily_returns).prod() - 1)
            ann_return = float((1 + total_return) ** (252 / max(len(daily_returns), 1)) - 1)
            ann_vol = float(daily_returns.std() * np.sqrt(252))
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            cum = (1 + daily_returns).cumprod()
            running_max = cum.cummax()
            drawdown = (cum - running_max) / running_max
            max_dd = float(drawdown.min())

            metrics = {
                "total_return": round(total_return, 4),
                "annual_return": round(ann_return, 4),
                "annual_volatility": round(ann_vol, 4),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
            }
        else:
            metrics = {"total_return": 0, "sharpe_ratio": 0}

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
            param_grid: dict of param_name -> list of values
            base_config_overrides: base config overrides applied to all runs
            metric: metric to optimize (default: sharpe_ratio)

        Returns:
            SweepResult with all results and best params
        """
        import itertools

        base_overrides = base_config_overrides or {}

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info("Starting parallel sweep: %d combinations, %d workers",
                     len(combinations), self.max_workers or 4)

        # Build task args
        tasks = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
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
