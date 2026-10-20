"""Summary dashboard: text-based and chart-based overview of results.

Presents a complete picture of backtest results in a single report, including:
- Performance metrics (Sharpe, Sortino, Calmar, drawdown analysis)
- Risk analysis (VaR, CVaR, stress tests)
- Exposure breakdown (sector concentration, effective N)
- Factor rankings (when IC data provided)
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_platform.backtest.metrics import all_metrics
from quant_platform.reporting.performance import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
)
from quant_platform.risk.exposure import exposure_report
from quant_platform.risk.stress import run_all_stress_tests
from quant_platform.risk.var import var_summary
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: Per-run artifact layout (issue S15):
#:   <results_dir>/runs/<timestamp>_<config_hash>_<run_id>/...
#:   <results_dir>/latest_manifest.json   (pointer to the published run)
MANIFEST_FILENAME = "manifest.json"
LATEST_MANIFEST_FILENAME = "latest_manifest.json"
RUNS_DIRNAME = "runs"
_PUBLISH_LOCK = ".publish.lock"


def generate_dashboard(
    results: dict,
    metadata: pd.DataFrame | None = None,
    output_dir: str = "./results",
    save_plots: bool = True,
    plot_format: str = "png",
    ic_results: dict[str, dict] | None = None,
    var_confidence: float = 0.95,
    var_horizon: int = 1,
) -> str:
    """Generate comprehensive backtest dashboard.

    Args:
        results: Dict from BacktestEngine.run().
        metadata: Stock metadata (sector, market cap group).
        output_dir: Directory for saving output.
        save_plots: Whether to save chart files.
        plot_format: Image format for charts.
        ic_results: Optional dict of {factor_name: ic_summary} for factor ranking.

    Returns:
        Text summary string.
    """
    os.makedirs(output_dir, exist_ok=True)

    strategy_returns = results["daily_returns"]
    benchmark_returns = results.get("benchmark_returns")
    weights_history = results.get("weights_history", {})

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------
    summary = results.get("summary", all_metrics(strategy_returns, benchmark_returns))

    # --- Drawdown analysis ---
    dd_detail = _analyze_drawdown_detail(strategy_returns)

    # --- Rolling stats ---
    rolling_stats = _compute_rolling_stats(strategy_returns)

    # Risk metrics
    risk = var_summary(
        strategy_returns, confidence=var_confidence, horizon=var_horizon,
    )
    stress = run_all_stress_tests(strategy_returns)

    # Exposure (latest weights)
    if weights_history:
        latest_date = max(weights_history.keys())
        latest_weights = weights_history[latest_date]
        sector_map = metadata["sector"] if metadata is not None else None
        cap_groups = metadata["market_cap_group"] if metadata is not None else None
        exposure = exposure_report(latest_weights, sector_map, cap_groups)
    else:
        exposure = {}

    # ------------------------------------------------------------------
    # Build text report
    # ------------------------------------------------------------------
    lines = [
        "=" * 70,
        "          QUANTITATIVE STRATEGY BACKTEST — DASHBOARD",
        "=" * 70,
        "",
        "--- PERFORMANCE ---",
        f"  Total Return:         {summary.get('total_return', 0)*100:8.2f}%",
        f"  Annual Return:        {summary.get('annual_return', 0)*100:8.2f}%",
        f"  Annual Volatility:    {summary.get('annual_volatility', 0)*100:8.2f}%",
        f"  Sharpe Ratio:         {summary.get('sharpe_ratio', 0):8.2f}",
        f"  Sortino Ratio:        {summary.get('sortino_ratio', 0):8.2f}",
        f"  Calmar Ratio:         {summary.get('calmar_ratio', 0):8.2f}",
        f"  Max Drawdown:         {summary.get('max_drawdown', 0)*100:8.2f}%",
        f"  Win Rate:             {summary.get('win_rate', 0)*100:8.2f}%",
        f"  P/L Ratio:            {summary.get('profit_loss_ratio', 0):8.2f}",
        "",
    ]

    if "information_ratio" in summary:
        lines.extend([
            f"  Information Ratio:    {summary['information_ratio']:8.2f}",
            f"  Tracking Error:       {summary['tracking_error']*100:8.2f}%",
            f"  Excess Return:        {summary.get('excess_return', 0)*100:8.2f}%",
            "",
        ])

    # --- Drawdown details ---
    if dd_detail:
        lines.extend([
            "--- DRAWDOWN ANALYSIS ---",
            f"  Peak Date:            {dd_detail['peak_date']}",
            f"  Trough Date:          {dd_detail['trough_date']}",
            f"  Recovery Date:        {dd_detail['recovery_date'] or 'Not yet recovered'}",
            f"  Drawdown Duration:    {dd_detail['duration_days']} days",
            f"  Recovery Time:        {dd_detail.get('recovery_days', 'N/A')} days",
            "",
        ])

    # --- Rolling statistics ---
    if rolling_stats:
        lines.extend([
            "--- ROLLING 252-DAY STATISTICS ---",
            f"  Best Rolling Sharpe:  {rolling_stats['best_sharpe']:.2f}",
            f"  Worst Rolling Sharpe: {rolling_stats['worst_sharpe']:.2f}",
            f"  Sharpe Stability:     {rolling_stats['sharpe_stability']:.2f} (std of rolling Sharpe)",
            "",
        ])

    lines.extend([
        "--- RISK ---",
        f"  Historical VaR (95%): {risk['historical_var']*100:8.2f}%",
        f"  Parametric VaR (95%): {risk['parametric_var']*100:8.2f}%",
        f"  Historical CVaR (95%):{risk['historical_cvar']*100:8.2f}%",
        "",
        "--- STRESS TESTS ---",
    ])

    for _, row in stress.iterrows():
        lines.append(f"  {row.name}: {row['estimated_cumulative_return']*100:+.2f}% cumulative, "
                      f"max DD {row['estimated_max_drawdown']*100:+.2f}%")

    # --- Factor rankings ---
    if ic_results:
        lines.extend(["", "--- FACTOR IC RANKINGS ---"])
        ranked = sorted(
            ic_results.items(),
            key=lambda x: (np.isnan(x[1].get("mean_ic", np.nan)),
                           -abs(x[1].get("mean_ic", 0) or 0)),
        )
        for name, ic_sum in ranked:
            mean_ic = ic_sum.get("mean_ic", 0)
            icir = ic_sum.get("icir", 0)
            n_obs = ic_sum.get("n_obs", 0)
            # `ic_summary` never returned an information ratio; the old line
            # printed a fabricated IR=+0.00 for every factor. Report the
            # number of evaluable cross-sections instead.
            if mean_ic is None or not np.isfinite(mean_ic):
                lines.append(
                    f"    {name:<22s} IC=n/a  (only {n_obs} evaluable observations)"
                )
                continue
            direction = "+" if mean_ic > 0 else " " if mean_ic == 0 else "-"
            lines.append(
                f"  {direction} {name:<22s} IC={mean_ic:+.4f}  "
                f"ICIR={icir:+.2f}  N={n_obs}"
            )

    if exposure:
        lines.extend([
            "",
            "--- EXPOSURE (Latest) ---",
            f"  Holdings:             {exposure.get('n_assets', 0)}",
            f"  Effective N:          {exposure.get('effective_n', 0):.1f}",
            f"  Top 5 Concentration:  {exposure.get('top5_concentration', 0)*100:.1f}%",
            f"  Top 10 Concentration: {exposure.get('top10_concentration', 0)*100:.1f}%",
        ])

        if "sector_exposure" in exposure:
            lines.append("  Sector Breakdown:")
            for sector, weight in sorted(exposure["sector_exposure"].items(),
                                          key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    {sector:<16s} {weight*100:5.1f}%")

    lines.extend([
        "",
        f"  Config: optimizer={summary.get('optimizer', 'N/A')}, "
        f"rebalances={summary.get('n_rebalances', 0)}",
        "=" * 70,
    ])

    report = "\n".join(lines)
    logger.info("Dashboard generated:\n%s", report)

    # ------------------------------------------------------------------
    # Save charts
    # ------------------------------------------------------------------
    if save_plots:
        plot_equity_curve(
            strategy_returns, benchmark_returns,
            save_path=os.path.join(output_dir, f"equity_curve.{plot_format}"),
        )
        plot_drawdown(
            strategy_returns,
            save_path=os.path.join(output_dir, f"drawdown.{plot_format}"),
        )
        plot_rolling_sharpe(
            strategy_returns,
            save_path=os.path.join(output_dir, f"rolling_sharpe.{plot_format}"),
        )

        try:
            plot_monthly_returns_heatmap(
                strategy_returns,
                save_path=os.path.join(output_dir, f"monthly_returns.{plot_format}"),
            )
        except Exception:
            pass  # Not enough data for heatmap

        plt.close("all")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    strategy_returns.to_csv(os.path.join(output_dir, "daily_returns.csv"))
    if benchmark_returns is not None:
        benchmark_returns.to_csv(os.path.join(output_dir, "benchmark_returns.csv"))
    if weights_history:
        weights_df = pd.DataFrame(weights_history).T
        weights_df.to_csv(os.path.join(output_dir, "weights_history.csv"))

    return report


# ------------------------------------------------------------------
# Per-run artifact directories (issue S15)
# ------------------------------------------------------------------

def create_run_dir(
    base_dir: str | Path,
    run_id: str,
    config_hash: str = "",
    timestamp: str | None = None,
) -> Path:
    """Create (atomically) and return a unique per-run output directory.

    The name carries the timestamp, config hash and run id, so two concurrent
    runs can never write over each other's CSVs even in the same results dir.
    """
    base = Path(base_dir)
    stamp = timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_hash = (config_hash or "nohash")[:12]
    safe_run = (run_id or stamp)[:16]
    run_dir = base / RUNS_DIRNAME / f"{stamp}_{safe_hash}_{safe_run}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(run_dir: str | Path, manifest: dict) -> Path:
    """Atomically write ``manifest.json`` inside a run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_FILENAME
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_manifest(run_dir: str | Path) -> dict | None:
    """Read a run's manifest, or None when absent/corrupt."""
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def resolve_run_dir(results_dir: str | Path) -> Path:
    """Return the run directory holding the results to analyze.

    Order: the path itself when it contains a manifest (a run dir passed
    directly), the published pointer, the newest complete run under
    ``runs/``, and finally the legacy flat ``results/`` layout.
    """
    base = Path(results_dir)
    if (base / MANIFEST_FILENAME).exists():
        return base

    pointer_path = base / LATEST_MANIFEST_FILENAME
    if pointer_path.exists():
        try:
            with open(pointer_path, encoding="utf-8") as f:
                pointer = json.load(f)
            candidate = base / str(pointer.get("run_dir", ""))
            if candidate.is_dir() and (candidate / MANIFEST_FILENAME).exists():
                return candidate
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    runs = base / RUNS_DIRNAME
    if runs.is_dir():
        candidates = sorted(
            (
                d for d in runs.iterdir()
                if d.is_dir() and (d / MANIFEST_FILENAME).exists()
            ),
            key=lambda d: d.name,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    return base


def publish_run_artifacts(
    results_dir: str | Path,
    run_dir: str | Path,
    manifest: dict,
    timeout: float = 2.0,
) -> bool:
    """Copy a run's artifacts to the top-level results dir, all-or-nothing.

    Top-level convenience copies are only written while holding an exclusive
    publish lock, so two concurrent runs cannot interleave files from
    different runs at the top level. If the lock cannot be acquired the
    copies are skipped (the run directory and its manifest remain the
    authoritative artifacts).

    Returns True when the copies/pointer were published.
    """
    base = Path(results_dir)
    run_dir = Path(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / _PUBLISH_LOCK
    deadline = time.monotonic() + timeout

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 120:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                logger.warning(
                    "Another run holds the publish lock; skipping top-level "
                    "copies for %s", run_dir,
                )
                return False
            time.sleep(0.01)

    try:
        artifacts = manifest.get("artifacts") or []
        for name in artifacts:
            source = run_dir / name
            if not source.is_file():
                continue
            tmp = base / f"{name}.tmp.{os.getpid()}"
            try:
                shutil.copyfile(source, tmp)
                os.replace(tmp, base / name)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        pointer = {
            "run_id": manifest.get("run_id", ""),
            "config_hash": manifest.get("config_hash", ""),
            "run_dir": (
                str(run_dir.relative_to(base)).replace("\\", "/")
                if run_dir.is_relative_to(base) else str(run_dir)
            ),
            "manifest": (
                str((run_dir / MANIFEST_FILENAME).relative_to(base)).replace("\\", "/")
                if run_dir.is_relative_to(base) else str(run_dir / MANIFEST_FILENAME)
            ),
            "published_at": datetime.now().isoformat(),
        }
        pointer_path = base / LATEST_MANIFEST_FILENAME
        tmp = pointer_path.with_name(f"{pointer_path.name}.tmp.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pointer, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pointer_path)
        return True
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _analyze_drawdown_detail(returns: pd.Series) -> dict | None:
    """Analyze maximum drawdown event with dates and duration.

    The initial wealth of 1.0 is part of the wealth path. Leaving it out made
    a first-day loss invisible and reported the first trading day as the peak.
    """
    sr = returns.dropna()
    if len(sr) == 0:
        return None

    wealth = np.concatenate([[1.0], np.cumprod((1 + sr).to_numpy())])
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1

    if drawdown.min() >= 0:
        return None

    trough_pos = int(drawdown.argmin())
    peak_pos = int(np.argmax(wealth[:trough_pos]))
    # Wealth position k >= 1 is reached at the close of the k-th return. The
    # series is labelled by return-period *start* (the row dated t is the
    # close(t) -> close(t+1) return), so that close carries the next label --
    # `sr.index[k]`, clamped at the end. The old `index[k - 1]` labelled the
    # peak/trough one trading day early (B10).
    def _close_date(pos: int):
        if pos <= 0:
            return sr.index[0]
        return sr.index[min(pos, len(sr) - 1)]

    peak_date = _close_date(peak_pos)
    trough_date = _close_date(trough_pos)
    peak_value = wealth[peak_pos]

    recovery_date = None
    recovery_days = None
    for pos in range(trough_pos + 1, len(wealth)):
        if wealth[pos] >= peak_value:
            recovery_date = sr.index[min(pos, len(sr) - 1)]
            recovery_days = (recovery_date - peak_date).days
            break

    duration_days = (trough_date - peak_date).days

    return {
        "peak_date": peak_date.strftime("%Y-%m-%d"),
        "trough_date": trough_date.strftime("%Y-%m-%d"),
        "recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date else None,
        "duration_days": duration_days,
        "recovery_days": recovery_days,
    }


def _compute_rolling_stats(returns: pd.Series, window: int = 252) -> dict | None:
    """Compute rolling performance statistics."""
    if len(returns.dropna()) < window:
        return None

    from quant_platform.backtest.metrics import TRADING_DAYS_PER_YEAR

    rolling_ret = returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_sharpe = rolling_ret / rolling_vol.replace(0, np.nan)
    rolling_sharpe = rolling_sharpe.dropna()

    if len(rolling_sharpe) == 0:
        return None

    return {
        "best_sharpe": rolling_sharpe.max(),
        "worst_sharpe": rolling_sharpe.min(),
        "sharpe_stability": rolling_sharpe.std(),
    }
