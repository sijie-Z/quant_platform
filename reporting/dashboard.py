"""Summary dashboard: text-based and chart-based overview of results.

Presents a complete picture of backtest results in a single report, including:
- Performance metrics (Sharpe, Sortino, Calmar, drawdown analysis)
- Risk analysis (VaR, CVaR, stress tests)
- Exposure breakdown (sector concentration, effective N)
- Factor rankings (when IC data provided)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import UTC, datetime
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

#: ``backtest.metrics.all_metrics`` returns an empty mapping below this many
#: observations. The text report must print n/a, never the 0.0 defaults.
MIN_OBSERVATIONS = 10


def _finite(value) -> float | None:
    """A finite float, or None for missing/non-numeric/non-finite values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_pct(value, signed: bool = True, decimals: int = 2) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    if signed:
        return f"{number * 100:+.{decimals}f}%"
    return f"{number * 100:.{decimals}f}%"


def _fmt_num(value, decimals: int = 2) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:,.{decimals}f}"


def _fmt_int(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{int(number):,}"


def _var_basis(risk: dict) -> str:
    """VaR/CVaR label suffix from the configured confidence/horizon (R97)."""
    confidence = _finite(risk.get("confidence"))
    horizon = _finite(risk.get("horizon_days"))
    parts = []
    if confidence is not None and 0 < confidence < 1:
        parts.append(f"{confidence * 100:g}%")
    if horizon is not None and horizon > 0:
        parts.append(f"{horizon:g}-day")
    return f" ({', '.join(parts)})" if parts else ""


def generate_dashboard(
    results: dict,
    metadata: pd.DataFrame | None = None,
    output_dir: str = "./results",
    save_plots: bool = True,
    plot_format: str = "png",
    ic_results: dict[str, dict] | None = None,
    var_confidence: float = 0.95,
    var_horizon: int = 1,
    provenance: dict | None = None,
) -> str:
    """Generate comprehensive backtest dashboard.

    Args:
        results: Dict from BacktestEngine.run().
        metadata: Stock metadata (sector, market cap group).
        output_dir: Directory for saving output.
        save_plots: Whether to save chart files.
        plot_format: Image format for charts.
        ic_results: Optional dict of {factor_name: ic_summary} for factor ranking.
        var_confidence: VaR/CVaR confidence level (labels use this value).
        var_horizon: VaR/CVaR holding period in days (labels use this value).
        provenance: Optional data provenance (provider/synthetic/date range),
            rendered in the report header so synthetic and real runs differ.

    Returns:
        Text summary string.
    """
    os.makedirs(output_dir, exist_ok=True)

    # `cmd_analyze` re-opens a completed run directory; its manifest carries
    # the provenance that the text report was missing (R94). Read it when the
    # caller did not pass provenance explicitly.
    if provenance is None:
        manifest = read_manifest(output_dir)
        if manifest:
            date_range = manifest.get("date_range") or {}
            provenance = {
                "requested_provider": manifest.get("requested_provider"),
                "effective_provider": manifest.get("provider"),
                "synthetic": bool(manifest.get("synthetic")),
                "start_date": date_range.get("start"),
                "end_date": date_range.get("end"),
                "data_date": date_range.get("data_date"),
            }

    strategy_returns = results["daily_returns"]
    benchmark_returns = results.get("benchmark_returns")
    weights_history = results.get("weights_history", {})

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------
    summary = results.get("summary") or all_metrics(strategy_returns, benchmark_returns)

    # Sample size drives the n/a policy: all_metrics returns {} for fewer
    # than 10 observations, and rendering the missing keys as 0.0 made an
    # unmeasured sample look like a flat strategy (R93).
    n_obs_value = _finite(summary.get("total_days"))
    if n_obs_value is None:
        n_obs_value = float(len(strategy_returns.dropna()))
    n_obs = int(n_obs_value)
    insufficient = n_obs < MIN_OBSERVATIONS
    metric_src = {} if insufficient else summary

    # --- Drawdown analysis ---
    dd_detail = None if insufficient else _analyze_drawdown_detail(strategy_returns)

    # --- Rolling stats ---
    rolling_stats = None if insufficient else _compute_rolling_stats(strategy_returns)

    # Risk metrics. VaR on a handful of points is meaningless, and the Monte
    # Carlo fit can raise; both paths must not crash the report.
    risk: dict = {}
    if not insufficient:
        try:
            risk = var_summary(
                strategy_returns, confidence=var_confidence, horizon=var_horizon,
            )
        except Exception as exc:
            logger.warning("Risk summary unavailable: %s", exc)

    stress = None
    if not insufficient:
        try:
            stress = run_all_stress_tests(strategy_returns)
        except Exception as exc:
            logger.warning("Stress tests unavailable: %s", exc)

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
    # No-data report (R100): an empty series must not crash the generator.
    # ------------------------------------------------------------------
    if n_obs == 0:
        report = "\n".join([
            "=" * 70,
            "          QUANTITATIVE STRATEGY BACKTEST — DASHBOARD",
            "=" * 70,
            "",
            "--- NO DATA ---",
            "  The strategy return series is empty; no metrics, charts or risk",
            "  statistics were produced.",
            "=" * 70,
        ])
        logger.warning("Dashboard requested for an empty return series")
        return report

    # ------------------------------------------------------------------
    # Build text report
    # ------------------------------------------------------------------
    # Return windows (R90): the headline total return is the full strategy
    # history ("since inception"), matching the equity chart and the annual
    # return/sharpe/drawdown that are also full-history. ``total_return`` in
    # the summary is computed on the benchmark-common window when a benchmark
    # was supplied, so it is displayed separately together with its coverage.
    full_total = _finite(metric_src.get("strategy_total_return_full_window"))
    common_total = _finite(metric_src.get("total_return"))
    coverage = _finite(metric_src.get("benchmark_coverage"))
    if full_total is not None:
        total_label, total_value = "Total Return (since inception)", full_total
    elif common_total is not None and coverage is not None and coverage < 0.9999:
        total_label, total_value = "Total Return (benchmark window)", common_total
    else:
        total_label, total_value = "Total Return", common_total

    lines = [
        "=" * 70,
        "          QUANTITATIVE STRATEGY BACKTEST — DASHBOARD",
        "=" * 70,
        "",
    ]

    if provenance:
        provider = provenance.get("effective_provider") or "N/A"
        if provenance.get("synthetic"):
            provider += " (SYNTHETIC)"
        requested = provenance.get("requested_provider") or "N/A"
        lines.append(f"  Provider:            {provider} (requested: {requested})")
        if provenance.get("start_date") or provenance.get("end_date"):
            data_date = provenance.get("data_date") or "n/a"
            lines.append(
                f"  Data Range:          {provenance.get('start_date', '?')} → "
                f"{provenance.get('end_date', '?')} (data date: {data_date})"
            )
        lines.append("")

    lines.append("--- PERFORMANCE ---")
    if insufficient:
        lines.append(
            f"  INSUFFICIENT DATA: n={n_obs} observations "
            f"(minimum {MIN_OBSERVATIONS} required); metrics are n/a."
        )
    if coverage is not None and coverage < 0.9999:
        if full_total is not None:
            lines.append(
                "  Windows: total return since inception; benchmark-based "
                "metrics (excess/IR/TE) use the common window."
            )
        else:
            # No full-window field in the summary: the headline total return
            # IS the common-window value (R90), so do not claim otherwise.
            lines.append(
                "  Windows: total return and benchmark-based metrics "
                "(excess/IR/TE) use the common strategy/benchmark window."
            )
    lines.extend([
        f"  {total_label + ':':<29} {_fmt_pct(total_value)}",
        f"  {'Annual Return:':<29} {_fmt_pct(metric_src.get('annual_return'))}",
        f"  {'Annual Volatility:':<29} {_fmt_pct(metric_src.get('annual_volatility'), signed=False)}",
        f"  {'Sharpe Ratio:':<29} {_fmt_num(metric_src.get('sharpe_ratio'))}",
        f"  {'Sortino Ratio:':<29} {_fmt_num(metric_src.get('sortino_ratio'))}",
        f"  {'Calmar Ratio:':<29} {_fmt_num(metric_src.get('calmar_ratio'))}",
        f"  {'Max Drawdown:':<29} {_fmt_pct(metric_src.get('max_drawdown'))}",
        f"  {'Win Rate:':<29} "
        f"{_fmt_pct(metric_src.get('win_rate'), signed=False, decimals=1)}",
        f"  {'P/L Ratio:':<29} {_fmt_num(metric_src.get('profit_loss_ratio'))}",
        "",
    ])

    if coverage is not None and not insufficient:
        if full_total is not None and common_total is not None and coverage < 0.9999:
            lines.append(f"  {'Benchmark-Window Return:':<29} {_fmt_pct(common_total)}")
        bench_days = _finite(metric_src.get("benchmark_days"))
        total_days = _finite(metric_src.get("total_days"))
        coverage_text = f"{coverage * 100:.1f}%"
        if bench_days is not None and total_days is not None and total_days > 0:
            coverage_text += f" ({int(bench_days)}/{int(total_days)} days)"
        lines.extend([
            f"  {'Benchmark Coverage:':<29} {coverage_text}",
            "",
        ])

    if "information_ratio" in metric_src and not insufficient:
        lines.extend([
            f"  {'Information Ratio (benchmark window):':<29} "
            f"{_fmt_num(metric_src.get('information_ratio'))}",
            f"  {'Tracking Error:':<29} "
            f"{_fmt_pct(metric_src.get('tracking_error'), signed=False)}",
            f"  {'Excess Return (benchmark window):':<29} "
            f"{_fmt_pct(metric_src.get('excess_return'))}",
            "",
        ])

    # --- Drawdown details ---
    if dd_detail:
        recovery_days = dd_detail.get("recovery_days_from_trough")
        recovery_text = (
            "N/A (not yet recovered)" if recovery_days is None
            else f"{recovery_days} days (trough→recovery)"
        )
        lines.extend([
            "--- DRAWDOWN ANALYSIS ---",
            f"  {'Peak Date:':<29} {dd_detail['peak_date']}",
            f"  {'Trough Date:':<29} {dd_detail['trough_date']}",
            f"  {'Recovery Date:':<29} "
            f"{dd_detail['recovery_date'] or 'Not yet recovered'}",
            f"  {'Duration (peak→trough):':<29} {dd_detail['duration_days']} days",
            f"  {'Recovery Time:':<29} {recovery_text}",
            "",
        ])

    # --- Rolling statistics ---
    if rolling_stats:
        lines.extend([
            "--- ROLLING 252-DAY SHARPE (rf = 0) ---",
            f"  {'Best Rolling Sharpe:':<29} {_fmt_num(rolling_stats['best_sharpe'])}",
            f"  {'Worst Rolling Sharpe:':<29} {_fmt_num(rolling_stats['worst_sharpe'])}",
            f"  {'Sharpe Stability:':<29} "
            f"{_fmt_num(rolling_stats['sharpe_stability'])} (std of rolling Sharpe)",
            "  (rolling window uses a zero risk-free rate; the headline Sharpe uses 3%)",
            "",
        ])

    var_basis = _var_basis({
        "confidence": risk.get("confidence", var_confidence),
        "horizon_days": risk.get("horizon_days", var_horizon),
    })
    lines.extend([
        "--- RISK ---",
        f"  {'Historical VaR' + var_basis + ':':<29} "
        f"{_fmt_pct(risk.get('historical_var'), signed=False)}",
        f"  {'Parametric VaR' + var_basis + ':':<29} "
        f"{_fmt_pct(risk.get('parametric_var'), signed=False)}",
        f"  {'Historical CVaR' + var_basis + ':':<29} "
        f"{_fmt_pct(risk.get('historical_cvar'), signed=False)}",
        "  (positive values = loss, scaled by sqrt(horizon))",
        "",
        "--- STRESS TESTS ---",
    ])

    if stress is None or len(stress) == 0:
        lines.append("  n/a (insufficient data)" if insufficient
                     else "  n/a (no stress results)")
    else:
        for _, row in stress.iterrows():
            lines.append(
                f"  {row.name}: {_fmt_pct(row['estimated_cumulative_return'])} cumulative, "
                f"max DD {_fmt_pct(row['estimated_max_drawdown'])}"
            )

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
            icir = _finite(ic_sum.get("icir"))
            n_obs_factor = ic_sum.get("n_obs", 0)
            # `ic_summary` never returned an information ratio; the old line
            # printed a fabricated IR=+0.00 for every factor. Report the
            # number of evaluable cross-sections instead.
            if _finite(mean_ic) is None:
                lines.append(
                    f"    {name:<22s} IC=n/a  (only {n_obs_factor} evaluable observations)"
                )
                continue
            direction = "+" if mean_ic > 0 else " " if mean_ic == 0 else "-"
            icir_text = f"{icir:+.2f}" if icir is not None else "n/a"
            lines.append(
                f"  {direction} {name:<22s} IC={mean_ic:+.4f}  "
                f"ICIR={icir_text}  N={n_obs_factor}"
            )

    if exposure:
        lines.extend([
            "",
            "--- EXPOSURE (Latest) ---",
            f"  {'Holdings:':<29} {_fmt_int(exposure.get('n_assets'))}",
            f"  {'Effective N:':<29} {_fmt_num(exposure.get('effective_n'), 1)}",
            f"  {'Top 5 Concentration:':<29} "
            f"{_fmt_pct(exposure.get('top5_concentration'), signed=False)}",
            f"  {'Top 10 Concentration:':<29} "
            f"{_fmt_pct(exposure.get('top10_concentration'), signed=False)}",
        ])

        if "sector_exposure" in exposure:
            lines.append("  Sector Breakdown:")
            for sector, weight in sorted(exposure["sector_exposure"].items(),
                                          key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    {sector:<16s} {_fmt_pct(weight, signed=False)}")

    # --- Configuration (R96: effective optimizer and fallbacks) ---
    optimizer_text = str(summary.get("optimizer") or "N/A")
    effective_optimizer = summary.get("effective_optimizer")
    if effective_optimizer and str(effective_optimizer) != optimizer_text:
        optimizer_text += f" (effective: {effective_optimizer})"
    n_fallbacks = _finite(summary.get("n_optimizer_fallbacks"))
    fallback_rate = _finite(summary.get("optimizer_fallback_rate"))
    if n_fallbacks is not None:
        optimizer_text += f", fallbacks={int(n_fallbacks)}"
        if fallback_rate is not None:
            optimizer_text += f" ({fallback_rate * 100:.1f}% of rebalances)"

    lines.extend([
        "",
        f"  Config: optimizer={optimizer_text}, "
        f"rebalances={_fmt_int(metric_src.get('n_rebalances'))}",
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
        except Exception as exc:
            # Presentation-only: the report must still be returned, but a
            # silently missing chart hid real failures (R105).
            logger.warning("Monthly returns heatmap unavailable: %s", exc)

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
    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
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
    """Read a run's manifest, or None when absent/corrupt.

    A corrupt manifest is logged: silently treating it as absent made the
    analysis fall back to a different run without any warning (R105).
    """
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        logger.warning("Run manifest %s is not a JSON object", path)
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable run manifest %s: %s", path, exc)
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
            if not isinstance(pointer, dict):
                raise ValueError("latest_manifest.json is not a JSON object")
            candidate = base / str(pointer.get("run_dir", ""))
            if candidate.is_dir() and (candidate / MANIFEST_FILENAME).exists():
                return candidate
            logger.warning(
                "latest_manifest.json points to a missing/corrupt run (%s); "
                "falling back to the newest complete run", candidate,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            # A corrupt pointer used to be swallowed and analysis silently
            # switched to a different (or legacy) run (R105).
            logger.warning("Ignoring unreadable latest_manifest.json: %s", exc)

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
            "published_at": datetime.now(UTC).isoformat(),
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
    recovery_days_from_trough = None
    for pos in range(trough_pos + 1, len(wealth)):
        if wealth[pos] >= peak_value:
            recovery_date = sr.index[min(pos, len(sr) - 1)]
            recovery_days = (recovery_date - peak_date).days
            # The interval that actually describes the recovery phase is
            # trough -> recovery; peak -> recovery was printed next to the
            # peak -> trough duration and read as if it were the same span.
            recovery_days_from_trough = (recovery_date - trough_date).days
            break

    duration_days = (trough_date - peak_date).days

    return {
        "peak_date": peak_date.strftime("%Y-%m-%d"),
        "trough_date": trough_date.strftime("%Y-%m-%d"),
        "recovery_date": recovery_date.strftime("%Y-%m-%d") if recovery_date else None,
        "duration_days": duration_days,
        "recovery_days": recovery_days,
        "recovery_days_from_trough": recovery_days_from_trough,
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
