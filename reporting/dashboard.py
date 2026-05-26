"""Summary dashboard: text-based and chart-based overview of results.

Presents a complete picture of backtest results in a single report, including:
- Performance metrics (Sharpe, Sortino, Calmar, drawdown analysis)
- Risk analysis (VaR, CVaR, stress tests)
- Exposure breakdown (sector concentration, effective N)
- Factor rankings (when IC data provided)
"""

from __future__ import annotations

import os

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


def generate_dashboard(
    results: dict,
    metadata: pd.DataFrame | None = None,
    output_dir: str = "./results",
    save_plots: bool = True,
    plot_format: str = "png",
    ic_results: dict[str, dict] | None = None,
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
    risk = var_summary(strategy_returns)
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
        ranked = sorted(ic_results.items(), key=lambda x: x[1].get("mean_ic", 0), reverse=True)
        for name, ic_sum in ranked:
            mean_ic = ic_sum.get("mean_ic", 0)
            icir = ic_sum.get("icir", 0)
            ir = ic_sum.get("information_ratio", 0)
            direction = "+" if mean_ic > 0 else " " if mean_ic == 0 else "-"
            lines.append(
                f"  {direction} {name:<22s} IC={mean_ic:+.4f}  ICIR={icir:+.2f}  IR={ir:+.2f}"
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
# Helpers
# ------------------------------------------------------------------

def _analyze_drawdown_detail(returns: pd.Series) -> dict | None:
    """Analyze maximum drawdown event with dates and duration."""
    cumulative = (1 + returns.dropna()).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / running_max - 1)

    if drawdown.min() >= 0:
        return None

    trough_idx = drawdown.idxmin()
    trough_date = returns.index[trough_idx] if isinstance(trough_idx, int) else trough_idx
    if not isinstance(trough_date, pd.Timestamp):
        return None

    # Find peak: highest point before trough
    pre_trough = cumulative.loc[:trough_date]
    if len(pre_trough) == 0:
        return None
    peak_date = pre_trough.idxmax()
    if not isinstance(peak_date, pd.Timestamp):
        return None

    # Find recovery: first date after trough where cumulative >= peak
    peak_value = cumulative.loc[peak_date]
    post_trough = cumulative.loc[trough_date:]
    recovery_mask = post_trough >= peak_value
    if recovery_mask.any():
        recovery_date = recovery_mask[recovery_mask].index[0]
        recovery_days = (recovery_date - peak_date).days
    else:
        recovery_date = None
        recovery_days = None

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
