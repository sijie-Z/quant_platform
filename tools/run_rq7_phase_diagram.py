#!/usr/bin/env python3
"""
RQ7: Sampling–Holding Phase Diagram

2D sweep: Sharpe(frequency, holding_period)
Find the execution manifold where A-share reversal alpha survives discretization.

Protocol frozen.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("rq7")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"

FREQUENCIES = [1, 5, 10, 20, 40]   # sampling intervals in days
HORIZONS = [5, 10, 20, 40, 80, 120, 160]  # holding periods


@dataclass
class Tranche:
    entry_idx: int
    exit_idx: int
    assets: list[str]
    weights: np.ndarray


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close(), pipeline.benchmark


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def run_cell(returns, freq_days, hold_days, select_pct=0.20):
    """Run one (f, H) cell with position overlap.

    Every `freq_days` days, open a new tranche.
    Each tranche holds for `hold_days`.
    Aggregate all active tranches → daily return stream.
    """
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * select_pct))
    dates = returns.index
    n_dates = len(dates)
    signal_h = max(5, hold_days // 2)

    # Precompute signal once
    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    signal_values = past_ret.values  # (n_dates, n_assets)
    asset_names = returns.columns.tolist()

    # Rebalance indices
    rebal_indices = list(range(freq_days, n_dates - max(signal_h, hold_days), freq_days))

    active_tranches: list[Tranche] = []
    daily_returns = np.zeros(n_dates)

    for i in range(n_dates):
        # Close matured tranches
        active_tranches = [t for t in active_tranches if t.exit_idx > i]

        # Open new tranche if this is a rebalance date
        if i in rebal_indices:
            sig_row = signal_values[i]
            # Sort assets by signal (ascending = most negative = strongest reversal)
            order = np.argsort(sig_row)
            valid_mask = ~np.isnan(sig_row)
            valid_indices = order[valid_mask[order]]
            selected = valid_indices[:n_select]

            if len(selected) > 0:
                assets = [asset_names[s] for s in selected]
                w = np.ones(len(selected)) / len(selected)
                tranche = Tranche(
                    entry_idx=i,
                    exit_idx=min(i + hold_days, n_dates - 1),
                    assets=assets,
                    weights=w,
                )
                active_tranches.append(tranche)

        # Compute daily portfolio return
        if active_tranches:
            port_ret = 0.0
            total_weight = 0.0
            for t in active_tranches:
                for j, asset in enumerate(t.assets):
                    col_idx = asset_names.index(asset)
                    ret_val = returns.values[i, col_idx]
                    if not np.isnan(ret_val):
                        port_ret += t.weights[j] * ret_val
                        total_weight += t.weights[j]

            if total_weight > 0:
                daily_returns[i] = port_ret / total_weight
            else:
                daily_returns[i] = 0.0

    # Compute metrics from return stream
    ret_series = pd.Series(daily_returns, index=dates)
    # Find first day with non-zero return
    first_date = dates[np.where(np.abs(daily_returns) > 1e-10)[0][0]] if np.any(np.abs(daily_returns) > 1e-10) else dates[-1]
    ret_series = ret_series[first_date:]

    if len(ret_series) < 20:
        return {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "turnover": 0, "n_positions": len(active_tranches)}

    metrics = all_metrics(ret_series, None)
    # Turnover = positions per year
    n_positions_total = len(rebal_indices)
    turnover = n_positions_total / (n_dates / 252)

    return {
        "sharpe": metrics.get("sharpe_ratio", 0),
        "ann_ret": metrics.get("annual_return", 0),
        "mdd": metrics.get("max_drawdown", 0),
        "turnover": turnover,
        "n_positions": n_positions_total,
        "n_days_active": int((ret_series != 0).sum()),
    }


def print_heatmap(results_dict, f_values, h_values):
    """Print Sharpe(f, H) as a heatmap."""
    print()
    print("=" * 100)
    print("  RQ7: SAMPLING–HOLDING PHASE DIAGRAM")
    print("=" * 100)
    print()
    print(f"  Sharpe(f, H) — Red=negative, Green=positive, --=NaN")
    print()

    # Header
    print(f"  {'f↓ H→':>8}", end="")
    for h in h_values:
        print(f"  {h:>8d}d", end="")
    print()
    print(f"  {'─'*8}", end="")
    for _ in h_values:
        print(f"  {'─'*8}", end="")
    print()

    for f_val in f_values:
        print(f"  {f_val:>4d}d  ", end="")
        for h_val in h_values:
            key = (f_val, h_val)
            v = results_dict.get(key, {}).get("sharpe", np.nan)
            if np.isnan(v):
                print(f"  {'  N/A':>8}", end="")
            elif v > 0.1:
                print(f"  {v:>+8.4f}+", end="")
            elif v < -0.1:
                print(f"  {v:>+8.4f} ", end="")
            else:
                print(f"  {v:>+8.4f}~", end="")
        print()

    print()
    print("  Legend: + = strong positive | ~ = near zero | space = negative")
    print()


def main():
    logger.info("=" * 60)
    logger.info("RQ7: Sampling–Holding Phase Diagram")
    logger.info("=" * 60)

    returns, prices, benchmark = load()
    logger.info("%d days, %d assets", len(returns), len(returns.columns))

    all_results = {}
    total_cells = len(FREQUENCIES) * len(HORIZONS)
    cell_count = 0

    for f_val, h_val in product(FREQUENCIES, HORIZONS):
        cell_count += 1
        logger.info("[%d/%d] f=%dd H=%dd (S=%dd)...", cell_count, total_cells,
                     f_val, h_val, max(5, h_val // 2))

        result = run_cell(returns, freq_days=f_val, hold_days=h_val)
        all_results[(f_val, h_val)] = result

        s = result["sharpe"]
        logger.info("  Sharpe=%.4f AnnRet=%.2f%% MDD=%.2f%% n=%d",
                     s, result["ann_ret"] * 100, result["mdd"] * 100, result["n_positions"])

        # Save intermediate results (in case of crash)
        if cell_count % 10 == 0:
            _save_intermediate(all_results)

    # Print heatmap
    print_heatmap(all_results, FREQUENCIES, HORIZONS)

    # Find optimal ridge: for each f, best H
    print("─── Optimal Ridge (per frequency) ───")
    print()
    print(f"  {'Frequency':>12} {'Best H':>8} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8}")
    for f_val in FREQUENCIES:
        best = None
        best_sharpe = -np.inf
        for h_val in HORIZONS:
            v = all_results.get((f_val, h_val), {}).get("sharpe", -np.inf)
            if not np.isnan(v) and v > best_sharpe:
                best_sharpe = v
                best = (f_val, h_val, v)

        if best:
            r = all_results[(best[0], best[1])]
            print(f"  {f_val:>4d}d:           {best[1]:>4d}d    {best[2]:>+8.4f}  "
                  f"{r['ann_ret']*100:>7.2f}% {r['mdd']*100:>7.2f}%")

    # Find f_critical: frequency where Sharpe first becomes positive
    print()
    print("─── Critical Sampling Frequency ───")
    print()
    for h_val in HORIZONS:
        for f_val in FREQUENCIES:
            v = all_results.get((f_val, h_val), {}).get("sharpe", np.nan)
            if not np.isnan(v) and v > 0:
                print(f"  H={h_val:>4d}d:  f_critical = {f_val:>2d}d  (Sharpe={v:.4f})")
                break
        else:
            print(f"  H={h_val:>4d}d:  no positive Sharpe found")

    # ── Ideal benchmark (f=1d) ──
    print()
    print("─── Ideal Benchmark (f=1d, daily rebalance) ───")
    print()
    for h_val in HORIZONS:
        r = all_results.get((1, h_val), {})
        s = r.get("sharpe", np.nan)
        if not np.isnan(s):
            print(f"  H={h_val:>4d}d:  Sharpe={s:>+.4f}  AnnRet={r.get('ann_ret',0)*100:.2f}%")

    # Save
    rows = []
    for (f_val, h_val), r in all_results.items():
        rows.append({"f": f_val, "H": h_val, **r})
    df = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rq7_phase_diagram.csv", index=False)
    logger.info("Saved to results/rq7_phase_diagram.csv")


def _save_intermediate(results):
    rows = [{"f": k[0], "H": k[1], **v} for k, v in results.items()]
    pd.DataFrame(rows).to_csv("results/rq7_phase_diagram_intermediate.csv", index=False)


if __name__ == "__main__":
    main()
