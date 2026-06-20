#!/usr/bin/env python3
"""
RQ5b: Signal Horizon vs Holding Horizon Alignment

Theory ↔ Execution Gap exploration.

Question: Why does IC(H) confirm reversal but reversal backtest loses money?
Hypothesis: Protocol's fixed monthly rebalance (20d) mismatches the optimal
reversal window (80d).

Experiment: Fix signal horizon at 80d (strongest reversal), scan holding periods.
Then 2D matrix: signal_horizon × holding_horizon.

Protocol frozen. Results only.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("rq5b")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
INITIAL_CAPITAL = 10_000_000

SIGNAL_HORIZONS = [5, 20, 40, 80, 120, 200]
HOLDING_HORIZONS = [5, 10, 20, 40, 60, 80, 120]


def load_real_data():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    returns = pipeline.returns
    benchmark = pipeline.benchmark
    logger.info("Loaded: %d days, %d assets", len(returns), len(returns.columns))
    return returns, benchmark


def compound_return(ret_series):
    """Compute compound return from a return series."""
    return np.prod(1 + ret_series) - 1


def backtest_horizon(returns, signal_horizon, holding_horizon):
    """Run a reversal backtest with matching signal and holding horizons.

    Strategy:
    - Every `holding_horizon` trading days:
      1. Compute past `signal_horizon` return for each stock
      2. Select bottom 20% (most oversold = reversal signal)
      3. Equal weight them
      4. Hold for `holding_horizon` days
    - Compute portfolio return for each period

    Returns: dict with metrics
    """
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * 0.2))  # Bottom 20%

    dates = returns.index
    step = holding_horizon

    # Compute rolling past returns at each relevant date
    past_ret = returns.rolling(signal_horizon, min_periods=signal_horizon).apply(
        lambda x: compound_return(x), raw=True
    )

    # Rebalance dates: every `step` days
    rebalance_indices = list(range(step, len(dates) - signal_horizon, step))
    rebalance_dates = [dates[i] for i in rebalance_indices]

    portfolio_returns = []
    weights_history = []

    for i, idx in enumerate(rebalance_indices):
        rdate = dates[idx]
        # Signal value at rebalance date
        sig = past_ret.loc[rdate]

        # Select bottom 20% (most negative past return = most oversold = reversal)
        valid = sig.dropna().sort_values()
        if len(valid) < n_select:
            continue

        selected = valid.head(n_select)
        w = 1.0 / len(selected)

        # Compute holding period return
        next_idx = idx + step
        if next_idx >= len(dates):
            break

        hold_returns = returns.iloc[idx + 1:next_idx + 1]
        if len(hold_returns) == 0:
            continue

        # Portfolio return = equal-weighted return of selected stocks
        port_ret = (hold_returns[selected.index].mean(axis=1) + 1).prod() - 1
        portfolio_returns.append(port_ret)

    if len(portfolio_returns) < 5:
        return {"sharpe": 0, "annual_return": 0, "max_drawdown": 0,
                "n_periods": len(portfolio_returns), "total_return": 0}

    port_series = pd.Series(portfolio_returns)
    ann_factor = np.sqrt(252 / holding_horizon)
    sharpe = port_series.mean() / port_series.std() * ann_factor if port_series.std() > 1e-10 else 0
    ann_ret = (1 + port_series.mean()) ** (252 / holding_horizon) - 1

    # Max drawdown
    cum = (1 + port_series).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    mdd = dd.min() if len(dd) > 0 else 0

    return {
        "signal_h": signal_horizon,
        "hold_h": holding_horizon,
        "sharpe": sharpe,
        "annual_return": ann_ret,
        "max_drawdown": mdd,
        "n_periods": len(portfolio_returns),
        "total_return": port_series.sum(),
    }


def run():
    logger.info("=" * 70)
    logger.info("RQ5b: Signal Horizon vs Holding Horizon Alignment")
    logger.info("Theory-Execution Gap exploration")
    logger.info("=" * 70)

    logger.info("[1/3] Loading data...")
    returns, benchmark = load_real_data()

    # ==================================================================
    # Phase 1: Fix signal=80d, scan holding periods
    # ==================================================================
    logger.info("[2/3] Phase 1: Signal=80d, scan holding periods...")
    phase1_results = []
    for hold_h in HOLDING_HORIZONS:
        r = backtest_horizon(returns, signal_horizon=80, holding_horizon=hold_h)
        phase1_results.append(r)
        logger.info("  Hold=%3dd  Sharpe=%+7.4f  AnnRet=%+7.2f%%  MDD=%6.2f%%  n=%d",
                     hold_h, r["sharpe"], r["annual_return"] * 100,
                     r["max_drawdown"] * 100, r["n_periods"])

    # ==================================================================
    # Phase 2: 2D Matrix — signal_h × hold_h
    # ==================================================================
    logger.info("[3/3] Phase 2: 2D Sharpe Matrix...")
    matrix_results = []
    for sig_h in SIGNAL_HORIZONS:
        for hold_h in HOLDING_HORIZONS:
            r = backtest_horizon(returns, signal_horizon=sig_h, holding_horizon=hold_h)
            matrix_results.append(r)

    # ==================================================================
    # Report
    # ==================================================================
    print()
    print("=" * 90)
    print("  RQ5b: SIGNAL-HOLDING HORIZON ALIGNMENT")
    print("=" * 90)

    print()
    print("─── Phase 1: Signal=80d, Scan Holding Period ───")
    print()
    print(f"  {'Hold(d)':>8} {'Sharpe':>10} {'AnnRet':>10} {'MDD':>10} {'n':>6}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*6}")
    for r in phase1_results:
        print(f"  {r['hold_h']:>8d} {r['sharpe']:>10.4f} {r['annual_return']:>10.2%} {r['max_drawdown']:>10.2%} {r['n_periods']:>6d}")

    print()
    print("─── Phase 2: 2D Sharpe Matrix ───")
    print()
    print(f"  Sharpe(Signal_H × Hold_H):")
    print()
    sig_labels = [f"S{sig_h}" for sig_h in SIGNAL_HORIZONS]
    hold_labels = [f"H{hold_h}" for hold_h in HOLDING_HORIZONS]

    # Create matrix
    matrix = {}
    for r in matrix_results:
        key = (r["signal_h"], r["hold_h"])
        matrix[key] = r["sharpe"]

    # Print header
    print(f"  {'Sig\\Hold':>10}", end="")
    for hh in HOLDING_HORIZONS:
        print(f"  H={hh:<5d}", end="")
    print()

    for sh in SIGNAL_HORIZONS:
        print(f"  S={sh:<5d}  ", end="")
        for hh in HOLDING_HORIZONS:
            v = matrix.get((sh, hh), 0)
            if -0.1 < v < 0.1:
                fmt = f"{v:>+7.4f}~"
            elif v >= 0.1:
                fmt = f"{v:>+7.4f}+"
            else:
                fmt = f"{v:>+7.4f} "
            print(fmt, end=" ")
        print()

    print()
    print("  Legend: ~ = near zero, + = positive, (space) = negative")
    print()

    # Find best cell
    best = max(matrix_results, key=lambda r: r["sharpe"])
    worst = min(matrix_results, key=lambda r: r["sharpe"])
    print(f"  Best:  Signal={best['signal_h']}d  Hold={best['hold_h']}d  Sharpe={best['sharpe']:.4f}")
    print(f"  Worst: Signal={worst['signal_h']}d  Hold={worst['hold_h']}d  Sharpe={worst['sharpe']:.4f}")

    # ==================================================================
    # Save
    # ==================================================================
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(phase1_results).to_csv(out / "rq5b_phase1_signal80_scan.csv", index=False)
    pd.DataFrame(matrix_results).to_csv(out / "rq5b_phase2_matrix.csv", index=False)

    # Print raw data
    print()
    print("  === RAW DATA: PHASE1 ===")
    print(pd.DataFrame(phase1_results).to_string())
    print()
    print("  === RAW DATA: PHASE2 ===")
    df_m = pd.DataFrame(matrix_results)
    pivot = df_m.pivot_table(index="signal_h", columns="hold_h", values="sharpe")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    logger.info("Results saved.")


if __name__ == "__main__":
    run()
