#!/usr/bin/env python3
"""
Regime Discovery v4 — Regime-Conditioned Heatmap

Key insight: Regime dates are non-contiguous, so we can't filter returns before
computing the backtest (rolling windows break). Instead:

1. Run backtest on FULL data to get portfolio returns
2. Decompose daily portfolio returns by regime
3. Compute regime-conditioned Sharpe

Protocol frozen. Results only.
"""

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np
from collections import defaultdict

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("rd4")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"

# Focus on the proven sweet spot and its neighbors
SIGNAL_HORIZONS = [5, 10, 20, 40, 60, 80, 120, 200]
HOLDING_HORIZONS = [5, 10, 20, 40, 60, 80, 120, 200]


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.benchmark


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def classify_regimes(benchmark_returns):
    monthly = benchmark_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    regimes = {}
    for date, ret in monthly.items():
        if ret > 0.02:
            regimes[date] = "Bull"
        elif ret < -0.02:
            regimes[date] = "Bear"
        else:
            regimes[date] = "Sideways"
    reg_series = pd.Series(regimes, name="regime")
    reg_series.index = pd.DatetimeIndex(reg_series.index)
    daily = reg_series.reindex(
        pd.date_range(REAL_START, REAL_END, freq="D"), method="ffill"
    )
    daily.index.name = "date"
    return daily, "基准月收益"


def classify_volatility(returns):
    monthly_vol = returns.mean(axis=1).resample("M").std()
    thirds = monthly_vol.quantile([1/3, 2/3])
    regimes = {}
    for date, vol in monthly_vol.items():
        if vol <= thirds.iloc[0]:
            regimes[date] = "LowVol"
        elif vol >= thirds.iloc[1]:
            regimes[date] = "HighVol"
        else:
            regimes[date] = "NormalVol"
    reg_series = pd.Series(regimes, name="regime")
    reg_series.index = pd.DatetimeIndex(reg_series.index)
    daily = reg_series.reindex(
        pd.date_range(REAL_START, REAL_END, freq="D"), method="ffill"
    )
    daily.index.name = "date"
    return daily, "月波动率"


def backtest_with_regime_tracking(returns, signal_h, hold_h, daily_regime):
    """Run reversal backtest on FULL data, decompose returns by regime.

    Rebalances every `hold_h` days, selects bottom 20% by past `signal_h` return.
    Returns dict with per-regime Sharpe and overall Sharpe.
    """
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * 0.2))
    dates = returns.index
    step = hold_h

    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    indices = list(range(step, len(dates) - max(signal_h, hold_h), step))

    all_port_rets = []
    all_regimes = []

    for i in indices:
        sig = past_ret.iloc[i]
        valid = sig.dropna().sort_values()
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)
        end = i + step
        if end >= len(dates):
            break

        hold_returns = returns.iloc[i + 1:end + 1]
        if len(hold_returns) == 0:
            continue

        # Portfolio return for this holding period
        port_ret = (hold_returns[selected.index].mean(axis=1) + 1).prod() - 1
        all_port_rets.append(port_ret)

        # Determine regime for this period (modal regime during the holding window)
        period_dates = hold_returns.index
        period_regimes = daily_regime.reindex(period_dates).dropna()
        if len(period_regimes) > 0:
            modal_regime = period_regimes.mode()
            regime = modal_regime.iloc[0] if len(modal_regime) > 0 else "Unknown"
        else:
            regime = "Unknown"
        all_regimes.append(regime)

    if len(all_port_rets) < 3:
        return {"overall": {"sharpe": np.nan, "ann_ret": 0, "n": 0}}

    port_series = pd.Series(all_port_rets)

    # Overall
    af = np.sqrt(252 / hold_h)
    overall_sharpe = port_series.mean() / port_series.std() * af if port_series.std() > 1e-10 else 0
    overall_ann = (1 + port_series.mean()) ** (252 / hold_h) - 1
    cum = (1 + port_series).cumprod()
    overall_mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    result = {
        "overall": {"sharpe": overall_sharpe, "ann_ret": overall_ann, "mdd": overall_mdd if not np.isnan(overall_mdd) else 0, "n": len(port_series)}
    }

    # Per-regime
    port_df = pd.DataFrame({"return": all_port_rets, "regime": all_regimes})
    for regime_name in port_df["regime"].unique():
        subset = port_df[port_df["regime"] == regime_name]["return"]
        if len(subset) < 2:
            continue
        sr = subset.mean() / subset.std() * af if subset.std() > 1e-10 else 0
        ar = (1 + subset.mean()) ** (252 / hold_h) - 1
        result[regime_name] = {"sharpe": sr, "ann_ret": ar, "n": len(subset)}

    return result


def run():
    logger.info("=" * 60)
    logger.info("Regime Discovery v4")
    logger.info("=" * 60)

    returns, benchmark = load()

    # Regime classifications
    classifiers = [
        ("基准收益(Bull/Bear/Sideways)", classify_regimes(benchmark)),
        ("波动率(Low/Normal/High)", classify_volatility(returns)),
    ]

    for clf_name, (daily_regime, method_desc) in classifiers:
        counts = daily_regime.value_counts()
        total = len(daily_regime)
        logger.info("Classifier: %s", clf_name)
        logger.info("  Distribution: %s", {k: f"{v/total*100:.1f}%" for k, v in counts.items()})

        # Run key cells with regime tracking
        KEY_CELLS = [(5, 5), (20, 20), (40, 80), (80, 80), (5, 120), (40, 120), (60, 120), (20, 60)]
        all_results = []

        for sh, hh in KEY_CELLS:
            r = backtest_with_regime_tracking(returns, sh, hh, daily_regime)
            row = {"signal_h": sh, "hold_h": hh}
            for regime_key, metrics in r.items():
                row[f"Sharpe_{regime_key}"] = metrics["sharpe"]
                row[f"n_{regime_key}"] = metrics["n"]
            all_results.append(row)
            regime_sharpes = {k: v["sharpe"] for k, v in r.items() if k != "overall"}
            logger.info("  S=%3d H=%3d Overall=%.4f Regimes: %s",
                         sh, hh, r["overall"]["sharpe"], regime_sharpes)

        # Print table
        print()
        print("=" * 100)
        print(f"  REGIME-CONDITIONED HEATMAP: {clf_name}")
        print("=" * 100)
        print()

        # Collect all regime names
        all_regime_names = set()
        for r in all_results:
            for k in r:
                if k.startswith("Sharpe_") and k != "Sharpe_overall":
                    all_regime_names.add(k.replace("Sharpe_", ""))

        # Print header
        print(f"  {'S':>4} {'H':>4} {'Overall':>8}", end="")
        for rn in sorted(all_regime_names):
            print(f" {rn:>10}", end="")
        print()

        for row in all_results:
            print(f"  {row['signal_h']:>4d} {row['hold_h']:>4d} {row['Sharpe_overall']:>+8.4f}", end="")
            for rn in sorted(all_regime_names):
                v = row.get(f"Sharpe_{rn}", np.nan)
                if not np.isnan(v):
                    print(f" {v:>+10.4f}", end="")
                else:
                    print(f" {'':>10}", end="")
            print()

        # Key insight: (40,80) across regimes
        print()
        k40 = [r for r in all_results if r["signal_h"] == 40 and r["hold_h"] == 80]
        if k40:
            print(f"  Key Cell S=40 H=80 across regimes:")
            for k in k40:
                for rn in sorted(all_regime_names):
                    v = k.get(f"Sharpe_{rn}", np.nan)
                    if not np.isnan(v):
                        print(f"    {rn:>12}: Sharpe={v:+.4f} (n={k.get(f'n_{rn}', 0)})")

        # State persistence
        print()
        print(f"  State Persistence:")
        print()
        transitions = defaultdict(int)
        state_durations = defaultdict(list)
        current_state = None
        current_start = None
        for date, state in daily_regime.items():
            if state != current_state:
                if current_state is not None and current_start is not None:
                    duration = (date - current_start).days
                    state_durations[current_state].append(duration)
                    transitions[(current_state, state)] += 1
                current_state = state
                current_start = date

        print(f"  {'State':<12} {'Avg Dur':>8} {'Max Dur':>8} {'N':>5}")
        for state in sorted([s for s in state_durations.keys() if isinstance(s, str)]):
            durations = state_durations[state]
            print(f"  {state:<12} {np.mean(durations):>8.1f}d {max(durations):>8.0f}d {len(durations):>5d}")

        print()

    print()
    print("=" * 100)
    print("  结论: 等待解读")
    print("=" * 100)


if __name__ == "__main__":
    run()
