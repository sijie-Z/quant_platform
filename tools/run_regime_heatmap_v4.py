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

from collections import defaultdict

import numpy as np
import pandas as pd
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import get_logger, setup_logging
from quant_platform.utils.research_window import resolve_end_date, stamp_provenance

# Reuse RQ3's corrected regime helpers: month labels are contemporaneous and
# each return is attributed to its realised date (no forward-filled month-end
# labels, no one-day cross-month leak), and %Time is measured per trading day.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rq3_stability import (  # noqa: E402  # isort: skip
    classify_regimes as _rq3_classify_regimes,
    regime_shares,
)

setup_logging()
logger = get_logger("rd4")

REAL_START = "2018-01-01"
REAL_END = resolve_end_date()  # as-of resolved at run time
PROVIDER = "baostock"

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


def classify_regimes(benchmark_returns, trading_days=None):
    """Contemporaneous benchmark-regime labels (RQ3 convention).

    The old local copy bucketed by MonthEnd and then reindexed onto a calendar
    day range with ``ffill``: every calendar day carried the previous
    month-end's label, so month M traded on month M-1's regime. RQ3's corrected
    helper labels each trading day with the regime of its own month and
    attributes returns to their realised dates; reuse it instead of a local
    copy that can drift again (R85).
    """
    daily = _rq3_classify_regimes(benchmark_returns, trading_days=trading_days)
    return daily, "基准月收益"


def classify_volatility(returns, trading_days=None):
    """Contemporaneous monthly-volatility labels on trading days (R85).

    Same defect class as the regime labels: the old code resampled to MonthEnd
    and forward-filled across calendar days, so each month carried the
    previous month's volatility bucket. Each trading day is now labelled by the
    volatility of its OWN month, and the forward conventions of the return
    panel are respected (a return labelled t is realised on t+1).
    """
    if trading_days is None:
        trading_days = (returns.index if returns is not None
                        else pd.DatetimeIndex([]))
    days = pd.DatetimeIndex(trading_days).sort_values()

    if returns is None or len(returns) == 0:
        return pd.Series(np.nan, index=days, name="regime", dtype=object), "月波动率"

    mean_ret = returns.mean(axis=1).sort_index().shift(1)
    monthly_vol = mean_ret.resample(pd.offsets.MonthEnd()).std().dropna()
    if monthly_vol.empty:
        logger.warning("No monthly volatility — volatility labels unavailable")
        return pd.Series(np.nan, index=days, name="regime", dtype=object), "月波动率"

    thirds = monthly_vol.quantile([1/3, 2/3])
    labels_by_month = {}
    for date, vol in monthly_vol.items():
        if vol <= thirds.iloc[0]:
            labels_by_month[date.to_period("M")] = "LowVol"
        elif vol >= thirds.iloc[1]:
            labels_by_month[date.to_period("M")] = "HighVol"
        else:
            labels_by_month[date.to_period("M")] = "NormalVol"

    day_labels = [labels_by_month.get(period, np.nan)
                  for period in days.to_period("M")]
    daily = pd.Series(day_labels, index=days, name="regime", dtype=object)
    daily.index.name = "date"
    return daily, "月波动率"


def realised_regime(daily_regime):
    """Align daily regime labels with the realised date of each return row.

    Backtest holding returns use the pipeline's forward convention (row t
    spans t -> t+1). A label series indexed by the month each return was
    realised in must therefore be shifted back one row so a holding-period
    return is matched with the regime of its realised date.
    """
    return daily_regime.sort_index().shift(-1)


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

        # Determine regime for this period (modal regime during the holding
        # window). Holding returns are forward-labelled (row t spans t->t+1),
        # and the daily labels now describe the month a return was REALISED
        # in, so shift the label series back one row to line each return up
        # with the regime of its realised date (R87/R85).
        period_dates = hold_returns.index
        period_regimes = realised_regime(daily_regime).reindex(period_dates).dropna()
        if len(period_regimes) > 0:
            modal_regime = period_regimes.mode()
            regime = modal_regime.iloc[0] if len(modal_regime) > 0 else "Unknown"
        else:
            regime = "Unknown"
        all_regimes.append(regime)

    if len(all_port_rets) < 3:
        # Not enough periods: report NaN, never a fabricated 0% return.
        return {"overall": {"sharpe": np.nan, "ann_ret": np.nan,
                            "mdd": np.nan, "n": 0}}

    port_series = pd.Series(all_port_rets)

    # Overall
    af = np.sqrt(252 / hold_h)
    overall_sharpe = port_series.mean() / port_series.std() * af if port_series.std() > 1e-10 else np.nan
    overall_ann = (1 + port_series.mean()) ** (252 / hold_h) - 1
    cum = (1 + port_series).cumprod()
    overall_mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    result = {
        "overall": {"sharpe": overall_sharpe, "ann_ret": overall_ann, "mdd": overall_mdd, "n": len(port_series)}
    }

    # Per-regime
    port_df = pd.DataFrame({"return": all_port_rets, "regime": all_regimes})
    for regime_name in port_df["regime"].unique():
        subset = port_df[port_df["regime"] == regime_name]["return"]
        if len(subset) < 2:
            continue
        sr = subset.mean() / subset.std() * af if subset.std() > 1e-10 else np.nan
        ar = (1 + subset.mean()) ** (252 / hold_h) - 1
        result[regime_name] = {"sharpe": sr, "ann_ret": ar, "n": len(subset)}

    return result


def run():
    logger.info("=" * 60)
    logger.info("Regime Discovery v4")
    logger.info("=" * 60)

    returns, benchmark = load()
    data_start = str(returns.index.min().date())
    data_end = str(returns.index.max().date())
    logger.info("Data window: %s → %s (requested end %s, provider=%s)",
                data_start, data_end, REAL_END, PROVIDER)

    # Regime classifications. Both classifiers label each *trading day* with
    # the regime of its own month (no calendar-day forward fill).
    classifiers = [
        ("基准收益(Bull/Bear/Sideways)",
         classify_regimes(benchmark, trading_days=returns.index)),
        ("波动率(Low/Normal/High)",
         classify_volatility(returns, trading_days=returns.index)),
    ]

    output_dir = _project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for clf_name, (daily_regime, method_desc) in classifiers:
        labelled = daily_regime.dropna()
        shares = regime_shares(labelled)
        logger.info("Classifier: %s", clf_name)
        logger.info("  Labels: %d/%d trading days (%d unlabelled skipped)",
                    len(labelled), len(daily_regime),
                    int(daily_regime.isna().sum()))
        logger.info("  Distribution (%% of labelled trading days): %s",
                    {k: f"{v:.1f}%" for k, v in shares.items()})

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
            print("  Key Cell S=40 H=80 across regimes:")
            for k in k40:
                for rn in sorted(all_regime_names):
                    v = k.get(f"Sharpe_{rn}", np.nan)
                    if not np.isnan(v):
                        print(f"    {rn:>12}: Sharpe={v:+.4f} (n={k.get(f'n_{rn}', 0)})")

        # State persistence
        print()
        print("  State Persistence:")
        print()
        transitions = defaultdict(int)
        state_durations = defaultdict(list)
        current_state = None
        current_start = None
        for date, state in daily_regime.dropna().items():
            if state != current_state:
                if current_state is not None and current_start is not None:
                    duration = (date - current_start).days
                    state_durations[current_state].append(duration)
                    transitions[(current_state, state)] += 1
                current_state = state
                current_start = date

        print(f"  {'State':<12} {'Avg Dur':>8} {'Max Dur':>8} {'N':>5}")
        for state in sorted([s for s in state_durations if isinstance(s, str)]):
            durations = state_durations[state]
            print(f"  {state:<12} {np.mean(durations):>8.1f}d {max(durations):>8.0f}d {len(durations):>5d}")

        print()

        # Persist the heatmap with its as-of window and provider. This script
        # used to be console-only, so a printed result could not be told apart
        # from a stale one (R84).
        slug = "benchmark" if "基准" in clf_name else "volatility"
        df_out = pd.DataFrame(all_results)
        df_out = stamp_provenance(df_out, data_start, data_end)
        df_out["provider"] = PROVIDER
        out_path = output_dir / f"regime_heatmap_v4_{slug}.csv"
        df_out.to_csv(out_path, index=False)
        logger.info("Heatmap artifact saved: %s", out_path)

    print()
    print("=" * 100)
    print("  结论: 等待解读")
    print("=" * 100)


if __name__ == "__main__":
    run()
