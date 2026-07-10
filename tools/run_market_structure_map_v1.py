#!/usr/bin/env python3
"""
Market Structure Map v1 — A 股市场结构测绘与稳定性验证

输出:
  1. Sharpe(Signal_H, Hold_H) 全矩阵热力图
  2. 最优区域逐年稳定性检验
  3. 一条明确的"结构结论"

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
logger = get_logger("msm1")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
SIGNAL_HORIZONS = [5, 10, 20, 40, 60, 80, 120, 200]
HOLDING_HORIZONS = [5, 10, 20, 40, 60, 80, 120, 200]
YEAR_WINDOWS = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2018, 2026)]


def load_data():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.benchmark


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def backtest_cell(returns, signal_h, hold_h, min_periods=3):
    """Run one cell of the heatmap: (signal_horizon, holding_horizon).

    Returns dict with sharpe, ann_ret, mdd, n_periods.
    """
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * 0.2))
    dates = returns.index
    step = hold_h

    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    indices = list(range(step, len(dates) - max(signal_h, hold_h), step))

    port_rets = []
    for i in indices:
        sig = past_ret.iloc[i]
        valid = sig.dropna().sort_values()
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)
        end = i + step
        if end >= len(dates):
            break
        hr = returns.iloc[i + 1:end + 1]
        if len(hr) == 0:
            continue
        pr = (hr[selected.index].mean(axis=1) + 1).prod() - 1
        port_rets.append(pr)

    if len(port_rets) < min_periods:
        return {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "n": len(port_rets)}

    ps = pd.Series(port_rets)
    af = np.sqrt(252 / hold_h)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann_ret = (1 + ps.mean()) ** (252 / hold_h) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann_ret, "mdd": mdd if not np.isnan(mdd) else 0, "n": len(port_rets)}


def run():
    logger.info("=" * 60)
    logger.info("Market Structure Map v1")
    logger.info("=" * 60)

    returns, benchmark = load_data()

    # ==================================================================
    # 1. Full-period heatmap
    # ==================================================================
    logger.info("[1/4] Full-period heatmap...")
    matrix_all = []
    for sh in SIGNAL_HORIZONS:
        for hh in HOLDING_HORIZONS:
            r = backtest_cell(returns, sh, hh)
            r["signal_h"], r["hold_h"] = sh, hh
            matrix_all.append(r)
        logger.info("  S=%3d done", sh)

    df_all = pd.DataFrame(matrix_all)
    pivot = df_all.pivot_table(index="signal_h", columns="hold_h", values="sharpe")

    # Find best cell
    best = df_all.loc[df_all["sharpe"].idxmax()]
    logger.info("  Best: S=%dd H=%dd Sharpe=%.4f", best["signal_h"], best["hold_h"], best["sharpe"])

    # ==================================================================
    # 2. Year-by-year stability of optimal region
    # ==================================================================
    logger.info("[2/4] Year-by-year stability of key cells...")
    KEY_CELLS = [(5, 5), (20, 20), (40, 80), (80, 80), (5, 120)]

    yearly = defaultdict(list)
    for year_name, ys, ye in YEAR_WINDOWS:
        yr = returns[ys:ye]
        if len(yr) < 100:
            continue
        for (sh, hh) in KEY_CELLS:
            r = backtest_cell(yr, sh, hh)
            yearly[(sh, hh)].append({"year": year_name, "sharpe": r["sharpe"], "n": r["n"]})
        logger.info("  Year %s done", year_name)

    # ==================================================================
    # 3. Rolling 2-year stability
    # ==================================================================
    logger.info("[3/4] Rolling 2-year stability of best cell...")
    sh_best, hh_best = int(best["signal_h"]), int(best["hold_h"])
    rolling_results = []
    step_days = 63  # ~3 months
    for start in range(0, len(returns) - 504, step_days):
        end = start + 504
        if end >= len(returns):
            break
        yr = returns.iloc[start:end]
        r = backtest_cell(yr, sh_best, hh_best, min_periods=2)
        if not np.isnan(r["sharpe"]):
            rolling_results.append({
                "start": str(returns.index[start])[:10],
                "end": str(returns.index[end])[:10],
                "sharpe": r["sharpe"],
                "ann_ret": r["ann_ret"],
                "n": r["n"],
            })

    df_roll = pd.DataFrame(rolling_results)
    if len(df_roll) > 0:
        positives = (df_roll["sharpe"] > 0).sum()
        logger.info("  Rolling windows: %d/%d positive (%.0f%%)",
                     positives, len(df_roll), positives / len(df_roll) * 100)

    # ==================================================================
    # 4. Report
    # ==================================================================
    print()
    print("=" * 100)
    print("  MARKET STRUCTURE MAP v1 — A股市场结构地图")
    print("=" * 100)

    print()
    print("─── 1. 全周期 Sharpe 矩阵 (2018-2025) ───")
    print()
    print(f"  {'Sig\\Hold':>8}", end="")
    for hh in HOLDING_HORIZONS:
        print(f"  {hh:>7d}", end="")
    print()
    for sh in SIGNAL_HORIZONS:
        print(f"  {sh:>8d}", end="")
        for hh in HOLDING_HORIZONS:
            v = pivot.loc[sh, hh]
            fmt = f"{v:>+7.4f}" if not np.isnan(v) else "    nan "
            if sh == sh_best and hh == hh_best:
                fmt = f" [{v:>+5.4f}]"
            print(f"  {fmt}", end="")
        print()

    print()
    print(f"  [bracketed] = Best cell: S={sh_best}d × H={hh_best}d")
    print()

    print("─── 2. 最优区域逐年稳定性 ───")
    print()
    print(f"  {'Cell':>12}", end="")
    for wn, _, _ in YEAR_WINDOWS:
        print(f"  {wn:>8}", end="")
    print(f"  {'Mean':>8}  {'Stable':>8}")
    print(f"  {'─'*12}", end="")
    for _ in YEAR_WINDOWS:
        print(f"  {'─'*8}", end="")
    print(f"  {'─'*8}  {'─'*8}")

    for (sh, hh) in KEY_CELLS:
        vals = [(r["year"], r["sharpe"]) for r in yearly[(sh, hh)]]
        print(f"  {f'{sh}d x {hh}d':>12}", end="")
        yvals = []
        for wn, _, _ in YEAR_WINDOWS:
            match = [v for y, v in vals if y == wn]
            v = match[0] if match else np.nan
            if not np.isnan(v):
                yvals.append(v)
                print(f"  {v:>+8.4f}", end="")
            else:
                print(f"  {'   N/A':>8}", end="")
        mean_v = np.mean(yvals) if yvals else 0
        pos_ratio = sum(1 for v in yvals if v > 0) / len(yvals) if yvals else 0
        print(f"  {mean_v:>+8.4f}  {pos_ratio:>8.0%}")

    print()
    print("─── 3. 滚动稳定性检验 (2年窗口, 3月步进) ───")
    print()
    if len(df_roll) > 0:
        print(f"  最优单元: S={sh_best}d × H={hh_best}d")
        print(f"  滚动窗口数: {len(df_roll)}")
        print(f"  正Sharpe占比: {positives}/{len(df_roll)} = {positives/len(df_roll)*100:.0f}%")
        print(f"  平均Sharpe: {df_roll['sharpe'].mean():.4f}")
        print(f"  Sharpe标准差: {df_roll['sharpe'].std():.4f}")
        print(f"  Sharpe稳定性(均值/标准差): {df_roll['sharpe'].mean()/df_roll['sharpe'].std() if df_roll['sharpe'].std()>0 else 0:.2f}")
        print()
        # Print some sample rolling windows
        print(f"  滚动窗口样本:")
        print(f"  {'Start':<12} {'End':<12} {'Sharpe':>8} {'AnnRet':>8} {'n':>4}")
        for _, r in df_roll.iterrows():
            print(f"  {r['start']:<12} {r['end']:<12} {r['sharpe']:>+8.4f} {r['ann_ret']:>+8.2%} {r['n']:>4d}")

    print()
    print("─── 4. 结构结论 ───")
    print()
    # Find all cells with Sharpe > 0.3
    strong = df_all[df_all["sharpe"] > 0.3].sort_values("sharpe", ascending=False)
    print(f"  Sharpe > 0.3 的单元: {len(strong)}")
    for _, r in strong.iterrows():
        print(f"    S={int(r['signal_h']):3d}d × H={int(r['hold_h']):3d}d  Sharpe={r['sharpe']:.4f}")

    # Compute regional averages
    print()
    short_short = df_all[(df_all["signal_h"] <= 20) & (df_all["hold_h"] <= 20)]["sharpe"].mean()
    short_long = df_all[(df_all["signal_h"] <= 20) & (df_all["hold_h"] >= 60)]["sharpe"].mean()
    long_short = df_all[(df_all["signal_h"] >= 60) & (df_all["hold_h"] <= 20)]["sharpe"].mean()
    long_long = df_all[(df_all["signal_h"] >= 40) & (df_all["hold_h"] >= 40)]["sharpe"].mean()
    print(f"  区域平均:")
    print(f"    短信号×短持有 (S≤20,H≤20): {short_short:.4f}")
    print(f"    短信号×长持有 (S≤20,H≥60): {short_long:.4f}")
    print(f"    长信号×短持有 (S≥60,H≤20): {long_short:.4f}")
    print(f"    中长信号×中长持有 (S≥40,H≥40): {long_long:.4f}")

    print()
    print("=" * 100)

    # ==================================================================
    # Save
    # ==================================================================
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out / "msm1_full_matrix.csv", index=False)
    if df_roll is not None and len(df_roll) > 0:
        pd.DataFrame(rolling_results).to_csv(out / "msm1_rolling_stability.csv", index=False)
    logger.info("Results saved to results/msm1_*")


if __name__ == "__main__":
    run()
