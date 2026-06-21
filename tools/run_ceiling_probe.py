#!/usr/bin/env python3
"""
Ceiling Probe: +0.45 是结构上限还是起点?

围绕最优单元 (S=40, H=80) 小幅扩展信号族:
  1. S 扫描 (信号窗口, 以H/2为中心)
  2. H 扫描 (持有期, 围绕80d)
  3. S+H 联合扫描 (最优区域附近网格)
  4. 简单复合信号 (S40 + volatility filter)

目标: 是否存在稳定 > 0.45 的区域?
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np
from itertools import product

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("ceiling")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
SELECT_PCT = 0.20


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def backtest(returns, signal_h, hold_h, select_pct=SELECT_PCT, min_trades=3):
    """80d 风格自定义回测。"""
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * select_pct))
    dates = returns.index
    step = hold_h

    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    indices = list(range(step, len(dates) - max(signal_h, hold_h), step))

    rets = []
    for i in indices:
        pr = past_ret.iloc[i]
        valid = pr.dropna().sort_values()
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)
        end = i + step
        if end >= len(dates):
            break
        hr = returns.iloc[i + 1:end + 1]
        if len(hr) == 0:
            continue
        ret = (hr[selected.index].mean(axis=1) + 1).prod() - 1
        rets.append(ret)

    if len(rets) < min_trades:
        return {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "n": len(rets)}

    ps = pd.Series(rets)
    af = np.sqrt(252 / hold_h)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / hold_h) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0, "n": len(rets)}


def main():
    print("=" * 90)
    print("  Ceiling Probe: +0.45 是上限还是起点?")
    print("=" * 90)

    returns = load()
    logger.info("Data: %d days, %d assets", len(returns), len(returns.columns))

    # ── 1. S 扫描 (H=80固定) ──
    print("\n─── 1. Signal Horizon 扫描 (H=80) ───")
    print(f"  {'S':>4} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8} {'n':>4}")
    s_results = []
    for S in [10, 20, 30, 40, 50, 60, 80]:
        r = backtest(returns, S, 80)
        s_results.append((S, r))
        print(f"  {S:>4d} {r['sharpe']:>8.4f} {r['ann_ret']*100:>7.2f}% {r['mdd']*100:>7.2f}% {r['n']:>4d}")

    best_s = max(s_results, key=lambda x: x[1]["sharpe"])
    print(f"  Best: S={best_s[0]}d, Sharpe={best_s[1]['sharpe']:.4f}")

    # ── 2. H 扫描 (S=40固定) ──
    print("\n─── 2. Holding Horizon 扫描 (S=40) ───")
    print(f"  {'H':>4} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8} {'n':>4}")
    h_results = []
    for H in [40, 60, 80, 100, 120]:
        r = backtest(returns, 40, H)
        h_results.append((H, r))
        print(f"  {H:>4d} {r['sharpe']:>8.4f} {r['ann_ret']*100:>7.2f}% {r['mdd']*100:>7.2f}% {r['n']:>4d}")

    best_h = max(h_results, key=lambda x: x[1]["sharpe"])
    print(f"  Best: H={best_h[0]}d, Sharpe={best_h[1]['sharpe']:.4f}")

    # ── 3. 联合扫描 (S=30-60, H=60-100) ──
    print("\n─── 3. 联合扫描 (S=30-60, H=60-100) ───")
    print(f"  {'S':>4} {'H':>4} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8} {'n':>4}")
    grid = []
    for S, H in product([30, 40, 50, 60], [60, 70, 80, 90, 100]):
        r = backtest(returns, S, H)
        grid.append((S, H, r))
        print(f"  {S:>4d} {H:>4d} {r['sharpe']:>8.4f} {r['ann_ret']*100:>7.2f}% {r['mdd']*100:>7.2f}% {r['n']:>4d}")

    best_g = max(grid, key=lambda x: x[2]["sharpe"])
    print(f"  Best: S={best_g[0]}d, H={best_g[1]}d, Sharpe={best_g[2]['sharpe']:.4f}")

    # ── 4. 复合信号: S40反转 + volatility filter ──
    print("\n─── 4. 复合信号: 反转 + Volatility Filter ───")
    print(f"  {'Filter':<20} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8} {'n':>4}")

    # Volatility filter: 市场波动率高时减少仓位
    market_vol = returns.mean(axis=1).rolling(20).std()
    vol_threshold = market_vol.quantile(0.7)  # top 30% vol = high vol

    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * SELECT_PCT))
    past_ret = returns.rolling(40, min_periods=40).apply(compound, raw=True)

    for desc, hold_h, vol_filter in [
        ("无filter (S=40, H=80)", 80, False),
        ("高vol减半仓位 (S=40, H=80)", 80, True),
        ("无filter (S=40, H=60)", 60, False),
        ("高vol减半仓位 (S=40, H=60)", 60, True),
    ]:
        indices = list(range(hold_h, len(returns) - 40, hold_h))
        rets = []
        for i in indices:
            if vol_filter:
                rdate = returns.index[i]
                if rdate in market_vol.index and market_vol[rdate] > vol_threshold:
                    continue  # skip high-vol periods

            pr = past_ret.iloc[i]
            valid = pr.dropna().sort_values()
            if len(valid) < n_select:
                continue
            selected = valid.head(n_select)
            end = i + hold_h
            if end >= len(returns):
                break
            hr = returns.iloc[i + 1:end + 1]
            if len(hr) == 0:
                continue
            ret = (hr[selected.index].mean(axis=1) + 1).prod() - 1
            rets.append(ret)

        if len(rets) >= 3:
            ps = pd.Series(rets)
            af = np.sqrt(252 / hold_h)
            sh = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
            ann = (1 + ps.mean()) ** (252 / hold_h) - 1
            cum = (1 + ps).cumprod()
            mdd = ((cum - cum.cummax()) / cum.cummax()).min()
            print(f"  {desc:<20} {sh:>8.4f} {ann*100:>7.2f}% {mdd*100:>7.2f}% {len(rets):>4d}")
        else:
            print(f"  {desc:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'0':>4d}")

    # ── 综合结论 ──
    print()
    print("=" * 90)
    baseline = backtest(returns, 40, 80)
    all_candidates = [("S40/H80 (baseline)", baseline)] + \
                     [(f"S{s}/H80", r) for s, r in s_results] + \
                     [(f"S40/H{h}", r) for h, r in h_results] + \
                     [(f"S{s}/H{h}", r) for s, h, r in grid]

    best = max(all_candidates, key=lambda x: x[1]["sharpe"])
    print(f"  Baseline Sharpe: {baseline['sharpe']:.4f}")
    print(f"  Best candidate: {best[0]} Sharpe={best[1]['sharpe']:.4f}")
    if best[1]["sharpe"] > baseline["sharpe"] + 0.05 and best[1]["n"] >= 8:
        print(f"  => +0.45 不是上限。存在稳定提升空间 (delta={best[1]['sharpe']-baseline['sharpe']:+.4f})")
    else:
        print(f"  => +0.45 接近结构上限。小幅扩展未产生稳定提升。")
    print("=" * 90)


if __name__ == "__main__":
    main()
