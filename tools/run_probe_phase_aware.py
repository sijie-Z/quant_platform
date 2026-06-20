#!/usr/bin/env python3
"""
Probe: Phase-Aware Execution — NO-GO ZONE 边界测试

问题: fixed-frequency failure 是 fundamental 还是 policy class artifact?
方法: 最简单的 phase proxy + conditional entry vs fixed-grid baseline

不是研究。是 falsification test。
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("probe")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close(), pipeline.benchmark


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def phase_proxy(returns, lookback=20):
    """最简单的相位估计: 过去 lookback 日的平均日收益符号.

    正值 → 上升相位 (bull phase)
    负值 → 下降相位 (bear phase)

    另一种相位: 滚动自相关 (市场动量持续性)
    """
    # 等权市场收益
    market_ret = returns.mean(axis=1)
    rolling_mean = market_ret.rolling(lookback).mean()
    # phase = +1 (bull), -1 (bear)
    phase = rolling_mean.apply(lambda x: 1 if x > 0 else -1)
    return phase


def phase_proxy2(returns, lookback=20):
    """第二种相位: 滚动自相关. 高自相关 = 趋势延续, 低/负 = 反转."""
    market_ret = returns.mean(axis=1)
    autocorr = market_ret.rolling(lookback).apply(
        lambda x: x.autocorr() if len(x) >= 4 else 0, raw=False
    )
    phase = autocorr.apply(lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0))
    return phase


def baseline_fixed_grid(signal, returns, past_ret, hold_h=80):
    """Baseline: 固定持有期反转. 每 hold_h 天调仓, 买跌幅最大 20%."""
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * SELECT_PCT))
    dates = returns.index
    indices = list(range(hold_h, len(dates) - SIGNAL_H, hold_h))

    rets = []
    for i in indices:
        pr = past_ret.iloc[i]
        valid = pr.dropna().sort_values()  # 升序: 跌最多的排最前
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)  # 选跌最多的
        end = i + hold_h
        if end >= len(dates):
            break
        hr = returns.iloc[i + 1:end + 1]
        if len(hr) == 0:
            continue
        pr_val = (hr[selected.index].mean(axis=1) + 1).prod() - 1
        rets.append(pr_val)

    ps = pd.Series(rets)
    af = np.sqrt(252 / hold_h)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / hold_h) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0, "n": len(ps)}


def phase_conditioned(signal, returns, past_ret, phase, hold_h=80, min_phase_duration=10):
    """Phase-conditioned: 只在 phase 满足条件时入仓.

    规则: 只在 phase 在入仓前连续 min_phase_duration 日为正时交易.
    选股: 买入跌幅最大的 20%.
    """
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * SELECT_PCT))
    dates = returns.index
    indices = list(range(hold_h, len(dates) - SIGNAL_H, hold_h))

    rets = []
    skipped = 0
    for i in indices:
        # 检查 phase 条件
        phase_window = phase.iloc[i - min_phase_duration:i]
        if len(phase_window) < min_phase_duration:
            continue
        if phase_window.sum() < min_phase_duration * 0.8:
            skipped += 1
            continue

        pr = past_ret.iloc[i]
        valid = pr.dropna().sort_values()  # 跌最多的排最前
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)
        end = i + hold_h
        if end >= len(dates):
            break
        hr = returns.iloc[i + 1:end + 1]
        if len(hr) == 0:
            continue
        pr_val = (hr[selected.index].mean(axis=1) + 1).prod() - 1
        rets.append(pr_val)

    if len(rets) < 3:
        return {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "n": len(rets), "skipped": skipped}

    ps = pd.Series(rets)
    af = np.sqrt(252 / hold_h)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / hold_h) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0,
            "n": len(rets), "skipped": skipped}


def main():
    print("=" * 70)
    print("  PROBE: Phase-Aware Execution — NO-GO ZONE 边界测试")
    print("=" * 70)

    returns, prices, benchmark = load()
    logger.info("Data: %d days, %d assets", len(returns), len(returns.columns))

    # 信号
    past_ret = returns.rolling(SIGNAL_H, min_periods=SIGNAL_H).apply(compound, raw=True)
    signal = -past_ret.rank(axis=1, pct=True)
    signal = signal - 0.5

    # Phase proxies
    phase_ma = phase_proxy(returns, lookback=20)
    phase_ac = phase_proxy2(returns, lookback=20)

    # ── Baseline: fixed-grid ──
    base = baseline_fixed_grid(signal, returns, past_ret, HOLD_H)
    print(f"\nBaseline (fixed-grid 80d):")
    print(f"  Sharpe={base['sharpe']:.4f}  AnnRet={base['ann_ret']*100:.2f}%  "
          f"MDD={base['mdd']*100:.2f}%  n={base['n']}")

    # ── Phase-conditioned (MA) ──
    pc_ma = phase_conditioned(signal, returns, past_ret, phase_ma, HOLD_H)
    if not np.isnan(pc_ma["sharpe"]):
        print(f"\nPhase-conditioned (MA proxy):")
        print(f"  Sharpe={pc_ma['sharpe']:.4f}  AnnRet={pc_ma['ann_ret']*100:.2f}%  "
              f"MDD={pc_ma['mdd']*100:.2f}%  trades={pc_ma['n']}  skipped={pc_ma['skipped']}")
    else:
        print(f"\nPhase-conditioned (MA proxy): insufficient trades (< 3)")

    # ── Phase-conditioned (AC) ──
    pc_ac = phase_conditioned(signal, returns, past_ret, phase_ac, HOLD_H)
    if not np.isnan(pc_ac["sharpe"]):
        print(f"\nPhase-conditioned (AC proxy):")
        print(f"  Sharpe={pc_ac['sharpe']:.4f}  AnnRet={pc_ac['ann_ret']*100:.2f}%  "
              f"MDD={pc_ac['mdd']*100:.2f}%  trades={pc_ac['n']}  skipped={pc_ac['skipped']}")
    else:
        print(f"\nPhase-conditioned (AC proxy): insufficient trades (< 3)")

    # ── 结论 ──
    print()
    print("=" * 70)
    pc_sharpes = [("MA", pc_ma["sharpe"]), ("AC", pc_ac["sharpe"])]
    improved = any(not np.isnan(s) and s > base["sharpe"] + 0.1 for _, s in pc_sharpes)
    if improved:
        print("  => Phase-conditioned improved Sharpe: NO-GO ZONE is policy class artifact")
        print("     (fixed-grid is bad, but control space expansion works)")
    else:
        print("  => Phase-conditioned did NOT improve: NO-GO ZONE appears fundamental")
        print("     (not solvable within explored control family)")
    print("=" * 70)


if __name__ == "__main__":
    main()
