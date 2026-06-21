#!/usr/bin/env python3
"""
OOS 验证: 80d Reversal + Vol Filter

时间分割: 2018-2021 (train/校准) vs 2022-2025 (OOS)
判定标准: OOS Sharpe > 0.2 为通过
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("oos")

SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20
VOL_THRESHOLD = 0.70


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date="2018-01-01", end_date="2025-12-31",
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def backtest_80d(returns, use_vol_filter=False):
    """80d 反转 ± vol filter. 返回 (ret_series, summary)."""
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * SELECT_PCT))
    dates = returns.index
    indices = list(range(HOLD_H, len(dates) - SIGNAL_H, HOLD_H))

    market_ret = returns.mean(axis=1)
    market_vol = market_ret.rolling(20).std()
    vol_threshold = market_vol.quantile(VOL_THRESHOLD)

    past_ret = returns.rolling(SIGNAL_H, min_periods=SIGNAL_H).apply(compound, raw=True)

    rets = []
    for i in indices:
        if use_vol_filter:
            rdate = dates[i]
            if rdate in market_vol.index and market_vol[rdate] > vol_threshold:
                continue

        pr = past_ret.iloc[i]
        valid = pr.dropna().sort_values()
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)
        end = i + HOLD_H
        if end >= len(dates):
            break
        hr = returns.iloc[i + 1:end + 1]
        if len(hr) == 0:
            continue
        ret = (hr[selected.index].mean(axis=1) + 1).prod() - 1
        rets.append(ret)

    ps = pd.Series(rets)
    if len(ps) < 3:
        return ps, {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "n": len(ps)}

    af = np.sqrt(252 / HOLD_H)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / HOLD_H) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ps, {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0, "n": len(ps)}


def main():
    print("=" * 80)
    print("  OOS 验证: 80d Reversal + Vol Filter")
    print("=" * 80)

    returns = load()

    splits = [
        ("In-Sample (2018-2021)", returns["2018":"2021"]),
        ("OOS (2022-2025)",      returns["2022":"2025"]),
        ("Full (2018-2025)",     returns),
    ]

    all_rows = []
    for label, r in splits:
        if len(r) < 200:
            continue

        rets_raw, sum_raw = backtest_80d(r, use_vol_filter=False)
        rets_filt, sum_filt = backtest_80d(r, use_vol_filter=True)

        all_rows.append({
            "period": label, "version": "裸策略",
            "sharpe": sum_raw["sharpe"], "ann_ret": sum_raw["ann_ret"],
            "mdd": sum_raw["mdd"], "n": sum_raw["n"],
        })
        all_rows.append({
            "period": label, "version": "+ Vol Filter",
            "sharpe": sum_filt["sharpe"], "ann_ret": sum_filt["ann_ret"],
            "mdd": sum_filt["mdd"], "n": sum_filt["n"],
        })

        print(f"\n{label}:")
        print(f"  裸策略:    Sharpe={sum_raw['sharpe']:.4f}  AnnRet={sum_raw['ann_ret']*100:.2f}%  "
              f"MDD={sum_raw['mdd']*100:.2f}%  trades={sum_raw['n']}")
        print(f"  +VolFilter: Sharpe={sum_filt['sharpe']:.4f}  AnnRet={sum_filt['ann_ret']*100:.2f}%  "
              f"MDD={sum_filt['mdd']*100:.2f}%  trades={sum_filt['n']}")

    # ── 判定 ──
    oos = [r for r in all_rows if "OOS" in r["period"]]
    oos_raw = [r for r in oos if r["version"] == "裸策略"]
    oos_filt = [r for r in oos if r["version"] == "+ Vol Filter"]

    print()
    print("=" * 80)
    print("  OOS 判定:")
    if oos_raw and oos_raw[0]["sharpe"] > 0.2:
        print(f"  裸策略 OOS Sharpe = {oos_raw[0]['sharpe']:.4f} > 0.2 -> PASS")
    else:
        v = oos_raw[0]["sharpe"] if oos_raw else 0
        print(f"  裸策略 OOS Sharpe = {v:.4f} <= 0.2 -> FAIL")

    if oos_filt and oos_filt[0]["sharpe"] > 0.2:
        print(f"  +VolFilter OOS Sharpe = {oos_filt[0]['sharpe']:.4f} > 0.2 -> PASS")
    else:
        v = oos_filt[0]["sharpe"] if oos_filt else 0
        print(f"  +VolFilter OOS Sharpe = {v:.4f} <= 0.2 -> FAIL")

    # ── 逐年 ──
    print()
    print("  逐年 Sharpe (裸策略 + vol filter):")
    print(f"  {'Year':<8} {'Raw_S':>8} {'Filt_S':>8} {'n':>4}")
    for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        yr = returns[f"{y}-01-01":f"{y}-12-31"]
        if len(yr) < 100:
            continue
        _, sr = backtest_80d(yr, False)
        _, sf = backtest_80d(yr, True)
        print(f"  {y:<8} {sr['sharpe']:>8.4f} {sf['sharpe']:>8.4f} {sr['n']:>4d}")

    # ── 保存 ──
    pd.DataFrame(all_rows).to_csv("results/oos_verification.csv", index=False)
    print()
    print("  Saved to results/oos_verification.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
