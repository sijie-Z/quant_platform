#!/usr/bin/env python3
"""
Cost Layer Test: 80d Reversal + Vol Filter

测试交易成本对净 Sharpe 的影响。
成本模型: commission + slippage + spread
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import numpy as np
import pandas as pd
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import get_logger, setup_logging
from quant_platform.utils.research_window import resolve_end_date

setup_logging()
logger = get_logger("cost")

SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20
INITIAL_CAPITAL = 10_000_000
# Resolved at run time (QUANT_RESEARCH_END_DATE or today).
END_DATE = resolve_end_date()


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date="2018-01-01", end_date=END_DATE,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close()


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


def backtest_with_cost(returns, prices, use_vol_filter=False,
                       commission_bps=10, slippage_bps=5,
                       stamp_tax_bps=10):
    """80d 反转 + 成本模型.

    cost per trade = commission + slippage (+ stamp tax on sells)
    A 股印花税仅卖出征收；旧场景全部漏掉，净 Sharpe 与 VIABLE 判定被高估。

    Entry is at close(i+1): the signal at row i uses pipeline forward returns
    (data/pipeline.py shift(-1)) through i, which are only known at close i+1.
    The old code entered at close(i), earning returns.values[i] -- the
    signal-day forward return.
    """
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * SELECT_PCT))
    dates = returns.index
    indices = list(range(HOLD_H, len(dates) - SIGNAL_H, HOLD_H))

    market_ret = returns.mean(axis=1)
    market_vol = market_ret.rolling(20).std()
    vol_threshold = market_vol.quantile(0.70)

    past_ret = returns.rolling(SIGNAL_H, min_periods=SIGNAL_H).apply(compound, raw=True)

    cost_buy = (commission_bps + slippage_bps) / 10000  # bps to decimal
    cost_sell = (commission_bps + slippage_bps + stamp_tax_bps) / 10000
    cash = float(INITIAL_CAPITAL)
    portfolio_log = []  # (date, value)

    for i in indices:
        rdate = dates[i]

        if use_vol_filter:
            if rdate in market_vol.index and market_vol[rdate] > vol_threshold:
                continue

        pr = past_ret.iloc[i]
        valid = pr.dropna().sort_values()
        if len(valid) < n_select:
            continue
        selected = valid.head(n_select)

        entry_idx = i + 1
        exit_idx = entry_idx + HOLD_H
        if exit_idx >= len(dates):
            break
        entry_date = dates[entry_idx]
        exit_date = dates[exit_idx]
        if entry_date not in prices.index or exit_date not in prices.index:
            continue

        # 每只股票投入资金
        invest_per_stock = cash / len(selected)

        # 买入价 (信号次日收盘) / 卖出价 (持有期末收盘)
        entry_prices = prices.loc[entry_date, selected.index]
        exit_prices = prices.loc[exit_date, selected.index]

        # 计算收益 (含成本)
        total_return = 0.0
        for asset in selected.index:
            if asset not in entry_prices.index or asset not in exit_prices.index:
                continue
            ep = entry_prices[asset]
            xp = exit_prices[asset]
            if pd.isna(ep) or pd.isna(xp) or ep == 0:
                continue

            shares = (invest_per_stock / ep)  # 买入股数
            cost_side_buy = invest_per_stock * cost_buy  # 买入成本
            gross_sale = shares * xp  # 卖出毛收入
            cost_side_sell = gross_sale * cost_sell  # 卖出成本(含印花税)
            net_return = (
                gross_sale - cost_side_sell - invest_per_stock - cost_side_buy
            ) / invest_per_stock
            total_return += net_return * (invest_per_stock / cash)

        cash = cash * (1 + total_return)

        # 记下净值
        portfolio_log.append({"date": exit_date, "value": cash})

    if len(portfolio_log) < 3:
        return {"sharpe": np.nan, "ann_ret": 0, "total_return": 0,
                "mdd": np.nan, "n": len(portfolio_log), "total_cost": 0}

    # 计算收益序列
    df = pd.DataFrame(portfolio_log)
    values = df["value"].values

    # 直接用每期收益
    ps = pd.Series([(values[t] / values[t-1] - 1) for t in range(1, len(values))])

    af = np.sqrt(252 / HOLD_H)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / HOLD_H) - 1
    total_ret = (values[-1] / values[0] - 1)

    return {"sharpe": sharpe, "ann_ret": ann, "total_return": total_ret,
            "mdd": _max_drawdown(values), "n": len(ps)}


def _max_drawdown(values):
    cum = values / values[0]
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return dd.min() if len(dd) > 0 else 0


def main():
    print("=" * 90)
    print("  Cost Layer Test: 80d Reversal + Vol Filter")
    print("=" * 90)

    returns, prices = load()
    data_start = str(returns.index.min().date())
    data_end = str(returns.index.max().date())
    logger.info("Data window: %s → %s (requested end %s)", data_start, data_end, END_DATE)
    print(f"  Data window: {data_start} → {data_end} (requested end {END_DATE})")
    print("  Sells include 10bps A-share stamp tax in every scenario.")

    # 成本场景 (comm, slippage, stamp tax; stamp applies to sells only)
    scenarios = [
        ("理想 (无成本)", 0, 0, 0),
        ("低 (5+2bps)",  5, 2, 10),
        ("中 (10+5bps)", 10, 5, 10),
        ("高 (20+10bps)", 20, 10, 10),
    ]

    print()
    print(f"  {'Scenario':<20} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8} {'Cost':>8} {'n':>4}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*4}")

    results = []
    for label, comm, slippage, stamp in scenarios:
        r = backtest_with_cost(returns, prices, use_vol_filter=True,
                               commission_bps=comm, slippage_bps=slippage,
                               stamp_tax_bps=stamp)
        results.append((label, r))
        print(f"  {label:<20} {r['sharpe']:>8.4f} {r['ann_ret']*100:>7.2f}% "
              f"{r['mdd']*100:>7.2f}% {comm+slippage:>5d}+{stamp:<3d}bps {r['n']:>4d}")

    print()
    print("=" * 90)

    # 判定
    baseline = results[0][1]["sharpe"]
    mid = results[2][1]["sharpe"]

    print("  判定:")
    print(f"  无成本 Sharpe: {baseline:.4f}")
    print(f"  中成本 Sharpe: {mid:.4f}")
    if not np.isfinite(mid):
        print("  -> UNDETERMINED: too few trades to judge viability")
    elif mid > 0.3:
        print("  -> Net Sharpe > 0.3: VIABLE (可交易)")
    elif mid > 0.2:
        print("  -> Net Sharpe 0.2-0.3: MARGINAL (边际)")
    else:
        print("  -> Net Sharpe < 0.2: NOT TRADABLE (不可交易)")
    print("=" * 90)


if __name__ == "__main__":
    main()
