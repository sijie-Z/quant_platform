"""alpha-v1.0 运行监控 — 每日检查 + 异常预警.

Usage:
    python monitoring/run_monitor.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import logging
from datetime import datetime

import pandas as pd
import numpy as np

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from trading.reversal_paper_trader import ReversalPaperTrader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('results/monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASELINE_SHARPE = 0.45
BASELINE_ANN_RET = 0.11
MAX_DRAWDOWN_ALERT = -0.30


def check_regime_drift(returns):
    """检测市场状态是否发生转变."""
    market_ret = returns.mean(axis=1)
    recent = market_ret.loc[market_ret.index[-252]:] if len(market_ret) > 252 else market_ret
    full = market_ret

    recent_vol = recent.std() * np.sqrt(252)
    full_vol = full.std() * np.sqrt(252)
    recent_mean = recent.mean() * 252

    flags = []
    if recent_vol > full_vol * 1.5:
        flags.append(f"VOL_SPIKE: recent_vol={recent_vol:.2f} vs hist={full_vol:.2f}")
    if recent_mean < -0.1:
        flags.append(f"BEAR_REGIME: recent_return={recent_mean*100:.1f}%")

    return flags


def main():
    print("=" * 65)
    print("  alpha-v1.0 — 系统运行监控")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # 加载数据
    logger.info("Loading data...")
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date='2018-01-01', end_date='2025-12-31',
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    returns = pipeline.returns
    prices = pipeline.get_close()
    logger.info("Data: %d days, %d assets", len(returns), len(returns.columns))

    # 运行策略
    logger.info("Running strategy...")
    trader = ReversalPaperTrader(initial_capital=10_000_000)
    trader.load_data(returns, prices)
    trader.run()

    # 解析结果
    df = pd.DataFrame(trader.equity_curve)
    totals = df["total"].values
    daily_rets = pd.Series([totals[t] / totals[t-1] - 1 for t in range(1, len(totals))])
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 1e-10 else 0
    ann_ret = (1 + daily_rets.mean()) ** 252 - 1
    cum = totals
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min()
    total_ret = totals[-1] / totals[0] - 1
    n_enter = sum(1 for t in trader.trades if t.action == "enter")

    # ── 状态检查 ──
    alerts = []

    # 1. Sharpe 是否低于基线
    if sharpe < BASELINE_SHARPE * 0.5:
        alerts.append(f"SHARPE_DEGRADE: {sharpe:.2f} vs baseline {BASELINE_SHARPE:.2f}")

    # 2. 回撤是否超限
    if mdd < MAX_DRAWDOWN_ALERT:
        alerts.append(f"DRAWDOWN_HIT: {mdd*100:.1f}% (limit {MAX_DRAWDOWN_ALERT*100:.0f}%)")

    # 3. 市场状态检测
    regime_flags = check_regime_drift(returns)
    alerts.extend(regime_flags)

    # ── 输出 ──
    print()
    print(f"  Performance:")
    print(f"    Sharpe:        {sharpe:>+8.4f} (baseline: {BASELINE_SHARPE})")
    print(f"    年化收益:      {ann_ret*100:>+8.2f}% (baseline: {BASELINE_ANN_RET*100:.0f}%)")
    print(f"    总收益:        {total_ret*100:>+8.2f}%")
    print(f"    最大回撤:      {mdd*100:>8.2f}%")
    print(f"    交易次数:      {n_enter:>8d}")

    if alerts:
        print(f"\n  !! ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a}]")
    else:
        print(f"\n  [OK] No alerts — system nominal")

    # ── 保存 ──
    df.to_csv('results/monitor_equity.csv', index=False)
    summary = pd.DataFrame([{
        "date": datetime.now().strftime('%Y-%m-%d'),
        "sharpe": round(sharpe, 4),
        "ann_ret": round(ann_ret, 4),
        "mdd": round(mdd, 4),
        "total_ret": round(total_ret, 4),
        "n_trades": n_enter,
        "alerts": len(alerts),
    }])
    summary.to_csv('results/monitor_summary.csv', index=False)
    logger.info("Monitor complete. %d alerts.", len(alerts))
    print()


if __name__ == "__main__":
    main()
