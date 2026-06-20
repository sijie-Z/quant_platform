#!/usr/bin/env python3
"""
ExecutionAdapter v1.1 — 三路对比验证

对比:
1. RQ5b 自定义回测 (S=40/H=80, ground truth, ~0.45)
2. 原始信号 + BacktestEngine (未适配, ~-0.27)
3. Adapter position overlap (适配后, 预期 ~0.35)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))
sys.path.insert(0, str(_project_root / "regime_router"))
from router import RegimeRouterStub
from adapter import ExecutionAdapter

import pandas as pd
import numpy as np

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.cost_model import CostModel
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
INITIAL_CAPITAL = 10_000_000


def load_data():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close(), pipeline.benchmark


def rq5b_baseline(returns):
    """RQ5b 自定义回测: S=40, H=80, 每 80 日调仓."""
    H, n_stocks = 80, len(returns.columns)
    n_select = max(1, int(n_stocks * 0.20))
    past_ret = returns.rolling(40, min_periods=40).apply(
        lambda x: np.prod(1 + x) - 1 if len(x) == 40 else np.nan, raw=True)
    indices = list(range(H, len(returns) - max(40, H), H))
    rets = []
    for i in indices:
        sig = past_ret.iloc[i].dropna().sort_values()
        if len(sig) < n_select:
            continue
        hr = returns.iloc[i + 1:i + H + 1]
        if len(hr) == 0:
            continue
        pr = (hr[sig.head(n_select).index].mean(axis=1) + 1).prod() - 1
        rets.append(pr)
    ps = pd.Series(rets)
    af = np.sqrt(252 / H)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / H) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0, "n": len(ps)}


def main():
    print("=" * 70)
    print("  ExecutionAdapter v1.1 — 三路对比验证")
    print("=" * 70)

    print("\n[1] 加载数据...")
    returns, prices, benchmark = load_data()
    print(f"  数据: {len(returns)} 天 x {len(returns.columns)} 只")

    # ── 基线: RQ5b (ground truth) ──
    baseline = rq5b_baseline(returns)
    print(f"\n[2] RQ5b 基线: Sharpe={baseline['sharpe']:.4f} "
          f"AnnRet={baseline['ann_ret']*100:.2f}% MDD={baseline['mdd']*100:.2f}%")

    # ── 原始信号 + 引擎 (未适配) ──
    router = RegimeRouterStub()
    raw_signal = router.generate_signal(returns, 40)
    constraints = PortfolioConstraints(long_only=True, max_weight=0.05,
                                       max_sector_exposure=0.30, max_turnover=0.30, lot_size=100)
    cost_model = CostModel(commission=0.0003, stamp_tax=0.001, slippage=0.0005, slippage_model="fixed")
    engine = BacktestEngine(INITIAL_CAPITAL, "monthly", cost_model, constraints, "equal_weight", "equal_weight")
    raw_bt = engine.run(raw_signal, prices, returns, benchmark, None, None)
    rs = raw_bt["summary"]
    print(f"[3] 原始+引擎: Sharpe={rs['sharpe_ratio']:.4f} "
          f"AnnRet={rs['annual_return']*100:.2f}% MDD={rs['max_drawdown']*100:.2f}%")

    # ── Adapter (position overlap) ──
    adapter = ExecutionAdapter(signal_h=40, hold_h=80)
    ed = returns.index.to_series()
    rebalance_dates = ed.groupby([ed.dt.year, ed.dt.month]).last().tolist()
    port_returns = adapter.run(returns, prices, rebalance_dates)
    port_metrics = all_metrics(port_returns, None)
    print(f"[4] Adapter:    Sharpe={port_metrics['sharpe_ratio']:.4f} "
          f"AnnRet={port_metrics['annual_return']*100:.2f}% "
          f"MDD={port_metrics['max_drawdown']*100:.2f}%")

    # ── 对比表 ──
    print("\n" + "=" * 70)
    print("  三路对比结果")
    print("=" * 70)
    print(f"  {'方法':<25} {'Sharpe':>8} {'AnnRet':>8} {'MDD':>8} {'n':>5}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
    print(f"  {'RQ5b 基线 (80d)':<25} {baseline['sharpe']:>8.4f} {baseline['ann_ret']*100:>7.2f}% "
          f"{baseline['mdd']*100:>7.2f}% {baseline['n']:>5d}")
    print(f"  {'原始+引擎 (月频)':<25} {rs['sharpe_ratio']:>8.4f} {rs['annual_return']*100:>7.2f}% "
          f"{rs['max_drawdown']*100:>7.2f}% {rs.get('n_rebalances',0):>5d}")
    print(f"  {'Adapter overlap':<25} {port_metrics['sharpe_ratio']:>8.4f} "
          f"{port_metrics['annual_return']*100:>7.2f}% "
          f"{port_metrics['max_drawdown']*100:>7.2f}% "
          f"{adapter.metrics['n_tranches_opened']:>5d}")

    alpha_decay = baseline["sharpe"] - port_metrics["sharpe_ratio"]
    print(f"\n  Alpha Decay (研究→执行): {alpha_decay:.4f} "
          f"({alpha_decay/abs(baseline['sharpe'])*100:.1f}%)")
    print(f"  Adapter tranches: {adapter.metrics['n_tranches_opened']} opened, "
          f"{adapter.metrics['n_tranches_closed']} closed")

    return {"baseline": baseline, "engine": rs, "adapter": port_metrics}


if __name__ == "__main__":
    main()
