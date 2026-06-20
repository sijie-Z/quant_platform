#!/usr/bin/env python3
"""
Regime Router v1 — 验证脚本

运行 RegimeRouterStub 在真实 A 股数据上的完整回测。
对比: S=40/H=80 反转 vs 等权基准.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

# 直接 import router (加入 regime_router 路径)
sys.path.insert(0, str(_project_root / "regime_router"))
from router import RegimeRouterStub

import pandas as pd
import numpy as np
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.utils.logging import setup_logging

setup_logging()
REAL_START = "2018-01-01"
REAL_END = "2025-12-31"


def load_data():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close(), pipeline.benchmark


def main():
    print("=" * 60)
    print("  RegimeRouterStub v1 — 验证")
    print("=" * 60)

    # 加载数据
    print("\n[1/3] 加载数据...")
    returns, prices, benchmark = load_data()
    print(f"  数据: {len(returns)} 天 × {len(returns.columns)} 只股票")

    # 初始化 Router
    router = RegimeRouterStub()

    # 运行回测
    print("\n[2/3] 运行回测 (S=40, H=80)...")
    results = router.run_backtest(returns, prices, benchmark)

    # 输出结果
    print("\n[3/3] 结果:")
    print(router.summary(results))

    # 与等权基准对比
    summary = results.get("summary", {})
    bench_ret = results.get("benchmark_returns", pd.Series(dtype=float))
    port_ret = results.get("daily_returns", pd.Series(dtype=float))

    if len(bench_ret) > 0 and len(port_ret) > 0:
        # 超额收益
        excess = port_ret - bench_ret
        print(f"  超额收益均值: {excess.mean()*100:.4f}%/日")
        print(f"  超额收益 t-stat: {excess.mean() / excess.std() * np.sqrt(len(excess)):.2f}" if excess.std() > 0 else "")

    # 与 RQ5b 结果对照
    print()
    print("  RQ5b 对照 (S=40, H=80):")
    print(f"    RQ5b 自定义回测 Sharpe: 0.4503")
    print(f"    本系统引擎回测 Sharpe: {summary.get('sharpe_ratio', 0):.4f}")
    print()

    # 返回结果供后续分析
    return results


if __name__ == "__main__":
    main()
