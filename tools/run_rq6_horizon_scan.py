#!/usr/bin/env python3
"""
RQ6: Execution-Aware Horizon Scan

找 H* = argmin |Research_Sharpe(H) - Execution_Sharpe(H)|

每个 H 运行两种回测:
  - Research: 自定义, 持有期=H (无执行约束)
  - Execution: BacktestEngine, 月频 (真实约束)

协议冻结. 不修改信号. 不改引擎.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.cost_model import CostModel
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("rq6")

REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
SIGNAL_H = 40  # 固定信号窗口 (发现 v3 的最佳反转信号)
HORIZONS = [5, 10, 20, 30, 40, 60, 80, 120, 160]


def load():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    return pipeline.returns, pipeline.get_close(), pipeline.benchmark


def compound(x):
    return np.prod(1 + x) - 1 if len(x) > 0 else 0


# ── Research 回测 (理想执行, 持有期=H) ──

def research_backtest(returns, signal_h, hold_h, select_pct=0.20):
    """自定义回测: 持有期 = hold_h, 无执行约束."""
    n_stocks = len(returns.columns)
    n_select = max(1, int(n_stocks * select_pct))
    dates = returns.index
    step = hold_h

    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    indices = list(range(step, len(dates) - max(signal_h, hold_h), step))

    rets = []
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
        rets.append(pr)

    if len(rets) < 3:
        return {"sharpe": np.nan, "ann_ret": 0, "mdd": 0, "n": len(rets), "turnover": 1.0}

    ps = pd.Series(rets)
    af = np.sqrt(252 / hold_h)
    sharpe = ps.mean() / ps.std() * af if ps.std() > 1e-10 else 0
    ann = (1 + ps.mean()) ** (252 / hold_h) - 1
    cum = (1 + ps).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    # Turnover: 每期买卖一次 = 200% / n_periods
    turnover = 2.0 / hold_h * 252 if hold_h > 0 else 1.0

    return {"sharpe": sharpe, "ann_ret": ann, "mdd": mdd if not np.isnan(mdd) else 0,
            "n": len(rets), "turnover": turnover}


# ── Execution 回测 (真实约束, 月频) ──

def execution_backtest(signal, prices, returns, benchmark):
    """通过 BacktestEngine 回测 (月频调仓)."""
    constraints = PortfolioConstraints(
        long_only=True, max_weight=0.05, max_sector_exposure=0.30,
        max_turnover=0.30, lot_size=100,
    )
    cost_model = CostModel(commission=0.0003, stamp_tax=0.001, slippage=0.0005, slippage_model="fixed")
    engine = BacktestEngine(
        10_000_000, "monthly", cost_model, constraints, "equal_weight", "equal_weight",
    )
    results = engine.run(signal=signal, prices=prices, returns=returns,
                         benchmark_returns=benchmark, sector_map=None, financials=None)
    s = results.get("summary", {})
    return {
        "sharpe": s.get("sharpe_ratio", 0),
        "ann_ret": s.get("annual_return", 0),
        "mdd": s.get("max_drawdown", 0),
        "n_rebalances": s.get("n_rebalances", 0),
        "turnover": s.get("turnover", s.get("avg_turnover", 0)),
    }


# ── 信号生成 ──

def generate_signal(returns, signal_h=SIGNAL_H):
    """反转信号: S=40."""
    past_ret = returns.rolling(signal_h, min_periods=signal_h).apply(compound, raw=True)
    signal = -past_ret.rank(axis=1, pct=True)
    signal = signal - 0.5
    return signal


# ── 主扫描 ──

def scan():
    logger.info("=" * 60)
    logger.info("RQ6: Execution-Aware Horizon Scan")
    logger.info("=" * 60)

    returns, prices, benchmark = load()
    raw_signal = generate_signal(returns)  # S=40 作参考
    logger.info("数据: %d 天 x %d 只", len(returns), len(returns.columns))

    results = []
    for H in HORIZONS:
        S = max(5, H // 2)  # 协议: 信号窗口 = 持有期的一半 (至少 5 天)
        logger.info("H=%d (S=%d)...", H, S)

        # Research 回测 (理想执行, 持有期=H)
        r = research_backtest(returns, signal_h=S, hold_h=H)
        research_sharpe = r["sharpe"]

        # Execution 回测 (月频, 信号=S)
        sig = generate_signal(returns, signal_h=S)
        e = execution_backtest(sig, prices, returns, benchmark)
        exec_sharpe = e["sharpe"]

        gap = research_sharpe - exec_sharpe if not np.isnan(research_sharpe) else 0
        decay_ratio = gap / abs(research_sharpe) if not np.isnan(research_sharpe) and abs(research_sharpe) > 1e-6 else np.nan

        results.append({
            "H": H, "S": S,
            "Research_Sharpe": research_sharpe,
            "Research_AnnRet": r["ann_ret"],
            "Research_MDD": r["mdd"],
            "Research_N": r["n"],
            "Execution_Sharpe": exec_sharpe,
            "Execution_AnnRet": e["ann_ret"],
            "Execution_MDD": e["mdd"],
            "Execution_N": e["n_rebalances"],
            "Gap": gap,
            "Decay_Ratio": decay_ratio,
            "Research_Turnover": r["turnover"],
        })

        logger.info("  Research=%.4f  Execution=%.4f  Gap=%.4f  Decay=%.2f%%",
                     research_sharpe, exec_sharpe, gap, decay_ratio * 100 if not np.isnan(decay_ratio) else 0)

    # ── 输出 ──
    df = pd.DataFrame(results)

    print()
    print("=" * 120)
    print("  RQ6: EXECUTION-AWARE HORIZON SCAN")
    print("=" * 120)
    print()
    print(f"  {'H':>4} {'S':>4} {'Research_S':>10} {'Exec_S':>10} {'Gap':>8} {'Decay%':>8} "
          f"{'Res_Ann':>8} {'Exec_Ann':>8} {'Res_MDD':>8} {'Exec_MDD':>8} {'Tech_Res':>8} {'Turn':>8}")
    print(f"  {'─'*4} {'─'*4} {'─'*10} {'─'*10} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for _, r in df.iterrows():
        d_r = r["Decay_Ratio"]
        decay_str = f"{d_r*100:>7.1f}%" if pd.notna(d_r) else f"{'':>8}"
        print(f"  {int(r['H']):>4d} {int(r['S']):>4d} {r['Research_Sharpe']:>10.4f} {r['Execution_Sharpe']:>10.4f} "
              f"{r['Gap']:>+8.4f} {decay_str} "
              f"{r['Research_AnnRet']*100:>7.2f}% {r['Execution_AnnRet']*100:>7.2f}% "
              f"{r['Research_MDD']*100:>7.2f}% {r['Execution_MDD']*100:>7.2f}% "
              f"{int(r['Research_N']):>8d} {r['Research_Turnover']:>8.2f}%")

    # 找 H*
    # 条件: Execution_Sharpe > 0 且 Decay_Ratio 最小
    valid = df[(df["Execution_Sharpe"] > 0) & df["Research_Sharpe"].notna()]
    if len(valid) > 0:
        h_star = valid.loc[valid["Decay_Ratio"].idxmin()]
        print()
        print(f"  H* = {int(h_star['H'])} (Execution Sharpe={h_star['Execution_Sharpe']:.4f}, "
              f"Decay Ratio={h_star['Decay_Ratio']*100:.1f}%)")
        print()

        # 最简 H*: min gap
        h_star_simple = df.loc[df["Gap"].idxmin()]
        print(f"  H* (min gap) = {int(h_star_simple['H'])} (Gap={h_star_simple['Gap']:.4f})")

    # 判断形态
    print()
    print("  形态判断:")
    r_sharpes = df["Research_Sharpe"].values
    e_sharpes = df["Execution_Sharpe"].values
    if np.argmax(r_sharpes) == np.argmax(e_sharpes):
        print(f"  单峰一致: Research peak ≈ Execution peak at H={HORIZONS[np.argmax(r_sharpes)]}")
    else:
        print(f"  偏移峰: Research peak at H={HORIZONS[np.argmax(r_sharpes)]}, "
              f"Execution peak at H={HORIZONS[np.argmax(e_sharpes)]}")

    # 保存
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rq6_horizon_scan.csv", index=False)
    logger.info("Saved to results/rq6_horizon_scan.csv")


if __name__ == "__main__":
    scan()
