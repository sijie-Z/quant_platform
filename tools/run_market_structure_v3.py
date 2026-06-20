#!/usr/bin/env python3
"""
Market Structure Discovery v3 — Real A-Share Market

RQ5: Horizon Scan — IC(H) curve from 5d to 240d
RQ6: Size Bucket — Small/Mid/Large cap decomposition
RQ7: Sector Analysis — Industry decomposition

Protocol frozen. Results only. No interpretation.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.cost_model import CostModel
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("ms3")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
INITIAL_CAPITAL = 10_000_000

HORIZONS = [5, 10, 20, 40, 60, 80, 120, 160, 200, 240]
SAMPLE_INTERVAL = 21  # Sample every ~1 month to avoid overlapping


def load_data():
    provider = BaostockDataProvider(cache_enabled=True)
    pipeline = DataPipeline(provider=provider, start_date=REAL_START, end_date=REAL_END,
                            exclude_st=True, exclude_suspended=True)
    pipeline.run()
    prices = pipeline.get_close()
    returns = pipeline.returns
    benchmark = pipeline.benchmark
    metadata = pipeline.metadata
    logger.info("Data: %d days, %d assets", len(prices), len(prices.columns))
    return prices, returns, benchmark, metadata


def compute_horizon_ic(returns, horizon, sample_interval=21):
    """Compute IC between past H-day return and future H-day return.

    For each stock, for each sampled date:
      factor = cumulative return over past H days
      forward = cumulative return over next H days
      IC = spearman rank correlation across stocks

    Samples every `sample_interval` days to reduce overlapping window bias.
    """
    # Forward cumulative returns
    forward_ret = returns.rolling(horizon).apply(
        lambda x: np.prod(1 + x) - 1 if len(x) == horizon else np.nan,
        raw=True
    ).shift(-horizon)

    # Past cumulative returns
    past_ret = returns.rolling(horizon).apply(
        lambda x: np.prod(1 + x) - 1 if len(x) == horizon else np.nan,
        raw=True
    )

    # Sample dates
    all_dates = returns.index
    sampled_dates = all_dates[::sample_interval]
    # Filter to valid range
    sampled_dates = sampled_dates[(sampled_dates >= all_dates[horizon]) &
                                   (sampled_dates <= all_dates[-horizon - 1])]

    ic_values = []
    for date in sampled_dates:
        pf = past_ret.loc[date]
        ff = forward_ret.loc[date]
        valid = pf.notna() & ff.notna()
        if valid.sum() < 10:
            continue
        ic, _ = scipy_stats.spearmanr(pf[valid], ff[valid])
        ic_values.append(ic)

    if not ic_values:
        return {"mean_ic": np.nan, "icir": np.nan, "t_stat": np.nan,
                "pos_ratio": np.nan, "n_obs": 0, "ic_series": []}

    ic_arr = np.array(ic_values)
    n = len(ic_arr)
    return {
        "mean_ic": float(np.nanmean(ic_arr)),
        "icir": float(np.nanmean(ic_arr) / np.nanstd(ic_arr)) if np.nanstd(ic_arr) > 1e-10 else 0,
        "t_stat": float(np.nanmean(ic_arr) / (np.nanstd(ic_arr) / np.sqrt(n))) if np.nanstd(ic_arr) > 1e-10 else 0,
        "pos_ratio": float(np.sum(np.array(ic_arr) > 0) / n),
        "n_obs": n,
        "ic_series": [float(x) for x in ic_arr],
    }


def run():
    logger.info("=" * 60)
    logger.info("Market Structure Discovery v3")
    logger.info("Protocol frozen. Results only.")
    logger.info("=" * 60)

    logger.info("[1/3] Loading data...")
    prices, returns, benchmark, metadata = load_data()

    # Build market cap and sector data
    mcap = None
    sector_map = None
    if metadata is not None:
        if "sector" in metadata.columns and metadata["sector"].nunique() > 1:
            sector_map = metadata["sector"]

    # ==================================================================
    # RQ5: Horizon Scan
    # ==================================================================
    logger.info("[2/3] RQ5: Horizon Scan — IC(H) curve...")
    rq5_results = []
    for H in HORIZONS:
        logger.info("  Horizon: %d days...", H)
        result = compute_horizon_ic(returns, H, SAMPLE_INTERVAL)
        result["horizon"] = H
        rq5_results.append(result)

        logger.info("    IC=%.4f  ICIR=%.4f  t=%.2f  pos=%.1f%% (n=%d)",
                     result["mean_ic"], result["icir"], result["t_stat"],
                     result["pos_ratio"] * 100, result["n_obs"])

    # RQ5a: Backtest reversal signals at key horizons
    logger.info("[2b] RQ5a: Backtest validation at key horizons...")
    bt_results_list = []
    for H in [5, 10, 20, 40, 60, 120]:
        # Build reversal signal: -rank(past_H_return)
        logger.info("  Backtest H=%d...", H)
        past_ret = returns.rolling(H).apply(
            lambda x: np.prod(1 + x) - 1 if len(x) == H else np.nan, raw=True
        )
        signal = -past_ret.rank(axis=1, pct=True) + 0.5  # Reversal signal

        constraints = PortfolioConstraints(
            long_only=True, max_weight=0.05, max_sector_exposure=0.30,
            max_turnover=0.30, lot_size=100,
        )
        cost_model = CostModel(commission=0.0003, stamp_tax=0.001, slippage=0.0005, slippage_model="fixed")
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL, rebalance_frequency="monthly",
            cost_model=cost_model, constraints=constraints,
            optimizer="equal_weight", benchmark="equal_weight",
        )
        bt = engine.run(signal=signal, prices=prices, returns=returns,
                        benchmark_returns=benchmark, sector_map=None, financials=None)
        s = bt.get("summary", {})
        bt_results_list.append({
            "Horizon": H, "Sharpe": s.get("sharpe_ratio", 0),
            "TotalReturn": s.get("total_return", 0),
            "MDD": s.get("max_drawdown", 0),
        })
        logger.info("    Sharpe=%.4f  Ret=%.2f%%  MDD=%.2f%%",
                     bt_results_list[-1]["Sharpe"],
                     bt_results_list[-1]["TotalReturn"] * 100,
                     bt_results_list[-1]["MDD"] * 100)

    # ==================================================================
    # RQ6: Size Bucket
    # ==================================================================
    logger.info("[3/3] RQ6: Size Bucket Analysis...")
    # Use proxy: divide stocks by average price (no mcap in data)
    avg_price = prices.mean()
    n_stocks = len(avg_price)
    terciles = [avg_price.quantile(q) for q in [1/3, 2/3]]

    small_mask = avg_price <= avg_price.quantile(1/3)
    mid_mask = (avg_price > avg_price.quantile(1/3)) & (avg_price <= avg_price.quantile(2/3))
    large_mask = avg_price > avg_price.quantile(2/3)

    size_labels = [
        ("Small", small_mask),
        ("Mid", mid_mask),
        ("Large", large_mask),
    ]

    rq6_results = []
    for size_name, mask in size_labels:
        assets = avg_price[mask].index.tolist()
        ret_subset = returns[assets]
        logger.info("  %s: %d stocks", size_name, len(assets))
        size_row = {"size": size_name}
        for H in [5, 10, 20, 40, 60, 120, 240]:
            result = compute_horizon_ic(ret_subset, H, SAMPLE_INTERVAL)
            size_row[f"IC_{H}d"] = result["mean_ic"]
            size_row[f"t_{H}d"] = result["t_stat"]
        rq6_results.append(size_row)
        logger.info("    ICs: %s", " | ".join(f"{size_row[f'IC_{H}d']:.4f}" for H in [5, 10, 20, 40, 60, 120, 240]))

    # ==================================================================
    # RQ7: Sector Analysis
    # ==================================================================
    logger.info("RQ7: Sector Analysis...")
    if sector_map is not None:
        rq7_results = []
        sectors = sector_map.value_counts()
        major_sectors = sectors[sectors >= 5].index  # Only sectors with 5+ stocks
        logger.info("  %d sectors with 5+ stocks", len(major_sectors))

        for sector in major_sectors:
            assets = sector_map[sector_map == sector].index.tolist()
            assets_in_data = [a for a in assets if a in returns.columns]
            if len(assets_in_data) < 5:
                continue
            ret_subset = returns[assets_in_data]
            sector_row = {"sector": sector, "n_stocks": len(assets_in_data)}
            for H in [5, 20, 60, 120]:
                result = compute_horizon_ic(ret_subset, H, SAMPLE_INTERVAL)
                sector_row[f"IC_{H}d"] = result["mean_ic"]
            rq7_results.append(sector_row)
            logger.info("  %-20s (%2d) 5d=%.4f 20d=%.4f 60d=%.4f 120d=%.4f",
                         sector[:20], len(assets_in_data),
                         sector_row["IC_5d"], sector_row["IC_20d"],
                         sector_row["IC_60d"], sector_row["IC_120d"])

    # ==================================================================
    # Report
    # ==================================================================
    print()
    print("=" * 90)
    print("  MARKET STRUCTURE DISCOVERY v3 — RESULTS")
    print("=" * 90)

    print()
    print("─── RQ5: IC(Horizon) Curve ───")
    print()
    print(f"  {'H':<6} {'IC':>8} {'ICIR':>8} {'t_stat':>8} {'pos_ratio':>10} {'n_obs':>6}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")
    for r in rq5_results:
        print(f"  {r['horizon']:<6} {r['mean_ic']:>8.4f} {r['icir']:>8.4f} {r['t_stat']:>8.2f} {r['pos_ratio']:>10.1%} {r['n_obs']:>6}")

    print()
    print("─── RQ5a: Reversal Backtest ───")
    print()
    print(f"  {'H':<6} {'Sharpe':>8} {'TotalRet':>10} {'MDD':>10}")
    for r in bt_results_list:
        print(f"  {r['Horizon']:<6} {r['Sharpe']:>8.4f} {r['TotalReturn']:>10.2%} {r['MDD']:>10.2%}")

    print()
    print("─── RQ6: Size Bucket IC(H) ───")
    print()
    header = f"  {'Size':<10}" + "".join(f"{'IC_'+str(H)+'d':>10}" for H in [5, 10, 20, 40, 60, 120, 240])
    print(header)
    for r in rq6_results:
        line = f"  {r['size']:<10}"
        for H in [5, 10, 20, 40, 60, 120, 240]:
            v = r.get(f"IC_{H}d", np.nan)
            line += f"{v:>10.4f}" if not np.isnan(v) else f"{'':>10}"
        print(line)

    if rq7_results:
        print()
        print("─── RQ7: Sector IC(H) ───")
        print()
        header = f"  {'Sector':<22} {'N':>4} {'IC_5d':>8} {'IC_20d':>8} {'IC_60d':>8} {'IC_120d':>8}"
        print(header)
        for r in sorted(rq7_results, key=lambda x: abs(x.get("IC_5d", 0)), reverse=True)[:20]:
            print(f"  {r['sector'][:20]:<22} {r['n_stocks']:>4} {r.get('IC_5d', 0):>8.4f} {r.get('IC_20d', 0):>8.4f} {r.get('IC_60d', 0):>8.4f} {r.get('IC_120d', 0):>8.4f}")

    print()
    print("=" * 90)

    # Save raw data
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rq5_results).drop(columns=["ic_series"], errors="ignore").to_csv(out / "ms3_rq5_horizon_ic.csv", index=False)
    pd.DataFrame(bt_results_list).to_csv(out / "ms3_rq5a_backtest.csv", index=False)
    pd.DataFrame(rq6_results).to_csv(out / "ms3_rq6_size_bucket.csv", index=False)
    if rq7_results:
        pd.DataFrame(rq7_results).to_csv(out / "ms3_rq7_sector.csv", index=False)
    logger.info("Results saved to results/ms3_*.csv")


if __name__ == "__main__":
    run()
