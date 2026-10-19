#!/usr/bin/env python3
"""
Market Structure Discovery v3 — Real A-Share Market

RQ5: Horizon Scan — IC(H) curve from 5d to 240d
RQ6: Size Bucket — Small/Mid/Large decomposition (same-day close price proxy)
RQ7: Sector Analysis — Industry decomposition

Protocol frozen. Results only. No interpretation.

Statistical notes (revised):
- ICs are sampled every ~21 trading days, but each horizon-H observation uses
  an H-day forward return window. For H > 21 consecutive IC observations share
  return days, so the naive t = mean / (std / sqrt(n)) overstates significance.
  Every result therefore reports BOTH the naive t and a Newey-West (Bartlett)
  HAC t with lag = ceil(H / sample_interval).
- RQ6 no longer splits on the full-sample average price (that used future
  prices and is not a market cap). Groups are formed cross-sectionally on each
  sampling date from that day's close, i.e. a same-day PRICE PROXY. The data
  pipeline exposes no point-in-time market cap.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import numpy as np
import pandas as pd
from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.utils.logging import get_logger, setup_logging
from scipy import stats as scipy_stats

setup_logging()
logger = get_logger("ms3")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
INITIAL_CAPITAL = 10_000_000

HORIZONS = [5, 10, 20, 40, 60, 80, 120, 160, 200, 240]
SAMPLE_INTERVAL = 21  # Sample every ~1 month; HAC below handles residual overlap

# RQ6 grouping: the pipeline has no point-in-time market cap, so size is a
# same-day close-price proxy. Never use a full-sample statistic for grouping.
SIZE_PROXY = "price_proxy_same_day_close"
SIZE_GROUP_LABELS = ("Small", "Mid", "Large")


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


def naive_tstat(values):
    """t-statistic of the mean under the i.i.d. assumption.

    This is the statistic the original protocol reported; it treats every
    sampled IC as an independent observation even when the forward windows
    overlap. Kept for comparison with the HAC version.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    mean = float(x.mean())
    std = float(x.std())
    if std <= 1e-10:
        return 0.0
    return float(mean / (std / np.sqrt(n)))


def newey_west_tstat(values, lag):
    """t-statistic of the mean with a Newey-West (Bartlett) HAC standard error.

    Overlapping forward windows induce positive autocorrelation in the IC
    series, so the naive standard error is too small. The HAC variance is

        gamma_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) * gamma_j

    with Bartlett weights, divided by n. ``lag`` is the number of
    autocovariance lags in sampled-observation units; for an H-day forward
    window sampled every ``sample_interval`` days, lag = ceil(H / interval).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan
    mean = float(x.mean())
    dev = x - mean
    gamma0 = float(np.dot(dev, dev) / n)
    if gamma0 <= 1e-20:
        return 0.0
    var = gamma0
    max_lag = min(int(lag), n - 1)
    for j in range(1, max_lag + 1):
        gamma_j = float(np.dot(dev[j:], dev[:-j]) / n)
        weight = 1.0 - j / (lag + 1.0)
        var += 2.0 * weight * gamma_j
    if not np.isfinite(var) or var <= 0:
        # Pathological negative autocorrelation can make the HAC variance
        # non-positive; fall back to the i.i.d. variance rather than returning
        # a meaningless complex/infinite t.
        var = gamma0
    return float(mean / np.sqrt(var / n))


def summarize_ic_series(ic_values, hac_lag):
    """Summarise a sampled IC series with both naive and HAC t-statistics."""
    if len(ic_values) == 0:
        return {
            "mean_ic": np.nan, "icir": np.nan, "t_stat": np.nan,
            "t_stat_naive": np.nan, "t_stat_hac": np.nan, "hac_lag": int(hac_lag),
            "pos_ratio": np.nan, "n_obs": 0, "ic_series": [],
        }

    ic_arr = np.asarray(ic_values, dtype=float)
    n = len(ic_arr)
    mean_ic = float(np.nanmean(ic_arr))
    std_ic = float(np.nanstd(ic_arr))
    return {
        "mean_ic": mean_ic,
        "icir": float(mean_ic / std_ic) if std_ic > 1e-10 else 0.0,
        # ``t_stat`` is retained for backward compatibility with older CSV
        # consumers; it is exactly the naive statistic.
        "t_stat": naive_tstat(ic_arr),
        "t_stat_naive": naive_tstat(ic_arr),
        "t_stat_hac": newey_west_tstat(ic_arr, hac_lag),
        "hac_lag": int(hac_lag),
        "pos_ratio": float(np.sum(ic_arr > 0) / n),
        "n_obs": n,
        "ic_series": [float(x) for x in ic_arr],
    }


def compute_past_forward_returns(returns, horizon):
    """H-day trailing and forward cumulative returns.

    ``returns[t]`` is the pipeline's forward one-day return r_{t -> t+1}.
    ``rolling(H)`` at t therefore covers rows t-H+1 .. t, i.e. the return from
    close t-H+1 to close t+1, and the forward window (shifted by -H) covers
    rows t+1 .. t+H, i.e. close t+1 -> close t+H+1. The factor is known at
    close t+1 and the forward window starts at that same close, so the two
    windows partition the price path without overlapping.
    """
    past_ret = (1.0 + returns).rolling(horizon).apply(
        lambda x: np.prod(x) - 1 if len(x) == horizon else np.nan,
        raw=True,
    )
    forward_ret = (1.0 + returns).rolling(horizon).apply(
        lambda x: np.prod(x) - 1 if len(x) == horizon else np.nan,
        raw=True,
    ).shift(-horizon)
    return past_ret, forward_ret


def sample_dates_for_horizon(index, horizon, sample_interval=SAMPLE_INTERVAL):
    """Dates where both the trailing and forward H-day windows are complete."""
    all_dates = index
    if len(all_dates) <= 2 * horizon:
        return all_dates[:0]
    sampled_dates = all_dates[::sample_interval]
    return sampled_dates[(sampled_dates >= all_dates[horizon]) &
                         (sampled_dates <= all_dates[-horizon - 1])]


def compute_horizon_ic(returns, horizon, sample_interval=21):
    """Compute IC between past H-day return and future H-day return.

    For each stock, for each sampled date:
      factor = cumulative return over the past H days (known at close t+1,
               matching the pipeline's forward-return convention)
      forward = cumulative return over the next H days
      IC = spearman rank correlation across stocks

    Returns naive and Newey-West HAC t-statistics. The HAC lag is
    ceil(H / sample_interval): for horizons longer than the sampling interval,
    consecutive sampled forward windows overlap by H - sample_interval days.
    """
    past_ret, forward_ret = compute_past_forward_returns(returns, horizon)
    sampled_dates = sample_dates_for_horizon(returns.index, horizon, sample_interval)

    ic_values = []
    for date in sampled_dates:
        pf = past_ret.loc[date]
        ff = forward_ret.loc[date]
        valid = pf.notna() & ff.notna()
        if valid.sum() < 10:
            continue
        ic, _ = scipy_stats.spearmanr(pf[valid], ff[valid])
        if np.isfinite(ic):
            ic_values.append(float(ic))

    hac_lag = int(np.ceil(horizon / sample_interval))
    return summarize_ic_series(ic_values, hac_lag)


def assign_price_groups(prices, date, n_groups=3):
    """Cross-sectional size groups from the close observed on ``date`` only.

    The pipeline provides no point-in-time market cap, so this is an explicit
    PRICE PROXY: low price -> "Small", high price -> "Large". Only the
    ``prices.loc[date]`` row is used, so future prices cannot change the group
    membership of a past date.

    Returns a Series (asset -> group label). Empty when the date is missing or
    too few names have a price.
    """
    if date not in prices.index:
        return pd.Series(dtype=object)
    px = prices.loc[date].dropna()
    if len(px) < n_groups:
        return pd.Series(dtype=object)

    # rank(method="first") breaks price ties deterministically, so equal
    # prices cannot make qcut fail on duplicate bin edges.
    ranks = px.rank(method="first").to_numpy()
    bucket = np.ceil(ranks * n_groups / len(px)).astype(int)
    bucket = np.clip(bucket, 1, n_groups)
    labels = SIZE_GROUP_LABELS if n_groups == 3 else tuple(f"G{i}" for i in range(1, n_groups + 1))
    return pd.Series([labels[b - 1] for b in bucket], index=px.index, name="size_group")


def compute_horizon_ic_by_price_group(returns, prices, horizon, sample_interval=21,
                                      n_groups=3, min_names=10):
    """Per-date cross-sectional IC within same-day price-proxy size groups.

    On every sampled date the cross-section is split into ``n_groups`` groups
    using that date's close only. The IC is then computed *within* each group
    for that date and averaged over dates. This replaces the old full-sample
    ``prices.mean()`` split, which both used future prices and was not a size
    measure.

    Returns {group_label: summary_dict}, where each summary_dict is produced
    by :func:`summarize_ic_series` (naive + HAC t).
    """
    past_ret, forward_ret = compute_past_forward_returns(returns, horizon)
    sampled_dates = sample_dates_for_horizon(returns.index, horizon, sample_interval)

    labels = SIZE_GROUP_LABELS if n_groups == 3 else tuple(f"G{i}" for i in range(1, n_groups + 1))
    per_group = {label: [] for label in labels}

    for date in sampled_dates:
        groups = assign_price_groups(prices, date, n_groups)
        if groups.empty:
            continue
        pf = past_ret.loc[date]
        ff = forward_ret.loc[date]
        valid = pf.notna() & ff.notna()
        for label in labels:
            members = groups.index[(groups == label) & valid.reindex(groups.index, fill_value=False)]
            if len(members) < min_names:
                continue
            ic, _ = scipy_stats.spearmanr(pf[members], ff[members])
            if np.isfinite(ic):
                per_group[label].append(float(ic))

    hac_lag = int(np.ceil(horizon / sample_interval))
    return {label: summarize_ic_series(values, hac_lag) for label, values in per_group.items()}


def run():
    logger.info("=" * 60)
    logger.info("Market Structure Discovery v3")
    logger.info("Protocol frozen. Results only.")
    logger.info("=" * 60)

    logger.info("[1/3] Loading data...")
    prices, returns, benchmark, metadata = load_data()

    # Build market cap and sector data
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

        logger.info("    IC=%.4f  ICIR=%.4f  t_naive=%.2f  t_hac=%.2f  pos=%.1f%% (n=%d)",
                     result["mean_ic"], result["icir"], result["t_stat_naive"],
                     result["t_stat_hac"], result["pos_ratio"] * 100, result["n_obs"])

    # RQ5a: Backtest reversal signals at key horizons
    logger.info("[2b] RQ5a: Backtest validation at key horizons...")
    bt_results_list = []
    for H in [5, 10, 20, 40, 60, 120]:
        # Build reversal signal: -rank(past_H_return)
        logger.info("  Backtest H=%d...", H)
        past_ret = returns.rolling(H).apply(
            lambda x, H=H: np.prod(1 + x) - 1 if len(x) == H else np.nan, raw=True
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
    logger.info("[3/3] RQ6: Size Bucket Analysis (size proxy: %s)...", SIZE_PROXY)
    logger.info("  Groups are same-day price terciles formed on each sampling date "
                "(no point-in-time market cap is available).")
    rq6_horizons = [5, 10, 20, 40, 60, 120, 240]
    rq6_by_horizon = {}
    for H in rq6_horizons:
        logger.info("  Horizon: %d days...", H)
        rq6_by_horizon[H] = compute_horizon_ic_by_price_group(
            returns, prices, H, SAMPLE_INTERVAL)

    rq6_results = []
    for size_name in SIZE_GROUP_LABELS:
        size_row = {"size": size_name, "size_proxy": SIZE_PROXY}
        for H in rq6_horizons:
            result = rq6_by_horizon[H][size_name]
            size_row[f"IC_{H}d"] = result["mean_ic"]
            size_row[f"t_naive_{H}d"] = result["t_stat_naive"]
            size_row[f"t_hac_{H}d"] = result["t_stat_hac"]
            size_row[f"n_{H}d"] = result["n_obs"]
        rq6_results.append(size_row)
        logger.info("    ICs: %s", " | ".join(f"{size_row[f'IC_{H}d']:.4f}" for H in rq6_horizons))

    # ==================================================================
    # RQ7: Sector Analysis
    # ==================================================================
    logger.info("RQ7: Sector Analysis...")
    rq7_results = []
    if sector_map is not None:
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
                sector_row[f"t_naive_{H}d"] = result["t_stat_naive"]
                sector_row[f"t_hac_{H}d"] = result["t_stat_hac"]
                sector_row[f"n_{H}d"] = result["n_obs"]
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
    print("  t_naive assumes independent samples; t_hac is Newey-West (Bartlett), lag=ceil(H/21).")
    print()
    print(f"  {'H':<6} {'IC':>8} {'ICIR':>8} {'t_naive':>8} {'t_hac':>8} {'lag':>4} {'pos_ratio':>10} {'n_obs':>6}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*4} {'─'*10} {'─'*6}")
    for r in rq5_results:
        print(f"  {r['horizon']:<6} {r['mean_ic']:>8.4f} {r['icir']:>8.4f} {r['t_stat_naive']:>8.2f} {r['t_stat_hac']:>8.2f} {r['hac_lag']:>4} {r['pos_ratio']:>10.1%} {r['n_obs']:>6}")

    print()
    print("─── RQ5a: Reversal Backtest ───")
    print()
    print(f"  {'H':<6} {'Sharpe':>8} {'TotalRet':>10} {'MDD':>10}")
    for r in bt_results_list:
        print(f"  {r['Horizon']:<6} {r['Sharpe']:>8.4f} {r['TotalReturn']:>10.2%} {r['MDD']:>10.2%}")

    print()
    print("─── RQ6: Size Bucket IC(H) ───")
    print(f"  Size proxy: {SIZE_PROXY} (same-day close terciles; NOT market cap).")
    print("  Full naive/HAC t statistics per horizon are in ms3_rq6_size_bucket.csv.")
    print()
    header = f"  {'Size':<10}" + "".join(f"{'IC_'+str(H)+'d':>10}" for H in rq6_horizons)
    header += f" {'t20_naive':>10} {'t20_hac':>9} {'t60_naive':>10} {'t60_hac':>9}"
    print(header)
    for r in rq6_results:
        line = f"  {r['size']:<10}"
        for H in rq6_horizons:
            v = r.get(f"IC_{H}d", np.nan)
            line += f"{v:>10.4f}" if not np.isnan(v) else f"{'':>10}"
        for H in (20, 60):
            line += f" {r.get(f't_naive_{H}d', np.nan):>10.2f} {r.get(f't_hac_{H}d', np.nan):>9.2f}"
        print(line)

    if rq7_results:
        print()
        print("─── RQ7: Sector IC(H) ───")
        print()
        header = f"  {'Sector':<22} {'N':>4} {'IC_5d':>8} {'IC_20d':>8} {'IC_60d':>8} {'IC_120d':>8}"
        header += f" {'t20_naive':>10} {'t20_hac':>9} {'t60_naive':>10} {'t60_hac':>9}"
        print(header)
        for r in sorted(rq7_results, key=lambda x: abs(x.get("IC_5d", 0)), reverse=True)[:20]:
            print(f"  {r['sector'][:20]:<22} {r['n_stocks']:>4} {r.get('IC_5d', 0):>8.4f} {r.get('IC_20d', 0):>8.4f} {r.get('IC_60d', 0):>8.4f} {r.get('IC_120d', 0):>8.4f} "
                  f"{r.get('t_naive_20d', np.nan):>10.2f} {r.get('t_hac_20d', np.nan):>9.2f} "
                  f"{r.get('t_naive_60d', np.nan):>10.2f} {r.get('t_hac_60d', np.nan):>9.2f}")

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
