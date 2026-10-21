#!/usr/bin/env python3
"""
RQ3: Stability + Regime — Alpha Discovery v2

Protocol-frozen experiment: Alpha 什么时候有效？

Experiment 3.1 — Year-by-year stability (fixed windows)
Experiment 3.2 — Market regime decomposition (Bull/Bear/Sideways)

Regime convention (revised): each trading day is labelled with the regime of
ITS OWN calendar month (month M's benchmark return on month M's trading days).
The previous implementation forward-filled month-end labels into the following
month, so February traded on January's label (62.7% of trading days carried a
stale label). Labels are contemporaneous and therefore usable only for ex-post
performance decomposition, not as a tradable signal. Days without a benchmark
label (e.g. before the benchmark starts) are explicitly skipped and counted.

只记录结果，不解释。
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import numpy as np
import pandas as pd
from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.metrics import all_metrics
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.factors.fundamental import register_all as register_fundamental
from quant_platform.factors.processing import process_factor
from quant_platform.factors.registry import get_registry
from quant_platform.factors.technical import register_all as register_technical
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.utils.logging import get_logger, setup_logging
from quant_platform.utils.research_window import (
    resolve_end_date,
    stamp_provenance,
    year_windows,
)

setup_logging()
logger = get_logger("rq3")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
ALPHA_STRENGTH = 0.06
N_STOCKS = 500
START_DATE = "2021-01-01"
# Resolved at run time; the old hardcoded 2025-12-31 aged silently.
END_DATE = resolve_end_date()
INITIAL_CAPITAL = 10_000_000

TOP8_FACTORS = [
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14d", "turnover_20d", "trend_stage", "breakout_proximity",
]

CLUSTERS = {
    "A_ShortReversal": ["rsi_14d", "momentum_1m", "breakout_proximity"],
    "B_MediumTrend":   ["trend_stage", "momentum_3m", "momentum_6m"],
    "C_LongTrend":     ["momentum_12m"],
    "D_Liquidity":     ["turnover_20d"],
}

CLUSTER_NAMES = list(CLUSTERS.keys())
ALL_NAME = "ALL"

# Experiment 3.1: Fixed yearly windows, clipped at the resolved end date.
YEAR_WINDOWS = year_windows(2021, END_DATE)

# Experiment 3.2: Regime thresholds (protocol frozen)
REGIME_THRESHOLDS = {
    "Bull":     0.02,   # benchmark monthly return > +2%
    "Bear":    -0.02,   # benchmark monthly return < -2%
    "Sideways": None,   # -2% <= return <= +2%
}


def load_data():
    """Load synthetic data with protocol-frozen parameters."""
    provider = SyntheticDataProvider(
        n_stocks=N_STOCKS,
        start_date=START_DATE,
        end_date=END_DATE,
        embedded_alpha=True,
        alpha_strength=ALPHA_STRENGTH,
    )
    pipeline = DataPipeline(
        provider=provider,
        start_date=START_DATE,
        end_date=END_DATE,
        exclude_st=True,
        exclude_suspended=True,
    )
    pipeline.run()

    prices = pipeline.get_close()
    returns = pipeline.returns
    benchmark = pipeline.benchmark
    metadata = pipeline.metadata
    turnover = pipeline.get_turnover()
    sector_map = metadata["sector"] if metadata is not None and "sector" in metadata.columns else pd.Series(dtype=object)

    logger.info("Data loaded: %d days, %d assets", len(prices), len(prices.columns))
    return prices, returns, benchmark, sector_map, turnover


def compute_all_factors(prices, returns, sector_map, turnover):
    """Compute all 8 top factors."""
    register_technical()
    register_fundamental()
    registry = get_registry()

    raw_factors = {}
    for name in registry.list_all():
        if name not in TOP8_FACTORS:
            continue
        cls = registry.get(name)
        inst = cls()
        try:
            kwargs = {}
            if turnover is not None:
                kwargs["turnover"] = turnover
            result = inst.run(prices, **kwargs)
            raw_factors[result.name] = result.values
        except Exception as e:
            logger.warning("Factor %s failed: %s", name, e)

    processed = {}
    for name, factor in raw_factors.items():
        processed[name] = process_factor(factor, sector_map=sector_map, market_cap=None)
    logger.info("Computed %d factors", len(processed))
    return processed


def build_all_signals(processed_factors):
    """Build cluster signals and the ALL combined signal."""
    cluster_signals = {}
    for cluster_name, factor_names in CLUSTERS.items():
        available = [f for f in factor_names if f in processed_factors]
        if not available:
            continue
        signal = sum(processed_factors[f] for f in available) / len(available)
        signal = signal.rank(axis=1, pct=True) - 0.5
        cluster_signals[cluster_name] = signal

    # ALL = equal-weight all clusters
    signals = [cluster_signals[c] for c in CLUSTER_NAMES if c in cluster_signals]
    common = sorted(set.intersection(*(set(s.columns) for s in signals)))
    aligned = [s[common] for s in signals]
    all_signal = sum(aligned) / len(aligned)
    all_signal = all_signal.rank(axis=1, pct=True) - 0.5
    cluster_signals[ALL_NAME] = all_signal

    return cluster_signals


def run_backtest(signal, prices, returns, benchmark, sector_map):
    """Run backtest with EqualWeight optimizer, return results."""
    constraints = PortfolioConstraints(
        long_only=True,
        max_weight=0.05,
        max_sector_exposure=0.30,
        max_turnover=0.30,
        lot_size=100,
    )
    cost_model = CostModel(
        commission=0.0003,
        stamp_tax=0.001,
        slippage=0.0005,
        slippage_model="fixed",
    )
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        rebalance_frequency="monthly",
        cost_model=cost_model,
        constraints=constraints,
        optimizer="equal_weight",
        benchmark="equal_weight",
    )
    return engine.run(
        signal=signal, prices=prices, returns=returns,
        benchmark_returns=benchmark, sector_map=sector_map,
        financials=None,
    )


def compute_ic(signal, returns):
    """Compute Rank IC summary."""
    ic = rank_ic(signal, returns)
    return ic_summary(ic)


def classify_regimes(benchmark_returns, trading_days=None):
    """Label each trading day with the regime of its OWN calendar month.

    Month M's compounded benchmark return is computed from the benchmark rows
    inside M and applied to the trading days inside M (contemporaneous, ex-post
    decomposition). No forward-fill into the next month: that made February
    trade on January's label and mislabelled 62.7% of trading days.

    Args:
        benchmark_returns: Series of benchmark returns (pipeline convention).
        trading_days: index of actual trading days to label. Defaults to the
            benchmark's own index. Days whose month has no benchmark data get
            NaN and are excluded by callers.

    Returns: Series(index=trading_days, value=regime_str or NaN)
    """
    if trading_days is None:
        trading_days = (benchmark_returns.index if benchmark_returns is not None
                        else pd.DatetimeIndex([]))
    trading_days = pd.DatetimeIndex(trading_days).sort_values()

    if benchmark_returns is None or len(benchmark_returns) == 0:
        logger.warning("No benchmark returns — regime labels unavailable")
        return pd.Series(np.nan, index=trading_days, name="regime", dtype=object)

    benchmark_returns = benchmark_returns.dropna()
    if benchmark_returns.empty:
        logger.warning("Benchmark series is empty after dropping NaN — regime labels unavailable")
        return pd.Series(np.nan, index=trading_days, name="regime", dtype=object)

    # min_count=1 keeps an all-NaN month as NaN instead of silently compounding
    # it to 0% (which would become a spurious "Sideways" label).
    monthly = benchmark_returns.resample(pd.offsets.MonthEnd()).apply(
        lambda x: (1 + x).prod(min_count=1) - 1
    ).dropna()

    def _label(ret):
        if ret > REGIME_THRESHOLDS["Bull"]:
            return "Bull"
        if ret < REGIME_THRESHOLDS["Bear"]:
            return "Bear"
        return "Sideways"

    labels_by_month = {idx.to_period("M"): _label(ret) for idx, ret in monthly.items()}
    day_labels = [labels_by_month.get(period, np.nan)
                  for period in trading_days.to_period("M")]
    return pd.Series(day_labels, index=trading_days, name="regime", dtype=object)


def regime_shares(daily_regime):
    """Percentage of *labelled* trading days in each regime (sums to 100).

    Unlabelled days (NaN, e.g. before the benchmark starts) are excluded from
    the denominator and must be reported separately.
    """
    labelled = daily_regime.dropna()
    if labelled.empty:
        return pd.Series(dtype=float)
    return labelled.value_counts(normalize=True) * 100.0


def regime_pct_time(regime_dates, daily_regime):
    """``%Time`` for one regime: share of labelled trading days, in percent.

    The denominator is the number of trading days carrying a regime label —
    never calendar days. This is the single expression used by the report so
    the displayed shares cannot drift from :func:`regime_shares`.
    """
    total_labelled = int(daily_regime.notna().sum())
    if total_labelled == 0:
        return 0.0
    return len(regime_dates) / total_labelled * 100.0


def print_yearly_results(results):
    """Print year-by-year stability table."""
    print()
    print("=" * 100)
    print("  EXPERIMENT 3.1 — YEAR-BY-YEAR STABILITY")
    print("=" * 100)
    header = f"  {'Cluster':<18} {'Year':<8} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'MDD':>8}"
    print(header)
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for row in results:
        print(f"  {row['Cluster']:<18} {row['Year']:<8} {row['IC']:>8.4f} {row['ICIR']:>8.4f} {row['Sharpe']:>8.4f} {row['MDD']:>8.4f}")
    print("=" * 100)
    print()


def print_regime_results(results):
    """Print regime decomposition table."""
    print()
    print("=" * 100)
    print("  EXPERIMENT 3.2 — MARKET REGIME DECOMPOSITION")
    print("=" * 100)
    header = f"  {'Cluster':<18} {'Regime':<12} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'%Time':>8}"
    print(header)
    print(f"  {'─'*18} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for row in results:
        print(f"  {row['Cluster']:<18} {row['Regime']:<12} {row['IC']:>8.4f} {row['ICIR']:>8.4f} {row['Sharpe']:>8.4f} {row['%Time']:>7.1f}%")
    print("=" * 100)
    print()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RQ3: Stability + Regime — Alpha Discovery v2")
    logger.info("Protocol frozen. Results only. No interpretation.")
    logger.info("=" * 60)

    # Load data
    logger.info("[1/5] Loading data...")
    prices, returns, benchmark, sector_map, turnover = load_data()
    data_start = str(prices.index[0].date())
    data_end = str(prices.index[-1].date())
    logger.info("Data window: %s → %s (requested end %s)", data_start, data_end, END_DATE)

    # Compute factors
    logger.info("[2/5] Computing factors...")
    processed_factors = compute_all_factors(prices, returns, sector_map, turnover)

    # Build signals
    logger.info("[3/5] Building signals...")
    all_signals = build_all_signals(processed_factors)

    # ==================================================================
    # Experiment 3.1: Year-by-year stability
    # ==================================================================
    logger.info("[4/5] Running Experiment 3.1 — Year-by-year stability...")

    yearly_records = []
    signal_names = CLUSTER_NAMES + [ALL_NAME]

    for year_name, year_start, year_end in YEAR_WINDOWS:
        logger.info("  Year: %s (%s to %s)", year_name, year_start, year_end)
        year_prices = prices[year_start:year_end]
        year_returns = returns[year_start:year_end]

        for sig_name in signal_names:
            if sig_name not in all_signals:
                continue
            sig = all_signals[sig_name]
            year_sig = sig[year_start:year_end]

            # IC
            ic_s = compute_ic(year_sig, year_returns)

            # Backtest
            status, error = "ok", ""
            try:
                bt = run_backtest(year_sig, year_prices, year_returns, benchmark, sector_map)
                summary = bt.get("summary", {})
                sharpe = summary.get("sharpe_ratio", np.nan)
                mdd = summary.get("max_drawdown", np.nan)
            except Exception as e:
                logger.warning("  Backtest failed for %s %s: %s", sig_name, year_name, e)
                # NaN, never 0: a failed calculation must not read as "no edge".
                sharpe = np.nan
                mdd = np.nan
                status, error = "failed", f"{type(e).__name__}: {e}"

            row = {
                "Cluster": sig_name, "Year": year_name,
                "IC": ic_s.get("mean_ic", np.nan), "ICIR": ic_s.get("icir", np.nan),
                "Sharpe": sharpe, "MDD": mdd,
                "Status": status, "Error": error,
            }
            yearly_records.append(row)
            logger.info("    %-18s → IC=%.4f  ICIR=%.4f  Sharpe=%.4f",
                         sig_name, row["IC"], row["ICIR"], row["Sharpe"])

    df_yearly = pd.DataFrame(yearly_records)
    print_yearly_results(yearly_records)

    # ==================================================================
    # Experiment 3.2: Market regime decomposition
    # ==================================================================
    logger.info("[5/5] Running Experiment 3.2 — Regime decomposition...")

    # Classify regimes on the actual trading calendar. Each month's own
    # return labels that month's trading days — no forward-fill into the
    # next month (the old behaviour put January's label on February).
    daily_regime = classify_regimes(benchmark, trading_days=returns.index)
    labelled_days = daily_regime.dropna()
    n_unlabelled = int(daily_regime.isna().sum())
    total_trading_days = len(labelled_days)
    logger.info("  Regime labels: %d/%d trading days labelled; %d unlabelled skipped",
                total_trading_days, len(daily_regime), n_unlabelled)
    if n_unlabelled:
        unlabelled_index = daily_regime.index[daily_regime.isna()]
        logger.info("  Unlabelled (skipped) days span %s → %s (no benchmark data for their month)",
                    unlabelled_index.min().date(), unlabelled_index.max().date())
    if total_trading_days == 0:
        logger.warning("  No regime labels available — Experiment 3.2 will be empty")
    else:
        shares = regime_shares(labelled_days)
        logger.info("  Regime distribution (%% of labelled trading days): %s",
                    {k: f"{v:.1f}%" for k, v in shares.items()})

    regime_records = []

    for regime_name in ["Bull", "Bear", "Sideways"]:
        # Dates carrying this regime label (NaN days are never matched)
        regime_dates = daily_regime.index[daily_regime == regime_name]

        # Filter returns/prices to regime dates
        # IC: compute only on regime dates
        # For backtest: we need to run on continuous data but can segment the P&L

        for sig_name in signal_names:
            if sig_name not in all_signals:
                continue
            sig = all_signals[sig_name]
            returns_aligned = returns

            # IC: subset to regime dates that exist in returns
            common_dates = sorted(set(sig.index) & set(returns_aligned.index) & set(regime_dates))
            if len(common_dates) < 10:
                logger.info("    %-18s %-10s → too few dates (%d), skipping (regime covers %.1f%% of labelled days)",
                            sig_name, regime_name, len(common_dates),
                            regime_pct_time(regime_dates, daily_regime))
                continue

            regime_sig = sig.loc[common_dates]
            regime_ret = returns_aligned.loc[common_dates]
            ic_s = compute_ic(regime_sig, regime_ret)

            # For Sharpe: run full backtest but extract returns only on regime dates
            status, error = "ok", ""
            try:
                bt = run_backtest(sig, prices, returns_aligned, benchmark, sector_map)
                bt_returns = bt.get("daily_returns", pd.Series(dtype=float))
                # Filter to regime dates
                regime_bt = bt_returns[bt_returns.index.isin(regime_dates)]
                if len(regime_bt) < 5:
                    sharpe = np.nan
                else:
                    # Regime subsets are daily returns on a sparse calendar;
                    # infer_periods_per_year would infer ~20-82 from the
                    # filtered span. State the daily basis explicitly.
                    regime_metrics = all_metrics(
                        regime_bt, benchmark, periods_per_year=252
                    )
                    sharpe = regime_metrics.get("sharpe_ratio", np.nan)
                    if not regime_metrics:
                        sharpe = np.nan
            except Exception as e:
                logger.warning("  Regime BT failed for %s %s: %s", sig_name, regime_name, e)
                sharpe = np.nan
                status, error = "failed", f"{type(e).__name__}: {e}"

            # Denominator is actual *labelled* trading days, not calendar days
            # (the old 1826-day denominator made regime shares sum to ~70%).
            pct_time = regime_pct_time(common_dates, daily_regime)
            row = {
                "Cluster": sig_name, "Regime": regime_name,
                "IC": ic_s.get("mean_ic", np.nan), "ICIR": ic_s.get("icir", np.nan),
                "Sharpe": sharpe, "%Time": round(pct_time, 1),
                "Status": status, "Error": error,
            }
            regime_records.append(row)
            logger.info("    %-18s %-10s → IC=%.4f  ICIR=%.4f  Sharpe=%.4f  (%4.1f%% of days)",
                         sig_name, regime_name, row["IC"], row["ICIR"], row["Sharpe"], pct_time)

    df_regime = pd.DataFrame(regime_records)
    print_regime_results(regime_records)

    # Save raw data
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_yearly = stamp_provenance(df_yearly, data_start, data_end)
    df_regime = stamp_provenance(df_regime, data_start, data_end)
    df_yearly.to_csv(output_dir / "rq3_yearly_stability.csv", index=False)
    df_regime.to_csv(output_dir / "rq3_regime_decomposition.csv", index=False)

    # Print raw data
    print()
    print("  === RAW DATA: RQ3_YEARLY ===")
    print(df_yearly.to_string())
    print("  === END RAW DATA ===")
    print()
    print("  === RAW DATA: RQ3_REGIME ===")
    print(df_regime.to_string())
    print("  === END RAW DATA ===")
    print()

    logger.info("Raw results saved to results/rq3_*.csv")
    logger.info("RQ3 complete. All protocol experiments finished.")
