#!/usr/bin/env python3
"""
RQ3: Stability + Regime — Alpha Discovery v2

Protocol-frozen experiment: Alpha 什么时候有效？

Experiment 3.1 — Year-by-year stability (fixed windows)
Experiment 3.2 — Market regime decomposition (Bull/Bear/Sideways)

只记录结果，不解释。
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root.parent))

import pandas as pd
import numpy as np

from quant_platform.factors.evaluation import rank_ic, ic_summary
from quant_platform.factors.technical import register_all as register_technical
from quant_platform.factors.fundamental import register_all as register_fundamental
from quant_platform.factors.registry import get_registry
from quant_platform.factors.processing import process_factor
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.cost_model import CostModel
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("rq3")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
ALPHA_STRENGTH = 0.06
N_STOCKS = 500
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
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

# Experiment 3.1: Fixed yearly windows
YEAR_WINDOWS = [
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
]

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


def classify_regimes(benchmark_returns):
    """Classify each month into Bull/Bear/Sideways based on benchmark returns.

    Returns: Series(index=date, value=regime_str)
    """
    if benchmark_returns is None or len(benchmark_returns) == 0:
        return pd.Series("Sideways", index=pd.date_range(START_DATE, END_DATE, freq="ME"))

    monthly = benchmark_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    regimes = {}
    for date, ret in monthly.items():
        if ret > REGIME_THRESHOLDS["Bull"]:
            regimes[date] = "Bull"
        elif ret < REGIME_THRESHOLDS["Bear"]:
            regimes[date] = "Bear"
        else:
            regimes[date] = "Sideways"

    # Forward-fill to daily
    regime_series = pd.Series(regimes, name="regime")
    regime_series.index = pd.DatetimeIndex(regime_series.index)
    daily_regime = regime_series.reindex(
        pd.date_range(START_DATE, END_DATE, freq="D"),
        method="ffill"
    )
    return daily_regime


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
            try:
                bt = run_backtest(year_sig, year_prices, year_returns, benchmark, sector_map)
                summary = bt.get("summary", {})
                sharpe = summary.get("sharpe_ratio", 0)
                mdd = summary.get("max_drawdown", 0)
            except Exception as e:
                logger.warning("  Backtest failed for %s %s: %s", sig_name, year_name, e)
                sharpe = 0
                mdd = 0

            row = {
                "Cluster": sig_name, "Year": year_name,
                "IC": ic_s.get("mean_ic", 0), "ICIR": ic_s.get("icir", 0),
                "Sharpe": sharpe, "MDD": mdd,
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

    # Classify regimes
    daily_regime = classify_regimes(benchmark)
    regime_counts = daily_regime.value_counts()
    total_days = len(daily_regime)
    logger.info("  Regime distribution: %s", {k: f"{v/total_days*100:.1f}%" for k, v in regime_counts.items()})

    regime_records = []

    for regime_name in ["Bull", "Bear", "Sideways"]:
        # Get dates for this regime
        regime_dates = daily_regime[daily_regime == regime_name].index
        regime_dates_str = regime_dates.strftime("%Y-%m-%d")

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
                logger.info("    %-18s %-10s → too few dates (%d), skipping",
                            sig_name, regime_name, len(common_dates))
                continue

            regime_sig = sig.loc[common_dates]
            regime_ret = returns_aligned.loc[common_dates]
            ic_s = compute_ic(regime_sig, regime_ret)

            # For Sharpe: run full backtest but extract returns only on regime dates
            try:
                bt = run_backtest(sig, prices, returns_aligned, benchmark, sector_map)
                bt_returns = bt.get("daily_returns", pd.Series(dtype=float))
                # Filter to regime dates
                regime_bt = bt_returns[bt_returns.index.isin(regime_dates)]
                if len(regime_bt) < 5:
                    sharpe = 0
                else:
                    regime_metrics = all_metrics(regime_bt, benchmark)
                    sharpe = regime_metrics.get("sharpe_ratio", 0)
            except Exception as e:
                logger.warning("  Regime BT failed for %s %s: %s", sig_name, regime_name, e)
                sharpe = 0

            pct_time = len(common_dates) / total_days * 100
            row = {
                "Cluster": sig_name, "Regime": regime_name,
                "IC": ic_s.get("mean_ic", 0), "ICIR": ic_s.get("icir", 0),
                "Sharpe": sharpe, "%Time": round(pct_time, 1),
            }
            regime_records.append(row)
            logger.info("    %-18s %-10s → IC=%.4f  ICIR=%.4f  Sharpe=%.4f  (%4.1f%% of days)",
                         sig_name, regime_name, row["IC"], row["ICIR"], row["Sharpe"], pct_time)

    df_regime = pd.DataFrame(regime_records)
    print_regime_results(regime_records)

    # Save raw data
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

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
