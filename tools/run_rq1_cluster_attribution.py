#!/usr/bin/env python3
"""
RQ1: Cluster Attribution — Alpha Discovery v2

Protocol-frozen experiment: Alpha 来源于哪些 Alpha 簇？

只记录结果，不解释。
"""

import sys
from pathlib import Path

# Ensure package is importable.
# The project directory is named "quant_platform" and acts as the package.
# We need its parent on sys.path so "import quant_platform" resolves.
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
from quant_platform.utils.logging import get_logger, setup_logging
from quant_platform.reporting.dashboard import generate_dashboard

setup_logging()
logger = get_logger("rq1")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
ALPHA_STRENGTH = 0.06       # "normal" strength per protocol
N_STOCKS = 500
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 10_000_000

# Top 8 factors from Ablation #001
TOP8_FACTORS = [
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14d", "turnover_20d", "trend_stage", "breakout_proximity",
]

# Cluster mapping from Ablation #002 (protocol frozen)
CLUSTERS = {
    "A_ShortReversal": ["rsi_14d", "momentum_1m", "breakout_proximity"],
    "B_MediumTrend":   ["trend_stage", "momentum_3m", "momentum_6m"],
    "C_LongTrend":     ["momentum_12m"],
    "D_Liquidity":     ["turnover_20d"],
}

# Experiment combinations
COMBOS = {
    "A":              ["A_ShortReversal"],
    "B":              ["B_MediumTrend"],
    "C":              ["C_LongTrend"],
    "D":              ["D_Liquidity"],
    "A+B":            ["A_ShortReversal", "B_MediumTrend"],
    "A+C":            ["A_ShortReversal", "C_LongTrend"],
    "A+D":            ["A_ShortReversal", "D_Liquidity"],
    "B+C":            ["B_MediumTrend", "C_LongTrend"],
    "B+D":            ["B_MediumTrend", "D_Liquidity"],
    "C+D":            ["C_LongTrend", "D_Liquidity"],
    "A+B+C":          ["A_ShortReversal", "B_MediumTrend", "C_LongTrend"],
    "A+B+D":          ["A_ShortReversal", "B_MediumTrend", "D_Liquidity"],
    "A+C+D":          ["A_ShortReversal", "C_LongTrend", "D_Liquidity"],
    "B+C+D":          ["B_MediumTrend", "C_LongTrend", "D_Liquidity"],
    "ALL":            ["A_ShortReversal", "B_MediumTrend", "C_LongTrend", "D_Liquidity"],
    "TOP8":           None,  # special: all 8 factors individually (baseline from v1)
}

# Evaluation metrics (protocol frozen)
METRICS = ["IC", "ICIR", "Sharpe", "MDD", "Turnover"]


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

    # Extract sector map from metadata DataFrame as pd.Series (dict causes .reindex() bugs)
    sector_map = metadata["sector"] if metadata is not None and "sector" in metadata.columns else pd.Series(dtype=object)

    logger.info("Data loaded: %d days, %d assets", len(prices), len(prices.columns))
    logger.info("Alpha strength: %.2f", ALPHA_STRENGTH)
    return prices, returns, benchmark, metadata, sector_map, turnover


def compute_all_factors(prices, returns, sector_map, turnover):
    """Compute all registered factors, return raw and processed."""
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

    # Process (winsorize → standardize → neutralize)
    processed = {}
    for name, factor in raw_factors.items():
        processed[name] = process_factor(factor, sector_map=sector_map, market_cap=None)

    logger.info("Computed %d factors: %s", len(processed), list(processed.keys()))
    return processed


def build_cluster_signals(processed_factors):
    """Build cluster-level signals by equal-weighting within each cluster."""
    cluster_signals = {}
    for cluster_name, factor_names in CLUSTERS.items():
        available = [f for f in factor_names if f in processed_factors]
        if not available:
            logger.warning("Cluster %s has no available factors", cluster_name)
            continue
        # Equal-weight within cluster (protocol frozen).
    # Stack factor DataFrames and average across factors.
        factor_stack = [processed_factors[f] for f in available]
        signal = sum(factor_stack) / len(factor_stack)
        # Rank-standardize to [-0.5, 0.5]
        signal = signal.rank(axis=1, pct=True) - 0.5
        cluster_signals[cluster_name] = signal
        logger.info("Cluster %s: %s → signal shape %s",
                     cluster_name, available, signal.shape)

    return cluster_signals


def build_experiment_signal(cluster_signals, combo_clusters):
    """Build the combined signal for an experiment.

    Equal-weights across clusters, then rank-standardize to [-0.5, 0.5].
    For TOP8: equal-weight all 8 individual factors.
    """
    if combo_clusters is None:
        # TOP8: all 8 individual factors, equal-weight
        signals = [processed_factors[f] for f in TOP8_FACTORS if f in processed_factors]
    else:
        # Cluster combinations
        signals = [cluster_signals[c] for c in combo_clusters if c in cluster_signals]

    if not signals:
        return None

    # Ensure all signals are aligned on same date/asset index
    # Use the intersection of all column sets
    common_assets = set.intersection(*(set(s.columns) for s in signals)) if len(signals) > 1 else set(signals[0].columns)
    common_assets = sorted(common_assets)
    aligned = [s[common_assets] for s in signals]

    # Equal-weight combination
    combined = sum(aligned) / len(aligned)

    # Cross-sectional rank to [-0.5, 0.5]
    signal = combined.rank(axis=1, pct=True) - 0.5
    return signal


def run_backtest(signal, prices, returns, benchmark, sector_map):
    """Run backtest with EqualWeight optimizer (protocol frozen)."""
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
    results = engine.run(
        signal=signal, prices=prices, returns=returns,
        benchmark_returns=benchmark, sector_map=sector_map,
        financials=None,
    )
    return results


def compute_signal_ic(signal, returns):
    """Compute Rank IC for a signal vs next-period returns."""
    # Signal is aligned with prices index; returns for t+1 already shifted in pipeline
    ic = rank_ic(signal, returns)
    summary = ic_summary(ic)
    return summary


def extract_metrics(bt_results, signal_ic):
    """Extract protocol metrics from backtest results and IC."""
    summary = bt_results.get("summary", {})
    metrics = {
        "IC":        signal_ic.get("mean_ic", 0),
        "ICIR":      signal_ic.get("icir", 0),
        "Sharpe":    summary.get("sharpe_ratio", 0),
        "MDD":       summary.get("max_drawdown", 0),
        "Turnover":  summary.get("avg_turnover", summary.get("turnover", 0)),
    }
    return metrics


def compute_correlation_matrix(cluster_signals):
    """Compute pairwise correlation between cluster signals (Experiment 1.3)."""
    # Align all to the same timestamp
    aligned = {}
    for name, sig in cluster_signals.items():
        # Take the mean across assets per day to get a single time series per cluster
        aligned[name] = sig.mean(axis=1)

    df = pd.DataFrame(aligned)
    pearson_corr = df.corr(method="pearson")
    spearman_corr = df.corr(method="spearman")

    return pearson_corr, spearman_corr


def print_results_table(results_list):
    """Print results as a formatted table (no interpretation)."""
    print()
    print("=" * 90)
    print("  RQ1 — CLUSTER ATTRIBUTION RESULTS")
    print("=" * 90)
    print(f"  Data: {START_DATE} → {END_DATE}, {N_STOCKS} stocks, alpha_strength={ALPHA_STRENGTH}")
    print(f"  Backtest: monthly rebalance, equal_weight, long-only, 5% max weight")
    print()
    print(f"  {'Experiment':<18} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'MDD':>8} {'Turnover':>10}")
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    for row in results_list:
        print(f"  {row['Experiment']:<18} {row['IC']:>8.4f} {row['ICIR']:>8.4f} {row['Sharpe']:>8.4f} {row['MDD']:>8.4f} {row['Turnover']:>10.4f}")

    print("=" * 90)
    print()


def print_correlation_matrix(pearson, spearman):
    """Print correlation matrix (Experiment 1.3)."""
    print()
    print("=" * 70)
    print("  EXPERIMENT 1.3 — CLUSTER CORRELATION MATRIX (PEARSON)")
    print("=" * 70)
    print(pearson.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print("=" * 70)
    print("  EXPERIMENT 1.3 — CLUSTER CORRELATION MATRIX (SPEARMAN)")
    print("=" * 70)
    print(spearman.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)
    print()


def print_raw_section(label, data):
    """Print raw data section for later analysis."""
    print()
    print(f"  === RAW DATA: {label} ===")
    print(data)
    print(f"  === END RAW DATA: {label} ===")
    print()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RQ1: Cluster Attribution — Alpha Discovery v2")
    logger.info("Protocol frozen. Results only. No interpretation.")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("[1/4] Loading data...")
    prices, returns, benchmark, metadata, sector_map, turnover = load_data()

    # Step 2: Compute all 8 factors
    logger.info("[2/4] Computing factors...")
    processed_factors = compute_all_factors(prices, returns, sector_map, turnover)

    # Step 3: Build cluster signals
    logger.info("[3/4] Building cluster signals...")
    cluster_signals = build_cluster_signals(processed_factors)

    # Experiment 1.3: Correlation matrix (doesn't depend on backtest)
    logger.info("Computing cluster correlation matrix...")
    pearson_corr, spearman_corr = compute_correlation_matrix(cluster_signals)
    print_correlation_matrix(pearson_corr, spearman_corr)

    # Step 4: Run all experiments
    logger.info("[4/4] Running %d experiments...", len(COMBOS))

    results_list = []
    raw_records = []  # Save all raw data for later analysis

    for exp_name, combo_clusters in COMBOS.items():
        logger.info("  Experiment: %s", exp_name)

        # Build signal
        signal = build_experiment_signal(cluster_signals, combo_clusters)
        if signal is None:
            logger.warning("  Skipping %s: no signal", exp_name)
            continue

        # Compute IC
        signal_ic = compute_signal_ic(signal, returns)

        # Run backtest
        try:
            bt_results = run_backtest(signal, prices, returns, benchmark, sector_map)
            metrics = extract_metrics(bt_results, signal_ic)
        except Exception as e:
            logger.error("  Backtest failed for %s: %s", exp_name, e)
            metrics = {"IC": signal_ic.get("mean_ic", 0), "ICIR": signal_ic.get("icir", 0),
                       "Sharpe": 0, "MDD": 0, "Turnover": 0}

        metrics["Experiment"] = exp_name
        results_list.append(metrics)
        raw_records.append(metrics)

        logger.info("  → IC=%.4f  ICIR=%.4f  Sharpe=%.4f  MDD=%.4f",
                     metrics["IC"], metrics["ICIR"], metrics["Sharpe"], metrics["MDD"])

    # Print results
    print_results_table(results_list)

    # Print raw data for archival
    df_raw = pd.DataFrame(raw_records)
    print_raw_section("RQ1_RAW", df_raw)

    # Save to CSV
    output_path = Path("results/rq1_cluster_attribution.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    logger.info("Raw results saved to: %s", output_path)

    # Save correlation matrices
    pearson_path = Path("results/rq1_cluster_correlation_pearson.csv")
    spearman_path = Path("results/rq1_cluster_correlation_spearman.csv")
    pearson_corr.to_csv(pearson_path)
    spearman_corr.to_csv(spearman_path)
    logger.info("Correlation matrices saved.")

    logger.info("RQ1 complete. %d experiments run.", len(results_list))
    logger.info("End of RQ1. Awaiting RQ2.")
