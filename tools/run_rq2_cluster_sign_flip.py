#!/usr/bin/env python3
"""
RQ2: Cluster Sign Flip — Alpha Discovery v2

Protocol-frozen experiment: 哪些 Alpha 方向错了？

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
from quant_platform.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("rq2")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters (identical to RQ1)
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

# Flip experiments
# Each entry: (name, {cluster: +1 or -1})
# +1 = original direction, -1 = flipped
FLIP_EXPERIMENTS = [
    ("Baseline",  {"A_ShortReversal": +1, "B_MediumTrend": +1, "C_LongTrend": +1, "D_Liquidity": +1}),
    ("Flip_A",    {"A_ShortReversal": -1, "B_MediumTrend": +1, "C_LongTrend": +1, "D_Liquidity": +1}),
    ("Flip_B",    {"A_ShortReversal": +1, "B_MediumTrend": -1, "C_LongTrend": +1, "D_Liquidity": +1}),
    ("Flip_C",    {"A_ShortReversal": +1, "B_MediumTrend": +1, "C_LongTrend": -1, "D_Liquidity": +1}),
    ("Flip_D",    {"A_ShortReversal": +1, "B_MediumTrend": +1, "C_LongTrend": +1, "D_Liquidity": -1}),
]


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

    logger.info("Computed %d factors: %s", len(processed), list(processed.keys()))
    return processed


def build_cluster_signals(processed_factors):
    """Build cluster-level signals by equal-weighting within each cluster."""
    cluster_signals = {}
    for cluster_name, factor_names in CLUSTERS.items():
        available = [f for f in factor_names if f in processed_factors]
        if not available:
            continue
        factor_stack = [processed_factors[f] for f in available]
        signal = sum(factor_stack) / len(factor_stack)
        signal = signal.rank(axis=1, pct=True) - 0.5
        cluster_signals[cluster_name] = signal
        logger.info("Cluster %s: shape %s", cluster_name, signal.shape)
    return cluster_signals


def build_flip_signal(cluster_signals, flip_map):
    """Build combined signal with flipped directions per experiment.

    Each cluster signal is multiplied by +1 (original) or -1 (flipped),
    then all are equal-weight combined and rank-standardized.
    """
    signals = []
    for cluster_name, direction in flip_map.items():
        if cluster_name not in cluster_signals:
            continue
        sig = cluster_signals[cluster_name] * direction
        signals.append(sig)

    if not signals:
        return None

    common_assets = set.intersection(*(set(s.columns) for s in signals)) if len(signals) > 1 else set(signals[0].columns)
    common_assets = sorted(common_assets)
    aligned = [s[common_assets] for s in signals]

    combined = sum(aligned) / len(aligned)
    signal = combined.rank(axis=1, pct=True) - 0.5
    return signal


def run_backtest(signal, prices, returns, benchmark, sector_map):
    """Run backtest with EqualWeight optimizer."""
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


def print_results_table(results_list):
    """Print results as a formatted table. Δ columns vs Baseline."""
    # Find baseline
    baseline = {r["Experiment"]: r for r in results_list}
    base = baseline.get("Baseline", {})

    print()
    print("=" * 120)
    print("  RQ2 — CLUSTER SIGN FLIP RESULTS")
    print("=" * 120)
    print(f"  Data: {START_DATE} → {END_DATE}, {N_STOCKS} stocks, alpha_strength={ALPHA_STRENGTH}")
    print()
    print(f"  {'Experiment':<12} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'MDD':>8} {'ΔIC':>8} {'ΔICIR':>8} {'ΔSharpe':>8} {'Flipped':>12}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")

    for row in results_list:
        exp = row["Experiment"]
        dic = row["IC"] - base.get("IC", 0) if base else 0
        dicir = row["ICIR"] - base.get("ICIR", 0) if base else 0
        dsharpe = row["Sharpe"] - base.get("Sharpe", 0) if base else 0
        # Which cluster is flipped in this experiment?
        flipped = ""
        if exp == "Baseline":
            flipped = "none"
        else:
            parts = exp.split("_")
            if len(parts) >= 2:
                flipped = {"A": "A_ShortRev", "B": "B_MidTrend", "C": "C_LongTrend", "D": "D_Liquidity"}.get(parts[1], parts[1])
        print(f"  {exp:<12} {row['IC']:>8.4f} {row['ICIR']:>8.4f} {row['Sharpe']:>8.4f} {row['MDD']:>8.4f} {dic:>+8.4f} {dicir:>+8.4f} {dsharpe:>+8.4f} {flipped:>12}")

    print("=" * 120)
    print()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RQ2: Cluster Sign Flip — Alpha Discovery v2")
    logger.info("Protocol frozen. Results only. No interpretation.")
    logger.info("=" * 60)

    # Load data
    logger.info("[1/4] Loading data...")
    prices, returns, benchmark, sector_map, turnover = load_data()

    # Compute factors
    logger.info("[2/4] Computing factors...")
    processed_factors = compute_all_factors(prices, returns, sector_map, turnover)

    # Build cluster signals
    logger.info("[3/4] Building cluster signals...")
    cluster_signals = build_cluster_signals(processed_factors)

    # Run 5 flip experiments
    logger.info("[4/4] Running %d flip experiments...", len(FLIP_EXPERIMENTS))

    results_list = []
    raw_records = []

    for exp_name, flip_map in FLIP_EXPERIMENTS:
        logger.info("  Experiment: %s — %s", exp_name, flip_map)

        signal = build_flip_signal(cluster_signals, flip_map)
        if signal is None:
            logger.warning("  Skipping: no signal")
            continue

        signal_ic = compute_signal_ic(signal, returns)

        try:
            bt_results = run_backtest(signal, prices, returns, benchmark, sector_map)
            metrics = extract_metrics(bt_results, signal_ic)
        except Exception as e:
            logger.error("  Backtest failed: %s", e)
            metrics = {"IC": signal_ic.get("mean_ic", 0), "ICIR": signal_ic.get("icir", 0),
                       "Sharpe": 0, "MDD": 0, "Turnover": 0}

        metrics["Experiment"] = exp_name
        results_list.append(metrics)
        raw_records.append(metrics)

        logger.info("  → IC=%.4f  ICIR=%.4f  Sharpe=%.4f  MDD=%.4f",
                     metrics["IC"], metrics["ICIR"], metrics["Sharpe"], metrics["MDD"])

    # Print results
    print_results_table(results_list)

    # Save raw data
    df_raw = pd.DataFrame(raw_records)
    print()
    print("  === RAW DATA: RQ2_RAW ===")
    print(df_raw.to_string())
    print("  === END RAW DATA: RQ2_RAW ===")
    print()

    output_path = Path("results/rq2_cluster_sign_flip.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    logger.info("Raw results saved to: %s", output_path)
    logger.info("RQ2 complete. Awaiting RQ3.")
