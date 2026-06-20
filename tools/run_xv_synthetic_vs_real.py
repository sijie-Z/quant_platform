#!/usr/bin/env python3
"""
Alpha Discovery v2 Cross-Validation — Real A-Share Market

Protocol-frozen reproduction of RQ1-RQ3 on real A-share data (Baostock).
Same experiments, same metrics, same evaluation criteria.

Purpose: Distinguish between "market规律" and "synthetic generator规律"
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
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.cost_model import CostModel
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("xvld")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
N_STOCKS = 500
SYNTHETIC_START = "2021-01-01"
SYNTHETIC_END = "2025-12-31"
REAL_START = "2018-01-01"
REAL_END = "2025-12-31"
ALPHA_STRENGTH = 0.06
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

COMBOS = {
    "A": ["A_ShortReversal"], "B": ["B_MediumTrend"], "C": ["C_LongTrend"], "D": ["D_Liquidity"],
    "A+B": ["A_ShortReversal", "B_MediumTrend"],
    "A+C": ["A_ShortReversal", "C_LongTrend"],
    "A+D": ["A_ShortReversal", "D_Liquidity"],
    "B+C": ["B_MediumTrend", "C_LongTrend"],
    "B+D": ["B_MediumTrend", "D_Liquidity"],
    "C+D": ["C_LongTrend", "D_Liquidity"],
    "A+B+C": ["A_ShortReversal", "B_MediumTrend", "C_LongTrend"],
    "A+B+D": ["A_ShortReversal", "B_MediumTrend", "D_Liquidity"],
    "A+C+D": ["A_ShortReversal", "C_LongTrend", "D_Liquidity"],
    "B+C+D": ["B_MediumTrend", "C_LongTrend", "D_Liquidity"],
    "ALL": ["A_ShortReversal", "B_MediumTrend", "C_LongTrend", "D_Liquidity"],
    "TOP8": None,
}

FLIP_EXPERIMENTS = [
    ("Baseline",  {"A_ShortReversal": +1, "B_MediumTrend": +1, "C_LongTrend": +1, "D_Liquidity": +1}),
    ("Flip_A",    {"A_ShortReversal": -1, "B_MediumTrend": +1, "C_LongTrend": +1, "D_Liquidity": +1}),
    ("Flip_B",    {"A_ShortReversal": +1, "B_MediumTrend": -1, "C_LongTrend": +1, "D_Liquidity": +1}),
    ("Flip_C",    {"A_ShortReversal": +1, "B_MediumTrend": +1, "C_LongTrend": -1, "D_Liquidity": +1}),
    ("Flip_D",    {"A_ShortReversal": +1, "B_MediumTrend": +1, "C_LongTrend": +1, "D_Liquidity": -1}),
]

YEAR_WINDOWS = [
    ("2018", "2018-01-01", "2018-12-31"),
    ("2019", "2019-01-01", "2019-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
]

REGIME_THRESHOLDS = {"Bull": 0.02, "Bear": -0.02, "Sideways": None}


# ======================================================================
# Data loading
# ======================================================================

def load_synthetic():
    provider = SyntheticDataProvider(
        n_stocks=N_STOCKS, start_date=SYNTHETIC_START, end_date=SYNTHETIC_END,
        embedded_alpha=True, alpha_strength=ALPHA_STRENGTH,
    )
    return _run_pipeline(provider, SYNTHETIC_START, SYNTHETIC_END)


def load_real():
    provider = BaostockDataProvider(cache_enabled=True)
    return _run_pipeline(provider, REAL_START, REAL_END)


def _run_pipeline(provider, start, end):
    pipeline = DataPipeline(
        provider=provider, start_date=start, end_date=end,
        exclude_st=True, exclude_suspended=True,
    )
    pipeline.run()
    prices = pipeline.get_close()
    returns = pipeline.returns
    benchmark = pipeline.benchmark
    metadata = pipeline.metadata
    turnover = pipeline.get_turnover()

    # Handle sector_map: use None when all sectors are identical or missing
    # (neutralize with one sector crashes LinearRegression)
    sector_map = None
    if metadata is not None and "sector" in metadata.columns:
        unique_sectors = metadata["sector"].nunique()
        if unique_sectors > 1:
            sector_map = metadata["sector"]
            logger.info("Sector map: %d unique sectors", unique_sectors)

    logger.info("Data: %d days, %d assets", len(prices), len(prices.columns))
    return prices, returns, benchmark, sector_map, turnover


# ======================================================================
# Factors + Signals
# ======================================================================

def compute_factors(prices, sector_map, turnover):
    register_technical()
    register_fundamental()
    registry = get_registry()
    raw = {}
    for name in registry.list_all():
        if name not in TOP8_FACTORS:
            continue
        cls = registry.get(name)
        inst = cls()
        try:
            kwargs = {}
            if turnover is not None:
                kwargs["turnover"] = turnover
            raw[name] = inst.run(prices, **kwargs).values
        except Exception as e:
            logger.warning("Factor %s failed: %s", name, e)
    processed = {}
    for name, factor in raw.items():
        processed[name] = process_factor(factor, sector_map=sector_map, market_cap=None)
    logger.info("Factors: %d", len(processed))
    return processed


def build_signals(pf):
    cs = {}
    for cn, fns in CLUSTERS.items():
        av = [f for f in fns if f in pf]
        if not av:
            continue
        sig = sum(pf[f] for f in av) / len(av)
        cs[cn] = sig.rank(axis=1, pct=True) - 0.5

    sigs = [cs[c] for c in CLUSTER_NAMES if c in cs]
    common = sorted(set.intersection(*(set(s.columns) for s in sigs)))
    aligned = [s[common] for s in sigs]
    all_sig = sum(aligned) / len(aligned)
    cs[ALL_NAME] = all_sig.rank(axis=1, pct=True) - 0.5
    return cs


# ======================================================================
# Backtest
# ======================================================================

def run_bt(signal, prices, returns, benchmark, sector_map):
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
    return engine.run(signal=signal, prices=prices, returns=returns,
                      benchmark_returns=benchmark, sector_map=sector_map, financials=None)


def compute_ic(signal, returns):
    ic = rank_ic(signal, returns)
    return ic_summary(ic)


def classify_regimes(benchmark_returns):
    if benchmark_returns is None or len(benchmark_returns) == 0:
        return pd.Series("Sideways", index=pd.date_range(REAL_START, REAL_END, freq="D"))
    monthly = benchmark_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    regimes = {}
    for date, ret in monthly.items():
        if ret > REGIME_THRESHOLDS["Bull"]:
            regimes[date] = "Bull"
        elif ret < REGIME_THRESHOLDS["Bear"]:
            regimes[date] = "Bear"
        else:
            regimes[date] = "Sideways"
    regime_series = pd.Series(regimes, name="regime")
    regime_series.index = pd.DatetimeIndex(regime_series.index)
    daily = regime_series.reindex(
        pd.date_range(REAL_START, REAL_END, freq="D"), method="ffill"
    )
    return daily


# ======================================================================
# Experiment Runners
# ======================================================================

def run_rq1(cs, prices, returns, benchmark, sector_map, label=""):
    results = []
    for exp_name, combo in COMBOS.items():
        if combo is None:  # TOP8
            sigs = [cs[c] for c in CLUSTER_NAMES if c in cs]
        else:
            sigs = [cs[c] for c in combo if c in cs]
        if not sigs:
            continue
        common = sorted(set.intersection(*(set(s.columns) for s in sigs)))
        comb = sum(s[common] for s in sigs) / len(sigs)
        signal = comb.rank(axis=1, pct=True) - 0.5

        ic_s = compute_ic(signal, returns)
        try:
            bt = run_bt(signal, prices, returns, benchmark, sector_map)
            s = bt.get("summary", {})
            results.append({"Experiment": exp_name, "IC": ic_s.get("mean_ic", 0),
                           "ICIR": ic_s.get("icir", 0), "Sharpe": s.get("sharpe_ratio", 0),
                           "MDD": s.get("max_drawdown", 0)})
        except Exception as e:
            results.append({"Experiment": exp_name, "IC": ic_s.get("mean_ic", 0),
                           "ICIR": ic_s.get("icir", 0), "Sharpe": 0, "MDD": 0})
        logger.info("  %s %-12s IC=%.4f Sharpe=%.2f", label, exp_name, results[-1]["IC"], results[-1]["Sharpe"])
    return pd.DataFrame(results)


def run_rq2(cs, prices, returns, benchmark, sector_map, label=""):
    results = []
    for exp_name, flip_map in FLIP_EXPERIMENTS:
        sigs = []
        for cn, direction in flip_map.items():
            if cn not in cs:
                continue
            sigs.append(cs[cn] * direction)
        common = sorted(set.intersection(*(set(s.columns) for s in sigs)))
        comb = sum(s[common] for s in sigs) / len(sigs)
        signal = comb.rank(axis=1, pct=True) - 0.5

        ic_s = compute_ic(signal, returns)
        try:
            bt = run_bt(signal, prices, returns, benchmark, sector_map)
            s = bt.get("summary", {})
            results.append({"Experiment": exp_name, "IC": ic_s.get("mean_ic", 0),
                           "ICIR": ic_s.get("icir", 0), "Sharpe": s.get("sharpe_ratio", 0),
                           "MDD": s.get("max_drawdown", 0)})
        except Exception as e:
            results.append({"Experiment": exp_name, "IC": ic_s.get("mean_ic", 0),
                           "ICIR": ic_s.get("icir", 0), "Sharpe": 0, "MDD": 0})
        logger.info("  %s %-12s IC=%.4f Sharpe=%.2f", label, exp_name, results[-1]["IC"], results[-1]["Sharpe"])
    return pd.DataFrame(results)


def run_rq3(cs, prices, returns, benchmark, sector_map, label=""):
    yearly = []
    for year_name, ys, ye in YEAR_WINDOWS:
        yp = prices[ys:ye]
        yr = returns[ys:ye]
        for sn in CLUSTER_NAMES + [ALL_NAME]:
            if sn not in cs:
                continue
            sig = cs[sn][ys:ye]
            ic_s = compute_ic(sig, yr)
            try:
                bt = run_bt(sig, yp, yr, returns, sector_map)
                s = bt.get("summary", {})
                yearly.append({"Cluster": sn, "Year": year_name, "IC": ic_s.get("mean_ic", 0),
                              "ICIR": ic_s.get("icir", 0), "Sharpe": s.get("sharpe_ratio", 0),
                              "MDD": s.get("max_drawdown", 0)})
            except Exception:
                pass
        logger.info("  %s Year %s done", label, year_name)

    regime_records = []
    daily_regime = classify_regimes(benchmark) if label == "REAL" else \
        pd.Series("Sideways", index=pd.date_range(SYNTHETIC_START, SYNTHETIC_END, freq="D"))

    total_days = len(daily_regime)
    for regime_name in ["Bull", "Bear", "Sideways"]:
        regime_dates = daily_regime[daily_regime == regime_name].index
        for sn in CLUSTER_NAMES + [ALL_NAME]:
            if sn not in cs:
                continue
            common_dates = sorted(set(cs[sn].index) & set(returns.index) & set(regime_dates))
            if len(common_dates) < 10:
                continue
            rsig = cs[sn].loc[common_dates]
            rret = returns.loc[common_dates]
            ic_s = compute_ic(rsig, rret)
            try:
                bt = run_bt(cs[sn], prices, returns, benchmark, sector_map)
                btr = bt.get("daily_returns", pd.Series(dtype=float))
                regime_bt = btr[btr.index.isin(regime_dates)]
                if len(regime_bt) >= 5:
                    rm = all_metrics(regime_bt, returns)
                    sharpe = rm.get("sharpe_ratio", 0)
                else:
                    sharpe = 0
            except Exception:
                sharpe = 0
            pct = len(common_dates) / total_days * 100
            regime_records.append({"Cluster": sn, "Regime": regime_name, "IC": ic_s.get("mean_ic", 0),
                                  "ICIR": ic_s.get("icir", 0), "Sharpe": sharpe, "%Time": round(pct, 1)})
        logger.info("  %s Regime %s done", label, regime_name)

    return pd.DataFrame(yearly), pd.DataFrame(regime_records)


# ======================================================================
# Report
# ======================================================================

def print_comparison(syn_r1, real_r1, syn_r2, real_r2, syn_r3y, real_r3y, syn_r3r, real_r3r):
    print()
    print("=" * 100)
    print("  ALPHA DISCOVERY v2 — CROSS-VALIDATION: SYNTHETIC vs REAL A-SHARE")
    print("=" * 100)

    # --- RQ1 Comparison ---
    print()
    print("─── RQ1: Cluster Attribution ───")
    print()
    merge = syn_r1.merge(real_r1, on="Experiment", suffixes=("_Syn", "_Real"))
    for _, r in merge.iterrows():
        ic_s = r["IC_Syn"]
        ic_r = r["IC_Real"]
        sh_s = r["Sharpe_Syn"]
        sh_r = r["Sharpe_Real"]
        ic_s_s = f"{ic_s:.4f}" if not (isinstance(ic_s, float) and np.isnan(ic_s)) else " nan"
        ic_r_s = f"{ic_r:.4f}" if not (isinstance(ic_r, float) and np.isnan(ic_r)) else " nan"
        print(f"  {r['Experiment']:<12} "
              f"Syn: IC={ic_s_s} Sharpe={sh_s:.2f} | "
              f"Real: IC={ic_r_s} Sharpe={sh_r:.2f}")

    # --- RQ2 Comparison ---
    print()
    print("─── RQ2: Sign Flip ───")
    print()
    base_syn = syn_r2[syn_r2["Experiment"] == "Baseline"].iloc[0]
    base_real = real_r2[real_r2["Experiment"] == "Baseline"].iloc[0]
    print(f"  Baseline (Syn):  IC={base_syn['IC']:.4f} Sharpe={base_syn['Sharpe']:.2f}")
    print(f"  Baseline (Real): IC={base_real['IC']:.4f} Sharpe={base_real['Sharpe']:.2f}")
    print()
    for _, rs in syn_r2.iterrows():
        if rs["Experiment"] == "Baseline": continue
        rr = real_r2[real_r2["Experiment"] == rs["Experiment"]].iloc[0]
        d_syn = rs["Sharpe"] - base_syn["Sharpe"]
        d_real = rr["Sharpe"] - base_real["Sharpe"]
        print(f"  {rs['Experiment']:<12} "
              f"Syn ΔSharpe={d_syn:+.2f} | "
              f"Real ΔSharpe={d_real:+.2f} | "
              f"DirMatch={d_syn < 0 and d_real < 0 or d_syn > 0 and d_real > 0}")

    # --- RQ3 Yearly ---
    print()
    print("─── RQ3: Year-by-Year (ALL Cluster) ───")
    print()
    syn_all = syn_r3y[syn_r3y["Cluster"] == "ALL"][["Year", "Sharpe", "IC"]].copy()
    real_all = real_r3y[real_r3y["Cluster"] == "ALL"][["Year", "Sharpe", "IC"]].copy()
    syn_all.columns = ["Year", "Sharpe_Syn", "IC_Syn"]
    real_all.columns = ["Year", "Sharpe_Real", "IC_Real"]
    merge_y = syn_all.merge(real_all, on="Year", how="outer")
    for _, r in merge_y.iterrows():
        ic_s = r.get("IC_Syn", np.nan)
        ic_r = r.get("IC_Real", np.nan)
        sh_s = r.get("Sharpe_Syn", np.nan)
        sh_r = r.get("Sharpe_Real", np.nan)
        ic_s_str = f"{ic_s:.4f}" if not (isinstance(ic_s, float) and np.isnan(ic_s)) else " nan"
        ic_r_str = f"{ic_r:.4f}" if not (isinstance(ic_r, float) and np.isnan(ic_r)) else " nan"
        sh_s_str = f"{sh_s:.2f}" if not (isinstance(sh_s, float) and np.isnan(sh_s)) else " nan"
        sh_r_str = f"{sh_r:.2f}" if not (isinstance(sh_r, float) and np.isnan(sh_r)) else " nan"
        print(f"  {r['Year']:<6} Syn: IC={ic_s_str} Sharpe={sh_s_str} | "
              f"Real: IC={ic_r_str} Sharpe={sh_r_str}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Alpha Discovery v2 Cross-Validation")
    logger.info("Synthetic vs Real A-Share Market")
    logger.info("=" * 60)

    # === Synthetic ===
    logger.info("[1/6] Synthetic data...")
    sp, sr, sb, ss, st = load_synthetic()
    pf_syn = compute_factors(sp, ss, st)
    cs_syn = build_signals(pf_syn)

    logger.info("[2/6] Synthetic RQ1...")
    syn_r1 = run_rq1(cs_syn, sp, sr, sb, ss, "SYN")

    logger.info("[3/6] Synthetic RQ2...")
    syn_r2 = run_rq2(cs_syn, sp, sr, sb, ss, "SYN")

    logger.info("[4/6] Synthetic RQ3...")
    syn_r3y, syn_r3r = run_rq3(cs_syn, sp, sr, sb, ss, "SYN")

    # === Real A-Share ===
    logger.info("[5/6] Real A-share data...")
    rp, rr, rb, rs, rt = load_real()
    pf_real = compute_factors(rp, rs, rt)
    cs_real = build_signals(pf_real)

    logger.info("[6/6] Real A-share RQ1+RQ2+RQ3...")
    real_r1 = run_rq1(cs_real, rp, rr, rb, rs, "REAL")
    real_r2 = run_rq2(cs_real, rp, rr, rb, rs, "REAL")
    real_r3y, real_r3r = run_rq3(cs_real, rp, rr, rb, rs, "REAL")

    # === Comparison Report ===
    print_comparison(syn_r1, real_r1, syn_r2, real_r2, syn_r3y, real_r3y, syn_r3r, real_r3r)

    # Save all
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    syn_r1.to_csv(out / "xv_syn_rq1.csv", index=False)
    syn_r2.to_csv(out / "xv_syn_rq2.csv", index=False)
    syn_r3y.to_csv(out / "xv_syn_rq3_yearly.csv", index=False)
    syn_r3r.to_csv(out / "xv_syn_rq3_regime.csv", index=False)
    real_r1.to_csv(out / "xv_real_rq1.csv", index=False)
    real_r2.to_csv(out / "xv_real_rq2.csv", index=False)
    real_r3y.to_csv(out / "xv_real_rq3_yearly.csv", index=False)
    real_r3r.to_csv(out / "xv_real_rq3_regime.csv", index=False)
    logger.info("All results saved.")
