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

import numpy as np
import pandas as pd
from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.metrics import all_metrics
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
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

# Reuse RQ3's corrected regime helpers: its month labels are contemporaneous
# and its %Time denominator is labelled trading days, not calendar days.
# The sibling directory is put on sys.path so the import works when the file
# is executed as a script, imported by path, or loaded by tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rq3_stability import classify_regimes, regime_pct_time  # noqa: E402

setup_logging()
logger = get_logger("xvld")

# ---------------------------------------------------------------------------
# Protocol-frozen parameters
# ---------------------------------------------------------------------------
N_STOCKS = 500
SYNTHETIC_START = "2021-01-01"
# End dates are resolved at run time; no hardcoded stale windows.
SYNTHETIC_END = resolve_end_date()
REAL_START = "2018-01-01"
REAL_END = resolve_end_date()
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

    # TOP8 = equal-weight combination of the eight INDIVIDUAL factors, as in
    # RQ1. The old code reused the four cluster signals here, so the TOP8 row
    # was numerically identical to ALL and one of the sixteen experiments was
    # fabricated.
    top8 = [pf[f] for f in TOP8_FACTORS if f in pf]
    if top8:
        common8 = sorted(set.intersection(*(set(s.columns) for s in top8)))
        aligned8 = [s[common8] for s in top8]
        top8_sig = sum(aligned8) / len(aligned8)
        cs["TOP8"] = top8_sig.rank(axis=1, pct=True) - 0.5
    else:
        logger.warning("TOP8: no individual factors available")
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


# `classify_regimes` is imported from run_rq3_stability. The local copy that
# used to live here forward-filled month-end labels across calendar days
# (month M's days carried month M-1's label) and has been deleted.


# ======================================================================
# Experiment Runners
# ======================================================================

def run_rq1(cs, prices, returns, benchmark, sector_map, label=""):
    results = []
    for exp_name, combo in COMBOS.items():
        if combo is None:  # TOP8: the eight individual factors, not the clusters
            sigs = [cs["TOP8"]] if "TOP8" in cs else []
        else:
            sigs = [cs[c] for c in combo if c in cs]
        if not sigs:
            continue
        common = sorted(set.intersection(*(set(s.columns) for s in sigs)))
        comb = sum(s[common] for s in sigs) / len(sigs)
        signal = comb.rank(axis=1, pct=True) - 0.5

        ic_s = compute_ic(signal, returns)
        record = {"Experiment": exp_name,
                  "IC": ic_s.get("mean_ic", np.nan),
                  "ICIR": ic_s.get("icir", np.nan),
                  "Status": "ok", "Error": ""}
        try:
            bt = run_bt(signal, prices, returns, benchmark, sector_map)
            s = bt.get("summary", {})
            record.update({"Sharpe": s.get("sharpe_ratio", np.nan),
                           "MDD": s.get("max_drawdown", np.nan)})
        except Exception as exc:
            # A failed calculation is not "no edge": record NaN + the error.
            logger.warning("  %s %-12s backtest failed: %s", label, exp_name, exc)
            record.update({"Sharpe": np.nan, "MDD": np.nan,
                           "Status": "failed",
                           "Error": f"{type(exc).__name__}: {exc}"})
        results.append(record)
        logger.info("  %s %-12s IC=%.4f Sharpe=%.2f", label, exp_name,
                    record["IC"], record["Sharpe"])
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
        record = {"Experiment": exp_name,
                  "IC": ic_s.get("mean_ic", np.nan),
                  "ICIR": ic_s.get("icir", np.nan),
                  "Status": "ok", "Error": ""}
        try:
            bt = run_bt(signal, prices, returns, benchmark, sector_map)
            s = bt.get("summary", {})
            record.update({"Sharpe": s.get("sharpe_ratio", np.nan),
                           "MDD": s.get("max_drawdown", np.nan)})
        except Exception as exc:
            logger.warning("  %s %-12s backtest failed: %s", label, exp_name, exc)
            record.update({"Sharpe": np.nan, "MDD": np.nan,
                           "Status": "failed",
                           "Error": f"{type(exc).__name__}: {exc}"})
        results.append(record)
        logger.info("  %s %-12s IC=%.4f Sharpe=%.2f", label, exp_name,
                    record["IC"], record["Sharpe"])
    return pd.DataFrame(results)


def run_rq3(cs, prices, returns, benchmark, sector_map, label=""):
    yearly = []
    windows = year_windows(int(prices.index.min().year), str(prices.index.max().date()))
    for year_name, ys, ye in windows:
        yp = prices[ys:ye]
        yr = returns[ys:ye]
        for sn in CLUSTER_NAMES + [ALL_NAME]:
            if sn not in cs:
                continue
            sig = cs[sn][ys:ye]
            ic_s = compute_ic(sig, yr)
            record = {"Cluster": sn, "Year": year_name,
                      "IC": ic_s.get("mean_ic", np.nan),
                      "ICIR": ic_s.get("icir", np.nan),
                      "Status": "ok", "Error": ""}
            try:
                bt = run_bt(sig, yp, yr, returns, sector_map)
                s = bt.get("summary", {})
                record.update({"Sharpe": s.get("sharpe_ratio", np.nan),
                               "MDD": s.get("max_drawdown", np.nan)})
            except Exception as exc:
                # Keep the row with NaN and the error instead of dropping it:
                # a silently missing year looked like "not tested".
                logger.warning("  %s %s %s backtest failed: %s", label, year_name, sn, exc)
                record.update({"Sharpe": np.nan, "MDD": np.nan,
                               "Status": "failed",
                               "Error": f"{type(exc).__name__}: {exc}"})
            yearly.append(record)
        logger.info("  %s Year %s done", label, year_name)

    regime_records = []
    # RQ3's corrected helper: each trading day carries its OWN calendar
    # month's label (no forward-fill across calendar days), and unlabelled
    # days are excluded rather than silently labelled. The old code applied
    # month-end labels by reindex(method="ffill") over calendar days.
    daily_regime = classify_regimes(benchmark, trading_days=returns.index)
    total_labelled = int(daily_regime.notna().sum())
    if total_labelled == 0:
        logger.warning("  %s: no regime labels available", label)

    for regime_name in ["Bull", "Bear", "Sideways"]:
        regime_dates = daily_regime.index[daily_regime == regime_name]
        for sn in CLUSTER_NAMES + [ALL_NAME]:
            if sn not in cs:
                continue
            common_dates = sorted(set(cs[sn].index) & set(returns.index) & set(regime_dates))
            if len(common_dates) < 10:
                continue
            rsig = cs[sn].loc[common_dates]
            rret = returns.loc[common_dates]
            ic_s = compute_ic(rsig, rret)
            status, error = "ok", ""
            try:
                bt = run_bt(cs[sn], prices, returns, benchmark, sector_map)
                btr = bt.get("daily_returns", pd.Series(dtype=float))
                regime_bt = btr[btr.index.isin(regime_dates)]
                if len(regime_bt) >= 5:
                    # Explicit daily basis: infer_periods_per_year would read
                    # the sparse regime calendar as ~20-82 periods/year.
                    rm = all_metrics(regime_bt, None, periods_per_year=252)
                    sharpe = rm.get("sharpe_ratio", np.nan) if rm else np.nan
                else:
                    sharpe = np.nan
            except Exception as exc:
                logger.warning("Regime metrics failed: %s", exc)
                sharpe = np.nan
                status, error = "failed", f"{type(exc).__name__}: {exc}"
            pct = regime_pct_time(common_dates, daily_regime)
            regime_records.append({"Cluster": sn, "Regime": regime_name,
                                   "IC": ic_s.get("mean_ic", np.nan),
                                   "ICIR": ic_s.get("icir", np.nan),
                                   "Sharpe": sharpe, "%Time": round(pct, 1),
                                   "Status": status, "Error": error})
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
    syn_start, syn_end = str(sp.index.min().date()), str(sp.index.max().date())
    logger.info("Synthetic data window: %s → %s (requested end %s)",
                syn_start, syn_end, SYNTHETIC_END)
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
    real_start, real_end = str(rp.index.min().date()), str(rp.index.max().date())
    logger.info("Real data window: %s → %s (requested end %s)",
                real_start, real_end, REAL_END)
    pf_real = compute_factors(rp, rs, rt)
    cs_real = build_signals(pf_real)

    logger.info("[6/6] Real A-share RQ1+RQ2+RQ3...")
    real_r1 = run_rq1(cs_real, rp, rr, rb, rs, "REAL")
    real_r2 = run_rq2(cs_real, rp, rr, rb, rs, "REAL")
    real_r3y, real_r3r = run_rq3(cs_real, rp, rr, rb, rs, "REAL")

    # === Comparison Report ===
    print_comparison(syn_r1, real_r1, syn_r2, real_r2, syn_r3y, real_r3y, syn_r3r, real_r3r)

    # Save all (each artifact declares the window it was computed on)
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    stamp_provenance(syn_r1, syn_start, syn_end).to_csv(out / "xv_syn_rq1.csv", index=False)
    stamp_provenance(syn_r2, syn_start, syn_end).to_csv(out / "xv_syn_rq2.csv", index=False)
    stamp_provenance(syn_r3y, syn_start, syn_end).to_csv(out / "xv_syn_rq3_yearly.csv", index=False)
    stamp_provenance(syn_r3r, syn_start, syn_end).to_csv(out / "xv_syn_rq3_regime.csv", index=False)
    stamp_provenance(real_r1, real_start, real_end).to_csv(out / "xv_real_rq1.csv", index=False)
    stamp_provenance(real_r2, real_start, real_end).to_csv(out / "xv_real_rq2.csv", index=False)
    stamp_provenance(real_r3y, real_start, real_end).to_csv(out / "xv_real_rq3_yearly.csv", index=False)
    stamp_provenance(real_r3r, real_start, real_end).to_csv(out / "xv_real_rq3_regime.csv", index=False)
    logger.info("All results saved.")
