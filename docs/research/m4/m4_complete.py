"""M4.2 Complete Analysis — starts from the PROVEN m4_backtest.py logic.
=========================================================
Uses EXACT same portfolio loop as m4_backtest.py (the one that
produced Sharpe +0.87 / CAGR +6.31%). No novel accounting.
Then computes IC→Excess leakage from the result.
"""
import json
import math
import os
import sqlite3
import sys
import time

sys.path.insert(0, "D:/Desktop")
import akshare as ak
import numpy as np
import pandas as pd

DB = "data/trading.db"
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
INITIAL = 1_000_000

print("=" * 72)
print("  M4.2 Complete — Verified Backtest + Attribution + Leakage Audit")
print("=" * 72, flush=True)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════
t0 = time.time()
df = ak.index_stock_cons(symbol="000300")
code_col = [c for c in df.columns][0]
codes = sorted({str(c).zfill(6) for c in df[code_col].tolist()})
n_stocks = len(codes)
print(f"CSI300 members: {n_stocks}", flush=True)

def txsym(code):
    return ("sh" if code.startswith(("60","68","9")) else "sz") + code

frames = {}
for i, c in enumerate(codes):
    for retry in range(3):
        try:
            r = ak.stock_zh_a_hist_tx(symbol=txsym(c), start_date="20210101", end_date="20260523", adjust="hfq")
            if r is not None and not r.empty:
                r = r.rename(columns={"date":"date","close":"close"}) if "close" in r.columns else r
                r["date"] = pd.to_datetime(r["date"])
                frames[c] = r.set_index("date")["close"].astype(float)
                break
        except Exception:
            time.sleep(0.5)
    if i and i % 50 == 0:
        print(f"  prices {i}/{n_stocks} ok={len(frames)}", flush=True)

prices = pd.DataFrame(frames)
prices.index.name = "date"
rets = prices.pct_change(fill_method=None)
fwd_ret = rets.shift(-1)
all_dates = prices.index.sort_values()
ny = (all_dates[-1] - all_dates[0]).days / 365.25
print(f"Price panel: {prices.shape} (fetch {int(time.time()-t0)}s)", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 2. FACTOR + IC
# ══════════════════════════════════════════════════════════════════════
vol20 = -rets.rolling(20).std()

ic_vals = []
for date in vol20.index:
    f = vol20.loc[date].dropna()
    r = fwd_ret.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(r.index)
    if len(common) >= 20:
        ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
        if not np.isnan(ic):
            ic_vals.append(ic)
mean_ic = float(np.mean(ic_vals))
cs_vol = float(rets.std(axis=1).mean())

# IC-implied: IC × daily_cs_vol × mills_ratio × sqrt(252)
p = N_LONG / max(n_stocks, 1)
z = math.sqrt(-2 * math.log(p)) if p > 0 and p < 0.5 else 0
mills = math.exp(-z*z/2) / math.sqrt(2*math.pi) / p if p > 0 else 0
theo_ann = mean_ic * cs_vol * mills * 252

# ══════════════════════════════════════════════════════════════════════
# 3. PORTFOLIO BACKTEST (IDENTICAL to m4_backtest_lo.py)
# ══════════════════════════════════════════════════════════════════════
month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(
    lambda x: x.index[-1])
rebals = sorted(month_end_dates.unique())

capital = INITIAL
daily_rets = []
pv_vals = []
turnover_log = []
current_w = {}
rebal_iter = iter(rebals)
next_rebal = next(rebal_iter, None)

for date in all_dates:
    if next_rebal is not None and date >= next_rebal:
        if date in vol20.index:
            f = vol20.loc[date].dropna()
            if len(f) >= 60:
                rank = f.rank(ascending=False)
                long_u = rank.nsmallest(N_LONG).index.tolist()
                short_u = rank.nlargest(N_SHORT).index.tolist()

                if current_w:
                    old_set = set(current_w.keys())
                    new_set = set(long_u + short_u)
                    sold = old_set - new_set
                    if sold and len(current_w) > 0:
                        to = sum(abs(current_w.get(a,0)) for a in sold)
                        turnover_log.append(float(to))
                        capital -= to * (C + P + S) * capital

                total = max(len(long_u) + len(short_u), 1)
                current_w = {}
                for a in long_u:
                    current_w[a] = 1.0 / total
                for a in short_u:
                    current_w[a] = -1.0 / total
        try:
            next_rebal = next(rebal_iter)
        except StopIteration:
            next_rebal = None

    if current_w:
        daily = 0.0
        for a, w in current_w.items():
            if a in rets.columns and date in rets.index:
                r = rets.loc[date, a]
                if pd.notna(r):
                    daily += w * r
        daily_rets.append(daily)
        capital *= (1 + daily)
    else:
        daily_rets.append(0.0)
    pv_vals.append(capital)

dr = pd.Series(daily_rets, index=all_dates)
pv = pd.Series(pv_vals, index=all_dates)

# Benchmark: CSI300 equal-weight
bpv = (1 + fwd_ret.mean(axis=1).reindex(all_dates).fillna(0)).cumprod()

# ══════════════════════════════════════════════════════════════════════
# 4. STYLE BENCHMARK — same factor, same construction, NO costs
#    Answers: is the strategy just delivering style exposure,
#    or is there residual alpha beyond the naive factor portfolio?
# ══════════════════════════════════════════════════════════════════════

style_capital = INITIAL
style_daily_rets = []
style_current_w = {}
style_rebal_iter = iter(rebals)
style_next_rebal = next(style_rebal_iter, None)

for date in all_dates:
    if style_next_rebal is not None and date >= style_next_rebal:
        if date in vol20.index:
            f = vol20.loc[date].dropna()
            if len(f) >= 60:
                rank = f.rank(ascending=False)
                long_u = rank.nsmallest(N_LONG).index.tolist()
                total = max(len(long_u), 1)
                style_current_w = {}
                for a in long_u:
                    style_current_w[a] = 1.0 / total
        try:
            style_next_rebal = next(style_rebal_iter)
        except StopIteration:
            style_next_rebal = None

    if style_current_w:
        daily = 0.0
        for a, w in style_current_w.items():
            if a in rets.columns and date in rets.index:
                r = rets.loc[date, a]
                if pd.notna(r):
                    daily += w * r
        style_daily_rets.append(daily)
        style_capital *= (1 + daily)
    else:
        style_daily_rets.append(0.0)

style_dr = pd.Series(style_daily_rets, index=all_dates)
style_pv = (1 + style_dr).cumprod() * INITIAL
style_cagr = (style_pv.iloc[-1]/INITIAL) ** (1/ny) - 1
style_vol = float(style_dr.std() * np.sqrt(252))
style_sharpe = float((style_dr.mean()/(style_dr.std()+1e-12)) * np.sqrt(252))
style_dd = float(((style_pv - style_pv.cummax()) / style_pv.cummax()).min())
cagr = (pv.iloc[-1]/INITIAL) ** (1/ny) - 1
ann_vol = float(dr.std() * np.sqrt(252))
sharpe = float((dr.mean()/(dr.std()+1e-12)) * np.sqrt(252))
sortino = float((dr.mean()/(dr[dr<0].std()+1e-12)) * np.sqrt(252))
dd_series = (pv - pv.cummax()) / pv.cummax()
max_dd = float(dd_series.min())
win_daily = float((dr > 0).mean())
monthly_r = dr.resample("ME").apply(lambda x: (1+x).prod()-1)
win_monthly = float((monthly_r > 0).mean())
avg_to = float(np.mean(turnover_log)) if turnover_log else 0
bench_cagr = (bpv.iloc[-1]/bpv.iloc[0]) ** (1/ny) - 1
excess = cagr - bench_cagr

# Yearly
yret = {}
for yr, g in dr.groupby(dr.index.year):
    yret[yr] = float((1+g).prod()-1)

# Roll 12m Sharpe
roll12 = dr.rolling(252).mean()/(dr.rolling(252).std()+1e-12)*np.sqrt(252)

# Temporal split
mid = int(len(dr) * 0.55)
early = dr.iloc[:mid]
late = dr.iloc[mid:]
early_b = fwd_ret.mean(axis=1).reindex(early.index).dropna()
late_b = fwd_ret.mean(axis=1).reindex(late.index).dropna()
early_c = (1+early).prod()**(252/len(early))-1 if len(early) else 0
late_c = (1+late).prod()**(252/len(late))-1 if len(late) else 0
early_bench = (1+early_b).prod()**(252/len(early_b))-1 if len(early_b)>0 else 0
late_bench = (1+late_b).prod()**(252/len(late_b))-1 if len(late_b)>0 else 0

# ══════════════════════════════════════════════════════════════════════
# 5. ALPHA LEAKAGE AUDIT (accounting from result, not interleaved)
# ══════════════════════════════════════════════════════════════════════
# Gross (pre-cost) ≈ net + cost drag from turnover events
n_rebals = len(turnover_log)
total_to = sum(turnover_log) if turnover_log else 0
# Each turnover event (fraction of AUM sold) pays (C+P+S) cost
total_cost_paid = sum(t * (C+P+S) for t in turnover_log)
annual_cost_drag = total_cost_paid / ny if ny > 0 else 0

# Gross CAGR = what CAGR would have been without costs
# Approximate: net CAGR + annual_cost_drag
gross_cagr = cagr + annual_cost_drag

# Construction gap = theory - gross
construction_gap = theo_ann - gross_cagr
# Cost drag = gross - net
cost_drag_ann = gross_cagr - cagr
# Benchmark gap = net CAGR - bench CAGR = excess
bench_gap = excess

total_leak = theo_ann - excess

# Percent attribution
leak_theory = abs(theo_ann) + 1e-8
constr_pct = abs(construction_gap) / leak_theory * 100 if construction_gap > 0 else 0
cost_pct = abs(cost_drag_ann) / leak_theory * 100
bench_pct = abs(bench_gap - construction_gap + cost_drag_ann) / leak_theory * 100

# ══════════════════════════════════════════════════════════════════════
# 6. PRINT
# ══════════════════════════════════════════════════════════════════════
print("")
print("=" * 72)
print("  M4.2 LONG-ONLY BACKTEST")
print("=" * 72)
print(f"  CAGR:            {cagr:>+8.2%}")
print(f"  Ann Vol:         {ann_vol:>8.2%}")
print(f"  Sharpe:          {sharpe:>+8.2f}")
print(f"  Sortino:         {sortino:>+8.2f}")
print(f"  Max DD:          {max_dd:>+8.2%}")
print(f"  Daily Win:       {win_daily:>7.1%}  Monthly Win: {win_monthly:>7.1%}")
print(f"  Avg Turnover/Rb: {avg_to*100:>7.1f}%")
print(f"  Benchmark CAGR:  {bench_cagr:>+8.2%} (CSI300 EW)")
print(f"  Excess vs BM:    {excess:>+8.2%}")
print(f"  N Rebalances:    {n_rebals}")
print("")
print("   Yearly Returns:")
for yr in sorted(yret):
    bm_yr = (1+fwd_ret.mean(axis=1).loc[fwd_ret.mean(axis=1).index.year == yr]).prod()-1
    print(f"    {yr}: strategy {yret[yr]:>+7.2%}  benchmark {bm_yr:>+7.2%}")

# Market benchmark daily returns for metrics
b_dr = fwd_ret.mean(axis=1).reindex(dr.index).dropna()
b_vol = float(b_dr.std() * np.sqrt(252))
b_sharpe = float((b_dr.mean()/(b_dr.std()+1e-12)) * np.sqrt(252))
b_maxdd = float((b_dr.cumsum() - b_dr.cumsum().cummax()).min())

print("")
print("=" * 72)
print("  BENCHMARK COMPARISON — Market vs Style")
print("=" * 72)
print(f"  {'Metric':<30} {'Strategy':>11} {'Market':>11} {'Style':>11}")
print(f"  {'':30} {'(vol_20d LO)':>11} {'(CSI300 EW)':>11} {'(no-cost)':>11}")
print(f"  {'-'*30} {'-'*11} {'-'*11} {'-'*11}")
for name, sv, mv, stv in [
    ("CAGR", cagr, bench_cagr, style_cagr),
    ("Ann Volatility", ann_vol, b_vol, style_vol),
    ("Sharpe Ratio", sharpe, b_sharpe, style_sharpe),
    ("Max Drawdown", max_dd, b_maxdd, style_dd),
]:
    print(f"  {name:<30} {sv:>+10.2%} {mv:>+10.2%} {stv:>+10.2%}")
print(f"  {'-'*30} {'-'*11} {'-'*11} {'-'*11}")

# Style excess: strategy vs naive no-cost factor portfolio
style_excess = cagr - style_cagr
print(f"  {'Excess vs Market':<30} {excess:>+10.2%}")
print(f"  {'Excess vs Style':<30} {style_excess:>+10.2%}")
print("")

# Yearly with style
print("  YEARLY — All Three Benchmarks")
print(f"  {'Year':<8} {'Strategy':>9} {'Market':>9} {'Style':>9} {'Ex-vs-Mkt':>9} {'Ex-vs-Style':>9}")
print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
for yr in sorted(yret):
    sy = (1+dr[dr.index.year==yr]).prod()-1
    my = (1+fwd_ret.mean(axis=1).loc[fwd_ret.mean(axis=1).index.year==yr]).prod()-1
    sty = (1+style_dr[style_dr.index.year==yr]).prod()-1
    print(f"  {yr:<8} {sy:>+8.2%} {my:>+8.2%} {sty:>+8.2%} {(sy-my):>+9.2%} {(sy-sty):>+9.2%}")

# Diagnosis
print("")
print("  DIAGNOSIS")
if abs(style_excess) < 0.02:
    print(f"  Strategy vs Style excess = {style_excess:+.2%}")
    print("  → Strategy ≈ Naive Low-Vol. The factor IS the style.")
    print("  → No evidence of alpha beyond simple low-vol selection.")
    print("  → Next: multi-factor to escape single-style ceiling.")
elif style_excess < -0.02:
    print(f"  Strategy vs Style excess = {style_excess:+.2%}")
    print("  → Strategy UNDERPERFORMS naive low-vol benchmark.")
    print("  → Costs or construction are destroying value beyond just style.")
    print("  → Next: audit cost model and rebalance frequency.")
else:
    print(f"  Strategy vs Style excess = {style_excess:+.2%}")
    print("  → Strategy OUTPERFORMS naive low-vol benchmark.")
    print("  → Residual alpha exists beyond pure factor exposure.")
    print("  → Next: Walk-Forward OOS to verify persistence.")

print("")
print("=" * 72)
print("  ALPHA LEAKAGE AUDIT — IC → Net Excess")
print("=" * 72)
print(f"  Mean IC:            {mean_ic:+.4f}")
print(f"  N stocks / Top-N:   {n_stocks} / {N_LONG}")
print("")
print(f"  {'Step':<5} {'Component':<40} {'Annual':>10}")
print(f"  {'-'*5} {'-'*40} {'-'*10}")
print(f"  {'[0]':<5} {'IC-implied theoretical alpha':<40} {theo_ann:>+9.2%}")
print(f"  {'[1]':<5} {'▼ Construction (theory→gross)':<40} {-construction_gap:>+9.2%}")
print(f"  {'[2]':<5} {'Gross portfolio (pre-cost)':<40} {gross_cagr:>+9.2%}")
print(f"  {'[3]':<5} {'▼ Cost drag (turnover+fees)':<40} {-cost_drag_ann:>+9.2%}")
print(f"  {'[4]':<5} {'Net portfolio (post-cost)':<40} {cagr:>+9.2%}")
print(f"  {'[5]':<5} {'▼ Benchmark gap':<40} {-excess:>+9.2%}")
print(f"  {'':<5} {'= Net Excess (vs CSI300 EW)':<40} {excess:>+9.2%}")
print("")
print("  Leakage decomposition:")
print(f"    Construction:       {constr_pct:.0f}%  ({-construction_gap:+.2%})")
print(f"    Trading costs:      {cost_pct:.0f}%  ({-cost_drag_ann:+.2%})")
print(f"    Benchmark mismatch: {bench_pct:.0f}%")
print("")
print("  Temporal decay:")
print(f"    Early: strat {early_c:+.2%}  bench {early_bench:+.2%}  excess {early_c-early_bench:+.2%}")
print(f"    Late:  strat {late_c:+.2%}  bench {late_bench:+.2%}  excess {late_c-late_bench:+.2%}")
print("")
if constr_pct > cost_pct and constr_pct > bench_pct:
    print(f"  → PRIMARY LEAK: Portfolio Construction ({constr_pct:.0f}%)")
    print("    Theory→Gross gap is the biggest loss. Score-weight or wider N.")
elif cost_pct > constr_pct and cost_pct > bench_pct:
    print(f"  → PRIMARY LEAK: Trading Costs ({cost_pct:.0f}%)")
    print("    37% monthly turnover at 18bp round-trip = material drain.")
    print("    Options: wider top-N (30→50), quarterly rebalance, lower-turnover weighting.")
else:
    print(f"  → PRIMARY LEAK: Benchmark mismatch ({bench_pct:.0f}%)")
    print("    Low-vol picks miss market beta in bull markets.")
print("=" * 72)

# ══════════════════════════════════════════════════════════════════════
# 7. REGISTRY
# ══════════════════════════════════════════════════════════════════════
rid = f"m4.2_lo_{int(time.time())}"
ev = {
    "experiment": "m4.2_long_only_complete",
    "hypothesis": "Short leg is primary drag; long-only improves risk-return.",
    "variable_changed": "N_SHORT: 20 → 0",
    "cagr": float(cagr), "sharpe": sharpe, "sortino": sortino,
    "max_dd": max_dd, "ann_vol": ann_vol,
    "win_daily": win_daily, "win_monthly": win_monthly,
    "avg_turnover": avg_to, "n_rebalances": n_rebals,
    "benchmark_cagr": float(bench_cagr), "excess": float(excess),
    "yearly": yret,
    "ic_mean": float(mean_ic), "ic_implied_ann": float(theo_ann),
    "construction_gap": float(construction_gap), "cost_drag_ann": float(-cost_drag_ann),
    "gross_cagr": float(gross_cagr),
    "leakage": {
        "construction_pct": float(constr_pct), "cost_pct": float(cost_pct),
        "bench_mismatch_pct": float(bench_pct),
        "primary": "construction" if constr_pct > max(cost_pct,bench_pct) else ("cost" if cost_pct > max(constr_pct,bench_pct) else "benchmark"),
    },
    "temporal": {"early_cagr": float(early_c), "late_cagr": float(late_c)},
    "cost_model": {"commission":C,"stamp_tax":S,"slippage":P},
    "setup": {"n_long":N_LONG,"n_short":N_SHORT},
    "baseline_m4_1_sharpe": -0.96,
}
conn = sqlite3.connect(DB)
conn.execute(
    """INSERT INTO research_runs
       (run_id, timestamp, status, reason, slice, data_source, data_meta,
        universe_provider, universe_meta, factor, factor_params,
        evaluation, input_hash, report_path, warnings)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (rid, pd.Timestamp.now('UTC').isoformat()[:19], "success", "",
     "m4.2_long_only_backtest", "akshare",
     json.dumps({"adjust":"hfq","provider":"akshare"}),
     "CurrentConstituentUniverseProvider",
     json.dumps({"pit":False,"bias_warning":["survivorship_bias_possible"]}),
     "volatility_20d",
     json.dumps({"n_long":N_LONG,"n_short":N_SHORT}),
     json.dumps(ev, default=str), "", "", "[]"))
conn.commit()
conn.close()
print(f"Registry: {rid}")
print("DONE")
