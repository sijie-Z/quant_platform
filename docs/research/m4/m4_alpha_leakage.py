"""M4.2 Alpha Leakage Audit — transaction-path decomposition.
=============================================================
Traces alpha from IC to net excess, step by step through the
actual portfolio construction & execution pipeline.

NO CAPM. NO factor regression. Pure accounting decomposition:
  Step 0: IC-implied alpha (daily rank IC × cross-sectional vol)
  Step 1: → Top-20 construction (what % of rank spread survives equal-weight)
  Step 2: → Gross portfolio return (before costs)
  Step 3: → Turnover drag (sold positions × cost rate)
  Step 4: → Trading costs (commission + stamp + slippage per rebalance)
  Step 5: → Benchmark gap (beta × benchmark return, or simple subtraction)
  Step 6: = Net excess (actual excess vs benchmark)

Returns one table: where the alpha dies.
"""
import json
import os
import sys
import time

sys.path.insert(0, "D:/Desktop")
import akshare as ak
import numpy as np
import pandas as pd

print("=" * 72)
print("  M4.2 — Alpha Leakage Audit: IC → Net Excess")
print("=" * 72, flush=True)

# ── 1. Data ──────────────────────────────────────────────────────────
start = time.time()
df = ak.index_stock_cons(symbol="000300")
code_col = [c for c in df.columns][0]
codes = sorted({str(c).zfill(6) for c in df[code_col].tolist()})
n_stocks = len(codes)
print(f"CSI300 members: {n_stocks}", flush=True)

def txsym(c):
    return ("sh" if c.startswith(("60", "68", "9")) else "sz") + c

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
ny = (prices.index[-1] - prices.index[0]).days / 365.25
print(f"Price panel: {prices.shape} (fetch {int(time.time()-start)}s)", flush=True)

# ── 2. Factor ───────────────────────────────────────────────────────
vol20 = -rets.rolling(20).std()
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0

# ── STEP 0: IC-implied theoretical alpha ─────────────────────────────
# For each day, compute rank IC & cross-sectional dispersion
# Theoretical gross alpha ≈ IC_mean × CS_vol(daily) × selection_breadth
ic_vals = []
cs_vols = []
for date in vol20.index:
    f = vol20.loc[date].dropna()
    r = fwd_ret.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(r.index)
    if len(common) >= 20:
        ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
        cs_vol = r.loc[common].std()
        if not np.isnan(ic) and not np.isnan(cs_vol):
            ic_vals.append((date, ic))
            cs_vols.append((date, cs_vol))

ic_series = pd.Series({d:v for d,v in ic_vals})
cs_vol_series = pd.Series({d:v for d,v in cs_vols})
ic_common = ic_series.index.intersection(cs_vol_series.index)
mean_ic = float(ic_series.loc[ic_common].mean())
mean_cs_vol = float(cs_vol_series.loc[ic_common].mean())

# Theoretical: if you could perfectly capture the top decile's alpha
# For a long-only top-20 from ~300 stocks: cutoff ≈ 93rd percentile
# Mills ratio λ = φ(z)/Φ(-z) for top fraction p
# p = N_LONG/n_stocks ≈ 20/288 ≈ 0.0694 → z ≈ 1.48, λ ≈ 2.12
p_select = N_LONG / n_stocks
# Rational approximation of inverse normal CDF
import math

t = math.sqrt(-2 * math.log(p_select)) if p_select > 0 and p_select < 0.5 else 0
z_cutoff = t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
selection_width = math.exp(-z_cutoff*z_cutoff/2) / math.sqrt(2*math.pi) / p_select
theoretical_daily_alpha = mean_ic * mean_cs_vol * selection_width
theoretical_ann_alpha = theoretical_daily_alpha * 252

# ── Run portfolio (same as M4.2) ──────────────────────────────────────
month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(lambda x: x.index[-1])
rebals = sorted(month_end_dates.unique())
all_dates = prices.index.sort_values()

# Track detailed transaction data for audit
txn_log = []  # per-rebalance: date, sold_pct, cost_paid, turnover_pct
holdings_history = {}  # date -> (assets, weights)
gross_daily = []  # before costs
net_daily = []    # after costs
bench_daily = []

capital = 1_000_000
initial = capital
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

                sold_pct = 0.0
                if current_w:
                    old_set = set(current_w.keys())
                    new_set = set(long_u + short_u)
                    sold = old_set - new_set
                    if sold and len(current_w) > 0:
                        sold_pct = sum(abs(current_w.get(a,0)) for a in sold)
                        cost_pct = sold_pct * (C + P + S)
                        capital -= cost_pct * capital
                        txn_log.append({
                            "date": date,
                            "sold_pct": float(sold_pct),
                            "cost_pct": float(cost_pct),
                            "n_held": len(old_set),
                            "capital": float(capital),
                        })

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
        holdings_history[date] = dict(current_w)
        # Gross return (before costs applied above)
        gross_r = sum(current_w.get(a,0) * rets.loc[date,a]
                      for a in current_w if a in rets.columns
                      and date in rets.index and pd.notna(rets.loc[date,a]))
        gross_daily.append((date, gross_r))
        # Cost was already deducted from capital at rebalance time
        # Net daily capital change already embedded in capital evolution
        net_daily.append((date, capital))
        # Benchmark
        valid_b = fwd_ret.loc[date].dropna()
        if len(valid_b) > 0:
            bench_daily.append((date, float(valid_b.mean())))
    else:
        net_daily.append((date, capital))

# ── Compute step-by-step chain ────────────────────────────────────────
# Align indices
gross_series = pd.Series({d:v for d,v in gross_daily if not np.isnan(v)})
net_vals = pd.Series({d:v for d,v in net_daily})
bench_series = pd.Series({d:v for d,v in bench_daily})

common = gross_series.index.intersection(bench_series.index)
gross_series = gross_series.loc[common]
bench_series = bench_series.loc[common]
# net_vals covers more dates (before first position), align
net_series = net_vals.pct_change(fill_method=None)
net_series = net_series.loc[common].dropna()

# Step 0: Theoretical IC-implied alpha
theo_daily_alpha = theoretical_daily_alpha

# Step 1: Gross portfolio return
gross_cagr = float((1+gross_series).prod() ** (1/ny) - 1)

# Step 2: Net portfolio return (after costs)
net_cagr = float((net_vals.iloc[-1]/initial) ** (1/ny) - 1)

# Step 3: Benchmark return
bench_cagr = float((1+bench_series).prod() ** (1/ny) - 1)

# Cost summary
if txn_log:
    total_cost_pct = sum(t["cost_pct"] for t in txn_log)
    avg_turnover_pct = float(np.mean([t["sold_pct"] for t in txn_log]))
    n_rebals = len(txn_log)
    annual_cost_bps = total_cost_pct / ny * 10000
else:
    total_cost_pct = 0
    avg_turnover_pct = 0
    n_rebals = 0
    annual_cost_bps = 0

# Step 5: Construction loss = IC-implied - gross (before costs)
construction_gap = theoretical_ann_alpha - gross_cagr

# Step 6: Cost drag = gross - net
cost_drag = gross_cagr - net_cagr

# Step 7: Benchmark gap = net CAGR - bench CAGR
excess = net_cagr - bench_cagr

# Step 8: IC→excess total leakage
total_leakage = theoretical_ann_alpha - excess

# ── Print audit table ──────────────────────────────────────────────────
print("")
print("=" * 72)
print("  ALPHA LEAKAGE AUDIT — IC → Net Excess")
print("=" * 72)
print(f"  Time period:    {common[0].strftime('%Y-%m-%d')} → {common[-1].strftime('%Y-%m-%d')} ({ny:.1f}yr)")
print(f"  Mean IC:        {mean_ic:+.4f}")
print(f"  Mean CS vol:    {mean_cs_vol*100:.2f}% daily")
print(f"  N stocks:       {n_stocks}")
print(f"  Top-N selected: {N_LONG}")
print("")
print(f"  {'Step':<5} {'Component':<40} {'Annual':>10} {'Cumulative':>10}")
print(f"  {'-'*5} {'-'*40} {'-'*10} {'-'*10}")
print(f"  {'[0]':<5} {'IC-implied theoretical alpha':<40} {theoretical_ann_alpha:>+9.2%} {'':>10}")
print(f"  {'[1]':<5} {'▼ Construction gap':<40} {construction_gap:>+9.2%} {theoretical_ann_alpha-construction_gap:>+9.2%}")
print(f"  {'[2]':<5} {'Gross portfolio return (pre-cost)':<40} {gross_cagr:>+9.2%} {gross_cagr:>+9.2%}")
print(f"  {'[3]':<5} {'▼ Cost drag':<40} {-cost_drag:>+9.2%} {net_cagr:>+9.2%}")
print(f"  {'[4]':<5} {'Net portfolio return (post-cost)':<40} {net_cagr:>+9.2%} {net_cagr:>+9.2%}")
print(f"  {'[5]':<5} {'▼ Benchmark gap':<40} {-excess:>+9.2%} {net_cagr-excess:>+9.2%}")
print(f"  {'':<5} {'= Net Excess (vs CSI300 EW)':<40} {excess:>+9.2%} {'':>10}")
print("")
print(f"  {'─'*5} {'─'*40} {'─'*10} {'─'*10}")
print(f"  {'':<5} {'TOTAL LEAKAGE (IC→Excess)':<40} {total_leakage:>+9.2%} {'':>10}")
print("")
print("  ── Cost Detail ──")
print(f"  N rebalances:     {n_rebals}")
print(f"  Avg turnover/rb:  {avg_turnover_pct*100:.1f}%")
print(f"  Total cost paid:  {total_cost_pct*100:.2f}% of AUM")
print(f"  Annual cost bps:  {annual_cost_bps:.0f} bps/yr")
print(f"  Cost params:      C={C*10000:.0f}bp S={S*10000:.0f}bp Slip={P*10000:.0f}bp")
print("")
print("  ── Leakage Diagnosis ──")
constr_pct = abs(construction_gap) / (abs(theoretical_ann_alpha) + 1e-8) * 100
cost_pct_leak = abs(cost_drag) / (abs(theoretical_ann_alpha) + 1e-8) * 100
bench_pct = abs(excess - construction_gap - cost_drag) / (abs(theoretical_ann_alpha) + 1e-8) * 100
print(f"  Construction loss: {constr_pct:.0f}% of total leakage")
print(f"  Cost drag:         {cost_pct_leak:.0f}% of total leakage")
print(f"  Beta/size timing:  {bench_pct:.0f}% of total leakage")
print("")
if constr_pct > cost_pct_leak and constr_pct > bench_pct:
    print("  → PRIMARY LEAK: Portfolio Construction")
    print("    The IC-implied alpha is REAL but the equal-weight top-20")
    print("    construction captures only a fraction of it.")
    print("    Next: vary top-N, score-weighting, rebalance frequency")
    print("    to narrow the construction gap BEFORE adding more factors.")
elif cost_pct_leak > constr_pct and cost_pct_leak > bench_pct:
    print("  → PRIMARY LEAK: Trading Costs")
    print("    Turnover is the dominant drag. Widen top-N or shift to")
    print("    quarterly rebalance to reduce costs.")
else:
    print("  → PRIMARY LEAK: Benchmark / Market structure")
    print("    The strategy's return profile doesn't match the benchmark.")
    print("    Low-vol bias means it underperforms in bull markets.")
    print("    This is structural — consider a different benchmark or")
    print("    add a market-beta factor to the mix.")
print("=" * 72)
print("DONE")
