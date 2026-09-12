"""M4.2 Return Attribution — where does the alpha leak?
=====================================================
Decomposes vol_20d Long-Only portfolio return into:
  1. Market Beta contribution   — capm_beta × benchmark_return
  2. Low-Vol Style contribution — factor_tilt × factor_premium
  3. Selection / Residual       — what's left (real alpha or noise)

Answers: WHY does a factor with IC=+0.033 produce Sharpe=+0.87
but Excess=-5.26% against CSI300 EW?
"""
import os
import sys
import time

sys.path.insert(0, "D:/Desktop")
import akshare as ak
import numpy as np
import pandas as pd

print("=" * 72)
print("  M4.2 — Return Attribution: Where Does the Alpha Leak?")
print("=" * 72, flush=True)

# ── 1. Data ──────────────────────────────────────────────────────────
start = time.time()
df = ak.index_stock_cons(symbol="000300")
code_col = [c for c in df.columns][0]
codes = sorted({str(c).zfill(6) for c in df[code_col].tolist()})
print(f"CSI300 members: {len(codes)}", flush=True)

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
        print(f"  prices {i}/{len(codes)} ok={len(frames)}", flush=True)

prices = pd.DataFrame(frames)
prices.index.name = "date"
rets = prices.pct_change(fill_method=None)
fwd_ret = rets.shift(-1)
print(f"Price panel: {prices.shape} (fetch {int(time.time()-start)}s)", flush=True)

# ── 2. Factor & Portfolio ────────────────────────────────────────────
vol20 = -rets.rolling(20).std()
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
initial = 1_000_000

month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(lambda x: x.index[-1])
rebals = sorted(month_end_dates.unique())
all_dates = prices.index.sort_values()

# Track portfolio holdings over time for attribution
holdings_history = {}  # date -> {asset: weight}
turnover_events = []   # (date, sold_assets_weight)

capital = initial
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
                        capital -= to * (C + P + S) * capital
                        turnover_events.append((date, float(to)))
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
    capital *= (1 + sum(current_w.get(a,0)*rets.loc[date,a]
                        for a in current_w if a in rets.columns
                        and date in rets.index and pd.notna(rets.loc[date,a])))

# Daily strategy returns
strategy_dr = pd.Series(index=all_dates, dtype=float)
for date in all_dates:
    if date in holdings_history:
        w = holdings_history[date]
        strategy_dr[date] = sum(w.get(a,0)*rets.loc[date,a]
                               for a in w if a in rets.columns
                               and date in rets.index and pd.notna(rets.loc[date,a]))
    else:
        strategy_dr[date] = 0.0

bench_dr = fwd_ret.mean(axis=1)
common = strategy_dr.dropna().index.intersection(bench_dr.dropna().index)
strategy_dr = strategy_dr.loc[common]
bench_dr = bench_dr.loc[common]

# ── 3. CAPM Decomposition ────────────────────────────────────────────
# R_portfolio = alpha + beta * R_market + epsilon
X = np.column_stack([bench_dr.values])
Y = strategy_dr.values
beta, alpha_daily = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]
market_contrib = beta * bench_dr
residual = strategy_dr - (alpha_daily + market_contrib)

# ── 4. Style Attribution: Is vol_20d just harvesting low-vol premium? ─
# Build a daily low-vol factor portfolio return (extreme decile)
# Low-vol decile = stocks with LOWEST 20d volatility → we go long
# The "low vol premium" = return of low-vol stocks - return of market
lowvol_ret = pd.Series(index=rets.index, dtype=float)
for date in rets.index:
    if date in vol20.index:
        v = vol20.loc[date].dropna()
        if len(v) >= 60:
            # vol20 is NEGATED std → higher = lower vol
            # Select lowest vol (highest vol20 value) stocks
            top_lo = v.nlargest(int(len(v)*0.2))
            wgt = top_lo / top_lo.sum() if top_lo.sum() != 0 else top_lo * 0
            # Weighted return of low-vol basket
            r = rets.loc[date]
            common_a = [a for a in top_lo.index if a in r.index and pd.notna(r[a])]
            if common_a:
                w = top_lo[common_a] / top_lo[common_a].sum()
                lowvol_ret[date] = (w * r[common_a]).sum()

lowvol_ret = lowvol_ret.dropna()
lv_common = strategy_dr.index.intersection(lowvol_ret.index)
lv_aligned = lowvol_ret.loc[lv_common]

# Build a SIZE proxy: log market cap (price itself as rough proxy)
# Stocks with higher price → larger cap (very rough, but directional)
size_ret = pd.Series(index=rets.index, dtype=float)
for date in rets.index:
    px = prices.loc[date].dropna()
    if len(px) >= 60:
        big = px.nlargest(int(len(px)*0.2))
        wgt = big / big.sum()
        r = rets.loc[date]
        common_a = [a for a in big.index if a in r.index and pd.notna(r[a])]
        if common_a:
            w = wgt[common_a] / wgt[common_a].sum()
            size_ret[date] = (w * r[common_a]).sum()

size_ret = size_ret.dropna()
sz_common = strategy_dr.index.intersection(size_ret.index)
sz_aligned = size_ret.loc[sz_common]

# ── 5. Three-Factor Regression ───────────────────────────────────────
# R_strat = alpha + b1*R_mkt + b2*R_lowvol + b3*R_size + e
# If alpha > 0 after controlling for low-vol and size → real signal
# If alpha ~ 0 → factor is just low-vol + size exposure
lv_reindexed = lv_aligned.reindex(common).fillna(0)
sz_reindexed = sz_aligned.reindex(common).fillna(0)
bm_reindexed = bench_dr.loc[common]
st_reindexed = strategy_dr.loc[common]

# Build factor matrix
F = np.column_stack([
    bm_reindexed.values,     # market
    lv_reindexed.values,     # low-vol factor
    sz_reindexed.values,     # size factor
])
mask = ~np.isnan(F).any(axis=1)
Yf = st_reindexed.values[mask]
Xf = np.column_stack([np.ones(mask.sum()), F[mask]])
coeff = np.linalg.lstsq(Xf, Yf, rcond=None)[0]
alpha3 = coeff[0] * 252  # annualized
beta_mkt = coeff[1]
beta_lv = coeff[2]
beta_sz = coeff[3]

# Contributions
mkt_contrib_ann = beta_mkt * bm_reindexed[mask].mean() * 252
lv_contrib_ann = beta_lv * lv_reindexed[mask].mean() * 252
sz_contrib_ann = beta_sz * sz_reindexed[mask].mean() * 252
total_ann = st_reindexed[mask].mean() * 252
residual_ann = total_ann - (alpha3 + mkt_contrib_ann + lv_contrib_ann + sz_contrib_ann)

# ── 6. Turnover Cost Attribution ─────────────────────────────────────
if turnover_events:
    to_dates = [t[0] for t in turnover_events]
    to_vals = [t[1] for t in turnover_events]
    avg_monthly_to = np.mean(to_vals)
    annual_to_cost = avg_monthly_to * 12 * (C + P + S)  # approximate
    # More precise: each turnover event cost = to * (C+P+S) already applied
    total_to_cost = sum(to_vals) * (C + P + S)
else:
    avg_monthly_to = 0
    annual_to_cost = 0
    total_to_cost = 0

# ── 7. Concentration / Effective N ────────────────────────────────────
# How concentrated is the portfolio?
avg_n_holdings = np.mean([len(w) for w in holdings_history.values()]) if holdings_history else 0
avg_hhi = np.mean([sum((v**2) for v in w.values()) / (sum(w.values())**2)
                   for w in holdings_history.values() if sum(w.values()) > 0]) if holdings_history else 0

# ── 8. Factor Decay: early vs late period ─────────────────────────────
midpoint = common[len(common)//2]
early_s = strategy_dr[common <= midpoint]
late_s = strategy_dr[common > midpoint]
early_b = bench_dr[common <= midpoint]
late_b = bench_dr[common > midpoint]
early_cagr = (1+early_s).prod()**(252/len(early_s))-1 if len(early_s)>0 else 0
late_cagr = (1+late_s).prod()**(252/len(late_s))-1 if len(late_s)>0 else 0
early_bench = (1+early_b).prod()**(252/len(early_b))-1 if len(early_b)>0 else 0
late_bench = (1+late_b).prod()**(252/len(late_b))-1 if len(late_b)>0 else 0

# ── 9. Print ──────────────────────────────────────────────────────────
print("")
print("=" * 72)
print("  RETURN ATTRIBUTION — Where Does the Alpha Leak?")
print("=" * 72)
print("")
print("  ── CAPM Single-Factor ──")
print(f"  Alpha (annualized):       {alpha_daily*252:>+10.2%}")
print(f"  Market Beta:              {beta:>10.2f}")
print(f"  Market Contribution:      {beta*bench_dr.mean()*252:>+10.2%}")
print(f"  Residual Vol (ann):       {float(residual.std()*np.sqrt(252)):>10.2%}")
print("")
print("  ── Three-Factor Decomposition ──")
print(f"  Alpha (annualized):       {alpha3:>+10.2%}")
print(f"  Market Beta:              {beta_mkt:>10.3f}")
print(f"  Low-Vol Beta:             {beta_lv:>10.3f}")
print(f"  Size Beta:                {beta_sz:>10.3f}")
print(f"  {'':>20} {'Annual':>10} {'% of Total':>12}")
print(f"  {'Market Contribution':<20} {mkt_contrib_ann:>+10.2%} {(mkt_contrib_ann/total_ann*100 if total_ann>0 else 0):>+11.1f}%")
print(f"  {'Low-Vol Contribution':<20} {lv_contrib_ann:>+10.2%} {(lv_contrib_ann/total_ann*100 if total_ann>0 else 0):>+11.1f}%")
print(f"  {'Size Contribution':<20} {sz_contrib_ann:>+10.2%} {(sz_contrib_ann/total_ann*100 if total_ann>0 else 0):>+11.1f}%")
print(f"  {'Alpha (residual)':<20} {alpha3:>+10.2%} {(alpha3/total_ann*100 if total_ann>0 else 0):>+11.1f}%")
print(f"  {'─'*20} {'─'*10}")
print(f"  {'TOTAL':<20} {total_ann:>+10.2%}")
print("")
print("  ── Turnover Cost Drain ──")
print(f"  Avg holdings:             {avg_n_holdings:>10.1f} stocks")
print(f"  Avg monthly turnover:     {avg_monthly_to*100:>10.1f}%")
print(f"  Avg HHI (concentration):  {avg_hhi:>10.4f}")
print(f"  Est. annual cost drag:    {annual_to_cost*100:>+10.2f}%")
print("")
print("  ── Temporal Decay ──")
print(f"  {'Period':<20} {'Strategy':>10} {'Benchmark':>10} {'Excess':>10}")
print(f"  {'Early (2021-2023)':<20} {early_cagr:>+9.2%} {early_bench:>+9.2%} {(early_cagr-early_bench):>+10.2%}")
print(f"  {'Late (2024-2026)':<20} {late_cagr:>+9.2%} {late_bench:>+9.2%} {(late_cagr-late_bench):>+10.2%}")
print("")
print("  ── Diagnosis ──")
if abs(alpha3) < 0.02 and beta_lv > 0.5:
    print("  >> vol_20d is a LOW-VOL FACTOR, NOT an alpha signal.")
    print(f"  >> Returns are explained by low-vol style exposure (beta_lv={beta_lv:.2f}).")
    print(f"  >> After controlling for low-vol + size, residual alpha ≈ {alpha3:.2%}.")
    print("  >> Low-vol premium itself provides Sharpe improvement, but not excess.")
    print("  >> To generate excess: need to combine with another factor")
    print("     that provides uncorrelated alpha (e.g. reversal, quality).")
elif alpha3 > 0.02:
    print("  >> Residual alpha EXISTS after controlling for style factors.")
    print("  >> The signal has genuine selection ability beyond factor exposure.")
    print("  >> Portfolio construction may be the leak — consider score-weighting.")
else:
    print("  >> Mixed or negative signal after factor controls.")
    print("  >> Check individual factor betas above for diagnosis.")

# ── 10. Cost vs Alpha comparison ─────────────────────────────────────
gross_alpha = alpha3 + annual_to_cost  # what alpha would be without costs
print("")
print("  ── Cost vs Alpha ──")
print(f"  Gross alpha (before costs):  {gross_alpha:>+10.2%}")
print(f"  Annual cost drag:            {annual_to_cost*100:>+10.2f}%")
print(f"  Net alpha (after costs):     {alpha3:>+10.2%}")
print(f"  Alpha / Cost ratio:          {abs(alpha3)/(annual_to_cost*100+1e-8):>.1f}x")
if annual_to_cost * 100 > abs(alpha3) * 0.5:
    print("  >> Costs are a MATERIAL fraction of alpha. Portfolio construction")
    print("     (equal-weight Top20 monthly rebalance) may be too high-turnover.")
    print("     Consider: score-weight, wider top-N, or quarterly rebalance.")

print("")
print("  ── Leak Chain Summary ──")
print(f"  IC (+0.033) → Sharpe (+0.87) → Excess ({early_cagr-early_bench:+.2%}/{late_cagr-late_bench:+.2%} early/late)")
print(f"  Primary leak: {'Low-vol style tilt (not alpha)' if beta_lv > 0.5 else 'Cost + construction' if annual_to_cost*100 > 0.03 else 'Unknown — need deeper decomposition'}")
print("=" * 72)
print("DONE")
