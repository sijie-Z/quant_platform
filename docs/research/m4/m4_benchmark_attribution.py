"""M4.2 Benchmark Attribution — risk-return decomposition vs CSI300 EW.
=============================================================
Answers: Is M4.2 Long-Only providing a genuinely better risk-return
structure, or just riding market beta?

Uses the same price data and methodology as m4_backtest_lo.py.
"""
import os
import sys
import time

sys.path.insert(0, "D:/Desktop")
import akshare as ak
import numpy as np
import pandas as pd

print("=" * 72)
print("  M4.2 — Benchmark Attribution: vol_20d Long-Only vs CSI300 EW")
print("=" * 72, flush=True)

# ── 1. Data (same as M4.1/M4.2) ─────────────────────────────────────
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
                r = r.rename(columns={"date": "date", "close": "close"}) if "close" in r.columns else r
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

# ── 2. Factor ───────────────────────────────────────────────────────
vol20 = -rets.rolling(20).std()

# ── 3. Portfolio (M4.2 config, identical to m4_backtest_lo.py) ──────
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
initial = 1_000_000
capital = initial

month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(lambda x: x.index[-1])
rebals = sorted(month_end_dates.unique())
all_dates = prices.index.sort_values()

daily_rets = []
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
                        to = sum(abs(current_w.get(a, 0)) for a in sold)
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

strategy_dr = pd.Series(daily_rets, index=all_dates)
# Benchmark: CSI300 equal-weight daily return
bench_dr = fwd_ret.mean(axis=1).reindex(all_dates).dropna()
common = strategy_dr.index.intersection(bench_dr.index)
strategy_dr = strategy_dr.loc[common]
bench_dr = bench_dr.loc[common]

ny = (common[-1] - common[0]).days / 365.25

# ── 4. Risk-Return Metrics ───────────────────────────────────────────
def metrics(name, dr):
    cagr = (np.prod(1 + dr)) ** (1 / ny) - 1
    ann_vol = float(dr.std() * np.sqrt(252))
    sharpe = float((dr.mean() / (dr.std() + 1e-12)) * np.sqrt(252))
    sortino = float((dr.mean() / (dr[dr < 0].std() + 1e-12)) * np.sqrt(252))
    pv = (1 + dr).cumprod()
    dd = (pv - pv.cummax()) / pv.cummax()
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else float('inf')
    win = float((dr > 0).mean())
    # Monthly
    m = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mwin = float((m > 0).mean())
    return dict(
        name=name, cagr=cagr, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
        max_dd=max_dd, calmar=calmar, win_rate=win, monthly_win=mwin,
        total_return=float(np.prod(1 + dr) - 1),
    )

s = metrics("vol_20d_LO", strategy_dr)
b = metrics("CSI300_EW", bench_dr)

# ── 5. Beta & Alpha (CAPM regression) ────────────────────────────────
excess_strat = strategy_dr - 0  # no risk-free proxy in daily
excess_bench = bench_dr - 0
cov = np.cov(excess_strat, excess_bench)
beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
# Annualized alpha via CAPM
alpha_daily = strategy_dr.mean() - beta * bench_dr.mean()
alpha_ann = alpha_daily * 252
# Information Ratio
tracking_error = (strategy_dr - bench_dr).std() * np.sqrt(252)
ir = (s["cagr"] - b["cagr"]) / tracking_error if tracking_error > 0 else 0

# ── 6. Print ─────────────────────────────────────────────────────────
print("")
print("=" * 72)
print("  BENCHMARK ATTRIBUTION")
print("=" * 72)
print(f"  {'Metric':<30} {'vol_20d LO':>12} {'CSI300 EW':>12} {'Δ':>12}")
print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
for k, label in [
    ("cagr", "CAGR"),
    ("ann_vol", "Ann Volatility"),
    ("sharpe", "Sharpe Ratio"),
    ("sortino", "Sortino Ratio"),
    ("max_dd", "Max Drawdown"),
    ("calmar", "Calmar Ratio"),
    ("win_rate", "Daily Win Rate"),
    ("monthly_win", "Monthly Win Rate"),
    ("total_return", "Total Return"),
]:
    v1 = s[k]
    v2 = b[k]
    if k in ("cagr", "ann_vol", "max_dd", "win_rate", "monthly_win", "total_return"):
        d = v1 - v2
        print(f"  {label:<30} {v1:>+11.2%} {v2:>+11.2%} {d:>+12.2%}")
    else:
        d = v1 - v2
        print(f"  {label:<30} {v1:>+11.2f} {v2:>+11.2f} {d:>+12.2f}")

print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
print(f"  {'Beta (CAPM)':<30} {beta:>12.2f}")
print(f"  {'Alpha (annualized)':<30} {alpha_ann:>+11.2%}")
print(f"  {'Information Ratio':<30} {ir:>+11.2f}")
print(f"  {'Tracking Error (ann)':<30} {tracking_error:>+11.2%}")
print("=" * 72)

# ── 7. Yearly breakdown ──────────────────────────────────────────────
print("")
print("  YEARLY BREAKDOWN")
print(f"  {'Year':<8} {'vol_20d LO':>12} {'CSI300 EW':>12} {'Excess':>12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
for yr in sorted(set(strategy_dr.index.year)):
    sy = strategy_dr[strategy_dr.index.year == yr]
    by = bench_dr[bench_dr.index.year == yr]
    if len(sy) > 5:
        sr = (1 + sy).prod() - 1
        br = (1 + by).prod() - 1
        print(f"  {yr:<8} {sr:>+11.2%} {br:>+11.2%} {(sr-br):>+12.2%}")

print("")
print("  ROLLING 12-MONTH SHARPE")
roll_s = strategy_dr.rolling(252).mean() / (strategy_dr.rolling(252).std() + 1e-12) * np.sqrt(252)
roll_b = bench_dr.rolling(252).mean() / (bench_dr.rolling(252).std() + 1e-12) * np.sqrt(252)
roll_s = roll_s.dropna()
roll_b = roll_b.dropna()
cr = roll_s.index.intersection(roll_b.index)
print(f"  {'Metric':<30} {'vol_20d LO':>12} {'CSI300 EW':>12}")
print(f"  {'-'*30} {'-'*12} {'-'*12}")
print(f"  {'Rolling Sharpe Min':<30} {roll_s.loc[cr].min():>+11.2f} {roll_b.loc[cr].min():>+11.2f}")
print(f"  {'Rolling Sharpe Max':<30} {roll_s.loc[cr].max():>+11.2f} {roll_b.loc[cr].max():>+11.2f}")
print(f"  {'Rolling Sharpe Mean':<30} {roll_s.loc[cr].mean():>+11.2f} {roll_b.loc[cr].mean():>+11.2f}")
print(f"  {'Rolling Sharpe Latest':<30} {roll_s.iloc[-1]:>+11.2f} {roll_b.iloc[-1]:>+11.2f}")
print("=" * 72)

# ── 8. Decision guide ────────────────────────────────────────────────
print("")
print("  DECISION GUIDE")
if s["sharpe"] > b["sharpe"] and s["max_dd"] > b["max_dd"] and abs(alpha_ann) < 0.01:
    print("  → Risk-adjusted returns BETTER than benchmark.")
    print("  → Strategy provides superior risk structure.")
    print("  → Recommend: M4.3 Walk-Forward OOS")
elif abs(beta - 1.0) < 0.2 and alpha_ann < 0:
    print("  → Strategy is largely a MARKET BETA proxy with negative alpha.")
    print("  → Returns are driven by broad market, not factor selection.")
    print("  → Recommend: M4.4 Multi-Factor to add genuine alpha")
elif s["sharpe"] > b["sharpe"] and abs(s["max_dd"]) < abs(b["max_dd"]):
    print("  → Sharper ratio higher, drawdown lower than benchmark.")
    print("  → Low-volatility bias providing genuine defensive advantage.")
    print("  → Recommend: M4.3 Walk-Forward OOS")
else:
    print("  → Mixed signals. Check specific metrics above.")
    print("  → Recommend: M4.4 Multi-Factor to address deficiencies")

print("DONE")
