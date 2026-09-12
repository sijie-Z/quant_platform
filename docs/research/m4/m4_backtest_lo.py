"""M4.2 Portfolio Backtest — Long-Only variant
=====================================================
Hypothesis: The negative Sharpe in M4.1 was driven by the short leg.
If long-only (eliminating short-side costs + negative carry) improves
risk-adjusted returns, the factor has directional value even if the
long-short construction fails.

This is an INDEPENDENT experiment, NOT a fix of M4.1.

Variables controlled (identical to M4.1):
  - Factor: volatility_20d (same: -rets.rolling(20).std())
  - Universe: CSI300 current constituents, AkShare hfq
  - Rebalance: monthly (last trading day of each month)
  - Weighting: equal weight
  - Cost model: commission 3bp + stamp tax 10bp (sell only) + slippage 5bp
  - Capital: 1,000,000 initial
  - Period: 2021-01 through 2026-05

Single variable change vs M4.1:
  - Short leg REMOVED (N_SHORT = 0, was 20)
  - Long leg kept at N_LONG = 20

Output:
  - CAGR, Sharpe, Sortino, MaxDD, Win Rate, Turnover
  - Yearly returns, Rolling 12m Sharpe
  - Comparison vs M4.1 baseline & CSI300 EW benchmark
  - Result recorded in Registry as independent experiment
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "D:/Desktop")
import akshare as ak
import numpy as np
import pandas as pd

print("=" * 72)
print("  M4.2 — Long-Only Portfolio Backtest: volatility_20d")
print("  Hypothesis: short leg is the primary drag on M4.1 performance")
print("=" * 72, flush=True)

DB = Path(__file__).resolve().parent / "data" / "trading.db"
EXPERIMENT_ID = f"m4.2_lo_{int(time.time())}"

# ── 1. Data (SAME as M4.1) ──────────────────────────────────────────
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
            r = ak.stock_zh_a_hist_tx(
                symbol=txsym(c), start_date="20210101", end_date="20260523",
                adjust="hfq",
            )
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
print(f"Price panel: {prices.shape} (fetch {int(time.time() - start)}s)", flush=True)

# ── 2. Factor (SAME as M4.1) ────────────────────────────────────────
vol20 = -rets.rolling(20).std()
print(f"Factor: {vol20.shape}", flush=True)

# ── 3. Portfolio Backtest (SAME logic, N_SHORT=0) ────────────────────
C, S, P = 0.0003, 0.001, 0.0005   # commission, stamp tax, slippage
N_LONG = 20
N_SHORT = 0                        # ← ONLY variable changed vs M4.1
initial = 1_000_000
capital = initial

month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(
    lambda x: x.index[-1]
)
rebals = sorted(month_end_dates.unique())
all_dates = prices.index.sort_values()

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
                short_u = rank.nlargest(N_SHORT).index.tolist()  # empty when N_SHORT=0

                # Turnover cost on sold positions
                if current_w:
                    old_set = set(current_w.keys())
                    new_set = set(long_u + short_u)
                    sold = old_set - new_set
                    if sold and len(current_w) > 0:
                        to = sum(abs(current_w.get(a, 0)) for a in sold)
                        turnover_log.append(float(to))
                        # Cost on sold leg: commission + stamp tax + slippage
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
ny = (pv.index[-1] - pv.index[0]).days / 365.25

# Benchmark: CSI300 equal-weight
bpv = (1 + fwd_ret.mean(axis=1).reindex(all_dates).fillna(0)).cumprod()

# ── 4. Metrics ──────────────────────────────────────────────────────
cagr = (pv.iloc[-1] / initial) ** (1 / ny) - 1
ann_vol = float(dr.std() * np.sqrt(252))
sharpe = float((dr.mean() / (dr.std() + 1e-12)) * np.sqrt(252))
sortino = float((dr.mean() / (dr[dr < 0].std() + 1e-12)) * np.sqrt(252))
dd = (pv - pv.cummax()) / pv.cummax()
max_dd = float(dd.min())
win = float((dr > 0).mean())
monthly_r = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly_w = float((monthly_r > 0).mean())
avg_to = float(np.mean(turnover_log)) if turnover_log else 0
bench_c = (bpv.iloc[-1] / bpv.iloc[0]) ** (1 / ny) - 1

# Yearly returns
yret = {}
for yr, g in dr.groupby(dr.index.year):
    yret[yr] = float((1 + g).prod() - 1)

roll12 = dr.rolling(252).mean() / (dr.rolling(252).std() + 1e-12) * np.sqrt(252)
roll12 = roll12.dropna()

# ── 5. Print results ────────────────────────────────────────────────
print("")
print("=" * 72)
print("  M4.2 RESULTS — Long-Only (N_LONG=20, N_SHORT=0)")
print("=" * 72)
print(f"  Total return:  {pv.iloc[-1]/initial - 1:+.2%}")
print(f"  CAGR:          {cagr:+.2%}")
print(f"  Ann Vol:       {ann_vol:.2%}")
print(f"  Sharpe:        {sharpe:+.2f}")
print(f"  Sortino:       {sortino:+.2f}")
print(f"  Max DD:        {max_dd:+.2%}")
print(f"  Daily win:     {win:.1%}  Monthly win: {monthly_w:.1%}")
print(f"  Avg turnover:  {avg_to*100:.1f}%")
print(f"  Benchmark CAGR:{bench_c:+.2%} (CSI300 EW)")
print(f"  Excess vs BM:  {cagr - bench_c:+.2%}")
print(f"  N rebalances:  {len(turnover_log)}")
print(f"  Cost model:    C={C*10000:.0f}bp S={S*10000:.0f}bp Slip={P*10000:.0f}bp")
print("-" * 72)
print("  Yearly:")
for yr in sorted(yret):
    print(f"    {yr}: {yret[yr]:+.2%}")
if len(roll12) > 0:
    print(f"  Rolling 12m Sharpe: min={roll12.min():+.2f} max={roll12.max():+.2f} latest={roll12.iloc[-1]:+.2f}")
print("=" * 72, flush=True)

# ── 6. Store in Registry (independent experiment) ────────────────────
conn = sqlite3.connect(str(DB))
rid = EXPERIMENT_ID

ev = {
    "experiment": "m4.2_long_only",
    "hypothesis": "Short leg is the primary drag on M4.1 long-short performance. Long-only should improve risk-adjusted returns if factor has directional value.",
    "variable_changed": "N_SHORT: 20 → 0",
    "variables_controlled": [
        "factor=volatility_20d", "universe=CSI300_current_constituents",
        "data=akshare_hfq", "rebalance=monthly", "weighting=equal_weight",
        "cost_model=C3bp+S10bp+Slip5bp", "N_LONG=20",
        "period=2021-01_to_2026-05", "initial_capital=1_000_000",
    ],
    "factor": "volatility_20d",
    "validation_type": "m4.2_portfolio_backtest_long_only",
    "cagr": float(cagr), "sharpe": sharpe, "sortino": sortino,
    "max_dd": max_dd, "ann_vol": ann_vol,
    "win_rate_daily": win, "win_rate_monthly": monthly_w,
    "avg_turnover": float(avg_to), "n_rebalances": len(turnover_log),
    "benchmark_cagr": float(bench_c), "excess_vs_benchmark": float(cagr - bench_c),
    "yearly_returns": yret,
    "rolling_12m_sharpe": {
        "min": float(roll12.min()) if len(roll12) > 0 else None,
        "max": float(roll12.max()) if len(roll12) > 0 else None,
        "latest": float(roll12.iloc[-1]) if len(roll12) > 0 else None,
    },
    "cost_model": {"commission": C, "stamp_tax": S, "slippage": P},
    "setup": {"n_long": N_LONG, "n_short": N_SHORT},
    "dsr": "insufficient_trials", "bh_fdr": "n/a",
    "baseline_m4_1_sharpe": -0.96,  # reference
}
conn.execute(
    """INSERT INTO research_runs
       (run_id, timestamp, status, reason, slice, data_source, data_meta,
        universe_provider, universe_meta, factor, factor_params,
        evaluation, input_hash, report_path, warnings)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (
        rid,
        pd.Timestamp.utcnow().isoformat()[:19],
        "success",
        "", "m4.2_long_only_backtest",
        "akshare",
        json.dumps({"adjust": "hfq", "provider": "akshare"}, default=str),
        "CurrentConstituentUniverseProvider",
        json.dumps({"pit": False, "bias_warning": ["survivorship_bias_possible"]}, default=str),
        "volatility_20d",
        json.dumps({"n_long": N_LONG, "n_short": N_SHORT, "experiment": "m4.2_long_only"}),
        json.dumps(ev, default=str),
        "", "", "[]",
    ),
)
conn.commit()
conn.close()

print(f"\nRegistry: {rid}")
print("M4.2 experiment recorded. Baseline M4.1 Sharpe was -0.96.")
print("DONE")
