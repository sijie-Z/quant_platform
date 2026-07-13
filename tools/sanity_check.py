"""Pipeline Sanity Check — verify the evaluation chain is not lying.
=============================================================
Three experiments against the M4.2 backtest + leakage audit pipeline:

  Test A: RANDOM FACTOR
    Factor = random Gaussian noise per stock per day.
    Expected: IC~~0, Sharpe~~0, excess~~0, leakage pattern flat.

  Test B: ORACLE FACTOR (perfect foresight)
    Factor = tomorrow's actual return.
    Expected: IC~~1, Sharpe >> 0, excess >> 0, leakage->zero.

  Test C: NOISE MONTE CARLO (100 draws)
    Factor = random noise (fresh seed each trial).
    Expected: IC distributed around 0, Sharpe around 0.
    If median Sharpe > 0.2 -> systematic upward bias.

All three run the IDENTICAL portfolio construction, cost model,
and metrics as M4.2. Only the factor definition changes.
"""
import sys, os, time, json, sqlite3, math
sys.path.insert(0, "D:/Desktop")
import akshare as ak
import pandas as pd
import numpy as np

DB = "data/trading.db"
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
INITIAL = 1_000_000
RNG = np.random.default_rng(42)

print("=" * 72)
print("  PIPELINE SANITY CHECK")
print("=" * 72, flush=True)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA (once)
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

prices = pd.DataFrame(frames); prices.index.name = "date"
rets = prices.pct_change(fill_method=None)
fwd_ret = rets.shift(-1)
all_dates = prices.index.sort_values()
ny = (all_dates[-1] - all_dates[0]).days / 365.25
print(f"Price panel: {prices.shape} (fetch {int(time.time()-t0)}s)", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 2. SHARED BACKTEST FUNCTION
# ══════════════════════════════════════════════════════════════════════

month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(
    lambda x: x.index[-1])
rebals = sorted(month_end_dates.unique())


def run_factor(factor_panel: pd.DataFrame) -> dict:
    """Run the EXACT M4.2 portfolio logic with a given factor panel.

    Returns dict with cagr, sharpe, max_dd, turnover, style_cagr, ic_mean.
    """
    capital = INITIAL
    daily_rets = []
    turnover_log = []
    current_w = {}
    rebal_iter = iter(rebals)
    next_rebal = next(rebal_iter, None)

    # Style benchmark (same loop, no costs)
    style_capital = INITIAL
    style_daily_rets = []
    style_current_w = {}
    style_rebal_iter = iter(rebals)
    style_next_rebal = next(style_rebal_iter, None)

    for date in all_dates:
        # ── Strategy (with costs) ──
        if next_rebal is not None and date >= next_rebal:
            if date in factor_panel.index:
                f = factor_panel.loc[date].dropna()
                if len(f) >= 60:
                    rank = f.rank(ascending=False)
                    long_u = rank.nsmallest(N_LONG).index.tolist()
                    if current_w:
                        old_set = set(current_w.keys())
                        new_set = set(long_u)
                        sold = old_set - new_set
                        if sold and len(current_w) > 0:
                            to = sum(abs(current_w.get(a,0)) for a in sold)
                            turnover_log.append(float(to))
                            capital -= to * (C + P + S) * capital
                    total = max(len(long_u), 1)
                    current_w = {}
                    for a in long_u:
                        current_w[a] = 1.0 / total
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

        # ── Style benchmark (no costs) ──
        if style_next_rebal is not None and date >= style_next_rebal:
            if date in factor_panel.index:
                f = factor_panel.loc[date].dropna()
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

    dr = pd.Series(daily_rets, index=all_dates)
    style_dr = pd.Series(style_daily_rets, index=all_dates)

    # IC
    ic_vals = []
    for date in factor_panel.index:
        f = factor_panel.loc[date].dropna()
        r = fwd_ret.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) >= 20:
            ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
            if not np.isnan(ic):
                ic_vals.append(ic)

    # Metrics
    cagr = (capital/INITIAL) ** (1/ny) - 1
    sharpe = float((dr.mean()/(dr.std()+1e-12)) * np.sqrt(252))
    dd = float(((dr.cumsum() + np.log(INITIAL) - (dr.cumsum() + np.log(INITIAL)).cummax()).min() / INITIAL))
    avg_to = float(np.mean(turnover_log)) if turnover_log else 0

    style_cagr = (style_capital/INITIAL) ** (1/ny) - 1
    style_excess = cagr - style_cagr
    annual_cost = sum(t*(C+P+S) for t in turnover_log)/ny if turnover_log else 0

    return {
        "cagr": cagr, "sharpe": sharpe, "max_dd": dd,
        "avg_turnover": avg_to, "style_cagr": style_cagr,
        "style_excess": style_excess, "cost_drag": annual_cost,
        "ic_mean": float(np.mean(ic_vals)) if ic_vals else float("nan"),
        "ic_std": float(np.std(ic_vals)) if ic_vals else float("nan"),
        "n_ic": len(ic_vals),
    }


# ══════════════════════════════════════════════════════════════════════
# 3. TEST A: RANDOM FACTOR
# ══════════════════════════════════════════════════════════════════════
print("")
print("  TEST A — Random Gaussian Factor (null hypothesis)")
print("  Expected: IC~~0, Sharpe~~0, Style Ex~~0")
random_factor = pd.DataFrame(
    RNG.normal(0, 1, (len(rets.index), len(rets.columns))),
    index=rets.index, columns=rets.columns,
)
r = run_factor(random_factor)
print(f"  IC mean:     {r['ic_mean']:+.4f}  (expect ~~0)")
print(f"  Sharpe:      {r['sharpe']:+.2f}   (expect ~~0)")
print(f"  CAGR:        {r['cagr']:+.2%}")
print(f"  Style Ex:    {r['style_excess']:+.2%} (expect ~~0)")
print(f"  Cost drag:   {r['cost_drag']:+.2%}")
result_a = "PASS" if abs(r["ic_mean"]) < 0.02 and abs(r["sharpe"]) < 0.5 and abs(r["style_excess"]) < 0.02 else "REVIEW"
print(f"  -> {result_a}")

# ══════════════════════════════════════════════════════════════════════
# 4. TEST B: ORACLE FACTOR (perfect foresight)
# ══════════════════════════════════════════════════════════════════════
print("")
print("  TEST B — Oracle Factor (tomorrow's return)")
print("  Expected: IC~~1, Sharpe >> 0, Style Ex >> 0")
oracle_factor = fwd_ret.copy()
r2 = run_factor(oracle_factor)
print(f"  IC mean:     {r2['ic_mean']:+.4f}  (expect ->1)")
print(f"  Sharpe:      {r2['sharpe']:+.2f}   (expect >>1)")
print(f"  CAGR:        {r2['cagr']:+.2%}")
print(f"  Max DD:      {r2['max_dd']:+.2%}")
print(f"  Style Ex:    {r2['style_excess']:+.2%} (expect >>0)")
print(f"  Cost drag:   {r2['cost_drag']:+.2%}")
result_b = "PASS" if r2["ic_mean"] > 0.5 and r2["sharpe"] > 2 else "REVIEW"
print(f"  -> {result_b}")

# ══════════════════════════════════════════════════════════════════════
# 5. TEST C: NOISE MONTE CARLO (100 draws)
# ══════════════════════════════════════════════════════════════════════
print("")
print("  TEST C — Noise Monte Carlo (100 draws)")
print("  Expected: IC ~ N(0, small), median Sharpe < 0.2")

N_TRIALS = 100
mc_ics, mc_sharpes, mc_style_ex = [], [], []
for trial in range(N_TRIALS):
    noise = pd.DataFrame(
        RNG.normal(0, 1, (len(rets.index), len(rets.columns))),
        index=rets.index, columns=rets.columns,
    )
    r3 = run_factor(noise)
    mc_ics.append(r3["ic_mean"])
    mc_sharpes.append(r3["sharpe"])
    mc_style_ex.append(r3["style_excess"])
    if (trial+1) % 25 == 0:
        print(f"    {trial+1}/{N_TRIALS} done", flush=True)

mc_ics = np.array(mc_ics)
mc_sharpes = np.array(mc_sharpes)
mc_style_ex = np.array(mc_style_ex)

print(f"  IC:        mean={np.mean(mc_ics):+.4f}  std={np.std(mc_ics):.4f}  p5={np.percentile(mc_ics,5):+.4f}  p95={np.percentile(mc_ics,95):+.4f}")
print(f"  Sharpe:    mean={np.mean(mc_sharpes):+.3f}  std={np.std(mc_sharpes):.3f}  median={np.median(mc_sharpes):+.3f}  max={np.max(mc_sharpes):+.3f}")
print(f"  Style Ex:  mean={np.mean(mc_style_ex):+.4f}  std={np.std(mc_style_ex):+.4f}")
print(f"  Sharpe>0:  {np.mean(mc_sharpes > 0)*100:.0f}%")
print(f"  Sharpe>0.5:{np.mean(mc_sharpes > 0.5)*100:.0f}%")
result_c = "PASS" if abs(np.mean(mc_ics)) < 0.01 and np.median(mc_sharpes) < 0.2 else "REVIEW"
print(f"  -> {result_c}")

# ══════════════════════════════════════════════════════════════════════
# 6. SUMMARY & REGISTRY
# ══════════════════════════════════════════════════════════════════════
print("")
print("=" * 72)
print("  SANITY CHECK SUMMARY")
print("=" * 72)
all_pass = all(r == "PASS" for r in [result_a, result_b, result_c])
print(f"  Test A (Random):    {result_a}")
print(f"  Test B (Oracle):    {result_b}")
print(f"  Test C (MC N={N_TRIALS}): {result_c}")
print(f"  {'─'*40}")
print(f"  OVERALL: {'PIPELINE VERIFIED' if all_pass else 'INVESTIGATE FIRST'}")
print("=" * 72)

# Registry
rid = f"sanity_{int(time.time())}"
ev = {
    "experiment": "pipeline_sanity_check",
    "test_a_random": {"ic_mean": r["ic_mean"], "sharpe": r["sharpe"], "style_excess": r["style_excess"], "result": result_a},
    "test_b_oracle": {"ic_mean": r2["ic_mean"], "sharpe": r2["sharpe"], "style_excess": r2["style_excess"], "result": result_b},
    "test_c_mc": {"n_trials": N_TRIALS, "mean_ic": float(np.mean(mc_ics)), "median_sharpe": float(np.median(mc_sharpes)), "sharpe_gt_0_5_pct": float(np.mean(mc_sharpes > 0.5)*100), "result": result_c},
    "overall": "PASS" if all_pass else "INVESTIGATE",
}
conn = sqlite3.connect(DB)
conn.execute(
    """INSERT INTO research_runs (run_id, timestamp, status, reason, slice, data_source, data_meta,
       universe_provider, universe_meta, factor, factor_params, evaluation, input_hash, report_path, warnings)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (rid, pd.Timestamp.now('UTC').isoformat()[:19], "success" if all_pass else "failed",
     "", "pipeline_sanity_check", "akshare",
     json.dumps({"adjust":"hfq","provider":"akshare"}),
     "CurrentConstituentUniverseProvider",
     json.dumps({"pit":False,"bias_warning":["survivorship_bias_possible"]}),
     "sanity_check", json.dumps({"tests":["random","oracle","mc_100"]}),
     json.dumps(ev, default=str), "", "", "[]"))
conn.commit(); conn.close()
print(f"Registry: {rid}")
print("DONE" if all_pass else "DONE (with issues — review above)")
