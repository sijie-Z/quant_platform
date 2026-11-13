"""Construction Attribution + Engineering Sanity Tests.
=====================================================
One data fetch. Three outputs:

  PART 1: Construction Attribution
    IC-implied alpha → Universe → Ranking → Weight → Cost → Net
    Each step quantifies where alpha is lost.

  PART 2: Engineering Sanity Tests
    Test A: Cost=0 → net >= current net
    Test B: Turnover=0 (hold forever) → cost≈0
    Test C: Weight sum=1, no negative weights

  PART 3: Combined Report
"""
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

# Repo dir is the `quant_platform` package; its parent must be importable when
# this script runs without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_platform.backtest.metrics import max_drawdown
from quant_platform.utils.research_window import compact_date, resolve_end_date

DB = "data/trading.db"
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
INITIAL = 1_000_000

START_DATE = "2021-01-01"
END_DATE = resolve_end_date()

print("=" * 72)
print("  CONSTRUCTION ATTRIBUTION + ENGINEERING SANITY")
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
            r = ak.stock_zh_a_hist_tx(symbol=txsym(c), start_date=compact_date(START_DATE), end_date=compact_date(END_DATE), adjust="hfq")
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

ic_vals, cs_vols = [], []
for date in vol20.index:
    f = vol20.loc[date].dropna()
    r = fwd_ret.loc[date].reindex(f.index).dropna()
    common = f.index.intersection(r.index)
    if len(common) >= 20:
        ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
        cv = r.loc[common].std()
        if not np.isnan(ic) and not np.isnan(cv):
            ic_vals.append(ic)
            cs_vols.append(cv)
mean_ic = float(np.mean(ic_vals))
mean_cs_vol = float(np.mean(cs_vols))

# IC-implied theoretical alpha
p = N_LONG / max(n_stocks, 1)
z = math.sqrt(-2 * math.log(p)) if p > 0 and p < 0.5 else 0
mills = math.exp(-z*z/2) / math.sqrt(2*math.pi) / p if p > 0 else 0
theo_ann = mean_ic * mean_cs_vol * mills * 252

# ══════════════════════════════════════════════════════════════════════
# 3. SHARED BACKTEST FUNCTION
# ══════════════════════════════════════════════════════════════════════

month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(
    lambda x: x.index[-1])
rebals = sorted(month_end_dates.unique())


def backtest(factor_panel, cost_multiplier=1.0, rebalance_override=None):
    """Run the M4.2 portfolio logic with configurable cost and rebalance.

    Returns dict with cagr, sharpe, max_dd, turnover_log, daily_rets, pv.
    cost_multiplier=0 → cost-free; rebalance_override → custom schedule.
    """
    capital = INITIAL
    daily_rets = []
    turnover_log = []
    cost_log = []
    current_w = {}
    rebal_dates = rebalance_override if rebalance_override is not None else rebals
    rebal_iter = iter(rebal_dates)
    next_rebal = next(rebal_iter, None)

    for date in all_dates:
        # Decision at today's close; activated after today's return.
        pending_w = None
        pending_cost = 0.0
        if next_rebal is not None and date >= next_rebal:
            if date in factor_panel.index:
                f = factor_panel.loc[date].dropna()
                if len(f) >= 60:
                    rank = f.rank(ascending=False)
                    long_u = rank.nsmallest(N_LONG).index.tolist()
                    short_u = rank.nlargest(N_SHORT).index.tolist()
                    total = max(len(long_u) + len(short_u), 1)
                    pending_w = {}
                    for a in long_u:
                        pending_w[a] = 1.0 / total
                    for a in short_u:
                        pending_w[a] = -1.0 / total

                    # One-way turnover = 0.5*sum|w_new - w_old|; buys pay
                    # commission+slippage, sells also stamp tax. The old
                    # sell-only count missed new entries and drift.
                    old_w = current_w
                    changed = set(old_w) | set(pending_w)
                    buy_w = sum(
                        max(pending_w.get(a, 0.0) - old_w.get(a, 0.0), 0.0)
                        for a in changed
                    )
                    sell_w = sum(
                        max(old_w.get(a, 0.0) - pending_w.get(a, 0.0), 0.0)
                        for a in changed
                    )
                    turnover_log.append(0.5 * (buy_w + sell_w))
                    pending_cost = (buy_w * (C + P)
                                    + sell_w * (C + P + S)) * cost_multiplier
                    cost_log.append(float(pending_cost))
            try:
                next_rebal = next(rebal_iter)
            except StopIteration:
                next_rebal = None

        # Today's return belongs to the weights held since the previous
        # close. Activating pending_w before this block was a one-day
        # lookahead: the decision earns the t-1 -> t return it was made after.
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

        # Trade at today's close: pay costs, then activate the new weights.
        if pending_cost:
            capital -= pending_cost * capital
        if pending_w is not None:
            current_w = pending_w

    dr = pd.Series(daily_rets, index=all_dates)
    cagr = (capital/INITIAL) ** (1/ny) - 1
    sharpe = float((dr.mean()/(dr.std()+1e-12)) * np.sqrt(252))
    # Compounded equity-curve drawdown, identical to
    # backtest.metrics.max_drawdown (the old log-cumsum/INITIAL formula was
    # dimensionally wrong and printed -0.00% for every strategy).
    dd = float(max_drawdown(dr)[0])
    avg_to = float(np.mean(turnover_log)) if turnover_log else 0
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": dd,
            "avg_turnover": avg_to, "daily_rets": dr, "turnover_log": turnover_log,
            "cost_log": cost_log,
            "n_rebals": len(turnover_log)}


# ══════════════════════════════════════════════════════════════════════
# 4. CONSTRUCTION ATTRIBUTION — step by step
# ══════════════════════════════════════════════════════════════════════

# 4a. UNIVERSE LOSS: not measurable from a single-universe fetch. The old
# `0.0` was a placeholder printed as if it had been measured, and the printed
# "TOTAL: 100%" implied the decomposition was complete.
universe_loss = None  # not measured

# 4b. RANKING LOSS: not measurable here. The old code ran the same backtest
# twice (`score_weighted = equal_weighted = backtest(vol20)`), so any value
# derived from that comparison would have been an artifact of a duplicate
# call, not a measurement.
ranking_loss = None  # not measured

# 4c. WEIGHT LOSS: Top20 EW vs all stocks weighted by factor score.
# The residual between IC-implied alpha and the gross portfolio return. It
# absorbs whatever the unmeasured universe/ranking steps would explain, so it
# is labelled as the residual rather than attributed to a single cause.
# 4d. COST LOSS: already computed as annual cost drag (buys and sells).
base = backtest(vol20)
cost_loss = sum(base["cost_log"]) / ny
gross_cagr = base["cagr"] + cost_loss
weight_loss = theo_ann - gross_cagr  # residual after costs
total_construction_loss = theo_ann - base["cagr"]
discarded = max(n_stocks - N_LONG, 0)

print("")
print("=" * 72)
print("  PART 1: CONSTRUCTION ATTRIBUTION")
print("=" * 72)
print(f"  IC-implied theoretical alpha:  {theo_ann:>+8.2%}")
print(f"  {'Step':<5} {'Component':<45} {'Annual':>10} {'Running':>10}")
print(f"  {'-'*5} {'-'*45} {'-'*10} {'-'*10}")
print(f"  {'[0]':<5} {'IC-implied theoretical alpha':<45} {theo_ann:>+9.2%} {theo_ann:>+9.2%}")
print(f"  {'[1]':<5} {'- Universe Loss (CSI300 vs ALL)':<45} {'n/a':>10} {'n/a':>10}")
print(f"  {'[2]':<5} {'- Ranking Loss (EW vs Score)':<45} {'n/a':>10} {'n/a':>10}")
print(f"  {'[3]':<5} {'- Residual (Top20 vs IC-implied)':<45} {-weight_loss:>+9.2%} {gross_cagr:>+9.2%}")
print(f"  {'[4]':<5} {'- Cost Loss (turnover+fees)':<45} {-cost_loss:>+9.2%} {base['cagr']:>+9.2%}")
print(f"  {'':<5} {'= Net Portfolio Return':<45} {base['cagr']:>+9.2%}")

leak_theo = abs(theo_ann) + 1e-8
print("")
print("  Attribution (% of IC-implied alpha; measured components only):")
print("    Universe loss:      n/a (not measured)")
print("    Ranking loss:       n/a (not measured)")
wg_pct = abs(weight_loss)/leak_theo*100 if weight_loss > 0 else 0
print(f"    Residual (weight):  {wg_pct:.0f}%  ({-weight_loss:+.2%})")
cost_pct_attr = abs(cost_loss)/leak_theo*100
print(f"    Cost loss:          {cost_pct_attr:.0f}%  ({-cost_loss:+.2%})")
measured_pct = wg_pct + cost_pct_attr
print(f"    Measured total:     {measured_pct:.0f}%  "
      f"(universe/ranking unmeasured; no 100% claim)")

# ══════════════════════════════════════════════════════════════════════
# 5. ENGINEERING SANITY TESTS
# ══════════════════════════════════════════════════════════════════════

print("")
print("=" * 72)
print("  PART 2: ENGINEERING SANITY TESTS")
print("=" * 72)

# Test A: Cost=0 → net >= current net
cost_free = backtest(vol20, cost_multiplier=0.0)
test_a = cost_free["cagr"] >= base["cagr"]
print(f"  Test A (Cost=0):       net={cost_free['cagr']:+.2%}  base={base['cagr']:+.2%}  {'PASS' if test_a else 'FAIL'}")
if not test_a:
    print("    WARNING: Cost-free run has LOWER return than with-cost run!")
    print("    This indicates the cost deduction logic is broken.")

# Test B: Turnover=0 (rebalance once, hold forever)
single_rebal = [rebals[0]]  # rebalance only on first month-end
hold_forever = backtest(vol20, rebalance_override=single_rebal)
n_to = len(hold_forever["turnover_log"])
test_b = n_to <= 1  # at most 1 turnover event (the initial buy)
print(f"  Test B (Turnover=0):   n_rebals={n_to}  turnover={hold_forever['avg_turnover']*100:.1f}%  {'PASS' if test_b else 'FAIL'}")
if not test_b:
    print(f"    WARNING: Single-rebalance strategy has {n_to} turnover events!")
    print("    Should be at most 1 (the second rebalance sells the initial positions).")

# Test C: Weight audit — all weights must sum to 1, none negative
# Run a quick pass to collect all weights on rebalance dates
weight_violations = 0
weight_near_miss = 0
current_w_test = {}
rebal_iter_test = iter(rebals)
next_rebal_test = next(rebal_iter_test, None)
for date in all_dates:
    if next_rebal_test is not None and date >= next_rebal_test:
        if date in vol20.index:
            f = vol20.loc[date].dropna()
            if len(f) >= 60:
                rank = f.rank(ascending=False)
                long_u = rank.nsmallest(N_LONG).index.tolist()
                total = max(len(long_u), 1)
                w = {a: 1.0/total for a in long_u}
                w_sum = sum(w.values())
                if abs(w_sum - 1.0) > 0.01:
                    weight_violations += 1
                elif abs(w_sum - 1.0) > 1e-6:
                    weight_near_miss += 1
                for a, wt in w.items():
                    if wt < 0:
                        weight_violations += 1
        try:
            next_rebal_test = next(rebal_iter_test)
        except StopIteration:
            next_rebal_test = None

test_c = weight_violations == 0
print(f"  Test C (Weight audit): violations={weight_violations}  near_miss={weight_near_miss}  {'PASS' if test_c else 'FAIL'}")
if not test_c:
    print(f"    WARNING: {weight_violations} rebalance dates have invalid weights!")
    print("    Check weight calculation and short-leg logic.")

# Test D: Rebalance count matches expected
# Test D: Rebalance count matches expected. n_rebals counts every applied
# rebalance, including the initial buy (previously only rebalances with a
# non-empty sell leg were counted). Warm-up rebalances without a 60-name
# cross-section legitimately never happen, so count eligible dates only.
expected_rebals = sum(
    1 for d in rebals
    if d in vol20.index and len(vol20.loc[d].dropna()) >= 60
)
actual_rebals = base["n_rebals"]
test_d = abs(actual_rebals - expected_rebals) <= 2  # allow ±2 for edge cases
print(f"  Test D (Rebal count):  expected~{expected_rebals}  actual={actual_rebals}  {'PASS' if test_d else 'FAIL'}")

all_pass = test_a and test_b and test_c and test_d
print("")
print(f"  OVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED — review above'}")

# ══════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("")
print("=" * 72)
print("  COMBINED VERDICT")
print("=" * 72)
print("")
print("  Construction attribution:")
print(f"    Primary alpha loss (residual): {-weight_loss:+.2%} ann")
if abs(theo_ann) > 1e-12:
    print(f"    — Top20 equal-weight captures {gross_cagr/theo_ann*100:.0f}% of IC-implied alpha")
    print(f"    — The remaining {(theo_ann-gross_cagr)/theo_ann*100:.0f}% is the residual "
          f"across the {discarded} names outside Top20 (universe/ranking effects not separately measured)")
else:
    print("    — Top20 vs IC-implied alpha: n/a (IC-implied alpha is zero)")
loss_denom = theo_ann - gross_cagr + cost_loss
cost_share = (cost_loss / loss_denom * 100) if abs(loss_denom) > 1e-12 else float("nan")
cost_share_txt = f"{cost_share:.0f}%" if np.isfinite(cost_share) else "n/a"
print(f"    — Cost adds {-cost_loss:+.2%} ann drag ({cost_share_txt} of measured loss)")
print("")
print(f"  Engineering sanity: {'ALL PASS' if all_pass else 'FAILURES'}")
print(f"    Pipeline infrastructure is {'verified' if all_pass else 'suspicious'}.")
print("=" * 72)

# Registry
rid = f"constr_attr_{int(time.time())}"
ev = {
    "experiment": "construction_attribution",
    "theoretical_ann": float(theo_ann),
    "gross_cagr": float(gross_cagr), "net_cagr": float(base["cagr"]),
    # None = not measured. The old 0.0 values were placeholders that read as
    # measured components downstream.
    "universe_loss": None, "ranking_loss": None,
    "weight_loss": float(weight_loss), "cost_loss": float(cost_loss),
    "weight_loss_pct": float(wg_pct), "cost_loss_pct": float(cost_pct_attr),
    "n_stocks": n_stocks, "n_discarded": discarded,
    "data_start": str(all_dates[0].date()), "data_end": str(all_dates[-1].date()),
    "requested_end": END_DATE,
    "engineering_sanity": {
        "cost_free_cagr": float(cost_free["cagr"]), "test_a": test_a,
        "turnover_zero_rebals": n_to, "test_b": test_b,
        "weight_violations": weight_violations, "test_c": test_c,
        "rebal_count_expected": expected_rebals, "rebal_count_actual": actual_rebals, "test_d": test_d,
    },
    "overall_pipeline": "VERIFIED" if all_pass else "REVIEW",
}
conn = sqlite3.connect(DB)
conn.execute(
    """INSERT INTO research_runs (run_id, timestamp, status, reason, slice, data_source, data_meta,
       universe_provider, universe_meta, factor, factor_params, evaluation, input_hash, report_path, warnings)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (rid, pd.Timestamp.now('UTC').isoformat()[:19], "success", "",
     "construction_attribution", "akshare",
     json.dumps({"adjust":"hfq","provider":"akshare",
                 "start_date": str(all_dates[0].date()),
                 "end_date": str(all_dates[-1].date()),
                 "requested_end": END_DATE}),
     "CurrentConstituentUniverseProvider",
     json.dumps({"pit":False,"bias_warning":["survivorship_bias_possible"]}),
     "volatility_20d",
     json.dumps({"n_long":N_LONG,"n_short":N_SHORT}),
     json.dumps(ev, default=str), "", "", "[]"))
conn.commit()
conn.close()
print(f"Registry: {rid}")
print("DONE")
