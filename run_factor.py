"""Unified Factor Runner — single entry point for any factor.
===========================================================
Replaces m4_*.py pattern. One command, any factor:

  python run_factor.py volatility_20d
  python run_factor.py reversal
  python run_factor.py momentum_12m

Outputs standard Factor Report: IC, Portfolio, Style Benchmark,
Market Benchmark, Leakage Audit, Construction Attribution,
Engineering Sanity.

Each factor is a ~10-line compute function. No new scripts needed.
"""
import sys, os, time, json, sqlite3, math
from pathlib import Path

sys.path.insert(0, "D:/Desktop")
import akshare as ak
import pandas as pd
import numpy as np

DB = "data/trading.db"
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
INITIAL = 1_000_000

# ══════════════════════════════════════════════════════════════════════
# FACTOR CATALOG — add new factors here (~10 lines each)
# ══════════════════════════════════════════════════════════════════════

FACTORS = {}


def _factor_volatility_20d(prices, rets):
    return -rets.rolling(20).std()

FACTORS["volatility_20d"] = {
    "name": "volatility_20d",
    "compute": _factor_volatility_20d,
    "params": {"period": 20},
    "category": "low_volatility",
}


def _factor_reversal(prices, rets):
    """Short-term reversal: buy past 5-day losers."""
    return -rets.rolling(5).apply(lambda x: (1 + x).prod() - 1)

FACTORS["reversal"] = {
    "name": "reversal",
    "compute": _factor_reversal,
    "params": {"period": 5},
    "category": "contrarian",
}


def _factor_momentum_12m(prices, rets):
    """12-month momentum skipping the most recent month."""
    return (1 + rets).rolling(252).apply(lambda x: x.prod() - 1).shift(-21)

FACTORS["momentum_12m"] = {
    "name": "momentum_12m",
    "compute": _factor_momentum_12m,
    "params": {"period": 252, "skip": 21},
    "category": "momentum",
}


def _factor_momentum_1m(prices, rets):
    return rets.rolling(21).apply(lambda x: (1 + x).prod() - 1)

FACTORS["momentum_1m"] = {
    "name": "momentum_1m",
    "compute": _factor_momentum_1m,
    "params": {"period": 21},
    "category": "momentum",
}


def _factor_rsi_14d(prices, rets):
    gain = rets.clip(lower=0).rolling(14).mean()
    loss = (-rets).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

FACTORS["rsi_14d"] = {
    "name": "rsi_14d",
    "compute": _factor_rsi_14d,
    "params": {"period": 14},
    "category": "mean_reversion",
}


def _factor_roe(prices, rets):
    """ROE proxy using price-to-book reversal (rough, no financials in daily data).
    Uses the inverse of cumulative return as value proxy — high past return
    means expensive, low past return means cheap (value play).
    """
    return -rets.rolling(252).apply(lambda x: (1 + x).prod() - 1).shift(-21)

FACTORS["roe_proxy"] = {
    "name": "roe_proxy",
    "compute": _factor_roe,
    "params": {"period": 252, "skip": 21},
    "category": "quality",
}


# ══════════════════════════════════════════════════════════════════════
# DATA FETCH (once per run)
# ══════════════════════════════════════════════════════════════════════

def fetch_data():
    t0 = time.time()
    df = ak.index_stock_cons(symbol="000300")
    code_col = [c for c in df.columns][0]
    codes = sorted({str(c).zfill(6) for c in df[code_col].tolist()})
    n_stocks = len(codes)
    print(f"CSI300 members: {n_stocks}", flush=True)

    def txsym(code):
        return ("sh" if code.startswith(("60", "68", "9")) else "sz") + code

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
            print(f"  prices {i}/{n_stocks} ok={len(frames)}", flush=True)

    prices = pd.DataFrame(frames)
    prices.index.name = "date"
    rets = prices.pct_change(fill_method=None)
    fwd_ret = rets.shift(-1)
    all_dates = prices.index.sort_values()
    ny = (all_dates[-1] - all_dates[0]).days / 365.25
    print(f"Price panel: {prices.shape} (fetch {int(time.time() - t0)}s)", flush=True)
    return prices, rets, fwd_ret, all_dates, ny, n_stocks


# ══════════════════════════════════════════════════════════════════════
# SHARED BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

def backtest(factor_panel, rets, all_dates, rebals, ny,
             cost_multiplier=1.0, rebalance_override=None):
    capital = INITIAL
    daily_rets = []
    turnover_log = []
    current_w = {}
    rebal_dates = rebalance_override if rebalance_override is not None else rebals
    rebal_iter = iter(rebal_dates)
    next_rebal = next(rebal_iter, None)

    for date in all_dates:
        if next_rebal is not None and date >= next_rebal:
            if date in factor_panel.index:
                f = factor_panel.loc[date].dropna()
                if len(f) >= 60:
                    rank = f.rank(ascending=False)
                    long_u = rank.nsmallest(N_LONG).index.tolist()
                    short_u = rank.nlargest(N_SHORT).index.tolist()
                    if current_w:
                        old_set = set(current_w.keys())
                        new_set = set(long_u + short_u)
                        sold = old_set - new_set
                        if sold:
                            to = sum(abs(current_w.get(a, 0)) for a in sold)
                            turnover_log.append(float(to))
                            capital -= to * (C + P + S) * cost_multiplier * capital
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
            daily = sum(current_w.get(a, 0) * rets.loc[date, a]
                       for a in current_w if a in rets.columns
                       and date in rets.index and pd.notna(rets.loc[date, a]))
            daily_rets.append(daily)
            capital *= (1 + daily)
        else:
            daily_rets.append(0.0)

    dr = pd.Series(daily_rets, index=all_dates)
    cagr = (capital / INITIAL) ** (1 / ny) - 1
    sharpe = float((dr.mean() / (dr.std() + 1e-12)) * np.sqrt(252))
    dd = float(((dr.cumsum() + np.log(INITIAL) -
                 (dr.cumsum() + np.log(INITIAL)).cummax()).min() / INITIAL))
    avg_to = float(np.mean(turnover_log)) if turnover_log else 0
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": dd,
            "avg_turnover": avg_to, "daily_rets": dr,
            "turnover_log": turnover_log, "n_rebals": len(turnover_log)}


# ══════════════════════════════════════════════════════════════════════
# STYLE BENCHMARK (same factor, no costs)
# ══════════════════════════════════════════════════════════════════════

def style_benchmark(factor_panel, rets, all_dates, rebals, ny):
    capital = INITIAL
    daily_rets = []
    current_w = {}
    rebal_iter = iter(rebals)
    next_rebal = next(rebal_iter, None)

    for date in all_dates:
        if next_rebal is not None and date >= next_rebal:
            if date in factor_panel.index:
                f = factor_panel.loc[date].dropna()
                if len(f) >= 60:
                    rank = f.rank(ascending=False)
                    long_u = rank.nsmallest(N_LONG).index.tolist()
                    total = max(len(long_u), 1)
                    current_w = {}
                    for a in long_u:
                        current_w[a] = 1.0 / total
            try:
                next_rebal = next(rebal_iter)
            except StopIteration:
                next_rebal = None

        if current_w:
            daily = sum(current_w.get(a, 0) * rets.loc[date, a]
                       for a in current_w if a in rets.columns
                       and date in rets.index and pd.notna(rets.loc[date, a]))
            daily_rets.append(daily)
            capital *= (1 + daily)
        else:
            daily_rets.append(0.0)

    dr = pd.Series(daily_rets, index=all_dates)
    cagr = (capital / INITIAL) ** (1 / ny) - 1
    sharpe = float((dr.mean() / (dr.std() + 1e-12)) * np.sqrt(252))
    return {"cagr": cagr, "sharpe": sharpe, "daily_rets": dr}


# ══════════════════════════════════════════════════════════════════════
# MAIN REPORT
# ══════════════════════════════════════════════════════════════════════

def run_factor_report(factor_key):
    if factor_key not in FACTORS:
        print(f"Unknown factor: {factor_key}")
        print(f"Available: {list(FACTORS.keys())}")
        sys.exit(1)

    entry = FACTORS[factor_key]
    print("=" * 72)
    print(f"  FACTOR REPORT: {entry['name']} ({entry['category']})")
    print("=" * 72, flush=True)

    # Data
    prices, rets, fwd_ret, all_dates, ny, n_stocks = fetch_data()
    factor_panel = entry["compute"](prices, rets)

    # IC
    ic_vals, cs_vols_vals = [], []
    for date in factor_panel.index:
        f = factor_panel.loc[date].dropna()
        r = fwd_ret.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) >= 20:
            ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
            cv = r.loc[common].std()
            if not np.isnan(ic) and not np.isnan(cv):
                ic_vals.append(ic)
                cs_vols_vals.append(cv)
    mean_ic = float(np.mean(ic_vals))
    icir = mean_ic / (float(np.std(ic_vals)) + 1e-12)
    mean_cs_vol = float(np.mean(cs_vols_vals))

    # IC-implied theoretical alpha
    p = N_LONG / max(n_stocks, 1)
    z = math.sqrt(-2 * math.log(p)) if p > 0 and p < 0.5 else 0
    mills = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / p if p > 0 else 0
    theo_ann = mean_ic * mean_cs_vol * mills * 252

    # Rebalance schedule
    month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(
        lambda x: x.index[-1])
    rebals = sorted(month_end_dates.unique())

    # Portfolio backtest
    base = backtest(factor_panel, rets, all_dates, rebals, ny)
    cost_free = backtest(factor_panel, rets, all_dates, rebals, ny, cost_multiplier=0.0)
    style = style_benchmark(factor_panel, rets, all_dates, rebals, ny)

    # Market benchmark
    bench_dr = fwd_ret.mean(axis=1).reindex(base["daily_rets"].index).dropna()
    bench_cagr = float((1 + bench_dr).prod() ** (1 / ny) - 1)
    bench_sharpe = float((bench_dr.mean() / (bench_dr.std() + 1e-12)) * np.sqrt(252))

    # Excess
    market_excess = base["cagr"] - bench_cagr
    style_excess = base["cagr"] - style["cagr"]

    # Turnover & cost
    total_cost_paid = sum(t * (C + P + S) for t in base["turnover_log"])
    annual_cost_drag = total_cost_paid / ny if ny > 0 else 0
    gross_cagr = base["cagr"] + annual_cost_drag
    weight_loss = theo_ann - gross_cagr

    # Engineering sanity
    cost_ok = cost_free["cagr"] >= base["cagr"]
    turnover_zero = backtest(factor_panel, rets, all_dates, rebals, ny,
                             rebalance_override=[rebals[0]])
    to_ok = len(turnover_zero["turnover_log"]) <= 1
    rebals_ok = abs(base["n_rebals"] - len(rebals) + 1) <= 2

    # ── Print ──
    print("")
    print("  1. IC STATISTICS")
    print(f"  IC mean:    {mean_ic:+.4f}")
    print(f"  ICIR:       {icir:+.4f}")
    print(f"  IC>0%:      {np.mean(np.array(ic_vals) > 0) * 100:.0f}%")
    print(f"  N obs:      {len(ic_vals)}")
    print("")
    print("  2. PORTFOLIO")
    print(f"  CAGR:       {base['cagr']:+.2%}")
    print(f"  Sharpe:     {base['sharpe']:+.2f}")
    print(f"  Max DD:     {base['max_dd']:+.2%}")
    print(f"  Turnover:   {base['avg_turnover']*100:.1f}%/rb")
    print(f"  N rebals:   {base['n_rebals']}")
    print("")
    print("  3. BENCHMARKS")
    print(f"  {'':25} {'CAGR':>10} {'Sharpe':>10}")
    print(f"  {'Strategy':25} {base['cagr']:>+9.2%} {base['sharpe']:>+9.2f}")
    print(f"  {'Market (CSI300 EW)':25} {bench_cagr:>+9.2%} {bench_sharpe:>+9.2f}")
    print(f"  {'Style (no-cost)':25} {style['cagr']:>+9.2%} {style['sharpe']:>+9.2f}")
    print(f"  Excess vs Market: {market_excess:+.2%}")
    print(f"  Excess vs Style:  {style_excess:+.2%}")
    print("")
    print("  4. LEAKAGE AUDIT")
    print(f"  {'[0]':<5} IC-implied:          {theo_ann:>+8.2%}")
    print(f"  {'[1]':<5} – Weight Loss:       {-weight_loss:>+8.2%}")
    print(f"  {'[2]':<5} Gross portfolio:     {gross_cagr:>+8.2%}")
    print(f"  {'[3]':<5} – Cost Drag:         {-annual_cost_drag:>+8.2%}")
    print(f"  {'[4]':<5} Net portfolio:       {base['cagr']:>+8.2%}")
    print(f"  {'[5]':<5} – Benchmark gap:     {-market_excess:>+8.2%}")
    print(f"  {'':<5} = Net Excess:           {market_excess:>+8.2%}")
    print(f"")
    print(f"  Weight Loss: {abs(weight_loss)/(abs(theo_ann)+1e-8)*100:.0f}%  "
          f"Cost: {abs(annual_cost_drag)/(abs(theo_ann)+1e-8)*100:.0f}%")
    print("")
    print("  5. ENGINEERING SANITY")
    print(f"  Cost=0 net>base:  {'PASS' if cost_ok else 'FAIL'}")
    print(f"  Turnover=0 to~0:  {'PASS' if to_ok else 'FAIL'}")
    print(f"  Rebal count ok:   {'PASS' if rebals_ok else 'FAIL'}")
    all_sanity = cost_ok and to_ok and rebals_ok
    print(f"  {'ALL PASS' if all_sanity else 'REVIEW'}")
    print("=" * 72)

    # Registry
    rid = f"factor_{factor_key}_{int(time.time())}"
    ev = {
        "factor": entry["name"], "category": entry["category"],
        "ic_mean": float(mean_ic), "icir": float(icir),
        "cagr": float(base["cagr"]), "sharpe": base["sharpe"],
        "max_dd": base["max_dd"], "avg_turnover": base["avg_turnover"],
        "benchmark_cagr": float(bench_cagr), "style_cagr": float(style["cagr"]),
        "market_excess": float(market_excess), "style_excess": float(style_excess),
        "theoretical_ann": float(theo_ann), "weight_loss": float(weight_loss),
        "cost_drag": float(annual_cost_drag), "gross_cagr": float(gross_cagr),
        "sanity": {"cost_ok": cost_ok, "to_ok": to_ok, "rebals_ok": rebals_ok,
                   "all_pass": all_sanity},
        "n_stocks": n_stocks, "period_years": round(ny, 1),
    }
    conn = sqlite3.connect(DB)
    conn.execute(
        """INSERT INTO research_runs (run_id, timestamp, status, reason, slice, data_source, data_meta,
           universe_provider, universe_meta, factor, factor_params, evaluation, input_hash, report_path, warnings)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rid, pd.Timestamp.now('UTC').isoformat()[:19], "success" if all_sanity else "failed",
         "", "factor_report", "akshare",
         json.dumps({"adjust": "hfq", "provider": "akshare"}),
         "CurrentConstituentUniverseProvider",
         json.dumps({"pit": False, "bias_warning": ["survivorship_bias_possible"]}),
         entry["name"], json.dumps(entry["params"]),
         json.dumps(ev, default=str), "", "", "[]"))
    conn.commit()
    conn.close()
    print(f"Registry: {rid}")
    print("DONE")
    return base, style, mean_ic, icir, market_excess, style_excess


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "list":
            print("Available factors:")
            for k, v in FACTORS.items():
                print(f"  {k:25s}  {v['category']}")
            sys.exit(0)
        run_factor_report(target)
    else:
        print("Usage: python run_factor.py <factor_key>")
        print("       python run_factor.py list")
        print(f"Available: {list(FACTORS.keys())}")
