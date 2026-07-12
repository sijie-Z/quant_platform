"""M4 Portfolio Backtest — volatility_20d Top20 Long / Bottom20 Short
Monthly rebalance, full A-share cost model (commission + stamp + slippage).
Output: CAGR, Sharpe, Sortino, MaxDD, Turnover, Yearly Returns.
"""
import sys, os, time, json, sqlite3
sys.path.insert(0, "D:/Desktop")
import akshare as ak
import pandas as pd
import numpy as np

print("=== M4 PORTFOLIO BACKTEST: volatility_20d ===", flush=True)

# 1. Fetch CSI300 prices
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

# 2. Factor
vol20 = -rets.rolling(20).std()
print(f"Factor: {vol20.shape}", flush=True)

# 3. Portfolio backtest
C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 20
initial = 1_000_000
capital = initial

month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(lambda x: x.index[-1])
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
                short_u = rank.nlargest(N_SHORT).index.tolist()

                if current_w:
                    old_set = set(current_w.keys())
                    new_set = set(long_u + short_u)
                    sold = old_set - new_set
                    if sold and len(current_w) > 0:
                        to = sum(abs(current_w.get(a, 0)) for a in sold)
                        turnover_log.append(float(to))
                        capital -= to * (C + P + S) * capital

                total = len(long_u) + len(short_u)
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

# Bench
bpv = (1 + fwd_ret.mean(axis=1).reindex(all_dates).fillna(0)).cumprod()

# 4. Metrics
cagr = (pv.iloc[-1]/initial)**(1/ny) - 1
ann_vol = float(dr.std() * np.sqrt(252))
sharpe = float((dr.mean()/(dr.std()+1e-12)) * np.sqrt(252))
sortino = float((dr.mean()/(dr[dr<0].std()+1e-12)) * np.sqrt(252))
dd = (pv - pv.cummax()) / pv.cummax()
max_dd = float(dd.min())
win = float((dr > 0).mean())
monthly_r = dr.resample("ME").apply(lambda x: (1+x).prod()-1)
monthly_w = float((monthly_r > 0).mean())
avg_to = float(np.mean(turnover_log)) if turnover_log else 0
bench_c = (bpv.iloc[-1]/bpv.iloc[0])**(1/ny) - 1

# Per-year
yret = {}
for yr, g in dr.groupby(dr.index.year):
    yret[yr] = float((1+g).prod()-1)

roll12 = dr.rolling(252).mean() / (dr.rolling(252).std() + 1e-12) * np.sqrt(252)
roll12 = roll12.dropna()

# 5. Print results
print("")
print("=== RESULTS ===")
print(f"Total return:  {pv.iloc[-1]/initial-1:+.2%}")
print(f"CAGR:          {cagr:+.2%}")
print(f"Ann Vol:       {ann_vol:.2%}")
print(f"Sharpe:        {sharpe:+.2f}")
print(f"Sortino:       {sortino:+.2f}")
print(f"Max DD:        {max_dd:+.2%}")
print(f"Daily win:     {win:.1%}  Monthly win: {monthly_w:.1%}")
print(f"Avg turnover:  {avg_to*100:.1f}%")
print(f"Bench CAGR:    {bench_c:+.2%}")
print(f"Excess:        {cagr-bench_c:+.2%}")
print(f"N rebals:      {len(turnover_log)}")
print(f"Cost:          C={C*10000:.0f}bp S={S*10000:.0f}bp Slip={P*10000:.0f}bp")
print("Yearly:")
for yr in sorted(yret):
    print(f"  {yr}: {yret[yr]:+.2%}")
if len(roll12) > 0:
    print(f"Rolling 12m Sharpe: min={roll12.min():+.2f} max={roll12.max():+.2f} latest={roll12.iloc[-1]:+.2f}")

# 6. Store in Registry
c = sqlite3.connect("D:/Desktop/quant_platform/data/trading.db")
rid = f"m4_pf_{int(time.time())}"
ev = {
    "validation_type":"m4_portfolio_backtest","factor":"volatility_20d",
    "cagr":float(cagr),"sharpe":sharpe,"sortino":sortino,"max_dd":max_dd,
    "ann_vol":ann_vol,"win_rate":win,"monthly_win":monthly_w,
    "avg_turnover":float(avg_to),"n_rebals":len(turnover_log),
    "benchmark_cagr":float(bench_c),"excess":float(cagr-bench_c),
    "yearly":yret,"cost_model":{"commission":C,"stamp":S,"slippage":P},
    "setup":{"n_long":N_LONG,"n_short":N_SHORT},
    "dsr":"insufficient_trials","bh_fdr":"n/a",
}
c.execute("""INSERT INTO research_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (rid, pd.Timestamp.utcnow().isoformat()[:19], "success", "", "m4_portfolio_backtest",
     "akshare", json.dumps({"adjust":"hfq","provider":"akshare"},default=str),
     "CurrentConstituentUniverseProvider", json.dumps({"pit":False,"bias_warning":["survivorship_bias_possible"]},default=str),
     "volatility_20d", json.dumps({"n_long":N_LONG,"n_short":N_SHORT}),
     json.dumps(ev,default=str), "", "", "[]"))
c.commit()
print(f"\nRegistry: {rid}")
print("DONE")
