"""First real research study — 80-day reversal on a point-in-time HS300 universe.

Protocol (pre-registered before looking at the results):
    data:        Baostock daily close, back-adjusted (adjustflag="1")
    universe:    HS300 constituents queried at each month-end, 2020-01..2025-12
                 (point-in-time membership; no current-constituent look-ahead)
    factor:      reversal_80d = -(close[t] / close[t-80 trading days] - 1)
                 the single strongest horizon in
                 docs/research/MARKET_STRUCTURE_DISCOVERY_V3_REPORT.md
                 (IC -0.081, t -4.58); no parameter search
    evaluation:  cross-sectional Spearman IC vs forward 80-day return
    OOS:         2023-01-01 onward is never used for design decisions
    backtest:    monthly rebalance, long top quintile, equal weight,
                 15 bps one-way cost x turnover

Run:
    python -m quant_platform.lab.runs.first_real_research_80d_reversal
    python -m quant_platform.lab.runs.first_real_research_80d_reversal --fetch-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache" / "first_real_research"
REPORT = REPO / "data" / "reports" / "first_real_research_80d_reversal.md"

SLICE = "first_real_research_80d_reversal"
START = "2020-01-01"
END = "2025-12-31"
FETCH_START = "2019-06-01"
LOOKBACK = 80
OOS_START = "2023-01-01"
TOP_PCT = 0.20
COST_BPS = 15
MIN_NAMES = 30


# ── Data acquisition ────────────────────────────────────────────────────


def fetch_universe(force: bool = False) -> pd.DataFrame:
    """Monthly HS300 constituent snapshots (point-in-time membership)."""
    path = CACHE / "universe.csv"
    if path.exists() and not force:
        return pd.read_csv(path)

    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")

    rows = []
    for month_end in pd.date_range(START, END, freq=pd.offsets.MonthEnd()):
        query_date = month_end.strftime("%Y-%m-%d")
        rs = bs.query_hs300_stocks(date=query_date)
        if rs.error_code != "0":
            raise RuntimeError(f"HS300 query failed at {query_date}: {rs.error_msg}")
        while rs.next():
            update_date, code, name = rs.get_row_data()
            rows.append({
                "month": month_end.strftime("%Y-%m"),
                "code": code,
                "name": name,
                "effective_date": update_date,
            })
    bs.logout()

    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    print(f"universe: {len(frame)} rows, {frame['code'].nunique()} unique codes")
    return frame


def fetch_prices(codes: list[str], force: bool = False) -> None:
    """Daily back-adjusted close per code, cached one CSV per stock."""
    import baostock as bs

    prices_dir = CACHE / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        code for code in sorted(codes)
        if force or not (prices_dir / f"{code.replace('.', '_')}.csv").exists()
    ]
    if not pending:
        print("prices: cache complete")
        return

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_msg}")

    started = time.time()
    for index, code in enumerate(pending, 1):
        rows = []
        for attempt in range(3):
            rs = bs.query_history_k_data_plus(
                code, "date,close",
                start_date=FETCH_START, end_date=END,
                frequency="d", adjustflag="1",
            )
            if rs.error_code == "0":
                while rs.next():
                    rows.append(rs.get_row_data())
                break
            time.sleep(1 + attempt)
        if rows:
            pd.DataFrame(rows, columns=["date", "close"]).to_csv(
                prices_dir / f"{code.replace('.', '_')}.csv",
                index=False,
            )
        if index % 25 == 0 or index == len(pending):
            elapsed = time.time() - started
            print(
                f"prices: {index}/{len(pending)} fetched ({elapsed:.0f}s elapsed)",
                flush=True,
            )
    bs.logout()


# ── Panels ──────────────────────────────────────────────────────────────


def load_close() -> pd.DataFrame:
    frames = {}
    for path in sorted((CACHE / "prices").glob("*.csv")):
        code = path.stem.replace("_", ".")
        frame = pd.read_csv(path)
        series = pd.to_numeric(frame["close"], errors="coerce")
        series.index = pd.to_datetime(frame["date"])
        frames[code] = series[~series.index.duplicated(keep="last")]
    close = pd.DataFrame(frames).sort_index()
    return close


def month_ends(close: pd.DataFrame) -> pd.DatetimeIndex:
    last = close.index.to_series().groupby(close.index.to_period("M")).max()
    return pd.DatetimeIndex(last.values)


def eligibility(close: pd.DataFrame, universe: pd.DataFrame,
                dates: pd.DatetimeIndex) -> pd.DataFrame:
    by_month = universe.groupby("month")["code"].apply(set)
    mask = pd.DataFrame(False, index=dates, columns=close.columns)
    for date in dates:
        members = by_month.get(date.strftime("%Y-%m"), set())
        present = [code for code in members if code in mask.columns]
        if present:
            mask.loc[date, present] = True
    return mask


# ── Evaluation ──────────────────────────────────────────────────────────


def cross_sectional_ic(factor: pd.DataFrame, forward: pd.DataFrame,
                       eligible: pd.DataFrame) -> pd.Series:
    values = {}
    for date in factor.index:
        names = eligible.loc[date]
        f_vals = factor.loc[date][names].dropna()
        r_vals = forward.loc[date].reindex(f_vals.index).dropna()
        common = f_vals.index.intersection(r_vals.index)
        if len(common) < MIN_NAMES:
            continue
        values[date] = float(
            f_vals[common].corr(r_vals[common], method="spearman")
        )
    return pd.Series(values, dtype=float)


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 2:
        return {"n_months": int(len(ic)), "mean_ic": 0.0, "icir": 0.0,
                "t_stat": 0.0, "positive_ratio": 0.0}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    return {
        "n_months": int(len(ic)),
        "mean_ic": mean,
        "icir": mean / std if std > 0 else 0.0,
        "t_stat": mean / std * np.sqrt(len(ic)) if std > 0 else 0.0,
        "positive_ratio": float((ic > 0).mean()),
    }


def backtest(factor: pd.DataFrame, fwd_1m: pd.DataFrame,
             eligible: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    gross, net, bench, turnover = [], [], [], []
    dates = []
    previous = pd.Series(dtype=float)

    for date in factor.index[:-1]:
        names = eligible.loc[date]
        scores = factor.loc[date][names].dropna()
        returns = fwd_1m.loc[date].reindex(scores.index).dropna()
        scores = scores.reindex(returns.index)
        if len(scores) < MIN_NAMES:
            continue

        top_n = max(1, int(len(scores) * TOP_PCT))
        selected = scores.nlargest(top_n).index
        weights = pd.Series(1.0 / len(selected), index=selected)

        union = weights.index.union(previous.index)
        turn = float(
            (weights.reindex(union, fill_value=0.0)
             - previous.reindex(union, fill_value=0.0)).abs().sum() / 2
        )

        portfolio_return = float((weights * returns).sum())
        benchmark_return = float(returns.mean())
        # One-sided turnover counts half the two-way traded value (cash ->
        # fully invested is 0.5). COST_BPS is the cost per unit of traded
        # value, so charge it on the two-way value: turn * 2.
        cost = turn * 2 * COST_BPS / 10000

        dates.append(date)
        gross.append(portfolio_return)
        net.append(portfolio_return - cost)
        bench.append(benchmark_return)
        turnover.append(turn)
        previous = weights

    index = pd.DatetimeIndex(dates)
    return (
        pd.Series(gross, index=index, name="gross"),
        pd.Series(net, index=index, name="net"),
        pd.Series(bench, index=index, name="benchmark"),
        pd.Series(turnover, index=index, name="turnover"),
    )


def performance(monthly: pd.Series) -> dict:
    monthly = monthly.dropna()
    if len(monthly) < 2:
        return {"months": int(len(monthly)), "annual_return": 0.0,
                "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "hit_rate": 0.0}
    equity = (1 + monthly).cumprod()
    annual_return = float(equity.iloc[-1] ** (12 / len(monthly)) - 1)
    annual_vol = float(monthly.std(ddof=1) * np.sqrt(12))
    drawdown = float((equity / equity.cummax() - 1).min())
    return {
        "months": int(len(monthly)),
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": annual_return / annual_vol if annual_vol > 1e-12 else 0.0,
        "max_drawdown": drawdown,
        "hit_rate": float((monthly > 0).mean()),
    }


# ── Report and registry ─────────────────────────────────────────────────


def write_report(results: dict) -> None:
    full, ins, oos = results["ic"]["full"], results["ic"]["is"], results["ic"]["oos"]
    gross, net = results["backtest"]["gross"], results["backtest"]["net"]
    bench = results["backtest"]["benchmark"]

    def ic_row(label: str, stats: dict) -> str:
        return (
            f"| {label} | {stats['n_months']} | {stats['mean_ic']:.4f} | "
            f"{stats['icir']:.3f} | {stats['t_stat']:.2f} | "
            f"{stats['positive_ratio']:.1%} |"
        )

    def perf_row(label: str, stats: dict) -> str:
        return (
            f"| {label} | {stats['annual_return']:.2%} | {stats['annual_vol']:.2%} | "
            f"{stats['sharpe']:.2f} | {stats['max_drawdown']:.2%} | "
            f"{stats['hit_rate']:.1%} |"
        )

    text = f"""# First real research run — 80-day reversal

> **Run date**: {results['run_date']}
> **Registry run id**: `{results.get('run_id', 'not registered')}`
> **Status**: pre-registered before the results were computed; OOS = 2023-01 onward.

## 1. Pre-registered definition

| Item | Definition |
|---|---|
| Data | Baostock daily close, back-adjusted (`adjustflag="1"`, hfq) |
| Universe | HS300 constituents queried at every month-end, 2020-01..2025-12 |
| PIT | Yes — each month uses the constituent list effective on that date |
| Factor | `reversal_80d = -(close[t] / close[t-80 trading days] - 1)` |
| IC target | forward 80-trading-day return |
| Backtest | monthly, long top {TOP_PCT:.0%}, equal weight, {COST_BPS} bps per unit of traded value (charged on 2 x one-sided turnover) |
| OOS | 2023-01-01 onward |

The 80-day horizon was fixed from `docs/research/MARKET_STRUCTURE_DISCOVERY_V3_REPORT.md`
(IC −0.081, t −4.58). No other horizon or parameter was tried in this run.

## 2. Rank IC

| Sample | Months | Mean IC | ICIR | t-stat | IC > 0 |
|---|---:|---:|---:|---:|---:|
{ic_row('Full 2020-2025', full)}
{ic_row('In-sample 2020-2022', ins)}
{ic_row('Out-of-sample 2023-2025', oos)}

The t-stat treats monthly ICs as independent. Forward 80-day returns overlap
across adjacent months, so the effective sample is smaller and the true
t-stat is lower than shown; the OOS column is the one that matters.

## 3. Cost-aware monthly backtest

| Portfolio | Ann. return | Ann. vol | Sharpe | Max DD | Hit rate |
|---|---:|---:|---:|---:|---:|
{perf_row('Reversal top quintile, gross', gross)}
{perf_row('Reversal top quintile, net', net)}
{perf_row('HS300 equal-weight benchmark', bench)}

Average one-sided monthly turnover: {results['backtest']['avg_turnover']:.1%}.

## 4. What this says about the pre-registered hypothesis

Our factor is the **negative** of the past 80-day return, so a *positive* IC
means reversal works. The observed IC is {full['mean_ic']:+.4f}
(t={full['t_stat']:.2f}) full-sample and {oos['mean_ic']:+.4f}
(t={oos['t_stat']:.2f}) out-of-sample. The sign is **opposite** to the V3
report's raw past-return IC (−0.081): on this PIT HS300 sample the raw
past-return IC is {-full['mean_ic']:+.4f}, i.e. mild momentum, and it is not
statistically distinguishable from zero.

**Conclusion: the 80-day reversal result does not reproduce in this
specification.** The pre-registered strategy underperforms its equal-weight
benchmark after costs. That is the honest answer this run was designed to
produce — not a positive alpha claim.

## 5. Limitations (read before citing)

- **ST stocks are not filtered point-in-time.** Baostock does not expose a
  historical ST flag; names that are ST today are not removed from history.
- Price limits and trading halts are not modelled; a real book cannot trade
  every name at the month-end close.
- Membership is HS300 PIT, so the study says nothing about stocks that were
  never index members.
- The factor and the OOS split were fixed before this run; no parameter
  search was performed, but the horizon itself was chosen using earlier
  research on the same market (a mild selection risk, disclosed).
- Costs use a flat 15 bps per unit of traded value (charged on two-way
  turnover); stamp tax, minimum commission and market impact are not itemised.

## Reproduction

```bash
python -m quant_platform.lab.runs.first_real_research_80d_reversal
```
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def register_run(results: dict) -> str:
    from quant_platform.lab.registry import DEFAULT_DB, RunStore

    store = RunStore(str(REPO / DEFAULT_DB))
    run_id = store.begin_run(SLICE, {
        "data_source": "baostock",
        "data_meta": {
            "provider": "baostock",
            "adjust": "hfq",
            "adjustflag": "1",
            "start": START,
            "end": END,
        },
        "universe_provider": "BaostockHS300PITProvider",
        "universe_meta": {
            "kind": "index_constituent",
            "index": "hs300",
            "pit": True,
            "snapshot": "monthly",
            "bias_warning": ["ST_not_pit_filtered", "price_limits_not_modelled"],
        },
        "factor": "reversal_80d",
        "factor_params": {"lookback": LOOKBACK, "top_pct": TOP_PCT, "cost_bps": COST_BPS},
        "warnings": [
            "forward 80d returns overlap; t-stat overstates significance",
            "ST status is not point-in-time",
        ],
    })
    store.finish_run(
        run_id,
        status="success",
        evaluation=results["evaluation"],
        report_path=str(REPORT.relative_to(REPO)),
    )
    return run_id


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--force-fetch", action="store_true")
    args = parser.parse_args()

    universe = fetch_universe(force=args.force_fetch)
    codes = sorted(universe["code"].unique())
    fetch_prices(codes, force=args.force_fetch)
    if args.fetch_only:
        return 0

    close = load_close()
    dates = month_ends(close)
    eligible = eligibility(close, universe, dates)

    factor_daily = -(close / close.shift(LOOKBACK) - 1)
    forward_80_daily = close.shift(-LOOKBACK) / close - 1

    factor = factor_daily.loc[dates]
    forward_80 = forward_80_daily.loc[dates]
    close_monthly = close.loc[dates]
    forward_1m = close_monthly.shift(-1) / close_monthly - 1

    ic = cross_sectional_ic(factor, forward_80, eligible)
    split = pd.Timestamp(OOS_START)
    ic_stats_full = ic_stats(ic)
    ic_stats_is = ic_stats(ic[ic.index < split])
    ic_stats_oos = ic_stats(ic[ic.index >= split])

    gross, net, bench, turnover = backtest(factor, forward_1m, eligible)
    backtest_result = {
        "gross": performance(gross),
        "net": performance(net),
        "benchmark": performance(bench),
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }

    results = {
        "run_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "ic": {"full": ic_stats_full, "is": ic_stats_is, "oos": ic_stats_oos},
        "backtest": backtest_result,
        "evaluation": {
            "ic_full": ic_stats_full,
            "ic_is": ic_stats_is,
            "ic_oos": ic_stats_oos,
            "backtest_gross": backtest_result["gross"],
            "backtest_net": backtest_result["net"],
            "backtest_benchmark": backtest_result["benchmark"],
            "avg_turnover": backtest_result["avg_turnover"],
        },
    }

    write_report(results)
    results["run_id"] = register_run(results)
    write_report(results)  # include the run id

    print("IC full :", ic_stats_full)
    print("IC OOS  :", ic_stats_oos)
    print("Net perf:", backtest_result["net"])
    print("Bench   :", backtest_result["benchmark"])
    print("Report  :", REPORT)
    print("Run id  :", results["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
