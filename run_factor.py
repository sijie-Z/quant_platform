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
import math
import sys
import time
import traceback
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

# The repo directory IS the `quant_platform` package, so its parent must be on
# sys.path when this file is executed directly from a fresh checkout (no
# editable install). Mirrors the sys.path bootstrap in tools/*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_platform.backtest.metrics import max_drawdown as _metrics_max_drawdown
from quant_platform.utils.research_window import compact_date, resolve_end_date

#: Outputs are anchored to the repo root, not the process cwd: running the
#: script from another directory used to scatter databases/reports or fail at
#: the final write after all the fetch work (R76).
REPO_ROOT = Path(__file__).resolve().parent
DB = str(REPO_ROOT / "data" / "trading.db")
REPORT_DIR = REPO_ROOT / "data" / "reports"

C, S, P = 0.0003, 0.001, 0.0005
N_LONG, N_SHORT = 20, 0
INITIAL = 1_000_000

#: Below this many IC observations a run has not been meaningfully evaluated;
#: it is recorded as failed with NaN metrics rather than a clean-looking zero.
MIN_IC_OBS = 20

#: Fixed start of the price panel; the end is resolved at run time (CLI flag,
#: ``QUANT_RESEARCH_END_DATE``, or today) instead of a hardcoded stale date.
START_DATE = "2021-01-01"

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
    return (1 + rets).rolling(252).apply(lambda x: x.prod() - 1).shift(21)

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
    return -rets.rolling(252).apply(lambda x: (1 + x).prod() - 1).shift(21)

FACTORS["roe_proxy"] = {
    "name": "roe_proxy",
    "compute": _factor_roe,
    "params": {"period": 252, "skip": 21},
    "category": "quality",
}


def _json_safe(value):
    """Replace non-finite floats with None (strict-JSON-safe registry data).

    Failed runs intentionally carry NaN metrics; ``json.dumps`` emits bare
    ``NaN`` tokens, which Python round-trips but strict consumers (JS, jq)
    reject. NULL is the honest JSON representation of "not measured".
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# ══════════════════════════════════════════════════════════════════════
# DATA FETCH (once per run)
# ══════════════════════════════════════════════════════════════════════

def fetch_data(end_date: str | None = None):
    end_date = resolve_end_date(end_date)
    start_s, end_s = compact_date(START_DATE), compact_date(end_date)
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
                r = ak.stock_zh_a_hist_tx(symbol=txsym(c), start_date=start_s, end_date=end_s, adjust="hfq")
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
    if prices.empty or prices.shape[1] == 0 or len(prices) < 2:
        # The old code failed later with an opaque IndexError on an empty
        # panel, after every per-stock fetch had already been retried (R70).
        raise RuntimeError(
            f"empty price panel after fetch: shape={prices.shape} "
            f"requested={n_stocks}"
        )
    rets = prices.pct_change(fill_method=None)
    fwd_ret = rets.shift(-1)
    all_dates = prices.index.sort_values()
    ny = (all_dates[-1] - all_dates[0]).days / 365.25
    print(f"Price panel: {prices.shape} (fetch {int(time.time() - t0)}s)", flush=True)
    # The actual data window, not the requested one: the provider may end
    # earlier than requested and individual symbols may fail. Both the
    # requested and the realised panel width are recorded so a silently
    # shrunken panel is visible in the output (R70).
    window = {
        "provider": "akshare",
        "adjust": "hfq",
        "requested_end": end_date,
        "data_start": str(all_dates[0].date()),
        "data_end": str(all_dates[-1].date()),
        "n_stocks_requested": n_stocks,
        "n_stocks_actual": int(prices.shape[1]),
        "n_fetch_failures": int(n_stocks - prices.shape[1]),
    }
    print(f"Data window: {window['data_start']} → {window['data_end']} "
          f"(requested end {end_date}, panel {prices.shape[1]}/{n_stocks})",
          flush=True)
    return prices, rets, fwd_ret, all_dates, ny, n_stocks, window


# ══════════════════════════════════════════════════════════════════════
# SHARED BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════

def backtest(factor_panel, rets, all_dates, rebals, ny,
             cost_multiplier=1.0, rebalance_override=None):
    capital = INITIAL
    daily_rets = []
    turnover_log = []
    cost_log = []
    current_w = {}
    active_start_idx = None
    rebal_dates = rebalance_override if rebalance_override is not None else rebals
    rebal_iter = iter(rebal_dates)
    next_rebal = next(rebal_iter, None)
    last_pos = len(all_dates) - 1

    for pos, date in enumerate(all_dates):
        pending_w = None
        pending_cost = 0.0
        if next_rebal is not None and date >= next_rebal:
            # A signal on the final bar has no following trading day to be
            # executed into. The engine drops such unexecuted signals (M13);
            # this loop used to charge their build cost anyway, so net CAGR
            # paid for a book that was never held. Skip terminal rebalances
            # consistently in the net, gross and style legs.
            if pos < last_pos and date in factor_panel.index:
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

                    # One-way turnover = 0.5 * sum(|w_new - w_old|). The old
                    # formula counted only names that *left* the book, so new
                    # entries were free and surviving names reset to 1/N
                    # without charging drift; true turnover is roughly double.
                    # Buys pay commission + slippage, sells also pay stamp
                    # tax, so both sides are charged.
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
                    to = 0.5 * (buy_w + sell_w)
                    pending_cost = (buy_w * (C + P)
                                    + sell_w * (C + P + S)) * cost_multiplier
                    turnover_log.append(float(to))
                    cost_log.append(float(pending_cost))
                    if active_start_idx is None:
                        # The strategy's performance starts on the day it
                        # first trades: the entry cost belongs to the net
                        # return series, and annualising from the prior
                        # (cash) day made CAGR one trading day too high.
                        active_start_idx = pos
            try:
                next_rebal = next(rebal_iter)
            except StopIteration:
                next_rebal = None

        # Today's return belongs to the portfolio held since the previous
        # close. Weights decided at today's close (pending_w) must not earn
        # today's already-realized t-1 -> t return; they take effect tomorrow.
        if current_w:
            gross = sum(current_w.get(a, 0) * rets.loc[date, a]
                        for a in current_w if a in rets.columns
                        and date in rets.index and pd.notna(rets.loc[date, a]))
        else:
            gross = 0.0
        # Costs are paid at today's close, after the day's gross return. The
        # daily series is therefore NET of costs: labelling Sharpe/MaxDD "net
        # of costs" while compounding only gross returns was a false claim
        # (the cost used to reduce capital without touching the series).
        daily = (1.0 + gross) * (1.0 - pending_cost) - 1.0
        daily_rets.append(daily)
        capital *= (1.0 + daily)
        if pending_w is not None:
            current_w = pending_w

    dr = pd.Series(daily_rets, index=all_dates)
    if active_start_idx is None:
        # The factor never produced an executable book: NaN metrics plus a
        # status flag, not a clean-looking 0% (R64/R86).
        cagr = sharpe = dd = float("nan")
        ny_active = float("nan")
        active_start = ""
        status = "no_active_period"
    else:
        # Warm-up cash days are not part of the strategy's performance: the
        # strategy was not invested. The active window starts on the first
        # executable rebalance (the day the book is established), so the entry
        # cost is inside the net return series and CAGR is not annualised over
        # a longer cash-only period (R86).
        active = dr.iloc[active_start_idx:]
        first = dr.index[active_start_idx]
        ny_active = max((dr.index[-1] - first).days / 365.25, 1.0 / 252.0)
        cagr = (capital / INITIAL) ** (1 / ny_active) - 1
        sharpe = float((active.mean() / (active.std() + 1e-12)) * np.sqrt(252))
        # Max drawdown of the compounded equity curve, identical to
        # backtest.metrics.max_drawdown. The old formula divided a log-equity
        # cumsum by INITIAL, giving values around -4e-7 (printed as "-0.00%")
        # for every strategy, including ones that lost 38%.
        dd = float(_metrics_max_drawdown(active)[0])
        active_start = str(first.date())
        status = "ok"
    avg_to = float(np.mean(turnover_log)) if turnover_log else 0
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": dd,
            "avg_turnover": avg_to, "daily_rets": dr,
            "turnover_log": turnover_log, "cost_log": cost_log,
            "n_rebals": len(turnover_log),
            "status": status, "active_start": active_start,
            "warmup_days": active_start_idx if active_start_idx is not None else 0,
            "ny_active": ny_active}


# ══════════════════════════════════════════════════════════════════════
# STYLE BENCHMARK (same factor, no costs)
# ══════════════════════════════════════════════════════════════════════

def style_benchmark(factor_panel, rets, all_dates, rebals, ny):
    capital = INITIAL
    daily_rets = []
    current_w = {}
    active_start_idx = None
    rebal_iter = iter(rebals)
    next_rebal = next(rebal_iter, None)
    last_pos = len(all_dates) - 1

    for pos, date in enumerate(all_dates):
        pending_w = None
        if next_rebal is not None and date >= next_rebal:
            # Terminal signals are unexecuted (no following trading day), so
            # they are skipped here exactly as in the net/gross legs.
            if pos < last_pos and date in factor_panel.index:
                f = factor_panel.loc[date].dropna()
                if len(f) >= 60:
                    rank = f.rank(ascending=False)
                    long_u = rank.nsmallest(N_LONG).index.tolist()
                    total = max(len(long_u), 1)
                    pending_w = {}
                    for a in long_u:
                        pending_w[a] = 1.0 / total
                    if active_start_idx is None:
                        # Same start date as the net/gross legs: the day the
                        # style book is first established.
                        active_start_idx = pos
            try:
                next_rebal = next(rebal_iter)
            except StopIteration:
                next_rebal = None

        # Same next-day activation as backtest(): today's return is earned by
        # the weights held since the previous close, not by today's decision.
        if current_w:
            daily = sum(current_w.get(a, 0) * rets.loc[date, a]
                       for a in current_w if a in rets.columns
                       and date in rets.index and pd.notna(rets.loc[date, a]))
        else:
            daily = 0.0
        daily_rets.append(daily)
        capital *= (1 + daily)
        if pending_w is not None:
            current_w = pending_w

    dr = pd.Series(daily_rets, index=all_dates)
    if active_start_idx is None:
        cagr = sharpe = float("nan")
        active_start = ""
        status = "no_active_period"
    else:
        active = dr.iloc[active_start_idx:]
        first = dr.index[active_start_idx]
        ny_active = max((dr.index[-1] - first).days / 365.25, 1.0 / 252.0)
        cagr = (capital / INITIAL) ** (1 / ny_active) - 1
        sharpe = float((active.mean() / (active.std() + 1e-12)) * np.sqrt(252))
        active_start = str(first.date())
        status = "ok"
    return {"cagr": cagr, "sharpe": sharpe, "daily_rets": dr,
            "status": status, "active_start": active_start}


# ══════════════════════════════════════════════════════════════════════
# MAIN REPORT
# ══════════════════════════════════════════════════════════════════════

def run_factor_report(factor_key, end_date: str | None = None):
    if factor_key not in FACTORS:
        print(f"Unknown factor: {factor_key}")
        print(f"Available: {list(FACTORS.keys())}")
        sys.exit(1)

    entry = FACTORS[factor_key]
    print("=" * 72)
    print(f"  FACTOR REPORT: {entry['name']} ({entry['category']})")
    print("=" * 72, flush=True)

    # Record the run BEFORE the fetch: a crash during data collection must
    # leave an honest failed row, and the registry schema is created by
    # RunStore instead of being assumed to exist (R77/R78).
    from quant_platform.lab.registry import RunStore
    from quant_platform.lab.reports import generate_report

    store = RunStore(DB)
    run_id = store.begin_run("factor_report", {
        "factor": entry["name"],
        "factor_params": entry["params"],
        "data_source": "akshare",
        "data_meta": {"provider": "akshare", "adjust": "hfq",
                      "requested_end": resolve_end_date(end_date)},
        "universe_provider": "CurrentConstituentUniverseProvider",
        "universe_meta": {"provider": "CurrentConstituentUniverseProvider",
                          "kind": "current_constituent", "pit": False,
                          "bias_warning": ["survivorship_bias_possible"]},
        "warnings": [],
    })
    warnings: list[str] = []
    evaluation: dict | None = None

    try:
        # Data
        prices, rets, fwd_ret, all_dates, ny, n_stocks_requested, window = fetch_data(end_date)
        # Actual panel width, not the requested CSI300 count: failed symbol
        # fetches used to be silently dropped while the requested count drove
        # the IC-implied alpha and the registry (R70).
        n_stocks = int(prices.shape[1])
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
        n_ic_obs = len(ic_vals)
        if ic_vals:
            mean_ic = float(np.mean(ic_vals))
            ic_std = float(np.std(ic_vals))
            icir = mean_ic / (ic_std + 1e-12)
            mean_cs_vol = float(np.mean(cs_vols_vals))
            ic_positive = float(np.mean(np.array(ic_vals) > 0))
        else:
            # No observations: NaN, never a clean-looking 0 and never a
            # success record (R64/R86).
            mean_ic = ic_std = icir = mean_cs_vol = ic_positive = float("nan")

        # IC-implied theoretical alpha
        p = N_LONG / max(n_stocks, 1)
        z = math.sqrt(-2 * math.log(p)) if 0 < p < 0.5 else 0
        mills = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / p if p > 0 else 0
        theo_ann = mean_ic * mean_cs_vol * mills * 252

        # Rebalance schedule
        month_end_dates = rets.groupby([rets.index.year, rets.index.month]).apply(
            lambda x: x.index[-1])
        rebals = sorted(month_end_dates.unique())

        # Portfolio backtest. `cost_free` IS the gross leg: identical weights,
        # schedule and returns, no cost deduction.
        base = backtest(factor_panel, rets, all_dates, rebals, ny)
        cost_free = backtest(factor_panel, rets, all_dates, rebals, ny, cost_multiplier=0.0)
        style = style_benchmark(factor_panel, rets, all_dates, rebals, ny)
        net_cagr = base["cagr"]
        gross_cagr = cost_free["cagr"]

        # Market benchmark over the strategy's ACTIVE window only. The old
        # comparison annualised both legs over the full panel, so the
        # strategy's cash warm-up days diluted its CAGR while the benchmark
        # earned returns over the same days (R86). daily_rets[t] is the return
        # earned over day t (close t-1 -> close t), matching rets.mean(axis=1).
        if base["status"] == "ok":
            # The strategy is invested at the close of the first rebalance
            # day; its first market return is the NEXT row. Skip the trade
            # day's pre-investment benchmark return but keep annualising over
            # the strategy's active span (trade day -> end).
            first_holding = min(base["warmup_days"] + 1, len(base["daily_rets"]))
            active_index = base["daily_rets"].index[first_holding:]
            bench_dr = rets.mean(axis=1).reindex(active_index).dropna()
            if len(bench_dr):
                bench_cagr = float((1 + bench_dr).prod() ** (1 / base["ny_active"]) - 1)
                bench_sharpe = float(
                    (bench_dr.mean() / (bench_dr.std() + 1e-12)) * np.sqrt(252)
                )
            else:
                bench_cagr = bench_sharpe = float("nan")
        else:
            bench_cagr = bench_sharpe = float("nan")

        # Excess. The style leg is explicitly no-cost, so its gap is measured
        # gross-vs-gross; the market gap is net-of-costs vs a gross benchmark
        # and is labelled as such. The old code compared net to gross and
        # reported the difference as if the books were like-for-like.
        market_excess = net_cagr - bench_cagr
        style_excess = gross_cagr - style["cagr"]

        # Cost drag in CAGR space, so the leakage decomposition is additive by
        # construction: gross - (gross - net) = net. The old code added the
        # average annual cost fraction to the net CAGR, mixing a simple return
        # with a compounded one and calling the result additive (R86).
        cost_drag_cagr = gross_cagr - net_cagr
        weight_loss = theo_ann - gross_cagr

        # Engineering sanity
        cost_ok = bool(cost_free["cagr"] >= base["cagr"]) if base["status"] == "ok" else False
        turnover_zero = backtest(
            factor_panel, rets, all_dates, rebals, ny,
            rebalance_override=[rebals[0]] if rebals else [],
        )
        to_ok = len(turnover_zero["turnover_log"]) <= 1
        # Long-warm-up factors (momentum_12m needs 252+21 days) legitimately
        # skip the first ~13 month-ends; comparing against len(rebals)
        # unconditionally made every such factor report FAIL and be recorded
        # as a failed run. The terminal month-end is not executed either.
        eligible_rebals = sum(
            1 for d in rebals
            if d != all_dates[-1] and d in factor_panel.index
            and len(factor_panel.loc[d].dropna()) >= 60
        )
        rebals_ok = abs(base["n_rebals"] - eligible_rebals) <= 2
        all_sanity = cost_ok and to_ok and rebals_ok

        ic_ok = n_ic_obs >= MIN_IC_OBS
        if not ic_ok:
            warnings.append(
                f"insufficient_ic_observations: n_ic_obs={n_ic_obs} < {MIN_IC_OBS}"
            )
        if base["status"] != "ok":
            warnings.append("no_active_period: factor never produced an executable book")
        if not all_sanity:
            warnings.append(
                "engineering_sanity_failed: "
                f"cost_ok={cost_ok} to_ok={to_ok} rebals_ok={rebals_ok}"
            )
        status = "success" if (ic_ok and all_sanity and base["status"] == "ok") else "failed"
        reason = None if status == "success" else "; ".join(warnings)

        # ── Print ──
        print("")
        print("  1. IC STATISTICS")
        print(f"  IC mean:    {mean_ic:+.4f}")
        print(f"  ICIR:       {icir:+.4f}")
        print(f"  IC>0%:      {ic_positive * 100:.0f}%" if ic_ok or n_ic_obs else "  IC>0%:      n/a")
        print(f"  N obs:      {n_ic_obs}")
        print("")
        print("  2. PORTFOLIO (net of costs)")
        print(f"  CAGR:       {base['cagr']:+.2%}")
        print(f"  Sharpe:     {base['sharpe']:+.2f}")
        print(f"  Max DD:     {base['max_dd']:+.2%}")
        print(f"  Turnover:   {base['avg_turnover']*100:.1f}%/rb")
        print(f"  N rebals:   {base['n_rebals']}")
        print(f"  Active:     {base['active_start'] or 'n/a'} → {window['data_end']} "
              f"({base['warmup_days']} warm-up days excluded)")
        print("")
        print("  3. BENCHMARKS")
        print(f"  {'':32} {'CAGR':>10} {'Sharpe':>10}")
        print(f"  {'Strategy (net of costs)':32} {base['cagr']:>+9.2%} {base['sharpe']:>+9.2f}")
        print(f"  {'Market (CSI300 EW, gross)':32} {bench_cagr:>+9.2%} {bench_sharpe:>+9.2f}")
        print(f"  {'Style (gross, no costs)':32} {style['cagr']:>+9.2%} {style['sharpe']:>+9.2f}")
        print(f"  Excess vs Market (net - gross):  {market_excess:+.2%}")
        print(f"  Excess vs Style  (gross - gross): {style_excess:+.2%}")
        print("")
        print("  4. LEAKAGE AUDIT (additive in CAGR space)")
        print(f"  {'[0]':<5} IC-implied:          {theo_ann:>+8.2%}")
        print(f"  {'[1]':<5} – Weight Loss:       {-weight_loss:>+8.2%}")
        print(f"  {'[2]':<5} Gross portfolio:     {gross_cagr:>+8.2%}")
        print(f"  {'[3]':<5} - Cost Drag (CAGR):  {-cost_drag_cagr:>+8.2%}")
        print(f"  {'[4]':<5} Net portfolio:       {net_cagr:>+8.2%}")
        print(f"  {'[5]':<5} – Benchmark gap:     {-market_excess:>+8.2%}")
        print(f"  {'':<5} = Net Excess:           {market_excess:>+8.2%}")
        print("")
        print(f"  Weight Loss: {abs(weight_loss)/(abs(theo_ann)+1e-8)*100:.0f}%  "
              f"Cost: {abs(cost_drag_cagr)/(abs(theo_ann)+1e-8)*100:.0f}%")
        print("")
        print("  5. ENGINEERING SANITY")
        print(f"  Cost=0 net>base:  {'PASS' if cost_ok else 'FAIL'}")
        print(f"  Turnover=0 to~0:  {'PASS' if to_ok else 'FAIL'}")
        print(f"  Rebal count ok:   {'PASS' if rebals_ok else 'FAIL'}")
        print(f"  {'ALL PASS' if all_sanity else 'REVIEW'}")
        print("=" * 72)

        # Registry (through RunStore; see R77/R78). The metric basis is part
        # of the record so no reader has to guess whether a number is gross
        # or net.
        evaluation = {
            "factor": entry["name"], "category": entry["category"],
            "ic_mean": float(mean_ic), "ic_std": float(ic_std),
            "icir": float(icir), "n_ic_obs": n_ic_obs,
            "cagr": float(base["cagr"]), "gross_cagr": float(gross_cagr),
            "sharpe": base["sharpe"], "max_dd": base["max_dd"],
            "avg_turnover": base["avg_turnover"],
            "benchmark_cagr": float(bench_cagr), "style_cagr": float(style["cagr"]),
            "market_excess": float(market_excess), "style_excess": float(style_excess),
            "theoretical_ann": float(theo_ann), "weight_loss": float(weight_loss),
            "cost_drag": float(cost_drag_cagr),
            "metric_basis": {
                "cagr": "net_of_costs", "sharpe": "net_of_costs",
                "market_excess": "net_vs_gross_benchmark",
                "style_excess": "gross_vs_gross",
                "cost_drag": "gross_minus_net_cagr",
            },
            "sanity": {"cost_ok": cost_ok, "to_ok": to_ok, "rebals_ok": rebals_ok,
                       "all_pass": all_sanity},
            "n_stocks": n_stocks,
            "n_stocks_requested": window["n_stocks_requested"],
            "n_fetch_failures": window["n_fetch_failures"],
            "period_years": round(float(base["ny_active"]), 2) if base["status"] == "ok" else None,
            "panel_years": round(float(ny), 2),
            "active_start": base["active_start"],
            "warmup_days": base["warmup_days"],
            "status_reason": reason or "ok",
            # The as-of window is part of the result: without it a reader
            # cannot tell a current report from one built on a stale window.
            "data_start": window["data_start"], "data_end": window["data_end"],
            "requested_end": window["requested_end"],
        }
        data_meta = {
            "provider": window["provider"], "adjust": window["adjust"],
            "start_date": window["data_start"], "end_date": window["data_end"],
            "requested_end": window["requested_end"], "as_of": window["data_end"],
            "n_stocks_requested": window["n_stocks_requested"],
            "n_stocks_actual": window["n_stocks_actual"],
            "n_fetch_failures": window["n_fetch_failures"],
        }
        evaluation = _json_safe(evaluation)
        store.finish_run(run_id, status=status, evaluation=evaluation,
                         reason=reason, warnings=warnings, data_meta=data_meta)
        report_path = generate_report(store.get_run(run_id), out_dir=str(REPORT_DIR))
        store.finish_run(run_id, status=status, evaluation=evaluation,
                         reason=reason, report_path=report_path, warnings=warnings)
        print(f"Registry: {run_id}")
        print(f"Report:   {report_path}")
        print("DONE")
        return base, style, mean_ic, icir, market_excess, style_excess
    except Exception as e:
        tb = traceback.format_exc()
        # Keep whatever evaluation was already built: a report-generation
        # failure after a successful run must not wipe its metrics (the old
        # second finish_run wrote evaluation={} on the failure path).
        store.finish_run(
            run_id, status="failed",
            evaluation=evaluation,
            reason=f"{type(e).__name__}: {e}",
            warnings=[f"traceback_excerpt: {tb[-400:]}"],
        )
        raise


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    argv = sys.argv[1:]
    end_arg = None
    if "--end-date" in argv:
        pos = argv.index("--end-date")
        try:
            end_arg = argv[pos + 1]
        except IndexError:
            print("--end-date requires a value (YYYY-MM-DD)")
            sys.exit(2)
        del argv[pos:pos + 2]
    if argv:
        target = argv[0]
        if target == "list":
            print("Available factors:")
            for k, v in FACTORS.items():
                print(f"  {k:25s}  {v['category']}")
            sys.exit(0)
        run_factor_report(target, end_date=end_arg)
    else:
        print("Usage: python run_factor.py <factor_key> [--end-date YYYY-MM-DD]")
        print("       python run_factor.py list")
        print(f"Available: {list(FACTORS.keys())}")
