"""Pre-registered evaluation report for the forward paper experiment.

Implements docs/research/EVALUATION_SPEC_2026-11.md exactly:

- reads only the append-only nav.csv ledgers (never summary.json);
- recomputes every metric from daily returns with the frozen formulas
  (Sharpe with rf = 0 as the primary, rf = 3% secondary);
- rebuilds the ST-excluded investable equal-weight benchmark B1 from the
  merged panel closes and Baostock PIT ``isST`` (cached under .scratch),
  and reports the ledger's own ST-including column as B2.

Usage:
    python tools/forward_evaluation.py [--skip-b1]

Outputs .scratch/evaluation/report_<date>.json and prints the fixed table.
Interim (3-month) runs are diagnostic only; the script makes no decision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "research" / "paper_track"
MERGED = ROOT / ".scratch" / "orchestrate" / "merged_wide.parquet"
ST_CACHE = ROOT / ".scratch" / "baostock" / "is_st_panel.parquet"
OUT = ROOT / ".scratch" / "evaluation"
START = pd.Timestamp("2026-10-26")
CAPITAL = 2_000_000.0

STRATEGIES = {
    "turnover": BASE / "forward_2026_turnover" / "nav.csv",
    "ep_turnover": BASE / "forward_2026" / "nav.csv",
}


def load_nav(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    frame = frame[frame.index >= START]
    return frame


def metrics(nav: pd.Series) -> dict:
    returns = nav.pct_change()
    returns.iloc[0] = nav.iloc[0] / CAPITAL - 1.0
    n = len(returns)
    cum = nav.iloc[-1] / CAPITAL - 1.0
    ann = (1.0 + cum) ** (252.0 / n) - 1.0
    vol = returns.std(ddof=1) * np.sqrt(252.0)
    sharpe = np.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
    sharpe_rf3 = np.sqrt(252.0) * (returns.mean() - 0.03 / 252.0) / returns.std(ddof=1)
    drawdown = (nav / nav.cummax() - 1.0).min()
    se = np.sqrt((1.0 + 0.5 * sharpe ** 2) / n)
    return {
        "n_days": int(n),
        "cum_return": float(cum),
        "annualized_return": float(ann),
        "annualized_vol": float(vol),
        "sharpe_rf0": float(sharpe),
        "sharpe_rf3": float(sharpe_rf3),
        "sharpe_95ci": [float(sharpe - 1.96 * se), float(sharpe + 1.96 * se)],
        "max_drawdown": float(drawdown),
        "daily_returns": returns,
    }


def st_panel(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """PIT isST per asset/date from Baostock, cached once."""
    if ST_CACHE.is_file():
        cached = pd.read_parquet(ST_CACHE)
        cached_start = pd.to_datetime(cached["date"]).min()
        cached_end = pd.to_datetime(cached["date"]).max()
        if cached_start <= start and cached_end >= end - pd.Timedelta(days=10):
            cached["date"] = pd.to_datetime(cached["date"])
            return cached
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")
    assets = sorted(p.stem for p in (ROOT / ".scratch" / "wide_sina_2026" / "stocks").glob("*.parquet"))
    rows = []
    try:
        for index, code in enumerate(assets, 1):
            market = "sh" if code.startswith("6") else "sz"
            rs = bs.query_history_k_data_plus(
                f"{market}.{code}", "date,isST",
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"), frequency="d",
            )
            while rs.error_code == "0" and rs.next():
                date, is_st = rs.get_row_data()
                rows.append((pd.Timestamp(date), code, is_st == "1"))
            if index % 50 == 0:
                print(f"st panel [{index}/{len(assets)}]", flush=True)
    finally:
        bs.logout()
    panel = pd.DataFrame(rows, columns=["date", "asset", "is_st"])
    ST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(ST_CACHE)
    return panel


def b1_returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """ST-excluded equal-weight daily returns of the pooled universe."""
    panel = pd.read_parquet(MERGED).reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    closes = panel.pivot_table(index="date", columns="asset", values="close").sort_index()
    closes = closes.loc[(closes.index >= start - pd.Timedelta(days=10)) & (closes.index <= end)]
    returns = closes.pct_change()
    st = st_panel(start, end).pivot_table(index="date", columns="asset", values="is_st")
    st = st.reindex(index=returns.index, columns=returns.columns).fillna(False).astype(bool)
    investable = returns.where(~st)
    return investable.mean(axis=1).dropna()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-b1", action="store_true")
    args = parser.parse_args()

    end_candidates = [load_nav(p).index.max() for p in STRATEGIES.values() if p.is_file()]
    end = max(end_candidates)
    b1 = None
    if not args.skip_b1:
        try:
            b1 = b1_returns(START, end)
            print(f"B1 rebuilt: {len(b1)} sessions", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"B1 unavailable ({type(exc).__name__}: {exc}); reporting B2 only", flush=True)

    report = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
              "window": f"{START.date()}..{end.date()}", "strategies": {}}
    for name, path in STRATEGIES.items():
        nav = load_nav(path)
        result = metrics(nav["nav"])
        bench_b2 = nav["benchmark_daily_return"].copy()
        bench_b2.iloc[0] = nav["benchmark_daily_return"].iloc[0]
        b2_cum = float((1.0 + bench_b2).prod() - 1.0)
        excess = {"b2": result["cum_return"] - b2_cum}
        if b1 is not None:
            aligned = b1.reindex(nav.index).dropna()
            if len(aligned) == len(nav):
                excess["b1"] = result["cum_return"] - float((1.0 + aligned).prod() - 1.0)
            else:
                excess["b1"] = None
        costs = float(nav["costs_paid"].sum())
        rebalances = int((nav["turnover"] > 0).sum())
        result.pop("daily_returns")
        result.update({"excess": excess, "cum_costs": costs,
                       "rebalances": rebalances, "b2_cum": b2_cum})
        report["strategies"][name] = result
        print(json.dumps({name: result}, ensure_ascii=False, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"report_{dt.date.today():%Y%m%d}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    print(f"report written: {out_path.name} sha256[:16]={digest}")
    print("NOTE: interim (3-month) runs are diagnostic only; no decision is made by this script.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
