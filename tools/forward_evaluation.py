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
DECISION_DATE = pd.Timestamp("2027-04-30")
MIN_DECISION_DAYS = 120

STRATEGIES = {
    "turnover": BASE / "forward_2026_turnover" / "nav.csv",
    "ep_turnover": BASE / "forward_2026" / "nav.csv",
}


def load_nav(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if frame["date"].duplicated().any():
        duplicated = frame.loc[frame["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"duplicate ledger dates in {path.name}: {duplicated[:10]}")
    frame = frame.set_index("date").sort_index()
    frame = frame[frame.index >= START]
    return frame


def metrics(nav: pd.Series) -> dict:
    if nav.empty or not np.isfinite(nav.astype(float)).all() or (nav <= 0).any():
        raise RuntimeError("NAV contains empty, non-finite, or non-positive values")
    returns = nav.pct_change()
    returns.iloc[0] = nav.iloc[0] / CAPITAL - 1.0
    n = len(returns)
    cum = nav.iloc[-1] / CAPITAL - 1.0
    compounded = float((1.0 + returns).prod() - 1.0)
    if abs(compounded - cum) > 1e-9:
        raise RuntimeError(
            f"daily returns compound to {compounded}, not cumulative NAV {cum}"
        )
    ann = (1.0 + cum) ** (252.0 / n) - 1.0
    vol = returns.std(ddof=1) * np.sqrt(252.0)
    sharpe = np.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
    sharpe_rf3 = np.sqrt(252.0) * (returns.mean() - 0.03 / 252.0) / returns.std(ddof=1)
    nav_path = pd.concat([
        pd.Series([CAPITAL], index=[nav.index[0] - pd.Timedelta(days=1)]), nav,
    ])
    drawdown = (nav_path / nav_path.cummax() - 1.0).min()
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
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        cached_start = cached["date"].min()
        cached_end = cached["date"].max()
        # PIT ST data must cover the actual evaluation end. A recent but
        # incomplete cache is not acceptable for the formal benchmark.
        if cached_start <= start and cached_end >= end:
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
    returns = returns.loc[returns.index >= start]
    st = st_panel(start, end).pivot_table(index="date", columns="asset", values="is_st")
    st = st.reindex(index=returns.index, columns=returns.columns)
    missing = returns.notna() & st.isna()
    if missing.to_numpy().any():
        rows, cols = np.where(missing.to_numpy())
        examples = [
            f"{returns.index[row].date()}:{returns.columns[col]}"
            for row, col in zip(rows[:10], cols[:10])
        ]
        raise RuntimeError(
            f"PIT ST mask missing for {int(missing.to_numpy().sum())} "
            f"tradable asset-days; examples={examples}"
        )
    investable = returns.where(~st.fillna(False).astype(bool))
    return investable.mean(axis=1).dropna()


def align_b1_returns(b1: pd.Series, ledger_dates: pd.DatetimeIndex) -> pd.Series:
    """Align B1 to the ledger window; inception-day return is zero.

    The strategy starts at the 2026-10-26 execution close. The raw B1
    pct_change on that date measures 10-23-close to 10-26-close, a period
    before the paper book existed, so it must not enter the comparison.
    """
    aligned = b1.reindex(ledger_dates)
    if aligned.isna().any() or len(aligned) != len(ledger_dates):
        missing = ledger_dates[aligned.isna()]
        raise RuntimeError(
            "B1 does not cover every strategy ledger date; "
            f"missing={list(missing.strftime('%Y-%m-%d'))[:10]}"
        )
    if len(aligned):
        aligned.iloc[0] = 0.0
    return aligned


def expected_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    from quant_platform.utils.exchange_calendar import is_trading_day

    days = pd.date_range(start, end, freq="D")
    return pd.DatetimeIndex([day.normalize() for day in days
                             if is_trading_day(day.date())])


def file_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    end_candidates = [load_nav(p).index.max() for p in STRATEGIES.values() if p.is_file()]
    if len(end_candidates) != len(STRATEGIES) or not end_candidates:
        raise RuntimeError("both strategy nav.csv files are required")
    if len(set(end_candidates)) != 1:
        raise RuntimeError(f"strategy ledgers do not share a common end date: {end_candidates}")
    end = end_candidates[0]
    expected = expected_sessions(START, end)
    for name, path in STRATEGIES.items():
        actual = load_nav(path).index
        if not actual.equals(expected):
            missing = expected.difference(actual)
            unexpected = actual.difference(expected)
            raise RuntimeError(
                f"{name} ledger dates do not match exchange sessions; "
                f"missing={list(missing.strftime('%Y-%m-%d'))[:10]}, "
                f"unexpected={list(unexpected.strftime('%Y-%m-%d'))[:10]}"
            )
    b1 = b1_returns(START, end)
    if b1.empty:
        raise RuntimeError("B1 (ST-excluded investable equal-weight) is empty")
    print(f"B1 rebuilt: {len(b1)} sessions", flush=True)

    report = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
              "window": f"{START.date()}..{end.date()}",
              "input_digests": {
                  "merged_panel": file_digest(MERGED),
                  "pit_st_cache": file_digest(ST_CACHE),
                  "spec": file_digest(ROOT / "docs" / "research" / "EVALUATION_SPEC_2026-11.md"),
                  "evaluator": file_digest(Path(__file__).resolve()),
              },
              "strategies": {}}
    for name, path in STRATEGIES.items():
        nav = load_nav(path)
        if nav.empty or nav.index.max() != end:
            raise RuntimeError(f"{name} nav.csv does not cover the common end date {end.date()}")
        metrics_result = metrics(nav["nav"])
        if not all(np.isfinite(metrics_result[key]) for key in (
                "cum_return", "annualized_return", "annualized_vol",
                "sharpe_rf0", "sharpe_rf3", "max_drawdown")):
            raise RuntimeError(f"{name} metrics contain non-finite values")
        if metrics_result["n_days"] != len(expected):
            raise RuntimeError(f"{name} has unexpected sample size")
        bench_b2 = nav["benchmark_daily_return"].astype(float).copy()
        if not np.isfinite(bench_b2.to_numpy()).all():
            raise RuntimeError(f"B2 has non-finite daily returns for {name}")
        if abs(float(bench_b2.iloc[0])) > 1e-9:
            raise RuntimeError(
                f"B2 inception-day return must be 0 for {name}; "
                f"got {bench_b2.iloc[0]}"
            )
        bench_b2.iloc[0] = 0.0
        b2_cum = float((1.0 + bench_b2).prod() - 1.0)
        aligned = align_b1_returns(b1, nav.index)
        b1_cum = float((1.0 + aligned).prod() - 1.0)
        excess = {
            "b1_st_excluded_investable_ew": metrics_result["cum_return"] - b1_cum,
            "b2_st_including_ew": metrics_result["cum_return"] - b2_cum,
        }
        daily_returns = metrics_result.pop("daily_returns")
        metrics_result.update({"excess": excess,
                               "cum_costs": float(nav["costs_paid"].sum()),
                               "rebalances": int((nav["turnover"] > 0).sum()),
                               "b1_cum": b1_cum, "b2_cum": b2_cum})
        report["strategies"][name] = metrics_result
        report.setdefault("daily_series", {})[name] = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "nav": float(nav.at[date, "nav"]),
                "strategy_return": float(daily_returns.loc[date]),
                "b1_st_excluded_ew_return": float(aligned.loc[date]),
                "b2_st_including_ew_return": float(bench_b2.loc[date]),
            }
            for date in nav.index
        ]
        print(json.dumps({name: metrics_result}, ensure_ascii=False, indent=2))

    strategy_names = list(STRATEGIES)
    daily_matrix = pd.DataFrame({
        name: [row["strategy_return"] for row in report["daily_series"][name]]
        for name in strategy_names
    })
    correlation = float(daily_matrix[strategy_names[0]].corr(daily_matrix[strategy_names[1]]))
    joint_required = correlation > 0.9
    decision_eligible = end >= DECISION_DATE and len(expected) >= MIN_DECISION_DAYS
    individual_pass = {
        name: (values["sharpe_rf0"] >= 0.2
               and values["excess"]["b1_st_excluded_investable_ew"] > 0
               and values["excess"]["b2_st_including_ew"] > 0)
        for name, values in report["strategies"].items()
    }
    report["cross_strategy"] = {
        "daily_return_correlation": correlation,
        "joint_rule_required": joint_required,
        "decision_eligible": decision_eligible,
        "individual_pass_diagnostic": individual_pass,
        "joint_pass": (all(individual_pass.values()) if joint_required else None),
    }
    report["disclaimer"] = (
        "通过门槛 ≠ 证明有效。样本短且 Sharpe 置信区间为 IID 近似，"
        "未校正序列相关；3个月节点仅诊断，不授权提前终止或调参。"
    )

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
