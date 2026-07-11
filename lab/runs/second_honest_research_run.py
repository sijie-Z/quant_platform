"""Second Honest Research Run — ROE (value/quality factor) verification.

Per Builder-mode discipline (v0.1 single Run, no Runner abstraction):
this script follows the EXACT same Trust → Factor → IC/ICIR → Registry → Report
path as first_honest_research_run.py. ZERO framework changes. Only the factor
and data-fetch paths differ (financial API call instead of daily prices).

Goal: prove the Research OS path is REUSABLE — not a one-off script.

Factor: ROE (净资产收益率) from akshare stock_financial_analysis_indicator.
         Latest fiscal-year ROE carried forward until next report. Higher ROE
         = higher quality/profitability = higher factor expected IC.

Trust metadata (identical structure, different adjust/completeness caveats):
    - data_meta.adjust = "none" (financial data, no price adjustment needed)
    - data_meta.trust_warning = ["quarterly_point_in_time"]
    - universe_meta.pit = False, same survivorship caveat as run 1

Run:  .venv/Scripts/python.exe run_slice.py  (or direct import)
"""

from __future__ import annotations

import sys
import time as time_module
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # D:/Desktop

import akshare as ak  # noqa: E402

from quant_platform.factors.evaluation import ic_summary  # noqa: E402
from quant_platform.lab.registry import DEFAULT_DB, RunStore  # noqa: E402
from quant_platform.lab.reports import generate_report  # noqa: E402

SLICE = "second_honest_research_run"

# ---- Trust metadata (same structure as first run) ---------------------------
DATA_META = {
    "provider": "akshare",
    "adjust": "none",  # financial data, no OHLCV adjustment needed
    "basis": "fiscal_year_end",
    "cross_sectional_use": True,
    "interface": "stock_financial_analysis_indicator",
    "trust_warning": ["quarterly_point_in_time", "no_pe_ttm_precision"],
    "description": "AkShare quarterly financial indicators; ROE from latest fiscal year",
}
UNIVERSE_META = {
    "provider": "CurrentConstituentUniverseProvider",
    "kind": "current_constituent",
    "pit": False,
    "bias_warning": ["survivorship_bias_possible"],
    "description": "CSI300 current constituents applied to full history (NOT PIT)",
}
FACTOR_PARAMS = {"name": "roe", "source": "stock_financial_analysis_indicator"}


def _fetch_csi300_members() -> list[str]:
    df = ak.index_stock_cons(symbol="000300")
    code_col = "品种代码" if "品种代码" in df.columns else df.columns[0]
    return [str(c).zfill(6) for c in df[code_col].tolist()]


def _fetch_prices_for_fwd(
    codes: list[str], start: str, end: str
) -> pd.DataFrame:
    """Daily prices ONLY for computing forward return (t->t+1 target var).
    Reuses tx source with retry.  Not affected by adjustment — only pct_change.
    """
    start_s = start.replace("-", "")
    end_s = end.replace("-", "") if isinstance(end, str) else end

    def _txsym(c):
        return ("sh" if c.startswith(("60", "68", "9")) else "sz") + c

    frames = {}
    for i, code in enumerate(codes):
        ok = False
        for _attempt in range(3):
            try:
                df = ak.stock_zh_a_hist_tx(
                    symbol=_txsym(code), start_date=start_s, end_date=end_s,
                    adjust="hfq",
                )
                if df is not None and not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    frames[code] = df.set_index("date")["close"].astype(float)
                    ok = True
                    break
            except Exception:
                time_module.sleep(1)
        if i and i % 50 == 0:
            print(f"[{SLICE}] prices {i}/{len(codes)} ok={len(frames)}", flush=True)
    panel = pd.DataFrame(frames)
    panel.index.name = "date"
    return panel


def _fetch_roe_scores(codes: list[str]) -> pd.DataFrame:
    """Build a (date x asset) ROE factor panel.

    AkShare's stock_financial_analysis_indicator returns quarterly records
    with the latest '净资产收益率(%)' per quarter. We group by fiscal-year-end
    date and carry forward the latest reported ROE until the next report.

    Returns date (pd.Timestamp) x asset (code) DataFrame.
    """
    roe_col = "净资产收益率(%)"
    records = []
    for i, code in enumerate(codes):
        for _attempt in range(3):
            try:
                df = ak.stock_financial_analysis_indicator(
                    symbol=code, start_year="2021",
                )
                if df is None or df.empty or roe_col not in df.columns:
                    break
                # '日期' column holds reporting-date end
                sub = df[["日期", roe_col]].rename(
                    columns={"日期": "date", roe_col: "roe"}
                )
                sub["date"] = pd.to_datetime(sub["date"])
                sub["roe"] = pd.to_numeric(sub["roe"], errors="coerce")
                sub = sub.dropna(subset=["roe"])
                sub["code"] = code
                records.append(sub)
                break
            except Exception:
                time_module.sleep(1)
        if i and i % 50 == 0:
            print(f"[{SLICE}] ROE {i}/{len(codes)} ok={len(records)}", flush=True)

    if not records:
        return pd.DataFrame()

    stacked = pd.concat(records, ignore_index=True)
    # Sort and forward-fill: for every date, the latest reported ROE is used
    stacked = stacked.sort_values(["code", "date"])
    stacked["roe"] = stacked.groupby("code")["roe"].ffill()
    # Build wide panel
    panel = stacked.pivot_table(
        index="date", columns="code", values="roe", aggfunc="last"
    )
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    return panel


def run() -> str:
    store = RunStore(DEFAULT_DB)
    inputs = {
        "slice": SLICE,
        "data_source": "akshare",
        "data_meta": DATA_META,
        "universe_provider": UNIVERSE_META["provider"],
        "universe_meta": UNIVERSE_META,
        "factor": FACTOR_PARAMS["name"],
        "factor_params": FACTOR_PARAMS,
        "warnings": [],
    }
    run_id = store.begin_run(SLICE, inputs)

    try:
        codes = _fetch_csi300_members()
        print(f"[{SLICE}] CSI300 members: {len(codes)}", flush=True)

        start, end = "2021-01-01", "2026-05-23"
        t0 = time_module.time()
        prices = _fetch_prices_for_fwd(codes, start, end)
        print(f"[{SLICE}] price panel: {prices.shape} (t={time_module.time() - t0:.0f}s)", flush=True)
        if prices.shape[1] < 50:
            raise RuntimeError(f"price panel too narrow: {prices.shape}")

        t1 = time_module.time()
        roe_panel = _fetch_roe_scores(codes)
        print(f"[{SLICE}] ROE panel: {roe_panel.shape} (t={time_module.time() - t1:.0f}s)", flush=True)
        if roe_panel.empty or roe_panel.shape[1] < 10:
            raise RuntimeError(f"ROE panel too narrow: {roe_panel.shape}")

        fwd_ret = prices.pct_change(fill_method=None).shift(-1)

        # Cross-sectional rank IC: align ROE to price dates
        # ROE is at quarterly frequency; forward-fill to daily
        roe_daily = roe_panel.reindex(fwd_ret.index, method="ffill")

        ic_series_vals = []
        for date in roe_daily.index:
            f = roe_daily.loc[date].dropna()
            r = fwd_ret.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 20:
                continue
            ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
            ic_series_vals.append((date, ic))

        ic_series = pd.Series(
            [v for _, v in ic_series_vals if pd.notna(v)],
            index=pd.DatetimeIndex([d for d, v in ic_series_vals if pd.notna(v)]),
            name="rank_ic",
        ).dropna()
        ic_stats = ic_summary(ic_series) if len(ic_series) else {}

        # Long-short daily return for honest Sharpe proxy
        daily_strategy_ret = []
        for date in ic_series.index:
            f = roe_daily.loc[date].dropna()
            r = fwd_ret.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                continue
            fr = f.loc[common].rank()
            hi = fr >= fr.quantile(0.7)
            lo = fr <= fr.quantile(0.3)
            if hi.sum() == 0 or lo.sum() == 0:
                continue
            ls_ret = r.loc[common][hi].mean() - r.loc[common][lo].mean()
            daily_strategy_ret.append((date, ls_ret))

        daily_strategy_ret = pd.Series(
            [v for _, v in daily_strategy_ret],
            index=pd.DatetimeIndex([d for d, _ in daily_strategy_ret]),
        ).dropna()

        ls_sharpe = round(
            float(daily_strategy_ret.mean() / (daily_strategy_ret.std() + 1e-12) * np.sqrt(252)),
            4,
        ) if len(daily_strategy_ret) > 5 else None

        evaluation = {
            "n_ic_obs": int(len(ic_series)),
            "ic_mean": round(float(ic_stats.get("mean_ic", float("nan"))), 4) if ic_stats else None,
            "icir": round(float(ic_stats.get("icir", float("nan"))), 4) if ic_stats else None,
            "ic_positive_ratio": round(float(ic_stats.get("ic_positive_ratio", float("nan"))), 4) if ic_stats else None,
            "long_short_sharpe_proxy": ls_sharpe,
            "dsr": {
                "status": "insufficient_trials",
                "reason": "single_research_run; DSR corrects for multiple testing, N=1 has nothing to correct",
                "observed_sharpe_proxy": ls_sharpe,
                "note": "Do not read this as formal validation. Run more factors first.",
            },
            "bh_fdr": "n/a (single factor; BH-FDR requires multiple factors)",
        }

        store.finish_run(run_id, status="success", evaluation=evaluation,
                         report_path=None, warnings=[])
        final_run = store.get_run(run_id)
        report_path = generate_report(final_run, out_dir="data/reports")
        store.finish_run(run_id, status="success", evaluation=evaluation,
                         report_path=report_path, warnings=[])

        print(f"[{SLICE}] SUCCESS run_id={run_id}", flush=True)
        print(f"[{SLICE}] IC={evaluation.get('ic_mean')} ICIR={evaluation.get('icir')}", flush=True)
        print(f"[{SLICE}] report: {report_path}", flush=True)
        print(f"[{SLICE}] registry: {DEFAULT_DB} research_runs", flush=True)
        return run_id

    except Exception as e:
        tb = traceback.format_exc()
        reason = f"{type(e).__name__}: {e}"
        store.finish_run(run_id, status="failed", reason=reason,
                         warnings=[f"traceback_excerpt: {tb[-400:]}"])
        report_path = generate_report(store.get_run(run_id), out_dir="data/reports")
        store.finish_run(run_id, status="failed", reason=reason, report_path=report_path)
        print(f"[{SLICE}] FAILED (recorded) run_id={run_id}: {e}", flush=True)
        print(f"[{SLICE}] failure report: {report_path}", flush=True)
        raise


if __name__ == "__main__":
    run()
