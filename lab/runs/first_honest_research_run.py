"""First Honest Research Run — v0.1 slice (Milestone 1).

Goal (NOT alpha): prove the OS can produce ONE research record that does not
hide its own untrustworthiness. Whatever the IC is — 0.02, 0, or negative —
the run is a success IF the Registry faithfully records the data lineage,
universe trust (pit=false + bias_warning), factor, evaluation, and auto-emit
report WARNINGs. A failed run that still records is also a success (criterion #6).

Pipeline (Builder mode minimum, single new path, does NOT touch the 4万行 legacy):
    AkShare (后复权 hfq)  →  CSI300 current constituents (pit=false)
    →  Momentum_12m (skip 21d)  →  cross-sectional rank IC / ICIR
    →  DSR / BH-FDR (research/validation)  →  RunStore  →  Markdown report

Trust decisions, machine-recorded into Registry:
    - adjust = "back" (hfq): back-adjusted; per-stock basis = ipo date (uniform),
      safe for cross-sectional ranking. NOT "front" because free-source 前复权
      uses a per-stock latest-date basis → cross-stock inconsistency.
    - universe.kind = "current_constituent", pit = False, bias_warning =
      ["survivorship_bias_possible"]: today's CSI300 members applied to history.
      Strict PIT is v0.2's PITConstituentProvider job. We declare, not hide, it.
    - num_trials fed to DSR honestly = 1 (we test one factor here); report shows
      DSR is only meaningful when N>1 — recorded so a reader is not misled.

Run:  .venv/Scripts/python.exe -m lab.runs.first_honest_research_run
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root's PARENT is importable, since the repo uses a flat layout
# where the repo dir name ("quant_platform") is the package name.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import akshare as ak  # noqa: E402

from quant_platform.factors.evaluation import ic_summary  # noqa: E402
from quant_platform.factors.technical import Momentum12M  # noqa: E402
from quant_platform.lab.registry import RunStore, DEFAULT_DB  # noqa: E402
from quant_platform.lab.reports import generate_report  # noqa: E402

SLICE = "first_honest_research_run"

# ---- Trust metadata (machine-recorded, not README) --------------------------
DATA_META = {
    "provider": "akshare",
    "adjust": "hfq",                       # 后复权; NOT a "true price" — a research price
    "adjust_basis": "ipo_date_per_stock",  # back-adj basis is each stock's ipo date (uniform vs front-adj)
    "cross_sectional_use": True,           # safe for cross-sectional ranking; caveat for absolute returns
    "interface": "stock_zh_a_hist_tx / stock_zh_a_hist",
    "trust_warning": [],
    "description": "AkShare daily OHLCV, back-adjusted (hfq)",
}
UNIVERSE_META = {
    "provider": "CurrentConstituentUniverseProvider",
    "kind": "current_constituent",
    "pit": False,
    "bias_warning": ["survivorship_bias_possible"],
    "description": "CSI300 current constituents applied to full history (NOT PIT)",
}
FACTOR_PARAMS = {"name": "momentum_12m", "period": 252, "skip": 21}


def _fetch_csi300_members() -> list[str]:
    """Current CSI300 constituent codes from AkShare."""
    df = ak.index_stock_cons(symbol="000300")
    # column may be '品种代码' or '000300' depending on version
    code_col = "品种代码" if "品种代码" in df.columns else df.columns[0]
    return [str(c).zfill(6) for c in df[code_col].tolist()]


def _fetch_prices(codes: list[str], start: str, end) -> pd.DataFrame:
    """Back-adjusted daily close for each code. Returns date×asset DataFrame.

    Per Builder-mode discipline: minimal retry only (retry=3, sleep=1s), no
    async/queue/cache/rate-limiter — those are future problems. Provider
    implemented against the tx interface first (Protocol over Implementation:
    the data failed on the em interface, so we swap the implementation, not the
    contract).
    """
    import time

    start_s = start.replace("-", "")
    end_s = end.replace("-", "") if isinstance(end, str) else end
    frames = {}
    fetch_fn_name = "stock_zh_a_hist_tx"  # tencent source; fallback to em below

    def _tx_symbol(code: str) -> str:
        # tx interface wants sz/sh prefix
        return ("sh" if code.startswith(("60", "68", "9")) else "sz") + code

    for i, code in enumerate(codes):
        ok = False
        for attempt in range(3):
            try:
                if fetch_fn_name == "stock_zh_a_hist_tx":
                    df = ak.stock_zh_a_hist_tx(
                        symbol=_tx_symbol(code), start_date=start_s, end_date=end_s, adjust="hfq"
                    )
                else:
                    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="hfq")
                if df is None or getattr(df, "empty", True):
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    break
                # tx columns are already lowercase (date, close); em uses 中文
                if "close" not in df.columns:
                    df = df.rename(columns={"日期": "date", "收盘": "close"})
                df["date"] = pd.to_datetime(df["date"])
                frames[code] = df.set_index("date")["close"].astype(float)
                ok = True
                break
            except Exception:
                if fetch_fn_name == "stock_zh_a_hist_tx" and attempt == 0:
                    # tx failing wholesale — fall back to em for the rest
                    fetch_fn_name = "stock_zh_a_hist"
                if attempt < 2:
                    time.sleep(1)
        if i and i % 50 == 0:
            print(f"[{SLICE}] fetched {i}/{len(codes)} ok={len(frames)}", flush=True)
    panel = pd.DataFrame(frames)
    panel.index.name = "date"
    return panel


def run() -> str:
    store = RunStore(DEFAULT_DB)
    # inputs fingerprint for reproducibility / lineage
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
        if len(codes) < 50:
            raise RuntimeError(f"CSI300 fetch too few members: {len(codes)}")
        print(f"[{SLICE}] CSI300 members: {len(codes)}", flush=True)

        start, end = "2016-01-01", "2026-05-23"
        prices = _fetch_prices(codes, start, end)
        print(f"[{SLICE}] price panel: {prices.shape}")
        if prices.shape[1] < 50:
            raise RuntimeError(f"price panel too narrow: {prices.shape}")

        # Factor: Momentum_12m (skip 21d). Returns date×asset cumulative return.
        factor_panel = Momentum12M().compute(prices)
        # Forward return for IC target: t → t+1 (point-in-time causal)
        fwd_ret = prices.pct_change(fill_method=None).shift(-1)

        # Cross-sectional rank IC series (daily, only on common assets)
        # ic_stats keys (from ic_summary): mean_ic, std_ic, icir, ic_positive_ratio
        ic_series_vals = []
        for date in factor_panel.index:
            f = factor_panel.loc[date].dropna()
            r = fwd_ret.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) >= 20:
                # single-day rank IC = spearman corr of ranks
                ic_series_vals.append(
                    (date, f.loc[common].rank().corr(r.loc[common].rank(), method="pearson"))
                )
        ic_series = pd.Series(
            [v for _, v in ic_series_vals],
            index=pd.DatetimeIndex([d for d, _ in ic_series_vals]),
            name="rank_ic",
        ).dropna()
        ic_stats = ic_summary(ic_series) if len(ic_series) else {}

        # DSR / BH-FDR: honest about single-trial limitation. Truth First means
        # we do NOT emit a DSR number as if a formal multiple-testing check was
        # performed. BH-FDR also n/a for one factor.
        daily_strategy_ret = []
        for date in factor_panel.index:
            f = factor_panel.loc[date].dropna()
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

        # Per Truth First: single factor → DSR formality is NOT applicable.
        dsr_dict = {
            "status": "insufficient_trials",
            "reason": "single_research_run; DSR corrects for multiple testing, N=1 has nothing to correct",
            "observed_sharpe_proxy": round(float(daily_strategy_ret.mean() / (daily_strategy_ret.std() + 1e-12) * np.sqrt(252)), 4) if len(daily_strategy_ret) > 5 else None,
            "note": "Do not read this as formal validation. Run more factors first, then re-evaluate with proper num_trials.",
        }

        evaluation = {
            "n_ic_obs": int(len(ic_series)),
            "ic_mean": round(float(ic_stats.get("mean_ic", float("nan"))), 4) if ic_stats else None,
            "icir": round(float(ic_stats.get("icir", float("nan"))), 4) if ic_stats else None,
            "ic_positive_ratio": round(float(ic_stats.get("ic_positive_ratio", float("nan"))), 4) if ic_stats else None,
            "dsr": dsr_dict,
            "bh_fdr": "n/a (single factor; BH-FDR requires multiple factors)",
        }

        # Persist evaluation, then generate report from the FULL persisted record
        # so the report's trust fields come straight out of the Registry.
        store.finish_run(run_id, status="success", evaluation=evaluation,
                         report_path=None, warnings=[])
        final_run = store.get_run(run_id)
        report_path = generate_report(final_run, out_dir="data/reports")
        store.finish_run(run_id, status="success", evaluation=evaluation,
                         report_path=report_path, warnings=[])

        print(f"[{SLICE}] SUCCESS run_id={run_id}", flush=True)
        print(f"[{SLICE}] IC={evaluation.get('ic_mean')} ICIR={evaluation.get('icir')}", flush=True)
        print(f"[{SLICE}] report: {report_path}", flush=True)
        print(f"[{SLICE}] registry: {DEFAULT_DB} table=research_runs", flush=True)
        return run_id

    except Exception as e:
        tb = traceback.format_exc()
        reason = f"{type(e).__name__}: {e}"
        store.finish_run(run_id, status="failed", reason=reason,
                         warnings=[f"traceback_excerpt: {tb[-400:]}"])
        report_path = generate_report(store.get_run(run_id), out_dir="data/reports")
        store.finish_run(run_id, status="failed", reason=reason,
                         report_path=report_path)
        print(f"[{SLICE}] FAILED (recorded) run_id={run_id}: {e}", flush=True)
        print(f"[{SLICE}] failure report: {report_path}", flush=True)
        raise


if __name__ == "__main__":
    run()
