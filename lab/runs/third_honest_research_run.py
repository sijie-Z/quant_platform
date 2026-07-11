"""Third Honest Research Run — Low Volatility (quality/defensive factor).

Per Builder mode v0.1 single Run discipline: this 3rd factor tests whether
Registered knowledge can be deployed to any arbitary run by solely adding
content-layer scripts with zero framework changes. Only new factor logic is
added; Registry, Report, Comparison infrastructure and Trust metadata are
fully reused.

Factor: Low Volatility (volatility_60d) — 60-day daily return standard
deviation, negated (high vol -> low factor, consistent with low-vol anomaly).
Uses daily OHLCV HFQ prices from AkShare TX source.
"""

from __future__ import annotations

import sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, akshare as ak

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quant_platform.factors.evaluation import ic_summary
from quant_platform.lab.registry import DEFAULT_DB, RunStore
from quant_platform.lab.reports import generate_report

SLICE = "third_honest_research_run"

DATA_META = {
    "provider": "akshare", "adjust": "hfq", "adjust_basis": "ipo_date_per_stock",
    "cross_sectional_use": True, "interface": "stock_zh_a_hist_tx",
    "trust_warning": [], "description": "AkShare daily OHLCV, back-adjusted (hfq)",
}
UNIVERSE_META = {
    "provider": "CurrentConstituentUniverseProvider", "kind": "current_constituent",
    "pit": False, "bias_warning": ["survivorship_bias_possible"],
    "description": "CSI300 current constituents applied to full history (NOT PIT)",
}
FACTOR_PARAMS = {"name": "volatility_60d", "period": 60}


def _txsym(c): return ("sh" if c.startswith(("60","68","9")) else "sz") + c


def run() -> str:
    store = RunStore(DEFAULT_DB)
    run_id = store.begin_run(SLICE, {
        "slice": SLICE, "data_source": "akshare", "data_meta": DATA_META,
        "universe_provider": UNIVERSE_META["provider"], "universe_meta": UNIVERSE_META,
        "factor": FACTOR_PARAMS["name"], "factor_params": FACTOR_PARAMS, "warnings": [],
    })
    try:
        codes = [str(c).zfill(6) for c in ak.index_stock_cons(symbol="000300").iloc[:, 0].tolist()]
        print(f"[{SLICE}] CSI300 members: {len(codes)}", flush=True)

        start_s, end_s = "20210101", "20260523"
        frames = {}
        for i, code in enumerate(codes):
            for _ in range(3):
                try:
                    df = ak.stock_zh_a_hist_tx(symbol=_txsym(code), start_date=start_s, end_date=end_s, adjust="hfq")
                    if df is not None and not df.empty:
                        df = df.rename(columns={'date':'date','close':'close'}) if 'close' in df.columns else df
                        df["date"] = pd.to_datetime(df["date"])
                        frames[code] = df.set_index("date")["close"].astype(float)
                        break
                except Exception:
                    time.sleep(1)
            if i and i % 50 == 0:
                print(f"[{SLICE}] fetched {i}/{len(codes)} ok={len(frames)}", flush=True)

        prices = pd.DataFrame(frames); prices.index.name = "date"
        print(f"[{SLICE}] price panel: {prices.shape}", flush=True)

        # Low vol factor: negated 60d rolling std of daily returns
        rets = prices.pct_change(fill_method=None)
        factor_panel = -rets.rolling(60).std()
        fwd_ret = rets.shift(-1)

        ic_vals = []
        for date in factor_panel.index:
            f = factor_panel.loc[date].dropna(); r = fwd_ret.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) >= 20:
                ic_vals.append((date, f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")))
        ic_series = pd.Series([v for _, v in ic_vals if not np.isnan(v)],
                              index=pd.DatetimeIndex([d for d, v in ic_vals if not np.isnan(v)]), name="rank_ic").dropna()
        stats = ic_summary(ic_series) if len(ic_series) else {}

        ls_rets = []
        for date in ic_series.index:
            f = factor_panel.loc[date].dropna(); r = fwd_ret.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30: continue
            ranks = f.loc[common].rank()
            hi, lo = ranks >= ranks.quantile(0.7), ranks <= ranks.quantile(0.3)
            if hi.sum() and lo.sum():
                ls_rets.append((date, r.loc[common][hi].mean() - r.loc[common][lo].mean()))
        ls_series = pd.Series([v for _, v in ls_rets],
                              index=pd.DatetimeIndex([d for d, _ in ls_rets])).dropna()
        ls_sr = round(float(ls_series.mean() / (ls_series.std() + 1e-12) * np.sqrt(252)), 4) if len(ls_series) > 5 else None

        evaluation = {
            "n_ic_obs": int(len(ic_series)),
            "ic_mean": round(float(stats.get("mean_ic", float("nan"))), 4) if stats else None,
            "icir": round(float(stats.get("icir", float("nan"))), 4) if stats else None,
            "ic_positive_ratio": round(float(stats.get("ic_positive_ratio", float("nan"))), 4) if stats else None,
            "long_short_sharpe_proxy": ls_sr,
            "dsr": {"status": "insufficient_trials", "reason": "single_research_run",
                     "observed_sharpe_proxy": ls_sr,
                     "note": "Do not read this as formal validation."},
            "bh_fdr": "n/a",
        }
        store.finish_run(run_id, status="success", evaluation=evaluation, report_path=None, warnings=[])
        rp = generate_report(store.get_run(run_id), out_dir="data/reports")
        store.finish_run(run_id, status="success", evaluation=evaluation, report_path=rp, warnings=[])
        print(f"[{SLICE}] SUCCESS run_id={run_id} IC={evaluation['ic_mean']} ICIR={evaluation['icir']} report={rp}", flush=True)
        return run_id
    except Exception as e:
        tb = traceback.format_exc()
        reason_text = f"{type(e).__name__}: {e}"
        store.finish_run(run_id, status="failed", reason=reason_text,
                         warnings=[f"traceback_excerpt: {tb[-400:]}"])
        rp = generate_report(store.get_run(run_id), out_dir="data/reports")
        store.finish_run(run_id, status="failed", reason=reason_text, report_path=rp)
        print(f"[{SLICE}] FAILED run_id={run_id}: {e}", flush=True)
        raise


if __name__ == "__main__":
    run()
