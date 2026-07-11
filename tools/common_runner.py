"""通用因子回测运行器 — 把任意因子跑成 Honest Run 的单文件脚本。

基于 first_honest_research_run.py 的完整模板：数据拉取 (CSI300 hfq) →
因子计算 → IC/ICIR → 注册表存储 → 报告生成。

新增第四个因子时只需新增一个 ~15 行的因子类,然后 `python common_runner.py`。

用法:
  python common_runner.py momentum_12m
  python common_runner.py volatility_60d
  python common_runner.py roe              # 需要基本面数据,当前用财务 API
"""

from __future__ import annotations

import sys, os, time, traceback, json
from pathlib import Path

# Flat-layout requires D:/Desktop on sys.path before importing quant_platform
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # D:/Desktop
sys.path.insert(0, "D:/Desktop")

import numpy as np, pandas as pd, akshare as ak  # noqa: E402

from quant_platform.factors.evaluation import ic_summary  # noqa: E402
from quant_platform.factors.technical import Momentum12M, Volatility20D, Volatility60D, Momentum1M, Momentum3M, Momentum6M, RSIFactor, MACDFactor, TurnoverFactor  # noqa: E402
from quant_platform.lab.registry import DEFAULT_DB, RunStore  # noqa: E402
from quant_platform.lab.reports import generate_report  # noqa: E402


# ---------------------------------------------------------------------------
# 因子目录 — 每个因子是一个 compute(prices, rets) -> (date x asset) DataFrame 的函数
# ---------------------------------------------------------------------------

def _factor_momentum_12m(prices, rets):
    """过去 252 个交易日的累计收益,跳过最近 21 天"""
    factor = (1 + rets).rolling(252).apply(lambda x: x.prod() - 1).shift(-21)
    return factor

def _factor_volatility_60d(prices, rets):
    """低波动率因子 — 60 日收益标准差取负 (高波动 → 低因子值,符合低波异象)"""
    return -rets.rolling(60).std()

def _factor_momentum_1m(prices, rets):
    return rets.rolling(21).apply(lambda x: (1 + x).prod() - 1)

def _factor_momentum_3m(prices, rets):
    return rets.rolling(63).apply(lambda x: (1 + x).prod() - 1)

def _factor_momentum_6m(prices, rets):
    return rets.rolling(126).apply(lambda x: (1 + x).prod() - 1)

def _factor_volatility_20d(prices, rets):
    return -rets.rolling(20).std()

def _factor_rsi_14d(prices, rets):
    gain = rets.clip(lower=0).rolling(14).mean()
    loss = (-rets).clip(lower=0).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

def _factor_turnover_20d(prices, rets):
    """换手率因子 — 用价格变动幅度代理 (无真实换手率数据时)"""
    return rets.abs().rolling(20).mean()

def _factor_reversal(prices, rets):
    """5 日反转 — 近 5 日收益取负"""
    return -rets.rolling(5).apply(lambda x: (1 + x).prod() - 1)

def _factor_amplitude(prices, rets):
    """振幅因子 — 20 日平均绝对日收益"""
    return rets.abs().rolling(20).mean()

# 因子登记表
FACTOR_CATALOG = {
    "momentum_12m":     {"name": "momentum_12m",     "compute": _factor_momentum_12m,     "params": {"period": 252, "skip": 21}},
    "volatility_60d":   {"name": "volatility_60d",   "compute": _factor_volatility_60d,   "params": {"period": 60}},
    "momentum_1m":      {"name": "momentum_1m",      "compute": _factor_momentum_1m,      "params": {"period": 21}},
    "momentum_3m":      {"name": "momentum_3m",      "compute": _factor_momentum_3m,      "params": {"period": 63}},
    "momentum_6m":      {"name": "momentum_6m",      "compute": _factor_momentum_6m,      "params": {"period": 126}},
    "volatility_20d":   {"name": "volatility_20d",   "compute": _factor_volatility_20d,   "params": {"period": 20}},
    "rsi_14d":          {"name": "rsi_14d",          "compute": _factor_rsi_14d,          "params": {"period": 14}},
    "turnover_20d":     {"name": "turnover_20d",     "compute": _factor_turnover_20d,     "params": {"period": 20}},
    "reversal":         {"name": "reversal",         "compute": _factor_reversal,          "params": {"period": 5}},
    "amplitude_20d":    {"name": "amplitude_20d",    "compute": _factor_amplitude,         "params": {"period": 20}},
}


# ---------------------------------------------------------------------------
# 共享管线逻辑
# ---------------------------------------------------------------------------

def _fetch_csi300_prices() -> pd.DataFrame:
    """拉取 CSI300 全部成分股 hfq 日线行情。"""
    df = ak.index_stock_cons(symbol="000300")
    code_col = [c for c in df.columns][0]
    codes = sorted({str(c).zfill(6) for c in df[code_col].tolist()})
    print(f"[common_runner] CSI300 members: {len(codes)}", flush=True)

    def _txsym(c):
        return ("sh" if c.startswith(("60", "68", "9")) else "sz") + c

    start_s, end_s = "20210101", "20260523"
    frames = {}
    for i, code in enumerate(codes):
        for _retry in range(3):
            try:
                raw = ak.stock_zh_a_hist_tx(symbol=_txsym(code), start_date=start_s, end_date=end_s, adjust="hfq")
                if raw is not None and not raw.empty:
                    raw = raw.rename(columns={"date": "date", "close": "close"}) if "close" in raw.columns else raw
                    raw["date"] = pd.to_datetime(raw["date"])
                    frames[code] = raw.set_index("date")["close"].astype(float)
                    break
            except Exception:
                time.sleep(0.5)
        if i and i % 50 == 0:
            print(f"[common_runner] prices {i}/{len(codes)} ok={len(frames)}", flush=True)

    panel = pd.DataFrame(frames)
    panel.index.name = "date"
    return panel


def run_single_factor(factor_key: str) -> str:
    """运行单个因子的完整 Honest Run 管线。"""
    entry = FACTOR_CATALOG[factor_key]
    slice_name = f"honest_{factor_key}"

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

    store = RunStore(DEFAULT_DB)
    inputs = {
        "slice": slice_name, "data_source": "akshare", "data_meta": DATA_META,
        "universe_provider": UNIVERSE_META["provider"], "universe_meta": UNIVERSE_META,
        "factor": entry["name"], "factor_params": entry["params"], "warnings": [],
    }
    run_id = store.begin_run(slice_name, inputs)

    try:
        # ---- 1. 数据 -----
        t0 = time.time()
        prices = _fetch_csi300_prices()
        print(f"[{slice_name}] price panel: {prices.shape} (t={int(time.time() - t0)}s)", flush=True)
        if prices.shape[1] < 50:
            raise RuntimeError(f"price panel too narrow: {prices.shape}")

        rets = prices.pct_change(fill_method=None)
        fwd_ret = rets.shift(-1)

        # ---- 2. 因子 -----
        factor_panel = entry["compute"](prices, rets)
        print(f"[{slice_name}] factor computed: {factor_panel.shape}", flush=True)

        # ---- 3. 横截面 IC -----
        ic_vals = []
        for dt in factor_panel.index:
            f = factor_panel.loc[dt].dropna()
            r = fwd_ret.loc[dt].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) >= 20:
                ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
                if not np.isnan(ic):
                    ic_vals.append((dt, ic))
        ic_series = pd.Series({d: v for d, v in ic_vals})
        stats = ic_summary(ic_series) if len(ic_series) else {}
        print(f"[{slice_name}] IC obs={len(ic_series)} mean={stats.get('mean_ic', float('nan')):.4f} icir={stats.get('icir', float('nan')):.4f}", flush=True)

        # ---- 4. Long-short Sharpe proxy -----
        ls_rets = []
        for dt in ic_series.index:
            f = factor_panel.loc[dt].dropna()
            r = fwd_ret.loc[dt].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                continue
            ranks = f.loc[common].rank()
            hi = ranks >= ranks.quantile(0.7)
            lo = ranks <= ranks.quantile(0.3)
            if hi.sum() and lo.sum():
                ls_rets.append((dt, r.loc[common][hi].mean() - r.loc[common][lo].mean()))
        if ls_rets:
            ls_series = pd.Series({d: v for d, v in ls_rets}).dropna()
            ls_sharpe = round(float(ls_series.mean() / (ls_series.std() + 1e-12) * np.sqrt(252)), 4) if len(ls_series) > 5 else None
        else:
            ls_sharpe = None

        # ---- 5. 评估 & 写 Registry -----
        evaluation = {
            "n_ic_obs": int(len(ic_series)),
            "ic_mean": round(float(stats.get("mean_ic", float("nan"))), 4) if stats else None,
            "icir": round(float(stats.get("icir", float("nan"))), 4) if stats else None,
            "ic_positive_ratio": round(float(stats.get("ic_positive_ratio", float("nan"))), 4) if stats else None,
            "long_short_sharpe_proxy": ls_sharpe,
            "dsr": {"status": "insufficient_trials", "reason": "single_research_run",
                     "observed_sharpe_proxy": ls_sharpe, "note": "Do not read this as formal validation."},
            "bh_fdr": "n/a (single factor)",
            "fetch_duration_s": int(time.time() - t0),
        }
        store.finish_run(run_id, status="success", evaluation=evaluation, report_path=None, warnings=[])
        final_run = store.get_run(run_id)
        report_path = generate_report(final_run, out_dir="data/reports")
        store.finish_run(run_id, status="success", evaluation=evaluation, report_path=report_path, warnings=[])
        print(f"[{slice_name}] SUCCESS  IC={evaluation['ic_mean']}  ICIR={evaluation['icir']}  report={report_path}", flush=True)
        return run_id

    except Exception as e:
        tb = traceback.format_exc()
        reason = f"{type(e).__name__}: {e}"
        store.finish_run(run_id, status="failed", reason=reason, warnings=[f"traceback_excerpt: {tb[-400:]}"])
        rp = generate_report(store.get_run(run_id), out_dir="data/reports")
        store.finish_run(run_id, status="failed", reason=reason, report_path=rp)
        print(f"[{slice_name}] FAILED run_id={run_id}: {e}", flush=True)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target not in FACTOR_CATALOG:
            print(f"Unknown factor: {target}")
            print(f"Available: {list(FACTOR_CATALOG.keys())}")
            sys.exit(1)
        run_single_factor(target)
    else:
        print("Usage: python common_runner.py <factor_key>")
        print(f"Available keys: {list(FACTOR_CATALOG.keys())}")
