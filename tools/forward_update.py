# -*- coding: utf-8 -*-
"""Daily forward update with monthly rolling rebalances.

- Refetches Sina prices through today, rebuilds the merged panel.
- On the last trading days of a month, generates the next monthly target with
  daily_signal (prev-weights chained) into the strategy's target directory.
- Rebuilds each strategy's rolling paper ledger with paper_track --targets-dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WIDE = ROOT / ".scratch" / "wide_sina_2026"
STOCKS = WIDE / "stocks"
VAL = ROOT / ".scratch" / "valuation_baidu" / "valuation_wide.parquet"
MERGED = ROOT / ".scratch" / "orchestrate" / "merged_wide.parquet"
IMAP = ROOT / ".cache" / "baostock" / "industry_map.parquet"
BASE = ROOT / "docs" / "research" / "paper_track"

STRATEGIES = [
    {"name": "ep_turnover", "factors": "ep,turnover20d",
     "targets": BASE / "forward_targets" / "ep_turnover",
     "ledger": BASE / "forward_2026"},
    {"name": "turnover", "factors": "turnover20d",
     "targets": BASE / "forward_targets" / "turnover",
     "ledger": BASE / "forward_2026_turnover"},
]


def log(msg):
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def refetch():
    today = dt.date.today().strftime("%Y%m%d")
    import time

    import akshare as ak

    files = sorted(STOCKS.glob("*.parquet"))
    log(f"refetch {len(files)} stocks through {today}")
    failed = []
    for i, p in enumerate(files, 1):
        code = p.stem
        sym = ("sh" if code.startswith("6") else "sz") + code
        for attempt in (1, 2):
            try:
                df = ak.stock_zh_a_daily(symbol=sym, start_date="20180101",
                                         end_date=today, adjust="hfq")
                if df is None or df.empty:
                    raise ValueError("empty")
                df["date"] = pd.to_datetime(df["date"])
                for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df["turnover"] = df["turnover"] * 100.0
                df["pct_chg"] = df["close"].pct_change() * 100.0
                df["asset"] = code
                keep = ["date", "open", "high", "low", "close", "volume",
                        "amount", "turnover", "pct_chg", "asset"]
                df[[c for c in keep if c in df.columns]].to_parquet(p, index=False)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    failed.append([code, f"{type(exc).__name__}: {str(exc)[:80]}"])
                else:
                    time.sleep(1.0)
        time.sleep(0.15)
        if i % 50 == 0:
            log(f"[{i}/{len(files)}] failed={len(failed)}")
    log(f"refetch done failed={len(failed)}")
    return failed


def rebuild():
    frames = [pd.read_parquet(p) for p in sorted(STOCKS.glob("*.parquet"))]
    big = pd.concat(frames, ignore_index=True)
    big = big.drop_duplicates(subset=["date", "asset"], keep="last")
    big = big.set_index(["date", "asset"]).sort_index()
    price = big.reset_index()
    val = pd.read_parquet(VAL).reset_index()
    val["date"] = pd.to_datetime(val["date"])
    out = []
    for code, grp in price.groupby("asset", sort=False):
        g = grp.set_index("date").sort_index()
        v = val[val["asset"] == code].set_index("date").sort_index()
        if not v.empty:
            v = v[["pe_ttm", "pb_mrq"]].reindex(g.index, method="ffill")
            g = g.join(v, how="left")
        else:
            g["pe_ttm"] = float("nan")
            g["pb_mrq"] = float("nan")
        out.append(g.reset_index())
    merged = pd.concat(out, ignore_index=True).set_index(["date", "asset"]).sort_index()
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED)
    log(f"merged panel {merged.shape} last={merged.index.get_level_values('date').max().date()}")
    return merged


def is_month_end(latest: pd.Timestamp) -> bool:
    """True only on the last *exchange* trading day of the month."""
    from quant_platform.core.exchange_calendar import is_trading_day

    last = (latest + pd.offsets.MonthEnd(0)).date()
    while not is_trading_day(last):
        last -= dt.timedelta(days=1)
    return latest.normalize().date() == last


def latest_target(targets: Path) -> Path | None:
    files = sorted(targets.glob("target_*.json"))
    return files[-1] if files else None


_ST_CACHE: dict[str, Path | None] = {}


def current_st_codes() -> Path | None:
    """Snapshot the current ST list from Sina and cache it as a code file.

    The wide panel has no ST history, so target generation must exclude ST
    names explicitly; otherwise the frozen book can buy ST stocks (observed
    on 2026-10-23: 5 of 40 holdings were ST).
    """
    today = dt.date.today().strftime("%Y%m%d")
    if today in _ST_CACHE:
        return _ST_CACHE[today]
    path = ROOT / ".scratch" / "orchestrate" / f"st_codes_{today}.txt"
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot()
        name_col = "名称" if "名称" in df.columns else "name"
        code_col = "代码" if "代码" in df.columns else "code"
        mask = df[name_col].astype(str).str.contains("ST", case=False, na=False)
        codes = sorted({str(value)[-6:] for value in df.loc[mask, code_col]})
        path.write_text("\n".join(codes), encoding="utf-8")
        log(f"ST snapshot: {len(codes)} codes -> {path.name}")
    except Exception as exc:  # noqa: BLE001
        log(f"ST snapshot failed ({type(exc).__name__}: {exc}); month-end "
            "target generation will be SKIPPED (fail closed)")
        path = None
    _ST_CACHE[today] = path
    return path


def ensure_monthly_target(strategy, latest, st_file=None):
    tdir = strategy["targets"]
    tdir.mkdir(parents=True, exist_ok=True)
    stamp = latest.strftime("%Y-%m-%d")
    out = tdir / f"target_{stamp}.json"
    if out.exists() or not is_month_end(latest):
        return out if out.exists() else None
    if st_file is None:
        log(f"{strategy['name']} month-end target SKIPPED: the ST snapshot "
            "is unavailable and an unfiltered target is not compliant")
        return None
    prev = latest_target(tdir)
    cmd = [sys.executable, str(ROOT / "tools" / "daily_signal.py"),
           "--cache", str(MERGED), "--as-of", stamp,
           "--factors", strategy["factors"], "--industry-map", str(IMAP),
           "--out-json", str(out), "--out-csv", str(out.with_suffix(".csv"))]
    if prev is not None:
        cmd += ["--prev-weights", str(prev)]
    if st_file is not None:
        cmd += ["--exclude-codes-file", str(st_file)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log(f"{strategy['name']} month-end target {stamp} rc={r.returncode}")
    if r.returncode != 0:
        print(r.stderr[-500:])
        return None
    if prev is not None:
        log(f"{strategy['name']} new target chained from {prev.name}")
    return out


def track(strategy):
    ledger = strategy["ledger"]
    ledger.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "tools" / "paper_track.py"),
           "--targets-dir", str(strategy["targets"]), "--cache", str(MERGED),
           "--out-dir", str(ledger), "--capital", "2000000",
           "--cost-mode", "impact", "--raw-price-mode", "auto"]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log(f"{strategy['name']} ledger rc={r.returncode}")
    print((r.stdout or "")[-900:])
    if r.returncode != 0:
        print((r.stderr or "")[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refetch", action="store_true")
    args = ap.parse_args()
    if not args.skip_refetch:
        refetch()
    merged = rebuild()
    latest = merged.index.get_level_values("date").max()
    st_file = current_st_codes()
    for s in STRATEGIES:
        try:
            ensure_monthly_target(s, latest, st_file)
            track(s)
        except Exception as exc:  # noqa: BLE001
            log(f"{s['name']} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
