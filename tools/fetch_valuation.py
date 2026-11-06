"""Refresh the Baidu PE(TTM)/PB valuation history for the forward universe.

The vendor publishes a new observation per stock on its own sampling dates
(observed roughly every few days, irregular). The forward pipeline had no
scheduled refresh, so the stored valuation last advanced on 2026-10-24 and
EP inputs kept ffilling stale values.

This tool appends only vendor-dated rows (never invents dates), drops any row
dated after today (causality), writes each asset atomically, and then rebuilds
the merged ``valuation_wide.parquet`` used by ``tools/forward_update.py``.
Re-running the same day is a no-op for assets with no new observations.

Exit codes: 0 = refreshed (or nothing new), 2 = too many failed assets or
the post-refresh vintage is so stale that the EP factor would silently decay.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOCKS = ROOT / ".scratch" / "wide_sina_2026" / "stocks"
OUT = ROOT / ".scratch" / "valuation_baidu"
PER_ASSET = OUT / "stocks"
MERGED = OUT / "valuation_wide.parquet"

MAX_FAILED_FRACTION = 0.05
MAX_VINTAGE_DAYS = 21


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def universe() -> list[str]:
    return sorted(path.stem for path in STOCKS.glob("*.parquet"))


def fetch(code: str) -> pd.DataFrame:
    import akshare as ak

    frames = []
    for indicator, column in (("市盈率(TTM)", "pe_ttm"), ("市净率", "pb_mrq")):
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                df = ak.stock_zh_valuation_baidu(
                    symbol=code, indicator=indicator, period="全部",
                )
                if df is None or df.empty:
                    raise ValueError("empty valuation frame")
                series = df.rename(columns={"date": "date", "value": column})
                series = series[["date", column]].copy()
                series["date"] = pd.to_datetime(series["date"]).dt.normalize()
                frames.append(series.set_index("date"))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.8 * attempt)
        else:
            raise RuntimeError(
                f"{indicator} unavailable: {type(last_error).__name__}: {last_error}"
            )
    return pd.concat(frames, axis=1).sort_index()


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def refresh_asset(code: str, frame: pd.DataFrame, today: dt.date) -> int:
    frame = frame[frame.index.date <= today]
    path = PER_ASSET / f"{code}.parquet"
    if path.is_file():
        existing = pd.read_parquet(path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
        latest = existing["date"].max()
        frame = frame[frame.index > latest]
        if frame.empty:
            return 0
        combined = pd.concat(
            [existing.set_index("date"), frame],
        ).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.reset_index()
        combined["asset"] = code
    else:
        combined = frame.reset_index()
        combined["asset"] = code
    atomic_write_parquet(combined, path)
    return len(frame)


def rebuild_merged() -> int:
    allowed = set(universe())
    frames = []
    for path in sorted(PER_ASSET.glob("*.parquet")):
        if path.name.endswith(".tmp") or path.stem not in allowed:
            continue
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame[["date", "asset", "pe_ttm", "pb_mrq"]])
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "asset"], keep="last")
    merged = merged.set_index(["date", "asset"]).sort_index()
    atomic_write_parquet(merged, MERGED)
    return len(merged)


def vintage_days(today: dt.date) -> dict[str, int]:
    latest = {}
    for path in sorted(PER_ASSET.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["date"])
        if frame.empty:
            continue
        latest[path.stem] = (today - frame["date"].max().date()).days
    return latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", default="", help="comma-separated subset for a smoke test")
    args = parser.parse_args()
    codes = args.codes.split(",") if args.codes else universe()
    today = dt.date.today()
    PER_ASSET.mkdir(parents=True, exist_ok=True)

    failed: list[list[str]] = []
    appended = 0
    for index, code in enumerate(codes, 1):
        try:
            new_rows = refresh_asset(code, fetch(code), today)
            appended += new_rows
        except Exception as exc:  # noqa: BLE001
            failed.append([code, f"{type(exc).__name__}: {str(exc)[:120]}"])
        time.sleep(0.15)
        if index % 50 == 0 or index == len(codes):
            log(f"[{index}/{len(codes)}] appended_rows={appended} failed={len(failed)}")

    if args.codes:
        return 0 if not failed else 2

    total_rows = rebuild_merged()
    vintage = vintage_days(today)
    median_vintage = pd.Series(list(vintage.values())).median() if vintage else None
    (OUT / "vintage.json").write_text(json.dumps({
        "refreshed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "assets": len(vintage),
        "appended_rows": appended,
        "median_vintage_days": None if median_vintage is None else float(median_vintage),
        "max_vintage_days": max(vintage.values()) if vintage else None,
        "vintage": vintage,
        "failed": failed,
    }, indent=2), encoding="utf-8")
    log(f"merged rows={total_rows} median_vintage={median_vintage} "
        f"max={max(vintage.values()) if vintage else None}")

    if failed and len(failed) > MAX_FAILED_FRACTION * len(codes):
        log(f"ERROR: {len(failed)}/{len(codes)} assets failed (>5%)")
        return 2
    if median_vintage is not None and median_vintage > MAX_VINTAGE_DAYS:
        log(f"ERROR: median valuation vintage {median_vintage} days exceeds "
            f"{MAX_VINTAGE_DAYS}; EP would silently decay")
        return 2
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
