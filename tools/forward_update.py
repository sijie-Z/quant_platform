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


def atomic_write_parquet(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Write a parquet file without exposing a partial replacement.

    If serialization or replacement fails, the previous file remains in
    place. The temporary artifact is deliberately left for diagnosis.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=index)
    temporary.replace(path)


def refetch():
    today_date = dt.date.today()
    today = today_date.strftime("%Y%m%d")
    import time

    import akshare as ak
    from quant_platform.utils.exchange_calendar import is_trading_day

    files = sorted(STOCKS.glob("*.parquet"))
    log(f"refetch {len(files)} stocks through {today}")
    failed = []
    for i, p in enumerate(files, 1):
        code = p.stem
        sym = ("sh" if code.startswith("6") else "sz") + code
        for attempt in (1, 2, 3):
            try:
                df = ak.stock_zh_a_daily(symbol=sym, start_date="20180101",
                                         end_date=today, adjust="hfq")
                if df is None or df.empty:
                    raise ValueError("empty")
                df["date"] = pd.to_datetime(df["date"])
                latest = df["date"].max().date()
                if is_trading_day(today_date) and latest < today_date:
                    # A non-empty but stale response is a failure, not proof
                    # of suspension. Retry; never overwrite good data with it.
                    if attempt < 3:
                        time.sleep(0.8 * attempt)
                        continue
                    failed.append([code, f"stale_response:{latest}"])
                    break
                for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df["turnover"] = df["turnover"] * 100.0
                df["pct_chg"] = df["close"].pct_change() * 100.0
                df["asset"] = code
                keep = ["date", "open", "high", "low", "close", "volume",
                        "amount", "turnover", "pct_chg", "asset"]
                atomic_write_parquet(df[[c for c in keep if c in df.columns]], p,
                                     index=False)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    failed.append([code, f"{type(exc).__name__}: {str(exc)[:80]}"])
                else:
                    time.sleep(1.0)
        time.sleep(0.15)
        if i % 50 == 0:
            log(f"[{i}/{len(files)}] failed={len(failed)}")
    log(f"refetch done failed={len(failed)}")
    if failed:
        failed, suspended = confirm_suspended_stale(failed)
        if suspended:
            log("failed refreshes confirmed suspended by spot volume=0: "
                + ", ".join(sorted(str(row[0]) for row in suspended)))
    return failed


def confirm_suspended_stale(failed: list) -> tuple[list, list]:
    """Reclassify failed daily-bar fetches that spot shows as untraded today.

    A suspended stock legitimately has no current-session bar, whatever the
    provider returned (a stale last bar, an empty frame, or an error), so its
    cached last close is correct rather than stale. Reclassify only when the
    spot snapshot is structurally trustworthy: it covers the whole universe,
    a market-wide frame shows genuine trading activity, the code's own rows
    traded zero volume, and the reclassified count stays small. Anything
    doubtful keeps the fetch as a failure (fail closed).
    """
    if not failed:
        return failed, []
    try:
        spot = today_spot_snapshot()
    except Exception as exc:  # noqa: BLE001
        log(f"suspension check unavailable ({type(exc).__name__}: {exc}); "
            "failed refreshes stay failures")
        return failed, []
    code_col = "代码" if "代码" in spot.columns else "code"
    vol_col = "成交量" if "成交量" in spot.columns else "volume"
    if code_col not in spot.columns or vol_col not in spot.columns:
        log("suspension check degraded: spot snapshot lacks 代码/成交量")
        invalidate_spot_cache()
        return failed, []
    codes = spot[code_col].astype(str).str[-6:]
    volume = pd.to_numeric(spot[vol_col], errors="coerce")
    frame = pd.DataFrame({"code": codes.to_numpy(), "volume": volume.to_numpy()})
    code_max = frame.groupby("code")["volume"].max()
    code_nan = frame["volume"].isna().groupby(frame["code"]).any()
    if len(code_max) < 100:
        log(f"suspension check degraded: partial spot frame covers only "
            f"{len(code_max)} codes; failed refreshes stay failures")
        invalidate_spot_cache()
        return failed, []
    traded = int((code_max > 0).sum())
    if traded < 0.5 * len(code_max):
        log(f"suspension check degraded: only {traded}/{len(code_max)} spot "
            "codes traded today; a zeroed feed cannot confirm suspensions")
        invalidate_spot_cache()
        return failed, []
    untraded = (code_max == 0) & (~code_nan.reindex(code_max.index).fillna(True))
    no_trade = set(code_max.index[untraded])
    candidates = [row for row in failed if str(row[0]).zfill(6) in no_trade]
    expected = {p.stem[-6:] for p in STOCKS.glob("*.parquet")}
    cap = max(10, len(expected) // 20)
    if len(candidates) > cap:
        log(f"suspension check suspicious: {len(candidates)} failed codes all "
            f"show zero volume (cap {cap}); keeping every failure")
        return failed, []
    remaining, suspended = [], []
    for row in failed:
        if str(row[0]).zfill(6) in no_trade:
            suspended.append(row)
        else:
            remaining.append(row)
    return remaining, suspended


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
    atomic_write_parquet(merged, MERGED)
    log(f"merged panel {merged.shape} last={merged.index.get_level_values('date').max().date()}")
    return merged


def _month_end(value: pd.Timestamp) -> pd.Timestamp:
    """Last exchange trading day in the value's month."""
    from quant_platform.utils.exchange_calendar import is_trading_day

    last = (pd.Timestamp(value) + pd.offsets.MonthEnd(0)).date()
    while not is_trading_day(last):
        last -= dt.timedelta(days=1)
    return pd.Timestamp(last)


def is_month_end(latest: pd.Timestamp) -> bool:
    """True only on the last *exchange* trading day of the month."""
    return latest.normalize().date() == _month_end(latest).date()


def stale_cache_reason(latest: pd.Timestamp, today) -> str | None:
    """Explain a missing latest expected close after a real refetch, if any."""
    from quant_platform.utils.exchange_calendar import is_trading_day

    expected = today
    while not is_trading_day(expected):
        expected -= dt.timedelta(days=1)
    latest_date = pd.Timestamp(latest).date()
    if latest_date >= expected:
        return None
    return (
        f"merged cache ends {latest_date}, but the latest expected exchange "
        f"session is {expected}; the provider has not published that close "
        "yet. Refusing to build a forward ledger from stale data"
    )


def _valid_target(path: Path) -> bool:
    """A usable target parses, matches its filename date, and carries weights."""
    stamp = path.stem[len("target_"):]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(payload, dict):
        return False
    if str(payload.get("as_of")) != stamp:
        return False
    weights = payload.get("weights")
    return isinstance(weights, dict) and bool(weights)


def latest_target(targets: Path) -> Path | None:
    files = [path for path in sorted(targets.glob("target_*.json"))
             if _valid_target(path)]
    return files[-1] if files else None


def target_before(targets: Path, as_of: pd.Timestamp) -> Path | None:
    """Newest valid target strictly before ``as_of`` (never a future book)."""
    stamp = pd.Timestamp(as_of).normalize()
    candidates = []
    for path in targets.glob("target_*.json"):
        if not _valid_target(path):
            continue
        date = _target_date(path)
        if date is not None and date < stamp:
            candidates.append(path)
    candidates.sort()
    return candidates[-1] if candidates else None


def _watched_codes(strategy: dict) -> set[str]:
    """Codes whose stale data could change current holdings or the next book."""
    codes: set[str] = set()
    target = latest_target(strategy["targets"])
    if target is not None:
        payload = json.loads(target.read_text(encoding="utf-8"))
        codes.update(str(code).zfill(6) for code in payload.get("weights", {}))
    summary_path = strategy["ledger"] / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        book = summary.get("book") or {}
        codes.update(str(position["code"]).zfill(6)
                     for position in book.get("positions", []))
    return codes


def failed_refresh_reason(failed: list, strategies: list[dict],
                          latest: pd.Timestamp,
                          strict_month_end: bool | None = None) -> str | None:
    """Explain why failed per-asset refreshes make this update unsafe."""
    failed_codes = {str(row[0]).zfill(6) for row in failed or []}
    if not failed_codes:
        return None
    if len(failed_codes) > 5:
        return (f"{len(failed_codes)} assets failed to refresh; a mixed-date "
                "cache would corrupt the universe benchmark")
    month_end_signal = (is_month_end(latest) if strict_month_end is None
                        else strict_month_end)
    if month_end_signal:
        return (f"{len(failed_codes)} asset(s) failed to refresh before a "
                f"month-end signal: {sorted(failed_codes)[:10]}")
    impacted = []
    for strategy in strategies:
        overlap = sorted(failed_codes & _watched_codes(strategy))
        if overlap:
            impacted.append(f"{strategy['name']}={overlap[:10]}")
    if impacted:
        return "failed refreshes affect current target/book assets: " + "; ".join(impacted)
    return None


def _target_date(path: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(path.stem[len("target_"):]).normalize()
    except (ValueError, TypeError):
        return None


def overdue_month_ends(targets: Path, latest: pd.Timestamp) -> list[pd.Timestamp]:
    """Every missing exchange month-end after the first valid target."""
    files = [path for path in sorted(targets.glob("target_*.json"))
             if _valid_target(path)]
    if not files:
        return []
    latest = pd.Timestamp(latest).normalize()
    known = {date for path in files if (date := _target_date(path)) is not None}
    if not known:
        return []
    first = min(known)
    period = first.to_period("M")
    last_period = latest.to_period("M") - 1
    overdue = []
    while period <= last_period:
        as_of = _month_end(period.start_time)
        if first < as_of < latest and as_of not in known:
            overdue.append(as_of)
        period += 1
    return overdue


_ST_CACHE: dict[str, Path | None] = {}
_SPOT_CACHE: dict[str, pd.DataFrame] = {}
_SPOT_FAILURE: dict[str, tuple[float, Exception]] = {}
SPOT_TIMEOUT_SECONDS = 90.0
SPOT_FAILURE_MEMO_SECONDS = 600.0


def _today_stamp() -> str:
    return dt.date.today().strftime("%Y%m%d")


def invalidate_spot_cache() -> None:
    _SPOT_CACHE.pop(_today_stamp(), None)


def today_spot_snapshot() -> pd.DataFrame:
    """One full-market Sina spot snapshot per process, retried with a deadline.

    ``akshare.stock_zh_a_spot`` pages through Sina without request timeouts, so
    each attempt runs behind a wall-clock deadline; a hung feed must not hang
    the daily run past its gate. The frame is cached for the process so one
    run performs at most one successful full-market pagination. A total fetch
    failure is memoized for ``SPOT_FAILURE_MEMO_SECONDS`` so callers cannot
    multiply the pagination count through nested retries.
    """
    stamp = _today_stamp()
    if stamp in _SPOT_CACHE:
        return _SPOT_CACHE[stamp]
    import time

    failure = _SPOT_FAILURE.get(stamp)
    if failure is not None:
        failed_at, error = failure
        if time.monotonic() - failed_at < SPOT_FAILURE_MEMO_SECONDS:
            raise RuntimeError(
                "spot snapshot unavailable (recent failure cached): "
                f"{type(error).__name__}: {error}"
            )
    import threading

    import akshare as ak

    last_error: Exception | None = None
    for attempt in (1, 2):
        box: dict[str, object] = {}

        def _fetch() -> None:
            try:
                box["frame"] = ak.stock_zh_a_spot()
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        worker = threading.Thread(target=_fetch, daemon=True)
        worker.start()
        worker.join(SPOT_TIMEOUT_SECONDS)
        try:
            if worker.is_alive():
                raise TimeoutError(
                    f"spot snapshot exceeded {SPOT_TIMEOUT_SECONDS:.0f}s"
                )
            error = box.get("error")
            if error is not None:
                raise error  # type: ignore[misc]
            frame = box.get("frame")
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                raise ValueError("empty spot snapshot")
            _SPOT_CACHE[stamp] = frame
            _SPOT_FAILURE.pop(stamp, None)
            return frame
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log(f"spot snapshot attempt {attempt}/2 failed "
                f"({type(exc).__name__}: {exc})")
            if attempt < 2:
                time.sleep(2.0 * attempt)
    _SPOT_FAILURE[stamp] = (time.monotonic(), last_error)  # type: ignore[assignment]
    raise RuntimeError(
        "spot snapshot unavailable after 2 attempts "
        f"({type(last_error).__name__}: {last_error})"
    )


def current_st_codes(as_of: pd.Timestamp | None = None) -> Path | None:
    """Return a current ST snapshot or an exact-dated historical PIT snapshot.

    The wide panel has no ST history, so target generation must exclude ST
    names explicitly. A partial provider frame is rejected. A historical
    target may use only ``st_codes_YYYYMMDD.txt`` whose snapshot date equals
    its as_of; the current ST list is never substituted.
    """
    if as_of is not None and pd.Timestamp(as_of).date() != dt.date.today():
        target_date = pd.Timestamp(as_of).date()
        path = (ROOT / ".scratch" / "orchestrate" /
                f"st_codes_{target_date:%Y%m%d}.txt")
        if path.is_file():
            log(f"ST snapshot: using dated PIT {path.name} for target "
                f"{target_date}")
            return path
        log(f"ST snapshot failed: historical target {target_date} requires a "
            f"dated PIT ST snapshot with snapshot_date == target as_of; "
            f"expected {path.name}. The current ST list will not be used "
            "(fail closed)")
        return None
    today = dt.date.today().strftime("%Y%m%d")
    if today in _ST_CACHE:
        return _ST_CACHE[today]
    path = ROOT / ".scratch" / "orchestrate" / f"st_codes_{today}.txt"
    import time

    expected = {p.stem[-6:] for p in STOCKS.glob("*.parquet")}
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            df = today_spot_snapshot()
        except Exception as exc:  # fetch already retried; a failure is memoized
            last_error = exc
            break
        try:
            name_col = "名称" if "名称" in df.columns else "name"
            code_col = "代码" if "代码" in df.columns else "code"
            if name_col not in df.columns or code_col not in df.columns:
                raise ValueError("spot snapshot lacks 名称/代码 columns")
            covered = {str(value)[-6:] for value in df[code_col]}
            missing = sorted(expected - covered)
            if missing:
                raise ValueError(
                    f"ST snapshot misses {len(missing)} universe assets: "
                    f"{missing[:10]}"
                )
            mask = df[name_col].astype(str).str.contains("ST", case=False, na=False)
            codes = sorted({str(value)[-6:] for value in df.loc[mask, code_col]})
            floor = max(1, len(covered) // 200)
            if len(codes) < floor:
                raise ValueError(
                    f"implausible ST list: {len(codes)} codes below floor "
                    f"{floor} of {len(covered)} covered assets"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text("\n".join(codes), encoding="utf-8")
            temporary.replace(path)
            log(f"ST snapshot: {len(codes)} codes across {len(covered)} assets "
                f"-> {path.name}")
            _ST_CACHE[today] = path
            return path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            invalidate_spot_cache()
            log(f"ST snapshot validation attempt {attempt}/2 failed "
                f"({type(exc).__name__}: {exc})")
            if attempt < 2:
                time.sleep(2.0 * attempt)
    if path.is_file():
        # Written only after a full-coverage fetch succeeded earlier today;
        # identical date, so reusing it is still PIT for today's target.
        # Re-check shape because reuse must not trust a malformed file.
        try:
            reused = [line.strip()
                      for line in path.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        except OSError as exc:
            log(f"ST snapshot: same-day file unreadable "
                f"({type(exc).__name__}: {exc}); not reused")
            return None
        floor = max(10, len(expected) // 10)
        if (len(reused) >= floor
                and all(c.isdigit() and len(c) == 6 for c in reused)):
            log(f"ST snapshot: fetch failed ({type(last_error).__name__}: "
                f"{last_error}); reusing same-day snapshot {path.name} written "
                "earlier today")
            _ST_CACHE[today] = path
            return path
        log(f"ST snapshot: same-day file {path.name} failed shape checks; "
            "not reused")
    log(f"ST snapshot failed ({type(last_error).__name__}: {last_error}); "
        "month-end target generation will be SKIPPED (fail closed)")
    return None


def ensure_monthly_target(strategy, latest, st_file=None):
    latest = pd.Timestamp(latest).normalize()
    tdir = strategy["targets"]
    tdir.mkdir(parents=True, exist_ok=True)
    current_out = tdir / f"target_{latest:%Y-%m-%d}.json"
    current_valid = current_out.exists() and _valid_target(current_out)
    pending = overdue_month_ends(tdir, latest)
    if not current_valid and is_month_end(latest):
        pending.append(latest)
    if not pending:
        return current_out if current_valid else None

    generated = None
    for as_of in pending:
        stamp = as_of.strftime("%Y-%m-%d")
        out = tdir / f"target_{stamp}.json"
        overdue = as_of < latest
        if overdue:
            log(f"{strategy['name']} detected overdue month-end target "
                f"{stamp}; dated-PIT catch-up required")
            target_st = current_st_codes(as_of)
            if target_st is None:
                expected = (ROOT / ".scratch" / "orchestrate" /
                            f"st_codes_{as_of:%Y%m%d}.txt")
                raise RuntimeError(
                    f"{strategy['name']} overdue month-end target {stamp} "
                    f"is missing: a dated PIT ST snapshot is required with "
                    f"snapshot_date == target as_of ({stamp}); expected "
                    f"{expected}. Refusing to apply the current ST list"
                )
        else:
            target_st = st_file
            if target_st is None:
                raise RuntimeError(
                    f"{strategy['name']} month-end target skipped: the ST "
                    "snapshot is unavailable and an unfiltered target is not "
                    "compliant"
                )
        prev = target_before(tdir, as_of)
        cmd = [sys.executable, str(ROOT / "tools" / "daily_signal.py"),
               "--cache", str(MERGED), "--as-of", stamp,
               "--factors", strategy["factors"], "--industry-map", str(IMAP),
               "--out-json", str(out), "--out-csv", str(out.with_suffix(".csv"))]
        if prev is not None:
            cmd += ["--prev-weights", str(prev)]
        cmd += ["--exclude-codes-file", str(target_st)]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        log(f"{strategy['name']} month-end target {stamp} rc={r.returncode}")
        if r.returncode != 0:
            raise RuntimeError(
                f"{strategy['name']} target generation failed for {stamp}: "
                f"{(r.stderr or '')[-500:]}"
            )
        if prev is not None:
            log(f"{strategy['name']} new target chained from {prev.name}")
        generated = out
    return current_out if current_out.exists() and _valid_target(current_out) else generated


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
    return r.returncode


def _month_end_session_today() -> bool:
    from quant_platform.utils.exchange_calendar import is_trading_day

    today = pd.Timestamp(dt.date.today()).normalize()
    return is_trading_day(today.date()) and is_month_end(today)


def _generable_overdue(latest: pd.Timestamp) -> bool:
    """True when a missed month-end still has its dated PIT ST snapshot."""
    for strategy in STRATEGIES:
        for as_of in overdue_month_ends(strategy["targets"], latest):
            dated = (ROOT / ".scratch" / "orchestrate" /
                     f"st_codes_{as_of:%Y%m%d}.txt")
            if dated.is_file():
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refetch", action="store_true")
    args = ap.parse_args()
    if not args.skip_refetch:
        if _month_end_session_today():
            # Write today's dated PIT snapshot before any data gate can return
            # early; the overdue catch-up on a later run requires this file.
            try:
                current_st_codes(pd.Timestamp(dt.date.today()))
            except Exception as exc:  # noqa: BLE001
                log(f"early ST capture failed ({type(exc).__name__}: {exc}); "
                    "continuing with the refresh")
        failed = refetch()
        if failed:
            log(f"WARNING: {len(failed)} stocks could not be refreshed; "
                "their last cached rows will be used")
    merged = rebuild()
    latest = merged.index.get_level_values("date").max()
    if not args.skip_refetch:
        stale_reason = stale_cache_reason(latest, dt.date.today())
        if stale_reason:
            log(f"ERROR: {stale_reason}")
            return 2
        strict = is_month_end(latest) or _generable_overdue(latest)
        failed_reason = failed_refresh_reason(failed, STRATEGIES, latest, strict)
        if failed_reason:
            log(f"ERROR: {failed_reason}")
            return 2
    st_file = current_st_codes(latest) if is_month_end(latest) else None
    errors = []
    for s in STRATEGIES:
        try:
            ensure_monthly_target(s, latest, st_file)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{s['name']}: {type(exc).__name__}: {exc}")
            log(errors[-1])
        # The ledger must rebuild even when the month-end target is blocked:
        # paper_track marks the existing book and later runs rebuild all dates.
        rc = track(s)
        if rc != 0:
            errors.append(f"{s['name']}: paper_track exited {rc}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
