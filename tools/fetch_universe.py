#!/usr/bin/env python
"""Resumable wide-pool A-share fetcher (R27).

Pulls daily back-adjusted (hfq) bars **one stock at a time** so that an
intermittently available network can still grow a wide universe across many
short runs. Each successful stock is written atomically to
``<out-dir>/stocks/<code>.parquet``; a stock that already has a file is
skipped on the next run (breakpoint resume), unless ``--refresh CODE`` names
it. After the fetch loop every ``stocks/*.parquet`` file is merged into
``<out-dir>/daily_wide.parquet`` with a ``(date, asset)`` MultiIndex.

Existing per-stock files are trusted as-is: widening ``--start``/``--end``
or switching ``--source`` does not refetch them; use ``--refresh`` for that.
Unreadable files in ``stocks/`` are recorded as ``ReadError`` failures and
excluded from the panel rather than aborting the run.

Universe selection: ``--codes A,B,C`` uses the codes verbatim (delisted names
included if asked for); otherwise ``--universe-file`` (a parquet whose index
holds 6-digit codes and whose ``delisting_date`` column is null for live
names) is filtered to listed stocks and sampled evenly down to
``--max-stocks`` positions.

Fetch semantics:

* AKShare: ``stock_zh_a_hist(symbol, period="daily", start_date, end_date,
  adjust="hfq")``; columns 日期/open/high/low/close/volume/amount/pct_chg/
  turnover are mapped to ``date/open/high/low/close/volume/amount/pct_chg/
  turnover``.
* Baostock: ``query_history_k_data_plus(..., adjustflag="1")`` with the same
  fields. Login failure (at most two attempts) aborts with exit code 1.
* A failed attempt is retried ``--retries`` times with exponential backoff
  (``--sleep`` * 1, 2, 4, ... capped at 60s per wait). ``--sleep`` is also
  the throttle between stocks. Exhausted retries are written to
  ``<out-dir>/failures.json`` and the run continues with the next stock.
  Nothing outside ``--out-dir`` is written.

Exit codes: 0 = at least one stock available (freshly fetched or resumed),
including when some stocks failed; 2 = zero stock data available or an
unusable universe/date argument; 1 = Baostock login failure.

Usage::

    python tools/fetch_universe.py \
      --source akshare --start 2018-01-01 --end 2025-12-31 \
      --universe-file .cache/baostock/metadata_v2.parquet \
      --max-stocks 300 --out-dir .scratch/wide_pool \
      --sleep 0.3 --retries 3

Tests mock the network entirely; this module never calls AKShare/Baostock at
import time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
}
BAOSTOCK_COLUMN_MAP = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "pctChg": "pct_chg",
    "turn": "turnover",
}
BAOSTOCK_FIELDS = ",".join(BAOSTOCK_COLUMN_MAP)
VALUE_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]
OUTPUT_COLUMNS = ["date", "asset", *VALUE_COLUMNS]

MAX_BACKOFF_SECONDS = 60.0
BAOSTOCK_LOGIN_ATTEMPTS = 2
DEFAULT_UNIVERSE_FILE = ".cache/baostock/metadata_v2.parquet"


class FetchError(RuntimeError):
    """A fetch attempt that failed for a classified reason."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _import_akshare() -> Any:
    import akshare as ak  # imported lazily so tests and --help need no network stack

    return ak


def _import_baostock() -> Any:
    import baostock as bs  # imported lazily so tests and --help need no network stack

    return bs


def normalize_code(value: Any) -> str:
    """Normalise ``600000`` / ``sh.600000`` / ``600000.SH`` to ``600000``."""
    code = str(value).strip()
    if "." in code:
        head, tail = code.split(".", 1)
        code = head if head.isdigit() else tail
    code = code.strip().upper()
    if not code:
        raise ValueError("empty stock code")
    if code.isdigit():
        code = code.zfill(6)
    return code


def _require_code(code: str) -> str:
    """Reject anything that is not a plain 6-digit A-share code (path safety)."""
    if len(code) == 6 and code.isascii() and code.isdigit():
        return code
    raise ValueError(f"unsupported stock code {code!r}: expected 6 ASCII digits")


def load_listed_codes(path: str | Path) -> list[str]:
    """Read a metadata parquet and return sorted codes with no delisting date."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"universe file not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    codes = [_require_code(normalize_code(code)) for code in frame.index]
    if "delisting_date" in frame.columns:
        delisted = frame["delisting_date"]
        if pd.api.types.is_datetime64_any_dtype(delisted):
            listed_mask = delisted.isna().to_numpy()
        else:
            text = delisted.astype(str).str.strip().str.lower()
            listed_mask = delisted.isna().to_numpy() | text.isin(["", "nan", "nat", "none", "null", "n/a", "na", "-"]).to_numpy()
    else:
        listed_mask = np.ones(len(frame), dtype=bool)
    return sorted({code for code, keep in zip(codes, listed_mask, strict=True) if keep})


def sample_evenly(codes: list[str], max_stocks: int | None) -> list[str]:
    """Pick at most ``max_stocks`` codes at even index positions (endpoints included)."""
    codes = list(codes)
    if max_stocks is None or max_stocks <= 0 or max_stocks >= len(codes):
        return codes
    if max_stocks == 1:
        return [codes[0]]
    positions = np.linspace(0, len(codes) - 1, max_stocks).round().astype(int)
    return [codes[index] for index in sorted(set(int(position) for position in positions))]


def resolve_codes(args: argparse.Namespace) -> list[str]:
    if args.codes:
        seen: set[str] = set()
        codes: list[str] = []
        for chunk in args.codes.split(","):
            if not chunk.strip():
                continue
            code = _require_code(normalize_code(chunk))
            if code not in seen:
                seen.add(code)
                codes.append(code)
        if not codes:
            raise ValueError("--codes did not contain a usable stock code")
        return codes
    if not args.universe_file:
        raise ValueError("either --codes or --universe-file is required")
    listed = load_listed_codes(args.universe_file)
    return sample_evenly(listed, args.max_stocks)


def _normalize_frame(raw: pd.DataFrame | None, mapping: dict[str, str], code: str) -> pd.DataFrame:
    """Map a provider response onto OUTPUT_COLUMNS, or raise FetchError."""
    if raw is None:
        raise FetchError("EmptyData", f"no data returned for {code}")
    missing = [column for column in mapping if column not in raw.columns]
    if missing:
        raise FetchError("MissingColumns", f"response for {code} missing columns: {missing}")
    if raw.empty:
        raise FetchError("EmptyData", f"no rows returned for {code}")
    frame = raw.loc[:, list(mapping)].rename(columns=mapping)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise FetchError("EmptyData", f"no usable dates returned for {code}")
    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    frame.insert(1, "asset", code)
    return frame.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def fetch_akshare(ak: Any, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="hfq",
    )
    return _normalize_frame(raw, AKSHARE_COLUMN_MAP, code)


def _baostock_code(code: str) -> str:
    if code.startswith(("6", "5", "9")):
        return f"sh.{code}"
    if code.startswith(("4", "8")):
        return f"bj.{code}"
    return f"sz.{code}"


def fetch_baostock(bs: Any, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    result = bs.query_history_k_data_plus(
        _baostock_code(code),
        BAOSTOCK_FIELDS,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="1",
    )
    if result.error_code != "0":
        raise FetchError("BaostockQuery", f"{result.error_code}: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    fields = getattr(result, "fields", BAOSTOCK_FIELDS.split(","))
    raw = pd.DataFrame(rows, columns=fields)
    return _normalize_frame(raw, BAOSTOCK_COLUMN_MAP, code)


def baostock_login(bs: Any, sleep_fn: Callable[[float], None]) -> str | None:
    """Log in, retrying once. Returns None on success, else an error message."""
    last: Any = None
    for attempt in range(BAOSTOCK_LOGIN_ATTEMPTS):
        last = bs.login()
        if last.error_code == "0":
            return None
        if attempt + 1 < BAOSTOCK_LOGIN_ATTEMPTS:
            sleep_fn(1.0)
    detail = f"{getattr(last, 'error_code', '?')} {getattr(last, 'error_msg', '')}".strip()
    return f"login failed after {BAOSTOCK_LOGIN_ATTEMPTS} attempts: {detail}"


def describe_error(exc: BaseException | None) -> tuple[str, str]:
    if exc is None:  # pragma: no cover - defensive; the retry loop always records one
        return "Unknown", "unknown error"
    kind = getattr(exc, "kind", None) or type(exc).__name__
    message = str(exc).strip() or repr(exc)
    if len(message) > 300:
        message = message[:297] + "..."
    return str(kind), message


def fetch_with_retries(
    fetch_call: Callable[[], pd.DataFrame],
    retries: int,
    sleep_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[pd.DataFrame | None, int, tuple[str, str] | None]:
    """Call ``fetch_call`` up to ``retries + 1`` times with exponential backoff."""
    attempts = 0
    last_exc: BaseException | None = None
    while attempts <= retries:
        attempts += 1
        try:
            return fetch_call(), attempts, None
        except Exception as exc:
            last_exc = exc
            if attempts <= retries:
                backoff = min(sleep_seconds * (2 ** (attempts - 1)), MAX_BACKOFF_SECONDS)
                sleep_fn(backoff)
    return None, attempts, describe_error(last_exc)


def write_stock(frame: pd.DataFrame, code: str, stocks_dir: Path) -> None:
    """Atomically write one stock parquet, replacing any previous file."""
    stocks_dir.mkdir(parents=True, exist_ok=True)
    target = stocks_dir / f"{code}.parquet"
    tmp = stocks_dir / f".{code}.tmp"
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def aggregate_stocks(
    stocks_dir: Path,
    out_path: Path,
    errors: list[dict[str, Any]] | None = None,
) -> pd.DataFrame | None:
    """Merge ``stocks/*.parquet`` into a deduplicated (date, asset) panel.

    When ``errors`` is a list, unreadable files are appended to it as
    ``ReadError`` entries (and warned about) instead of aborting the run.
    """
    files = sorted(Path(stocks_dir).glob("*.parquet"))
    frames = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
            if "date" not in frame.columns:
                frame = frame.reset_index()
            if "asset" not in frame.columns:
                frame["asset"] = path.stem
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"])
            if frame.empty:
                continue
            for column in OUTPUT_COLUMNS:
                if column not in frame.columns:
                    frame[column] = np.nan
            frames.append(frame.loc[:, OUTPUT_COLUMNS])
        except Exception as exc:
            if errors is None:
                raise
            message = f"{type(exc).__name__}: {exc}"
            if len(message) > 300:
                message = message[:297] + "..."
            errors.append({"code": path.stem, "error_type": "ReadError", "reason": message, "attempts": 0})
            print(f"WARN: cannot read {path}: {message}", file=sys.stderr)
    if not frames:
        return None
    panel = pd.concat(frames, ignore_index=True)
    panel["asset"] = panel["asset"].astype(str)
    panel = panel.drop_duplicates(subset=["date", "asset"], keep="last")
    panel = panel.sort_values(["date", "asset"]).set_index(["date", "asset"]).sort_index()
    panel = panel.loc[:, VALUE_COLUMNS]
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        panel.to_parquet(tmp)
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return panel


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(
    args: argparse.Namespace,
    *,
    sleep_fn: Callable[[float], None] | None = None,
    ak_module: Any = None,
    bs_module: Any = None,
) -> int:
    if sleep_fn is None:
        sleep_fn = _sleep
    start, end = args.start, args.end
    if start > end:
        print(f"ERROR: --start {start.date()} is after --end {end.date()}", file=sys.stderr)
        return 2
    try:
        codes = resolve_codes(args)
    except Exception as exc:  # user input / parquet read errors are reported, not raised
        print(f"ERROR: cannot resolve stock universe: {exc}", file=sys.stderr)
        return 2
    if not codes:
        print("WARN: universe selection produced 0 codes", file=sys.stderr)

    try:
        refresh = {_require_code(normalize_code(code)) for code in (args.refresh or [])}
    except ValueError as exc:
        print(f"ERROR: invalid --refresh code: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    stocks_dir = out_dir / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in stocks_dir.glob("*.parquet")}
    selected = set(codes)
    to_fetch = [code for code in codes if code in refresh or code not in existing]
    for code in sorted(refresh - selected):
        if code in existing:
            to_fetch.append(code)
        else:
            print(f"WARN: --refresh {code}: not selected and no existing file; skipped", file=sys.stderr)
    skipped = [code for code in codes if code not in to_fetch]

    fetch_call: Callable[[str], pd.DataFrame]
    bs: Any = None
    bs_logged_in = False
    if args.source == "akshare":
        ak = ak_module if ak_module is not None else _import_akshare()
        fetch_call = lambda code: fetch_akshare(ak, code, start, end)  # noqa: E731
    else:
        bs = bs_module if bs_module is not None else _import_baostock()
        if to_fetch:  # a fully resumed pool needs no login at all
            login_error = baostock_login(bs, sleep_fn)
            if login_error is not None:
                print(f"ERROR: Baostock {login_error}", file=sys.stderr)
                return 1
            bs_logged_in = True
        fetch_call = lambda code: fetch_baostock(bs, code, start, end)  # noqa: E731

    failures: list[dict[str, Any]] = []
    fetched: list[str] = []
    try:
        total = len(to_fetch)
        for index, code in enumerate(to_fetch, start=1):
            frame, attempts, error = fetch_with_retries(
                lambda code=code: fetch_call(code), args.retries, args.sleep, sleep_fn
            )
            if frame is not None:
                try:
                    write_stock(frame, code, stocks_dir)
                except OSError as exc:
                    frame = None
                    error = ("WriteError", str(exc))
            if frame is not None:
                fetched.append(code)
                print(f"[{index}/{total}] {code}: ok ({len(frame)} rows)")
            else:
                kind, message = error if error is not None else ("Unknown", "unknown error")
                failures.append({"code": code, "error_type": kind, "reason": message, "attempts": attempts})
                print(f"[{index}/{total}] {code}: failed after {attempts} attempt(s): {kind}: {message}", file=sys.stderr)
            if index < total and args.sleep > 0:
                sleep_fn(args.sleep)
    finally:
        if bs is not None and bs_logged_in:
            try:
                bs.logout()
            except Exception:  # pragma: no cover - logout errors must not mask fetch results
                pass

    failures_path = out_dir / "failures.json"
    write_json_atomic(failures_path, {"generated_at": _utc_now(), "count": len(failures), "failures": failures})

    out_path = out_dir / "daily_wide.parquet"
    read_errors: list[dict[str, Any]] = []
    panel = aggregate_stocks(stocks_dir, out_path, errors=read_errors)
    if read_errors:  # a pre-aggregation file survives a panel-write failure
        failures.extend(read_errors)
        write_json_atomic(failures_path, {"generated_at": _utc_now(), "count": len(failures), "failures": failures})
    if panel is None and out_path.exists():
        print(f"WARN: {out_path} was not rebuilt this run and may be stale", file=sys.stderr)
    success_codes = sorted(panel.index.get_level_values("asset").unique()) if panel is not None else []

    distribution = Counter(failure["error_type"] for failure in failures)
    print()
    print(f"Universe selected : {len(codes)}")
    print(f"Fetched this run  : {len(fetched)}")
    print(f"Skipped (resume)  : {len(skipped)}")
    print(f"Failed this run   : {len(failures)}")
    print(f"Stocks available  : {len(success_codes)}")
    print(f"Failures file     : {failures_path}")
    if distribution:
        print("Failure reasons   : " + ", ".join(f"{kind}={count}" for kind, count in sorted(distribution.items())))
    if panel is not None:
        print(f"Wide panel        : {panel.shape[0]} rows x {panel.shape[1]} cols -> {out_path}")
    if not success_codes:
        print("ERROR: no stock data available (0 successes); exit code 2", file=sys.stderr)
        return 2
    return 0


def _date_arg(text: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(text, "%Y-%m-%d"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable wide-pool A-share fetcher: sample a universe, fetch one stock at a time, "
        "write stocks/<code>.parquet, resume across runs, aggregate to daily_wide.parquet.",
    )
    parser.add_argument("--source", choices=["akshare", "baostock"], default="akshare", help="data source (default: akshare)")
    parser.add_argument("--start", type=_date_arg, required=True, help="first date, YYYY-MM-DD")
    parser.add_argument("--end", type=_date_arg, required=True, help="last date, YYYY-MM-DD")
    parser.add_argument(
        "--universe-file",
        default=DEFAULT_UNIVERSE_FILE,
        help=f"metadata parquet (index = 6-digit codes, delisting_date column); default: {DEFAULT_UNIVERSE_FILE}",
    )
    parser.add_argument("--codes", default=None, help="comma-separated codes; overrides --universe-file")
    parser.add_argument("--max-stocks", type=_positive_int, default=300, help="evenly sample at most this many stocks (default: 300)")
    parser.add_argument("--out-dir", default=".scratch/wide_pool", help="output directory (default: .scratch/wide_pool)")
    parser.add_argument("--sleep", type=_non_negative_float, default=0.3, help="throttle between stocks, also the retry backoff base (default: 0.3)")
    parser.add_argument("--retries", type=_non_negative_int, default=3, help="retries after the first attempt (default: 3)")
    parser.add_argument("--refresh", action="append", default=None, metavar="CODE", help="force refetch of CODE; repeatable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
