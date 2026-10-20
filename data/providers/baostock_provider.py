"""Baostock real market data provider — free, no API key required.

Fetches real A-share prices, financials, and benchmark data from baostock.
Includes permanent caching for immutable historical data.

Contract notes
--------------
* Prices are 后复权 (hfq, ``adjustflag="1"``) with ``adj_factor`` = 1.0, so
  the pipeline applies the adjustment exactly once.
* ``volume`` is in shares and ``turnover`` is a fraction (Baostock reports
  ``turn`` in percent).
* Financials are returned as a ``(date, asset)`` panel keyed to the first
  trading day on/after each report's publication date, with ``publish_date``
  retained for point-in-time filtering and no backward fill.
* The stock universe includes delisted names (``status=0``) with
  ``delisting_date``; the sample for a requested range is built from names
  whose [listing, delisting] window overlaps that range.
* There is no ST history in this source.  ``st_available`` is False and no
  ``is_st`` column is fabricated (an all-False flag would silently pretend
  that ST screening works).
* Per-symbol failures are retried; if any symbol still fails the fetch
  raises :class:`DataFetchIncompleteError` and nothing is cached.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import pandas as pd

from quant_platform.data.providers.base import DataFetchIncompleteError, DataProvider
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "baostock"

# Bump these when cache semantics change; every cache key carries the version
# so a format/unit change can never silently replay old data.
_CACHE_VERSION = "v2"
_UNIVERSE_CACHE_VERSION = 2
_UNIVERSE_CACHE_TTL_DAYS = 7

# Board prefixes used for stratified sampling: SH main, STAR, SZ main,
# ChiNext, and Beijing (43/83/87/92).
_BOARD_PREFIXES = ("60", "68", "00", "30", "43", "83", "87", "92")
_SH_PREFIXES = ("60", "68")
_SZ_PREFIXES = ("00", "30")
_MAJOR_LIST_SIZE = 500
_SAMPLE_SEED = 42
# Dedicated JSON cache: the generic parquet/CSV helpers expect a two-level
# (date, asset) index and cannot read back a flat code list without pyarrow.
_STOCK_BASIC_CACHE_FILE = f"stock_basic_universe_v{_UNIVERSE_CACHE_VERSION}.json"

_PROFIT_FIELDS = (
    "code,pubDate,statDate,roeAvg,npMargin,gpMargin,netProfit,epsTTM,"
    "MBRevenue,totalShare,liqaShare"
)
_BALANCE_FIELDS = "code,pubDate,statDate,liabilityToAsset,assetToEquity"
_FINANCIAL_COLUMNS = [
    "market_cap", "total_assets", "net_assets", "revenue", "net_profit",
    "roe", "pb_ratio", "pe_ratio", "asset_growth", "publish_date",
]
_PRICE_COLUMNS = [
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turnover", "pct_chg", "pe_ttm", "pb_mrq", "vwap", "adj_factor",
]


def _interleave(buckets: list[list[str]]) -> list[str]:
    """Round-robin merge so the head of the list already spans all boards."""
    result: list[str] = []
    active = [iter(bucket) for bucket in buckets if bucket]
    while active:
        remaining = []
        for it in active:
            try:
                result.append(next(it))
            except StopIteration:
                continue
            remaining.append(it)
        active = remaining
    return result


def _ensure_exchange_quotas(quotas: dict[str, int], ordered: dict[str, list[str]], n: int) -> None:
    """When ``n >= 2``, never let a non-empty exchange lose all its seats.

    Proportional quotas can put every seat on one exchange for tiny, extremely
    skewed samples; move one seat from the dominant exchange when that happens.
    """
    if n < 2:
        return
    for missing_group, other_group in ((_SH_PREFIXES, _SZ_PREFIXES), (_SZ_PREFIXES, _SH_PREFIXES)):
        missing = [prefix for prefix in missing_group if prefix in ordered]
        other = [prefix for prefix in other_group if prefix in ordered]
        if not missing or not other:
            continue
        if sum(quotas[prefix] for prefix in missing) > 0:
            continue
        other_seats = sum(quotas[prefix] for prefix in other)
        if other_seats < 2:  # cannot donate without emptying the other exchange
            continue
        donor = max((prefix for prefix in other if quotas[prefix] > 0), key=quotas.get)
        recipient = max(missing, key=lambda prefix: (len(ordered[prefix]), prefix))
        quotas[donor] -= 1
        quotas[recipient] += 1


def _stratified_sample(codes, n: int, seed: int = _SAMPLE_SEED) -> list[str]:
    """Deterministically sample up to ``n`` codes across exchange boards.

    Codes are bucketed by two-digit prefix (60/68/00/30/43/83/87/92), each
    bucket is sorted and sampled with a fixed seed, so the result does not
    depend on the input ordering. Quotas are proportional to bucket size via
    the largest-remainder method; every non-empty board gets a seat when
    ``n`` allows it. The picked codes are interleaved so slicing the head
    still covers both exchanges.
    """
    buckets: dict[str, set[str]] = {prefix: set() for prefix in _BOARD_PREFIXES}
    for raw in codes:
        if not str(raw).strip():
            continue
        code = str(raw).zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        prefix = code[:2]
        if prefix in buckets:
            buckets[prefix].add(code)
    ordered = {prefix: sorted(bucket) for prefix, bucket in buckets.items() if bucket}
    total = sum(len(bucket) for bucket in ordered.values())
    if n <= 0 or total == 0:
        return []

    if total <= n:
        quotas = {prefix: len(bucket) for prefix, bucket in ordered.items()}
    else:
        quotas: dict[str, int] = {}
        remainders: dict[str, int] = {}
        for prefix, bucket in ordered.items():
            quota, remainder = divmod(n * len(bucket), total)
            quotas[prefix] = quota
            remainders[prefix] = remainder
        # Guarantee a seat for every non-empty board when the budget allows.
        if n >= len(ordered):
            for prefix in quotas:
                if quotas[prefix] == 0:
                    quotas[prefix] = 1
        assigned = sum(quotas.values())
        by_remainder = sorted(ordered, key=lambda prefix: (-remainders[prefix], prefix))
        while assigned < n:
            progressed = False
            for prefix in by_remainder:
                if assigned >= n:
                    break
                if quotas[prefix] < len(ordered[prefix]):
                    quotas[prefix] += 1
                    assigned += 1
                    progressed = True
            if not progressed:
                break
        # The minimum-one-seat rule may overshoot when n is tiny; trim it back.
        while assigned > n:
            donor = max((p for p in quotas if quotas[p] > 1), key=quotas.get, default=None)
            if donor is None:
                break
            quotas[donor] -= 1
            assigned -= 1

    _ensure_exchange_quotas(quotas, ordered, n)

    rng = random.Random(seed)
    picked = [
        rng.sample(ordered[prefix], quotas[prefix])
        for prefix in _BOARD_PREFIXES
        if prefix in ordered
    ]
    return _interleave(picked)


def _to_bs_code(code: str) -> str:
    """Convert an A-share code to baostock format (sh./sz./bj. + 6 digits).

    Accepts bare six-digit codes, ``sh.600000`` prefixes and ``600000.SH``
    suffixes.  Beijing names (4/8/920 prefixes) map to ``bj.``.
    """
    text = str(code).strip().lower()
    if "." in text:
        left, right = text.split(".", 1)
        if left in ("sh", "sz", "bj"):
            return f"{left}.{right.zfill(6)}"
        if right in ("sh", "sz", "bj"):
            return f"{right}.{left.zfill(6)}"
        text = "".join(ch for ch in left if ch.isdigit()) or text
    elif text.startswith(("sh", "sz", "bj")):
        exchange, digits = text[:2], text[2:]
        if digits.isdigit():
            return f"{exchange}.{digits.zfill(6)}"

    digits = text.zfill(6)
    if not digits.isdigit():
        raise ValueError(f"Invalid A-share code: {code!r}")
    if digits.startswith(("4", "8")) or digits.startswith("920"):
        return f"bj.{digits}"
    if digits.startswith(("6", "9")):
        return f"sh.{digits}"
    return f"sz.{digits}"


def _from_bs_code(bs_code: str) -> str:
    """Convert baostock code (sh.600000 / bj.830001) to bare 6 digits."""
    return str(bs_code).split(".")[-1].zfill(6)


def _as_date(value) -> pd.Timestamp:
    """Parse a baostock date string; empty/invalid values become NaT."""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def _as_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _empty_prices() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
    return pd.DataFrame(index=index, columns=_PRICE_COLUMNS)


def _empty_financials() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
    return pd.DataFrame(index=index, columns=_FINANCIAL_COLUMNS)


class BaostockDataProvider(DataProvider):
    """Real A-share market data from baostock.

    Features:
    - Free, no registration or API key required
    - Daily OHLCV + adj_factor for all A-share stocks
    - CSI 300 benchmark returns
    - Permanent CSV caching (historical data is immutable)
    - First run ~60-120s for full stock universe, subsequent runs <1s from cache

    ``st_available`` is False: baostock exposes no ST history (``status`` is
    the listing status, not an ST flag), so no ``is_st`` column is fabricated.
    """

    def __init__(self, cache_enabled: bool = True, max_retries: int = 2,
                 retry_backoff: float = 0.5):
        self._cache_enabled = cache_enabled
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._bs = None
        # No point-in-time ST history is supplied by this source.
        self.st_available = False

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _login(self):
        if self._bs is None:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"baostock login failed: {lg.error_msg}")
            self._bs = bs

    def _logout(self):
        if self._bs is not None:
            self._bs.logout()
            self._bs = None

    def _call_with_retries(self, what: str, func, *args, **kwargs):
        """Run one baostock call with bounded retries/backoff."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - baostock raises generic errors
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    delay = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, attempt + 1, self.max_retries, exc, delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
        raise DataFetchIncompleteError(
            f"{what} failed after {self.max_retries} attempt(s): {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Generic cache
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.parquet"

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        if not self._cache_enabled:
            return None
        # Try parquet first
        path = self._cache_path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        # Fallback to CSV
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, df: pd.DataFrame):
        if not self._cache_enabled:
            return
        try:
            df.to_parquet(self._cache_path(key), index=True)
        except Exception:
            # Fallback to CSV if parquet fails
            csv_path = self._cache_path(key).with_suffix(".csv")
            df.to_csv(csv_path, index=True)

    # ------------------------------------------------------------------
    # Daily prices
    # ------------------------------------------------------------------

    @staticmethod
    def _history_fields() -> str:
        return ("date,code,open,high,low,close,preclose,volume,amount,"
                "turn,pctChg,peTTM,pbMRQ")

    def _fetch_one_history(self, bs, bs_code: str, start_date: str, end_date: str,
                           adjustflag: str) -> list[list[str]]:
        rs = bs.query_history_k_data_plus(
            bs_code,
            self._history_fields(),
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        if rs.error_code != "0":
            raise RuntimeError(
                f"query_history_k_data_plus({bs_code}) error {rs.error_code}: "
                f"{getattr(rs, 'error_msg', '')}"
            )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(
                f"query_history_k_data_plus({bs_code}) failed mid-stream: "
                f"{rs.error_code}"
            )
        return rows

    def _fetch_daily_data(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        adjustflag: str = "1",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for multiple stocks from baostock.

        Every failed symbol is retried; if any still fails the fetch raises
        :class:`DataFetchIncompleteError` and the cache is not written, so a
        truncated panel can never be replayed as complete.
        """
        # Hash the full sorted code list: keying on len(codes) alone made any
        # two same-sized universes share one cache entry.  adjustflag is part
        # of the key because hfq and raw panels must not collide.
        codes_digest = hashlib.md5(
            ",".join(sorted(str(code).zfill(6) for code in codes)).encode()
        ).hexdigest()[:8]
        cache_key = (
            f"daily_{_CACHE_VERSION}_{start_date}_{end_date}_"
            f"{len(codes)}_{codes_digest}_adj{adjustflag}"
        )
        cached = self._load_cache(cache_key)
        if cached is not None and len(cached) > 0:
            logger.info("Cache hit: %s (%d rows)", cache_key, len(cached))
            return cached

        self._login()
        bs = self._bs

        all_rows: list[list[str]] = []
        failed: set[str] = set()
        t0 = time.time()

        for code in codes:
            bs_code = _to_bs_code(code)
            try:
                rows = self._call_with_retries(
                    f"history {bs_code}",
                    self._fetch_one_history,
                    bs, bs_code, start_date, end_date, adjustflag,
                )
                all_rows.extend(rows)
            except DataFetchIncompleteError as exc:
                logger.error("%s", exc)
                failed.add(str(code).zfill(6))

        elapsed = time.time() - t0
        logger.info("Fetched %d rows for %d stocks in %.1fs (failed: %d)",
                     len(all_rows), len(codes), elapsed, len(failed))

        if failed:
            raise DataFetchIncompleteError(
                "Baostock price fetch incomplete: "
                f"{len(failed)} symbol(s) failed after retries "
                f"(e.g. {sorted(failed)[:10]})"
            )

        if not all_rows:
            return _empty_prices()

        cols = ["date", "code", "open", "high", "low", "close",
                "preclose", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ"]
        df = pd.DataFrame(all_rows, columns=cols)
        df["code"] = df["code"].apply(_from_bs_code)

        # Convert to numeric
        for c in ["open", "high", "low", "close", "preclose", "volume",
                   "amount", "turn", "pctChg", "peTTM", "pbMRQ"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Rename to standard column names.  Baostock's `turn` is a percent;
        # the provider contract is a fraction.  Volume is already in shares,
        # amount in CNY.
        df.rename(columns={
            "turn": "turnover",
            "pctChg": "pct_chg",
            "peTTM": "pe_ttm",
            "pbMRQ": "pb_mrq",
        }, inplace=True)
        df["turnover"] = df["turnover"] / 100.0

        # VWAP approximation from raw amount/volume (same units, so it needs
        # no adjustment); only used for staleness detection by the pipeline.
        df["vwap"] = df["amount"] / df["volume"]
        df["vwap"] = df["vwap"].where(df["volume"] > 0)

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "code"]).sort_index()
        df.index.names = ["date", "asset"]
        df = df[df["close"] > 0]

        if df.empty:
            return _empty_prices()

        self._save_cache(cache_key, df)
        return df

    # ------------------------------------------------------------------
    # Universe / metadata
    # ------------------------------------------------------------------

    def _universe_cache_path(self) -> Path:
        return CACHE_DIR / _STOCK_BASIC_CACHE_FILE

    @staticmethod
    def _normalise_universe_records(payload) -> list[dict] | None:
        """Accept the current record format or the legacy bare-code list."""
        if not isinstance(payload, list) or not payload:
            return None
        records: list[dict] = []
        for item in payload:
            if isinstance(item, str):
                records.append({"code": _from_bs_code(item), "ipoDate": "",
                                "outDate": "", "status": ""})
            elif isinstance(item, dict) and item.get("code"):
                record = dict(item)
                record["code"] = _from_bs_code(str(record["code"]))
                records.append(record)
        return records or None

    def _load_stock_universe_cache(self) -> list[dict] | None:
        if not self._cache_enabled:
            return None
        path = self._universe_cache_path()
        if not path.exists():
            return None
        if _UNIVERSE_CACHE_TTL_DAYS > 0:
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds > _UNIVERSE_CACHE_TTL_DAYS * 86400:
                logger.info("Stock universe cache expired (%.1f days old)", age_seconds / 86400)
                return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return self._normalise_universe_records(payload)

    def _save_stock_universe_cache(self, records: list[dict]) -> None:
        if not self._cache_enabled:
            return
        try:
            path = self._universe_cache_path()
            path.write_text(json.dumps(records), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not cache stock universe: %s", exc)

    def _fetch_stock_basic_universe(self) -> list[dict] | None:
        """All type=1 A-share records (listed, paused and delisted).

        Returns None when the call is unavailable (older baostock, network
        failure, error result) so the caller can fall back to a static pool.
        """
        try:
            self._login()
            bs = self._bs
            rs = bs.query_stock_basic()
            if rs.error_code != "0":
                logger.warning("query_stock_basic returned error_code=%s", rs.error_code)
                return None
            fields = list(getattr(rs, "fields", None) or
                          ["code", "code_name", "ipoDate", "outDate", "type", "status"])
            i_code = fields.index("code") if "code" in fields else 0
            i_name = fields.index("code_name") if "code_name" in fields else 1
            i_ipo = fields.index("ipoDate") if "ipoDate" in fields else 2
            i_out = fields.index("outDate") if "outDate" in fields else 3
            i_type = fields.index("type") if "type" in fields else 4
            i_status = fields.index("status") if "status" in fields else 5
            records: list[dict] = []
            while rs.next():
                row = rs.get_row_data()
                try:
                    # type=1 is a stock (indices are type=2).  Status 0 means
                    # delisted and must stay in the universe for historical
                    # ranges -- filtering it out is survivorship bias.
                    if row[i_type] != "1":
                        continue
                    records.append({
                        "code": _from_bs_code(row[i_code]),
                        "code_name": row[i_name] if i_name < len(row) else "",
                        "ipoDate": row[i_ipo] if i_ipo < len(row) else "",
                        "outDate": row[i_out] if i_out < len(row) else "",
                        "status": row[i_status] if i_status < len(row) else "",
                    })
                except (IndexError, TypeError):
                    continue
            if rs.error_code != "0":
                logger.warning("query_stock_basic failed mid-stream (%s)", rs.error_code)
                return None
            return records or None
        except Exception as exc:
            logger.warning("query_stock_basic unavailable (%s); using fallback pool", exc)
            return None

    def _get_universe_records(self) -> list[dict] | None:
        records = self._load_stock_universe_cache()
        if records is None:
            records = self._fetch_stock_basic_universe()
            if records:
                self._save_stock_universe_cache(records)
        return records

    @staticmethod
    def _records_for_range(
        records: list[dict],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[str]:
        """Codes whose [listing, delisting] window overlaps the range."""
        start = _as_date(start_date) if start_date else pd.NaT
        end = _as_date(end_date) if end_date else pd.NaT
        codes: list[str] = []
        for record in records:
            ipo = _as_date(record.get("ipoDate"))
            out = _as_date(record.get("outDate"))
            if pd.notna(end) and pd.notna(ipo) and ipo > end:
                continue
            if pd.notna(start) and pd.notna(out) and out < start:
                continue
            codes.append(str(record["code"]).zfill(6))
        return codes

    def _get_major_stocks_list(
        self,
        n: int = _MAJOR_LIST_SIZE,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[str]:
        """Deterministic cross-board sample of up to ``n`` A-share stocks.

        Uses baostock's real security list (``query_stock_basic``) including
        delisted names whose window overlaps the requested range, and falls
        back to the static cross-exchange candidate pool when unavailable.
        Unlike the old ``sorted(...)[:500]`` this never truncates to Shenzhen
        main-board codes only.
        """
        records = self._get_universe_records()
        if records:
            universe = self._records_for_range(records, start_date, end_date)
        else:
            logger.warning("stock universe unavailable; sampling the cross-exchange fallback pool")
            universe = sorted(self._get_major_stocks())
        return _stratified_sample(universe, n)

    def _get_major_stocks(self) -> set:
        """Fallback: predefined list of ~500 major A-share stocks."""
        codes = set()
        # SH main board: 600000-603999
        for i in range(600000, 604000):
            codes.add(str(i).zfill(6))
        # SZ main board: 000001-002999
        for i in range(1, 3000):
            codes.add(str(i).zfill(6))
        # ChiNext: 300001-301999
        for i in range(300001, 302000):
            codes.add(str(i).zfill(6))
        # STAR: 688001-688999
        for i in range(688001, 689000):
            codes.add(str(i).zfill(6))
        return codes

    def get_metadata(self) -> pd.DataFrame:
        """Get stock metadata, including delisted names with ``outDate``.

        ``is_st`` is intentionally absent (``st_available`` is False): the
        source has no ST history and an all-False column would silently
        disable ST screening.
        """
        cache_key = f"metadata_{_CACHE_VERSION}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            cached.attrs["st_available"] = False
            return cached

        records = self._get_universe_records()
        rows = []
        if records:
            for record in records:
                rows.append({
                    "asset": str(record["code"]).zfill(6),
                    "listing_date": _as_date(record.get("ipoDate")),
                    "delisting_date": _as_date(record.get("outDate")),
                })
        else:
            logger.warning(
                "query_stock_basic unavailable: metadata falls back to a "
                "current-code sample with an assumed listing date; delisting "
                "dates are unknown and ST screening is unavailable"
            )
            rows = [{
                "asset": code,
                "listing_date": pd.Timestamp("1990-01-01"),
                "delisting_date": pd.NaT,
            } for code in self._get_major_stocks_list()]

        if not rows:
            return pd.DataFrame(
                columns=["sector", "market_cap_group", "listing_date", "delisting_date"],
                index=pd.Index([], name="asset"),
            )

        df = pd.DataFrame(rows).set_index("asset")
        df.index.name = "asset"
        df["market_cap_group"] = "mid"  # Placeholder; not used by the pipeline

        # Get real industry data
        df["sector"] = "Unknown"
        try:
            industry_map = self._get_industry_map()
            df["sector"] = df.index.map(lambda x: industry_map.get(x, "Unknown"))
        except Exception as exc:
            logger.warning("industry map unavailable: %s", exc)

        result = df[["sector", "market_cap_group", "listing_date", "delisting_date"]]
        result.attrs["st_available"] = False
        self._save_cache(cache_key, result)
        return result

    def get_st_timeseries(self) -> pd.DataFrame:
        """Explicitly expose that no ST history is available."""
        return pd.DataFrame(
            columns=["asset", "trigger_date", "announce_date", "end_date", "is_st"]
        )

    def _get_industry_map(self) -> dict[str, str]:
        """Get industry classification for all stocks."""
        cache_key = f"industry_map_{_CACHE_VERSION}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return dict(zip(cached.index, cached["industry"], strict=False))

        self._login()
        bs = self._bs

        rs = bs.query_stock_industry()
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return {}

        df = pd.DataFrame(rows, columns=rs.fields)
        df["code"] = df["code"].apply(_from_bs_code)
        df = df.set_index("code")

        # Extract industry code (e.g., "J66货币金融服务" -> "J66")
        df["industry"] = df["industry"].apply(
            lambda x: x[:3] if isinstance(x, str) and len(x) >= 3 else "Unknown")

        self._save_cache(cache_key, df[["industry"]])
        return dict(zip(df.index, df["industry"], strict=False))

    # ------------------------------------------------------------------
    # Prices (public)
    # ------------------------------------------------------------------

    @staticmethod
    def _project_price_fields(df: pd.DataFrame, fields: list[str] | None) -> pd.DataFrame:
        if not fields:
            return df
        unknown = [col for col in fields if col not in df.columns]
        if unknown:
            raise ValueError(f"Unknown price fields: {unknown}")
        return df[list(fields)]

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get daily price data for a range-overlapping stock sample."""
        codes = self._get_major_stocks_list(
            start_date=start_date, end_date=end_date,
        )
        df = self._fetch_daily_data(codes, start_date, end_date)
        if df.empty:
            return self._project_price_fields(df, fields)

        # Prices are already 后复权 (adjustflag="1"), so the pipeline's
        # divisor is 1; the adjustment is applied exactly once.
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0

        return self._project_price_fields(df, fields)

    # ------------------------------------------------------------------
    # Financials (public)
    # ------------------------------------------------------------------

    @staticmethod
    def _quarter_end(year: int, quarter: int) -> pd.Timestamp:
        return pd.Timestamp(
            year=year,
            month={1: 3, 2: 6, 3: 9, 4: 12}[quarter],
            day={1: 31, 2: 30, 3: 30, 4: 31}[quarter],
        )

    @classmethod
    def _quarters_between(cls, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
        """Quarters whose report can be published inside/around the range.

        Includes the previous year so YoY asset growth can be computed.
        """
        quarters = []
        for year in range(start.year - 1, end.year + 1):
            for quarter in (1, 2, 3, 4):
                qend = cls._quarter_end(year, quarter)
                if qend >= start - pd.Timedelta(days=400) and qend <= end:
                    quarters.append((year, quarter))
        return quarters

    def _profit_row(self, bs, bs_code: str, year: int, quarter: int) -> dict | None:
        rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
        if rs.error_code != "0":
            raise RuntimeError(
                f"query_profit_data({bs_code}, {year}Q{quarter}) error "
                f"{rs.error_code}: {getattr(rs, 'error_msg', '')}"
            )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        fields = list(getattr(rs, "fields", None) or [])
        row = rows[0]
        if not fields or len(fields) != len(row):
            fields = ["code", "pubDate", "statDate", "roeAvg", "npMargin",
                      "gpMargin", "netProfit", "epsTTM", "MBRevenue",
                      "totalShare", "liqaShare"][:len(row)]
        return dict(zip(fields, row, strict=False))

    def _balance_row(self, bs, bs_code: str, year: int, quarter: int) -> dict | None:
        try:
            rs = bs.query_balance_data(code=bs_code, year=year, quarter=quarter)
            if rs.error_code != "0":
                return None
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None
            fields = list(getattr(rs, "fields", None) or [])
            row = rows[0]
            if not fields or len(fields) != len(row):
                fields = ["code", "pubDate", "statDate", "liabilityToAsset",
                          "assetToEquity"][:len(row)]
            return dict(zip(fields, row, strict=False))
        except Exception as exc:  # balance data is optional for the panel
            logger.debug("balance data unavailable for %s %sQ%s: %s",
                         bs_code, year, quarter, exc)
            return None

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Quarterly fundamentals as a PIT ``(date, asset)`` panel.

        Each report is keyed to the first trading day on/after its
        publication date (``pubDate``; when absent, the first trading day
        after the period end with a warning).  Columns: market_cap,
        total_assets, net_assets, revenue, net_profit, roe, pb_ratio,
        pe_ratio, asset_growth, publish_date.  No backward fill.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        codes = self._get_major_stocks_list(
            n=100, start_date=start_date, end_date=end_date,
        )
        if not codes:
            return _empty_financials()

        codes_digest = hashlib.md5(
            ",".join(sorted(codes)).encode()
        ).hexdigest()[:8]
        cache_key = (
            f"fin_{_CACHE_VERSION}_{start_date}_{end_date}_"
            f"{len(codes)}_{codes_digest}"
        )
        cached = self._load_cache(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        # Raw (unadjusted) prices supply the price leg of market cap and the
        # trading calendar used to map publication dates.
        raw_prices = self._fetch_daily_data(
            codes, start_date, end_date, adjustflag="3",
        )
        if raw_prices.empty:
            return _empty_financials()
        raw_close = raw_prices["close"].unstack("asset")
        pb_panel = (
            raw_prices["pb_mrq"].unstack("asset")
            if "pb_mrq" in raw_prices.columns else None
        )
        pe_panel = (
            raw_prices["pe_ttm"].unstack("asset")
            if "pe_ttm" in raw_prices.columns else None
        )
        trading_dates = pd.DatetimeIndex(
            raw_prices.index.get_level_values("date").unique()
        ).sort_values()
        if len(trading_dates) == 0:
            return _empty_financials()

        self._login()
        bs = self._bs
        quarters = self._quarters_between(start, end)

        records: list[dict] = []
        failed: set[str] = set()

        for code in codes:
            bs_code = _to_bs_code(code)
            code_failed = False
            for year, quarter in quarters:
                try:
                    profit = self._call_with_retries(
                        f"profit {bs_code} {year}Q{quarter}",
                        self._profit_row,
                        bs, bs_code, year, quarter,
                    )
                except DataFetchIncompleteError as exc:
                    logger.error("%s", exc)
                    code_failed = True
                    break
                if profit is None:
                    continue

                balance = self._balance_row(bs, bs_code, year, quarter)
                stat_date = _as_date(profit.get("statDate"))
                pub_date = _as_date(profit.get("pubDate"))
                roe_pct = _as_float(profit.get("roeAvg"))
                net_profit = _as_float(profit.get("netProfit"))
                revenue = _as_float(profit.get("MBRevenue"))
                total_share = _as_float(profit.get("totalShare"))

                if pd.isna(pub_date):
                    # Fallback: first trading day strictly after period end.
                    if pd.notna(stat_date):
                        pos = trading_dates.searchsorted(stat_date, side="right")
                        if pos < len(trading_dates):
                            pub_date = trading_dates[pos]
                            logger.warning(
                                "Report %s %sQ%s has no pubDate; using first "
                                "trading day after period end (%s)",
                                bs_code, year, quarter, pub_date.date(),
                            )
                if pd.isna(pub_date):
                    logger.warning(
                        "Report %s %sQ%s has neither pubDate nor a usable "
                        "statDate; dropping it", bs_code, year, quarter,
                    )
                    continue

                pos = trading_dates.searchsorted(pub_date, side="left")
                if pos >= len(trading_dates):
                    continue  # published after the sample ends
                visible_date = trading_dates[pos]

                net_assets = float("nan")
                if math.isfinite(roe_pct) and roe_pct != 0.0:
                    net_assets = net_profit / (roe_pct / 100.0)
                total_assets = float("nan")
                if balance is not None:
                    asset_to_equity = _as_float(balance.get("assetToEquity"))
                    if math.isfinite(asset_to_equity) and math.isfinite(net_assets):
                        total_assets = net_assets * asset_to_equity

                market_cap = float("nan")
                if math.isfinite(total_share):
                    try:
                        price = float(raw_close.at[visible_date, code])
                        if math.isfinite(price) and price > 0:
                            market_cap = total_share * price
                    except KeyError:
                        pass

                def _panel_value(panel, column, code=code, visible_date=visible_date):
                    if panel is None or code not in panel.columns:
                        return float("nan")
                    try:
                        value = float(panel.at[visible_date, code])
                    except KeyError:
                        return float("nan")
                    return value if math.isfinite(value) else float("nan")

                records.append({
                    "date": visible_date,
                    "asset": code,
                    "stat_date": stat_date,
                    "market_cap": market_cap,
                    "total_assets": total_assets,
                    "net_assets": net_assets,
                    "revenue": revenue,
                    "net_profit": net_profit,
                    "roe": roe_pct / 100.0 if math.isfinite(roe_pct) else float("nan"),
                    "pb_ratio": _panel_value(pb_panel, "pb_mrq"),
                    "pe_ratio": _panel_value(pe_panel, "pe_ttm"),
                    "publish_date": pub_date,
                })
            if code_failed:
                failed.add(str(code).zfill(6))

        if failed:
            raise DataFetchIncompleteError(
                "Baostock financial fetch incomplete: "
                f"{len(failed)} symbol(s) failed after retries "
                f"(e.g. {sorted(failed)[:10]})"
            )
        if not records:
            return _empty_financials()

        df = pd.DataFrame(records)
        df = df.sort_values(["asset", "stat_date"])
        df["asset_growth"] = (
            df.groupby("asset", sort=False)["total_assets"]
            .pct_change(4, fill_method=None)
        )
        df = df.drop(columns=["stat_date"])
        df = df.drop_duplicates(subset=["date", "asset"], keep="last")
        df = df.set_index(["date", "asset"]).sort_index()

        # Forward-fill to the daily grid (no bfill: before a report's
        # publication there is genuinely nothing to know).
        full_idx = pd.MultiIndex.from_product(
            [trading_dates, pd.Index(codes, name="asset")],
            names=["date", "asset"],
        )
        df_full = df.reindex(full_idx)
        df_full = df_full.groupby("asset").ffill()

        if self._cache_enabled and len(df_full) > 0:
            self._save_cache(cache_key, df_full)
        return df_full

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Get CSI 300 daily trailing returns (pctChg / 100)."""
        cache_key = f"bench_{_CACHE_VERSION}_{start_date}_{end_date}"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            series = cached.iloc[:, 0]
            series.name = "benchmark"
            return series

        self._login()
        bs = self._bs

        rows = self._call_with_retries(
            "benchmark history",
            lambda: self._query_benchmark(bs, start_date, end_date),
        )

        if not rows:
            return pd.Series(dtype=float, name="benchmark")

        df = pd.DataFrame(rows, columns=["date", "close", "preclose", "pctChg"])
        df["date"] = pd.to_datetime(df["date"])
        df["pctChg"] = pd.to_numeric(df["pctChg"], errors="coerce") / 100
        series = df.set_index("date")["pctChg"].dropna()
        series.name = "benchmark"

        self._save_cache(cache_key, series.to_frame("benchmark_return"))
        return series

    @staticmethod
    def _query_benchmark(bs, start_date: str, end_date: str) -> list[list[str]]:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,close,preclose,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="1",
        )
        if rs.error_code != "0":
            raise RuntimeError(
                f"benchmark query error {rs.error_code}: {getattr(rs, 'error_msg', '')}"
            )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(f"benchmark query failed mid-stream: {rs.error_code}")
        return rows

    def close(self):
        """Logout from baostock."""
        self._logout()
