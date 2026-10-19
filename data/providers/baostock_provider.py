"""Baostock real market data provider — free, no API key required.

Fetches real A-share prices, financials, and benchmark data from baostock.
Includes permanent caching for immutable historical data.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "baostock"

# Board prefixes used for stratified sampling: SH main, STAR, SZ main, ChiNext.
_BOARD_PREFIXES = ("60", "68", "00", "30")
_SH_PREFIXES = ("60", "68")
_SZ_PREFIXES = ("00", "30")
_MAJOR_LIST_SIZE = 500
_SAMPLE_SEED = 42
# Dedicated JSON cache: the generic parquet/CSV helpers expect a two-level
# (date, asset) index and cannot read back a flat code list without pyarrow.
_STOCK_BASIC_CACHE_FILE = "stock_basic_universe.json"


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

    Codes are bucketed by two-digit prefix (60/68/00/30), each bucket is sorted
    and sampled with a fixed seed, so the result does not depend on the input
    ordering. Quotas are proportional to bucket size via the largest-remainder
    method; every non-empty board gets a seat when ``n`` allows it. The picked
    codes are interleaved so slicing the head still covers both exchanges.
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
    """Convert 6-digit code to baostock format (sh.600000 / sz.000001)."""
    code = str(code).zfill(6)
    code_int = int(code)
    if code_int >= 600000:
        return f"sh.{code}"
    return f"sz.{code}"


def _from_bs_code(bs_code: str) -> str:
    """Convert baostock code to 6-digit format."""
    return bs_code.split(".")[-1].zfill(6)


class BaostockDataProvider(DataProvider):
    """Real A-share market data from baostock.

    Features:
    - Free, no registration or API key required
    - Daily OHLCV + adj_factor for all A-share stocks
    - CSI 300 benchmark returns
    - Permanent CSV caching (historical data is immutable)
    - First run ~60-120s for full stock universe, subsequent runs <1s from cache
    """

    def __init__(self, cache_enabled: bool = True):
        self._cache_enabled = cache_enabled
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._bs = None

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

    def _fetch_daily_data(
        self, codes: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for multiple stocks from baostock."""
        # Hash the full sorted code list: keying on len(codes) alone made any
        # two same-sized universes share one cache entry.
        codes_digest = hashlib.md5(
            ",".join(sorted(str(code).zfill(6) for code in codes)).encode()
        ).hexdigest()[:8]
        cache_key = f"daily_{start_date}_{end_date}_{len(codes)}_{codes_digest}"
        cached = self._load_cache(cache_key)
        if cached is not None and len(cached) > 0:
            logger.info("Cache hit: %s (%d rows)", cache_key, len(cached))
            return cached

        self._login()
        import baostock as bs

        all_rows = []
        failed = 0
        t0 = time.time()

        for code in codes:
            bs_code = _to_bs_code(code)
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="1",  # 后复权: dividends/splits are already in the path
                )
                while rs.error_code == "0" and rs.next():
                    all_rows.append(rs.get_row_data())
            except Exception:
                failed += 1

        elapsed = time.time() - t0
        logger.info("Fetched %d rows for %d stocks in %.1fs (failed: %d)",
                     len(all_rows), len(codes), elapsed, failed)

        if not all_rows:
            return pd.DataFrame()

        cols = ["date", "code", "open", "high", "low", "close",
                "preclose", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ"]
        df = pd.DataFrame(all_rows, columns=cols)
        df["code"] = df["code"].apply(_from_bs_code)

        # Convert to numeric
        for c in ["open", "high", "low", "close", "preclose", "volume",
                   "amount", "turn", "pctChg", "peTTM", "pbMRQ"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Rename to standard column names
        df.rename(columns={
            "turn": "turnover",
            "pctChg": "pct_chg",
            "peTTM": "pe_ttm",
            "pbMRQ": "pb_mrq",
        }, inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "code"]).sort_index()
        df.index.names = ["date", "asset"]
        df = df[df["close"] > 0]

        self._save_cache(cache_key, df)
        return df

    def _fetch_stock_list(self) -> list[str]:
        """Get all A-share stock codes for a date range."""
        cache_key = "stock_list"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached["code"].tolist()

        self._login()
        import baostock as bs

        # Get all stocks from CSI 300 + CSI 500 + CSI 1000
        all_codes = set()
        for index_code in ["sh.000300", "sh.000905", "sh.000852"]:
            try:
                rs = bs.query_history_k_data_plus(
                    index_code, "date,code",
                    start_date="2024-01-01", end_date="2024-01-02",
                    frequency="d", adjustflag="3",
                )
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if len(row) > 1:
                        all_codes.add(_from_bs_code(row[1]))
            except Exception:
                pass

        # Fallback: use a predefined list of major stocks
        if not all_codes:
            all_codes = self._get_major_stocks()

        codes_list = sorted(all_codes)
        self._save_cache(cache_key, pd.DataFrame({"code": codes_list}))
        return codes_list

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

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get daily price data for all available stocks."""
        codes = self._get_major_stocks_list()
        df = self._fetch_daily_data(codes, start_date, end_date)
        if df.empty:
            return df

        # Add adj_factor. The prices above are already 后复权 (adjustflag="1"),
        # so the pipeline's divisor is 1; requesting unadjusted prices while
        # hardcoding this to 1 made ex-dividend dates look like crashes.
        if "adj_factor" not in df.columns:
            df["adj_factor"] = 1.0

        return df

    def _load_stock_universe_cache(self) -> list[str] | None:
        if not self._cache_enabled:
            return None
        path = CACHE_DIR / _STOCK_BASIC_CACHE_FILE
        if not path.exists():
            return None
        try:
            codes = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(codes, list) or not codes:
            return None
        return [str(code).zfill(6) for code in codes]

    def _save_stock_universe_cache(self, codes) -> None:
        if not self._cache_enabled:
            return
        try:
            path = CACHE_DIR / _STOCK_BASIC_CACHE_FILE
            path.write_text(json.dumps(sorted(codes)), encoding="utf-8")
        except OSError as exc:
            logger.warning("could not cache stock universe: %s", exc)

    def _fetch_stock_basic_universe(self) -> list[str] | None:
        """All listed A-share codes from baostock ``query_stock_basic``.

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
            i_type = fields.index("type") if "type" in fields else 4
            i_status = fields.index("status") if "status" in fields else 5
            codes = set()
            while rs.next():
                row = rs.get_row_data()
                try:
                    if row[i_type] != "1" or row[i_status] != "1":
                        continue
                    codes.add(_from_bs_code(row[i_code]))
                except (IndexError, TypeError):
                    continue
            if rs.error_code != "0":
                logger.warning("query_stock_basic failed mid-stream (%s)", rs.error_code)
                return None
            return sorted(codes) if codes else None
        except Exception as exc:
            logger.warning("query_stock_basic unavailable (%s); using fallback pool", exc)
            return None

    def _get_major_stocks_list(self, n: int = _MAJOR_LIST_SIZE) -> list[str]:
        """Deterministic cross-board sample of ~``n`` listed A-share stocks.

        Prefers baostock's real security list (``query_stock_basic``, type=1
        stocks with status=1) and falls back to the static cross-exchange
        candidate pool. Unlike the old ``sorted(...)[:500]`` this never
        truncates to Shenzhen main-board codes only.
        """
        universe = self._load_stock_universe_cache()
        if universe is None:
            universe = self._fetch_stock_basic_universe()
            if universe:
                self._save_stock_universe_cache(universe)
        if not universe:
            logger.warning("stock universe unavailable; sampling the cross-exchange fallback pool")
            universe = self._get_major_stocks()
        return _stratified_sample(universe, n)

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Get quarterly financial data from baostock."""
        cache_key = f"fin_{start_date}_{end_date}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._login()
        import baostock as bs

        codes = self._get_major_stocks_list()[:100]  # Limit for speed
        all_rows = []

        for code in codes:
            bs_code = _to_bs_code(code)
            try:
                rs = bs.query_profit_data(code=bs_code, year=2024, quarter=4)
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if row:
                        all_rows.append([code] + row)
            except Exception:
                continue

        if not all_rows:
            return pd.DataFrame()

        # Build financial DataFrame
        df = pd.DataFrame(all_rows)
        # baostock financial columns vary, use what we get
        if len(df.columns) >= 5:
            df = df.rename(columns={0: "code"})
            df = df.set_index("code")
            df.index.name = "asset"

        self._save_cache(cache_key, df)
        return df

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Get CSI 300 daily returns."""
        cache_key = f"bench_{start_date}_{end_date}"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            return cached.iloc[:, 0]

        self._login()
        import baostock as bs

        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,close,preclose,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="1",
        )

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.Series(dtype=float)

        df = pd.DataFrame(rows, columns=["date", "close", "preclose", "pctChg"])
        df["date"] = pd.to_datetime(df["date"])
        df["pctChg"] = pd.to_numeric(df["pctChg"], errors="coerce") / 100
        df = df.set_index("date")["pctChg"].dropna()

        self._save_cache(cache_key, df.to_frame("benchmark_return"))
        return df

    def get_metadata(self) -> pd.DataFrame:
        """Get stock metadata (sector info from baostock)."""
        cache_key = "metadata"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._login()
        import baostock as bs

        codes = self._get_major_stocks_list()[:500]
        rows = []

        for code in codes:
            bs_code = _to_bs_code(code)
            try:
                rs = bs.query_stock_basic(code=bs_code)
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if row:
                        rows.append([code] + list(row[1:]))
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["code", "code_name", "ipoDate",
                                          "outDate", "type", "status"])
        df = df.set_index("code")

        # Map Baostock fields to pipeline-expected columns
        df["is_st"] = df["status"].apply(
            lambda x: x in ("2", "ST", "PT", "*ST") if isinstance(x, str) else False)
        df["listing_date"] = pd.to_datetime(df["ipoDate"], errors="coerce")
        df["delisting_date"] = pd.to_datetime(df["outDate"], errors="coerce")

        # Get real industry data
        df["sector"] = "Unknown"
        try:
            industry_map = self._get_industry_map()
            df["sector"] = df.index.map(lambda x: industry_map.get(x, "Unknown"))
        except Exception:
            pass

        self._save_cache(cache_key, df)
        return df

    def _get_industry_map(self) -> dict[str, str]:
        """Get industry classification for all stocks."""
        cache_key = "industry_map"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return dict(zip(cached.index, cached["industry"], strict=False))

        self._login()
        import baostock as bs

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

    def close(self):
        """Logout from baostock."""
        self._logout()
