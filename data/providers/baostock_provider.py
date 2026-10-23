"""Baostock real market data provider — free, no API key required.

Fetches real A-share prices, financials, and benchmark data from baostock.
Includes permanent caching for immutable historical data.

Contract notes
--------------
* Prices are returned as **raw** OHLCV (``adjustflag="3"``) plus the
  corporate-action ``adj_factor`` divisor.  With the default
  ``adjustment="hfq"`` the provider also fetches the 后复权 series
  (``adjustflag="1"``) and derives ``adj_factor = raw_close / hfq_close``,
  so the pipeline's ``close_adj = close / adj_factor`` reproduces the hfq
  levels research used before this wave while ``close`` is a tradeable CNY
  price (D15/D18).  Pass ``adjustment="raw"`` to skip the second fetch and
  run with ``adj_factor=1`` (``adjustment_factor_available`` is then False
  and the research series is raw).  Never label hfq data as raw.
* ``volume`` is in shares and ``turnover`` is a fraction (Baostock reports
  ``turn`` in percent).
* Financials are returned as a ``(date, asset)`` panel keyed to the first
  trading day on/after each report's publication date, with ``publish_date``
  retained for point-in-time filtering and no backward fill.  A report
  without a usable ``pubDate`` has no honest visibility date and is dropped
  (with a warning); the fiscal period end is never used as a substitute,
  because that would make the figures visible weeks before the announcement.
* The stock universe includes delisted names (``status=0``) with
  ``delisting_date``; the sample for a requested range is built from names
  whose [listing, delisting] window overlaps that range.
* ST history is derived from the daily ``isST`` column, requested in the same
  ``query_history_k_data_plus`` call as the prices (no extra request).  A run
  of consecutive ``isST == "1"`` observations is one episode whose
  ``announce_date`` is the first day ST was actually observed (conservative
  point-in-time: nothing is inferred before the fetched sample) and whose
  ``trigger_date`` equals ``announce_date``, because baostock exposes no
  earlier announcement date.  ``st_available`` is True only when the derived
  episode table is non-empty; a response without ``isST`` (missing column or
  only empty values) yields an empty episode table rather than a fabricated
  ``is_st`` flag.
* Per-symbol failures are retried; if any symbol still fails the fetch
  raises :class:`DataFetchIncompleteError` and nothing is cached.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import (
    CORPORATE_ACTION_COLUMNS,
    INCREMENTAL_OVERLAP_SESSIONS,
    DataFetchIncompleteError,
    DataProvider,
    cache_manifest_matches,
    canonicalize_panel_index,
    complete_panel_marker,
    deduplicate_panel,
    empty_corporate_actions,
    exchange_sessions,
    incremental_fetch_start,
    incremental_marker_extra,
    last_day_missing_symbols,
    normalize_corporate_actions,
    panel_last_date,
    provider_cache_lock,
    read_cache_manifest,
    read_incremental_status,
    sessions_after,
    symbols_on_date,
    write_cache_manifest,
    write_incremental_status,
)
from quant_platform.shared.symbols import normalize_symbol
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "baostock"

# Bump these when cache semantics change; every cache key carries the version
# so a format/unit change can never silently replay old data. The schema
# generation (DATA_SCHEMA_VERSION) is mixed into `_cache_version()` at call
# time, so one bump invalidates the pipeline pickle *and* these provider
# caches (D13).
# v3: price semantics changed from pre-adjusted hfq levels (adj_factor=1) to
# raw OHLCV plus a derived divisor; v2 files must never be replayed.
# v4: financial panels no longer fabricate visibility dates for reports whose
# pubDate is missing or unparseable; v3 financial caches may contain those
# look-ahead rows and must never be replayed either.
# v5: asset_growth is now a same-quarter-last-year ratio resolved on the full
# fetched report sequence; v4 panels may carry growth values computed across a
# shifted positional window and must not be replayed.
# v6: the daily price query also requests ``isST`` and the observed flag is
# persisted in the panel as the side column ``_is_st_observed`` so derived ST
# episodes survive cache reloads; v5 panels carry no ST evidence and must not
# be replayed (they would silently disable ST screening on real data).
_CACHE_VERSION = "v6"
#: Basis of the stored ``close`` column (raw traded price).
_PRICE_ADJUSTMENT = "raw"
#: Basis of ``close / adj_factor`` when adjustment is enabled.
_ADJUSTED_PRICE_BASIS = "hfq"
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
#: Side column carried only inside the cached daily panel: the raw ``isST``
#: observation.  Never exposed on the price contract (``get_prices`` drops it).
_ST_OBSERVED_COLUMN = "_is_st_observed"
#: Episode table contract, identical to SyntheticDataProvider's.
_ST_EPISODE_COLUMNS = [
    "asset", "trigger_date", "announce_date", "end_date", "is_st",
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


def _empty_st_timeseries() -> pd.DataFrame:
    return pd.DataFrame(columns=_ST_EPISODE_COLUMNS)


class BaostockDataProvider(DataProvider):
    """Real A-share market data from baostock.

    Features:
    - Free, no registration or API key required
    - Daily OHLCV + adj_factor for all A-share stocks
    - CSI 300 benchmark returns
    - Permanent CSV caching (historical data is immutable)
    - First run ~60-120s for full stock universe, subsequent runs <1s from cache

    ``st_available`` is True when the daily ``isST`` column produced at least
    one ST episode; the episode table is derived conservatively (see
    :meth:`get_st_timeseries`) and no ``is_st`` metadata column is fabricated.
    """

    price_adjustment = _PRICE_ADJUSTMENT
    adjusted_price_basis = _ADJUSTED_PRICE_BASIS
    #: Sessions can be appended to the parquet/CSV daily cache via
    #: :meth:`update`; universe/range changes fall back to a full fetch.
    supports_incremental = True

    def __init__(self, cache_enabled: bool = True, max_retries: int = 2,
                 retry_backoff: float = 0.5, adjustment: str = "hfq"):
        self._cache_enabled = cache_enabled
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        mode = str(adjustment or "hfq").strip().lower()
        if mode not in {"hfq", "raw"}:
            raise ValueError(
                f"adjustment must be 'hfq' (raw + derived divisor) or 'raw' "
                f"(raw only, adj_factor=1); got {adjustment!r}"
            )
        #: "hfq" = fetch raw + hfq and derive the divisor (research series is
        #: hfq, preserving prior returns); "raw" = raw only with adj_factor=1
        #: (research series is raw; adjustment_factor_available is False).
        self.adjustment_mode = mode
        self.adjustment_factor_available = mode == "hfq"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._bs = None
        #: True only after a non-empty ST episode table was derived; False when
        #: the source omitted ``isST`` or no sample day was flagged ST.
        self.st_available = False
        #: Derived ST episode table (``None`` = not resolved yet, so a later
        #: ``get_prices`` can still populate it).  Empty means "resolved: no
        #: ST evidence".
        self._st_timeseries: pd.DataFrame | None = None
        # Universe sample pinned by `cache_universe()` so the pipeline cache
        # key and the actual fetch use the *same* symbols even if the
        # underlying stock list changes between resolution and fetch.
        self._pinned_range: tuple[str, str] | None = None
        self._pinned_price_codes: list[str] | None = None
        self._pinned_financial_codes: list[str] | None = None
        #: Completion marker/provenance of the loaded panel (D32).
        self._data_status: dict = {}
        #: Panel kept when an incremental update is incomplete: the pipeline
        #: continues on the last complete data (reported as stale) instead of
        #: re-issuing the same partial full-range fetch.
        self._fallback_panel: pd.DataFrame | None = None
        self._fallback_range: tuple[str, str] | None = None
        #: Sidecar of the latest incremental attempt (for data_status).
        self._last_status_path: Path | None = None

        #: Corporate-action feed (``query_dividend_data``).  Fetched lazily;
        #: ``corporate_actions_available`` is True only after at least one
        #: successful query.  ``corporate_action_codes`` optionally restricts
        #: the (per-symbol, per-year) fetch; None means the same sample the
        #: price fetch would use.
        self.corporate_action_codes: list[str] | None = None
        self._corporate_actions: pd.DataFrame | None = None
        self._corporate_actions_key: tuple[str, str] | None = None
        self.corporate_actions_available = False

    # ------------------------------------------------------------------
    # Cache identity / versioning (D12/D13)
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_version() -> str:
        """Cache format version including the global data-schema generation."""
        from quant_platform.data.providers.base import schema_version

        return f"{_CACHE_VERSION}-s{schema_version()}"

    def cache_identity(self) -> dict:
        return {
            "provider": "baostock",
            "adjustment": _PRICE_ADJUSTMENT,
            "adjusted_price_basis": (
                _ADJUSTED_PRICE_BASIS
                if self.adjustment_mode == "hfq" else "raw"
            ),
            "adjustment_mode": self.adjustment_mode,
            "cache_version": self._cache_version(),
            "cache_enabled": bool(self._cache_enabled),
            "synthetic": False,
        }

    def data_status(self) -> dict:
        """Completion marker of the loaded daily panel (D32).

        A partial incremental attempt never touches the data file or its
        manifest; its ``incomplete`` flag lives in a sidecar, surfaced here
        when it refers to the same completion marker.
        """
        status = dict(self._data_status)
        if self._last_status_path is not None:
            sidecar = read_incremental_status(self._last_status_path)
            if (
                sidecar
                and sidecar.get("incomplete")
                and str(sidecar.get("last_complete_date"))
                == str(status.get("last_complete_date"))
            ):
                status["incomplete"] = True
                status["reason"] = sidecar.get("reason")
                status["last_incremental_attempt"] = sidecar
        return status

    @staticmethod
    def _normalise_range(
        start_date, end_date,
    ) -> tuple[str, str]:
        """Canonical YYYY-MM-DD pair for pin matching (accepts compact ISO)."""

        def _one(value) -> str:
            if value is None:
                return ""
            try:
                return pd.Timestamp(value).date().isoformat()
            except (TypeError, ValueError):
                return str(value)

        return _one(start_date), _one(end_date)

    def cache_universe(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        n_symbols: int | None = None,
    ) -> list[str] | None:
        """The exact symbol set ``get_prices``/``get_financials`` will fetch.

        ``get_prices`` always samples ``_MAJOR_LIST_SIZE`` stocks (it does not
        consult ``universe.n_stocks``) and ``get_financials`` samples 100;
        both are pinned here so the pipeline cache key describes the panel
        that is actually fetched. Returns ``None`` when the real universe is
        unavailable: a fallback-pool sample cannot be attested, so the caller
        must not cache the run.
        """
        records = self._get_universe_records()
        if records is None:
            logger.warning(
                "Stock universe unavailable; the baostock data stage cannot "
                "be cached for this run"
            )
            return None
        universe = self._records_for_range(records, start_date, end_date)
        price_codes = _stratified_sample(universe, _MAJOR_LIST_SIZE)
        financial_codes = _stratified_sample(universe, 100)
        self._pinned_range = self._normalise_range(start_date, end_date)
        self._pinned_price_codes = price_codes
        self._pinned_financial_codes = financial_codes
        return sorted(set(price_codes) | set(financial_codes))

    def _pinned_codes(
        self, start_date: str, end_date: str, *, financial: bool,
    ) -> list[str] | None:
        """Codes pinned for this exact range (see :meth:`cache_universe`)."""
        if self._pinned_range != self._normalise_range(start_date, end_date):
            return None
        if financial:
            return self._pinned_financial_codes
        return self._pinned_price_codes

    def _fallback_for(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame | None:
        """The last complete panel kept after an incomplete update, if any."""
        if self._fallback_panel is None or self._fallback_range is None:
            return None
        if self._fallback_range != self._normalise_range(start_date, end_date):
            return None
        return self._fallback_panel

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

    @staticmethod
    def _key_adjustment(key: str) -> str:
        """Basis of the data stored under this cache key.

        ``_adj1`` files hold the 后复权 series (hfq); ``_adj3`` files hold raw
        prices.  The manifest records the file's own basis so a raw and an
        adjusted panel can never be confused even if their keys collide.
        """
        if "_adj1" in key:
            return _ADJUSTED_PRICE_BASIS
        if "_adj3" in key:
            return "raw"
        return "n/a"

    def _manifest_valid(self, key: str) -> bool:
        expected_adjustment = self._key_adjustment(key)
        return cache_manifest_matches(
            self._cache_path(key),
            adjustment=(
                None if expected_adjustment == "n/a" else expected_adjustment
            ),
            cache_version=self._cache_version(),
            key=key,
        )

    def _write_manifest(self, key: str, extra: dict | None = None) -> None:
        payload = {"key": key}
        if extra:
            payload.update(extra)
        write_cache_manifest(
            self._cache_path(key),
            provider="baostock",
            adjustment=self._key_adjustment(key),
            cache_version=self._cache_version(),
            extra=payload,
        )

    def _discard_cache(self, key: str) -> None:
        base = self._cache_path(key)
        for path in (
            base,
            base.with_suffix(".csv"),
            base.with_name(f"{base.name}.manifest.json"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        if not self._cache_enabled:
            return None
        path = self._cache_path(key)
        csv_path = path.with_suffix(".csv")
        if not path.exists() and not csv_path.exists():
            return None
        if not self._manifest_valid(key):
            logger.info(
                "Ignoring cache %s: missing/stale version manifest "
                "(schema or adjustment changed); refetching", key,
            )
            self._discard_cache(key)
            return None
        # Usually exactly one format exists (writes remove the twin). If a
        # Windows sharing violation prevented removing the old parquet after
        # a CSV fallback write, serve the *newer* file rather than the stale
        # parquet. Never fall back to the older twin on a read error.
        source = path if path.exists() else csv_path
        if path.exists() and csv_path.exists():
            try:
                if csv_path.stat().st_mtime > path.stat().st_mtime:
                    source = csv_path
            except OSError:
                source = path
        try:
            if source == csv_path:
                # The asset column must stay a zero-filled string: pandas
                # would otherwise infer ``000001`` as integer ``1``, and every
                # incremental merge / hfq reindex compares labels exactly.
                return pd.read_csv(
                    source,
                    index_col=[0, 1],
                    parse_dates=[0],
                    dtype={1: str},
                )
            return pd.read_parquet(source)
        except Exception:
            return None

    def _save_cache(
        self,
        key: str,
        df: pd.DataFrame,
        extra: dict | None = None,
        *,
        raise_errors: bool = False,
    ):
        """Atomically publish a cache panel + manifest.

        Publication happens under the cross-process provider lock; the data
        file is replaced first and the manifest afterwards, so a crash can
        never leave new rows advertised under an old completion marker. With
        ``raise_errors`` (the incremental path) a publication failure
        propagates so the caller does not report a marker advance.
        """
        if not self._cache_enabled:
            return
        base = self._cache_path(key)
        csv_path = base.with_suffix(".csv")
        tmp = base.with_name(f"{base.name}.tmp.{os.getpid()}")
        try:
            try:
                df.to_parquet(tmp, index=True)
                with provider_cache_lock(base.parent) as locked:
                    if not locked:
                        raise TimeoutError(
                            f"timed out acquiring the provider cache lock "
                            f"for {base}"
                        ) from None
                    os.replace(tmp, base)
                    # The parquet file is authoritative; drop a stale CSV
                    # twin so a later parquet failure cannot resurrect older
                    # values.
                    csv_path.unlink(missing_ok=True)
                    self._write_manifest(key, extra=extra)
            except Exception:
                # Fallback to CSV if parquet fails. Write atomically too, and
                # remove the stale parquet (see MINOR 7 / reviewer finding).
                tmp.unlink(missing_ok=True)
                csv_tmp = csv_path.with_name(
                    f"{csv_path.name}.tmp.{os.getpid()}"
                )
                df.to_csv(csv_tmp, index=True)
                with provider_cache_lock(base.parent) as locked:
                    if not locked:
                        raise TimeoutError(
                            f"timed out acquiring the provider cache lock "
                            f"for {csv_path}"
                        ) from None
                    os.replace(csv_tmp, csv_path)
                    base.unlink(missing_ok=True)
                    self._write_manifest(key, extra=extra)
        except Exception as exc:  # a cache failure must not fail the research run
            logger.warning(
                "Could not cache baostock panel %s: %s", key, exc,
            )
            for leftover in (tmp,):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            if raise_errors:
                raise

    # ------------------------------------------------------------------
    # Daily prices
    # ------------------------------------------------------------------

    @staticmethod
    def _history_fields() -> str:
        # ``isST`` rides along with the price query (a separate ST query would
        # double the per-symbol network calls).  It is parsed into the side
        # column ``_is_st_observed`` and never enters the price contract.
        return ("date,code,open,high,low,close,preclose,volume,amount,"
                "turn,pctChg,peTTM,pbMRQ,isST")

    @staticmethod
    def _rows_to_frame(rows: list[list[str]], fields: list[str]) -> pd.DataFrame:
        """Build the daily frame, tolerating an interface without ``isST``.

        Older/stand-in baostock implementations may ignore the requested field
        list and return the thirteen legacy columns.  ``isST`` is the trailing
        optional field, so short rows are padded with empty strings (i.e.
        "no ST information") instead of poisoning the frame; wider rows are a
        genuine contract violation and raise.
        """
        width = len(fields)
        if any(len(row) > width for row in rows):
            raise RuntimeError(
                "query_history_k_data_plus returned rows wider than the "
                f"requested fields ({width})"
            )
        if all(len(row) == width for row in rows):
            return pd.DataFrame(rows, columns=fields)
        padded = [list(row) + [""] * (width - len(row)) for row in rows]
        return pd.DataFrame(padded, columns=fields)

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

    @staticmethod
    def _code_digest(codes: list[str]) -> str:
        return hashlib.md5(
            ",".join(sorted(str(code).zfill(6) for code in codes)).encode()
        ).hexdigest()[:8]

    def _daily_cache_key(
        self, codes: list[str], start_date: str, end_date: str, adjustflag: str,
    ) -> str:
        """Cache key for one (universe, range, adjustment) daily panel.

        The full sorted code list is hashed (not just its length), so any two
        same-sized universes cannot share an entry, and ``adjustflag`` is part
        of the key because hfq and raw panels must not collide.
        """
        codes_digest = self._code_digest(codes)
        return (
            f"daily_{self._cache_version()}_{start_date}_{end_date}_"
            f"{len(codes)}_{codes_digest}_adj{adjustflag}"
        )

    def _fetch_daily_data(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        adjustflag: str = "1",
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for multiple stocks from baostock.

        Every failed symbol is retried; if any still fails the fetch raises
        :class:`DataFetchIncompleteError` and the cache is not written, so a
        truncated panel can never be replayed as complete.

        ``use_cache=False`` is used for incremental *windows*: a window
        response is only a candidate append until the completeness checks in
        :meth:`update` pass, and caching it would make a retry for a fixed end
        date reuse the same partial window forever.
        """
        # The full sorted code list is hashed (not just its length) and the
        # adjustment flag is part of the key; see `_daily_cache_key`.
        cache_key = self._daily_cache_key(
            codes, start_date, end_date, adjustflag,
        )
        if use_cache:
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

        fields = self._history_fields().split(",")
        df = self._rows_to_frame(all_rows, fields)
        df["code"] = df["code"].apply(_from_bs_code)
        # ``isST`` is auxiliary data, not a price column: keep it in the cached
        # panel as ``_is_st_observed`` (so episodes survive a reload) and drop
        # it from the public frame in `get_prices`.
        if "isST" in df.columns:
            df = df.rename(columns={"isST": _ST_OBSERVED_COLUMN})
        else:  # defensive: _rows_to_frame always materialises the column
            df[_ST_OBSERVED_COLUMN] = ""
        df[_ST_OBSERVED_COLUMN] = (
            df[_ST_OBSERVED_COLUMN].fillna("").astype(str).str.strip()
        )

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
        df = canonicalize_panel_index(df)

        if df.empty:
            return _empty_prices()

        if use_cache:
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
        if not cache_manifest_matches(path, cache_version=self._cache_version()):
            logger.info(
                "Ignoring stock universe cache %s: missing/stale version "
                "manifest (schema changed); refetching", path.name,
            )
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
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(records), encoding="utf-8")
            os.replace(tmp, path)
            write_cache_manifest(
                path,
                provider="baostock",
                adjustment="n/a",
                cache_version=self._cache_version(),
                extra={"universe_cache_version": _UNIVERSE_CACHE_VERSION},
            )
        except OSError as exc:
            logger.warning("could not cache stock universe: %s", exc)
        except Exception as exc:  # manifest failure must not break the fetch
            logger.warning("could not cache stock universe manifest: %s", exc)

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

    # ------------------------------------------------------------------
    # ST history (derived from the daily isST field)
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_st_episodes(panel: pd.DataFrame | None) -> pd.DataFrame:
        """Derive conservative ST episodes from observed ``isST`` flags.

        Rules (also documented on :meth:`get_st_timeseries`):

        * an episode is a run of consecutive observed trading days whose flag
          is ``"1"``; an unknown/blank flag breaks the run (nothing is
          assumed about days without evidence);
        * ``announce_date`` is the first day ST was observed -- baostock has no
          earlier announcement date and none is inferred;
        * ``end_date`` is the last ST day of the run, or the last sample day
          when the sample ends while the stock is still ST;
        * ``trigger_date`` equals ``announce_date`` (no fabricated lag).
        """
        if (
            panel is None
            or panel.empty
            or _ST_OBSERVED_COLUMN not in panel.columns
        ):
            return _empty_st_timeseries()
        flags = panel[_ST_OBSERVED_COLUMN].astype("string").str.strip()
        known = flags.isin(["0", "1"]).to_numpy(dtype=bool)
        if not bool(known.any()):
            return _empty_st_timeseries()
        is_st = (flags == "1").fillna(False).to_numpy(dtype=bool)
        # 0 = observed not-ST, 1 = observed ST, 2 = unknown (breaks a run).
        state = np.full(len(flags), 2, dtype=np.int8)
        state[known] = is_st[known].astype(np.int8)
        frame = pd.DataFrame({
            "asset": panel.index.get_level_values("asset").astype(str),
            "date": pd.DatetimeIndex(panel.index.get_level_values("date")),
            "state": state,
        }).sort_values(["asset", "date"], kind="stable")
        same_asset = frame["asset"].eq(frame["asset"].shift())
        state_change = frame["state"].ne(frame["state"].shift())
        frame["run"] = (~same_asset | state_change).cumsum()
        st_runs = frame[frame["state"] == 1]
        if st_runs.empty:
            return _empty_st_timeseries()
        episodes = st_runs.groupby("run", sort=True).agg(
            asset=("asset", "first"),
            announce_date=("date", "min"),
            end_date=("date", "max"),
        ).reset_index(drop=True)
        episodes["trigger_date"] = episodes["announce_date"]
        episodes["is_st"] = True
        return episodes[_ST_EPISODE_COLUMNS]

    def _ingest_st_episodes(self, panel: pd.DataFrame | None) -> pd.DataFrame:
        """Replace the resolved episode table from a full daily panel."""
        episodes = self._derive_st_episodes(panel)
        self._st_timeseries = episodes
        self.st_available = not episodes.empty
        if episodes.empty:
            logger.info("Baostock ST history: no ST episodes in the sample")
        else:
            logger.info("Baostock ST history: %d episode(s)", len(episodes))
        return episodes

    def _resolve_st_timeseries(self) -> pd.DataFrame:
        """Resolve the episode table lazily (from a pinned range or prices).

        The pipeline loads point-in-time data *before* it asks for prices, so
        on the standard pinned-universe path this fetches the same raw daily
        panel `get_prices` will use -- the later call is a cache hit, so the
        network request count is unchanged. Without a pinned range and without
        a prior `get_prices` call there is no honest date range to ask for, so
        the table stays empty rather than guessing one.
        """
        if self._st_timeseries is not None:
            return self._st_timeseries
        if self._pinned_range is None or not self._pinned_price_codes:
            self._st_timeseries = _empty_st_timeseries()
            self.st_available = False
            return self._st_timeseries
        start, end = self._pinned_range
        try:
            panel = self._fetch_daily_data(
                self._pinned_price_codes, start, end, adjustflag="3",
            )
        except Exception as exc:  # ST history must never abort a run
            logger.warning(
                "Baostock ST history unavailable (%s); reporting no ST "
                "history for this run", exc,
            )
            self._st_timeseries = _empty_st_timeseries()
            self.st_available = False
            return self._st_timeseries
        return self._ingest_st_episodes(panel)

    def get_metadata(self) -> pd.DataFrame:
        """Get stock metadata, including delisted names with ``outDate``.

        ``is_st`` is intentionally absent: a whole-sample boolean would silently
        disable per-date ST screening. ST information is exposed as the
        episode table from :meth:`get_st_timeseries`, and
        ``attrs["st_available"]`` is True exactly when that table is non-empty.
        """
        cache_key = f"metadata_{self._cache_version()}"
        st_available = not self._resolve_st_timeseries().empty
        cached = self._load_cache(cache_key)
        if cached is not None:
            cached.attrs["st_available"] = st_available
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
        result.attrs["st_available"] = st_available
        self._save_cache(cache_key, result)
        return result

    def get_st_timeseries(self) -> pd.DataFrame:
        """ST episodes derived from the daily ``isST`` observations.

        Derivation (point-in-time, no fabrication):

        * for each asset, consecutive observed trading days with
          ``isST == "1"`` form one episode; a blank/unknown flag breaks the
          run because nothing is known about that day;
        * ``announce_date`` is the first day ST was *observed* -- baostock
          exposes no earlier announcement date, so nothing is inferred before
          the fetched sample (conservative: a stock already ST on the sample's
          first day is announced from that first day, not earlier);
        * ``end_date`` is the run's last ST day; if the sample ends while the
          stock is still ST, it is the sample's last day;
        * ``trigger_date`` equals ``announce_date``;
        * ``is_st`` is True.

        Returns the empty contract frame when the source did not return
        ``isST`` (missing column or only empty values): no flag is fabricated.
        """
        return self._resolve_st_timeseries().copy()

    # ------------------------------------------------------------------
    # Corporate actions (best effort)
    # ------------------------------------------------------------------

    def get_corporate_actions(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Cash dividends and bonus/transfer shares from ``query_dividend_data``.

        Baostock exposes dividends only per symbol and operating year, so the
        fetch is inherently N-symbols x Y-years of calls: it uses the pinned
        price sample when available (``cache_universe``), otherwise the same
        stratified universe the price fetch would use, and callers can pin a
        narrower list via ``corporate_action_codes``.  ``dividOperateDate`` is
        the ex-date, ``dividCashPsBeforeTax`` the pre-tax amount per share
        (the engine withholds tax per its config) and
        ``dividStocksPs + dividReserveToStockPs`` the bonus/transfer ratio
        (``split_ratio = 1 + bonus``).  A combined event becomes two rows.

        Best effort: per-symbol failures are logged and skipped; if every
        query fails (baostock missing, login refused, network down) the
        documented empty frame is returned and
        ``corporate_actions_available`` stays False.  Amounts are never
        inferred or fabricated.
        """
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        key = self._normalise_range(start_date, end_date)
        if self._corporate_actions is not None and self._corporate_actions_key == key:
            frame = self._corporate_actions
            mask = (frame["ex_date"] >= start) & (frame["ex_date"] <= end)
            return frame.loc[mask].reset_index(drop=True).copy()

        try:
            self._login()
        except Exception as exc:  # baostock not installed / login refused
            logger.warning(
                "Baostock unavailable for corporate actions: %s", exc,
            )
            self.corporate_actions_available = False
            self._corporate_actions = empty_corporate_actions()
            self._corporate_actions_key = key
            return empty_corporate_actions()

        bs = self._bs
        codes = self.corporate_action_codes
        if codes is None:
            codes = self._pinned_codes(start_date, end_date, financial=False)
        if codes is None:
            codes = self._get_major_stocks_list(
                start_date=start_date, end_date=end_date,
            )

        records: list[dict] = []
        successes = 0
        queried = 0
        for code in codes:
            bs_code = _to_bs_code(code)
            for year in range(start.year, end.year + 1):
                queried += 1
                try:
                    rs = self._call_with_retries(
                        f"query_dividend_data[{code}:{year}]",
                        bs.query_dividend_data,
                        code=bs_code,
                        year=str(year),
                        yearType="operate",
                    )
                except Exception as exc:  # noqa: BLE001 - best effort feed
                    logger.warning(
                        "Baostock dividend query failed for %s/%d: %s",
                        code, year, exc,
                    )
                    continue
                if getattr(rs, "error_code", "0") != "0":
                    logger.warning(
                        "Baostock dividend query error for %s/%d: %s",
                        code, year, getattr(rs, "error_msg", ""),
                    )
                    continue
                successes += 1
                fields = list(getattr(rs, "fields", []) or [])
                while rs.next():
                    rec = dict(zip(fields, rs.get_row_data(), strict=False))
                    records.extend(
                        self._dividend_records(rec, code),
                    )

        if queried and successes == 0:
            logger.warning(
                "Baostock dividend queries all failed (%d call(s)); "
                "corporate actions are unavailable", queried,
            )
            self.corporate_actions_available = False
            self._corporate_actions = empty_corporate_actions()
            self._corporate_actions_key = key
            return empty_corporate_actions()

        self.corporate_actions_available = True
        if not records:
            frame = empty_corporate_actions()
        else:
            frame = normalize_corporate_actions(
                pd.DataFrame(records, columns=list(CORPORATE_ACTION_COLUMNS))
            )
            mask = (frame["ex_date"] >= start) & (frame["ex_date"] <= end)
            frame = frame.loc[mask].reset_index(drop=True)
        self._corporate_actions = frame
        self._corporate_actions_key = key
        return frame.copy()

    @staticmethod
    def _dividend_records(rec: dict, code: str) -> list[dict]:
        """Contract rows for one Baostock dividend record (cash and/or split)."""
        ex_date = _as_date(rec.get("dividOperateDate"))
        if pd.isna(ex_date):
            return []
        asset = normalize_symbol(_to_bs_code(code))
        cash = _as_float(rec.get("dividCashPsBeforeTax"))
        if not math.isfinite(cash) or cash <= 0:
            cash = _as_float(rec.get("dividCashPsAfterTax"))
        cash = cash if math.isfinite(cash) and cash > 0 else 0.0
        bonus = 0.0
        for key in ("dividStocksPs", "dividReserveToStockPs"):
            value = _as_float(rec.get(key))
            if math.isfinite(value) and value > 0:
                bonus += value
        rows: list[dict] = []
        if cash > 0:
            rows.append({
                "ex_date": ex_date,
                "asset": asset,
                "kind": "cash",
                "cash_per_share": round(cash, 6),
                "split_ratio": 1.0,
            })
        if bonus > 0:
            rows.append({
                "ex_date": ex_date,
                "asset": asset,
                "kind": "split",
                "cash_per_share": 0.0,
                "split_ratio": round(1.0 + bonus, 6),
            })
        return rows

    def _get_industry_map(self) -> dict[str, str]:
        """Get industry classification for all stocks."""
        cache_key = f"industry_map_{self._cache_version()}"
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
        """Get daily price data for a range-overlapping stock sample.

        The OHLCV path is fetched raw (``adjustflag="3"``).  In the default
        ``hfq`` mode the 后复权 series is fetched too and the divisor
        ``adj_factor = raw_close / hfq_close`` is attached, so
        ``close / adj_factor`` is the hfq series research used before the
        explicit basis contract.  In ``raw`` mode no factor is available
        (``adj_factor=1``) and the research series is honestly raw.
        """
        codes = self._pinned_codes(
            start_date, end_date, financial=False,
        )
        if codes is None:
            codes = self._get_major_stocks_list(
                start_date=start_date, end_date=end_date,
            )
        fallback = self._fallback_for(start_date, end_date)
        if fallback is not None:
            raw = fallback.loc[
                pd.Timestamp(start_date):pd.Timestamp(end_date)
            ].copy()
        else:
            raw = self._fetch_daily_data(
                codes, start_date, end_date, adjustflag="3",
            )

        if not raw.empty:
            # Side channel: the same daily response also carries ``isST``;
            # derive the episode table here (cache hits included) before the
            # auxiliary column is dropped from the public price frame.
            self._ingest_st_episodes(raw)

        if raw.empty:
            empty = self._project_price_fields(raw, fields)
        elif self.adjustment_mode == "raw":
            df = raw.copy()
            df["adj_factor"] = 1.0
            empty = None
        elif "adj_factor" in raw.columns and raw["adj_factor"].notna().any():
            # An incrementally published panel already carries the divisor;
            # refetching the whole hfq history would defeat the append.
            df = raw.copy()
            df["adj_factor"] = (
                pd.to_numeric(df["adj_factor"], errors="coerce")
                .groupby(level="asset").ffill()
            )
            missing_factor = df["close"].notna() & df["adj_factor"].isna()
            if bool(missing_factor.any()):
                raise DataFetchIncompleteError(
                    "Baostock cached panel is missing adjustment divisors "
                    f"for {int(missing_factor.sum())} priced row(s); "
                    "refusing to guess adj_factor=1.0"
                )
            empty = None
        else:
            hfq_end = end_date
            if fallback is not None:
                # Never issue a network fetch past the complete cached range.
                last = panel_last_date(raw)
                hfq_end = str(last.date()) if last is not None else end_date
            hfq_key = self._daily_cache_key(
                codes, start_date, hfq_end, "1",
            )
            hfq = self._fetch_daily_data(
                codes, start_date, hfq_end, adjustflag="1",
            )
            try:
                df = self._merge_adjustment_divisor(raw, hfq)
            except DataFetchIncompleteError:
                # The hfq panel was persisted before validation; a panel that
                # cannot produce a divisor for every raw close is unusable and
                # must not be replayed on retries (it would wedge the append
                # path too, which derives cached-range divisors from it).
                self._discard_cache(hfq_key)
                raise
            empty = None

        if empty is not None:
            empty.attrs.update({
                "price_basis": "raw",
                "adjusted_price_basis": "raw",
                "adjustment_factor_available": False,
            })
            return empty

        # Completion marker (D32): a full fetch that published a new raw
        # cache records ``last_complete_date`` so the *next* run can append
        # instead of refetching. A legacy cache without a marker is left
        # alone here; `update()` rejects it and performs a full refetch.
        if fallback is None and not raw.empty:
            key3 = self._daily_cache_key(codes, start_date, end_date, "3")
            manifest = read_cache_manifest(self._cache_path(key3)) or {}
            marker = manifest.get("last_complete_date")
            if not marker:
                panel_end = panel_last_date(raw)
                marker, incomplete = complete_panel_marker(
                    raw,
                    delisted=self._delisted_from_records(
                        self._get_universe_records(), panel_end,
                    ),
                )
                extra = incremental_marker_extra(
                    last_complete_date=marker,
                    requested_end=end_date,
                    incomplete=incomplete,
                    source="full_fetch",
                )
                self._write_marker(key3, extra)
                manifest = {**manifest, **extra}
                marker = extra["last_complete_date"]
            self._data_status = {
                "last_complete_date": marker,
                "requested_end": manifest.get("end_date") or end_date,
                "updated_at": manifest.get("updated_at"),
                "data_age_days": manifest.get("data_age_days"),
                "incomplete": bool(manifest.get("incomplete", False)),
                "source": "cache_manifest",
            }
            # A failed append attempt leaves its flag in a sidecar; surface
            # it when it refers to the same marker (restarted processes).
            sidecar = read_incremental_status(self._cache_path(key3))
            if (
                sidecar
                and sidecar.get("incomplete")
                and str(sidecar.get("last_complete_date")) == str(marker)
            ):
                self._data_status["incomplete"] = True
                self._data_status["reason"] = sidecar.get("reason")
                self._last_status_path = self._cache_path(key3)

        if _ST_OBSERVED_COLUMN in df.columns:
            df = df.drop(columns=[_ST_OBSERVED_COLUMN])
        df = self._project_price_fields(df, fields)
        df.attrs.update({
            "price_basis": "raw",
            "adjusted_price_basis": (
                _ADJUSTED_PRICE_BASIS
                if self.adjustment_mode == "hfq" else "raw"
            ),
            "adjustment_factor_available": self.adjustment_factor_available,
        })
        return df

    @staticmethod
    def _merge_adjustment_divisor(
        raw: pd.DataFrame, hfq: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach ``adj_factor = raw_close / hfq_close`` to the raw panel.

        Baostock's hfq series is ``raw * cumulative_factor``, so this ratio is
        exactly the divisor the pipeline needs (``close / adj_factor`` is
        hfq).  A ratio is only meaningful where both closes are present; short
        gaps are carried across per asset (corporate-action factors are
        piecewise constant between actions).  If a row still has a raw close
        but no derivable factor the fetch fails closed: silently assuming 1.0
        would fabricate a corporate-action return in the research series.
        """
        if hfq is None or hfq.empty:
            raise DataFetchIncompleteError(
                "Baostock hfq fetch returned no data; cannot derive the "
                "adjustment divisor. Use adjustment='raw' to run unadjusted."
            )
        raw_close = pd.to_numeric(raw["close"], errors="coerce")
        hfq_close = pd.to_numeric(
            hfq["close"].reindex(raw.index), errors="coerce",
        )
        factor = raw_close / hfq_close
        valid = np.isfinite(factor) & (factor > 0)
        factor = factor.where(valid)
        factor = factor.groupby(level="asset").ffill()
        factor = factor.groupby(level="asset").bfill()
        bad = raw_close.notna() & factor.isna()
        if bool(bad.any()):
            sample = [
                f"{asset}@{str(date)[:10]}"
                for date, asset in bad[bad].index[:5]
            ]
            raise DataFetchIncompleteError(
                "Baostock adjustment divisor unavailable for rows that carry "
                f"a raw close ({int(bad.sum())} row(s), e.g. {sample}); "
                "refusing to guess adj_factor=1.0"
            )
        merged = raw.copy()
        merged["adj_factor"] = factor.astype(float)
        return merged

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
        publication date (``pubDate``).  A report whose ``pubDate`` is
        missing or unparseable is dropped with a warning -- the period end is
        not a visibility date, and using it would leak figures before they
        were announced.  If no report row has a usable publication date the
        fetch raises :class:`DataFetchIncompleteError` rather than returning
        a panel built from fabricated dates.  Columns: market_cap,
        total_assets, net_assets, revenue, net_profit, roe, pb_ratio,
        pe_ratio, asset_growth, publish_date.  ``asset_growth`` is the
        year-over-year change in total assets against the same fiscal quarter
        one year earlier, computed on the fetched report sequence before
        undated rows are dropped.  No backward fill.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        codes = self._pinned_codes(
            start_date, end_date, financial=True,
        )
        if codes is None:
            codes = self._get_major_stocks_list(
                n=100, start_date=start_date, end_date=end_date,
            )
        if not codes:
            return _empty_financials()

        codes_digest = hashlib.md5(
            ",".join(sorted(codes)).encode()
        ).hexdigest()[:8]
        cache_key = (
            f"fin_{self._cache_version()}_{start_date}_{end_date}_"
            f"{len(codes)}_{codes_digest}_adj3"
        )
        cached = self._load_cache(cache_key)
        if cached is not None and len(cached) > 0:
            return cached

        if self._fallback_for(start_date, end_date) is not None:
            # An incomplete incremental update keeps the last complete
            # financial panel rather than re-issuing a fetch that may fail
            # the same way the price update did.
            fallback_fin = self._load_fallback_financials(
                codes, start_date, end_date,
            )
            if fallback_fin is not None:
                return fallback_fin

        # Raw (unadjusted) prices supply the price leg of market cap and the
        # trading calendar used to map publication dates. After an incomplete
        # incremental update the last complete panel is used instead of
        # re-issuing the same partial full-range fetch.
        fallback = self._fallback_for(start_date, end_date)
        if fallback is not None:
            raw_prices = fallback.loc[
                pd.Timestamp(start_date):pd.Timestamp(end_date)
            ].copy()
        else:
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
        report_rows = 0
        dropped_no_pub = 0
        dropped_after_end = 0
        # total_assets per (asset, fiscal year, quarter) from every fetched
        # report, including rows later dropped from the panel. asset_growth
        # is a same-quarter-last-year ratio computed from this basis, so
        # dropping an undated report cannot shift the YoY window.
        growth_assets: dict[tuple[str, int, int], float] = {}

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
                report_rows += 1

                # Balances are fetched before the pubDate check on purpose: a
                # report dropped for a missing pubDate still anchors next
                # year's same-quarter asset growth.
                balance = self._balance_row(bs, bs_code, year, quarter)
                stat_date = _as_date(profit.get("statDate"))
                pub_date = _as_date(profit.get("pubDate"))
                roe_pct = _as_float(profit.get("roeAvg"))
                net_profit = _as_float(profit.get("netProfit"))
                revenue = _as_float(profit.get("MBRevenue"))
                total_share = _as_float(profit.get("totalShare"))

                net_assets = float("nan")
                if math.isfinite(roe_pct) and roe_pct != 0.0:
                    net_assets = net_profit / (roe_pct / 100.0)
                total_assets = float("nan")
                if balance is not None:
                    asset_to_equity = _as_float(balance.get("assetToEquity"))
                    if math.isfinite(asset_to_equity) and math.isfinite(net_assets):
                        total_assets = net_assets * asset_to_equity
                growth_assets[(code, year, quarter)] = total_assets

                if pd.isna(pub_date):
                    # No publication date -> no honest visibility date. The
                    # fiscal period end is NOT a substitute: reports are
                    # usually announced weeks later, so keying the row to the
                    # first trading day after the period end would expose
                    # figures before they existed (look-ahead).
                    dropped_no_pub += 1
                    logger.debug(
                        "Report %s %sQ%s has no usable pubDate; dropping it "
                        "instead of fabricating a visibility date "
                        "(statDate=%s)",
                        bs_code, year, quarter,
                        stat_date.date() if pd.notna(stat_date) else "n/a",
                    )
                    continue

                pos = trading_dates.searchsorted(pub_date, side="left")
                if pos >= len(trading_dates):
                    dropped_after_end += 1
                    continue  # published after the sample ends
                visible_date = trading_dates[pos]

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
                    "_year": year,
                    "_quarter": quarter,
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

        if dropped_no_pub:
            logger.warning(
                "baostock: dropped %d/%d financial rows without a usable "
                "publish date",
                dropped_no_pub, report_rows,
            )
        if failed:
            raise DataFetchIncompleteError(
                "Baostock financial fetch incomplete: "
                f"{len(failed)} symbol(s) failed after retries "
                f"(e.g. {sorted(failed)[:10]})"
            )
        if not records and report_rows and dropped_no_pub == report_rows:
            raise DataFetchIncompleteError(
                f"Baostock financials: none of the {report_rows} report "
                "row(s) has a usable pubDate; refusing to fabricate "
                "visibility dates from fiscal period ends, so no "
                "point-in-time panel can be built"
            )
        if dropped_after_end and not records:
            logger.warning(
                "baostock: %d financial row(s) were published after the "
                "sample end; returning an empty panel",
                dropped_after_end,
            )
        if not records:
            return _empty_financials()

        # YoY asset growth against the same fiscal quarter one year earlier,
        # resolved on the full fetched report sequence. Doing this before the
        # old positional pct_change(4) means a dropped undated report cannot
        # make a later quarter compare against the wrong quarter.
        for record in records:
            prior_assets = growth_assets.get(
                (
                    record["asset"],
                    record.pop("_year") - 1,
                    record.pop("_quarter"),
                ),
                float("nan"),
            )
            total_assets = record["total_assets"]
            growth = float("nan")
            if (
                math.isfinite(total_assets)
                and math.isfinite(prior_assets)
                and prior_assets != 0.0
            ):
                growth = total_assets / prior_assets - 1.0
            record["asset_growth"] = growth

        df = pd.DataFrame(records)
        df = df.sort_values(["asset", "stat_date"])
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
        cache_key = (
            f"bench_{self._cache_version()}_{start_date}_{end_date}_adj1"
        )
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            series = cached.iloc[:, 0]
            series.name = "benchmark"
            return series

        if self._fallback_for(start_date, end_date) is not None:
            fallback = self._load_fallback_benchmark(start_date, end_date)
            if fallback is not None:
                return fallback

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

    # ------------------------------------------------------------------
    # Incremental updates (D32)
    # ------------------------------------------------------------------

    @staticmethod
    def _payload(**overrides) -> dict:
        status = {
            "supported": True,
            "performed": False,
            "full_refetch": False,
            "new_sessions": 0,
            "requested_end": None,
            "last_complete_date": None,
            "previous_last_complete_date": None,
            "updated_at": None,
            "data_age_days": None,
            "incomplete": False,
            "missing_symbols": [],
            "reason": "",
        }
        status.update(overrides)
        return status

    def _remember_status(self, status: dict, *, source: str) -> dict:
        self._data_status = {
            "last_complete_date": status.get("last_complete_date"),
            "requested_end": status.get("requested_end"),
            "updated_at": status.get("updated_at"),
            "data_age_days": status.get("data_age_days"),
            "incomplete": bool(status.get("incomplete")),
            "source": source,
        }
        return status

    def _candidate_daily_caches(
        self, codes: list[str], start_date: str,
    ) -> list[tuple]:
        """Valid cached raw (``adj3``) panels for this universe + start."""
        prefix = f"daily_{self._cache_version()}_{start_date}_"
        suffix = f"_{len(codes)}_{self._code_digest(codes)}_adj3"
        keys: dict[str, pd.Timestamp] = {}
        for pattern in (f"{prefix}*{suffix}.parquet", f"{prefix}*{suffix}.csv"):
            for path in CACHE_DIR.glob(pattern):
                name = path.name
                if ".tmp." in name:
                    continue
                key = name.rsplit(".", 1)[0]
                if key in keys:
                    continue
                end_str = key[len(prefix):-len(suffix)]
                try:
                    end_ts = pd.Timestamp(end_str).normalize()
                except (TypeError, ValueError):
                    continue
                if not self._manifest_valid(key):
                    continue
                keys[key] = end_ts
        return sorted(
            ((end_ts, key) for key, end_ts in keys.items()),
            key=lambda item: item[0],
        )

    @staticmethod
    def _manifest_marker(key_path: Path) -> dict:
        return read_cache_manifest(key_path) or {}

    @staticmethod
    def _delisted_from_records(
        records: list[dict] | None, before,
    ) -> set[str]:
        """Codes whose baostock ``outDate`` is <= ``before``."""
        cutoff = pd.Timestamp(before).normalize()
        exempt: set[str] = set()
        for record in records or []:
            out = _as_date(record.get("outDate"))
            if pd.notna(out) and pd.Timestamp(out).normalize() <= cutoff:
                exempt.add(str(record["code"]).zfill(6))
        return exempt

    def _write_marker(self, key: str, extra: dict) -> None:
        """Attach the completion marker after the data file was published."""
        base = self._cache_path(key)
        if not base.exists() and not base.with_suffix(".csv").exists():
            # The panel was never published (cache disabled or a save
            # failure); a marker without data would make the next append read
            # an unknown boundary, so leave it unwritten.
            logger.warning(
                "Not writing a completion marker for %s: no cached data file",
                key,
            )
            return
        try:
            with provider_cache_lock(base.parent) as locked:
                if not locked:
                    raise TimeoutError(
                        f"timed out acquiring the provider cache lock for {key}"
                    )
                self._write_manifest(key, extra=extra)
        except Exception as exc:  # marker is best effort; missing => refetch
            logger.warning(
                "Could not write the incremental completion marker for %s: %s",
                key, exc,
            )

    def _full_refetch_status(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        reason: str,
        previous_marker=None,
    ) -> dict:
        """Fetch the requested range from scratch and mark it complete."""
        logger.warning(
            "Incremental update fell back to a full refetch: %s", reason,
        )
        self._fallback_panel = None
        self._fallback_range = None
        # A full refetch must not replay the cache that triggered it (a
        # corrupt/partial panel or a panel/marker race).
        raw = self._fetch_daily_data(
            codes, start_date, end_date, adjustflag="3", use_cache=False,
        )
        if raw.empty:
            status = self._payload(
                requested_end=end_date,
                reason=f"{reason} (provider returned no rows)",
            )
            return self._remember_status(status, source="full_refetch")
        if self.adjustment_mode == "hfq":
            hfq = self._fetch_daily_data(
                codes, start_date, end_date, adjustflag="1",
                use_cache=False,
            )
            self._merge_adjustment_divisor(raw, hfq)
        panel_end = panel_last_date(raw)
        marker, incomplete = complete_panel_marker(
            raw,
            delisted=self._delisted_from_records(
                self._get_universe_records(), panel_end,
            ),
        )
        extra = incremental_marker_extra(
            last_complete_date=marker,
            requested_end=end_date,
            previous_last_complete_date=previous_marker,
            new_sessions=0,
            incomplete=incomplete,
            source="full_refetch",
        )
        key = self._daily_cache_key(codes, start_date, end_date, "3")
        # Publish the freshly fetched panel (the cache that triggered this
        # fallback is deliberately not replayed) with its completion marker.
        self._save_cache(key, raw, extra=extra, raise_errors=True)
        status = self._payload(
            requested_end=end_date,
            last_complete_date=extra["last_complete_date"],
            previous_last_complete_date=extra["previous_last_complete_date"],
            updated_at=extra["updated_at"],
            data_age_days=extra["data_age_days"],
            full_refetch=True,
            incomplete=incomplete,
            reason=reason,
        )
        self._last_status_path = self._cache_path(key)
        write_incremental_status(self._last_status_path, status)
        return self._remember_status(status, source="full_refetch")

    def _incomplete_status(
        self,
        old_panel: pd.DataFrame,
        codes: list[str],
        start_date: str,
        end_date: str,
        reason: str,
        *,
        marker,
        missing: set[str] | None = None,
    ) -> dict:
        """Partial response: keep the old cache/panel, marker unchanged."""
        logger.warning(
            "Incremental update incomplete (%s); keeping the cached panel "
            "through %s and not advancing the completion marker",
            reason, marker,
        )
        self._fallback_panel = old_panel
        self._fallback_range = self._normalise_range(start_date, end_date)
        status = self._payload(
            requested_end=end_date,
            last_complete_date=(
                pd.Timestamp(marker).date().isoformat()
                if marker is not None else None
            ),
            incomplete=True,
            missing_symbols=sorted(str(code) for code in (missing or set()))[:20],
            reason=reason,
        )
        key = self._daily_cache_key(codes, start_date, end_date, "3")
        self._last_status_path = self._cache_path(key)
        write_incremental_status(self._last_status_path, status)
        return self._remember_status(status, source="cache_manifest")

    def _load_fallback_financials(
        self, codes: list[str], start_date: str, end_date: str,
    ) -> pd.DataFrame | None:
        """Financial panel from the last complete cache after a partial update."""
        prefix = f"fin_{self._cache_version()}_{start_date}_"
        suffix = f"_{len(codes)}_{self._code_digest(codes)}_adj3"
        requested_end = pd.Timestamp(end_date).normalize()
        best: tuple[pd.Timestamp, str] | None = None
        for pattern in (f"{prefix}*{suffix}.parquet", f"{prefix}*{suffix}.csv"):
            for path in CACHE_DIR.glob(pattern):
                name = path.name
                if ".tmp." in name:
                    continue
                key = name.rsplit(".", 1)[0]
                end_str = key[len(prefix):-len(suffix)]
                try:
                    end_ts = pd.Timestamp(end_str).normalize()
                except (TypeError, ValueError):
                    continue
                if end_ts > requested_end or not self._manifest_valid(key):
                    continue
                if best is None or end_ts > best[0]:
                    best = (end_ts, key)
        if best is None:
            return None
        cached = self._load_cache(best[1])
        if cached is None or cached.empty:
            return None
        return cached.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def _load_fallback_benchmark(
        self, start_date: str, end_date: str,
    ) -> pd.Series | None:
        """Benchmark from the last complete cache when an update was partial."""
        prefix = f"bench_{self._cache_version()}_{start_date}_"
        best: tuple[pd.Timestamp, str] | None = None
        for pattern in (f"{prefix}*_adj1.parquet", f"{prefix}*_adj1.csv"):
            for path in CACHE_DIR.glob(pattern):
                name = path.name
                if ".tmp." in name:
                    continue
                key = name.rsplit(".", 1)[0]
                end_str = key[len(prefix):-len("_adj1")]
                try:
                    end_ts = pd.Timestamp(end_str).normalize()
                except (TypeError, ValueError):
                    continue
                if end_ts > pd.Timestamp(end_date).normalize():
                    continue
                if not self._manifest_valid(key):
                    continue
                if best is None or end_ts > best[0]:
                    best = (end_ts, key)
        if best is None:
            return None
        cached = self._load_cache(best[1])
        if cached is None or cached.empty:
            return None
        series = cached.iloc[:, 0]
        series.name = "benchmark"
        return series.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def update(self, start_date: str = None, end_date: str = None) -> dict:
        """Append new exchange sessions to the cached raw panel (D32).

        Baostock's raw (``adjustflag=3``) panel is cached per
        ``(universe, range)``. The update locates the newest compatible cache
        for the same universe and an earlier end date, reads its completion
        marker, fetches ``marker + 1 session`` (plus one overlap session)
        through the requested end for both raw and hfq series, derives the
        divisor for the new rows, merges with ``keep="last"`` and publishes
        atomically. If the pinned universe or the adjustment basis changed --
        the cache key no longer matches -- a full fetch is performed instead.

        A partial response keeps the old cache and returns
        ``incomplete=True``; the completion marker only advances after a
        complete publish.
        """
        if start_date is None and self._pinned_range is not None:
            start_date = self._pinned_range[0]
        if end_date is None and self._pinned_range is not None:
            end_date = self._pinned_range[1]
        if start_date is None or end_date is None:
            raise ValueError("update() requires an explicit start and end date")
        start_s = str(pd.Timestamp(start_date).date())
        end_s = str(pd.Timestamp(end_date).date())
        requested_end = pd.Timestamp(end_s).normalize()

        codes = self._pinned_codes(start_s, end_s, financial=False)
        if codes is None:
            codes = self._get_major_stocks_list(
                start_date=start_s, end_date=end_s,
            )
        if not codes:
            return self._remember_status(
                self._payload(
                    requested_end=end_s,
                    reason="empty stock universe; nothing to append",
                ),
                source="full_refetch",
            )
        if not self._cache_enabled:
            return self._remember_status(
                self._payload(
                    requested_end=end_s,
                    full_refetch=True,
                    reason="provider cache disabled; performing a full fetch",
                ),
                source="full_refetch",
            )

        candidates = self._candidate_daily_caches(codes, start_s)
        covering = [item for item in candidates if item[0] <= requested_end]
        if not covering:
            return self._full_refetch_status(
                codes, start_s, end_s,
                "no compatible cached panel for this universe/start range",
            )
        end_ts, key = covering[-1]
        cached_end_s = str(end_ts.date())
        old_panel = self._load_cache(key)
        if old_panel is None or old_panel.empty:
            return self._full_refetch_status(
                codes, start_s, end_s,
                f"cached panel {key} could not be loaded",
            )
        manifest = self._manifest_marker(self._cache_path(key))
        marker_raw = manifest.get("last_complete_date")
        if not marker_raw or manifest.get("incomplete"):
            return self._full_refetch_status(
                codes, start_s, end_s,
                "cached panel has no trustworthy completion marker",
                previous_marker=marker_raw,
            )
        marker = pd.Timestamp(marker_raw).normalize()
        panel_end = panel_last_date(old_panel)
        if panel_end is None or pd.Timestamp(panel_end).normalize() < marker:
            # The panel and its manifest were read without a lock; a
            # concurrent publish can leave the data file older than the
            # marker. Appending from that base could skip sessions between
            # the panel end and the marker, so refetch in full instead.
            return self._full_refetch_status(
                codes, start_s, end_s,
                "cached panel ends before its completion marker "
                "(concurrent writer or corrupt cache)",
                previous_marker=marker_raw,
            )
        if requested_end <= marker:
            status = self._payload(
                requested_end=end_s,
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                data_age_days=manifest.get("data_age_days"),
                reason="completion marker already covers the requested end",
            )
            return self._remember_status(status, source="cache_manifest")

        # A full-fetch raw cache carries no divisor (it is derived from the
        # hfq series on demand). The append needs the cached rows' factors to
        # compute the overlap correction and to publish a self-contained
        # panel, so derive them here (the hfq fetch normally hits the cache
        # written by the run that populated the panel). In raw mode the factor
        # is the placeholder 1.0 -- no hfq call is needed at all.
        if "adj_factor" not in old_panel.columns:
            if self.adjustment_mode == "raw":
                old_panel = old_panel.copy()
                old_panel["adj_factor"] = 1.0
            else:
                try:
                    hfq_key = self._daily_cache_key(
                        codes, start_s, cached_end_s, "1",
                    )
                    hfq_old = self._fetch_daily_data(
                        codes, start_s, cached_end_s, adjustflag="1",
                    )
                    try:
                        old_panel = self._merge_adjustment_divisor(
                            old_panel, hfq_old,
                        )
                    except DataFetchIncompleteError:
                        # Never replay an hfq panel that failed validation.
                        self._discard_cache(hfq_key)
                        raise
                except DataFetchIncompleteError as exc:
                    return self._incomplete_status(
                        old_panel, codes, start_s, end_s,
                        f"cached-range adjustment divisor unavailable: {exc}",
                        marker=marker,
                    )

        window_start = incremental_fetch_start(
            marker, requested_end, overlap=INCREMENTAL_OVERLAP_SESSIONS,
        )
        if window_start is None:
            status = self._payload(
                requested_end=end_s,
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                reason="no exchange session between the marker and the end",
            )
            return self._remember_status(status, source="cache_manifest")
        window_start_s = str(pd.Timestamp(window_start).date())

        try:
            # Windows are fetched *without* the cache: a window response is
            # only a candidate append until the completeness checks below
            # pass. Caching it would make a retry for the same end date reuse
            # the same partial window forever.
            raw_window = self._fetch_daily_data(
                codes, window_start_s, end_s, adjustflag="3",
                use_cache=False,
            )
        except DataFetchIncompleteError as exc:
            return self._incomplete_status(
                old_panel, codes, start_s, end_s, str(exc), marker=marker,
            )
        if raw_window.empty:
            logger.warning(
                "Incremental fetch for %s..%s returned no rows; keeping the "
                "cached panel (reported as incomplete/stale, not fresh)",
                window_start_s, end_s,
            )
            # Serve the last complete panel for the requested range instead
            # of re-issuing a full-range fetch that would return the same
            # rows (or fail on a genuinely partial provider).
            self._fallback_panel = old_panel
            self._fallback_range = self._normalise_range(start_s, end_s)
            status = self._payload(
                requested_end=end_s,
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                incomplete=True,
                reason="provider returned no rows for the new sessions",
            )
            self._last_status_path = self._cache_path(
                self._daily_cache_key(codes, start_s, end_s, "3")
            )
            write_incremental_status(self._last_status_path, status)
            return self._remember_status(status, source="cache_manifest")

        try:
            if self.adjustment_mode == "raw":
                new_panel = raw_window.copy()
                new_panel["adj_factor"] = 1.0
            else:
                hfq_window = self._fetch_daily_data(
                    codes, window_start_s, end_s, adjustflag="1",
                    use_cache=False,
                )
                new_panel = self._merge_adjustment_divisor(
                    raw_window, hfq_window,
                )
        except DataFetchIncompleteError as exc:
            return self._incomplete_status(
                old_panel, codes, start_s, end_s,
                f"hfq/adjustment fetch failed: {exc}", marker=marker,
            )

        boundary_symbols = symbols_on_date(old_panel, marker)
        delisted = self._delisted_from_records(
            self._get_universe_records(), window_start,
        )
        new_dates = [
            day for day in pd.DatetimeIndex(
                new_panel.index.get_level_values("date").unique()
            ).normalize() if day > marker
        ]
        if not new_dates:
            logger.warning(
                "Incremental fetch for %s..%s returned only overlap rows; "
                "keeping the cached panel (reported as incomplete/stale)",
                window_start_s, end_s,
            )
            return self._incomplete_status(
                old_panel, codes, start_s, end_s,
                "provider returned no rows after the completion marker",
                marker=marker,
            )
        window_symbols = {
            str(code) for code in new_panel.index[
                new_panel.index.get_level_values("date").isin(new_dates)
            ].get_level_values("asset")
        }
        missing_window = boundary_symbols - window_symbols - delisted
        if missing_window:
            return self._incomplete_status(
                old_panel, codes, start_s, end_s,
                "provider response is missing symbols that traded on the "
                "completion marker",
                marker=marker, missing=missing_window,
            )

        # A corporate action changes the hfq basis. If the overlap ratio
        # drifted, rescale cached divisors so adjusted history stays on the
        # same basis as the appended rows (defensive: hfq normally leaves
        # pre-action history untouched, but a vendor revision must not split
        # the series).
        merged_old = old_panel
        overlap = old_panel.index.intersection(new_panel.index)
        if len(overlap):
            old_factor = pd.to_numeric(
                old_panel.loc[overlap, "adj_factor"], errors="coerce",
            )
            new_factor = pd.to_numeric(
                new_panel.loc[overlap, "adj_factor"], errors="coerce",
            )
            ratio = (new_factor / old_factor).replace(
                [np.inf, -np.inf], np.nan,
            ).dropna()
            ratio = ratio[ratio > 0]
            if len(ratio):
                median = float(ratio.median())
                if abs(median - 1.0) > 1e-9:
                    merged_old = old_panel.copy()
                    values = merged_old["adj_factor"].copy()
                    before = (
                        merged_old.index.get_level_values("date")
                        < pd.Timestamp(window_start)
                    )
                    values.loc[before] = values.loc[before] * median
                    merged_old["adj_factor"] = values
                    logger.info(
                        "Incremental update rescaled cached adjustment "
                        "factors by %.8f to match the provider's new basis",
                        median,
                    )

        merged = pd.concat([merged_old, new_panel]).sort_index()
        merged = deduplicate_panel(merged, "prices")

        merged_dates = pd.DatetimeIndex(
            merged.index.get_level_values("date").unique()
        ).sort_values()
        previous_symbols = (
            symbols_on_date(merged, merged_dates[-2])
            if len(merged_dates) >= 2 else set()
        )
        delisted_on_last = self._delisted_from_records(
            self._get_universe_records(), panel_last_date(merged),
        )
        missing_last = last_day_missing_symbols(
            merged, delisted=delisted_on_last,
        )
        # Zero tolerance on the last session: without a suspension feed a
        # truncated last day is indistinguishable from a suspended name,
        # and the marker must never advance past a partial cross-section.
        if missing_last:
            return self._incomplete_status(
                old_panel, codes, start_s, end_s,
                "last session of the provider response is missing symbols "
                f"present on the previous session ({len(missing_last)} of "
                f"{len(previous_symbols)})",
                marker=marker, missing=missing_last,
            )

        merged_date_set = set(merged_dates.normalize())
        expected_sessions = exchange_sessions(
            marker + pd.Timedelta(days=1), panel_last_date(merged),
        )
        missing_dates = [
            day for day in expected_sessions if day not in merged_date_set
        ]
        if missing_dates:
            return self._incomplete_status(
                old_panel, codes, start_s, end_s,
                "provider response is missing exchange sessions: "
                + ", ".join(str(day.date()) for day in missing_dates[:5]),
                marker=marker,
            )

        new_marker = panel_last_date(merged)
        new_sessions = sessions_after(marker, new_marker)
        extra = incremental_marker_extra(
            last_complete_date=new_marker,
            requested_end=end_s,
            previous_last_complete_date=marker,
            new_sessions=new_sessions,
            incomplete=False,
            source="incremental",
        )
        key_new = self._daily_cache_key(codes, start_s, end_s, "3")
        self._last_status_path = self._cache_path(key_new)
        try:
            self._save_cache(key_new, merged, extra=extra, raise_errors=True)
        except Exception as exc:
            # Publication failed: the old cache and marker remain intact.
            self._fallback_panel = old_panel
            self._fallback_range = self._normalise_range(start_s, end_s)
            status = self._payload(
                requested_end=end_s,
                last_complete_date=marker.date().isoformat(),
                incomplete=True,
                reason=f"atomic publish failed: {exc}",
            )
            write_incremental_status(self._cache_path(key_new), status)
            return self._remember_status(status, source="cache_manifest")

        # The appended panel is now authoritative: re-derive ST episodes from
        # the merged raw flags (windows are fetched without touching the
        # in-memory table, so this is the only ingest for an incremental run).
        self._ingest_st_episodes(merged)
        self._fallback_panel = None
        self._fallback_range = None
        status = self._payload(
            requested_end=end_s,
            performed=True,
            new_sessions=new_sessions,
            last_complete_date=extra["last_complete_date"],
            previous_last_complete_date=extra["previous_last_complete_date"],
            updated_at=extra["updated_at"],
            data_age_days=extra["data_age_days"],
            reason=(
                f"appended {new_sessions} session(s) from "
                f"{window_start_s}..{end_s}"
            ),
        )
        write_incremental_status(self._cache_path(key_new), status)
        logger.info(
            "Incremental update appended %d session(s); completion marker %s",
            new_sessions, extra["last_complete_date"],
        )
        return self._remember_status(status, source="incremental")

    def close(self):
        """Logout from baostock."""
        self._logout()
