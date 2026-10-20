"""Tushare real data provider for A-share market.

Connects to Tushare Pro API to fetch actual CSI 300 constituent data with:
- 后复权 (hfq) adjusted prices: raw prices are multiplied by the cumulative
  ``adj_factor`` and the pipeline's divisor is set to 1.0, so the adjustment
  is applied exactly once.  (The module previously claimed 前复权/qfq while
  the code produced hfq levels.)
- HDF5 local caching for fast reloads (atomic temp-file + ``os.replace``)
- Real suspension (停牌) handling
- Delisting tracking (survivorship-bias aware: metadata includes
  ``list_status`` D/P names and per-date index membership is retained)
- Bounded retries with exponential backoff; a failed batch raises
  :class:`DataFetchIncompleteError` instead of silently shrinking the universe
- Unit normalisation at the provider boundary: volume in shares, amount in
  CNY, turnover as a fraction, market cap in CNY.

Requires: TUSHARE_TOKEN env var or config value (free registration at tushare.pro)
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import (
    DataFetchIncompleteError,
    DataProvider,
    cache_manifest_matches,
    read_cache_manifest,
    write_cache_manifest,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache format/semantics version. Combined with DATA_SCHEMA_VERSION and the
# adjustment convention in the sidecar manifest, so a schema bump (or a change
# in what the HDF5 file means) refetches instead of replaying stale data (D13).
_PROVIDER_CACHE_VERSION = "v3"
_PRICE_ADJUSTMENT = "hfq"

# Tushare unit conventions (see https://tushare.pro/document/2):
#   vol is in 手 (lots, 100 shares), amount in 千元 (thousand CNY),
#   total_mv in 万元 (ten thousand CNY), turnover_rate in percent.
_LOTS_TO_SHARES = 100.0
_THOUSAND_CNY_TO_CNY = 1_000.0
_TEN_THOUSAND_CNY_TO_CNY = 10_000.0


class TushareProvider(DataProvider):
    """Real A-share data via Tushare Pro API with HDF5 caching.

    Usage:
        provider = TushareProvider(token="your_token")
        # or set env: TUSHARE_TOKEN=your_token
        provider = TushareProvider()

    Data is cached to HDF5 files after first fetch. Subsequent runs
    reload from cache in ~1 second vs ~30 seconds for API fetch.

    ``st_available`` is False: Tushare's stock list only carries the current
    name, not an ST history, so this provider does not fabricate an
    all-False ``is_st`` column.  Callers must treat ST screening as
    unavailable (fail closed or label the result).
    """

    price_adjustment = _PRICE_ADJUSTMENT

    def __init__(
        self,
        token: str | None = None,
        start_date: str = "2021-01-01",
        end_date: str = "2025-12-31",
        index_code: str = "000300.SH",  # CSI 300
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        self.start_date = start_date
        self.end_date = end_date
        self.index_code = index_code
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

        self._ts_api = None
        self._prices: pd.DataFrame | None = None
        self._financials: pd.DataFrame | None = None
        self._benchmark: pd.Series | None = None
        self._metadata: pd.DataFrame | None = None
        self._constituent_history: pd.DataFrame | None = None

        # No point-in-time ST history is available from this source.
        self.st_available = False
        # Where get_benchmark() data came from: "index_daily" or
        # "equal_weight_fallback". Recorded in the cache manifest and pipeline
        # provenance so a degraded benchmark is not mistaken for CSI 300.
        self.benchmark_source = "unknown"
        # True only when *date-stamped* index membership (index_weight)
        # succeeded; the fallbacks are a single end-date snapshot.
        self.universe_history_available = False

        self._has_token = self.token is not None and len(self.token) > 10

        # Fail early if no token — allows caller to catch RuntimeError and fall back
        if not self._has_token:
            raise RuntimeError(
                "Tushare token required. Get one at https://tushare.pro (free registration).\n"
                "Then: set TUSHARE_TOKEN=your_token  or  pass token='your_token'"
            )

    # ------------------------------------------------------------------
    # Tushare API (lazy init)
    # ------------------------------------------------------------------

    @property
    def ts_api(self):
        if self._ts_api is None and self._has_token:
            try:
                import tushare as ts
                ts.set_token(self.token)
                self._ts_api = ts.pro_api()
                logger.info("Tushare Pro API connected")
            except Exception as e:
                logger.error("Failed to connect Tushare: %s", e)
                self._has_token = False
        return self._ts_api

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _call_with_retries(self, what: str, func, *args, **kwargs):
        """Run ``func`` with bounded retries and exponential backoff.

        Raises :class:`DataFetchIncompleteError` after the final failed attempt.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - API errors are provider-specific
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    delay = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, attempt + 1, self.max_retries, exc, delay,
                    )
                    if delay > 0:
                        time.sleep(delay)
        logger.error("%s failed after %d attempts: %s", what, self.max_retries, last_exc)
        raise DataFetchIncompleteError(
            f"{what} failed after {self.max_retries} attempt(s): {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def _project_fields(df: pd.DataFrame, fields: list[str] | None) -> pd.DataFrame:
        """Apply the shared ``fields=`` contract.

        Unknown names raise ``ValueError`` (consistent with the other
        providers) and the requested order is preserved.
        """
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
        if self._prices is None:
            self._load_or_fetch_all()
        df = self._prices.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]
        return self._project_fields(df, fields)

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if self._financials is None:
            self._load_or_fetch_all()
        return self._financials.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        if self._benchmark is None:
            self._load_or_fetch_all()
        benchmark = self._benchmark
        if isinstance(benchmark, pd.DataFrame):
            # Defensive: old caches stored the benchmark as a one-column frame.
            benchmark = benchmark.iloc[:, 0]
        return benchmark.loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]

    def get_metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._load_or_fetch_all()
        return self._metadata

    def get_st_timeseries(self) -> pd.DataFrame:
        """Explicitly expose that no ST history is available.

        Returning an empty frame (instead of inventing an all-False column)
        lets the pipeline fail closed or label ST screening as unavailable.
        """
        return pd.DataFrame(
            columns=["asset", "trigger_date", "announce_date", "end_date", "is_st"]
        )

    def get_index_membership_mask(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Per-date (date x asset) index-membership mask.

        Membership snapshots are kept as ``(trade_date, ts_code)`` history and
        forward-filled between snapshot dates: a constituent stays a member
        until the next snapshot says otherwise.  Dates before the first
        snapshot are False (unknown, never assumed to be members).
        """
        if self._constituent_history is None:
            self._load_or_fetch_all()
        history = self._constituent_history
        empty = pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=pd.Index([], name="asset"),
            dtype=bool,
        )
        if history is None or history.empty:
            return empty

        hist = history.dropna(subset=["trade_date", "ts_code"]).copy()
        hist["trade_date"] = pd.to_datetime(hist["trade_date"])
        dates = pd.DatetimeIndex(sorted(hist["trade_date"].unique()), name="date")
        assets = pd.Index(sorted(hist["ts_code"].unique()), name="asset")
        mask = pd.DataFrame(False, index=dates, columns=assets)
        for trade_date, group in hist.groupby("trade_date"):
            members = [c for c in group["ts_code"].unique() if c in mask.columns]
            if members:
                mask.loc[pd.Timestamp(trade_date), members] = True

        # A snapshot describes membership *until the next snapshot*.
        mask = mask.where(mask, np.nan).ffill().fillna(False).astype(bool)
        mask = mask.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        return mask

    # Alias exposing the same panel through the generic "universe" name.
    def get_universe_mask(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Alias of :meth:`get_index_membership_mask`."""
        return self.get_index_membership_mask(start_date, end_date)

    # ------------------------------------------------------------------
    # Cache identity / versioning (D12/D13)
    # ------------------------------------------------------------------

    def _cache_file(self) -> Path:
        return self.cache_dir / (
            f"tushare_{self.index_code}_{self.start_date}_{self.end_date}.h5"
        )

    def _cache_manifest_valid(self, cache_file: Path) -> bool:
        """Whether the HDF5 file was written under the current semantics."""
        if not cache_manifest_matches(
            cache_file,
            adjustment=_PRICE_ADJUSTMENT,
            cache_version=_PROVIDER_CACHE_VERSION,
        ):
            return False
        # A file copied/renamed from another index or date range must not be
        # served: the sidecar records the selection it was fetched for.
        manifest = read_cache_manifest(cache_file) or {}
        return (
            str(manifest.get("index_code")) == str(self.index_code)
            and str(manifest.get("start_date")) == str(self.start_date)
            and str(manifest.get("end_date")) == str(self.end_date)
        )

    def _write_cache_manifest(self, cache_file: Path) -> None:
        write_cache_manifest(
            cache_file,
            provider="tushare",
            adjustment=_PRICE_ADJUSTMENT,
            cache_version=_PROVIDER_CACHE_VERSION,
            extra={
                "index_code": self.index_code,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "benchmark_source": self.benchmark_source,
            },
        )

    def _discard_cache(self, cache_file: Path) -> None:
        for path in (cache_file, cache_file.with_name(
                f"{cache_file.name}.manifest.json")):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete stale cache %s: %s", path, exc)

    def cache_identity(self) -> dict:
        """Selection inputs that change the fetched panel (used in the key)."""
        return {
            "provider": "tushare",
            "adjustment": _PRICE_ADJUSTMENT,
            "cache_version": _PROVIDER_CACHE_VERSION,
            "index_code": self.index_code,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "synthetic": False,
        }

    def cache_universe(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        n_symbols: int | None = None,
    ) -> list[str] | None:
        """Constituent code set, loaded without deserializing the price panel.

        Reads only the ``/constituents`` table from an existing, version-valid
        HDF5 file; otherwise fetches the constituent history (the same data
        the full fetch needs, so it is cached in memory for reuse). Returns
        ``None`` when unavailable.
        """
        if self._constituent_history is None:
            cache_file = self._cache_file()
            if cache_file.exists() and self._cache_manifest_valid(cache_file):
                try:
                    with pd.HDFStore(cache_file, mode="r") as store:
                        if "/constituents" in set(store.keys()):
                            history = store["constituents"]
                            history["trade_date"] = pd.to_datetime(
                                history["trade_date"], errors="coerce"
                            )
                            self._constituent_history = history
                            self.universe_history_available = bool(
                                len(history) > 0
                            )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Could not read constituents from %s: %s",
                        cache_file, exc,
                    )
            if self._constituent_history is None:
                api = self.ts_api
                if api is None:
                    return None
                try:
                    self._constituent_history = self._fetch_constituents(api)
                except Exception as exc:
                    logger.warning(
                        "Could not fetch the constituent universe for the "
                        "cache identity: %s", exc,
                    )
                    return None
        history = self._constituent_history
        if history is None or history.empty or "ts_code" not in history.columns:
            return None
        return sorted(history["ts_code"].dropna().astype(str).unique())

    # ------------------------------------------------------------------
    # Main data loading with caching
    # ------------------------------------------------------------------

    def _load_or_fetch_all(self) -> None:
        """Load from HDF5 cache or fetch from Tushare API.

        A cache whose sidecar manifest is missing or records a different
        schema/cache/adjustment version is not trusted: it is deleted and
        refetched (D13). A cache file that cannot be read is likewise deleted
        and refetched. The cache is written only after *every* fetch
        completed; a :class:`DataFetchIncompleteError` therefore never leaves a
        partial panel behind.
        """
        cache_file = self._cache_file()

        cache_usable = cache_file.exists() and self._cache_manifest_valid(cache_file)
        if cache_file.exists() and not cache_usable:
            logger.warning(
                "Cache %s has a missing/mismatched version manifest "
                "(schema or adjustment changed); refetching", cache_file,
            )
            self._discard_cache(cache_file)
            cache_usable = False

        if cache_usable:
            logger.info("Loading from cache: %s", cache_file)
            try:
                self._load_from_cache(cache_file)
                return
            except MemoryError:
                # Not evidence of corruption: preserve the file and let the
                # failure surface (D34 applies to provider caches too).
                raise
            except OSError as exc:
                # Sharing/permission/transient IO: keep the file; refetching
                # is cheaper than losing a valid cache permanently.
                logger.warning(
                    "Cache %s could not be read (%s); keeping it and refetching",
                    cache_file, exc,
                )
            except Exception as exc:  # noqa: BLE001 - unreadable => corrupt
                logger.warning(
                    "Cache %s is corrupt/unreadable (%s); deleting and refetching",
                    cache_file, exc,
                )
                self._discard_cache(cache_file)
                if cache_file.exists():
                    raise

        if not self._has_token:
            logger.error(
                "No Tushare token found. Set TUSHARE_TOKEN env var or pass token=...\n"
                "Register for free at: https://tushare.pro"
            )
            raise RuntimeError(
                "Tushare token required. Get one at https://tushare.pro (free registration).\n"
                "Then: set TUSHARE_TOKEN=your_token  or  pass token='your_token'"
            )

        logger.info("Fetching real data from Tushare (this takes ~30s first time)...")
        self._fetch_all_from_tushare()
        try:
            self._save_to_cache(cache_file)
        except Exception as exc:  # a cache failure must not fail the research run
            logger.warning(
                "Data fetched but could not be cached to %s: %s",
                cache_file, exc,
            )
        else:
            logger.info("Data cached to: %s", cache_file)

    def _load_from_cache(self, cache_file: Path) -> None:
        """Reload all data from HDF5 cache.

        Refuses a file whose version manifest is missing or stale, so direct
        callers get the same protection as :meth:`_load_or_fetch_all` (D13).
        """
        if not self._cache_manifest_valid(cache_file):
            raise ValueError(
                "cache version manifest missing or does not match the current "
                "schema/adjustment versions"
            )
        manifest = read_cache_manifest(cache_file) or {}
        self.benchmark_source = str(
            manifest.get("benchmark_source", "unknown")
        )
        if self.benchmark_source == "equal_weight_fallback":
            logger.warning(
                "Cached Tushare benchmark is an equal-weight fallback, not "
                "the CSI 300 index; metrics compare against a constructed "
                "benchmark"
            )
        with pd.HDFStore(cache_file, mode="r") as store:
            keys = set(store.keys())
            if "/prices" not in keys:
                raise ValueError("cache is missing the prices table")
            self._prices = store["prices"]
            self._financials = store["financials"] if "/financials" in keys else pd.DataFrame()
            self._metadata = store["metadata"] if "/metadata" in keys else pd.DataFrame()

            benchmark = store["benchmark"]
            if isinstance(benchmark, pd.DataFrame):
                benchmark = benchmark.iloc[:, 0]
            benchmark.name = "benchmark"
            self._benchmark = benchmark

            if "/constituents" in keys:
                constituents = store["constituents"]
                constituents["trade_date"] = pd.to_datetime(constituents["trade_date"])
                self._constituent_history = constituents
                self.universe_history_available = bool(len(constituents) > 0)
            else:
                self._constituent_history = pd.DataFrame(
                    columns=["trade_date", "ts_code"]
                )
                self.universe_history_available = False

        logger.info("Loaded from cache: %d assets, %d price rows",
                     len(self._metadata), len(self._prices))

    def _save_to_cache(self, cache_file: Path) -> None:
        """Persist all data to HDF5 atomically (temp file + ``os.replace``).

        The version manifest is written right after the data file. A crash
        between the two leaves a cache without a valid manifest, which the
        loader treats as stale and refetches -- never as current data.
        """
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_name(
            f"{cache_file.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            with pd.HDFStore(tmp_file, mode="w", complevel=5, complib="zlib") as store:
                store["prices"] = self._prices
                store["financials"] = self._financials
                store["benchmark"] = self._benchmark.to_frame("benchmark")
                store["metadata"] = self._metadata
                if self._constituent_history is not None:
                    store["constituents"] = self._constituent_history
            os.replace(tmp_file, cache_file)
            try:
                self._write_cache_manifest(cache_file)
            except Exception as exc:  # cache metadata is best effort
                logger.warning(
                    "Could not write cache version manifest for %s: %s",
                    cache_file, exc,
                )
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    logger.warning("Could not remove temp cache file %s", tmp_file)

    # ------------------------------------------------------------------
    # Tushare API fetching
    # ------------------------------------------------------------------

    def _fetch_all_from_tushare(self) -> None:
        """Fetch complete dataset from Tushare Pro API.

        Any failure raises :class:`DataFetchIncompleteError`; the caller
        (`_load_or_fetch_all`) only writes the cache afterwards, so a
        partial dataset is never persisted.
        """
        api = self.ts_api
        if api is None:
            raise RuntimeError("Tushare API not available")

        # 1. Get CSI 300 constituents over time
        if self._constituent_history is None or self._constituent_history.empty:
            logger.info("  Fetching CSI 300 constituent history...")
            self._constituent_history = self._fetch_constituents(api)
        else:
            # Already resolved for the cache key; reuse instead of re-fetching.
            logger.info("  Reusing pre-resolved CSI 300 constituent history...")
            self.universe_history_available = bool(
                len(self._constituent_history) > 0
            )
        constituents = self._constituent_history
        all_codes = list(constituents["ts_code"].unique())
        logger.info("  Got %d unique constituent stocks", len(all_codes))

        # 2. Get stock basic info (metadata; includes delisted names)
        logger.info("  Fetching stock basic info...")
        self._metadata = self._fetch_stock_basic(api, all_codes)
        logger.info("  Metadata: %d stocks", len(self._metadata))

        # 3. Get daily price data (hfq; adjustment applied exactly once)
        logger.info("  Fetching daily prices (hfq, may take a while)...")
        self._prices = self._fetch_daily_prices(api, all_codes)
        logger.info("  Prices: %d rows", len(self._prices))

        # 4. Get financial data (PIT: keyed on announcement date)
        logger.info("  Fetching financial statements...")
        self._financials = self._fetch_financials(api, all_codes)
        logger.info("  Financials: %d rows", len(self._financials))

        # 5. Get benchmark (CSI 300 index)
        logger.info("  Fetching CSI 300 index data...")
        self._benchmark = self._fetch_index_daily(api)
        logger.info("  Benchmark: %d rows", len(self._benchmark))

    def _fetch_constituents(self, api) -> pd.DataFrame:
        """Fetch CSI 300 constituent changes over time.

        The returned frame keeps one row per ``(trade_date, ts_code)``
        snapshot.  Only the ``index_weight`` path supplies real per-date
        history; the fallbacks are a single end-date snapshot and are
        flagged through ``universe_history_available``.
        """
        frames = []
        self.universe_history_available = False
        # CSI 300 = 000300.SH
        try:
            df = self._call_with_retries(
                "index_weight",
                api.index_weight,
                index_code=self.index_code,
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
            )
            if df is not None and len(df) > 0:
                df = df[["trade_date", "ts_code"]].drop_duplicates().copy()
                df["trade_date"] = pd.to_datetime(
                    df["trade_date"], format="%Y%m%d", errors="coerce"
                )
                df = df.dropna(subset=["trade_date", "ts_code"])
                frames.append(df)
                self.universe_history_available = True
            else:
                logger.warning("index_weight returned no rows; using a snapshot fallback")
        except Exception as e:
            logger.warning("index_weight failed: %s, trying index_member", e)

        if not frames:
            # Fallback: fetch current members only (single end-date snapshot).
            snapshot_date = self.end_date.replace("-", "")
            try:
                df = self._call_with_retries(
                    "index_member",
                    api.index_member,
                    index_code=self.index_code,
                    trade_date=snapshot_date,
                )
                if df is not None and len(df) > 0:
                    df = df[["ts_code"]].drop_duplicates().copy()
                    df["trade_date"] = pd.to_datetime(
                        snapshot_date, format="%Y%m%d", errors="coerce"
                    )
                    frames.append(df)
            except Exception as e2:
                logger.warning("index_member also failed: %s", e2)

        if not frames:
            # Last resort: get CSI 300 stock list directly
            logger.warning("Cannot fetch constituents, trying HS300 stocks directly")
            snapshot_date = self.end_date.replace("-", "")
            try:
                df = self._call_with_retries(
                    "hs300",
                    api.hs300,
                    trade_date=snapshot_date,
                    fields="ts_code,weight",
                )
                if df is not None and len(df) > 0:
                    df = df[["ts_code"]].drop_duplicates().copy()
                    df["trade_date"] = pd.to_datetime(
                        snapshot_date, format="%Y%m%d", errors="coerce"
                    )
                    frames.append(df)
            except Exception as exc:
                logger.warning("hs300 also failed: %s", exc)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            return result.drop_duplicates(subset=["trade_date", "ts_code"])
        return pd.DataFrame(columns=["trade_date", "ts_code"])

    def _fetch_stock_basic(self, api, codes: list[str]) -> pd.DataFrame:
        """Fetch stock metadata: name, industry, list date, delisting date.

        ``list_status`` L (listed), P (paused) and D (delisted) are all
        fetched so historical cross-sections do not silently lose delisted
        constituents.  ``is_st`` is deliberately **not** fabricated: Tushare
        only carries the current name, so an all-False flag would pretend ST
        screening works.  ``st_available`` stays False.
        """
        frames = []
        for status in ("L", "P", "D"):
            df = self._call_with_retries(
                f"stock_basic[{status}]",
                api.stock_basic,
                exchange="",
                list_status=status,
                fields="ts_code,name,industry,list_date,delist_date,is_hs",
            )
            if df is not None and len(df) > 0:
                frames.append(df)

        if frames:
            raw = pd.concat(frames, ignore_index=True)
            raw = raw.drop_duplicates(subset=["ts_code"], keep="last")
            raw = raw.set_index("ts_code")
            df = raw.reindex(codes)

            # Map Tushare industry names to our sector names
            df["sector"] = df["industry"].fillna("综合")
            df["sector"] = df["sector"].apply(self._map_sector)
            df["market_cap_group"] = "mid"  # Placeholder; not used by the pipeline
            list_date = (
                df["list_date"] if "list_date" in df.columns
                else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            )
            delist_date = (
                df["delist_date"] if "delist_date" in df.columns
                else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            )
            df["listing_date"] = pd.to_datetime(
                list_date, format="%Y%m%d", errors="coerce"
            )
            df["delisting_date"] = pd.to_datetime(
                delist_date, format="%Y%m%d", errors="coerce"
            )

            result = df[
                ["sector", "market_cap_group", "listing_date", "delisting_date"]
            ].copy()
            result.index.name = "asset"
            result.attrs["st_available"] = False
            return result

        # API succeeded but returned nothing at all: keep the panel usable
        # with an explicit, conservative fallback (no delisting knowledge).
        logger.warning(
            "stock_basic returned no rows for any list_status; metadata has no "
            "delisting information (survivorship bias not mitigated)"
        )
        result = pd.DataFrame({
            "sector": "综合",
            "market_cap_group": "mid",
            "listing_date": pd.Timestamp(self.start_date),
            "delisting_date": pd.NaT,
        }, index=pd.Index(codes, name="asset"))
        result.attrs["st_available"] = False
        return result

    def _map_sector(self, industry: str) -> str:
        """Map Tushare industry names to our standard sectors."""
        mapping = {
            "银行": "银行", "保险": "非银金融", "证券": "非银金融", "多元金融": "非银金融",
            "房地产": "房地产", "房地产开发": "房地产",
            "医药": "医药生物", "医疗保健": "医药生物", "医药生物": "医药生物",
            "食品饮料": "食品饮料", "酿酒": "食品饮料",
            "电子": "电子", "半导体": "电子", "元器件": "电子",
            "计算机": "计算机", "软件服务": "计算机", "IT设备": "计算机",
            "通信设备": "通信", "通信": "通信",
            "传媒娱乐": "传媒", "互联网": "传媒",
            "汽车": "汽车", "汽车类": "汽车",
            "家用电器": "家用电器", "家电": "家用电器",
            "电气设备": "电力设备", "电力": "公用事业", "电力设备": "电力设备",
            "化工": "基础化工", "基础化工": "基础化工",
            "有色": "有色金属", "有色金属": "有色金属",
            "钢铁": "钢铁", "煤炭": "煤炭", "石油": "石油石化", "石油石化": "石油石化",
            "机械": "机械设备", "工业机械": "机械设备", "机械设备": "机械设备",
            "建筑": "建筑装饰", "建材": "建筑材料", "建筑材料": "建筑材料",
            "交通运输": "交通运输", "交通设施": "交通运输",
            "环保": "环保",
            "农林牧渔": "农林牧渔", "农业": "农林牧渔",
            "商贸": "商贸零售", "商业连锁": "商贸零售",
            "纺织": "纺织服饰", "服饰": "纺织服饰",
            "军工": "国防军工", "航空": "国防军工",
        }
        for key, val in mapping.items():
            if key in (industry or ""):
                return val
        return "综合"

    # ------------------------------------------------------------------
    # Daily prices
    # ------------------------------------------------------------------

    def _fetch_batch_daily(self, api, batch: list[str]) -> pd.DataFrame:
        """Fetch one daily batch; raise if any requested symbol is absent."""
        ts_codes = ",".join(batch)
        df = api.daily(
            ts_code=ts_codes,
            start_date=self.start_date.replace("-", ""),
            end_date=self.end_date.replace("-", ""),
            fields="ts_code,trade_date,open,high,low,close,vol,amount,turnover_rate",
        )
        if df is None or len(df) == 0:
            raise ValueError("daily returned no rows")
        missing = set(batch) - set(df["ts_code"].astype(str).unique())
        if missing:
            raise ValueError(f"daily response missing symbols: {sorted(missing)}")
        return df

    def _fetch_batch_adj(self, api, batch: list[str]) -> pd.DataFrame:
        """Fetch one adj_factor batch; raise if any requested symbol is absent."""
        ts_codes = ",".join(batch)
        adj_df = api.adj_factor(
            ts_code=ts_codes,
            start_date=self.start_date.replace("-", ""),
            end_date=self.end_date.replace("-", ""),
        )
        if adj_df is None or len(adj_df) == 0:
            raise ValueError("adj_factor returned no rows")
        missing = set(batch) - set(adj_df["ts_code"].astype(str).unique())
        if missing:
            raise ValueError(f"adj_factor response missing symbols: {sorted(missing)}")
        return adj_df[["ts_code", "trade_date", "adj_factor"]]

    def _fetch_daily_prices(self, api, codes: list[str]) -> pd.DataFrame:
        """Fetch daily OHLCV data with 后复权 (hfq) adjustment.

        Tushare reports raw prices; multiplying by the cumulative
        ``adj_factor`` produces hfq levels.  The pipeline then divides by the
        stored ``adj_factor`` column (constant 1.0 here), so the adjustment
        is applied exactly once.

        Every batch is retried with exponential backoff.  If any batch (daily
        or adj_factor) still fails, the whole fetch raises
        :class:`DataFetchIncompleteError` — a partially adjusted panel (raw prices
        next to hfq prices) must never be returned or cached.
        """
        all_frames: list[pd.DataFrame] = []
        adj_frames: list[pd.DataFrame] = []
        failed_symbols: set[str] = set()
        batch_size = 50  # Tushare API limit

        for i in range(0, len(codes), batch_size):
            batch = list(codes[i:i + batch_size])

            try:
                all_frames.append(
                    self._call_with_retries(
                        f"daily batch {i // batch_size}", self._fetch_batch_daily, api, batch
                    )
                )
            except DataFetchIncompleteError as exc:
                logger.error("%s", exc)
                failed_symbols.update(batch)

            try:
                adj_frames.append(
                    self._call_with_retries(
                        f"adj_factor batch {i // batch_size}", self._fetch_batch_adj, api, batch
                    )
                )
            except DataFetchIncompleteError as exc:
                logger.error("%s", exc)
                failed_symbols.update(batch)

        if failed_symbols:
            raise DataFetchIncompleteError(
                "Tushare price fetch incomplete: "
                f"{len(failed_symbols)} symbol(s) failed after retries "
                f"(e.g. {sorted(failed_symbols)[:10]})"
            )
        if not all_frames or not adj_frames:
            raise DataFetchIncompleteError("Tushare price fetch returned no data")

        df = pd.concat(all_frames, ignore_index=True)

        # Process
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.rename(columns={
            "ts_code": "asset",
            "trade_date": "date",
            "vol": "volume",
            "amount": "amount",
            "turnover_rate": "turnover",
        })

        # Unit normalisation at the provider boundary.
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * _LOTS_TO_SHARES
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * _THOUSAND_CNY_TO_CNY
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0

        # VWAP before adjustment: amount (CNY) / volume (shares).  Zero-volume
        # rows yield inf; replace non-finite values with close below.
        df["vwap"] = df["amount"] / df["volume"]
        df["vwap"] = df["vwap"].where(np.isfinite(df["vwap"]), df["close"])

        adj = pd.concat(adj_frames, ignore_index=True)
        adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")
        adj = adj.rename(columns={
            "ts_code": "asset", "trade_date": "date", "adj_factor": "cum_factor",
        })
        adj["cum_factor"] = pd.to_numeric(adj["cum_factor"], errors="coerce")

        df = df.merge(adj, on=["date", "asset"], how="left")
        bad_factor = (
            df["cum_factor"].isna()
            | ~np.isfinite(df["cum_factor"])
            | (df["cum_factor"] <= 0)
        )
        if bool(bad_factor.any()):
            missing = sorted(
                df.loc[bad_factor, "asset"].astype(str).unique().tolist()
            )
            raise DataFetchIncompleteError(
                "Tushare adj_factor incomplete: missing/invalid cumulative "
                f"factor for {int(bad_factor.sum())} row(s), symbols {missing[:10]}"
            )

        # Apply the factor exactly once.  `adj_factor` (the pipeline divisor)
        # is 1.0 afterwards because the factor is already in the price level.
        for column in ("open", "high", "low", "close", "vwap"):
            df[column] = pd.to_numeric(df[column], errors="coerce") * df["cum_factor"]
        df = df.drop(columns=["cum_factor"])

        # Set MultiIndex
        df = df.set_index(["date", "asset"]).sort_index()

        df["adj_factor"] = 1.0  # hfq adjustment already applied above

        # Handle suspensions: volume=0 means suspended
        suspended = df["volume"].isna() | (df["volume"] <= 0)
        df.loc[suspended, ["open", "high", "low", "close", "vwap"]] = np.nan

        # Forward-fill suspension gaps (max 5 days for real data)
        df = df.groupby("asset").ffill(limit=5)

        # Drop rows where close is still NaN after ffill
        df = df.dropna(subset=["close"])

        return df

    # ------------------------------------------------------------------
    # Financials (PIT)
    # ------------------------------------------------------------------

    def _fetch_financials(self, api, codes: list[str]) -> pd.DataFrame:
        """Fetch quarterly financial indicators, keyed to announcement dates.

        Rows are mapped to the first trading day on/after their announcement
        (``f_ann_date``, falling back to ``ann_date``), then forward-filled —
        never backward-filled.  If any batch fails after retries, or the API
        returns no usable rows, this raises instead of substituting synthetic
        financials that would then be cached as real data.
        """
        if self._prices is None:
            self._prices = self._fetch_daily_prices(api, codes)

        all_frames: list[pd.DataFrame] = []
        failed_symbols: set[str] = set()

        for i in range(0, len(codes), 30):
            batch = list(codes[i:i + 30])
            ts_codes = ",".join(batch)
            try:
                df = self._call_with_retries(
                    f"fina_indicator batch {i // 30}",
                    api.fina_indicator,
                    ts_code=ts_codes,
                    start_date=self.start_date.replace("-", ""),
                    end_date=self.end_date.replace("-", ""),
                    fields="ts_code,ann_date,f_ann_date,end_date,total_mv,total_assets,"
                           "total_hldr_eqy_inc_min_int,revenue,net_profit,roe,pb,pe,eps",
                )
            except DataFetchIncompleteError as exc:
                logger.error("%s", exc)
                failed_symbols.update(batch)
                continue
            if df is not None and len(df) > 0:
                all_frames.append(df)

        if failed_symbols:
            raise DataFetchIncompleteError(
                "Tushare financial fetch incomplete: "
                f"{len(failed_symbols)} symbol(s) failed after retries "
                f"(e.g. {sorted(failed_symbols)[:10]})"
            )
        if not all_frames:
            raise DataFetchIncompleteError(
                "Tushare returned no financial data; refusing to substitute "
                "synthetic financials and cache them as real"
            )

        df = pd.concat(all_frames, ignore_index=True)
        required = {"ts_code", "end_date", "ann_date", "f_ann_date"}
        missing = required - set(df.columns)
        if missing:
            raise DataFetchIncompleteError(
                f"Tushare fina_indicator response missing announcement columns: "
                f"{sorted(missing)}"
            )

        df["period_end"] = pd.to_datetime(
            df["end_date"], format="%Y%m%d", errors="coerce"
        )
        ann_date = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
        f_ann_date = pd.to_datetime(df["f_ann_date"], format="%Y%m%d", errors="coerce")
        # Actual announcement date wins; ann_date is the fallback.
        publish_date = f_ann_date.where(f_ann_date.notna(), ann_date)
        missing_ann = int(publish_date.isna().sum())
        if missing_ann:
            logger.warning(
                "Tushare financials: %d row(s) have no ann_date/f_ann_date and "
                "cannot be made point-in-time; dropping them",
                missing_ann,
            )
            keep = publish_date.notna()
            df = df.loc[keep].copy()
            publish_date = publish_date.loc[keep]
        if df.empty:
            raise DataFetchIncompleteError(
                "Tushare financials have no announcement dates; cannot build a "
                "point-in-time panel"
            )

        df = df.rename(columns={"ts_code": "asset"})
        df["publish_date"] = publish_date.to_numpy()
        df["market_cap"] = (
            pd.to_numeric(df["total_mv"], errors="coerce") * _TEN_THOUSAND_CNY_TO_CNY
        )
        df["total_assets"] = pd.to_numeric(df["total_assets"], errors="coerce")
        df["net_assets"] = pd.to_numeric(
            df["total_hldr_eqy_inc_min_int"], errors="coerce"
        )
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        df["net_profit"] = pd.to_numeric(df["net_profit"], errors="coerce")
        # Tushare reports roe in percent; the provider contract is a fraction.
        df["roe"] = pd.to_numeric(df["roe"], errors="coerce") / 100.0
        df["pb_ratio"] = pd.to_numeric(df["pb"], errors="coerce")
        df["pe_ratio"] = pd.to_numeric(df["pe"], errors="coerce")

        # Asset growth YoY (4 quarters) on the report sequence.
        df = df.sort_values(["asset", "period_end"])
        df["asset_growth"] = (
            df.groupby("asset", sort=False)["total_assets"]
            .pct_change(4, fill_method=None)
        )

        # Map each report to the first trading day on/after its announcement.
        trading_dates = pd.DatetimeIndex(
            self._prices.index.get_level_values("date").unique()
        ).sort_values()
        if len(trading_dates) == 0:
            raise DataFetchIncompleteError("Tushare financials: empty price calendar")
        positions = trading_dates.searchsorted(
            df["publish_date"].to_numpy(), side="left"
        )
        in_sample = positions < len(trading_dates)
        dropped = int((~in_sample).sum())
        if dropped:
            logger.warning(
                "Tushare financials: %d asset-report(s) published after the "
                "last trading day (%s) are outside the sample",
                dropped, trading_dates[-1].date(),
            )
        df = df.loc[in_sample].copy()
        if df.empty:
            raise DataFetchIncompleteError(
                "Tushare financials: no report was published inside the sample"
            )
        df["date"] = trading_dates[positions[in_sample]]

        keep_cols = [
            "date", "asset", "market_cap", "total_assets", "net_assets",
            "revenue", "net_profit", "roe", "pb_ratio", "pe_ratio",
            "asset_growth", "publish_date",
        ]
        df = df[keep_cols]
        df = df.drop_duplicates(subset=["date", "asset"], keep="last")
        df = df.set_index(["date", "asset"]).sort_index()

        # Forward-fill to daily frequency (a report stays valid until the next
        # one); deliberately no bfill.
        full_idx = pd.MultiIndex.from_product(
            [trading_dates, pd.Index(codes, name="asset")],
            names=["date", "asset"],
        )
        df_full = df.reindex(full_idx)
        df_full = df_full.groupby("asset").ffill()

        return df_full

    def _fetch_index_daily(self, api) -> pd.Series:
        """Fetch CSI 300 index daily data (trailing returns convention)."""
        try:
            df = self._call_with_retries(
                "index_daily",
                api.index_daily,
                ts_code=self.index_code,
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
                fields="trade_date,pct_chg",
            )
            if df is not None and len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                df = df.set_index("trade_date").sort_index()
                returns = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
                returns.name = "benchmark"
                self.benchmark_source = "index_daily"
                return returns.dropna()
        except Exception as e:
            logger.warning("Failed to fetch CSI 300 index: %s", e)

        # Fallback: compute equal-weight benchmark from prices
        logger.info("Using equal-weight benchmark as fallback")
        self.benchmark_source = "equal_weight_fallback"
        if self._prices is None or self._prices.empty:
            return pd.Series(dtype=float, name="benchmark")
        close = self._prices["close"].unstack("asset")
        daily_ret = close.pct_change(fill_method=None).mean(axis=1)
        daily_ret.name = "benchmark"
        return daily_ret.fillna(0)
