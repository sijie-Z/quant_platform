"""Tushare real data provider for A-share market.

Connects to Tushare Pro API to fetch actual CSI 300 constituent data with:
- **Raw** OHLCV prices plus the corporate-action ``adj_factor`` divisor.
  Tushare's native ``adj_factor`` is a cumulative multiplier
  (``hfq = raw * adj_factor``); the provider inverts it to the shared
  divisor contract (``adj_factor = 1 / native_factor``), so the pipeline's
  ``close_adj = close / adj_factor`` reproduces the same hfq series as
  before while ``close`` remains a tradeable CNY price (D15/D18).  Research
  returns are therefore unchanged; money paths use ``close``.
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

# Cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache format/semantics version. Combined with DATA_SCHEMA_VERSION and the
# adjustment convention in the sidecar manifest, so a schema bump (or a change
# in what the HDF5 file means) refetches instead of replaying stale data (D13).
# v4: price semantics changed from pre-multiplied hfq levels (adj_factor=1) to
# raw OHLCV plus a divisor; v3 files must never be replayed.
_PROVIDER_CACHE_VERSION = "v4"
#: Basis of the stored ``close`` column (raw traded price).
_PRICE_ADJUSTMENT = "raw"
#: Basis of ``close / adj_factor`` (the return-generating series).
_ADJUSTED_PRICE_BASIS = "hfq"

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
    adjusted_price_basis = _ADJUSTED_PRICE_BASIS
    adjustment_factor_available = True
    #: Sessions appended from an existing cache via :meth:`update`; a full
    #: fetch is always available as the fallback.
    supports_incremental = True

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
        #: Completion marker/provenance of the loaded panel (D32). Populated
        #: from the cache manifest on load and after every successful fetch.
        self._data_status: dict = {}

        #: Corporate-action feed (``dividend`` endpoint).  Fetched lazily by
        #: :meth:`get_corporate_actions`; ``corporate_actions_available`` is
        #: True only after at least one successful API call.
        self._corporate_actions: pd.DataFrame | None = None
        self._corporate_actions_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self.corporate_actions_available = False

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

                # tushare 1.4.x posts the token and payload over plain HTTP
                # (`http://api.waditu.com/dataapi`). Force TLS before the
                # client is built so credentials never travel in clear text.
                from tushare.pro.client import DataApi

                if not getattr(DataApi, "_DataApi__http_url", "").startswith("https://"):
                    DataApi._DataApi__http_url = "https://api.waditu.com/dataapi"
                ts.set_token(self.token)
                self._ts_api = ts.pro_api()
                logger.info("Tushare Pro API connected (https)")
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
        df = self._project_fields(df, fields)
        df.attrs.update({
            "price_basis": _PRICE_ADJUSTMENT,
            "adjusted_price_basis": _ADJUSTED_PRICE_BASIS,
            "adjustment_factor_available": True,
        })
        return df

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

    def data_status(self) -> dict:
        """Completion marker of the loaded panel (D32).

        A failed incremental attempt leaves the data file and its manifest
        untouched, so its ``incomplete`` flag lives in a sidecar. It is
        surfaced here when it refers to the same completion marker, making a
        restarted process aware that the last append attempt was partial.
        """
        status = dict(self._data_status)
        sidecar = read_incremental_status(self._cache_file())
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

    def get_st_timeseries(self) -> pd.DataFrame:
        """Explicitly expose that no ST history is available.

        Returning an empty frame (instead of inventing an all-False column)
        lets the pipeline fail closed or label ST screening as unavailable.
        """
        return pd.DataFrame(
            columns=["asset", "trigger_date", "announce_date", "end_date", "is_st"]
        )

    # ------------------------------------------------------------------
    # Corporate actions (best effort)
    # ------------------------------------------------------------------

    def get_corporate_actions(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Cash dividends and splits from Tushare's ``dividend`` endpoint.

        Best effort: the request window is widened by 400 calendar days on
        the left because ``start_date``/``end_date`` are interpreted against
        the announcement date, then rows are filtered on ``ex_date``.  Only
        implemented plans (``div_proc`` containing 实施) are kept, so a
        proposal is never paid twice.  ``cash_div_tax`` (pre-tax, per share)
        is the contract amount -- the paper engine withholds tax per its own
        config.  If the API is unavailable, returns the documented empty
        frame and leaves ``corporate_actions_available`` False; no amount is
        ever inferred.
        """
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if (
            self._corporate_actions is not None
            and self._corporate_actions_range is not None
            and self._corporate_actions_range[0] <= start
            and end <= self._corporate_actions_range[1]
        ):
            frame = self._corporate_actions
            mask = (frame["ex_date"] >= start) & (frame["ex_date"] <= end)
            return frame.loc[mask].reset_index(drop=True).copy()

        frame = self._fetch_corporate_actions(start, end)
        self._corporate_actions = frame
        self._corporate_actions_range = (start, end)
        return frame.copy()

    def _dividend_code_batches(self) -> list[str | None]:
        """Symbol batches for the dividend call (``None`` = market-wide).

        Per-code batches avoid the endpoint's per-call row cap truncating a
        multi-year market-wide response.  When the constituent panel has not
        been loaded yet (the accessor is deliberately usable before a price
        fetch), fall back to one market-wide call.
        """
        codes: list[str] = []
        if self._metadata is not None and len(self._metadata):
            codes = sorted(str(code) for code in self._metadata.index)
        elif self._constituent_history is not None and len(self._constituent_history):
            codes = sorted(
                self._constituent_history["ts_code"].astype(str).unique()
            )
        if not codes:
            return [None]
        return [
            ",".join(codes[i:i + 50]) for i in range(0, len(codes), 50)
        ]

    def _fetch_corporate_actions(
        self, start: pd.Timestamp, end: pd.Timestamp,
    ) -> pd.DataFrame:
        api = self.ts_api
        if api is None:
            logger.warning(
                "Tushare API unavailable; corporate actions are unavailable"
            )
            self.corporate_actions_available = False
            return empty_corporate_actions()

        fields = (
            "ts_code,ex_date,cash_div,cash_div_tax,stk_div,"
            "stk_bo_rate,stk_co_rate,div_proc"
        )
        fetch_start = (start - pd.Timedelta(days=400)).strftime("%Y%m%d")
        fetch_end = end.strftime("%Y%m%d")
        batches = self._dividend_code_batches()
        frames: list[pd.DataFrame] = []
        failures = 0
        for batch in batches:
            kwargs = {
                "start_date": fetch_start,
                "end_date": fetch_end,
                "fields": fields,
            }
            if batch is not None:
                kwargs["ts_code"] = batch
            try:
                df = self._call_with_retries(
                    "dividend", api.dividend, **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - best effort feed
                failures += 1
                logger.warning(
                    "Tushare dividend fetch failed for %s: %s",
                    "market" if batch is None else f"{len(batch.split(','))} codes",
                    exc,
                )
                continue
            if df is not None and len(df):
                if len(df) >= 2000 and batch is None:
                    logger.warning(
                        "Tushare dividend returned %d rows in one market-wide "
                        "call (endpoint cap ~2000); the feed may be truncated. "
                        "Load the constituent panel first for per-code batches.",
                        len(df),
                    )
                frames.append(df)

        if batches and failures == len(batches):
            self.corporate_actions_available = False
            return empty_corporate_actions()
        self.corporate_actions_available = True

        if not frames:
            return empty_corporate_actions()
        raw = pd.concat(frames, ignore_index=True)
        return self._parse_dividend_rows(raw, start, end)

    @staticmethod
    def _parse_dividend_rows(
        raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Map Tushare ``dividend`` rows to the corporate-action contract."""
        records: list[dict] = []
        post_tax_fallbacks = 0
        has_process = "div_proc" in raw.columns
        for _, row in raw.iterrows():
            if has_process:
                process = str(row.get("div_proc") or "").strip()
                if process and "实施" not in process:
                    continue
            ex_date = pd.to_datetime(
                row.get("ex_date"), format="%Y%m%d", errors="coerce",
            )
            if pd.isna(ex_date) or ex_date < start or ex_date > end:
                continue
            ts_code = row.get("ts_code")
            if ts_code is None or pd.isna(ts_code):
                continue
            asset = normalize_symbol(ts_code)
            cash = pd.to_numeric(
                pd.Series([row.get("cash_div_tax")]), errors="coerce",
            ).iloc[0]
            if pd.isna(cash) or cash <= 0:
                # `cash_div` is the *after-tax* per-share amount.  Falling
                # back to it keeps the feed usable where the pre-tax column
                # is absent, but a non-zero withholding config would then
                # withhold twice; warn instead of silently changing basis.
                cash = pd.to_numeric(
                    pd.Series([row.get("cash_div")]), errors="coerce",
                ).iloc[0]
                if pd.notna(cash) and cash > 0:
                    post_tax_fallbacks += 1
            cash = float(cash) if pd.notna(cash) and cash > 0 else 0.0
            bonus = pd.to_numeric(
                pd.Series([row.get("stk_div")]), errors="coerce",
            ).iloc[0]
            if pd.isna(bonus) or bonus <= 0:
                parts = [
                    pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
                    for key in ("stk_bo_rate", "stk_co_rate")
                ]
                parts = [float(p) for p in parts if pd.notna(p) and p > 0]
                bonus = sum(parts) if parts else 0.0
            bonus = float(bonus) if pd.notna(bonus) and bonus > 0 else 0.0
            if cash <= 0 and bonus <= 0:
                continue
            if cash > 0:
                records.append({
                    "ex_date": ex_date,
                    "asset": asset,
                    "kind": "cash",
                    "cash_per_share": round(cash, 6),
                    "split_ratio": 1.0,
                })
            if bonus > 0:
                records.append({
                    "ex_date": ex_date,
                    "asset": asset,
                    "kind": "split",
                    "cash_per_share": 0.0,
                    "split_ratio": round(1.0 + bonus, 6),
                })
        if post_tax_fallbacks:
            logger.warning(
                "Tushare dividend: %d row(s) had no pre-tax amount; using the "
                "after-tax cash_div. Configure cash_dividend_tax_rate=0 for "
                "those rows or the tax is withheld twice",
                post_tax_fallbacks,
            )
        if not records:
            return empty_corporate_actions()
        return normalize_corporate_actions(
            pd.DataFrame(records, columns=list(CORPORATE_ACTION_COLUMNS))
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

        Every snapshot row lists both current members (True) and non-members
        (False), so a plain forward fill propagates an explicit *exit* as
        faithfully as an inclusion.  An earlier implementation masked the
        False cells to NaN first, which made a name that ever entered the
        index a member for the rest of the sample.
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

        # A snapshot describes membership *until the next snapshot*: forward
        # fill both True (inclusion persists) and False (explicit exit), then
        # default dates before the first snapshot to non-members.
        mask = mask.ffill().fillna(False).astype(bool)
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

    def _cache_manifest_valid(
        self, cache_file: Path, *, expected_end: str | None = None,
    ) -> bool:
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
        expected_end = self.end_date if expected_end is None else expected_end
        return (
            str(manifest.get("index_code")) == str(self.index_code)
            and str(manifest.get("start_date")) == str(self.start_date)
            and str(manifest.get("end_date")) == str(expected_end)
        )

    def _write_cache_manifest(
        self, cache_file: Path, extra: dict | None = None,
    ) -> None:
        payload = {
            "index_code": self.index_code,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "benchmark_source": self.benchmark_source,
            "adjusted_price_basis": _ADJUSTED_PRICE_BASIS,
            "adj_factor_role": "divisor",
        }
        if extra:
            payload.update(extra)
        write_cache_manifest(
            cache_file,
            provider="tushare",
            adjustment=_PRICE_ADJUSTMENT,
            cache_version=_PROVIDER_CACHE_VERSION,
            extra=payload,
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
            "adjusted_price_basis": _ADJUSTED_PRICE_BASIS,
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
        panel_end = panel_last_date(self._prices)
        delisted = self._delisted_symbols(self._metadata, panel_end)
        marker, incomplete = complete_panel_marker(
            self._prices, delisted=delisted,
        )
        marker_extra = incremental_marker_extra(
            last_complete_date=marker,
            requested_end=self.end_date,
            new_sessions=0,
            incomplete=incomplete,
            source="full_fetch",
        )
        self._data_status = {
            "last_complete_date": marker_extra["last_complete_date"],
            "requested_end": marker_extra["requested_end"],
            "updated_at": marker_extra["updated_at"],
            "data_age_days": marker_extra["data_age_days"],
            "incomplete": bool(incomplete),
            "source": "full_fetch",
        }
        try:
            self._save_to_cache(cache_file, extra=marker_extra)
        except Exception as exc:  # a cache failure must not fail the research run
            logger.warning(
                "Data fetched but could not be cached to %s: %s",
                cache_file, exc,
            )
        else:
            logger.info("Data cached to: %s", cache_file)

    def _load_from_cache(
        self, cache_file: Path, *, expected_end: str | None = None,
    ) -> None:
        """Reload all data from HDF5 cache.

        Refuses a file whose version manifest is missing or stale, so direct
        callers get the same protection as :meth:`_load_or_fetch_all` (D13).
        ``expected_end`` validates a candidate written for an earlier end
        date during the incremental-update scan; the default validates the
        provider's current request.
        """
        if not self._cache_manifest_valid(cache_file, expected_end=expected_end):
            raise ValueError(
                "cache version manifest missing or does not match the current "
                "schema/adjustment versions"
            )
        manifest = read_cache_manifest(cache_file) or {}
        self.benchmark_source = str(
            manifest.get("benchmark_source", "unknown")
        )
        self._data_status = {
            "last_complete_date": manifest.get("last_complete_date"),
            "requested_end": manifest.get("end_date"),
            "updated_at": manifest.get("updated_at"),
            "data_age_days": manifest.get("data_age_days"),
            "incomplete": bool(manifest.get("incomplete", False)),
            "source": "cache_manifest",
        }
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

    def _save_to_cache(self, cache_file: Path, extra: dict | None = None) -> None:
        """Persist all data to HDF5 atomically (temp file + ``os.replace``).

        Publication (data file + version/completion manifest) happens under a
        cross-process lock. The data file is replaced first and the manifest
        only afterwards, so a crash between the two leaves either the old
        entry or an unversioned file that the loader rejects -- never new data
        advertised under an old completion marker.
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
            with provider_cache_lock(cache_file.parent) as locked:
                if not locked:
                    raise TimeoutError(
                        f"timed out acquiring the provider cache lock for "
                        f"{cache_file}"
                    )
                os.replace(tmp_file, cache_file)
                self._write_cache_manifest(cache_file, extra=extra)
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
        # Sorted so a full fetch and an incremental update describe the same
        # panel in the same level order (byte-identical output).
        all_codes = sorted(constituents["ts_code"].astype(str).unique())
        logger.info("  Got %d unique constituent stocks", len(all_codes))

        # 2. Get stock basic info (metadata; includes delisted names)
        logger.info("  Fetching stock basic info...")
        self._metadata = self._fetch_stock_basic(api, all_codes)
        logger.info("  Metadata: %d stocks", len(self._metadata))

        # 3. Get daily price data (raw OHLCV + adjustment divisor)
        logger.info("  Fetching daily prices (raw + adj_factor, may take a while)...")
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

    def _fetch_batch_daily(
        self,
        api,
        batch: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Fetch one daily batch.

        ``strict`` is the full-fetch contract: any requested symbol absent
        from the response makes the batch unusable. Incremental windows relax
        it because a symbol suspended for the whole window legitimately has no
        rows; :meth:`update` then applies its own completion rules.
        """
        ts_codes = ",".join(batch)
        start = (start_date or self.start_date).replace("-", "")
        end = (end_date or self.end_date).replace("-", "")
        df = api.daily(
            ts_code=ts_codes,
            start_date=start,
            end_date=end,
            fields="ts_code,trade_date,open,high,low,close,vol,amount,turnover_rate",
        )
        if df is None or len(df) == 0:
            if strict:
                raise ValueError("daily returned no rows")
            return pd.DataFrame(columns=[
                "ts_code", "trade_date", "open", "high", "low", "close",
                "vol", "amount", "turnover_rate",
            ])
        missing = set(batch) - set(df["ts_code"].astype(str).unique())
        if missing and strict:
            raise ValueError(f"daily response missing symbols: {sorted(missing)}")
        return df

    def _fetch_batch_adj(
        self,
        api,
        batch: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        """Fetch one adj_factor batch.

        ``strict`` mirrors :meth:`_fetch_batch_daily`; the incremental path
        additionally verifies a factor for every returned price row, so a
        missing symbol here can only mean "no rows in the window".
        """
        ts_codes = ",".join(batch)
        start = (start_date or self.start_date).replace("-", "")
        end = (end_date or self.end_date).replace("-", "")
        adj_df = api.adj_factor(
            ts_code=ts_codes,
            start_date=start,
            end_date=end,
        )
        if adj_df is None or len(adj_df) == 0:
            if strict:
                raise ValueError("adj_factor returned no rows")
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        missing = set(batch) - set(adj_df["ts_code"].astype(str).unique())
        if missing and strict:
            raise ValueError(f"adj_factor response missing symbols: {sorted(missing)}")
        return adj_df[["ts_code", "trade_date", "adj_factor"]]

    def _fetch_price_frames(
        self,
        api,
        codes: list[str],
        start_date: str,
        end_date: str,
        *,
        strict: bool,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame], set[str]]:
        """Fetch raw daily + adj_factor batches for one date window.

        Returns ``(daily_frames, adj_frames, missing_symbols)``. API failures
        after retries raise :class:`DataFetchIncompleteError` (the whole
        update is incomplete); with ``strict=False`` symbols absent from a
        successful response are reported instead of raising.
        """
        all_frames: list[pd.DataFrame] = []
        adj_frames: list[pd.DataFrame] = []
        failed_symbols: set[str] = set()
        missing: set[str] = set()
        batch_size = 50  # Tushare API limit

        for i in range(0, len(codes), batch_size):
            batch = list(codes[i:i + batch_size])

            try:
                df = self._call_with_retries(
                    f"daily batch {i // batch_size}",
                    self._fetch_batch_daily, api, batch,
                    start_date, end_date, strict=strict,
                )
            except DataFetchIncompleteError as exc:
                logger.error("%s", exc)
                failed_symbols.update(batch)
            else:
                all_frames.append(df)
                if df is not None:
                    missing |= set(batch) - set(
                        df["ts_code"].astype(str).unique()
                    )

            try:
                adj_frames.append(
                    self._call_with_retries(
                        f"adj_factor batch {i // batch_size}",
                        self._fetch_batch_adj, api, batch,
                        start_date, end_date, strict=strict,
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
        return all_frames, adj_frames, missing

    @staticmethod
    def _assemble_price_frames(
        all_frames: list[pd.DataFrame],
        adj_frames: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge daily + adjustment batches into a raw price panel.

        Raw OHLCV is kept untouched; the returned ``adj_factor`` column is the
        divisor ``1 / native_factor`` the pipeline divides by (D15/D18).
        Suspension fill and row dropping happen in
        :meth:`_finalize_daily_prices`, so an incremental merge can finalise
        the concatenation of cached and fresh rows exactly like a full fetch.
        """
        frames = [frame for frame in all_frames if frame is not None and len(frame)]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

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

        # VWAP on raw prices: amount (CNY) / volume (shares).  Zero-volume
        # rows yield inf; replace non-finite values with close below.
        df["vwap"] = df["amount"] / df["volume"]
        df["vwap"] = df["vwap"].where(np.isfinite(df["vwap"]), df["close"])

        adj_list = [frame for frame in adj_frames if frame is not None and len(frame)]
        if not adj_list:
            raise DataFetchIncompleteError("Tushare adj_factor fetch returned no data")
        adj = pd.concat(adj_list, ignore_index=True)
        adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")
        adj = adj.rename(columns={
            "ts_code": "asset", "trade_date": "date", "adj_factor": "native_factor",
        })
        adj["native_factor"] = pd.to_numeric(adj["native_factor"], errors="coerce")

        df = df.merge(adj, on=["date", "asset"], how="left")
        bad_factor = (
            df["native_factor"].isna()
            | ~np.isfinite(df["native_factor"])
            | (df["native_factor"] <= 0)
        )
        if bool(bad_factor.any()):
            missing = sorted(
                df.loc[bad_factor, "asset"].astype(str).unique().tolist()
            )
            raise DataFetchIncompleteError(
                "Tushare adj_factor incomplete: missing/invalid cumulative "
                f"factor for {int(bad_factor.sum())} row(s), symbols {missing[:10]}"
            )

        # Keep raw prices untouched.  The pipeline divides by this column, so
        # it must be the reciprocal of Tushare's cumulative multiplier.
        df["adj_factor"] = 1.0 / df["native_factor"]
        df = df.drop(columns=["native_factor"])

        # Set MultiIndex
        df = df.set_index(["date", "asset"]).sort_index()
        return df

    @staticmethod
    def _finalize_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Suspension fill + row dropping, identical for full and incremental.

        Applying this to the *concatenated* panel (cached rows + fresh overlap
        rows) makes an append produce the same frame a full fetch would:
        suspension rows are blanked, short gaps are forward-filled (limit 5)
        and rows still lacking a close are dropped.
        """
        if df is None or df.empty:
            return df
        suspended = df["volume"].isna() | (df["volume"] <= 0)
        df.loc[suspended, ["open", "high", "low", "close", "vwap"]] = np.nan
        df = df.groupby("asset").ffill(limit=5)
        df = df.dropna(subset=["close"])
        return canonicalize_panel_index(df)

    def _fetch_daily_prices(self, api, codes: list[str]) -> pd.DataFrame:
        """Fetch daily OHLCV as **raw** prices plus the adjustment divisor.

        Tushare reports raw prices and a cumulative multiplier
        ``native_factor`` (``hfq = raw * native_factor``).  The shared
        provider contract is raw ``close`` plus a *divisor*
        ``adj_factor = 1 / native_factor``, so the pipeline's
        ``close_adj = close / adj_factor`` reproduces the same hfq levels
        that this module used to pre-multiply into ``close`` while the stored
        price is now the tradeable CNY price (D15/D18).  The adjustment is
        applied exactly once, by the pipeline.

        Every batch is retried with exponential backoff.  If any batch (daily
        or adj_factor) still fails, the whole fetch raises
        :class:`DataFetchIncompleteError` — a partially adjusted panel (raw prices
        next to adjusted prices) must never be returned or cached.
        """
        all_frames, adj_frames, missing = self._fetch_price_frames(
            api, codes, self.start_date, self.end_date, strict=True,
        )
        if missing:
            raise DataFetchIncompleteError(
                "Tushare price fetch incomplete: "
                f"{len(missing)} symbol(s) missing from the response "
                f"(e.g. {sorted(missing)[:10]})"
            )
        df = self._assemble_price_frames(all_frames, adj_frames)
        if df.empty:
            raise DataFetchIncompleteError("Tushare price fetch returned no data")
        return self._finalize_daily_prices(df)

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

        return canonicalize_panel_index(df_full)

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
        # Use the adjusted (return-generating) series: raw prices jump on
        # ex-dividend/split dates, which would fabricate benchmark returns.
        adjusted = self._prices["close"] / self._prices["adj_factor"]
        close = adjusted.unstack("asset")
        daily_ret = close.pct_change(fill_method=None).mean(axis=1)
        daily_ret.name = "benchmark"
        return daily_ret.fillna(0)

    # ------------------------------------------------------------------
    # Incremental updates (D32)
    # ------------------------------------------------------------------

    def _incremental_cache_candidates(self) -> list[tuple]:
        """Valid cached panels for this index/start, oldest end date first.

        A candidate is trusted only when its sidecar manifest matches the
        current schema/cache/adjustment versions *and* the index/start/end
        recorded for its own file name. Legacy files without a completion
        marker are still listed here; :meth:`update` rejects them and does a
        full refetch (never an append with an unknown boundary).
        """
        prefix = f"tushare_{self.index_code}_{self.start_date}_"
        candidates: list[tuple] = []
        for path in self.cache_dir.glob(
            f"tushare_{self.index_code}_{self.start_date}_*.h5"
        ):
            name = path.name
            if not (name.startswith(prefix) and name.endswith(".h5")):
                continue
            end_str = name[len(prefix):-len(".h5")]
            try:
                end_ts = pd.Timestamp(end_str).normalize()
            except (TypeError, ValueError):
                continue
            if not self._cache_manifest_valid(path, expected_end=end_str):
                continue
            manifest = read_cache_manifest(path) or {}
            candidates.append((end_ts, end_str, path, manifest))
        candidates.sort(key=lambda item: item[0])
        return candidates

    @staticmethod
    def _delisted_symbols(metadata: pd.DataFrame | None, before) -> set[str]:
        """Symbols whose provider-stated delisting date is <= ``before``."""
        if metadata is None or "delisting_date" not in metadata.columns:
            return set()
        dates = pd.to_datetime(metadata["delisting_date"], errors="coerce")
        cutoff = pd.Timestamp(before).normalize()
        return {
            str(asset) for asset, value in dates.items()
            if pd.notna(value) and pd.Timestamp(value).normalize() <= cutoff
        }

    def _fetch_adj_map(
        self, api, codes: list[str], start_date: str, end_date: str,
    ) -> tuple[pd.Series, set[str]]:
        """Cumulative native factors for the whole cached range.

        Corporate actions rescale adjusted history and vendors revise
        factors; an append that only fetches factors for the new window can
        therefore leave the stored adjusted series inconsistent with a full
        fetch. Tushare's adj_factor endpoint is cheap, so every incremental
        update refreshes it for the full cached range (D32 requirement 3).

        Returns ``(factors, missing_symbols)``; the caller treats any missing
        symbol as an incomplete update rather than silently keeping old
        factors for part of the panel (a mixed adjustment basis).
        """
        frames: list[pd.DataFrame] = []
        failed: set[str] = set()
        missing: set[str] = set()
        batch_size = 50
        for i in range(0, len(codes), batch_size):
            batch = list(codes[i:i + batch_size])
            try:
                frame = self._call_with_retries(
                    f"adj_factor refresh batch {i // batch_size}",
                    self._fetch_batch_adj, api, batch,
                    start_date, end_date, strict=False,
                )
            except DataFetchIncompleteError as exc:
                logger.error("%s", exc)
                failed.update(batch)
                continue
            frames.append(frame)
            if frame is not None:
                missing |= set(batch) - set(
                    frame["ts_code"].astype(str).unique()
                )
        if failed:
            raise DataFetchIncompleteError(
                "Tushare adj_factor refresh incomplete: "
                f"{len(failed)} symbol(s) failed after retries "
                f"(e.g. {sorted(failed)[:10]})"
            )
        frames = [frame for frame in frames if frame is not None and len(frame)]
        if not frames:
            raise DataFetchIncompleteError("Tushare adj_factor refresh returned no data")
        adj = pd.concat(frames, ignore_index=True)
        adj["trade_date"] = pd.to_datetime(adj["trade_date"], format="%Y%m%d")
        adj = adj.rename(columns={
            "ts_code": "asset", "trade_date": "date", "adj_factor": "native_factor",
        })
        adj["native_factor"] = pd.to_numeric(adj["native_factor"], errors="coerce")
        valid = np.isfinite(adj["native_factor"]) & (adj["native_factor"] > 0)
        adj = adj.loc[valid]
        factors = 1.0 / adj["native_factor"]
        factors.index = pd.MultiIndex.from_arrays(
            [adj["date"], adj["asset"]], names=["date", "asset"],
        )
        factors = factors[~factors.index.duplicated(keep="last")].sort_index()
        factors.name = "adj_factor"
        return factors, missing

    def _status_payload(self, **overrides) -> dict:
        status = {
            "supported": True,
            "performed": False,
            "full_refetch": False,
            "new_sessions": 0,
            "requested_end": str(self.end_date),
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
        """Expose the marker through :meth:`data_status` for provenance."""
        self._data_status = {
            "last_complete_date": status.get("last_complete_date"),
            "requested_end": status.get("requested_end"),
            "updated_at": status.get("updated_at"),
            "data_age_days": status.get("data_age_days"),
            "incomplete": bool(status.get("incomplete")),
            "source": source,
        }
        return status

    def _full_refetch_status(self, reason: str, previous_marker=None) -> dict:
        """Fallback: fetch the requested range from scratch and publish it."""
        logger.warning(
            "Incremental update fell back to a full refetch: %s", reason,
        )
        # A fallback exists precisely because the cached selection may be
        # stale or unreadable; resolve the constituent universe afresh rather
        # than reusing whatever the old cache left in memory.
        self._constituent_history = None
        self._fetch_all_from_tushare()
        panel_end = panel_last_date(self._prices)
        marker, incomplete = complete_panel_marker(
            self._prices,
            delisted=self._delisted_symbols(self._metadata, panel_end),
        )
        extra = incremental_marker_extra(
            last_complete_date=marker,
            requested_end=self.end_date,
            previous_last_complete_date=previous_marker,
            new_sessions=0,
            incomplete=incomplete,
            source="full_refetch",
        )
        self._save_to_cache(self._cache_file(), extra=extra)
        status = self._status_payload(
            performed=False,
            full_refetch=True,
            last_complete_date=extra["last_complete_date"],
            previous_last_complete_date=extra["previous_last_complete_date"],
            updated_at=extra["updated_at"],
            data_age_days=extra["data_age_days"],
            incomplete=incomplete,
            reason=reason,
        )
        write_incremental_status(self._cache_file(), status)
        return self._remember_status(status, source="full_refetch")

    def _incomplete_status(
        self, reason: str, *, marker, missing: set[str] | None = None,
    ) -> dict:
        """Partial response: keep the old cache and do not advance the marker."""
        logger.warning(
            "Incremental update incomplete (%s); keeping the cached panel "
            "through %s and not advancing the completion marker",
            reason, marker,
        )
        status = self._status_payload(
            performed=False,
            last_complete_date=(
                pd.Timestamp(marker).date().isoformat()
                if marker is not None else None
            ),
            incomplete=True,
            missing_symbols=sorted(str(code) for code in (missing or set()))[:20],
            reason=reason,
        )
        write_incremental_status(self._cache_file(), status)
        return self._remember_status(status, source="cache_manifest")

    def update(self, start_date: str = None, end_date: str = None) -> dict:
        """Append new exchange sessions to the cached panel (D32).

        Steps: locate the newest compatible cache for this index and
        (start, earlier end); read its completion marker (never the last row
        alone); fetch ``marker + 1 session`` through the requested end with one
        overlap session; verify the constituent universe is unchanged (a
        change forces a full refetch); merge with ``keep="last"``; refresh
        ``adj_factor`` for the whole cached range; refetch financials,
        metadata and benchmark for the requested range; and publish
        data+manifest atomically under the provider cache lock.

        Any partial response returns ``incomplete=True`` with the old cache
        untouched -- the marker only advances after a complete publish.
        """
        if start_date:
            self.start_date = str(start_date)
        if end_date:
            self.end_date = str(end_date)
        requested_end = pd.Timestamp(self.end_date).normalize()

        candidates = self._incremental_cache_candidates()
        if not candidates:
            return self._full_refetch_status(
                "no compatible cached panel for this index/start range"
            )

        # Coverage is judged by the completion marker, never by the cache file
        # name: a cache fetched for a longer range can still have a marker
        # behind the requested end (the last day was partial or the provider
        # lagged). A longer file is a usable *base*, not proof of coverage.
        covering = [item for item in candidates if item[0] <= requested_end]
        end_ts, end_str, path, manifest = (
            covering[-1] if covering else candidates[-1]
        )
        try:
            self._load_from_cache(path, expected_end=end_str)
        except Exception as exc:
            return self._full_refetch_status(
                f"cached panel could not be loaded: {exc}"
            )

        marker_raw = manifest.get("last_complete_date")
        if not marker_raw or manifest.get("incomplete"):
            return self._full_refetch_status(
                "cached panel has no trustworthy completion marker"
            )
        marker = pd.Timestamp(marker_raw).normalize()
        panel_end = panel_last_date(self._prices)
        if panel_end is None or pd.Timestamp(panel_end).normalize() < marker:
            # The data file is older than its marker (concurrent publish or a
            # corrupt pair). Appending from this base could create a hole, so
            # refetch instead.
            return self._full_refetch_status(
                "cached panel ends before its completion marker "
                "(concurrent writer or corrupt cache)"
            )
        if requested_end <= marker:
            status = self._status_payload(
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                data_age_days=manifest.get("data_age_days"),
                reason="completion marker already covers the requested end",
            )
            return self._remember_status(status, source="cache_manifest")
        if end_ts > requested_end:
            # The loaded panel runs past the requested end but its marker does
            # not: use the rows up to the request as the append base and
            # publish a requested-end cache.
            self._prices = self._prices.loc[:pd.Timestamp(requested_end)]
            logger.info(
                "Newest cache %s extends past the requested end %s but its "
                "marker is %s; appending up to the requested end",
                end_str, self.end_date, marker.date(),
            )

        api = self.ts_api
        if api is None:
            raise RuntimeError("Tushare API not available")

        # Universe/adjustment-basis guard: an index rebalance changes the
        # panel beyond an append; refetch in full instead of mixing universes.
        cached_codes: set[str] = set()
        if self._constituent_history is not None and len(self._constituent_history):
            cached_codes = {
                str(code) for code in
                self._constituent_history["ts_code"].dropna().unique()
            }
        fresh = self._fetch_constituents(api)
        fresh_codes = {
            str(code) for code in fresh["ts_code"].dropna().unique()
        } if fresh is not None and len(fresh) else set()
        if not fresh_codes:
            return self._full_refetch_status(
                "constituent universe could not be verified"
            )
        if not cached_codes:
            return self._full_refetch_status(
                "cached panel has no constituent history to append to"
            )
        if fresh_codes != cached_codes:
            return self._full_refetch_status(
                "constituent universe changed "
                f"({len(cached_codes)} -> {len(fresh_codes)} symbols)"
            )
        self._constituent_history = fresh
        codes = sorted(fresh_codes)
        try:
            self._metadata = self._fetch_stock_basic(api, codes)
        except DataFetchIncompleteError as exc:
            return self._incomplete_status(
                f"metadata refetch failed: {exc}", marker=marker,
            )

        window_start = incremental_fetch_start(
            marker, requested_end, overlap=INCREMENTAL_OVERLAP_SESSIONS,
        )
        if window_start is None:
            status = self._status_payload(
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                reason="no exchange session between the marker and the end",
            )
            return self._remember_status(status, source="cache_manifest")

        try:
            frames, adj_frames, _missing_daily = self._fetch_price_frames(
                api, codes, str(window_start.date()), str(requested_end.date()),
                strict=False,
            )
            new_prices = self._assemble_price_frames(frames, adj_frames)
        except DataFetchIncompleteError as exc:
            return self._incomplete_status(str(exc), marker=marker)

        window_label = (
            f"{pd.Timestamp(window_start).date()}..{requested_end.date()}"
        )
        if new_prices is None or new_prices.empty:
            logger.warning(
                "Incremental fetch for %s returned no rows; keeping the "
                "cached panel (reported as incomplete/stale, not fresh)",
                window_label,
            )
            status = self._status_payload(
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                incomplete=True,
                reason="provider returned no rows for the new sessions",
            )
            write_incremental_status(self._cache_file(), status)
            return self._remember_status(status, source="cache_manifest")

        boundary_symbols = symbols_on_date(self._prices, marker)
        delisted_before_window = self._delisted_symbols(
            self._metadata, window_start,
        )
        new_dates = [
            day for day in pd.DatetimeIndex(
                new_prices.index.get_level_values("date").unique()
            ).normalize() if day > marker
        ]
        if not new_dates:
            logger.warning(
                "Incremental fetch for %s returned only overlap rows; "
                "keeping the cached panel (reported as incomplete/stale)",
                window_label,
            )
            status = self._status_payload(
                performed=True,
                last_complete_date=marker.date().isoformat(),
                updated_at=manifest.get("updated_at"),
                incomplete=True,
                reason="provider returned no rows after the completion marker",
            )
            write_incremental_status(self._cache_file(), status)
            return self._remember_status(status, source="cache_manifest")
        window_symbols = {
            str(code) for code in new_prices.index[
                new_prices.index.get_level_values("date").isin(new_dates)
            ].get_level_values("asset")
        }
        missing_window = (
            boundary_symbols - window_symbols - delisted_before_window
        )
        if missing_window:
            return self._incomplete_status(
                "provider response is missing symbols that traded on the "
                "completion marker",
                marker=marker,
                missing=missing_window,
            )

        merged = pd.concat([self._prices, new_prices]).sort_index()
        merged = deduplicate_panel(merged, "prices")

        # Refresh the divisor over the whole cached range (D32 req. 3): a
        # corporate action / factor revision must never leave old rows on a
        # different adjustment basis than the rows appended after it.
        try:
            factor_map, missing_refresh = self._fetch_adj_map(
                api, codes,
                str(merged.index.get_level_values("date").min().date()),
                str(requested_end.date()),
            )
        except DataFetchIncompleteError as exc:
            return self._incomplete_status(
                f"adjustment-factor refresh failed: {exc}", marker=marker,
            )
        priced_assets = {
            str(asset) for asset in merged.index.get_level_values("asset")
        }
        factor_assets = {
            str(asset)
            for asset in factor_map.index.get_level_values("asset")
        }
        uncovered = priced_assets - factor_assets
        if missing_refresh or uncovered:
            return self._incomplete_status(
                "adjustment-factor refresh did not cover every priced symbol "
                f"({len(missing_refresh | uncovered)} uncovered)",
                marker=marker,
                missing=missing_refresh | uncovered,
            )
        refreshed = factor_map.reindex(merged.index)
        refreshed_mask = refreshed.notna() & np.isfinite(refreshed)
        merged = merged.copy()
        merged.loc[refreshed_mask, "adj_factor"] = refreshed[refreshed_mask]
        # New rows must have received a refreshed factor; cached suspension
        # fills may legitimately have no factor row on a suspended date and
        # keep the factor carried from the last action.
        appended_rows = (
            merged.index.get_level_values("date") > marker
        ) & merged["close"].notna()
        unprotected = appended_rows & ~refreshed_mask
        if bool(unprotected.any()) or bool(
            (merged["close"].notna() & merged["adj_factor"].isna()).any()
        ):
            return self._incomplete_status(
                "adjustment factors could not be refreshed for every priced "
                "row", marker=marker,
            )

        merged = self._finalize_daily_prices(merged)

        # A truncated last day must not advance the marker. Zero
        # tolerance: without a suspension feed a missing name cannot be
        # distinguished from truncation.
        merged_dates = pd.DatetimeIndex(
            merged.index.get_level_values("date").unique()
        ).sort_values()
        previous_symbols = (
            symbols_on_date(merged, merged_dates[-2])
            if len(merged_dates) >= 2 else set()
        )
        delisted_on_last = self._delisted_symbols(
            self._metadata, panel_last_date(merged),
        )
        missing_last = last_day_missing_symbols(
            merged, delisted=delisted_on_last,
        )
        if missing_last:
            return self._incomplete_status(
                "last session of the provider response is missing symbols "
                f"present on the previous session ({len(missing_last)} of "
                f"{len(previous_symbols)})",
                marker=marker,
                missing=missing_last,
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
                "provider response is missing exchange sessions: "
                + ", ".join(str(day.date()) for day in missing_dates[:5]),
                marker=marker,
            )

        new_marker = panel_last_date(merged)
        new_sessions = sessions_after(marker, new_marker)
        old_prices = self._prices
        self._prices = merged

        # Financials/benchmark are date-range products, not per-session
        # appends: refetch them against the extended range so the incremental
        # dataset equals a full fetch byte-for-byte. A failure here keeps the
        # old cache/marker intact and must not abort the research run.
        try:
            financials = self._fetch_financials(api, codes)
            benchmark = self._fetch_index_daily(api)
        except DataFetchIncompleteError as exc:
            self._prices = old_prices
            return self._incomplete_status(
                f"financials/benchmark refetch failed: {exc}", marker=marker,
            )
        self._financials = financials
        self._benchmark = benchmark

        extra = incremental_marker_extra(
            last_complete_date=new_marker,
            requested_end=self.end_date,
            previous_last_complete_date=marker,
            new_sessions=new_sessions,
            incomplete=False,
            source="incremental",
        )
        self._save_to_cache(self._cache_file(), extra=extra)
        status = self._status_payload(
            performed=True,
            new_sessions=new_sessions,
            last_complete_date=extra["last_complete_date"],
            previous_last_complete_date=extra["previous_last_complete_date"],
            updated_at=extra["updated_at"],
            data_age_days=extra["data_age_days"],
            reason=f"appended {new_sessions} session(s) from {window_label}",
        )
        write_incremental_status(self._cache_file(), status)
        logger.info(
            "Incremental update appended %d session(s); completion marker %s",
            new_sessions, extra["last_complete_date"],
        )
        return self._remember_status(status, source="incremental")
