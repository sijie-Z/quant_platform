"""Data pipeline for ETL: clean, align, filter, and prepare data.

Transforms raw data provider output into analysis-ready DataFrames with:
- Contract validation of every provider output at the boundary
  (`validate_prices`/`validate_financials`/`validate_metadata` in
  `data/schema.py`), so provider drift fails with an actionable error that
  names the provider and the violated column/index instead of a downstream
  ``KeyError``.
- Suspension handling (forward-fill, NaN masking, derived from the data)
- Point-in-time asset-day eligibility mask (ST episodes, listing window,
  mid-sample delisting, suspension), no look-ahead
- Date alignment across prices, financials, and benchmark
- Derived fields (daily returns, adjusted prices)

Data policies fixed in this module (regression-tested):

* **Duplicate ``(date, asset)`` rows** are deduplicated deterministically
  with ``keep="last"`` *before* any ``unstack``.  A warning reports the
  count and a sample of the offending keys; the counts are exposed as
  ``duplicate_rows_dropped``.  Previously a provider overlap aborted the run
  with a raw ``ValueError: Index contains duplicate entries``.
* **``adj_factor``** must be finite and > 0 wherever the close is present.
  Invalid/missing factors are forward-filled per asset from the last valid
  factor with a warning and a count (``adj_factor_repairs``); a factor that
  cannot be repaired (no earlier valid value for that asset) raises
  :class:`~quant_platform.data.schema.DataQualityError`.  ``close_adj`` is
  therefore never ``inf``/``NaN`` because of a zero factor.
* **Unknown listing dates** (``NaT``) are kept in the universe with a
  warning and a count, not silently dropped by a ``NaT <= Timestamp``
  comparison.  A *missing* ``listing_date`` column is a contract error
  naming the provider.
* **Delisting inference** from a terminal price gap uses
  ``delisting_gap_days`` (separate from ``max_suspension_days``, default 60
  sessions) and logs every inferred date; an end-of-sample suspension shorter
  than that threshold is not reclassified as a delisting.  Provider-stated
  ``delisting_date`` metadata always wins.
* **Suspension detection** treats an unchanged close with no volume/OHLC
  evidence of a trade as stale (close-only forward-fill detection), instead
  of booking a fake zero return.
* **Symbol normalization** for board price-limit rules accepts suffix
  (``300001.SZ``), prefix (``sz300001``/``sh.688001``) and Beijing
  (``4``/``8``/``920``) codes.  A shared normalizer is still needed in
  ``core/instrument.py`` (owned by another wave); this module's local
  helper is the reference implementation.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import (
    DataContractError,
    DataQualityError,
    provider_label,
    validate_financials,
    validate_metadata,
    validate_prices,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Symbol normalization / board price-limit rules
# ---------------------------------------------------------------------------

_EXCHANGES = ("SH", "SZ", "BJ")


def _symbol_code(symbol) -> str | None:
    """Extract the six-digit A-share code from any accepted symbol format.

    Accepts ``600001.SH``, ``sh600001``, ``SH.600001``, ``sz.300001``,
    ``bj430001`` and bare ``600001``.  Returns ``None`` when no six-digit
    code can be found.
    """
    match = re.search(r"\d{6}", str(symbol))
    return match.group(0) if match else None


def _infer_exchange(code: str) -> str:
    """Infer the exchange for a bare six-digit code (best effort)."""
    if code.startswith("920") or code.startswith(("4", "8")):
        return "BJ"
    if code.startswith(("0", "2", "3")):
        return "SZ"
    if code.startswith(("6", "9")):
        return "SH"
    return "SZ"


def normalize_symbol(symbol) -> str:
    """Return a canonical ``CODE.EXCHANGE`` symbol (e.g. ``300001.SZ``).

    Handles suffix (``300001.SZ``), prefix (``sz300001``), dotted prefix
    (``sh.688001``) and Beijing codes.  Unrecognised formats are returned
    upper-cased unchanged so callers can still log them.
    """
    s = str(symbol).strip().upper()
    m = re.fullmatch(r"(\d{6})[.\-_]?(SH|SZ|BJ)", s)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.fullmatch(r"(SH|SZ|BJ)[.\-_]?(\d{6})", s)
    if m:
        return f"{m.group(2)}.{m.group(1)}"
    m = re.fullmatch(r"(\d{6})", s)
    if m:
        return f"{m.group(1)}.{_infer_exchange(m.group(1))}"
    return s


def price_limit_threshold(symbol) -> float:
    """Daily price-limit threshold for an A-share symbol.

    Board rules: ChiNext/STAR (``300``/``301``/``688``/``689``) 20%, Beijing
    (``4``/``8``/``920``) 30%, everything else 10%.  The symbol is normalized
    first, so ``sz300001`` and ``sh.688001`` get the correct board rule
    instead of falling through ``str.startswith`` on the raw label.
    Unknown formats default to 10%.
    """
    code = _symbol_code(symbol)
    if code is None:
        return 0.10
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8", "920")):
        return 0.30
    return 0.10


def realized_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Shift a forward-return panel into its realized (trailing) view.

    ``DataPipeline.returns[t]`` spans t -> t+1; at the close of t that return
    has not happened yet.  ``shift(1)[t] = returns[t-1]`` is the return
    realized from the close of t-1 to the close of t — the only return a
    signal or risk statistic computed at t may use.  Live-sim consumers
    (reversal paper trader, regime router/adapter) must use this instead of
    the raw forward panel; doing otherwise is a one-day look-ahead.
    """
    return returns.shift(1)


class DataPipeline:
    """Transforms raw provider data into clean, aligned datasets."""

    def __init__(
        self,
        provider: DataProvider,
        start_date: str,
        end_date: str,
        exclude_st: bool = True,
        exclude_suspended: bool = True,
        max_suspension_days: int = 30,
        benchmark_is_trailing: bool | None = None,
        delisting_gap_days: int | None = None,
    ):
        self.provider = provider
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.exclude_st = exclude_st
        self.exclude_suspended = exclude_suspended
        self.max_suspension_days = max_suspension_days

        # Terminal missing runs longer than this are inferred as delistings
        # when the provider supplies no ``delisting_date``.  Deliberately
        # separate from (and larger than) ``max_suspension_days``: a stock
        # suspended through the end of the sample must not be reclassified as
        # delisted.  Defaults to 60 trading sessions (~3 months), scaled up
        # for callers that raised the suspension limit.
        if delisting_gap_days is None:
            delisting_gap_days = max(60, 2 * max_suspension_days)
        if delisting_gap_days <= max_suspension_days:
            raise ValueError(
                "delisting_gap_days must be larger than max_suspension_days "
                f"(got {delisting_gap_days} <= {max_suspension_days}); the "
                "two thresholds exist to distinguish a long suspension from "
                "a delisting"
            )
        self.delisting_gap_days = int(delisting_gap_days)

        # Processed data
        self.prices: pd.DataFrame | None = None
        self.financials: pd.DataFrame | None = None
        self.benchmark: pd.Series | None = None
        self.metadata: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None

        # Assets that survive the static universe filters (listing window,
        # and — for providers without a point-in-time ST timeseries — an
        # ever-ST flag).  This is the *ever eligible* universe: it is
        # deliberately not narrowed by per-date ST windows.  Use
        # `get_valid_assets()` for the assets that are eligible on at least
        # one date, or `st_eligible` for the per-date detail.
        self.valid_assets: pd.Index | None = None

        # Point-in-time tradability: (date x asset) boolean frame, True where
        # the asset may be held on that date.  False from a stock's ST
        # announcement date to the end of that ST episode.  Only populated
        # when `exclude_st` is set.
        self.st_eligible: pd.DataFrame | None = None

        # Full asset-day eligibility (date x asset, True = may receive a new
        # signal/weight on that date): ST, listing window (no pre-listing, no
        # post-delisting) and suspension days all excluded.  Unlike
        # `st_eligible` this is always populated; it is the single mask the
        # factor stage, the volume/turnover accessors and `get_factor_prices`
        # use.  Suspension rows stay priced (stale mark) in `self.prices` so
        # valuation is unaffected -- the mask is what keeps them out of
        # cross-sections and new weights.
        self.tradable_mask: pd.DataFrame | None = None

        # Hard exclusions only (ST/listing/delisting, not suspension), used by
        # the volume/turnover/market-cap accessors so a suspension does not
        # punch a hole in the rolling history (factor output is still blanked
        # on suspended days by `tradable_mask`).
        self._hard_eligible: pd.DataFrame | None = None

        # (date x asset) suspension flags derived inside `_clean_prices` from
        # the data itself (missing/stale/zero-volume rows), not merely from an
        # optional provider column.
        self._suspended: pd.DataFrame | None = None

        # Per-asset delisting dates: provider metadata where stated, plus
        # price disappearance for providers that simply stop pricing without
        # a delisting_date.  `_inferred_delistings` records only the latter
        # (asset -> inferred first missing date) for auditing/reporting.
        self._delisting_dates: pd.Series | None = None
        self._inferred_delistings: dict = {}

        # Data-quality counters, reset per run().  `duplicate_rows_dropped`
        # counts deduplicated (date, asset) rows per dataset (keep="last");
        # `adj_factor_repairs` counts invalid factors repaired by a per-asset
        # forward-fill; `unknown_listing_assets` counts names kept with a
        # NaT listing_date.
        self.duplicate_rows_dropped: dict[str, int] = {}
        self.adj_factor_repairs: int = 0
        self.unknown_listing_assets: int = 0

        # True when `financials` carries a publish_date column, i.e. the
        # panel supports point-in-time queries via get_financials_as_of().
        self.financials_pit_available: bool = False

        # Point-in-time data (populated during run if provider supports it)
        self._st_timeseries: pd.DataFrame | None = None
        self._industry_changes: pd.DataFrame | None = None

        # Benchmark convention.  Provider benchmarks are trailing returns
        # (r_t = P_t / P_{t-1} - 1) while `self.returns` is forward
        # (r_t = P_{t+1} / P_t - 1).  `_align` shifts provider benchmarks by
        # one day when this flag is True.  A benchmark that is already in
        # forward convention -- e.g. the backtest engine's internal
        # equal-weight benchmark, which averages `returns` date by date --
        # must not be shifted again: pass `benchmark_is_trailing=False` or
        # set the provider attribute with the same name.
        self.benchmark_is_trailing: bool = True
        self._benchmark_is_trailing_override: bool | None = benchmark_is_trailing

        # Cached (date x asset) point-in-time sector panel.
        self._sector_panel: pd.DataFrame | None = None

    def run(self) -> None:
        """Execute the full ETL pipeline.

        Every provider output is contract-validated at the boundary
        (`data/schema.py`): a wrong index, missing column or empty metadata
        raises a :class:`DataContractError` that names the provider and the
        violated field instead of failing later inside an ``unstack`` or a
        factor.
        """
        logger.info("Running data pipeline: %s to %s",
                     self.start_date.date(), self.end_date.date())

        # Per-run caches/counters; a second run() must not serve the first
        # one's panel or stale quality counts.
        self._sector_panel = None
        self.duplicate_rows_dropped = {}
        self.adj_factor_repairs = 0
        self.unknown_listing_assets = 0
        self._inferred_delistings = {}
        self.financials_pit_available = False

        self.metadata = self.provider.get_metadata()
        validate_metadata(self.metadata, provider=self.provider)
        # Ensure date columns are datetime (Baostock returns strings)
        for col in ["listing_date", "delisting_date"]:
            if col in self.metadata.columns:
                self.metadata[col] = pd.to_datetime(self.metadata[col], errors="coerce")

        # Point-in-time data must be loaded *before* the universe is built:
        # the per-date ST mask is derived from it, and it needs the unfiltered
        # metadata to fall back on when the provider has no ST timeseries.
        self._load_point_in_time_data()

        self._filter_universe()

        prices = self.provider.get_prices(
            str(self.start_date.date()), str(self.end_date.date())
        )
        validate_prices(prices, provider=self.provider)
        # Duplicate (date, asset) rows are a provider overlap, not a reason to
        # abort the run with pandas' opaque unstack error.  keep="last" is the
        # documented policy: the later observation supersedes the earlier one.
        self.prices = self._deduplicate_rows(prices, "prices")

        financials = self.provider.get_financials(
            str(self.start_date.date()), str(self.end_date.date())
        )
        validate_financials(financials, provider=self.provider)
        self.financials = self._deduplicate_rows(financials, "financials")
        self.financials_pit_available = bool(
            self.financials is not None
            and not self.financials.empty
            and "publish_date" in self.financials.columns
        )

        self.benchmark = self.provider.get_benchmark(
            str(self.start_date.date()), str(self.end_date.date())
        )
        # External providers return trailing (t-day) benchmark returns; the
        # engine's internal equal-weight benchmark is built from `returns`
        # (already forward) and never passes through here.  An explicit
        # constructor argument wins; otherwise the provider may declare its
        # convention, defaulting to trailing.
        if self._benchmark_is_trailing_override is not None:
            self.benchmark_is_trailing = self._benchmark_is_trailing_override
        else:
            self.benchmark_is_trailing = bool(
                getattr(self.provider, "benchmark_is_trailing", True)
            )

        self._clean_prices()
        self._apply_pit_tradability_mask()
        self._compute_returns()
        self._align()

        logger.info("Pipeline complete: %d assets, %d dates",
                     len(self.valid_assets) if self.valid_assets is not None else 0,
                     len(self.prices.index.get_level_values("date").unique()))

    # ------------------------------------------------------------------
    # Universe filtering
    # ------------------------------------------------------------------

    def _deduplicate_rows(
        self, df: pd.DataFrame | None, dataset: str
    ) -> pd.DataFrame | None:
        """Drop duplicate ``(date, asset)`` rows with the documented policy.

        ``keep="last"`` wins: when a provider returns the same key twice, the
        later row supersedes the earlier one (the common append/overlap bug).
        The count and a sample of keys are logged at WARNING and exposed via
        ``self.duplicate_rows_dropped[dataset]``; the pipeline never lets a
        raw pandas "Index contains duplicate entries" escape from ``unstack``.
        """
        if df is None or df.empty:
            return df
        duplicated = df.index.duplicated(keep="last")
        count = int(duplicated.sum())
        self.duplicate_rows_dropped[dataset] = count
        if not count:
            return df
        sample = list(df.index[duplicated][:5])
        logger.warning(
            "%s: dropped %d duplicate (date, asset) rows (keep='last', "
            "count reset per run): %s",
            dataset, count, sample,
        )
        return df[~duplicated]

    def _filter_universe(self) -> None:
        """Filter investable universe to the static (non-time-varying) rules.

        Only listing/delisting windows are applied here, plus — for providers
        that have no point-in-time ST timeseries — a legacy ever-ST exclusion.
        ST membership is otherwise handled per date by
        `_apply_pit_tradability_mask()`, because a stock's metadata flag
        records whether it is ST *at some point*, not whether it is ST today;
        excluding on it from the start of the sample would be a look-ahead.

        Sets `self.valid_assets`, the assets eligible under these static rules.
        """
        meta = self.metadata

        # `exclude_st` is applied per date by the tradability mask.  Providers
        # without an ST timeseries can be excluded at the sample level only;
        # that is what the metadata flag means in that case, and *not*
        # applying it would leave ST names in the universe for every date.
        # The flag may be absent or empty (nullable SQL columns): absence
        # must not be read as "not ST", so say so loudly instead of silently
        # disabling the screen.
        if self.exclude_st and not self._has_usable_st_timeseries():
            if "is_st" in meta.columns:
                static_st = meta["is_st"]
                if static_st.notna().any():
                    logger.warning(
                        "Provider has no usable ST timeseries — falling back "
                        "to the static is_st flag (excludes the whole history "
                        "of every ever-ST stock)"
                    )
                    meta = meta[~static_st.fillna(False).astype(bool)]
                else:
                    logger.warning(
                        "Provider supplied an is_st column with no values and "
                        "no ST timeseries — ST screening is unavailable for "
                        "this run (missing flags are NOT treated as False)"
                    )
            else:
                logger.warning(
                    "Provider supplied neither an ST timeseries nor an is_st "
                    "column — ST screening is unavailable for this run "
                    "(absence is NOT treated as False)"
                )

        # Filter stocks not yet listed.  `NaT <= Timestamp` is False, so a
        # raw comparison silently dropped every asset with an unknown listing
        # date; keep them with a warning and a count instead — the listing
        # window simply starts at the sample beginning for those names.
        if "listing_date" not in meta.columns:
            raise DataContractError(
                f"{provider_label(self.provider)}: metadata is missing "
                "'listing_date'; cannot build the listing window"
            )
        listing = pd.to_datetime(meta["listing_date"], errors="coerce")
        unknown_listing = listing.isna()
        self.unknown_listing_assets = int(unknown_listing.sum())
        if self.unknown_listing_assets:
            logger.warning(
                "Kept %d asset(s) with unknown (NaT) listing_date; their "
                "listing window starts at the sample beginning: %s",
                self.unknown_listing_assets,
                list(meta.index[unknown_listing][:10]),
            )
        meta = meta[unknown_listing | (listing <= self.end_date)]
        meta = meta[
            meta["delisting_date"].isna()
            | (meta["delisting_date"] >= self.start_date)
        ]

        self.metadata = meta
        self.valid_assets = meta.index

    # ------------------------------------------------------------------
    # Point-in-time tradability
    # ------------------------------------------------------------------

    def _has_usable_st_timeseries(self) -> bool:
        """True when the provider gave us per-date ST information we can use.

        A provider that ships no timeseries, an empty one, or one missing the
        date columns cannot support per-date exclusion — `_filter_universe`
        must fall back to the static `is_st` flag, and
        `_build_st_eligibility_mask` must leave every asset eligible.
        """
        ts = self._st_timeseries
        if ts is None or ts.empty:
            return False
        return {"asset", "announce_date", "end_date"}.issubset(ts.columns)

    def _build_st_eligibility_mask(self) -> pd.DataFrame:
        """Build a (date x asset) boolean mask: True = holdable on that date.

        An asset is ineligible from its ST announcement date through the end
        of that ST episode.  Announcement dates — not trigger dates — are used
        so the mask only reflects what a trader could have known at the time.

        Only `_st_timeseries` is consulted, never the static `is_st` metadata
        flag: that flag is a whole-sample property and using it here would
        re-introduce the look-ahead this mask exists to remove.  Providers
        without a timeseries are handled in `_filter_universe`.

        The mask is built over the assets and dates that actually survived
        cleaning, so it always lines up with `prices` / `returns`.
        """
        assets = pd.Index(
            self.prices.index.get_level_values("asset").unique(), name="asset"
        )
        dates = pd.DatetimeIndex(
            self.prices.index.get_level_values("date").unique()
        ).sort_values()
        mask = pd.DataFrame(True, index=dates, columns=assets)

        if not self._has_usable_st_timeseries():
            # Either no timeseries at all (the static fallback in
            # `_filter_universe` already ran) or one whose dates we cannot
            # read.  Nothing is excluded per date rather than guessing.
            return mask
        ts = self._st_timeseries

        # Restrict to episodes whose ST window overlaps the price sample, so
        # the loop below stays small (episodes, not asset-days).
        announce = pd.to_datetime(ts["announce_date"])
        end = pd.to_datetime(ts["end_date"])
        overlaps = (announce <= dates.max()) & (end >= dates.min())
        episodes = ts.loc[overlaps]

        for asset, group in episodes.groupby("asset"):
            if asset not in mask.columns:
                continue
            ineligible = pd.Series(False, index=dates)
            for start, stop in zip(
                pd.to_datetime(group["announce_date"]),
                pd.to_datetime(group["end_date"]),
                strict=False,
            ):
                ineligible |= (dates >= start) & (dates <= stop)
            mask.loc[ineligible, asset] = False

        return mask

    def _build_listing_window_mask(self, dates, assets) -> pd.DataFrame:
        """True where an asset is inside its listed window on a date.

        Pre-listing and post-delisting asset-days are not tradable and must
        not enter any cross-section.  ``delisting_date`` comes from provider
        metadata; when the metadata is silent but the provider simply stops
        pricing, ``_delisting_dates`` carries the first missing date of the
        terminal gap (see `_derive_delisting_dates`).
        """
        mask = pd.DataFrame(True, index=dates, columns=assets)
        meta = self.metadata.reindex(assets)

        listing = (
            pd.to_datetime(meta["listing_date"], errors="coerce")
            if "listing_date" in meta.columns
            else pd.Series(pd.NaT, index=assets, dtype="datetime64[ns]")
        )
        delisting = pd.Series(pd.NaT, index=assets, dtype="datetime64[ns]")
        if self._delisting_dates is not None:
            delisting = pd.to_datetime(
                self._delisting_dates.reindex(assets), errors="coerce"
            )

        for asset in assets:
            listed_at = listing.get(asset)
            if pd.notna(listed_at):
                mask.loc[dates < listed_at, asset] = False
            delisted_at = delisting.get(asset)
            if pd.notna(delisted_at):
                mask.loc[dates >= delisted_at, asset] = False
        return mask

    def _derive_delisting_dates(self, close: pd.DataFrame) -> pd.Series:
        """Delisting dates from metadata, falling back to price disappearance.

        The stated ``delisting_date`` is authoritative; inference never
        overrides it.  When a provider omits the date but stops producing
        prices for good (the common shape of a real delisting), the first
        date of the trailing missing run is used — but only when that run is
        longer than ``delisting_gap_days`` (default 60 sessions), a threshold
        deliberately separate from and larger than ``max_suspension_days``.
        A stock suspended through the end of the sample is therefore not
        silently reclassified as delisted, and every inferred date is logged
        at WARNING.
        """
        dates = pd.DatetimeIndex(close.index).sort_values()
        result = pd.Series(pd.NaT, index=close.columns, dtype="datetime64[ns]")

        if self.metadata is not None and "delisting_date" in self.metadata.columns:
            stated = pd.to_datetime(
                self.metadata["delisting_date"].reindex(close.columns),
                errors="coerce",
            )
            result.loc[stated.notna()] = stated[stated.notna()]

        if len(dates) == 0:
            return result

        inferred: dict = {}
        for asset in close.columns:
            if pd.notna(result.loc[asset]):
                continue
            observed = close[asset].dropna()
            if observed.empty:
                continue
            last_seen = observed.index[-1]
            if last_seen >= dates[-1]:
                continue
            trailing_missing = dates[dates > last_seen]
            if len(trailing_missing) > self.delisting_gap_days:
                result.loc[asset] = trailing_missing[0]
                inferred[asset] = trailing_missing[0]

        if inferred:
            logger.warning(
                "Inferred delisting for %d asset(s) from terminal price gaps "
                "longer than %d sessions (provider supplied no delisting_date): %s",
                len(inferred),
                self.delisting_gap_days,
                {a: str(d.date()) for a, d in list(inferred.items())[:10]},
            )
        self._inferred_delistings = inferred
        return result

    def _derive_suspensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive (date x asset) suspension flags from the price data itself.

        Real providers do not all ship ``is_suspended``: Tushare and Baostock
        forward-fill short gaps and emit nothing.  Trusting the optional column
        left those rows in the return panel, where a forward-filled close
        produced a run of fake zero returns followed by a multi-day "resume"
        move.  A row is treated as suspended when any of the following holds
        inside the asset's listed window:

        * the close is missing (dropped rows / provider NaN);
        * volume is zero;
        * the close equals the previous close and no intraday evidence
          contradicts a fill (OHLC/VWAP unchanged or missing, volume
          unchanged or missing) — this covers both a provider that
          forward-fills the whole row and one that fills only ``close``;
        * the provider explicitly flags it.

        Leading NaNs (pre-listing) and trailing NaNs (delisting) are excluded
        by the listed window, so they are not mislabelled as suspensions.
        """
        dates = pd.DatetimeIndex(
            df.index.get_level_values("date").unique()
        ).sort_values()
        assets = pd.Index(
            df.index.get_level_values("asset").unique(), name="asset"
        )
        close = df["close"].unstack("asset").reindex(index=dates, columns=assets)

        suspended = close.isna()

        # A row is stale when the close is unchanged from the previous row
        # and nothing contradicts a forward-fill: every present OHLC(VWAP)
        # value equals the previous row's, and volume is unchanged (or
        # missing).  Any *changed* intraday value or volume proves a real
        # trade and keeps the row live.
        #
        # The old rule required every OHLC(VWAP) panel to be identical to the
        # previous row, so a provider that forward-fills only `close` (leaving
        # OHLC NaN) slipped through and booked a fake zero return followed by
        # a compounded resume jump.
        stale = close.eq(close.shift(1)) & close.notna()
        for col in ("open", "high", "low", "vwap"):
            if col in df.columns:
                panel = df[col].unstack("asset").reindex(
                    index=dates, columns=assets
                )
                previous = panel.shift(1)
                changed = panel.notna() & previous.notna() & panel.ne(previous)
                stale &= ~changed

        if "volume" in df.columns:
            volume = df["volume"].unstack("asset").reindex(
                index=dates, columns=assets
            )
            previous_volume = volume.shift(1)
            volume_changed = (
                volume.notna()
                & previous_volume.notna()
                & volume.ne(previous_volume)
            )
            stale &= ~volume_changed
        suspended |= stale

        if "volume" in df.columns:
            volume = df["volume"].unstack("asset").reindex(
                index=dates, columns=assets
            )
            suspended |= volume.eq(0) & close.notna()

        if "is_suspended" in df.columns:
            flag = (
                df["is_suspended"]
                .unstack("asset")
                .reindex(index=dates, columns=assets)
                .fillna(False)
                .astype(bool)
            )
            suspended |= flag

        window = self._build_listing_window_mask(dates, assets)
        suspended &= window
        return suspended.fillna(False)

    def _apply_pit_tradability_mask(self) -> None:
        """Blank prices outside the listed window and build the eligibility mask.

        Two different guarantees are needed:

        * **Hard exclusions** (ST while excluded, pre-listing, post-delisting)
          are blanked in ``self.prices`` itself.  Every downstream consumer
          (factors, alpha, optimizer, backtest) already treats a NaN price as
          untradable, so blanking is what makes the exclusion effective
          without plumbing a new input through every layer.
        * **Suspension** keeps its stale price in ``self.prices`` (valuation
          must show the last clean mark), but is False in ``tradable_mask``
          and therefore in the ``excluded_mask`` attached by
          `get_factor_prices`, so factors, signals and new target weights
          never see a suspended asset-day.

        ``st_eligible`` keeps its original meaning (ST episodes only) for
        reporting; ``tradable_mask`` is the combined ST/listing/delisting/
        suspension mask consumed by the factor stage.

        Delisted names: factors and signals are NaN from the delisting date,
        so the optimizer cannot open a new position.  A position still marked
        at the next rebalance has a target weight of zero (it is dropped from
        the signal cross-section) and is liquidated at that rebalance; between
        rebalances its return is NaN, which the engine's reindex/fillna(0)
        treats as a frozen mark — documented frozen-holding behaviour rather
        than a fabricated return.
        """
        prices = self.prices
        if prices is None:
            return

        dates = pd.DatetimeIndex(
            prices.index.get_level_values("date").unique()
        ).sort_values()
        assets = pd.Index(
            prices.index.get_level_values("asset").unique(), name="asset"
        )

        if self.exclude_st:
            st_mask = self._build_st_eligibility_mask().reindex(
                index=dates, columns=assets, fill_value=True
            )
        else:
            st_mask = pd.DataFrame(True, index=dates, columns=assets)

        window = self._build_listing_window_mask(dates, assets)
        hard_eligible = st_mask & window

        suspended = pd.DataFrame(False, index=dates, columns=assets)
        if self._suspended is not None and not self._suspended.empty:
            suspended = self._suspended.reindex(
                index=dates, columns=assets, fill_value=False
            )
        tradable = hard_eligible & ~suspended

        flat = hard_eligible.stack()
        flat = flat.reindex(prices.index, fill_value=True).astype(bool)

        price_cols = [
            c for c in ("open_adj", "high_adj", "low_adj", "close_adj")
            if c in prices.columns
        ]
        prices.loc[~flat, price_cols] = float("nan")

        # `valid_assets` describes the ever-eligible universe: the names that
        # survived cleaning and are not excluded on every single date.  Names
        # already dropped by `_remove_long_suspensions` are not in the mask
        # and so disappear from it here too.
        ever_eligible = hard_eligible.any(axis=0)
        dropped = hard_eligible.columns[~ever_eligible]
        if len(dropped):
            logger.info(
                "Tradability mask excludes %d assets for the whole sample: %s",
                len(dropped), list(dropped[:10]),
            )
        kept = hard_eligible.columns[ever_eligible]
        self.valid_assets = pd.Index(kept, name="asset")
        self.metadata = self.metadata.loc[
            self.metadata.index.isin(self.valid_assets)
        ]

        self.st_eligible = (
            st_mask.loc[:, ever_eligible] if self.exclude_st else None
        )
        self._hard_eligible = hard_eligible.loc[:, ever_eligible]
        self.tradable_mask = tradable.loc[:, ever_eligible]
        self.prices = prices

    def get_valid_assets(self) -> pd.Index:
        """Assets eligible on at least one date of the sample."""
        if self.st_eligible is not None:
            return self.st_eligible.columns
        return self.valid_assets if self.valid_assets is not None else pd.Index([])

    def universe_stats(self) -> dict:
        """Report how the per-date mask narrows the universe.

        `n_static_assets` counts the names that survived the cleanup (price
        cleaning included); `n_ever_st` of them are ST at some point.  Assets
        the mask excludes on every date, and assets cleaning already removed
        (long suspensions), are not counted at all — so the numbers reconcile:
        ``n_static_assets - n_ever_st`` names are eligible throughout.

        Returns zeros when no mask was built (e.g. `exclude_st=False`), so the
        caller does not have to special-case it.
        """
        if self.st_eligible is None or self.st_eligible.empty:
            return {
                "n_static_assets": 0,
                "n_ever_st": 0,
                "n_eligible_at_peak": 0,
                "n_eligible_at_trough": 0,
            }
        per_date = self.st_eligible.sum(axis=1)
        return {
            "n_static_assets": int(self.st_eligible.shape[1]),
            "n_ever_st": int((~self.st_eligible.all(axis=0)).sum()),
            "n_eligible_at_peak": int(per_date.max()),
            "n_eligible_at_trough": int(per_date.min()),
        }

    # ------------------------------------------------------------------
    # Point-in-time data loading
    # ------------------------------------------------------------------

    def _load_point_in_time_data(self) -> None:
        """Load ST timeseries and industry changes from provider if available."""
        if hasattr(self.provider, 'get_st_timeseries'):
            try:
                self._st_timeseries = self.provider.get_st_timeseries()
                logger.info("Loaded ST timeseries: %d records", len(self._st_timeseries))
            except Exception as e:
                logger.warning("Failed to load ST timeseries: %s", e)

        if hasattr(self.provider, 'get_industry_changes'):
            try:
                self._industry_changes = self.provider.get_industry_changes()
                logger.info("Loaded industry changes: %d records", len(self._industry_changes))
            except Exception as e:
                logger.warning("Failed to load industry changes: %s", e)

    def get_financials_as_of(self, as_of_date: pd.Timestamp) -> pd.DataFrame:
        """Get financial data available as of a given date (no look-ahead).

        Only includes financials whose publish_date <= as_of_date, hides
        pre-listing reports, and drops names already delisted as of the query
        date.  For each asset, returns the most recently published record.

        Point-in-time semantics require a ``publish_date`` column.  A panel
        without one has no honest PIT basis, so this accessor raises
        :class:`~quant_platform.data.schema.DataQualityError` instead of
        returning the full panel as if it were safe (the old behaviour gave
        false assurance for Tushare/Baostock/Postgres panels, which had no
        publication timestamps).

        Args:
            as_of_date: The date to filter by (inclusive).

        Returns:
            DataFrame with the latest available financials per asset.

        Raises:
            DataQualityError: when the provider supplies no ``publish_date``.
        """
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        )
        if self.financials is None or self.financials.empty:
            return empty

        if "publish_date" not in self.financials.columns:
            raise DataQualityError(
                f"{provider_label(self.provider)}: financials have no "
                "'publish_date' column, so no point-in-time view exists. "
                "Refusing to return the full panel as if it were PIT-safe; "
                "fix the provider (map reports to their first trading day "
                "on/after the announcement date) or query the raw panel "
                "explicitly."
            )

        as_of_date = pd.Timestamp(as_of_date)
        available = self.financials[
            pd.to_datetime(self.financials["publish_date"], errors="coerce")
            <= as_of_date
        ]
        if available.empty:
            return empty

        # A company has no fundamentals before it lists; reports generated
        # for pre-listing periods (or carried over from a provider) must not
        # be visible to a cross-section that predates the listing.
        if self.metadata is not None and "listing_date" in self.metadata.columns:
            listing = pd.to_datetime(self.metadata["listing_date"], errors="coerce")
            listed_by_then = listing[listing <= as_of_date].index
            available = available[
                available.index.get_level_values("asset").isin(listed_by_then)
            ]
            if available.empty:
                return empty

        # Delisted names have no tradable cross-section after their delisting
        # date (provider-stated or inferred from a terminal gap).
        if self._delisting_dates is not None:
            delisting = pd.to_datetime(self._delisting_dates, errors="coerce")
        elif self.metadata is not None and "delisting_date" in self.metadata.columns:
            delisting = pd.to_datetime(self.metadata["delisting_date"], errors="coerce")
        else:
            delisting = None
        if delisting is not None:
            delisted_by_then = delisting[
                delisting.notna() & (delisting <= as_of_date)
            ].index
            if len(delisted_by_then):
                available = available[
                    ~available.index.get_level_values("asset").isin(delisted_by_then)
                ]
                if available.empty:
                    return empty

        # For each asset, take the record with the latest publish_date
        available = available.reset_index()
        idx = available.groupby("asset")["publish_date"].idxmax()
        latest = available.loc[idx].set_index(["date", "asset"])

        return latest

    def get_st_status(self, as_of_date: pd.Timestamp) -> set:
        """Get set of assets that are ST as of a given date.

        Uses announcement dates (not trigger dates) to prevent look-ahead.
        A stock is ST from its announce_date until its end_date.

        Args:
            as_of_date: The date to check ST status for.

        Returns:
            Set of asset codes that are ST on as_of_date.
        """
        if self._st_timeseries is None or self._st_timeseries.empty:
            # Fallback to static metadata
            if self.metadata is not None and "is_st" in self.metadata.columns:
                return set(self.metadata[self.metadata["is_st"]].index)
            return set()

        st_assets = set()
        for _, row in self._st_timeseries.iterrows():
            announce = pd.Timestamp(row["announce_date"])
            end = pd.Timestamp(row["end_date"])
            if announce <= as_of_date <= end:
                st_assets.add(row["asset"])

        return st_assets

    def get_industry_map(self, as_of_date: pd.Timestamp) -> dict:
        """Get industry classification as of a given date.

        Uses effective_date to return the classification that was in effect
        on as_of_date, preventing look-ahead from future reclassifications.

        Args:
            as_of_date: The date to get classification for.

        Returns:
            Dict mapping asset -> industry string.
        """
        if self._industry_changes is None or self._industry_changes.empty:
            # Fallback to metadata
            if self.metadata is not None and "sector" in self.metadata.columns:
                return self.metadata["sector"].to_dict()
            return {}

        # Filter to changes effective on or before as_of_date
        available = self._industry_changes[
            self._industry_changes["effective_date"] <= as_of_date
        ]
        if available.empty:
            return {}

        # For each asset, take the latest effective_date
        idx = available.groupby("asset")["effective_date"].idxmax()
        latest = available.loc[idx]
        return dict(zip(latest["asset"], latest["industry"], strict=False))

    def has_industry_history(self) -> bool:
        """True when the provider supplied a non-empty industry-change table.

        Distinguishes "no history at all" -- where the static metadata map is
        the only answer ``get_industry_map`` can give and is therefore a
        legitimate fallback -- from "history exists but does not classify
        these dates", where falling back to the static (whole-sample) map
        would reintroduce the look-ahead this panel exists to remove.
        """
        return (
            self._industry_changes is not None
            and not self._industry_changes.empty
        )

    def get_sector_panel(self) -> pd.DataFrame:
        """Point-in-time sector panel (date x asset).

        Row t is exactly ``get_industry_map(date_t)``: the classification in
        effect *on* that date.  Using this panel for neutralization keeps a
        historical reclassification from leaking into earlier cross-sections
        (the static metadata map is a single whole-sample classification --
        typically the latest known one -- and is a look-ahead whenever the
        provider ships industry changes).

        Providers without an industry-change history fall back to the static
        metadata ``sector`` map repeated for every date -- the same answer
        ``get_industry_map`` gives them -- so callers see one shape either
        way.  The panel is cached per run.
        """
        if self._sector_panel is not None:
            return self._sector_panel

        if self.prices is None:
            return pd.DataFrame()

        dates = pd.DatetimeIndex(
            self.prices.index.get_level_values("date").unique()
        ).sort_values()
        assets = pd.Index(self.get_valid_assets())
        if len(assets) == 0:
            assets = pd.Index(
                self.prices.index.get_level_values("asset").unique(),
                name="asset",
            )

        has_history = self.has_industry_history()
        if has_history:
            # One get_industry_map() call per date -- hoisted out of the
            # asset loop. Calling it per (date, asset) made the panel
            # quadratic: 46,800 calls / ~98s at 390x120, ~35 minutes at the
            # default 1,304x500 sample.
            rows = []
            for date in dates:
                mapping = self.get_industry_map(date)
                rows.append([mapping.get(asset) for asset in assets])
            panel = pd.DataFrame(rows, index=dates, columns=assets)
        elif self.metadata is not None and "sector" in self.metadata.columns:
            base = self.metadata["sector"].reindex(assets).to_numpy(
                dtype=object, copy=True,
            )
            panel = pd.DataFrame(
                [base.copy() for _ in range(len(dates))],
                index=dates,
                columns=assets,
            )
        else:
            panel = pd.DataFrame(index=dates, columns=assets, dtype=object)

        panel.index.name = "date"
        panel.columns.name = "asset"
        self._sector_panel = panel
        return panel

    # ------------------------------------------------------------------
    # Price cleaning
    # ------------------------------------------------------------------

    def _clean_prices(self) -> None:
        """Clean price data: handle suspensions, adjust prices."""
        df = self.prices

        # Filter to valid assets (explicit copy to avoid SettingWithCopyWarning)
        df = df[df.index.get_level_values("asset").isin(self.valid_assets)].copy()

        # Mark long suspensions (>max_suspension_days consecutive NaN)
        if self.exclude_suspended:
            df = self._remove_long_suspensions(df)

        # Delisting dates must be known before suspension derivation: a
        # terminal NaN run is a delisting, not a suspension.
        raw_close = df["close"].unstack("asset")
        self._delisting_dates = self._derive_delisting_dates(raw_close)

        # Suspension flags from the data itself: providers that ffill without
        # an `is_suspended` column (Tushare/Baostock) must not book fake zero
        # returns across the fill.
        self._suspended = self._derive_suspensions(df)

        # adj_factor must be finite and > 0 wherever close exists; a zero
        # factor used to produce inf close_adj and corrupt returns/limit flags.
        self._repair_adj_factor(df)

        # Price adjustment
        df["close_adj"] = df["close"] / df["adj_factor"]
        df["open_adj"] = df["open"] / df["adj_factor"]
        df["high_adj"] = df["high"] / df["adj_factor"]
        df["low_adj"] = df["low"] / df["adj_factor"]

        # Handle price limits: stocks at limit ( ±10%) are untradable
        close = df["close_adj"].unstack("asset")
        change = close.pct_change(fill_method=None)
        # A resume-day jump is a multi-day move, not a limit move: do not
        # label rows whose price spans a forward-filled suspension.
        suspended = self._suspended.reindex(
            index=close.index, columns=close.columns, fill_value=False
        )
        change = change.mask(suspended | suspended.shift(1))
        limit_thresholds = pd.DataFrame(0.10, index=close.index, columns=close.columns)
        for col in close.columns:
            # Board rule from the normalized six-digit code, so prefix-style
            # symbols (`sz300001`, `sh.688001`) and Beijing codes
            # (`4`/`8`/`920`) do not fall through to the 10% default.
            threshold = price_limit_threshold(col)
            if threshold != 0.10:
                limit_thresholds[col] = threshold
        up = (change >= limit_thresholds - 1e-4) & change.notna()
        down = (change <= -limit_thresholds + 1e-4) & change.notna()
        df["is_limit_up"] = up.stack().reindex(df.index).fillna(False).astype(bool)
        df["is_limit_down"] = down.stack().reindex(df.index).fillna(False).astype(bool)

        # Materialise the derived flags on the rows that exist.  Rows the
        # provider dropped entirely cannot carry a column value; consumers
        # that need the full grid use `self._suspended`.
        df["is_suspended"] = (
            self._suspended.stack().reindex(df.index).fillna(False).astype(bool)
        )

        self.prices = df

    def _repair_adj_factor(self, df: pd.DataFrame) -> None:
        """Validate ``adj_factor`` and repair missing/invalid values.

        Policy (documented in the module docstring): an invalid or missing
        factor is forward-filled from the last valid factor of the same asset
        with a WARNING and a count (``self.adj_factor_repairs``).  A factor
        that cannot be repaired — no earlier valid value for that asset, e.g.
        a zero on the asset's first priced day — raises
        :class:`~quant_platform.data.schema.DataQualityError` naming the
        provider and the offending keys.  Rows without a close do not need a
        factor (``close_adj`` is NaN from the missing close itself), so an
        irreparable factor there is left alone.
        """
        factor = df["adj_factor"].astype(float)
        invalid = ~np.isfinite(factor) | (factor <= 0)
        if not invalid.any():
            return

        close = df["close"]
        needs_repair = invalid & close.notna()
        if not needs_repair.any():
            return

        repaired = factor.mask(invalid).groupby(level="asset").ffill()
        unrepairable = needs_repair & repaired.isna()
        if unrepairable.any():
            offending = unrepairable[unrepairable]
            sample = [
                f"{asset}@{str(date)[:10]}"
                for date, asset in offending.index[:5]
            ]
            raise DataQualityError(
                f"{provider_label(self.provider)}: adj_factor must be finite "
                f"and > 0 wherever close is present; {len(offending)} row(s) "
                "have no earlier valid factor to forward-fill from "
                f"(e.g. {sample}). Fix the provider's adjustment-factor data."
            )

        repaired_count = int((needs_repair & repaired.notna()).sum())
        if repaired_count:
            repaired_keys = needs_repair & repaired.notna()
            sample = [
                f"{asset}@{str(date)[:10]}"
                for date, asset in repaired_keys[repaired_keys].index[:5]
            ]
            self.adj_factor_repairs = repaired_count
            logger.warning(
                "adj_factor: repaired %d invalid/missing row(s) by "
                "forward-filling the last valid factor per asset (e.g. %s)",
                repaired_count, sample,
            )
        df["adj_factor"] = factor.where(~invalid, repaired)

    def _remove_long_suspensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop assets that never trade; keep history around long gaps.

        The old rule removed an asset from the *entire sample* whenever any
        consecutive NaN run exceeded the limit. A stock that suspended for 35
        days in 2024 vanished from every 2021-2023 cross-section, and a stock
        listing 40 days into the sample was dropped although it traded
        afterwards -- universe membership on early dates was decided by later
        events. Long gaps already remain NaN (the forward-fill is capped at
        `max_suspension_days`), so NaN-aware consumers skip those dates
        without losing the asset's valid history.
        """
        close = df["close"].unstack("asset")

        never_traded = [
            asset for asset in close.columns
            if close[asset].isna().all()
        ]
        if never_traded:
            df = df[
                ~df.index.get_level_values("asset").isin(never_traded)
            ]
            logger.info(
                "Dropping %d assets with no price observations", len(never_traded),
            )

        gapped = []
        for asset in close.columns:
            if asset in never_traded:
                continue
            is_nan = close[asset].isna()
            groups = (is_nan != is_nan.shift()).cumsum()
            run_lengths = is_nan.astype(int).groupby(groups).transform("sum")
            if (run_lengths > self.max_suspension_days).any():
                gapped.append(asset)
        if gapped:
            logger.info(
                "Assets with suspension gaps longer than %d days (history "
                "retained, gap stays NaN): %d, e.g. %s",
                self.max_suspension_days, len(gapped), gapped[:10],
            )
        return df

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def _compute_returns(self) -> None:
        """Compute daily returns from adjusted close prices.

        ST/listing/delisting asset-days were already blanked in the price
        panel by `_apply_pit_tradability_mask()`, so their returns are NaN and
        the backtest cannot hold them.  Because `returns[t]` spans t to t+1,
        the ST window's first day (the announcement date, when the position
        must be exited) also comes out NaN — the conservative reading, since
        the ST-day price move is not a return the strategy could have earned.

        Suspension days keep their stale mark in `prices` (valuation), so the
        return series needs the mask explicitly: a forward-filled close would
        otherwise book a fake zero on every suspended day and a compounded
        multi-day jump on the resume day.  The mask is derived from the data
        (`self._suspended`), so providers without `is_suspended` are covered.
        """
        close = self.prices["close_adj"].unstack("asset")
        returns = close.pct_change(fill_method=None).shift(-1)
        # shift(-1): return from today's close to tomorrow's close

        # A suspension leaves a stale (forward-filled) close. `returns[s-1]`
        # then becomes a fake zero and `returns[s]` the compounded multi-day
        # move -- adjacent contaminated observations that produce a spurious
        # reversal signal. Neither is a one-day return, so drop both.
        suspended = pd.DataFrame(False, index=returns.index, columns=returns.columns)
        if self._suspended is not None and not self._suspended.empty:
            suspended = self._suspended.reindex(
                index=returns.index, columns=returns.columns, fill_value=False
            )
        returns = returns.mask(suspended | suspended.shift(-1))

        self.returns = returns

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _align(self) -> None:
        """Ensure all datasets share the same date/asset index.

        The benchmark needs more than a reindex.  `returns[t]` spans t -> t+1,
        but every external provider's benchmark series is trailing:
        `benchmark[t] = P_t / P_{t-1} - 1`.  Aligning them by date alone lines
        the strategy's t -> t+1 return up against the benchmark's t-1 -> t
        return -- one day of offset.  A strategy that tracks the index exactly
        then shows a tracking error of roughly sqrt(2) times the index
        volatility instead of ~0.  Shifting the trailing series by -1 puts it
        in the same forward convention as `returns`.  Benchmarks already in
        forward convention (flagged via `benchmark_is_trailing=False` or the
        provider attribute) are left alone; the engine's internal equal-weight
        benchmark never reaches this method at all.
        """
        dates = self.prices.index.get_level_values("date").unique()
        assets = self.prices.index.get_level_values("asset").unique()

        # Align benchmark to price dates
        if self.benchmark is not None:
            benchmark = self.benchmark
            if self.benchmark_is_trailing:
                # trailing[t] = P_t/P_{t-1}-1  ->  forward[t] = P_{t+1}/P_t-1
                # shift is positional, so order the labels first.
                benchmark = benchmark.sort_index().shift(-1)
            self.benchmark = benchmark.reindex(dates)

        # Align financials.  A provider may legitimately have no fundamentals
        # (e.g. every Baostock query failed, or the source has no statements):
        # give the empty frame the documented (date, asset) MultiIndex instead
        # of calling get_level_values("asset") on a RangeIndex and killing the
        # whole run.
        if self.financials is None:
            self.financials = pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
            )
        elif self.financials.empty:
            self.financials = pd.DataFrame(
                columns=self.financials.columns,
                index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"]),
            )
        else:
            self.financials = self.financials[
                self.financials.index.get_level_values("asset").isin(assets)
            ]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_close(self) -> pd.DataFrame:
        """Return close prices (unstacked: date x asset)."""
        return self.prices["close_adj"].unstack("asset")

    def get_factor_prices(self) -> pd.DataFrame:
        """Close prices with suspension rows blanked for research inputs.

        `get_close()` keeps the stale mark, which is what valuation wants on a
        day a stock could not trade. Research inputs are different: a factor
        that differences across the fill sees a fake zero followed by a
        multi-day move. Blanking those rows makes pandas propagate NaN across
        both boundaries instead of fabricating a reversal.  Suspension is
        derived from the data (`self._suspended`), not taken on trust from an
        optional provider column.

        The full asset-day eligibility mask travels in
        ``attrs["excluded_mask"]``: factor computation forward-fills the price
        path for level continuity, and it must re-blank these rows instead of
        resurrecting ST, pre-listing, delisted or suspended names.  It is the
        same `tradable_mask` used for volume/turnover and, via main.py, for
        the raw factor panels and the final signal.
        """
        close = self.prices["close_adj"].unstack("asset")
        if self._suspended is not None and not self._suspended.empty:
            suspended = self._suspended.reindex(
                index=close.index, columns=close.columns, fill_value=False
            )
            close = close.mask(suspended)
        if self.tradable_mask is not None and not self.tradable_mask.empty:
            close.attrs["excluded_mask"] = (
                ~self.tradable_mask
            ).reindex(
                index=close.index, columns=close.columns, fill_value=True,
            )
        return close

    def get_tradable_mask(self) -> pd.DataFrame | None:
        """(date x asset) True where a new signal/weight is allowed.

        False on ST days (when `exclude_st`), before listing, from the
        delisting date onward, and on suspension days.
        """
        return self.tradable_mask

    def get_execution_gates(self) -> dict[str, pd.DataFrame]:
        """Panels the backtest needs to block untradeable execution days.

        A signal generated at the close of day *d* is filled at the close of
        *d+1*. Checking ``tradable_mask`` only on *d* let the backtest buy at
        a limit-up close, buy a suspended name, or sell at limit-down. The
        engine accepts ``tradable_mask`` / ``limit_up`` / ``limit_down`` and
        checks them on the execution date; this accessor is the wiring point.
        Panels are date x asset and empty when the pipeline did not derive
        them (the engine then treats every day as tradeable, as before).
        """
        gates: dict[str, pd.DataFrame] = {}
        mask = self.tradable_mask
        if mask is not None and not mask.empty:
            gates["tradable_mask"] = mask
        if self.prices is not None:
            for name in ("is_limit_up", "is_limit_down"):
                if name not in self.prices.columns:
                    continue
                panel = self.prices[name].unstack("asset").astype(bool)
                gates["limit_up" if name == "is_limit_up" else "limit_down"] = panel
        return gates

    def get_factor_ohlc(self) -> dict[str, pd.DataFrame]:
        """Adjusted open/high/low panels for factors that need them.

        Candle factors used to receive no OHLC at all and silently returned
        all-zero panels; they now declare ``required_inputs`` and refuse to
        run without them. Rows a factor must not see (suspended or otherwise
        ineligible) are blanked exactly like ``get_factor_prices()``.
        """
        panels: dict[str, pd.DataFrame] = {}
        if self.prices is None:
            return panels
        for field in ("open", "high", "low"):
            column = f"{field}_adj"
            if column not in self.prices.columns:
                continue
            panel = self.prices[column].unstack("asset")
            if self._suspended is not None and not self._suspended.empty:
                panel = panel.mask(
                    self._suspended.reindex(
                        index=panel.index, columns=panel.columns, fill_value=False,
                    )
                )
            panels[field] = panel
        return panels

    def _blank_ineligible_rows(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Blank hard-ineligible rows (ST, pre-listing, post-delisting).

        Suspension is deliberately *not* blanked here: a stale volume/turnover
        observation keeps the rolling history continuous, and the suspended
        asset-day itself is excluded from factor outputs and the signal by
        `tradable_mask`.
        """
        mask = self._hard_eligible
        if mask is None or mask.empty:
            return panel
        excluded = (~mask).reindex(
            index=panel.index, columns=panel.columns, fill_value=True,
        )
        return panel.mask(excluded)

    def get_volume(self) -> pd.DataFrame:
        """Return volume (unstacked: date x asset), hard-ineligible rows blanked.

        Blanked outside the listed window (no volume before listing or after
        delisting) and while ST.  Suspended rows keep the provider's stale
        observation; `tradable_mask` excludes them from signals/weights.
        """
        return self._blank_ineligible_rows(self.prices["volume"].unstack("asset"))

    def get_turnover(self) -> pd.DataFrame:
        """Return turnover rate (unstacked: date x asset), hard-ineligible rows blanked."""
        return self._blank_ineligible_rows(self.prices["turnover"].unstack("asset"))

    def get_market_cap(self) -> pd.DataFrame:
        """Return market cap from financials (unstacked: date x asset).

        Blanked before listing, after delisting and while ST: a name has no
        meaningful market cap before it lists, and a fundamental cross-section
        must not include it there.

        A provider with legitimately no fundamentals yields an all-NaN
        date x asset panel (the documented empty shape) instead of a KeyError.
        """
        if self.prices is not None and (
            self.financials is None
            or self.financials.empty
            or "market_cap" not in self.financials.columns
        ):
            dates = pd.DatetimeIndex(
                self.prices.index.get_level_values("date").unique()
            ).sort_values()
            assets = pd.Index(
                self.prices.index.get_level_values("asset").unique(), name="asset"
            )
            return pd.DataFrame(np.nan, index=dates, columns=assets)

        return self._blank_ineligible_rows(
            self.financials["market_cap"].unstack("asset")
        )
