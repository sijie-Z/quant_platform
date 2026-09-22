"""Data pipeline for ETL: clean, align, filter, and prepare data.

Transforms raw data provider output into analysis-ready DataFrames with:
- Suspension handling (forward-fill, NaN masking)
- Point-in-time ST exclusion (per-date tradability mask, no look-ahead)
- Date alignment across prices, financials, and benchmark
- Derived fields (daily returns, adjusted prices)
"""

from __future__ import annotations

import pandas as pd
from quant_platform.data.providers.base import DataProvider
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


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
    ):
        self.provider = provider
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.exclude_st = exclude_st
        self.exclude_suspended = exclude_suspended
        self.max_suspension_days = max_suspension_days

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

        # Point-in-time data (populated during run if provider supports it)
        self._st_timeseries: pd.DataFrame | None = None
        self._industry_changes: pd.DataFrame | None = None

    def run(self) -> None:
        """Execute the full ETL pipeline."""
        logger.info("Running data pipeline: %s to %s",
                     self.start_date.date(), self.end_date.date())

        self.metadata = self.provider.get_metadata()
        # Ensure date columns are datetime (Baostock returns strings)
        for col in ["listing_date", "delisting_date"]:
            if col in self.metadata.columns:
                self.metadata[col] = pd.to_datetime(self.metadata[col], errors="coerce")

        # Point-in-time data must be loaded *before* the universe is built:
        # the per-date ST mask is derived from it, and it needs the unfiltered
        # metadata to fall back on when the provider has no ST timeseries.
        self._load_point_in_time_data()

        self._filter_universe()

        self.prices = self.provider.get_prices(
            str(self.start_date.date()), str(self.end_date.date())
        )
        self.financials = self.provider.get_financials(
            str(self.start_date.date()), str(self.end_date.date())
        )
        self.benchmark = self.provider.get_benchmark(
            str(self.start_date.date()), str(self.end_date.date())
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
        if self.exclude_st and not self._has_usable_st_timeseries() and "is_st" in meta.columns:
            logger.warning(
                "Provider has no usable ST timeseries — falling back to the "
                "static is_st flag (excludes the whole history of every "
                "ever-ST stock)"
            )
            meta = meta[~meta["is_st"]]

        # Filter stocks not yet listed
        meta = meta[meta["listing_date"] <= self.end_date]
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

    def _apply_pit_tradability_mask(self) -> None:
        """Blank prices and returns for assets while they are ST.

        Ineligible asset-days become NaN — the same representation the
        pipeline already uses for suspensions, so every downstream consumer
        (factors, alpha, optimizer, backtest) treats them as untradable
        without needing to know the mask exists.

        Note this is deliberately *not* plumbed into the backtest as a
        separate input: the backtest reads `self.returns` and `self.prices`,
        and blanking them here is what makes the mask effective.  A consumer
        that needs the mask itself (e.g. for reporting) can read
        `self.st_eligible`.
        """
        prices = self.prices
        if prices is None:
            return
        if not self.exclude_st:
            return

        mask = self._build_st_eligibility_mask()
        flat = mask.stack()
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
        ever_eligible = mask.any(axis=0)
        dropped = mask.columns[~ever_eligible]
        if len(dropped):
            logger.info(
                "ST mask excludes %d assets for the whole sample: %s",
                len(dropped), list(dropped[:10]),
            )
        kept = mask.columns[ever_eligible]
        self.valid_assets = pd.Index(kept, name="asset")
        self.metadata = self.metadata.loc[
            self.metadata.index.isin(self.valid_assets)
        ]

        self.st_eligible = mask.loc[:, ever_eligible]
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

        Only includes financials whose publish_date <= as_of_date.
        For each asset, returns the most recently published record.

        Args:
            as_of_date: The date to filter by (inclusive).

        Returns:
            DataFrame with latest available financials per asset.
        """
        if self.financials is None:
            return pd.DataFrame()

        if "publish_date" not in self.financials.columns:
            # No publish_date column — use all data (legacy provider)
            logger.warning("Financials missing publish_date — using all data (no PIT filter)")
            return self.financials

        # Filter: only records whose publish_date <= as_of_date
        available = self.financials[self.financials["publish_date"] <= as_of_date]
        if available.empty:
            return pd.DataFrame()

        # For each asset, take the record with the latest publish_date
        # Reset index to work with date/asset columns
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

        # Price adjustment
        df["close_adj"] = df["close"] / df["adj_factor"]
        df["open_adj"] = df["open"] / df["adj_factor"]
        df["high_adj"] = df["high"] / df["adj_factor"]
        df["low_adj"] = df["low"] / df["adj_factor"]

        # Handle price limits: stocks at limit ( ±10%) are untradable
        close = df["close_adj"].unstack("asset")
        change = close.pct_change(fill_method=None)
        limit_thresholds = pd.DataFrame(0.10, index=close.index, columns=close.columns)
        for col in close.columns:
            code = str(col)
            if code.startswith(("300", "301", "688", "689")):
                limit_thresholds[col] = 0.20
            elif code.startswith(("8", "4")):
                limit_thresholds[col] = 0.30
        up = (change >= limit_thresholds - 1e-4) & change.notna()
        down = (change <= -limit_thresholds + 1e-4) & change.notna()
        df["is_limit_up"] = up.stack().reindex(df.index).fillna(False).astype(bool)
        df["is_limit_down"] = down.stack().reindex(df.index).fillna(False).astype(bool)

        self.prices = df

    def _remove_long_suspensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop assets with a consecutive suspension run longer than the limit."""
        close = df["close"].unstack("asset")
        drop_assets = []
        for asset in close.columns:
            is_nan = close[asset].isna()
            groups = (is_nan != is_nan.shift()).cumsum()
            run_lengths = is_nan.astype(int).groupby(groups).transform("sum")
            if (run_lengths > self.max_suspension_days).any():
                drop_assets.append(asset)
        if drop_assets:
            logger.info("Dropping %d assets with long suspensions: %s",
                        len(drop_assets), drop_assets[:10])
            df = df[~df.index.get_level_values("asset").isin(drop_assets)]
        return df

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def _compute_returns(self) -> None:
        """Compute daily returns from adjusted close prices.

        ST asset-days were already blanked by `_apply_pit_tradability_mask()`,
        so their returns are NaN and the backtest cannot hold them.  Because
        `returns[t]` spans t to t+1, the ST window's first day (the
        announcement date, when the position must be exited) also comes out
        NaN — the conservative reading, since the ST-day price move is not a
        return the strategy could have earned.
        """
        close = self.prices["close_adj"].unstack("asset")
        self.returns = close.pct_change(fill_method=None).shift(-1)
        # shift(-1): return from today's close to tomorrow's close

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _align(self) -> None:
        """Ensure all datasets share the same date/asset index."""
        dates = self.prices.index.get_level_values("date").unique()
        assets = self.prices.index.get_level_values("asset").unique()

        # Align benchmark to price dates
        self.benchmark = self.benchmark.reindex(dates)

        # Align financials
        self.financials = self.financials[
            self.financials.index.get_level_values("asset").isin(assets)
        ]

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_close(self) -> pd.DataFrame:
        """Return close prices (unstacked: date x asset)."""
        return self.prices["close_adj"].unstack("asset")

    def get_volume(self) -> pd.DataFrame:
        """Return volume (unstacked: date x asset)."""
        return self.prices["volume"].unstack("asset")

    def get_turnover(self) -> pd.DataFrame:
        """Return turnover rate (unstacked: date x asset)."""
        return self.prices["turnover"].unstack("asset")

    def get_market_cap(self) -> pd.DataFrame:
        """Return market cap from financials (unstacked: date x asset)."""
        return self.financials["market_cap"].unstack("asset")
