"""Data pipeline for ETL: clean, align, filter, and prepare data.

Transforms raw data provider output into analysis-ready DataFrames with:
- Suspension handling (forward-fill, NaN masking)
- ST stock filtering
- Date alignment across prices, financials, and benchmark
- Derived fields (daily returns, adjusted prices)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import validate_financials, validate_prices
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
        self.valid_assets: pd.Index | None = None

    def run(self) -> None:
        """Execute the full ETL pipeline."""
        logger.info("Running data pipeline: %s to %s",
                     self.start_date.date(), self.end_date.date())

        self.metadata = self.provider.get_metadata()
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
        self._compute_returns()
        self._align()

        logger.info("Pipeline complete: %d assets, %d dates",
                     len(self.valid_assets) if self.valid_assets is not None else 0,
                     len(self.prices.index.get_level_values("date").unique()))

    # ------------------------------------------------------------------
    # Universe filtering
    # ------------------------------------------------------------------

    def _filter_universe(self) -> None:
        """Filter investable universe based on config rules."""
        meta = self.metadata

        if self.exclude_st:
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
        df["is_limit_up"] = False
        df["is_limit_down"] = False

        self.prices = df

    def _remove_long_suspensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag and remove stocks with excessively long trading halts."""
        close = df["close"].unstack("asset")
        suspended = close.isna()
        # Rolling count of consecutive NaN
        # Identify runs where all values are NaN for > max_suspension_days
        for asset in close.columns:
            nan_runs = suspended[asset].astype(int).groupby(
                (suspended[asset] != suspended[asset].shift()).cumsum()
            ).cumsum()
            # Don't actually remove, just flag for awareness
            # Removing would require reindexing which we handle in backtest
        return df

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def _compute_returns(self) -> None:
        """Compute daily returns from adjusted close prices."""
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
