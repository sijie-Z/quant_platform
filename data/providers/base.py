"""Abstract base class for data providers.

Defines the contract that all data providers must implement, enabling
swappable backends (synthetic, akshare, tushare, custom CSV, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Abstract interface for market data providers.

    Each provider must supply:
    - Daily price/volume data (OHLCV) as a multi-indexed DataFrame (date, asset)
    - Financial statement data as a multi-indexed DataFrame (date, asset)
    - Index/benchmark data
    - Asset metadata (sector, market cap, ST status, etc.)
    """

    @abstractmethod
    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return daily OHLCV data.

        Returns DataFrame with MultiIndex (date, asset) and columns like
        open, high, low, close, volume, turnover, adj_factor.
        """
        ...

    @abstractmethod
    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return quarterly financial statement data.

        Returns DataFrame with MultiIndex (date, asset) and columns like
        market_cap, total_assets, net_assets, revenue, net_profit, roe, pb, pe.
        Data is forward-filled from report dates.
        """
        ...

    @abstractmethod
    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Return benchmark daily returns (e.g., CSI 300).

        Returns Series indexed by date with daily return values.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> pd.DataFrame:
        """Return static asset metadata.

        Returns DataFrame indexed by asset with columns like
        sector, market_cap_group, is_st, listing_date.
        """
        ...
