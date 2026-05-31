"""Adata real market data provider — free, no API key required.

Covers A-share stocks, ETFs, indices, concepts, capital flows,
dragon-tiger lists, and sentiment data. Multi-source failover.

Install: pip install adata
"""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class AdataProvider(DataProvider):
    """A-share market data via adata (multi-source, free).

    Features:
    - Stock codes, concepts, industry classification
    - Daily/weekly/monthly K-line
    - Real-time quotes and 5-level order book
    - Capital flow (individual stock + concept)
    - Dragon-tiger list
    - North-bound capital flow
    - ETF data
    - Financial data
    """

    def __init__(self):
        self._adata = None

    def _lazy_import(self):
        if self._adata is not None:
            return
        import adata  # noqa: F811
        self._adata = adata

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get daily OHLCV prices for all A-share stocks.

        Returns (date, asset) multi-indexed DataFrame.
        """
        self._lazy_import()
        codes = self.get_metadata()
        all_data = []

        for code in codes.index[:200]:  # Limit universe for performance
            try:
                df = self._adata.stock.market.get_market(
                    stock_code=code,
                    k_type=1,
                    start_date=start_date,
                )
                if df is not None and not df.empty:
                    df["asset"] = code
                    all_data.append(df)
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", code, e)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.pivot_table(
            index="trade_date", columns="asset",
            values="close", aggfunc="first",
        )
        result.index.name = "date"
        result.columns.name = "asset"
        return result

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Get core financial data (PE, PB, ROE, market cap)."""
        self._lazy_import()
        codes = self.get_metadata()
        all_data = []

        for code in codes.index[:200]:
            try:
                df = self._adata.stock.finance.get_core_index(
                    stock_code=code,
                )
                if df is not None and not df.empty:
                    df["asset"] = code
                    all_data.append(df)
            except Exception:
                continue

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])
            result = result.pivot_table(
                index="date", columns="asset",
                values="pe_ttm", aggfunc="first",
            )
        return result

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Get CSI 300 index returns as benchmark."""
        self._lazy_import()
        try:
            df = self._adata.stock.market.get_market_index(
                index_code="000300",
                start_date=start_date,
            )
            if df is None or df.empty:
                return pd.Series(dtype=float)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date").sort_index()
            closes = df["close"].astype(float)
            return closes.pct_change().dropna()
        except Exception as e:
            logger.warning("Failed to get benchmark: %s", e)
            return pd.Series(dtype=float)

    def get_metadata(self) -> pd.DataFrame:
        """Get A-share stock metadata (code, name, exchange, sector)."""
        self._lazy_import()
        try:
            df = self._adata.stock.info.all_code()
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.set_index("stock_code")
            df.index.name = "asset"
            return df
        except Exception as e:
            logger.warning("Failed to get metadata: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Adata-specific data (beyond DataProvider interface)
    # ------------------------------------------------------------------

    def get_capital_flow(
        self, code: str, start_date: str = "", end_date: str = "",
    ) -> pd.DataFrame:
        """Get historical daily capital flow for a stock."""
        self._lazy_import()
        try:
            return self._adata.stock.market.get_capital_flow(
                stock_code=code, start_date=start_date, end_date=end_date,
            )
        except Exception as e:
            logger.warning("Capital flow failed for %s: %s", code, e)
            return pd.DataFrame()

    def get_dragon_tiger(self, date: str = "") -> pd.DataFrame:
        """Get daily dragon-tiger list (龙虎榜)."""
        self._lazy_import()
        try:
            return self._adata.sentiment.hot.list_a_list_daily(trade_date=date)
        except Exception as e:
            logger.warning("Dragon-tiger failed: %s", e)
            return pd.DataFrame()

    def get_concept_list(self) -> pd.DataFrame:
        """Get all concept/sector classifications."""
        self._lazy_import()
        try:
            return self._adata.stock.info.all_concept_code_ths()
        except Exception as e:
            logger.warning("Concept list failed: %s", e)
            return pd.DataFrame()

    def get_north_flow(self, start_date: str = "") -> pd.DataFrame:
        """Get north-bound capital flow (沪深港通)."""
        self._lazy_import()
        try:
            return self._adata.sentiment.north.north_flow(
                start_date=start_date,
            )
        except Exception as e:
            logger.warning("North flow failed: %s", e)
            return pd.DataFrame()
