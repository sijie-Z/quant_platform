"""Adata real market data provider — free, no API key required.

Covers A-share stocks, ETFs, indices, concepts, capital flows,
dragon-tiger lists, and sentiment data. Multi-source failover.

Contract status
---------------
Adata's stock API is not shaped like the platform's ``DataProvider`` price
panel.  Its K-line endpoint does not declare a price basis and does not
expose the raw-close + ``adj_factor`` divisor pair the pipeline divides by,
so returning its OHLCV frame would hand out levels that look like traded
CNY prices while possibly carrying adjusted (qfq/hfq) values.  This
provider therefore **fails closed for prices**: ``get_prices`` raises
:class:`NotImplementedError` with an actionable message instead of
returning that frame.  It remains usable as a secondary/validation source
for metadata and benchmark data, and for its Adata-specific research
helpers (capital flow, dragon-tiger list, concepts, north flow).

``get_financials`` likewise raises :class:`NotImplementedError` rather than
returning a wide close-only frame that looks like a valid financial panel.
Supporting fetches that take a date range are sliced to ``end_date``.

Install: pip install adata
"""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Date-column aliases adata may return; the benchmark fetch normalises them.
_COLUMN_ALIASES = {
    "trade_date": "date",
    "trade_time": "date",
}


class AdataProvider(DataProvider):
    """A-share market data via adata (multi-source, free).

    Features:
    - Stock codes, concepts, industry classification
    - Real-time quotes and 5-level order book
    - Capital flow (individual stock + concept)
    - Dragon-tiger list
    - North-bound capital flow
    - ETF data

    Prices are deliberately unsupported (see the module docstring): the
    K-line endpoint has no declared raw/adjusted basis and no divisor.
    ``get_prices`` raises :class:`NotImplementedError` so this source can
    never feed a money path or a research return series by accident.

    ``st_available`` is False: adata's stock list does not expose an ST
    history, so no ``is_st`` column is fabricated.
    """

    #: Explicit contract declaration: no price basis is offered at all.
    price_adjustment = "unsupported"
    adjustment_factor_available = False

    def __init__(self):
        self._adata = None
        self.st_available = False

    def _lazy_import(self):
        if self._adata is not None:
            return
        import adata
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
        """Not supported: fail closed instead of returning a look-alike panel.

        Adata's K-line frame carries no declared price basis and no
        ``adj_factor`` divisor column, so a caller (or the validated
        wrapper) could fuse qfq levels into a raw-marked panel.  The
        platform's contract is raw OHLCV plus a divisor; this source must
        not pretend to satisfy it.
        """
        raise NotImplementedError(
            "AdataProvider.get_prices cannot supply the platform's price "
            "contract (raw OHLCV + adj_factor divisor): adata's K-line "
            "endpoint does not declare whether its close is raw, qfq or hfq, "
            "so any panel returned here could be mistaken for traded CNY "
            "prices. Use AdataProvider only as a secondary/validation "
            "source (metadata, benchmark, research helpers); configure "
            "Tushare, Baostock or Postgres for prices and money paths."
        )

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Not supported.

        Adata's core-index endpoint does not provide the complete
        ``(date, asset)`` panel (market_cap / total_assets / net_assets /
        revenue / net_profit / roe / pb / pe / asset_growth / publish_date)
        the factors require.  Returning a PE-only pivot would look valid and
        silently drop the other factors, so this raises instead.
        """
        raise NotImplementedError(
            "AdataProvider.get_financials cannot supply the documented "
            "(date, asset) financial panel; use Tushare/Baostock/Postgres"
        )

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Get CSI 300 index trailing returns as the benchmark."""
        self._lazy_import()
        try:
            try:
                df = self._adata.stock.market.get_market_index(
                    index_code="000300",
                    start_date=start_date,
                    end_date=end_date,
                )
            except TypeError:
                df = self._adata.stock.market.get_market_index(
                    index_code="000300",
                    start_date=start_date,
                )
            if df is None or df.empty:
                return pd.Series(dtype=float, name="benchmark")
            df = df.rename(columns=_COLUMN_ALIASES)
            date_col = "date" if "date" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            df = df.loc[:pd.Timestamp(end_date)]
            closes = pd.to_numeric(df["close"], errors="coerce")
            returns = closes.pct_change().dropna()
            returns.name = "benchmark"
            return returns
        except Exception as e:
            logger.warning("Failed to get benchmark: %s", e)
            return pd.Series(dtype=float, name="benchmark")

    def get_metadata(self) -> pd.DataFrame:
        """Get A-share stock metadata mapped to the platform contract."""
        self._lazy_import()
        columns = ["sector", "market_cap_group", "listing_date", "delisting_date"]
        try:
            raw = self._adata.stock.info.all_code()
        except Exception as e:
            logger.warning("Failed to get metadata: %s", e)
            return pd.DataFrame(columns=columns, index=pd.Index([], name="asset"))
        if raw is None or raw.empty:
            return pd.DataFrame(columns=columns, index=pd.Index([], name="asset"))

        df = raw.rename(columns={"stock_code": "asset"}).copy()
        if "asset" not in df.columns:
            return pd.DataFrame(columns=columns, index=pd.Index([], name="asset"))
        df = df.set_index("asset")
        df.index = df.index.map(lambda x: str(x).split(".")[-1].zfill(6))
        df.index.name = "asset"

        sector_col = next(
            (c for c in ("sector", "industry", "industry_name") if c in df.columns),
            None,
        )
        df["sector"] = df[sector_col] if sector_col else "Unknown"
        df["market_cap_group"] = "mid"  # Placeholder; not used by the pipeline
        listing_col = next(
            (c for c in ("listing_date", "list_date", "ipo_date") if c in df.columns),
            None,
        )
        delisting_col = next(
            (c for c in ("delisting_date", "delist_date", "out_date") if c in df.columns),
            None,
        )
        df["listing_date"] = (
            pd.to_datetime(df[listing_col], errors="coerce")
            if listing_col else pd.NaT
        )
        df["delisting_date"] = (
            pd.to_datetime(df[delisting_col], errors="coerce")
            if delisting_col else pd.NaT
        )
        result = df[columns].copy()
        result.attrs["st_available"] = False
        return result

    # ------------------------------------------------------------------
    # Adata-specific data (beyond DataProvider interface)
    # ------------------------------------------------------------------

    def get_capital_flow(
        self, code: str, start_date: str = "", end_date: str = "",
    ) -> pd.DataFrame:
        """Get historical daily capital flow for a stock."""
        self._lazy_import()
        try:
            df = self._adata.stock.market.get_capital_flow(
                stock_code=code, start_date=start_date, end_date=end_date,
            )
            return self._slice_dates(df, end_date)
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

    def get_north_flow(self, start_date: str = "", end_date: str = "") -> pd.DataFrame:
        """Get north-bound capital flow (沪深港通), sliced to ``end_date``."""
        self._lazy_import()
        try:
            df = self._adata.sentiment.north.north_flow(start_date=start_date)
            return self._slice_dates(df, end_date)
        except Exception as e:
            logger.warning("North flow failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    def _slice_dates(df, end_date: str):
        """Drop rows after ``end_date`` when the source ignores the bound."""
        if df is None or getattr(df, "empty", True) or not end_date:
            return df
        date_col = next(
            (c for c in ("trade_date", "date", "trade_time") if c in df.columns),
            None,
        )
        if date_col is None:
            return df
        dates = pd.to_datetime(df[date_col], errors="coerce")
        return df.loc[dates <= pd.Timestamp(end_date)].copy()
