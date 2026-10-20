"""Adata real market data provider — free, no API key required.

Covers A-share stocks, ETFs, indices, concepts, capital flows,
dragon-tiger lists, and sentiment data. Multi-source failover.

Contract status
---------------
Adata's stock API is not shaped like the platform's ``DataProvider`` panel.
This provider therefore implements the parts it can map reliably
(``get_prices`` / ``get_benchmark`` / ``get_metadata``) and raises
:class:`NotImplementedError` for ``get_financials`` rather than returning a
wide close-only frame that looks like a valid financial panel.  Every fetch
is sliced to the requested date range and a failed symbol raises
:class:`DataFetchIncompleteError` instead of silently shrinking the universe.

Install: pip install adata
"""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataFetchIncompleteError, DataProvider
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Columns adata may return; mapped to the provider contract names.
_COLUMN_ALIASES = {
    "trade_date": "date",
    "trade_time": "date",
    "stock_code": "asset",
    "vol": "volume",
    "turnover_rate": "turnover",
}

_REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def _empty_prices(columns=None) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
    return pd.DataFrame(index=index, columns=list(columns or []))


class AdataProvider(DataProvider):
    """A-share market data via adata (multi-source, free).

    Features:
    - Stock codes, concepts, industry classification
    - Daily/weekly/monthly K-line (mapped to the (date, asset) panel)
    - Real-time quotes and 5-level order book
    - Capital flow (individual stock + concept)
    - Dragon-tiger list
    - North-bound capital flow
    - ETF data

    ``st_available`` is False: adata's stock list does not expose an ST
    history, so no ``is_st`` column is fabricated.
    """

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

    @staticmethod
    def _project_fields(df: pd.DataFrame, fields: list[str] | None) -> pd.DataFrame:
        if not fields:
            return df
        unknown = [col for col in fields if col not in df.columns]
        if unknown:
            raise ValueError(f"Unknown price fields: {unknown}")
        return df[list(fields)]

    def _fetch_one(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch one stock's K-line and normalise it to the contract."""
        kwargs = {
            "stock_code": code,
            "k_type": 1,
            "start_date": start_date,
        }
        try:
            df = self._adata.stock.market.get_market(end_date=end_date, **kwargs)
        except TypeError:
            # Older adata versions do not accept end_date; slice locally.
            df = self._adata.stock.market.get_market(**kwargs)
        if df is None or df.empty:
            raise ValueError(f"adata returned no rows for {code}")

        df = df.rename(columns=_COLUMN_ALIASES).copy()
        if "date" not in df.columns or "close" not in df.columns:
            raise ValueError(f"adata K-line for {code} has no date/close columns")
        missing = [col for col in _REQUIRED_PRICE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"adata K-line for {code} is missing contract columns: {missing}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["asset"] = code
        for col in ("open", "high", "low", "close", "volume", "amount", "turnover"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Enforce the requested end date even when the API ignores it.
        end = pd.Timestamp(end_date)
        df = df[df["date"] <= end]
        return df

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get daily OHLCV prices as a (date, asset) MultiIndex panel.

        Every symbol failure is retried once; if any symbol still fails the
        request raises instead of returning a close-only/truncated frame.
        """
        self._lazy_import()
        codes = self._list_codes()
        all_data = []
        failed: list[str] = []

        for code in codes[:200]:  # Limit universe for performance
            try:
                df = self._fetch_one(code, start_date, end_date)
                if not df.empty:
                    all_data.append(df)
            except Exception as exc:  # noqa: BLE001 - provider-specific errors
                logger.warning("adata fetch failed for %s: %s", code, exc)
                try:
                    df = self._fetch_one(code, start_date, end_date)
                    if not df.empty:
                        all_data.append(df)
                except Exception as retry_exc:  # noqa: BLE001
                    logger.error("adata retry failed for %s: %s", code, retry_exc)
                    failed.append(code)

        if failed:
            raise DataFetchIncompleteError(
                "Adata price fetch incomplete: "
                f"{len(failed)} symbol(s) failed (e.g. {failed[:10]})"
            )
        if not all_data:
            return self._project_fields(_empty_prices(_REQUIRED_PRICE_COLUMNS), fields)

        result = pd.concat(all_data, ignore_index=True)
        result = result.set_index(["date", "asset"]).sort_index()
        result.index = result.index.set_names(["date", "asset"])
        return self._project_fields(result, fields)

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

    def _list_codes(self) -> list[str]:
        metadata = self.get_metadata()
        if metadata.empty:
            return []
        return [str(code) for code in metadata.index]

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
