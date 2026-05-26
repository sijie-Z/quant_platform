"""Real-time A-share market data via AKShare.

Provides live quotes, sector data, and market snapshots.
No API key required — scrapes 东方财富 public endpoints.

Usage:
    rt = RealTimeMarket()
    quotes = rt.get_quotes(['600519', '000001', '300750'])
    snapshot = rt.get_market_snapshot()  # all A-shares
    sectors = rt.get_sector_data()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Try importing akshare
try:
    import akshare as ak
    HAS_AKSHARE = True
except ImportError:
    HAS_AKSHARE = False
    logger.warning("akshare not installed. Install with: pip install akshare")


@dataclass
class Quote:
    """Single stock real-time quote."""
    code: str
    name: str
    price: float
    change_pct: float      # 涨跌幅 %
    change_amt: float      # 涨跌额
    volume: float          # 成交量 (手)
    amount: float          # 成交额 (元)
    high: float
    low: float
    open: float
    prev_close: float
    turnover_rate: float   # 换手率 %
    pe_ratio: float        # 市盈率
    pb_ratio: float        # 市净率
    market_cap: float      # 总市值
    circulating_cap: float # 流通市值
    amplitude: float       # 振幅 %
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name, "price": self.price,
            "change_pct": self.change_pct, "change_amt": self.change_amt,
            "volume": self.volume, "amount": self.amount,
            "high": self.high, "low": self.low, "open": self.open,
            "prev_close": self.prev_close, "turnover_rate": self.turnover_rate,
            "pe_ratio": self.pe_ratio, "pb_ratio": self.pb_ratio,
            "market_cap": self.market_cap, "circulating_cap": self.circulating_cap,
            "amplitude": self.amplitude, "timestamp": self.timestamp,
        }


class RealTimeMarket:
    """Real-time A-share market data provider.

    Uses AKShare (东方财富) for live data. No broker account needed.
    Data has ~15-30 second delay vs true real-time.
    """

    def __init__(self, cache_ttl: int = 10):
        """
        Args:
            cache_ttl: Cache TTL in seconds to avoid rate limiting.
        """
        if not HAS_AKSHARE:
            raise ImportError("akshare is required. Install: pip install akshare")

        self._cache_ttl = cache_ttl
        self._snapshot_cache: pd.DataFrame | None = None
        self._snapshot_time: float = 0

    def get_market_snapshot(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get real-time snapshot of ALL A-shares (~5000 stocks).

        Returns DataFrame with columns:
            代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额,
            振幅, 最高, 最低, 今开, 昨收, 量比, 换手率, 市盈率, 市净率,
            总市值, 流通市值, 60日涨跌幅, 年初至今涨跌幅
        """
        now = time.time()
        if not force_refresh and self._snapshot_cache is not None:
            if now - self._snapshot_time < self._cache_ttl:
                return self._snapshot_cache

        try:
            df = ak.stock_zh_a_spot_em()
            self._snapshot_cache = df
            self._snapshot_time = now
            logger.info("Market snapshot refreshed: %d stocks", len(df))
            return df
        except Exception as e:
            logger.error("Failed to get market snapshot: %s", e)
            if self._snapshot_cache is not None:
                logger.warning("Returning stale cache")
                return self._snapshot_cache
            raise

    def get_quotes(self, codes: list[str]) -> list[Quote]:
        """Get real-time quotes for specific stocks.

        Args:
            codes: Stock codes without exchange suffix, e.g. ['600519', '000001']
        """
        df = self.get_market_snapshot()
        code_set = set(codes)

        # Filter matching stocks
        mask = df['代码'].isin(code_set)
        matched = df[mask]

        quotes = []
        for _, row in matched.iterrows():
            try:
                q = Quote(
                    code=str(row.get('代码', '')),
                    name=str(row.get('名称', '')),
                    price=float(row.get('最新价', 0) or 0),
                    change_pct=float(row.get('涨跌幅', 0) or 0),
                    change_amt=float(row.get('涨跌额', 0) or 0),
                    volume=float(row.get('成交量', 0) or 0),
                    amount=float(row.get('成交额', 0) or 0),
                    high=float(row.get('最高', 0) or 0),
                    low=float(row.get('最低', 0) or 0),
                    open=float(row.get('今开', 0) or 0),
                    prev_close=float(row.get('昨收', 0) or 0),
                    turnover_rate=float(row.get('换手率', 0) or 0),
                    pe_ratio=float(row.get('市盈率-动态', 0) or 0),
                    pb_ratio=float(row.get('市净率', 0) or 0),
                    market_cap=float(row.get('总市值', 0) or 0),
                    circulating_cap=float(row.get('流通市值', 0) or 0),
                    amplitude=float(row.get('振幅', 0) or 0),
                )
                quotes.append(q)
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse quote for %s: %s", row.get('代码'), e)

        return quotes

    def get_top_gainers(self, n: int = 20) -> pd.DataFrame:
        """Top N gainers by change %."""
        df = self.get_market_snapshot()
        df = df.dropna(subset=['涨跌幅'])
        return df.nlargest(n, '涨跌幅')[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']]

    def get_top_losers(self, n: int = 20) -> pd.DataFrame:
        """Top N losers by change %."""
        df = self.get_market_snapshot()
        df = df.dropna(subset=['涨跌幅'])
        return df.nsmallest(n, '涨跌幅')[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']]

    def get_top_volume(self, n: int = 20) -> pd.DataFrame:
        """Top N by trading volume."""
        df = self.get_market_snapshot()
        df = df.dropna(subset=['成交额'])
        return df.nlargest(n, '成交额')[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']]

    def get_sector_data(self) -> pd.DataFrame:
        """Get real-time sector/industry board data."""
        try:
            df = ak.stock_board_industry_name_em()
            logger.info("Sector data: %d sectors", len(df))
            return df
        except Exception as e:
            logger.error("Failed to get sector data: %s", e)
            raise

    def get_sector_stocks(self, sector_name: str) -> pd.DataFrame:
        """Get stocks in a specific sector."""
        try:
            return ak.stock_board_industry_cons_em(symbol=sector_name)
        except Exception as e:
            logger.error("Failed to get sector stocks for %s: %s", sector_name, e)
            raise

    def get_historical(self, code: str, period: str = "daily",
                       start_date: str = "20240101", end_date: str = "",
                       adjust: str = "qfq") -> pd.DataFrame:
        """Get historical K-line data for a stock.

        Args:
            code: Stock code without suffix, e.g. '600519'
            period: 'daily', 'weekly', 'monthly'
            start_date: YYYYMMDD
            end_date: YYYYMMDD (empty = today)
            adjust: 'qfq' (前复权), 'hfq' (后复权), '' (不复权)
        """
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period=period,
                start_date=start_date, end_date=end_date,
                adjust=adjust,
            )
            return df
        except Exception as e:
            logger.error("Failed to get historical data for %s: %s", code, e)
            raise

    def get_market_status(self) -> dict:
        """Get current market status (open/closed/break)."""
        now = datetime.now()
        hour, minute = now.hour, now.minute
        t = hour * 100 + minute

        if now.weekday() >= 5:
            status = "closed_weekend"
        elif 915 <= t <= 930:
            status = "call_auction"
        elif 930 <= t <= 1130:
            status = "trading_am"
        elif 1130 <= t <= 1300:
            status = "break"
        elif 1300 <= t <= 1457:
            status = "trading_pm"
        elif 1457 <= t <= 1500:
            status = "closing_auction"
        else:
            status = "closed"

        return {
            "status": status,
            "is_trading": status.startswith("trading"),
            "time": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
        }
