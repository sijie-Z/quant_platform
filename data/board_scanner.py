"""A-share market board scanner — real-time screening using East Money API.

Scans the entire A-share market in real-time for:
- Limit-up stocks (涨停板)
- Strong stocks (强势股 >7%)
- Consecutive limit-up streaks (连板)
- Market sentiment indicators

Data source: 东方财富 push2 API (free, no API key required)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# East Money field codes
FIELDS = {
    "code": "f12",
    "name": "f14",
    "price": "f2",
    "change_pct": "f3",
    "change": "f4",
    "volume": "f5",
    "amount": "f6",
    "turnover_rate": "f8",
    "volume_ratio": "f10",
    "high": "f15",
    "low": "f16",
    "open": "f17",
    "total_mv": "f20",
    "circ_mv": "f21",
    "rise_speed": "f22",
    "pe": "f9",
    "continuous_board": "f66",
    "board_time": "f75",
    "open_count": "f107",
    "last_board": "f168",
}

MARKET_CODES = {
    "a_share": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
    "sh": "m:1+t:2,m:1+t:23",
    "sz": "m:0+t:6,m:0+t:80,m:0+t:81+s:2048",
    "gem": "m:0+t:80",      # 创业板
    "star": "m:1+t:23",     # 科创板
    "be": "m:0+t:81+s:2048", # 北交所
}

EASTMONEY_API = "https://push2.eastmoney.com/api/qt/clist/get"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://data.eastmoney.com/",
}


@dataclass
class BoardStock:
    """A single stock in a board scan result."""
    code: str
    name: str
    price: float = 0.0
    change_pct: float = 0.0
    turnover_rate: float = 0.0
    volume_ratio: float = 0.0
    amount: float = 0.0
    total_mv: float = 0.0
    circ_mv: float = 0.0
    continuous_board: int = 0
    board_time: str = ""
    open_count: int = 0
    rise_speed: float = 0.0
    pe: float = 0.0


@dataclass
class MarketSentiment:
    """Aggregated market sentiment from board scan."""
    limit_up_count: int = 0
    strong_count: int = 0
    limit_up_break_count: int = 0
    total_up: int = 0
    total_down: int = 0


class BoardScanner:
    """Real-time A-share board scanner.

    Usage:
        scanner = BoardScanner()
        limit_ups = scanner.scan_limit_up()
        strong = scanner.scan_strong()
        sentiment = scanner.analyze_market()
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    def scan_limit_up(
        self,
        market: str = "a_share",
        limit: int = 100,
        min_change: float = 9.5,
    ) -> list[BoardStock]:
        """Scan for limit-up stocks in real-time.

        Args:
            market: Market segment (a_share/sh/sz/gem/star/be).
            limit: Max results.
            min_change: Minimum change % to count as limit-up.

        Returns:
            List of BoardStock objects sorted by time of hitting limit-up.
        """
        data = self._fetch(market, limit, "f75")  # Sort by board time
        results = []
        for item in data:
            change_pct = float(item.get(FIELDS["change_pct"], 0)) / 100
            if change_pct >= min_change:
                results.append(self._parse_item(item))
        return results

    def scan_strong(
        self,
        market: str = "a_share",
        limit: int = 100,
        min_change: float = 7.0,
        max_change: float = 9.5,
    ) -> list[BoardStock]:
        """Scan for strong stocks (>7%, not yet limit-up)."""
        data = self._fetch(market, limit, "f3")
        results = []
        for item in data:
            change_pct = float(item.get(FIELDS["change_pct"], 0)) / 100
            if min_change <= change_pct < max_change:
                results.append(self._parse_item(item))
        return results

    def scan_continuous(
        self,
        market: str = "a_share",
        limit: int = 50,
    ) -> list[BoardStock]:
        """Scan for stocks with consecutive limit-ups (连板).

        Returns stocks with continuous_board >= 2, sorted by streak.
        """
        limit_ups = self.scan_limit_up(market, limit)
        return [s for s in limit_ups if s.continuous_board >= 2]

    def analyze_market(self) -> MarketSentiment:
        """Get aggregated market sentiment indicators."""
        limit_ups = self.scan_limit_up(limit=500)
        strong = self.scan_strong(limit=500)
        all_data = self._fetch("a_share", 5000, "f3")

        up_count = 0
        down_count = 0
        for item in all_data:
            change = float(item.get(FIELDS["change_pct"], 0))
            if change > 0:
                up_count += 1
            elif change < 0:
                down_count += 1

        return MarketSentiment(
            limit_up_count=len(limit_ups),
            strong_count=len(strong),
            limit_up_break_count=sum(
                1 for s in limit_ups if s.open_count > 0
            ),
            total_up=up_count,
            total_down=down_count,
        )

    def scan_top_gainers(
        self,
        market: str = "a_share",
        limit: int = 20,
    ) -> list[BoardStock]:
        """Get top gainers sorted by change %."""
        data = self._fetch(market, limit, "f3")
        return [self._parse_item(item) for item in data]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(
        self, market: str, limit: int, sort_field: str
    ) -> list[dict[str, Any]]:
        """Fetch raw data from East Money API."""
        market_code = MARKET_CODES.get(market, MARKET_CODES["a_share"])
        params = {
            "pn": 1,
            "pz": limit,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": sort_field,
            "fs": market_code,
            "fields": ",".join(FIELDS.values()),
        }
        try:
            resp = self._session.get(EASTMONEY_API, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", {}).get("diff", [])
        except Exception as e:
            logger.warning("East Money API call failed: %s", e)
        return []

    def _parse_item(self, item: dict[str, Any]) -> BoardStock:
        """Parse a raw API item into BoardStock."""
        return BoardStock(
            code=str(item.get(FIELDS["code"], "")),
            name=str(item.get(FIELDS["name"], "")),
            price=float(item.get(FIELDS["price"], 0)) / 100,
            change_pct=float(item.get(FIELDS["change_pct"], 0)) / 100,
            turnover_rate=float(item.get(FIELDS["turnover_rate"], 0)),
            volume_ratio=round(float(item.get(FIELDS["volume_ratio"], 0)), 2),
            amount=float(item.get(FIELDS["amount"], 0)),
            total_mv=float(item.get(FIELDS["total_mv"], 0)) / 1e8,
            circ_mv=float(item.get(FIELDS["circ_mv"], 0)) / 1e8,
            continuous_board=int(item.get(FIELDS["continuous_board"], 0)),
            board_time=str(item.get(FIELDS["board_time"], "")),
            open_count=int(item.get(FIELDS["open_count"], 0)),
            rise_speed=float(item.get(FIELDS["rise_speed"], 0)) / 100,
            pe=float(item.get(FIELDS["pe"], 0)),
        )
