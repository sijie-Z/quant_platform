"""Real-time fundamental data provider.

Fetches real-time PE/PB and market-cap data from the public East Money API.
When the API is unreachable, the provider falls back to synthetic values and
**marks them** (`source="synthetic"`, `is_synthetic=True`) so callers and the
API can tell fabricated numbers from real ones.

Includes:
- In-memory cache with configurable TTL
- Rate limiting
- Bulk fetch for portfolio-level analysis

Usage:
    fd = FundamentalDataProvider()
    metrics = fd.get_fundamentals("600519")
    # {'pe_ttm': 33.5, 'pb': 11.2, 'market_cap': 2.1e12, ...}
"""

from __future__ import annotations

import math
import threading
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FundamentalMetrics:
    """Real-time fundamental metrics for a stock.

    `source` names where the numbers came from ("eastmoney" or "synthetic");
    `is_synthetic` is the machine-readable flag. Unavailable ratios stay at
    0.0, which consumers must treat as unknown rather than as a real zero.
    """
    code: str
    name: str = ""
    pe_ttm: float = 0.0
    pb: float = 0.0
    ps_ttm: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    gross_margin: float = 0.0
    net_margin: float = 0.0
    revenue_growth: float = 0.0
    profit_growth: float = 0.0
    debt_ratio: float = 0.0
    current_ratio: float = 0.0
    dividend_yield: float = 0.0
    market_cap: float = 0.0
    circulating_cap: float = 0.0
    total_assets: float = 0.0
    net_assets: float = 0.0
    revenue: float = 0.0
    net_profit: float = 0.0
    eps: float = 0.0
    bvps: float = 0.0
    source: str = ""
    is_synthetic: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

    def to_frame(self) -> pd.DataFrame:
        """One-row frame that **keeps** provenance (``source``/``is_synthetic``).

        Dropping those columns made fabricated fallback values
        indistinguishable from live data once they left the dataclass.
        """
        return pd.DataFrame([self.to_dict()])


class FundamentalDataProvider:
    """Real-time fundamental data provider with caching and rate limiting.

    Fetches PE/PB and market cap from the public East Money API. No API key
    required. When the API fails, synthetic values are returned with
    `source="synthetic"` and `is_synthetic=True`.

    Args:
        cache_ttl: Cache time-to-live in seconds
        rate_limit: Minimum seconds between API calls per source
        source: Preferred data source (only "eastmoney" is implemented)
        strict: When True, an unavailable live API raises instead of
            returning flagged synthetic values (fail closed).
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        rate_limit: float = 0.5,
        source: str = "eastmoney",
        strict: bool = False,
    ):
        self._cache_ttl = cache_ttl
        self._rate_limit = rate_limit
        self._source = source
        self._strict = bool(strict)

        self._cache: dict[str, tuple[float, FundamentalMetrics]] = {}
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._request_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._live_count = 0
        self._synthetic_count = 0
        self._last_source = ""

    def get_fundamentals(self, code: str) -> FundamentalMetrics:
        """Get real-time fundamental metrics for a stock.

        Returns cached data if available and fresh, otherwise fetches.

        Args:
            code: Stock code (e.g., "600519")

        Returns:
            FundamentalMetrics dataclass
        """
        # Check cache
        cached = self._check_cache(code)
        if cached:
            return cached

        # Fetch from API
        self._rate_limit_wait()
        metrics = self._fetch_from_api(code)

        with self._lock:
            self._cache[code] = (time.time(), metrics)
            self._cache_misses += 1

        return metrics

    def get_bulk(self, codes: list[str]) -> dict[str, FundamentalMetrics]:
        """Get fundamentals for multiple stocks.

        Uses cache where possible, fetches in batch otherwise.

        Args:
            codes: List of stock codes

        Returns:
            Dict of code -> FundamentalMetrics
        """
        result = {}
        to_fetch = []

        # Check cache first
        for code in codes:
            cached = self._check_cache(code)
            if cached:
                result[code] = cached
            else:
                to_fetch.append(code)

        # Fetch remaining
        if to_fetch:
            for code in to_fetch:
                self._rate_limit_wait()
                metrics = self._fetch_from_api(code)
                result[code] = metrics

                with self._lock:
                    self._cache[code] = (time.time(), metrics)
                    self._cache_misses += 1

        return result

    def get_as_dataframe(self, codes: list[str]) -> pd.DataFrame:
        """Get fundamentals as a DataFrame (stocks x metrics).

        Provenance columns (``source``/``is_synthetic``, plus ``name`` and
        ``timestamp``) are preserved so callers can tell live numbers from
        synthetic fallbacks; they are not coerced to numeric.
        """
        metrics = self.get_bulk(codes)
        data = {code: m.to_dict() for code, m in metrics.items()}
        df = pd.DataFrame(data).T
        df.index.name = "code"
        provenance_cols = [
            col for col in ("name", "timestamp", "source", "is_synthetic")
            if col in df.columns
        ]
        numeric_cols = [col for col in df.columns if col not in provenance_cols]
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        return df

    def clear_cache(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> dict:
        return {
            "source": self._source,
            "last_source": self._last_source,
            "live_count": self._live_count,
            "synthetic_count": self._synthetic_count,
            "cached": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "strict": self._strict,
            "request_count": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(
                self._cache_hits / max(self._cache_hits + self._cache_misses, 1) * 100, 1
            ),
        }

    # ── Internal ──

    def _check_cache(self, code: str) -> FundamentalMetrics | None:
        with self._lock:
            if code in self._cache:
                ts, metrics = self._cache[code]
                if time.time() - ts < self._cache_ttl:
                    self._cache_hits += 1
                    return metrics
                del self._cache[code]
        return None

    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.time()

    def _fetch_from_api(self, code: str) -> FundamentalMetrics:
        """Fetch fundamental data, falling back to *flagged* synthetic values.

        The fallback used to be indistinguishable from live data: the object
        carried no provenance and `stats` still reported the configured
        source. Every caller now receives `is_synthetic=True` when the numbers
        were generated.
        """
        self._request_count += 1

        try:
            metrics = self._fetch_eastmoney(code)
            self._live_count += 1
            self._last_source = metrics.source
            return metrics
        except Exception as e:
            logger.debug("East Money API failed for %s: %s", code, e)
            if self._strict:
                # Fail closed: never fabricate ratios in strict mode.
                raise RuntimeError(
                    f"live fundamentals unavailable for {code} (strict mode): {e}"
                ) from e

        logger.info("Using synthetic fundamentals for %s (live API unavailable)", code)
        metrics = self._generate_synthetic(code)
        self._synthetic_count += 1
        self._last_source = metrics.source
        return metrics

    def _fetch_eastmoney(self, code: str) -> FundamentalMetrics:
        """Fetch from East Money's `qt/stock/get` endpoint.

        Field ids: f58 name, f162 PE(TTM), f167 PB, f116 total market cap,
        f117 circulating market cap. Prices/ratios in this endpoint are
        scaled by 100 (f162=1860 means 18.60), market caps are in yuan.

        The previous request asked for f9/f23, which belong to the unrelated
        `clist` endpoint: `stock/get` never returns them, so PE and PB were
        always 0.0 while the response looked successful.
        """
        import json
        import urllib.request

        if code.startswith("6"):
            secid = f"1.{code}"
        else:
            secid = f"0.{code}"

        url = (
            f"http://push2.eastmoney.com/api/qt/stock/get"
            f"?secid={secid}"
            f"&fields=f58,f162,f167,f116,f117"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        if "data" not in data or data["data"] is None:
            raise ValueError("No data returned")

        d = data["data"]

        def _num(key: str) -> float:
            value = d.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        pe_ttm = _num("f162") / 100.0
        pb = _num("f167") / 100.0
        if pe_ttm == 0.0 and pb == 0.0:
            raise ValueError("East Money returned no valuation fields")

        return FundamentalMetrics(
            code=code,
            name=str(d.get("f58") or ""),
            pe_ttm=pe_ttm,
            pb=pb,
            market_cap=_num("f116"),
            circulating_cap=_num("f117"),
            source="eastmoney",
        )

    def _generate_synthetic(self, code: str) -> FundamentalMetrics:
        """Generate stable synthetic fundamentals, flagged as such."""
        import numpy as np

        # `hash(code)` is salted per process, so the same stock reported
        # different PE/PB after every server restart. crc32 is stable.
        seed = zlib.crc32(code.encode("utf-8"))
        rng = np.random.default_rng(seed)

        # Realistic ranges for A-share stocks
        pe = max(1, rng.lognormal(3, 0.8))  # median ~20, range ~3-200
        pb = max(0.3, rng.lognormal(0.5, 0.6))  # median ~1.6
        roe = rng.normal(12, 8)  # mean 12%, std 8%
        market_cap = rng.lognormal(23, 1.5)  # median ~1e10

        return FundamentalMetrics(
            code=code,
            pe_ttm=round(pe, 2),
            pb=round(pb, 2),
            ps_ttm=round(max(0.5, rng.lognormal(1, 0.8)), 2),
            roe=round(roe, 2),
            roa=round(roe * rng.uniform(0.3, 0.7), 2),
            gross_margin=round(rng.uniform(15, 70), 2),
            net_margin=round(rng.uniform(3, 35), 2),
            revenue_growth=round(rng.normal(10, 20), 2),
            profit_growth=round(rng.normal(8, 25), 2),
            debt_ratio=round(rng.uniform(20, 75), 2),
            current_ratio=round(rng.uniform(0.8, 3.0), 2),
            dividend_yield=round(max(0, rng.normal(2, 1.5)), 2),
            market_cap=round(market_cap, 0),
            circulating_cap=round(market_cap * rng.uniform(0.4, 0.95), 0),
            total_assets=round(market_cap * rng.uniform(1.5, 5), 0),
            net_assets=round(market_cap / pb, 0),
            revenue=round(market_cap * rng.uniform(0.3, 1.5), 0),
            net_profit=round(market_cap * rng.uniform(0.02, 0.15), 0),
            eps=round(rng.uniform(0.5, 10), 2),
            bvps=round(rng.uniform(3, 50), 2),
            source="synthetic",
            is_synthetic=True,
        )


def _known(value: float | None) -> bool:
    """Whether a metric is real rather than the dataclass's "unavailable" 0.

    A listed company cannot have a PE, PB, ROE, market cap or debt ratio of
    exactly zero; 0.0 is what the provider leaves behind when the API did not
    supply the field. Screening used to treat those zeros as values, so
    `pe_max=20` passed every stock and `pb_min=1` rejected every stock.
    """
    if value is None:
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value != 0.0


class FundamentalScreener:
    """Screen stocks by fundamental criteria.

    Useful for filtering the stock universe by quality/value metrics.

    Usage:
        screener = FundamentalScreener(provider)
        value_stocks = screener.screen(codes, pe_max=20, roe_min=15, pb_max=3)
    """

    def __init__(self, provider: FundamentalDataProvider):
        self._provider = provider

    def screen(
        self,
        codes: list[str],
        pe_min: float | None = None,
        pe_max: float | None = None,
        pb_min: float | None = None,
        pb_max: float | None = None,
        roe_min: float | None = None,
        roe_max: float | None = None,
        market_cap_min: float | None = None,
        market_cap_max: float | None = None,
        dividend_yield_min: float | None = None,
        debt_ratio_max: float | None = None,
    ) -> list[str]:
        """Screen stocks by fundamental criteria.

        Args:
            codes: Stock codes to screen
            pe_min/max: PE ratio range
            pb_min/max: PB ratio range
            roe_min/max: ROE range (%)
            market_cap_min/max: Market cap range (yuan)
            dividend_yield_min: Minimum dividend yield (%)
            debt_ratio_max: Maximum debt ratio (%)

        Returns:
            List of codes passing all filters
        """
        metrics = self._provider.get_bulk(codes)
        passed = []

        for code, m in metrics.items():
            if pe_min is not None and not (_known(m.pe_ttm) and m.pe_ttm >= pe_min):
                continue
            if pe_max is not None and not (_known(m.pe_ttm) and m.pe_ttm <= pe_max):
                continue
            if pb_min is not None and not (_known(m.pb) and m.pb >= pb_min):
                continue
            if pb_max is not None and not (_known(m.pb) and m.pb <= pb_max):
                continue
            if roe_min is not None and not (_known(m.roe) and m.roe >= roe_min):
                continue
            if roe_max is not None and not (_known(m.roe) and m.roe <= roe_max):
                continue
            if market_cap_min is not None and not (
                _known(m.market_cap) and m.market_cap >= market_cap_min
            ):
                continue
            if market_cap_max is not None and not (
                _known(m.market_cap) and m.market_cap <= market_cap_max
            ):
                continue
            if dividend_yield_min is not None and not (
                _known(m.dividend_yield) and m.dividend_yield >= dividend_yield_min
            ):
                continue
            if debt_ratio_max is not None and not (
                _known(m.debt_ratio) and m.debt_ratio <= debt_ratio_max
            ):
                continue
            passed.append(code)

        logger.info("Screened %d/%d stocks", len(passed), len(codes))
        return passed

    def rank_by(
        self,
        codes: list[str],
        metric: str = "roe",
        ascending: bool = False,
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rank stocks by a fundamental metric.

        Args:
            codes: Stock codes to rank
            metric: Metric name (pe_ttm, pb, roe, market_cap, etc.)
            ascending: Sort order
            top_n: Return only top N

        Returns:
            List of (code, metric_value) tuples
        """
        metrics = self._provider.get_bulk(codes)
        ranked = []
        for code, m in metrics.items():
            value = getattr(m, metric, None)
            if _known(value):
                ranked.append((code, float(value)))

        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        if top_n:
            ranked = ranked[:top_n]
        return ranked
