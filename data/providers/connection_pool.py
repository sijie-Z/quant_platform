"""Connection pool and failover router for data providers.

Provides:
- Multi-source data routing with automatic failover
- Circuit breaker pattern for failed sources
- Request deduplication and caching
- Health checking and latency tracking

Data source priority chain:
    Tushare (highest quality, needs token)
    → Baostock (free, good quality)
    → Synthetic (always available, for demo)

Usage:
    pool = DataProviderPool(config)
    prices = pool.get_prices(start_date, end_date)  # Auto-routes to best source
    health = pool.health_check()  # Check all sources
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class SourceStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CIRCUIT_OPEN = "circuit_open"
    DISABLED = "disabled"


@dataclass
class SourceHealth:
    """Health tracking for a single data source."""
    name: str
    status: SourceStatus = SourceStatus.HEALTHY
    priority: int = 0  # Lower = higher priority
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    avg_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list, repr=False)
    circuit_open_until: float = 0.0

    @property
    def available(self) -> bool:
        if self.status == SourceStatus.DISABLED:
            return False
        if self.status == SourceStatus.CIRCUIT_OPEN:
            if time.time() > self.circuit_open_until:
                self.status = SourceStatus.DEGRADED
                return True
            return False
        return True

    def record_success(self, latency_ms: float):
        self.success_count += 1
        self.consecutive_failures = 0
        self.last_success = time.time()
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
        if self.status == SourceStatus.DEGRADED:
            self.status = SourceStatus.HEALTHY

    def record_failure(self):
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure = time.time()
        # Circuit breaker: open after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.status = SourceStatus.CIRCUIT_OPEN
            self.circuit_open_until = time.time() + 60  # 60s cooldown
            logger.warning("Circuit breaker OPEN for %s (cooldown 60s)", self.name)
        elif self.consecutive_failures >= 1:
            self.status = SourceStatus.DEGRADED


class DataProviderPool:
    """Connection pool with failover routing for data providers.

    Automatically routes requests to the best available data source,
    with circuit breaker protection and health monitoring.

    Args:
        providers: dict of name -> DataProvider instance
        priorities: dict of name -> priority (lower = preferred)
    """

    def __init__(
        self,
        providers: dict[str, Any],
        priorities: dict[str, int] | None = None,
    ):
        self._providers = providers
        self._lock = threading.Lock()

        # Default priorities
        default_priorities = {
            "tushare": 1,
            "baostock": 2,
            "synthetic": 10,
        }
        prios = priorities or default_priorities

        self._health: dict[str, SourceHealth] = {}
        for name in providers:
            self._health[name] = SourceHealth(
                name=name,
                priority=prios.get(name, 5),
            )

        # Cache for deduplication
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = 60  # seconds

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Get price data from the best available source.

        Tries sources in priority order. Falls back automatically.
        """
        cache_key = f"prices_{start_date}_{end_date}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        for name, provider in self._get_ordered_providers():
            health = self._health[name]
            if not health.available:
                logger.debug("Skipping %s (status: %s)", name, health.status.value)
                continue

            try:
                start = time.perf_counter()
                result = provider.get_prices(start_date, end_date, **kwargs)
                latency = (time.perf_counter() - start) * 1000

                health.record_success(latency)
                self._set_cache(cache_key, result)
                logger.info("Data from %s (%.0fms, %d rows)", name, latency, len(result))
                return result

            except Exception as e:
                health.record_failure()
                logger.warning("Source %s failed: %s", name, e)

        logger.error("All data sources failed for prices %s to %s", start_date, end_date)
        return None

    def get_financials(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Get financial data from the best available source."""
        cache_key = f"financials_{start_date}_{end_date}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        for name, provider in self._get_ordered_providers():
            health = self._health[name]
            if not health.available:
                continue

            if not hasattr(provider, 'get_financials'):
                continue

            try:
                start = time.perf_counter()
                result = provider.get_financials(start_date, end_date, **kwargs)
                latency = (time.perf_counter() - start) * 1000

                health.record_success(latency)
                self._set_cache(cache_key, result)
                return result

            except Exception as e:
                health.record_failure()
                logger.warning("Source %s financials failed: %s", name, e)

        return None

    def health_check(self) -> dict[str, dict]:
        """Check health of all data sources."""
        result = {}
        for name, health in self._health.items():
            result[name] = {
                "status": health.status.value,
                "priority": health.priority,
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "consecutive_failures": health.consecutive_failures,
                "avg_latency_ms": round(health.avg_latency_ms, 1),
                "available": health.available,
            }
        return result

    def get_best_source(self) -> str:
        """Get the name of the current best available source."""
        for name, _ in self._get_ordered_providers():
            if self._health[name].available:
                return name
        return "none"

    def reset_circuit_breaker(self, name: str):
        """Manually reset a circuit breaker."""
        if name in self._health:
            self._health[name].status = SourceStatus.HEALTHY
            self._health[name].consecutive_failures = 0
            logger.info("Circuit breaker reset for %s", name)

    def clear_cache(self):
        """Clear the request cache."""
        self._cache.clear()

    # ── Internal ──

    def _get_ordered_providers(self):
        """Get providers sorted by priority (lowest first)."""
        items = sorted(
            self._providers.items(),
            key=lambda x: self._health[x[0]].priority,
        )
        return items

    def _check_cache(self, key: str) -> Any | None:
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (time.time(), data)
