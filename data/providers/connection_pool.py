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

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Bump when the cached payload's schema/semantics change; it is part of every
# cache key so a unit/contract fix cannot be masked by a stale in-process hit.
CACHE_SCHEMA_VERSION = "2"


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
    last_error: str = ""

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

    def record_failure(self, reason: str = ""):
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure = time.time()
        self.last_error = str(reason)
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
        min_rows: int = 1,
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
        # A response with fewer rows than this is treated as empty.  One
        # empty frame must not disable failover to a healthy secondary source.
        self._min_rows = max(0, int(min_rows))

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Get price data from the best available source.

        Tries sources in priority order. Falls back automatically.
        """
        cache_key = self._make_cache_key("prices", start_date, end_date, kwargs)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return self._copy_result(cached)

        for name, provider in self._get_ordered_providers():
            health = self._health[name]
            if not health.available:
                logger.debug("Skipping %s (status: %s)", name, health.status.value)
                continue

            try:
                start = time.perf_counter()
                result = provider.get_prices(start_date, end_date, **kwargs)
                latency = (time.perf_counter() - start) * 1000

                if result is None:
                    raise ValueError("provider returned None")
                if self._is_unusable_result(result) and self._another_source_available(name):
                    raise ValueError(
                        f"provider returned an empty/short frame "
                        f"(<{self._min_rows} row(s)); failing over"
                    )
                health.record_success(latency)
                # Store a copy: the caller receives `result` and may mutate
                # it, which used to corrupt the cache for the next request.
                self._set_cache(cache_key, self._copy_result(result))
                logger.info("Data from %s (%.0fms, %d rows)", name, latency, len(result))
                return result

            except Exception as e:
                health.record_failure(str(e))
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
        cache_key = self._make_cache_key("financials", start_date, end_date, kwargs)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return self._copy_result(cached)

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

                if result is None:
                    raise ValueError("provider returned None")
                if self._is_unusable_result(result) and self._another_source_available(name):
                    raise ValueError(
                        f"provider returned an empty/short financial frame "
                        f"(<{self._min_rows} row(s)); failing over"
                    )
                health.record_success(latency)
                self._set_cache(cache_key, self._copy_result(result))
                return result

            except Exception as e:
                health.record_failure(str(e))
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
                "last_error": health.last_error,
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

    def _is_unusable_result(self, value: Any) -> bool:
        """True for an empty/undersized frame that should trigger failover."""
        try:
            return len(value) < self._min_rows
        except TypeError:
            return False

    def _another_source_available(self, name: str) -> bool:
        """Whether a different, currently usable source exists for failover."""
        return any(
            other != name and health.available
            for other, health in self._health.items()
        )

    def _make_cache_key(
        self, prefix: str, start_date: str, end_date: str, kwargs: dict,
    ) -> str:
        """Cache key includes the provider identities and a schema version.

        The same request routed through a different provider set (or after a
        payload-contract bump) must not be served another configuration's
        result.  ``kwargs`` are included so a field projection is not served
        the cache of a different one.
        """
        identities = "|".join(
            f"{name}={type(provider).__module__}.{type(provider).__name__}"
            for name, provider in sorted(self._providers.items())
        )
        try:
            suffix = json.dumps(kwargs, sort_keys=True, default=str)
        except TypeError:
            suffix = repr(sorted(kwargs.items(), key=lambda item: item[0]))
        return (
            f"{prefix}_{CACHE_SCHEMA_VERSION}_{identities}_"
            f"{start_date}_{end_date}_{suffix}"
        )

    @staticmethod
    def _copy_result(value: Any) -> Any:
        """Return a copy so a caller cannot mutate the shared cache."""
        if isinstance(value, pd.DataFrame):
            return value.copy()
        return copy.deepcopy(value)

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
