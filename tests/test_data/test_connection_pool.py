"""Tests for data provider connection pool."""

import time

import pandas as pd
from quant_platform.data.providers.connection_pool import (
    DataProviderPool,
    SourceHealth,
    SourceStatus,
)


class MockProvider:
    """Mock data provider for testing."""
    def __init__(self, name="mock", should_fail=False, delay=0):
        self.name = name
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0

    def get_prices(self, start_date, end_date, **kwargs):
        self.call_count += 1
        if self.delay:
            time.sleep(self.delay)
        if self.should_fail:
            raise ConnectionError(f"{self.name} is down")
        return pd.DataFrame({"close": [100, 101, 102]}, index=["d1", "d2", "d3"])


class TestSourceHealth:
    def test_initial_status(self):
        h = SourceHealth(name="test")
        assert h.status == SourceStatus.HEALTHY
        assert h.available is True

    def test_record_success(self):
        h = SourceHealth(name="test")
        h.record_success(50.0)
        assert h.success_count == 1
        assert h.consecutive_failures == 0
        assert h.avg_latency_ms == 50.0

    def test_record_failure_circuit_breaker(self):
        h = SourceHealth(name="test")
        h.record_failure()
        h.record_failure()
        h.record_failure()
        assert h.status == SourceStatus.CIRCUIT_OPEN
        assert h.available is False

    def test_circuit_breaker_recovery(self):
        h = SourceHealth(name="test")
        for _ in range(3):
            h.record_failure()
        # Force cooldown to expire
        h.circuit_open_until = time.time() - 1
        assert h.available is True
        assert h.status == SourceStatus.DEGRADED

    def test_degraded_recovery(self):
        h = SourceHealth(name="test")
        h.record_failure()
        assert h.status == SourceStatus.DEGRADED
        h.record_success(10.0)
        assert h.status == SourceStatus.HEALTHY


class TestDataProviderPool:
    def test_basic_routing(self):
        primary = MockProvider("primary")
        pool = DataProviderPool(
            providers={"primary": primary},
            priorities={"primary": 1},
        )
        result = pool.get_prices("2021-01-01", "2021-12-31")
        assert result is not None
        assert len(result) == 3
        assert primary.call_count == 1

    def test_failover(self):
        primary = MockProvider("primary", should_fail=True)
        fallback = MockProvider("fallback")
        pool = DataProviderPool(
            providers={"primary": primary, "fallback": fallback},
            priorities={"primary": 1, "fallback": 2},
        )
        result = pool.get_prices("2021-01-01", "2021-12-31")
        assert result is not None
        assert primary.call_count == 1
        assert fallback.call_count == 1

    def test_all_fail(self):
        p1 = MockProvider("p1", should_fail=True)
        p2 = MockProvider("p2", should_fail=True)
        pool = DataProviderPool(
            providers={"p1": p1, "p2": p2},
            priorities={"p1": 1, "p2": 2},
        )
        result = pool.get_prices("2021-01-01", "2021-12-31")
        assert result is None

    def test_circuit_breaker_skip(self):
        # First 3 failures open the circuit
        primary = MockProvider("primary", should_fail=True)
        fallback = MockProvider("fallback")
        pool = DataProviderPool(
            providers={"primary": primary, "fallback": fallback},
            priorities={"primary": 1, "fallback": 2},
        )

        # Exhaust circuit breaker
        for _ in range(3):
            pool.get_prices("2021-01-01", "2021-12-31")

        # Primary should be circuit-open, go directly to fallback
        fallback.call_count = 0
        result = pool.get_prices("2022-01-01", "2022-12-31")
        assert result is not None

    def test_cache(self):
        provider = MockProvider("cached")
        pool = DataProviderPool(
            providers={"cached": provider},
            priorities={"cached": 1},
        )
        # First call
        pool.get_prices("2021-01-01", "2021-12-31")
        # Second call (should be cached)
        pool.get_prices("2021-01-01", "2021-12-31")
        assert provider.call_count == 1  # Only called once

    def test_health_check(self):
        primary = MockProvider("primary")
        pool = DataProviderPool(
            providers={"primary": primary},
            priorities={"primary": 1},
        )
        pool.get_prices("2021-01-01", "2021-12-31")
        health = pool.health_check()
        assert "primary" in health
        assert health["primary"]["success_count"] == 1
        assert health["primary"]["status"] == "healthy"

    def test_get_best_source(self):
        primary = MockProvider("primary")
        fallback = MockProvider("fallback")
        pool = DataProviderPool(
            providers={"primary": primary, "fallback": fallback},
            priorities={"primary": 1, "fallback": 2},
        )
        assert pool.get_best_source() == "primary"

    def test_get_best_source_all_down(self):
        p1 = MockProvider("p1", should_fail=True)
        pool = DataProviderPool(
            providers={"p1": p1},
            priorities={"p1": 1},
        )
        # Open circuit
        for _ in range(3):
            pool.get_prices("2021-01-01", "2021-12-31")
        assert pool.get_best_source() == "none"

    def test_reset_circuit_breaker(self):
        provider = MockProvider("test", should_fail=True)
        pool = DataProviderPool(
            providers={"test": provider},
            priorities={"test": 1},
        )
        for _ in range(3):
            pool.get_prices("2021-01-01", "2021-12-31")

        pool.reset_circuit_breaker("test")
        health = pool.health_check()
        assert health["test"]["status"] == "healthy"

    def test_clear_cache(self):
        provider = MockProvider("test")
        pool = DataProviderPool(
            providers={"test": provider},
            priorities={"test": 1},
        )
        pool.get_prices("2021-01-01", "2021-12-31")
        pool.clear_cache()
        pool.get_prices("2021-01-01", "2021-12-31")
        assert provider.call_count == 2

    def test_priority_ordering(self):
        p1 = MockProvider("p1")
        p2 = MockProvider("p2")
        p3 = MockProvider("p3")
        pool = DataProviderPool(
            providers={"p1": p1, "p2": p2, "p3": p3},
            priorities={"p1": 3, "p2": 1, "p3": 2},
        )
        ordered = [name for name, _ in pool._get_ordered_providers()]
        assert ordered == ["p2", "p3", "p1"]
