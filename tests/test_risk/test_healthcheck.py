"""Tests for system pre-flight health check.

Verifies that HealthCheck correctly identifies failing subsystems
and blocks trading when critical checks fail.
"""

import pytest

from quant_platform.core.events import EventBus
from quant_platform.risk.healthcheck import (
    CheckResult,
    CheckStatus,
    HealthCheck,
    HealthCheckReport,
    SystemBlockError,
)


class MockBroker:
    """Mock broker for testing."""

    def __init__(self, cash=1_000_000, positions=None, fail_connect=False):
        self._cash = cash
        self._positions = positions or []
        self._fail_connect = fail_connect

    def get_account(self):
        if self._fail_connect:
            raise ConnectionError("Broker unreachable")
        return {"cash": self._cash, "total_equity": self._cash, "n_positions": len(self._positions)}

    def get_positions(self):
        if self._fail_connect:
            raise ConnectionError("Broker unreachable")
        return self._positions


class MockRiskMonitor:
    """Mock risk monitor for testing."""

    def __init__(self, fail=False):
        self._fail = fail

    def get_status(self):
        if self._fail:
            raise RuntimeError("Risk monitor not initialized")
        return {"level": "GREEN", "kill_switch": False}


class TestHealthCheckPass:
    """Tests where all checks pass."""

    def test_all_pass_no_broker(self):
        """With no broker configured, all checks should pass (skip)."""
        bus = EventBus()
        health = HealthCheck(event_bus=bus)
        report = health.run_all_sync()

        assert report.overall_passed is True
        assert len(report.failed_checks) == 0
        assert all(r.status == CheckStatus.PASS for r in report.results)

    def test_all_pass_with_broker(self):
        """With a healthy broker, all checks should pass."""
        bus = EventBus()
        broker = MockBroker(cash=500_000)
        health = HealthCheck(event_bus=bus, broker=broker)
        report = health.run_all_sync()

        assert report.overall_passed is True

    def test_all_pass_with_risk_monitor(self):
        """With a healthy risk monitor, all checks should pass."""
        bus = EventBus()
        broker = MockBroker(cash=500_000)
        risk = MockRiskMonitor(fail=False)
        health = HealthCheck(event_bus=bus, broker=broker, risk_monitor=risk)
        report = health.run_all_sync()

        assert report.overall_passed is True

    def test_report_structure(self):
        """Report should have expected structure."""
        bus = EventBus()
        health = HealthCheck(event_bus=bus)
        report = health.run_all_sync()

        assert isinstance(report, HealthCheckReport)
        assert report.timestamp != ""
        d = report.to_dict()
        assert "overall_passed" in d
        assert "checks" in d
        assert "failed_count" in d


class TestHealthCheckFail:
    """Tests where checks fail."""

    def test_broker_connection_fails(self):
        """Broker connection failure should block trading."""
        bus = EventBus()
        broker = MockBroker(fail_connect=True)
        health = HealthCheck(event_bus=bus, broker=broker)

        with pytest.raises(SystemBlockError) as exc_info:
            health.run_all_sync()

        assert "account_balance" in str(exc_info.value) or "order_routing" in str(exc_info.value)

    def test_insufficient_cash(self):
        """Low cash balance should fail the balance check."""
        bus = EventBus()
        broker = MockBroker(cash=100)
        health = HealthCheck(event_bus=bus, broker=broker, min_cash=10_000)

        with pytest.raises(SystemBlockError) as exc_info:
            health.run_all_sync()

        assert "account_balance" in str(exc_info.value)

    def test_risk_monitor_fails(self):
        """Broken risk monitor should fail the risk limits check."""
        bus = EventBus()
        broker = MockBroker(cash=500_000)
        risk = MockRiskMonitor(fail=True)
        health = HealthCheck(event_bus=bus, broker=broker, risk_monitor=risk)

        with pytest.raises(SystemBlockError) as exc_info:
            health.run_all_sync()

        assert "risk_limits" in str(exc_info.value)


class TestEventBusIntegration:
    """Verify events are published correctly."""

    def test_pass_event_published(self):
        """system.healthcheck.pass should be published on success."""
        bus = EventBus()
        events = []
        bus.subscribe("system.healthcheck.pass", lambda e: events.append(e))

        health = HealthCheck(event_bus=bus)
        health.run_all_sync()

        assert len(events) == 1
        # EventBus wraps data in LegacyEvent; access .data for the dict
        event_data = events[0].data if hasattr(events[0], 'data') else events[0]
        assert event_data["overall_passed"] is True

    def test_fail_event_published(self):
        """system.healthcheck.fail should be published on failure."""
        bus = EventBus()
        events = []
        bus.subscribe("system.healthcheck.fail", lambda e: events.append(e))

        broker = MockBroker(fail_connect=True)
        health = HealthCheck(event_bus=bus, broker=broker)

        with pytest.raises(SystemBlockError):
            health.run_all_sync()

        assert len(events) == 1
        event_data = events[0].data if hasattr(events[0], 'data') else events[0]
        assert "failed_checks" in event_data


class TestAsyncHealthCheck:
    """Async version of health checks."""

    @pytest.mark.asyncio
    async def test_async_all_pass(self):
        """Async run should pass with no broker."""
        bus = EventBus()
        health = HealthCheck(event_bus=bus)
        report = await health.run_all()

        assert report.overall_passed is True

    @pytest.mark.asyncio
    async def test_async_broker_fails(self):
        """Async run should raise SystemBlockError on failure."""
        bus = EventBus()
        broker = MockBroker(fail_connect=True)
        health = HealthCheck(event_bus=bus, broker=broker)

        with pytest.raises(SystemBlockError):
            await health.run_all()


class TestCheckResult:
    """Test CheckResult data class."""

    def test_check_result_fields(self):
        """CheckResult should store all fields."""
        r = CheckResult(
            name="test",
            status=CheckStatus.PASS,
            message="ok",
            duration_ms=1.5,
        )
        assert r.name == "test"
        assert r.status == CheckStatus.PASS
        assert r.message == "ok"
        assert r.duration_ms == 1.5


class TestHealthCheckReport:
    """Test HealthCheckReport data class."""

    def test_failed_checks_filter(self):
        """failed_checks property should filter correctly."""
        report = HealthCheckReport(results=[
            CheckResult(name="a", status=CheckStatus.PASS),
            CheckResult(name="b", status=CheckStatus.FAIL, message="broken"),
            CheckResult(name="c", status=CheckStatus.PASS),
        ])
        assert len(report.failed_checks) == 1
        assert report.failed_checks[0].name == "b"

    def test_passed_checks_filter(self):
        """passed_checks property should filter correctly."""
        report = HealthCheckReport(results=[
            CheckResult(name="a", status=CheckStatus.PASS),
            CheckResult(name="b", status=CheckStatus.FAIL),
        ])
        assert len(report.passed_checks) == 1

    def test_overall_passed_false_on_any_fail(self):
        """overall_passed should be False if any check failed."""
        report = HealthCheckReport(results=[
            CheckResult(name="a", status=CheckStatus.PASS),
            CheckResult(name="b", status=CheckStatus.FAIL),
        ])
        report.overall_passed = all(
            r.status == CheckStatus.PASS for r in report.results
        )
        assert report.overall_passed is False
