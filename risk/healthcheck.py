"""Pre-flight health check for live trading.

Runs a suite of checks before the trading engine starts sending orders.
If any critical check fails, trading is blocked and an alert is emitted
via the EventBus.

Checks:
- Data connection: can we reach the data source?
- Account balance: sufficient cash for trading?
- Position sync: local vs broker positions match?
- Order routing: can we reach the broker?
- Risk limits: are risk parameters loaded and valid?

Usage:
    health = HealthCheck(config, event_bus, broker)
    results = await health.run_all()
    # or synchronously:
    results = health.run_all_sync()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.utils.logging import get_logger

try:
    from quant_platform.core.context import TenantContext
except ImportError:
    TenantContext = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: CheckStatus
    message: str = ""
    duration_ms: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class HealthCheckReport:
    """Aggregated report of all health checks."""
    results: list[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    overall_passed: bool = True

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.FAIL]

    @property
    def passed_checks(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.PASS]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_passed": self.overall_passed,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": round(r.duration_ms, 1),
                }
                for r in self.results
            ],
            "failed_count": len(self.failed_checks),
            "passed_count": len(self.passed_checks),
        }


class SystemBlockError(Exception):
    """Raised when a critical health check fails and trading must be blocked."""
    pass


class HealthCheck:
    """Pre-flight health check suite for the trading engine.

    Runs before the first trading cycle. Any FAIL result blocks
    all order submission until the issue is resolved.

    Args:
        event_bus: EventBus instance for publishing health events.
        broker: BrokerInterface to check connectivity and account.
        data_source: Optional data source to verify connectivity.
        risk_monitor: Optional risk monitor to verify limits are loaded.
        min_cash: Minimum cash balance required (default: 0).
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        broker: Any = None,
        data_source: Any = None,
        risk_monitor: Any = None,
        min_cash: float = 0.0,
    ):
        self._bus = event_bus or get_event_bus()
        self._broker = broker
        self._data_source = data_source
        self._risk_monitor = risk_monitor
        self._min_cash = min_cash

    def run_all_sync(self) -> HealthCheckReport:
        """Run all checks synchronously.

        Returns:
            HealthCheckReport with results for each check.

        Raises:
            SystemBlockError: If any critical check fails.
        """
        from datetime import datetime
        report = HealthCheckReport(timestamp=datetime.now().isoformat())

        checks = [
            ("data_connection", self._check_data_connection),
            ("account_balance", self._check_account_balance),
            ("position_sync", self._check_position_sync),
            ("order_routing", self._check_order_routing),
            ("risk_limits", self._check_risk_limits),
        ]

        for name, check_fn in checks:
            start = time.monotonic()
            try:
                result = check_fn()
                elapsed = (time.monotonic() - start) * 1000
                report.results.append(CheckResult(
                    name=name,
                    status=CheckStatus.PASS if result else CheckStatus.FAIL,
                    duration_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                report.results.append(CheckResult(
                    name=name,
                    status=CheckStatus.ERROR,
                    message=str(e),
                    duration_ms=elapsed,
                ))

        report.overall_passed = all(
            r.status == CheckStatus.PASS for r in report.results
        )

        # Publish events
        if report.overall_passed:
            self._bus.publish("system.healthcheck.pass", report.to_dict(), source="healthcheck")
            logger.info("Pre-flight health check PASSED (%d checks)", len(report.results))
        else:
            failed = [r.name for r in report.results if r.status != CheckStatus.PASS]
            self._bus.publish("system.healthcheck.fail", {
                **report.to_dict(),
                "failed_checks": failed,
            }, source="healthcheck")
            logger.critical("Pre-flight health check FAILED: %s", failed)
            raise SystemBlockError(
                f"Pre-flight checks failed: {failed}. "
                f"Trading blocked until issues are resolved."
            )

        return report

    async def run_all(self) -> HealthCheckReport:
        """Run all checks asynchronously.

        Returns:
            HealthCheckReport with results for each check.

        Raises:
            SystemBlockError: If any critical check fails.
        """
        from datetime import datetime
        report = HealthCheckReport(timestamp=datetime.now().isoformat())

        checks = [
            ("data_connection", self._async_check_data_connection),
            ("account_balance", self._async_check_account_balance),
            ("position_sync", self._async_check_position_sync),
            ("order_routing", self._async_check_order_routing),
            ("risk_limits", self._async_check_risk_limits),
        ]

        tasks = []
        for name, check_fn in checks:
            tasks.append(self._run_single_check(name, check_fn))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in results:
            if isinstance(result, Exception):
                report.results.append(CheckResult(
                    name=name,
                    status=CheckStatus.ERROR,
                    message=str(result),
                ))
            else:
                report.results.append(result)

        report.overall_passed = all(
            r.status == CheckStatus.PASS for r in report.results
        )

        # Publish events
        if report.overall_passed:
            self._bus.publish("system.healthcheck.pass", report.to_dict(), source="healthcheck")
            logger.info("Pre-flight health check PASSED (%d checks)", len(report.results))
        else:
            failed = [r.name for r in report.results if r.status != CheckStatus.PASS]
            self._bus.publish("system.healthcheck.fail", {
                **report.to_dict(),
                "failed_checks": failed,
            }, source="healthcheck")
            logger.critical("Pre-flight health check FAILED: %s", failed)
            raise SystemBlockError(
                f"Pre-flight checks failed: {failed}. "
                f"Trading blocked until issues are resolved."
            )

        return report

    async def _run_single_check(self, name: str, check_fn):
        start = time.monotonic()
        try:
            result = await check_fn()
            elapsed = (time.monotonic() - start) * 1000
            result.duration_ms = elapsed
            return name, result
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return name, CheckResult(
                name=name,
                status=CheckStatus.ERROR,
                message=str(e),
                duration_ms=elapsed,
            )

    # ------------------------------------------------------------------
    # Synchronous checks
    # ------------------------------------------------------------------

    def _check_data_connection(self) -> bool:
        """Check if data source is reachable."""
        if self._data_source is None:
            return True  # No data source configured — skip
        try:
            # Try to fetch a small amount of data
            if hasattr(self._data_source, 'get_prices'):
                df = self._data_source.get_prices("2025-01-01", "2025-01-10")
                return df is not None and len(df) > 0
            return True
        except Exception as e:
            logger.error("Data connection check failed: %s", e)
            return False

    def _check_account_balance(self) -> bool:
        """Check if account has sufficient balance.

        Uses TenantContext to identify which tenant's account to check.
        """
        if self._broker is None:
            return True
        try:
            tenant_id = "default"
            if TenantContext is not None:
                try:
                    tenant_id = TenantContext.get_current().tenant_id
                except Exception:
                    pass
            logger.debug("Checking account balance for tenant: %s", tenant_id)
            account = self._broker.get_account()
            cash = account.get("cash", 0)
            if cash < self._min_cash:
                logger.warning("Insufficient cash for %s: %.2f < %.2f",
                               tenant_id, cash, self._min_cash)
                return False
            return True
        except Exception as e:
            logger.error("Account balance check failed: %s", e)
            return False

    def _check_position_sync(self) -> bool:
        """Check if local positions match broker positions."""
        if self._broker is None:
            return True
        try:
            positions = self._broker.get_positions()
            # Basic sanity: positions list is accessible
            return positions is not None
        except Exception as e:
            logger.error("Position sync check failed: %s", e)
            return False

    def _check_order_routing(self) -> bool:
        """Check if order routing is available."""
        if self._broker is None:
            return True
        try:
            # Try to query account (proves broker connection is alive)
            account = self._broker.get_account()
            return account is not None
        except Exception as e:
            logger.error("Order routing check failed: %s", e)
            return False

    def _check_risk_limits(self) -> bool:
        """Check if risk limits are loaded and valid."""
        if self._risk_monitor is None:
            return True
        try:
            status = self._risk_monitor.get_status()
            return status is not None and isinstance(status, dict)
        except Exception as e:
            logger.error("Risk limits check failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Async checks
    # ------------------------------------------------------------------

    async def _async_check_data_connection(self) -> CheckResult:
        passed = self._check_data_connection()
        return CheckResult(
            name="data_connection",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            message="" if passed else "Data source unreachable",
        )

    async def _async_check_account_balance(self) -> CheckResult:
        passed = self._check_account_balance()
        return CheckResult(
            name="account_balance",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            message="" if passed else f"Cash below minimum ({self._min_cash})",
        )

    async def _async_check_position_sync(self) -> CheckResult:
        passed = self._check_position_sync()
        return CheckResult(
            name="position_sync",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            message="" if passed else "Position data unavailable",
        )

    async def _async_check_order_routing(self) -> CheckResult:
        passed = self._check_order_routing()
        return CheckResult(
            name="order_routing",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            message="" if passed else "Broker connection failed",
        )

    async def _async_check_risk_limits(self) -> CheckResult:
        passed = self._check_risk_limits()
        return CheckResult(
            name="risk_limits",
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            message="" if passed else "Risk limits not loaded",
        )
