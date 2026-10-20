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

A check whose input is not configured reports ``SKIP`` (not ``PASS``): an
unconfigured subsystem was never verified, and reporting it as passing made
the gate vacuous. ``ERROR`` results count as failures in ``failed_checks`` /
``failed_count`` and block trading, like ``FAIL``.

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
from enum import StrEnum
from typing import Any

from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.utils.logging import get_logger

try:
    from quant_platform.core.context import TenantContext
except ImportError:
    TenantContext = None  # type: ignore[assignment,misc]

logger = get_logger(__name__)


class CheckStatus(StrEnum):
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
        """Checks that block trading: FAIL *and* ERROR.

        ERROR used to be excluded here (and from ``failed_count``) even
        though ``overall_passed`` was false, so monitoring under-reported
        failed subsystems.
        """
        return [
            r for r in self.results
            if r.status in (CheckStatus.FAIL, CheckStatus.ERROR)
        ]

    @property
    def passed_checks(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.PASS]

    @property
    def skipped_checks(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == CheckStatus.SKIP]

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
            "skipped_count": len(self.skipped_checks),
        }


class SystemBlockError(Exception):
    """Raised when a critical health check fails and trading must be blocked."""
    pass


class HealthCheck:
    """Pre-flight health check suite for the trading engine.

    Runs before the first trading cycle. Any FAIL or ERROR result blocks
    all order submission until the issue is resolved. SKIP means the
    subsystem is not configured, so it could not be verified either way.

    Args:
        event_bus: EventBus instance for publishing health events.
        broker: BrokerInterface to check connectivity and account.
        data_source: Optional data source to verify connectivity.
        risk_monitor: Optional risk monitor to verify limits are loaded.
        min_cash: Minimum cash balance required. ``0`` explicitly disables
            the minimum; a missing cash figure then cannot fail the check.
        local_positions: Callable returning local/Store position rows, or
            a list. Without it the sync check reports SKIP.
    """

    _REQUIRED_RISK_KEYS = ("risk_level", "kill_switch", "limits")

    def __init__(
        self,
        event_bus: EventBus | None = None,
        broker: Any = None,
        data_source: Any = None,
        risk_monitor: Any = None,
        min_cash: float = 0.0,
        local_positions: Any = None,
    ):
        self._bus = event_bus or get_event_bus()
        self._broker = broker
        self._data_source = data_source
        self._risk_monitor = risk_monitor
        self._min_cash = min_cash
        # Callable returning the local/Store position rows, or a list.
        self._local_positions = local_positions

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _evaluators(self) -> list[tuple[str, Any]]:
        return [
            ("data_connection", self._evaluate_data_connection),
            ("account_balance", self._evaluate_account_balance),
            ("position_sync", self._evaluate_position_sync),
            ("order_routing", self._evaluate_order_routing),
            ("risk_limits", self._evaluate_risk_limits),
        ]

    @staticmethod
    def _is_acceptable(result: CheckResult) -> bool:
        return result.status in (CheckStatus.PASS, CheckStatus.SKIP)

    def _finalize(self, report: HealthCheckReport) -> HealthCheckReport:
        report.overall_passed = all(
            self._is_acceptable(r) for r in report.results
        )

        if report.overall_passed:
            self._bus.publish("system.healthcheck.pass", report.to_dict(), source="healthcheck")
            logger.info("Pre-flight health check PASSED (%d checks)", len(report.results))
        else:
            failed = [r.name for r in report.failed_checks]
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

    def run_all_sync(self) -> HealthCheckReport:
        """Run all checks synchronously.

        Returns:
            HealthCheckReport with results for each check.

        Raises:
            SystemBlockError: If any critical check fails or errors.
        """
        from datetime import datetime
        report = HealthCheckReport(timestamp=datetime.now().isoformat())

        for name, check_fn in self._evaluators():
            start = time.monotonic()
            try:
                result = check_fn()
            except Exception as e:
                result = CheckResult(
                    name=name, status=CheckStatus.ERROR, message=str(e),
                )
            result.duration_ms = (time.monotonic() - start) * 1000
            report.results.append(result)

        return self._finalize(report)

    async def run_all(self) -> HealthCheckReport:
        """Run all checks asynchronously.

        Returns:
            HealthCheckReport with results for each check.

        Raises:
            SystemBlockError: If any critical check fails or errors.
        """
        from datetime import datetime
        report = HealthCheckReport(timestamp=datetime.now().isoformat())

        tasks = [
            self._run_single_check(name, check_fn)
            for name, check_fn in self._evaluators()
        ]
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

        return self._finalize(report)

    async def _run_single_check(self, name: str, check_fn):
        start = time.monotonic()
        try:
            result = check_fn()
            result.duration_ms = (time.monotonic() - start) * 1000
            return name, result
        except Exception as e:
            return name, CheckResult(
                name=name,
                status=CheckStatus.ERROR,
                message=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    # ------------------------------------------------------------------
    # Check implementations (return CheckResult: PASS / FAIL / SKIP)
    # ------------------------------------------------------------------

    def _evaluate_data_connection(self) -> CheckResult:
        if self._data_source is None:
            return CheckResult(
                name="data_connection", status=CheckStatus.SKIP,
                message="no data source configured; connectivity not verified",
            )
        try:
            if hasattr(self._data_source, "get_prices"):
                df = self._data_source.get_prices("2025-01-01", "2025-01-10")
                if df is not None and len(df) > 0:
                    return CheckResult(
                        name="data_connection", status=CheckStatus.PASS,
                    )
                return CheckResult(
                    name="data_connection", status=CheckStatus.FAIL,
                    message="data source returned no rows",
                )
            return CheckResult(name="data_connection", status=CheckStatus.PASS)
        except Exception as e:
            logger.error("Data connection check failed: %s", e)
            return CheckResult(
                name="data_connection", status=CheckStatus.FAIL,
                message=f"data source unreachable: {e}",
            )

    def _evaluate_account_balance(self) -> CheckResult:
        if self._broker is None:
            return CheckResult(
                name="account_balance", status=CheckStatus.SKIP,
                message="no broker configured; balance not verified",
            )
        try:
            tenant_id = "default"
            if TenantContext is not None:
                try:
                    tenant_id = TenantContext.get_current().tenant_id
                except Exception:
                    pass
            logger.debug("Checking account balance for tenant: %s", tenant_id)
            account = self._broker.get_account()
            if not isinstance(account, dict):
                return CheckResult(
                    name="account_balance", status=CheckStatus.FAIL,
                    message=f"broker account query returned {type(account).__name__}",
                )
            cash = account.get("cash")
            if cash is None:
                if self._min_cash == 0:
                    return CheckResult(
                        name="account_balance", status=CheckStatus.PASS,
                        message="no cash figure; no minimum configured",
                    )
                return CheckResult(
                    name="account_balance", status=CheckStatus.FAIL,
                    message=(
                        "account balance has no cash figure and a minimum "
                        f"cash of {self._min_cash} is configured"
                    ),
                )
            if float(cash) < self._min_cash:
                logger.warning(
                    "Insufficient cash for %s: %.2f < %.2f",
                    tenant_id, float(cash), self._min_cash,
                )
                return CheckResult(
                    name="account_balance", status=CheckStatus.FAIL,
                    message=f"cash {float(cash):.2f} below minimum {self._min_cash}",
                )
            return CheckResult(name="account_balance", status=CheckStatus.PASS)
        except Exception as e:
            logger.error("Account balance check failed: %s", e)
            return CheckResult(
                name="account_balance", status=CheckStatus.FAIL,
                message=f"balance query failed: {e}",
            )

    def _evaluate_position_sync(self) -> CheckResult:
        """Compare the broker's book with the local/Store book.

        The check used to return `positions is not None` -- it never compared
        anything. Without a local book there is nothing to compare, which is
        now reported as SKIP instead of a vacuous PASS.
        """
        if self._broker is None:
            return CheckResult(
                name="position_sync", status=CheckStatus.SKIP,
                message="no broker configured; positions not verified",
            )
        try:
            broker_positions = self._broker.get_positions()
            if broker_positions is None:
                return CheckResult(
                    name="position_sync", status=CheckStatus.FAIL,
                    message="broker position query returned UNKNOWN (None)",
                )

            local = self._local_positions
            if local is None:
                logger.warning(
                    "Position sync check has no local book; cannot compare",
                )
                return CheckResult(
                    name="position_sync", status=CheckStatus.SKIP,
                    message="no local book supplied; broker/local books not compared",
                )
            if callable(local):
                local = local()

            local_map = {}
            for row in local or []:
                code = str(row.get("code") or row.get("ticker") or "")
                quantity = int(row.get("quantity", 0) or 0)
                if code and quantity > 0:
                    local_map[code] = quantity
            broker_map = {}
            for p in broker_positions:
                quantity = int(getattr(p, "quantity", 0) or 0)
                # Zero-quantity rows are "flat", not a holding: a broker that
                # reports closed positions alongside open ones must not make
                # the sync check fail against the Store's quantity>0 rows.
                if quantity > 0:
                    broker_map[str(p.code)] = quantity

            mismatches = [
                f"{code}: broker={broker_map.get(code, 0)} local={local_map.get(code, 0)}"
                for code in sorted(set(local_map) | set(broker_map))
                if local_map.get(code, 0) != broker_map.get(code, 0)
            ]
            if mismatches:
                logger.error(
                    "Position mismatch between broker and local book: %s",
                    "; ".join(mismatches[:5]),
                )
                return CheckResult(
                    name="position_sync", status=CheckStatus.FAIL,
                    message="; ".join(mismatches[:5]),
                    details={"mismatches": mismatches},
                )
            return CheckResult(name="position_sync", status=CheckStatus.PASS)
        except Exception as e:
            logger.error("Position sync check failed: %s", e)
            return CheckResult(
                name="position_sync", status=CheckStatus.FAIL,
                message=f"position sync failed: {e}",
            )

    def _evaluate_order_routing(self) -> CheckResult:
        if self._broker is None:
            return CheckResult(
                name="order_routing", status=CheckStatus.SKIP,
                message="no broker configured; routing not verified",
            )
        try:
            # Try to query account (proves broker connection is alive)
            account = self._broker.get_account()
            if account is None:
                return CheckResult(
                    name="order_routing", status=CheckStatus.FAIL,
                    message="broker account query returned None",
                )
            # QMT/XTP return a fallback account dict with connected=False when
            # the terminal is down; treating that as healthy let the engine
            # start and reject every order.
            if isinstance(account, dict) and account.get("connected") is False:
                logger.error("Order routing check failed: broker reports disconnected")
                return CheckResult(
                    name="order_routing", status=CheckStatus.FAIL,
                    message="broker reports disconnected",
                )
            return CheckResult(name="order_routing", status=CheckStatus.PASS)
        except Exception as e:
            logger.error("Order routing check failed: %s", e)
            return CheckResult(
                name="order_routing", status=CheckStatus.FAIL,
                message=f"broker connection failed: {e}",
            )

    def _evaluate_risk_limits(self) -> CheckResult:
        if self._risk_monitor is None:
            return CheckResult(
                name="risk_limits", status=CheckStatus.SKIP,
                message="no risk monitor configured; limits not verified",
            )
        try:
            status = self._risk_monitor.get_status()
            if not isinstance(status, dict):
                return CheckResult(
                    name="risk_limits", status=CheckStatus.FAIL,
                    message=(
                        "risk monitor returned no status mapping "
                        f"({type(status).__name__})"
                    ),
                )
            missing = [k for k in self._REQUIRED_RISK_KEYS if k not in status]
            if missing:
                return CheckResult(
                    name="risk_limits", status=CheckStatus.FAIL,
                    message=f"risk status is missing required keys: {missing}",
                )
            return CheckResult(name="risk_limits", status=CheckStatus.PASS)
        except Exception as e:
            logger.error("Risk limits check failed: %s", e)
            return CheckResult(
                name="risk_limits", status=CheckStatus.FAIL,
                message=f"risk status query failed: {e}",
            )

    # ------------------------------------------------------------------
    # Boolean compatibility wrappers (used by older callers/tests)
    # ------------------------------------------------------------------

    def _check_data_connection(self) -> bool:
        return self._evaluate_data_connection().status == CheckStatus.PASS

    def _check_account_balance(self) -> bool:
        return self._evaluate_account_balance().status == CheckStatus.PASS

    def _check_position_sync(self) -> bool:
        return self._evaluate_position_sync().status == CheckStatus.PASS

    def _check_order_routing(self) -> bool:
        return self._evaluate_order_routing().status == CheckStatus.PASS

    def _check_risk_limits(self) -> bool:
        return self._evaluate_risk_limits().status == CheckStatus.PASS

    # ------------------------------------------------------------------
    # Async wrappers (kept for callers that awaited these directly)
    # ------------------------------------------------------------------

    async def _async_check_data_connection(self) -> CheckResult:
        return self._evaluate_data_connection()

    async def _async_check_account_balance(self) -> CheckResult:
        return self._evaluate_account_balance()

    async def _async_check_position_sync(self) -> CheckResult:
        return self._evaluate_position_sync()

    async def _async_check_order_routing(self) -> CheckResult:
        return self._evaluate_order_routing()

    async def _async_check_risk_limits(self) -> CheckResult:
        return self._evaluate_risk_limits()
