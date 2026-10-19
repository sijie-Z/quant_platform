"""Audit log — compliance-grade decision tracking.

Every signal, order, fill, and state change is logged with:
- Who (component/strategy)
- What (action)
- When (timestamp)
- Why (reason/trigger)
- Result (outcome)

Stored in SQLite for queryability. Published on EventBus for real-time monitoring.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from enum import StrEnum

from quant_platform.core.events import EventBus, get_event_bus
from quant_platform.core.store import Store
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class AuditAction(StrEnum):
    SIGNAL_GENERATED = "signal_generated"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_BREACH = "risk_breach"
    STATE_CHANGE = "state_change"
    REBALANCE_START = "rebalance_start"
    REBALANCE_END = "rebalance_end"
    ENGINE_START = "engine_start"
    ENGINE_STOP = "engine_stop"
    CONFIG_CHANGE = "config_change"
    MANUAL_ORDER = "manual_order"


class AuditLog:
    """Compliance-grade audit trail.

    All decisions are logged to:
    1. SQLite (queryable, persistent)
    2. EventBus (real-time monitoring)
    3. Python logger (stdout/file)
    """

    def __init__(
        self,
        store: Store,
        bus: EventBus | None = None,
        tenant_id: str = "",
    ):
        self._store = store
        self._bus = bus or get_event_bus()
        # An explicit tenant wins over the context: engines create their
        # AuditLog on one thread and log from a worker, where the ContextVar
        # set in the constructor is not visible.
        self._tenant_id = tenant_id

    def _tenant(self) -> str:
        if self._tenant_id:
            return self._tenant_id
        try:
            from quant_platform.core.context import TenantContext

            return TenantContext.get_current().tenant_id
        except Exception:
            return "default"

    def log(
        self,
        action: AuditAction,
        component: str,
        details: dict | None = None,
        reason: str = "",
        result: str = "success",
    ):
        """Log an audit event."""
        event_id = uuid.uuid4().hex[:12]
        datetime.now().isoformat()

        record = {
            "event_id": event_id,
            "tenant_id": self._tenant(),
            "topic": f"audit.{action.value}",
            "data": {
                "action": action.value,
                "component": component,
                "details": details or {},
                "reason": reason,
                "result": result,
                "tenant_id": self._tenant(),
            },
            "source": component,
            "timestamp": time.time(),
        }

        # Store in SQLite
        self._store.log_event(record)

        # Publish on bus
        self._bus.publish(f"audit.{action.value}", record["data"], source=component)

        # Log
        logger.info("[AUDIT] %s | %s | %s | %s",
                     action.value, component, reason or "-", result)

    def log_signal(self, code: str, direction: str, strength: float,
                   strategy: str = "", factors: dict | None = None):
        """Log a signal generation."""
        self._store.save_signal({
            "signal_id": uuid.uuid4().hex[:12],
            "tenant_id": self._tenant(),
            "code": code,
            "direction": direction,
            "strength": strength,
            "factor_values": factors or {},
            "strategy_id": strategy,
            "generated_at": datetime.now().isoformat(),
        })
        self.log(
            AuditAction.SIGNAL_GENERATED,
            component=strategy or "engine",
            details={"code": code, "direction": direction, "strength": strength},
            reason=f"signal: {direction} {code} (strength={strength:.4f})",
        )

    def log_order(self, order: dict, action: AuditAction, reason: str = ""):
        """Log an order event."""
        tenant_id = self._tenant()
        order = {**order, "tenant_id": order.get("tenant_id") or tenant_id}
        if action == AuditAction.ORDER_FILLED:
            # Order + fill in one transaction: a second fill on the same
            # order used to fail on the trades UNIQUE key *after* the order
            # row had been written, losing the fill and its audit record.
            # The store generates the trade_id (uuid4).
            self._store.save_order_with_trade(order, {
                "tenant_id": tenant_id,
                "order_id": order["order_id"],
                "code": order["code"],
                "side": order["side"],
                "quantity": order.get("filled_quantity", order["quantity"]),
                "price": order.get("filled_price", order["price"]),
                "commission": order.get("commission", 0),
                "tax": order.get("tax", 0),
            })
        else:
            self._store.save_order(order)
        self.log(
            action,
            component="broker",
            details={
                "order_id": order.get("order_id", ""),
                "code": order["code"],
                "side": order["side"],
                "quantity": order["quantity"],
                "price": order["price"],
            },
            reason=reason,
        )

    def log_position(self, position: dict, action: AuditAction):
        """Log a position change."""
        tenant_id = self._tenant()
        if action == AuditAction.POSITION_CLOSED:
            # Without the tenant this deleted every tenant's row for the code.
            self._store.delete_position(position["code"], tenant_id=tenant_id)
        else:
            self._store.save_position({
                **position, "tenant_id": position.get("tenant_id") or tenant_id,
            })
        self.log(
            action,
            component="portfolio",
            details={"code": position.get("code", ""), "quantity": position.get("quantity", 0)},
        )

    def log_state_change(self, from_state: str, to_state: str, reason: str = ""):
        """Log a state machine transition."""
        self.log(
            AuditAction.STATE_CHANGE,
            component="state_machine",
            details={"from": from_state, "to": to_state},
            reason=reason,
        )

    def log_risk_breach(self, breach_type: str, details: dict, severity: str = "warning"):
        """Log a risk breach."""
        self.log(
            AuditAction.RISK_BREACH,
            component="risk",
            details={"breach_type": breach_type, "severity": severity, **details},
            reason=f"Risk breach: {breach_type}",
            result="warning",
        )

    def get_recent(self, action: str = "", limit: int = 50,
                   tenant_id: str = "") -> list[dict]:
        """Get recent audit events, optionally for one tenant."""
        topic = f"audit.{action}" if action else ""
        return self._store.get_events(
            topic=topic, limit=limit, tenant_id=tenant_id,
        )
