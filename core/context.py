"""Tenant context — coroutine-safe tenant identity propagation.

Uses contextvars to propagate tenant_id through async call chains
without explicit parameter threading. Each async task gets its own
context copy, so concurrent tenants are properly isolated.

Usage:
    from quant_platform.core.context import TenantContext

    ctx = TenantContext(tenant_id="fund_001", strategy_id="momentum_v2")
    TenantContext.set_current(ctx)

    # Later, in any downstream code:
    ctx = TenantContext.get_current()
    print(ctx.tenant_id)  # "fund_001"
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_tenant_context: ContextVar[TenantContext | None] = ContextVar(
    "tenant_context", default=None
)


@dataclass
class TenantContext:
    """Tenant-scoped context for multi-tenant isolation.

    Attributes:
        tenant_id: Unique tenant identifier (e.g., fund ID).
        strategy_id: Strategy running under this tenant.
        account_mapping: Maps logical accounts to broker accounts.
        risk_limits: Per-tenant risk limit overrides.
    """
    tenant_id: str = "default"
    strategy_id: str = ""
    account_mapping: dict[str, str] = field(default_factory=dict)
    risk_limits: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def set_current(cls, context: TenantContext) -> None:
        """Set the current tenant context for this coroutine/thread."""
        _tenant_context.set(context)

    @classmethod
    def get_current(cls) -> TenantContext:
        """Get the current tenant context. Returns default if not set."""
        ctx = _tenant_context.get()
        if ctx is None:
            return cls()  # default tenant
        return ctx

    @classmethod
    def clear(cls) -> None:
        """Clear the current context (useful in tests)."""
        _tenant_context.set(None)
