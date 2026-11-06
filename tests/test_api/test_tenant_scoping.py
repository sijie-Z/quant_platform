"""Tenant scoping for /api/core/store/* and /api/core/audit.

Before the fix these endpoints called Store without a tenant, and every Store
predicate is skipped when the tenant is falsey, so one request merged every
tenant's orders/trades/P&L/signals/sessions/audit rows. Cross-tenant reads now
require an explicit ``all_tenants=true`` from a loopback client; the shared
API token is *not* a tenant boundary.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.app import create_app
from quant_platform.core.audit import AuditLog
from quant_platform.core.store import Store


class _RemoteClient:
    """Rewrite the ASGI client address so a TestClient looks remote."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope)
            scope["client"] = ("203.0.113.9", 54321)
        await self.app(scope, receive, send)


def _seed(store: Store, tenant: str) -> None:
    store.save_order({
        "order_id": f"o-{tenant}", "tenant_id": tenant, "code": "600519",
        "side": "buy", "quantity": 100, "price": 10.0,
    })
    store.save_trade({
        "trade_id": f"t-{tenant}", "tenant_id": tenant,
        "order_id": f"o-{tenant}", "code": "600519", "side": "buy",
        "quantity": 100, "price": 10.0,
        "executed_at": "2026-10-01T10:00:00",
    })
    store.save_signal({
        "signal_id": f"s-{tenant}", "tenant_id": tenant, "code": "600519",
        "direction": "long", "strength": 0.5,
        "generated_at": "2026-10-01T09:00:00",
    })
    store.save_pnl_snapshot({
        "tenant_id": tenant, "total_equity": 100.0, "cash": 50.0,
        "market_value": 50.0, "timestamp": "2026-10-01T15:00:00",
    })
    store.save_session({
        "session_id": f"session-{tenant}", "tenant_id": tenant,
        "started_at": "2026-10-01T09:30:00",
    })
    store.log_event({
        "event_id": f"e-{tenant}", "tenant_id": tenant,
        "topic": "audit.test", "data": {"tenant": tenant},
        "timestamp": time.time(),
    })


@pytest.fixture
def tenant_clients(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "tenant.db"))
    _seed(store, "fund_a")
    _seed(store, "fund_b")
    monkeypatch.setattr(routes, "_core_store", store)
    monkeypatch.setattr(routes, "_core_audit", AuditLog(store))
    app = create_app(serve_frontend=False)
    return store, TestClient(app), TestClient(_RemoteClient(app))


def test_store_reads_default_to_the_default_tenant(tenant_clients):
    _, client, _ = tenant_clients

    for path in (
        "/api/core/store/orders",
        "/api/core/store/trades",
        "/api/core/store/pnl",
        "/api/core/store/signals",
        "/api/core/store/sessions",
        "/api/core/audit",
    ):
        payload = client.get(path).json()
        assert payload["tenant_id"] == "default", path
        assert payload["all_tenants"] is False, path
        rows = payload.get("orders", payload.get("trades", payload.get(
            "history", payload.get("signals", payload.get(
                "sessions", payload.get("events"))))))
        assert rows == [], (path, rows)


def test_store_reads_scope_to_tenant_header_and_query(tenant_clients):
    _, client, _ = tenant_clients

    by_header = client.get(
        "/api/core/store/orders", headers={"X-Tenant-Id": "fund_a"},
    ).json()
    assert [row["tenant_id"] for row in by_header["orders"]] == ["fund_a"]

    by_query = client.get(
        "/api/core/store/trades", params={"tenant_id": "fund_b"},
    ).json()
    assert [row["tenant_id"] for row in by_query["trades"]] == ["fund_b"]

    for path, key in (
        ("/api/core/store/pnl", "history"),
        ("/api/core/store/signals", "signals"),
        ("/api/core/store/sessions", "sessions"),
        ("/api/core/audit", "events"),
    ):
        payload = client.get(path, params={"tenant_id": "fund_a"}).json()
        assert {row["tenant_id"] for row in payload[key]} == {"fund_a"}, path
        assert payload["tenant_id"] == "fund_a"


def test_stats_are_tenant_scoped(tenant_clients):
    _, client, _ = tenant_clients

    payload = client.get(
        "/api/core/store/stats", headers={"X-Tenant-Id": "fund_a"},
    ).json()
    assert payload["tenant_id"] == "fund_a"
    assert payload["orders"] == 1
    assert payload["trades"] == 1
    assert payload["signals"] == 1
    assert payload["sessions"] == 1
    assert payload["events"] == 1


def test_all_tenants_is_loopback_only(tenant_clients, monkeypatch):
    _, client, remote = tenant_clients

    both = client.get(
        "/api/core/store/orders", params={"all_tenants": "true"},
    ).json()
    assert both["all_tenants"] is True
    assert {row["tenant_id"] for row in both["orders"]} == {"fund_a", "fund_b"}

    # Even a valid token does not grant a cross-tenant read from a remote IP.
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    denied = remote.get(
        "/api/core/store/orders",
        params={"all_tenants": "true"},
        headers={"Authorization": "Bearer secret"},
    )
    assert denied.status_code == 403
    assert "loopback" in denied.json()["detail"]


def test_invalid_tenant_id_is_rejected(tenant_clients):
    _, client, _ = tenant_clients

    resp = client.get(
        "/api/core/store/orders", headers={"X-Tenant-Id": "../evil tenant"},
    )
    assert resp.status_code == 422


def test_delete_position_requires_a_tenant(tmp_path):
    store = Store(str(tmp_path / "positions.db"))
    store.save_position({"code": "600519", "quantity": 100, "avg_cost": 10.0})

    with pytest.raises(ValueError):
        store.delete_position("600519")
    assert len(store.get_positions()) == 1

    store.delete_position("600519", tenant_id="default")
    assert store.get_positions() == []
