"""Ops wave 2 regression — the monitor API reports/persists kill-switch state.

S21: the dashboard's emergency action used to be memory-only and its response
carried no persistence status.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import quant_platform.api.monitor as monitor_module
from quant_platform.api.monitor import router
from quant_platform.core.events import EventBus
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr(
        monitor_module, "_core_store", Store(str(tmp_path / "api.db")),
    )
    monkeypatch.setattr(monitor_module, "_core_risk", RiskMonitor())
    monkeypatch.setattr(monitor_module, "_core_bus", EventBus())
    return TestClient(app)


def test_a_dashboard_poll_does_not_disarm_the_switch(client, tmp_path):
    """S21 regression: monitor._get_risk re-attached its own Store object on
    every poll, which reset the in-memory kill switch."""
    monitor = monitor_module._core_risk
    monitor.activate_kill_switch("armed via core API")

    # Production shape: the monitor owns a different Store handle for the
    # same SQLite file than routes/engine do.
    monitor_module._core_store = Store(str(tmp_path / "api.db"))

    resp = client.get("/api/monitor/risk-overview")

    assert resp.status_code == 200
    assert resp.json()["kill_switch"] is True
    assert monitor.kill_switch_active is True


def test_kill_switch_api_persists_and_reports_it(client, tmp_path):
    resp = client.post("/api/monitor/kill-switch", json={
        "activate": True, "reason": "API test",
    })

    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is True
    assert data["persisted"] is True

    store = monitor_module._core_store
    state = store.get_engine_state("risk_state:default")
    assert state is not None
    assert state["kill_switch_active"] is True

    # A fresh monitor restores the armed switch from the same store.
    restarted = RiskMonitor()
    restarted.attach_persistence(store)
    assert restarted.kill_switch_active is True

    overview = client.get("/api/monitor/risk-overview").json()
    assert overview["kill_switch"] is True
    assert overview["persisted"] is True

    deactivated = client.post("/api/monitor/kill-switch", json={
        "activate": False, "reason": "resolved",
    }).json()
    assert deactivated["active"] is False
    assert deactivated["persisted"] is True
    assert store.get_engine_state("risk_state:default")["kill_switch_active"] is False
