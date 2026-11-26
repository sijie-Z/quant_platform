"""Ops wave 2 regressions — durable risk state (S21).

Risk level, breaches and the kill switch lived only in memory: a restart
cleared a manually armed kill switch and every breach, and the monitor API's
emergency action left no audit record.
"""

from __future__ import annotations

from quant_platform.core.audit import AuditLog
from quant_platform.core.events import EventBus
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskLevel, RiskMonitor


def _monitor(store, tenant="default"):
    audit = AuditLog(store, EventBus(), tenant_id=tenant)
    return RiskMonitor(store=store, tenant_id=tenant, audit=audit)


def test_armed_kill_switch_survives_a_restart_and_blocks_orders(tmp_path):
    store = Store(str(tmp_path / "risk.db"))
    first = _monitor(store)
    first.activate_kill_switch("manual emergency", actor="operator:alice")

    restarted = RiskMonitor()
    restarted.attach_persistence(store)

    assert restarted.kill_switch_active is True
    approved, _breaches = restarted.check_pre_trade({
        "ticker": "600519", "side": "buy", "quantity": 100, "price": 10.0,
    })
    assert approved is False
    assert restarted.get_status()["persisted"] is True


def test_deactivation_is_persisted_too(tmp_path):
    store = Store(str(tmp_path / "risk.db"))
    first = _monitor(store)
    first.activate_kill_switch("manual")
    first.deactivate_kill_switch(reason="resolved", actor="operator:bob")

    restarted = RiskMonitor()
    restarted.attach_persistence(store)

    assert restarted.kill_switch_active is False


def test_kill_switch_activation_and_deactivation_are_audited(tmp_path):
    store = Store(str(tmp_path / "risk.db"))
    monitor = _monitor(store, tenant="acme")

    monitor.activate_kill_switch("drawdown emergency", actor="operator:alice")
    monitor.deactivate_kill_switch(reason="market stable", actor="operator:bob")

    events = store.get_events(topic="audit.risk_breach", tenant_id="acme")
    details = [event["data"]["details"] for event in events]
    activated = [
        row for row in details
        if row.get("breach_type") == "kill_switch_activated"
    ]
    deactivated = [
        row for row in details
        if row.get("breach_type") == "kill_switch_deactivated"
    ]
    assert activated and activated[0]["actor"] == "operator:alice"
    assert "drawdown emergency" in activated[0]["reason"]
    assert deactivated and deactivated[0]["actor"] == "operator:bob"


def test_risk_level_and_breaches_survive_a_restart(tmp_path):
    store = Store(str(tmp_path / "risk.db"))
    first = _monitor(store)
    first.update_portfolio_state(
        portfolio_value=1_000_000.0, daily_pnl=-40_000.0,
        positions={}, sector_weights={},
    )
    assert first.risk_level == RiskLevel.RED

    restarted = RiskMonitor()
    restarted.attach_persistence(store)

    assert restarted.risk_level == RiskLevel.RED
    assert restarted.portfolio_value == 1_000_000.0
    assert any(
        breach.breach_type.value == "daily_loss"
        for breach in restarted.breaches
    )


def test_state_is_tenant_scoped(tmp_path):
    store = Store(str(tmp_path / "risk.db"))
    tenant_a = _monitor(store, tenant="tenant-a")
    tenant_a.activate_kill_switch("a only")

    tenant_b = RiskMonitor()
    tenant_b.attach_persistence(store, tenant_id="tenant-b")

    assert tenant_b.kill_switch_active is False
    # Tenant B has no persisted row yet, so nothing is durable for it.
    assert tenant_b.get_status()["persisted"] is False


def test_missing_store_means_not_persisted(tmp_path):
    monitor = RiskMonitor()
    assert monitor.get_status()["persisted"] is False


def test_a_failed_persist_is_not_reported_as_persisted(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "risk.db"))
    monitor = RiskMonitor()
    monitor.attach_persistence(store, restore=False)

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_engine_state", _boom)

    ok = monitor.activate_kill_switch("emergency")

    assert ok is False
    status = monitor.get_status()
    assert status["persisted"] is False
    assert status["persist_error"] is True
    assert store.get_engine_state("risk_state:default") is None


def test_switching_tenant_does_not_inherit_the_previous_state(tmp_path):
    store_a = Store(str(tmp_path / "a.db"))
    store_b = Store(str(tmp_path / "b.db"))
    monitor = RiskMonitor()
    monitor.attach_persistence(store_a, tenant_id="tenant-a", restore=False)
    monitor.activate_kill_switch("a only")

    monitor.attach_persistence(store_b, tenant_id="tenant-b", restore=True)

    assert monitor.kill_switch_active is False
    assert monitor.breaches == []
    assert monitor.risk_level == RiskLevel.GREEN


def test_a_second_store_on_the_same_file_keeps_the_armed_switch(tmp_path):
    """Both API modules may lazily build their own Store handle for one DB; a
    re-attach with the same tenant must not reset live risk state."""
    db = str(tmp_path / "risk.db")
    store_a = Store(db)
    monitor = RiskMonitor()
    monitor.attach_persistence(store_a, restore=False)
    monitor.activate_kill_switch("armed")

    store_b = Store(db)
    monitor.attach_persistence(store_b, tenant_id="default", restore=False)

    assert monitor.kill_switch_active is True
    assert monitor.breaches, "breach history must survive a store-handle swap"


def test_a_kept_armed_switch_is_not_reported_as_persisted(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "risk.db"))
    monitor = RiskMonitor()
    monitor.attach_persistence(store, restore=False)
    store.save_engine_state(
        {
            "version": 1, "tenant_id": "default", "risk_level": "green",
            "kill_switch_active": False, "breaches": [],
            "portfolio_value": 0.0, "peak_value": 0.0, "daily_pnl": 0.0,
            "updated_at": "2026-10-20T09:00:00",
        },
        "risk_state:default",
    )
    monitor.attach_persistence(store, restore=True)
    original_save = store.save_engine_state

    def _boom(*_args, **_kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(store, "save_engine_state", _boom)
    assert monitor.activate_kill_switch("armed while store is down") is False

    # The engine-restart path: keep the armed switch, but do not pretend the
    # older disarmed row recorded it.
    monitor.attach_persistence(store, restore=True)
    assert monitor.kill_switch_active is True
    status = monitor.get_status()
    assert status["persisted"] is False
    assert status["persist_error"] is True
    assert store.get_engine_state("risk_state:default")["kill_switch_active"] is False

    # Once the store recovers, the next persist writes the armed state.
    monkeypatch.setattr(store, "save_engine_state", original_save)
    assert monitor._persist_state() is True
    assert monitor.get_status()["persisted"] is True
    assert store.get_engine_state("risk_state:default")["kill_switch_active"] is True


def test_automatic_drawdown_arming_is_audited(tmp_path):
    store = Store(str(tmp_path / "risk.db"))
    monitor = _monitor(store)
    monitor.peak_value = 1_000_000.0

    monitor.update_portfolio_state(
        portfolio_value=700_000.0, daily_pnl=0.0,
        positions={}, sector_weights={},
    )

    assert monitor.kill_switch_active is True
    events = store.get_events(topic="audit.risk_breach", tenant_id="default")
    details = [event["data"]["details"] for event in events]
    auto = [
        row for row in details
        if row.get("breach_type") == "kill_switch_activated"
        and row.get("actor") == "risk_monitor:auto"
    ]
    assert auto, "automatic kill-switch arming must carry an audit record"
