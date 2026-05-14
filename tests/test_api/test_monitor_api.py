"""Tests for monitor API endpoints — aggregation layer for dashboard."""

import pytest
from fastapi.testclient import TestClient

from quant_platform.api.monitor import (
    CapacityGaugeResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    KillSwitchRequest,
    KillSwitchResponse,
    RiskOverviewResponse,
    FactorStatusResponse,
    TCASummaryResponse,
    router,
)
from quant_platform.core.store import Store
from fastapi import FastAPI


@pytest.fixture
def app(tmp_path):
    """Create a minimal FastAPI app with just the monitor router."""
    _app = FastAPI()
    _app.include_router(router)

    # Patch the store singleton to use a temp DB
    import quant_platform.api.monitor as mod
    mod._core_store = Store(str(tmp_path / "test.db"))
    mod._core_risk = None
    mod._core_bus = None

    return _app


@pytest.fixture
def client(app):
    return TestClient(app)


# ── Risk Overview ──


class TestRiskOverview:
    def test_returns_200(self, client):
        resp = client.get("/api/monitor/risk-overview")
        assert resp.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/api/monitor/risk-overview").json()
        assert "factor_exposures" in data
        assert "sector_concentration" in data
        assert "current_drawdown" in data
        assert "volatility" in data
        assert "var_95" in data
        assert "cvar_95" in data
        assert "risk_level" in data
        assert "portfolio_value" in data
        assert "daily_pnl" in data
        assert "n_positions" in data

    def test_risk_level_default(self, client):
        data = client.get("/api/monitor/risk-overview").json()
        assert data["risk_level"].upper() in ("GREEN", "YELLOW", "ORANGE", "RED", "KILL")

    def test_with_positions(self, app, client):
        store = app._store if hasattr(app, '_store') else None
        import quant_platform.api.monitor as mod
        store = mod._core_store
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 1800, "market_value": 180000})
        store.save_position({"code": "000001", "quantity": 200, "avg_cost": 15, "market_value": 3000})
        data = client.get("/api/monitor/risk-overview").json()
        assert len(data["sector_concentration"]) > 0

    def test_with_pnl_history(self, app, client):
        import quant_platform.api.monitor as mod
        store = mod._core_store
        for i in range(20):
            store.save_pnl_snapshot({
                "total_equity": 10_000_000 + i * 10_000,
                "cash": 5_000_000,
                "market_value": 5_000_000 + i * 10_000,
            })
        data = client.get("/api/monitor/risk-overview").json()
        # Should have computed VaR with enough data
        assert isinstance(data["var_95"], float)


# ── TCA Summary ──


class TestTCASummary:
    def test_returns_200(self, client):
        resp = client.get("/api/monitor/tca-summary")
        assert resp.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/api/monitor/tca-summary").json()
        assert "n_orders" in data
        assert "mean_is_bps" in data
        assert "mean_delay_bps" in data
        assert "mean_impact_bps" in data
        assert "daily_trend" in data
        assert "cost_breakdown" in data
        assert "by_ticker" in data

    def test_with_trades(self, app, client):
        import quant_platform.api.monitor as mod
        store = mod._core_store
        for i in range(5):
            store.save_order({
                "order_id": f"o{i}", "code": f"60000{i}", "side": "buy",
                "quantity": 100, "price": 10 + i, "status": "filled",
                "created_at": "2024-07-15T10:00:00",
            })
            store.save_trade({
                "trade_id": f"t{i}", "order_id": f"o{i}", "code": f"60000{i}",
                "side": "buy", "quantity": 100, "price": 10 + i,
                "executed_at": "2024-07-15T10:00:00",
            })
        data = client.get("/api/monitor/tca-summary").json()
        assert data["n_orders"] == 5


# ── Factor Status ──


class TestFactorStatus:
    def test_returns_200(self, client):
        resp = client.get("/api/monitor/factor-status")
        assert resp.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/api/monitor/factor-status").json()
        assert "factors" in data
        assert "rolling_ic" in data
        assert "ic_dates" in data
        assert "attribution" in data
        assert "decay_alerts" in data
        assert "disabled_factors" in data

    def test_factors_have_required_fields(self, client):
        data = client.get("/api/monitor/factor-status").json()
        for f in data["factors"]:
            assert "name" in f
            assert "current_ic" in f
            assert "icir" in f
            assert "trend" in f
            assert "alert" in f

    def test_rolling_ic_length(self, client):
        data = client.get("/api/monitor/factor-status").json()
        if data["rolling_ic"]:
            for name, series in data["rolling_ic"].items():
                assert len(series) == 60

    def test_ic_dates_length(self, client):
        data = client.get("/api/monitor/factor-status").json()
        # Either 60 dates (with data) or 0 (no signals)
        assert len(data["ic_dates"]) in (0, 60)


# ── Capacity Gauge ──


class TestCapacityGauge:
    def test_returns_200(self, client):
        resp = client.get("/api/monitor/capacity-gauge")
        assert resp.status_code == 200

    def test_response_structure(self, client):
        data = client.get("/api/monitor/capacity-gauge").json()
        assert "current_aum" in data
        assert "capacity_aum" in data
        assert "usage_pct" in data
        assert "participation_rate" in data
        assert "aum_curve" in data

    def test_aum_curve_structure(self, client):
        data = client.get("/api/monitor/capacity-gauge").json()
        for point in data["aum_curve"]:
            assert "aum" in point
            assert "sharpe" in point
            assert "return" in point

    def test_with_positions(self, app, client):
        import quant_platform.api.monitor as mod
        store = mod._core_store
        store.save_pnl_snapshot({
            "total_equity": 15_000_000, "cash": 5_000_000,
            "market_value": 10_000_000,
        })
        data = client.get("/api/monitor/capacity-gauge").json()
        assert data["current_aum"] == 15_000_000


# ── Config Update ──


class TestConfigUpdate:
    def test_returns_200_with_valid_config(self, app, client):
        # Patch risk monitor
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/config", json={
            "max_position_pct": 0.08,
            "max_sector_pct": 0.35,
        })
        assert resp.status_code == 200

    def test_response_structure(self, app, client):
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        data = client.post("/api/monitor/config", json={
            "max_position_pct": 0.08,
        }).json()
        assert "updated" in data
        assert "limits" in data
        assert "max_position_pct" in data["updated"]

    def test_rejects_position_over_20_pct(self, app, client):
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/config", json={
            "max_position_pct": 0.25,
        })
        assert resp.status_code == 422  # Validation error

    def test_rejects_drawdown_over_30_pct(self, app, client):
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/config", json={
            "max_drawdown_pct": 0.35,
        })
        assert resp.status_code == 422

    def test_returns_503_without_risk_monitor(self, app, client, monkeypatch):
        import quant_platform.api.monitor as mod
        monkeypatch.setattr(mod, "_get_risk", lambda: None)

        resp = client.post("/api/monitor/config", json={
            "max_position_pct": 0.05,
        })
        assert resp.status_code == 503


# ── Kill Switch ──


class TestKillSwitch:
    def test_activate_returns_200(self, app, client):
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/kill-switch", json={
            "activate": True,
            "reason": "Test activation",
        })
        assert resp.status_code == 200

    def test_activate_response(self, app, client):
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        data = client.post("/api/monitor/kill-switch", json={
            "activate": True, "reason": "Test",
        }).json()
        assert data["active"] is True
        assert "ACTIVATED" in data["message"]

    def test_deactivate_response(self, app, client):
        from quant_platform.risk.circuit_breaker import RiskMonitor
        import quant_platform.api.monitor as mod
        mod._core_risk = RiskMonitor()

        # First activate
        client.post("/api/monitor/kill-switch", json={"activate": True, "reason": "Test"})
        # Then deactivate
        data = client.post("/api/monitor/kill-switch", json={
            "activate": False, "reason": "Test",
        }).json()
        assert data["active"] is False
        assert "DEACTIVATED" in data["message"]

    def test_returns_503_without_risk_monitor(self, app, client, monkeypatch):
        import quant_platform.api.monitor as mod
        monkeypatch.setattr(mod, "_get_risk", lambda: None)

        resp = client.post("/api/monitor/kill-switch", json={
            "activate": True, "reason": "Test",
        })
        assert resp.status_code == 503


# ── Pydantic Models ──


class TestResponseModels:
    def test_risk_overview_defaults(self):
        r = RiskOverviewResponse()
        assert r.risk_level == "GREEN"
        assert r.portfolio_value == 0.0

    def test_tca_summary_defaults(self):
        t = TCASummaryResponse()
        assert t.n_orders == 0
        assert t.daily_trend == []

    def test_factor_status_defaults(self):
        f = FactorStatusResponse()
        assert f.factors == []
        assert f.disabled_factors == []

    def test_capacity_gauge_defaults(self):
        c = CapacityGaugeResponse()
        assert c.usage_pct == 0.0
        assert c.aum_curve == []

    def test_config_update_request_validation(self):
        with pytest.raises(Exception):
            ConfigUpdateRequest(max_position_pct=0.25)

    def test_kill_switch_defaults(self):
        k = KillSwitchRequest()
        assert k.activate is True
        assert "Manual" in k.reason


# ── Helper ──


class TestCodeToSector:
    def test_sector_mapping(self):
        from quant_platform.api.monitor import _code_to_sector
        assert _code_to_sector("600519") == "沪市主板"
        assert _code_to_sector("688001") == "科创板"
        assert _code_to_sector("000001") == "深市主板"
        assert _code_to_sector("002001") == "中小板"
        assert _code_to_sector("300001") == "创业板"
        assert _code_to_sector("999999") == "其他"
        assert _code_to_sector("") == "Unknown"
