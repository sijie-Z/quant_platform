"""Tests for monitor API endpoints — aggregation layer for dashboard."""

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api.monitor import (
    CapacityGaugeResponse,
    ConfigUpdateRequest,
    FactorStatusResponse,
    KillSwitchRequest,
    RiskOverviewResponse,
    TCASummaryResponse,
    router,
)
from quant_platform.core.store import Store
from tests.test_api.research_helpers import make_context


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


@pytest.fixture
def short_context(monkeypatch):
    """A 50-session context: too short for rolling IC and without volume."""
    import quant_platform.api.monitor as mod

    index = pd.bdate_range("2024-01-01", periods=50)
    assets = [f"{i:06d}.SZ" for i in range(20)]
    prices = pd.DataFrame(100.0, index=index, columns=assets)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    context = make_context(
        prices, returns,
        {"momentum": prices.pct_change(fill_method=None)},
        turnover=None,
        fin_unstacked=None,
    )

    async def _context(start_date=None, end_date=None):
        return context

    monkeypatch.setattr(mod, "_research_context_or_503", _context)
    return context


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

    def test_no_synthetic_factor_exposures(self, app, client):
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor

        risk = RiskMonitor()
        risk.update_portfolio_state(
            portfolio_value=1_000_000,
            daily_pnl=0,
            positions={"600519": {"value": 100_000, "weight": 0.1, "sector": "白酒"}},
            sector_weights={"白酒": 0.1},
        )
        mod._core_risk = risk
        store = mod._core_store
        store.save_position({"code": "600519", "quantity": 100, "avg_cost": 1800, "market_value": 180000})
        data = client.get("/api/monitor/risk-overview").json()
        assert data["factor_exposures"] == {}

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

    def test_with_trades_computes_shortfall_from_the_order_price(self, app, client):
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
        assert data["n_trades"] == 5
        assert data["available"] is True
        assert data["n_analyzed"] == 5
        assert data["mean_is_bps"] == 0.0  # executed at the reference price

    def test_trade_without_an_order_has_no_invented_metrics(self, app, client):
        import quant_platform.api.monitor as mod
        store = mod._core_store
        store.save_trade({
            "trade_id": "t1", "order_id": "o1", "code": "600000",
            "side": "buy", "quantity": 100, "price": 10,
            "executed_at": "2024-07-15T10:00:00",
        })
        data = client.get("/api/monitor/tca-summary").json()
        assert data["available"] is False
        assert data["daily_trend"] == []
        assert data["cost_breakdown"] == []
        assert data["mean_is_bps"] == 0
        assert "reference price" in data["reason"]


# ── Factor Status ──


class TestFactorStatus:
    def test_returns_200(self, client, short_context):
        resp = client.get("/api/monitor/factor-status")
        assert resp.status_code == 200

    def test_response_structure(self, client, short_context):
        data = client.get("/api/monitor/factor-status").json()
        assert "factors" in data
        assert "rolling_ic" in data
        assert "ic_dates" in data
        assert "attribution" in data
        assert "decay_alerts" in data
        assert "disabled_factors" in data

    def test_insufficient_history_explains_the_requirement(self, client, short_context):
        data = client.get("/api/monitor/factor-status").json()
        assert data["available"] is False
        assert data["factors"] == []
        assert data["required_sessions"] > 50
        assert "sessions" in data["reason"]
        assert "at least" in data["reason"]

    def test_stored_signals_do_not_become_factor_stats(self, client, short_context):
        """Factor status is computed from factor panels, never from the
        factor_values blobs on stored signals."""
        import quant_platform.api.monitor as mod
        store = mod._core_store
        store.save_signal({
            "signal_id": "s1", "code": "600000", "direction": "buy",
            "strength": 0.5, "factor_values": {"momentum_3m": 0.1},
            "generated_at": "2024-07-15T10:00:00",
        })
        data = client.get("/api/monitor/factor-status").json()
        assert data["factors"] == []
        assert data["rolling_ic"] == {}
        assert data["attribution"] == []


# ── Capacity Gauge ──


class TestCapacityGauge:
    def test_returns_200(self, client, short_context):
        resp = client.get("/api/monitor/capacity-gauge")
        assert resp.status_code == 200

    def test_response_structure(self, client, short_context):
        data = client.get("/api/monitor/capacity-gauge").json()
        assert "current_aum" in data
        assert "capacity_aum" in data
        assert "usage_pct" in data
        assert "participation_rate" in data
        assert "aum_curve" in data

    def test_aum_curve_structure(self, client, short_context):
        data = client.get("/api/monitor/capacity-gauge").json()
        for point in data["aum_curve"]:
            assert "aum" in point
            assert "sharpe" in point
            assert "return" in point

    def test_missing_volume_panel_is_explained(self, client, short_context):
        """No turnover/market-cap panel means no ADV, so no gauge."""
        data = client.get("/api/monitor/capacity-gauge").json()
        assert data["available"] is False
        assert "no volume panel" in data["reason"]
        assert data["aum_curve"] == []
        assert data["capacity_aum"] == 0
        assert data["usage_pct"] == 0

    def test_current_aum_still_comes_from_the_live_book(self, app, client, short_context):
        import quant_platform.api.monitor as mod
        store = mod._core_store
        store.save_pnl_snapshot({
            "total_equity": 15_000_000, "cash": 5_000_000,
            "market_value": 10_000_000,
        })
        data = client.get("/api/monitor/capacity-gauge").json()
        assert data["current_aum"] == 15_000_000
        assert data["aum_source"] == "pnl_history"
        assert data["usage_available"] is True


# ── Config Update ──


class TestConfigUpdate:
    def test_returns_200_with_valid_config(self, app, client):
        # Patch risk monitor
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/config", json={
            "max_position_pct": 0.08,
            "max_sector_pct": 0.35,
        })
        assert resp.status_code == 200

    def test_response_structure(self, app, client):
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
        mod._core_risk = RiskMonitor()

        data = client.post("/api/monitor/config", json={
            "max_position_pct": 0.08,
        }).json()
        assert "updated" in data
        assert "limits" in data
        assert "max_position_pct" in data["updated"]

    def test_rejects_position_over_20_pct(self, app, client):
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/config", json={
            "max_position_pct": 0.25,
        })
        assert resp.status_code == 422  # Validation error

    def test_rejects_drawdown_over_30_pct(self, app, client):
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
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
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
        mod._core_risk = RiskMonitor()

        resp = client.post("/api/monitor/kill-switch", json={
            "activate": True,
            "reason": "Test activation",
        })
        assert resp.status_code == 200

    def test_activate_response(self, app, client):
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
        mod._core_risk = RiskMonitor()

        data = client.post("/api/monitor/kill-switch", json={
            "activate": True, "reason": "Test",
        }).json()
        assert data["active"] is True
        assert "ACTIVATED" in data["message"]

    def test_deactivate_response(self, app, client):
        import quant_platform.api.monitor as mod
        from quant_platform.risk.circuit_breaker import RiskMonitor
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
        # An unread risk state is not a green all-clear.
        assert r.available is False
        assert r.risk_level == "UNKNOWN"
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


class TestUnavailableMetricsAreExplicit:
    """An uncomputed metric must say so instead of returning zeros."""

    def test_tca_is_marked_unavailable_without_trade_data(self, client):
        data = client.get("/api/monitor/tca-summary").json()
        assert data["available"] is False
        assert data["reason"]

    def test_factor_status_is_marked_unavailable_without_history(
        self, client, short_context,
    ):
        data = client.get("/api/monitor/factor-status").json()
        assert data["available"] is False
        assert data["reason"]

    def test_capacity_is_marked_unavailable_without_volume(
        self, client, short_context,
    ):
        data = client.get("/api/monitor/capacity-gauge").json()
        assert data["available"] is False
        assert data["reason"]

    def test_risk_overview_explains_empty_factor_exposures(self, client):
        data = client.get("/api/monitor/risk-overview").json()
        assert data["factor_exposures"] == {}
        assert data["factor_exposures_available"] is False
        assert data["factor_exposures_reason"]
