"""Singleton races, error sanitization, WS token transport and lifespan.

Companion regressions for audit items A7, A14, A15 and U1.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from quant_platform.api import routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# A14: singleton getters, OMS lock scope, atomic limit replacement
# ---------------------------------------------------------------------------

class TestSingletonRaces:
    def test_concurrent_oms_getters_construct_once(self, monkeypatch):
        import quant_platform.execution.oms as oms_module

        constructed = []
        real_order_manager = oms_module.OrderManager

        class _SlowOrderManager(real_order_manager):
            def __init__(self, *args, **kwargs):
                constructed.append(1)
                time.sleep(0.05)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(routes, "_oms_instance", None)
        monkeypatch.setattr(oms_module, "OrderManager", _SlowOrderManager)

        results = []
        barrier = threading.Barrier(8)

        def get():
            barrier.wait(timeout=5)
            results.append(routes._get_oms())

        threads = [threading.Thread(target=get) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert len(constructed) == 1
        assert len({id(instance) for instance in results}) == 1

    def test_create_order_holds_the_lock_across_check_and_insert(self, monkeypatch):
        import quant_platform.execution.oms as oms_module

        manager = oms_module.OrderManager(initial_cash=1_000_000)
        real_order = oms_module.Order

        class _CheckingOrder(real_order):
            def __init__(self, **kwargs):
                assert manager._lock._is_owned(), (
                    "create_order released the RLock before inserting the order"
                )
                super().__init__(**kwargs)

        monkeypatch.setattr(oms_module, "Order", _CheckingOrder)
        order = manager.create_order("600519", "buy", 100)
        assert order.order_id in manager.orders

    def test_submit_order_holds_the_lock_across_check_and_set(self):
        import quant_platform.execution.oms as oms_module
        from quant_platform.execution.models import OrderStatus

        manager = oms_module.OrderManager(initial_cash=1_000_000)
        order = manager.create_order("600519", "buy", 100)

        class _ProbeOrder:
            def __init__(self):
                self._status = OrderStatus.PENDING
                self.submitted_at = None

            @property
            def status(self):
                return self._status

            @status.setter
            def status(self, value):
                assert manager._lock._is_owned(), (
                    "submit_order check+set is not atomic"
                )
                self._status = value

        manager.orders[order.order_id] = _ProbeOrder()
        manager.submit_order(order.order_id)

    def test_monitor_limit_update_swaps_the_limits_object(self, monkeypatch):
        from fastapi import FastAPI as LocalApp
        from fastapi.testclient import TestClient as LocalClient

        import quant_platform.api.monitor as monitor
        from quant_platform.risk.circuit_breaker import RiskMonitor

        app = LocalApp()
        app.include_router(monitor.router)
        risk = RiskMonitor()
        monkeypatch.setattr(monitor, "_core_risk", risk)
        old_limits = risk.limits

        resp = LocalClient(app).post(
            "/api/monitor/config", json={"max_position_pct": 0.08},
        )

        assert resp.status_code == 200
        assert risk.limits is not old_limits, "limits were mutated in place"
        assert risk.limits.max_single_position_pct == 0.08
        assert old_limits.max_single_position_pct == 0.05


# ---------------------------------------------------------------------------
# A15: generic 5xx bodies, stable codes, short monitor reasons
# ---------------------------------------------------------------------------

class TestErrorSanitization:
    def test_screen_does_not_echo_exception_text(self, client, monkeypatch):
        secret = "SECRETPATH /etc/passwd SQL SELECT * FROM users"

        def fail(*args, **kwargs):
            raise RuntimeError(secret)

        monkeypatch.setattr(routes, "_load_data", fail)
        resp = client.post(
            "/api/screen",
            json={
                "rules": [
                    {"factor": "pe_ratio", "operator": "lt", "value": 30},
                ],
            },
        )

        assert resp.status_code == 503
        body = resp.json()["detail"]
        assert "Data load failed" in body
        assert "SECRETPATH" not in body
        assert "SELECT" not in body
        assert resp.headers.get("X-Error-Code") == "screen_data_unavailable"

    def test_market_endpoint_does_not_echo_exception_text(self, client, monkeypatch):
        monkeypatch.setattr(
            routes, "_get_rt_market",
            lambda: (_ for _ in ()).throw(
                RuntimeError("TOPSECRET /srv/db.sqlite"),
            ),
        )
        resp = client.get("/api/market/snapshot")

        assert resp.status_code == 500
        body = resp.json()["detail"]
        assert "TOPSECRET" not in body
        assert "db.sqlite" not in body
        assert resp.headers.get("X-Error-Code") == "market_data_failed"

    def test_monitor_reason_does_not_echo_exception_text(self, monkeypatch):
        import quant_platform.api.monitor as monitor

        class _FailingStore:
            def get_positions(self):
                raise RuntimeError("SECRETPATH /srv/trading.db")

            def get_pnl_history(self, days=30):
                return []

        monkeypatch.setattr(monitor, "_core_store", _FailingStore())
        monkeypatch.setattr(monitor, "_get_risk", lambda: None)

        result = asyncio.run(monitor.get_risk_overview())

        assert "SECRETPATH" not in result.reason
        assert result.sector_concentration == {}

    def test_baostock_health_returns_a_short_reason(self, client, monkeypatch):
        import sys
        import types

        fake_bs = types.ModuleType("baostock")

        def fail():
            raise RuntimeError("SECRETPATH /usr/lib/secret")

        fake_bs.login = fail
        monkeypatch.setitem(sys.modules, "baostock", fake_bs)

        data = client.get("/api/market/baostock/health").json()

        assert data["status"] == "error"
        assert data["code"] == "baostock_unavailable"
        assert "SECRETPATH" not in json.dumps(data)


# ---------------------------------------------------------------------------
# U1: WebSocket token via subprotocol, query fallback preserved
# ---------------------------------------------------------------------------

class TestWebSocketTokenTransport:
    def test_bearer_subprotocol_is_accepted(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_TOKEN", "secret")
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)

        with client.websocket_connect(
            "/api/ws", subprotocols=["bearer.secret"],
        ) as ws:
            ws.send_text("ping")

    def test_wrong_subprotocol_token_is_refused(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_TOKEN", "secret")
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/api/ws", subprotocols=["bearer.wrong"],
            ):
                pass

    def test_query_token_fallback_still_works(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_TOKEN", "secret")
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)

        with client.websocket_connect("/api/ws?token=secret") as ws:
            ws.send_text("ping")


# ---------------------------------------------------------------------------
# A7: lifespan shutdown / re-creatable executors
# ---------------------------------------------------------------------------

class TestLifespan:
    def test_executors_are_recreated_on_the_next_startup(self):
        from quant_platform.app import create_app

        app = create_app(serve_frontend=False)

        with TestClient(app) as client:
            first_pipeline = routes._get_pipeline_executor()
            first_analytics = routes._get_analytics_executor()
            assert client.get("/api/health").status_code == 200

        assert routes._pipeline_executor is None
        assert routes._analytics_executor is None

        with TestClient(app):
            second_pipeline = routes._get_pipeline_executor()
            second_analytics = routes._get_analytics_executor()

        assert second_pipeline is not first_pipeline
        assert second_analytics is not first_analytics

    def test_shutdown_stops_the_live_engine(self):
        from quant_platform.app import create_app

        class _Engine:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        engine = _Engine()
        old_engine = routes._live_engine
        routes._live_engine = engine
        try:
            app = create_app(serve_frontend=False)
            with TestClient(app):
                pass
            assert engine.stopped is True
            assert routes._live_engine is None
        finally:
            routes._live_engine = old_engine

    def test_shutdown_closes_the_store_and_it_remains_reusable(self):
        from quant_platform.app import create_app

        app = create_app(serve_frontend=False)
        with TestClient(app):
            pass
        assert routes._core_store.get_stats()["backend"] == "sqlite"


# ---------------------------------------------------------------------------
# A17/A18 companion assertions (full tests live in their own modules)
# ---------------------------------------------------------------------------

def test_engine_bulk_position_writes_are_used():
    import inspect

    from quant_platform.trading import engine as engine_module

    source = inspect.getsource(engine_module.LiveTradingEngine._sync_store_book)
    assert "save_positions(" in source
    assert "save_position(" not in source.replace("save_positions(", "")


def test_exporter_uses_the_dedicated_export_queries():
    import inspect

    from quant_platform.compliance import exporter as exporter_module

    trade_source = inspect.getsource(
        exporter_module.ComplianceExporter.export_trade_log,
    )
    order_source = inspect.getsource(
        exporter_module.ComplianceExporter.export_order_log,
    )
    assert "get_trades_for_export" in trade_source
    assert "get_orders_for_export" in order_source
