"""Regression tests for the trading-engine start guard (BUG-04).

`POST /api/trading/start` used to build and start a new engine unconditionally;
`_live_engine` was only ever overwritten. A second start therefore orphaned the
first engine -- its scheduler thread and broker kept running with no handle left
to stop them, both engines wrote positions/P&L/orders into the same store, and
the orphan's kill switch became unreachable because `_core_risk` was rebound
too. `LiveTradingEngine.start()`'s own `if self._running: return` guard is
per-instance and cannot prevent this.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes


class _FakeEngine:
    def __init__(self, running):
        self._running = running


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_a_second_start_is_refused_while_one_is_running(client, monkeypatch):
    monkeypatch.setattr(routes, "_live_engine", _FakeEngine(running=True))

    resp = client.post("/api/trading/start", json={"broker": "simulated"})

    assert resp.status_code == 409
    assert "already running" in resp.json()["detail"]


def test_a_stopped_engine_does_not_block_a_restart(client, monkeypatch):
    """The guard keys on the engine's running flag, not on the global merely
    being set -- otherwise a stopped engine could never be restarted."""
    monkeypatch.setattr(routes, "_live_engine", _FakeEngine(running=False))

    started = {}

    class _StubEngine:
        def __init__(self, **kwargs):
            self._running = False

        def set_universe(self, codes):
            started["universe"] = codes

        def start(self):
            self._running = True
            started["started"] = True

    import quant_platform.trading.engine as engine_module
    import quant_platform.trading.realtime as realtime_module

    monkeypatch.setattr(engine_module, "LiveTradingEngine", _StubEngine)

    def _no_market():
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(realtime_module, "RealTimeMarket", _no_market)

    resp = client.post("/api/trading/start", json={"broker": "simulated"})

    assert resp.status_code == 200, resp.text
    assert started.get("started") is True


def test_the_orphaned_engine_is_the_one_the_guard_protects(client, monkeypatch):
    """Starting twice must not leave two engines started."""
    monkeypatch.setattr(routes, "_live_engine", _FakeEngine(running=True))

    for _ in range(3):
        assert client.post("/api/trading/start", json={}).status_code == 409
