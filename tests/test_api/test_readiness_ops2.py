"""Ops wave 2 regressions — /api/readiness dead-man switch and interval guard.

S4: /api/health was unconditional "ok" and was the only probe; a dead engine
thread or stalled scheduler was invisible. Readiness fails 503 on stale
heartbeats. S20: a zero/negative rebalance interval must be rejected with 422.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.utils.metrics import get_metrics

_GAUGES = (
    "last_engine_cycle_timestamp",
    "last_scheduler_tick_timestamp",
    "last_data_success_timestamp",
    "risk_kill_switch_active",
    "trading_equity",
    "trading_positions",
)


class FakeScheduler:
    def __init__(self, market_open: bool):
        self._running = True
        self._market_open = market_open

    def is_market_open(self) -> bool:
        return self._market_open


class FakeEngine:
    def __init__(self, interval: int = 300, market_open: bool = True,
                 cycles_outside: bool = False):
        self._running = True
        self._rebalance_interval = interval
        self._scheduler = FakeScheduler(market_open)
        self._cycles_outside = cycles_outside

    def cycles_outside_session(self) -> bool:
        return self._cycles_outside


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    monkeypatch.setattr(routes, "_live_engine", None)
    metrics = get_metrics()
    for name in _GAUGES:
        metrics.gauge(name).set(0.0)
    return TestClient(app)


def _set_gauges(**values):
    metrics = get_metrics()
    for name, value in values.items():
        metrics.gauge(name).set(value)


class TestReadiness:
    def test_503_when_the_engine_cycle_is_stale(self, client, monkeypatch):
        monkeypatch.setattr(routes, "_live_engine", FakeEngine(interval=300))
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now - 700,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        engine_check = next(
            c for c in data["checks"] if c["name"] == "engine_cycle"
        )
        assert engine_check["ok"] is False

    def test_200_after_a_fresh_tick(self, client, monkeypatch):
        monkeypatch.setattr(routes, "_live_engine", FakeEngine(market_open=True))
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "ready"

    def test_stale_loop_heartbeat_fails_even_outside_session(
        self, client, monkeypatch,
    ):
        """The loop beats every iteration; a stale beat means the loop is dead
        even when the market is legitimately closed."""
        monkeypatch.setattr(
            routes, "_live_engine",
            FakeEngine(market_open=False, cycles_outside=False),
        )
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now - 1800,
            last_scheduler_tick_timestamp=now,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503
        check = next(
            c for c in resp.json()["checks"] if c["name"] == "engine_cycle"
        )
        assert check["ok"] is False

    def test_market_data_is_not_required_outside_the_session(
        self, client, monkeypatch,
    ):
        monkeypatch.setattr(
            routes, "_live_engine",
            FakeEngine(market_open=False, cycles_outside=False),
        )
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now,
            # No fetch since Friday; a live broker does not fetch pre-market.
            last_data_success_timestamp=now - 5 * 86_400,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 200, resp.text
        check = next(
            c for c in resp.json()["checks"] if c["name"] == "market_data"
        )
        assert check["status"] == "market_closed"

    def test_stale_engine_cycle_still_fails_for_paper_mode(self, client, monkeypatch):
        monkeypatch.setattr(
            routes, "_live_engine",
            FakeEngine(market_open=False, cycles_outside=True),
        )
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now - 1800,
            last_scheduler_tick_timestamp=now,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503

    def test_503_when_the_scheduler_tick_is_stale_in_session(
        self, client, monkeypatch,
    ):
        monkeypatch.setattr(routes, "_live_engine", FakeEngine(market_open=True))
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now - 200,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503
        check = next(
            c for c in resp.json()["checks"] if c["name"] == "scheduler_tick"
        )
        assert check["ok"] is False

    def test_stale_scheduler_is_flagged_even_when_the_market_is_closed(
        self, client, monkeypatch,
    ):
        """The scheduler ticks every 30s regardless of market hours; a stale
        tick outside the session means the loop is dead."""
        monkeypatch.setattr(routes, "_live_engine", FakeEngine(market_open=False))
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now - 200,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503
        check = next(
            c for c in resp.json()["checks"] if c["name"] == "scheduler_tick"
        )
        assert check["ok"] is False

    def test_engine_running_without_a_scheduler_is_not_ready(
        self, client, monkeypatch,
    ):
        engine = FakeEngine(market_open=True)
        engine._scheduler = None
        monkeypatch.setattr(routes, "_live_engine", engine)
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now,
            last_data_success_timestamp=now,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503
        check = next(
            c for c in resp.json()["checks"] if c["name"] == "scheduler_tick"
        )
        assert check["ok"] is False

    def test_503_when_market_data_is_older_than_two_trading_days(
        self, client, monkeypatch,
    ):
        monkeypatch.setattr(routes, "_live_engine", FakeEngine(market_open=True))
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now,
            last_data_success_timestamp=now - 5 * 86_400,
        )

        resp = client.get("/api/readiness")

        assert resp.status_code == 503
        check = next(
            c for c in resp.json()["checks"] if c["name"] == "market_data"
        )
        assert check["ok"] is False
        assert check["age_trading_days"] > 2

    def test_a_research_only_server_is_ready(self, client):
        resp = client.get("/api/readiness")

        assert resp.status_code == 200
        assert resp.json()["engine_running"] is False

    def test_health_stays_ok_and_reports_last_seen(self, client, monkeypatch):
        monkeypatch.setattr(routes, "_live_engine", FakeEngine())
        now = time.time()
        _set_gauges(
            last_engine_cycle_timestamp=now,
            last_scheduler_tick_timestamp=now - 10,
            last_data_success_timestamp=now - 20,
        )

        resp = client.get("/api/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["last_engine_cycle"] == pytest.approx(now)
        assert data["last_scheduler_tick"] == pytest.approx(now - 10)
        assert data["last_data_success"] == pytest.approx(now - 20)


class TestTradingStartInterval:
    @pytest.mark.parametrize("bad", [0, -1, 2, 4, "soon", 5.5, True])
    def test_bad_rebalance_intervals_are_rejected_with_422(
        self, client, bad,
    ):
        resp = client.post(
            "/api/trading/start", json={"rebalance_interval": bad},
        )
        assert resp.status_code == 422, resp.text
        detail = str(resp.json()["detail"])
        assert "rebalance_interval" in detail
