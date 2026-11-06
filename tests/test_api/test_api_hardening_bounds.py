"""Response bounds and bounded-pool regressions (audit items A11-A13)."""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.core.store import Store


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# A11: response bounds
# ---------------------------------------------------------------------------

class TestResponseBounds:
    def test_sessions_limit_is_clamped(self, client, tmp_path, monkeypatch):
        store = Store(str(tmp_path / "sessions.db"))
        for i in range(120):
            store.save_session({
                "session_id": f"s{i}",
                "started_at": f"2024-01-{i % 28 + 1:02d}T00:00:00",
            })
        monkeypatch.setattr(routes, "_core_store", store)

        assert client.get(
            "/api/core/store/sessions?limit=-1",
        ).json()["sessions"] == []
        assert client.get(
            "/api/core/store/sessions?limit=0",
        ).json()["sessions"] == []
        huge = client.get(
            "/api/core/store/sessions?limit=1000000000",
        ).json()
        assert len(huge["sessions"]) == 120

    def test_events_limit_is_clamped_before_reaching_the_bus(
        self, client, monkeypatch,
    ):
        seen = {}

        class _Bus:
            def get_history(self, topic="", limit=50):
                seen["limit"] = limit
                return []

        monkeypatch.setattr(routes, "_core_bus", _Bus())

        client.get("/api/core/events?limit=1000000000")
        assert seen["limit"] == 1000
        client.get("/api/core/events?limit=-1")
        assert seen["limit"] == 0

    @pytest.mark.parametrize("query", ["limit=0", "limit=-5", "limit=5001"])
    def test_l2_tick_limit_is_validated(self, client, monkeypatch, query):
        class _Provider:
            is_running = True

        monkeypatch.setattr(routes, "_get_l2_provider", lambda: _Provider())
        resp = client.get(f"/api/l2/ticks/600519?{query}")
        assert resp.status_code == 422

    def test_l2_tick_limit_within_range_is_forwarded(self, client, monkeypatch):
        seen = {}

        class _Provider:
            is_running = True

            def get_ticks(self, code, limit=100):
                seen["limit"] = limit
                return []

        monkeypatch.setattr(routes, "_get_l2_provider", lambda: _Provider())
        resp = client.get("/api/l2/ticks/600519?limit=5000")

        assert resp.status_code == 200
        assert seen["limit"] == 5000

    def test_oms_blotter_limit_is_bounded(self, client, monkeypatch):
        seen = {}

        class _OMS:
            def get_order_blotter(self, tenant_id="", limit=500):
                seen["limit"] = limit
                return [{"order_id": "o1"}]

        monkeypatch.setattr(routes, "_get_oms", lambda: _OMS())

        assert client.get("/api/oms/blotter?limit=0").status_code == 422
        assert client.get("/api/oms/blotter?limit=2001").status_code == 422
        resp = client.get("/api/oms/blotter?limit=5")
        assert resp.status_code == 200
        assert seen["limit"] == 5

    def test_ml_prediction_matrix_is_truncated_to_recent_dates(self):
        dates = pd.bdate_range("2010-01-01", periods=5000)
        assets = [f"S{i:03d}" for i in range(500)]
        signal = pd.DataFrame(1.0, index=dates, columns=assets)

        truncated = routes._truncate_signal_matrix(signal)

        assert len(truncated.index) * len(truncated.columns) <= (
            routes._MAX_ML_MATRIX_CELLS
        )
        assert truncated.index[-1] == dates[-1]

    def test_chart_series_are_bounded(self):
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2000-01-03", periods=5000)
        series = pd.Series(rng.normal(0, 0.01, 5000), index=dates)

        chart = routes._build_chart_data(series)

        assert len(chart.dates) <= routes._MAX_CHART_POINTS + 1
        assert len(chart.equity) == len(chart.dates)
        assert chart.dates[-1] == str(dates[-1].date())
        assert len(chart.drawdown) <= routes._MAX_CHART_POINTS + 1

    def test_store_session_limit_clamps_negative_values(self, tmp_path):
        store = Store(str(tmp_path / "clamp.db"))
        store.save_session({
            "session_id": "s1", "started_at": "2024-01-01T00:00:00",
        })
        assert store.get_sessions(limit=-1) == []
        assert len(store.get_sessions(limit=1_000_000_000)) == 1


# ---------------------------------------------------------------------------
# A12: full-pipeline compute endpoints are bounded
# ---------------------------------------------------------------------------

class TestBoundedComputeEndpoints:
    @staticmethod
    def _hold_all():
        acquired = 0
        while routes._COMPUTE_SLOTS.acquire(blocking=False):
            acquired += 1
            if acquired >= 2:
                break
        return acquired

    def test_screen_returns_429_when_compute_is_saturated(self, client):
        acquired = self._hold_all()
        try:
            resp = client.post(
                "/api/screen",
                json={
                    "rules": [
                        {"factor": "pe_ratio", "operator": "lt", "value": 30},
                    ],
                },
            )
        finally:
            for _ in range(acquired):
                routes._COMPUTE_SLOTS.release()

        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "10"

    def test_ml_predict_returns_429_when_compute_is_saturated(self, client):
        acquired = self._hold_all()
        try:
            resp = client.post("/api/ml/predict", json={})
        finally:
            for _ in range(acquired):
                routes._COMPUTE_SLOTS.release()

        assert resp.status_code == 429

    def test_slot_is_released_after_an_exception(self):
        with pytest.raises(ZeroDivisionError):
            routes._run_bounded_compute(lambda: 1 / 0)

        acquired = self._hold_all()
        try:
            assert acquired == 2
        finally:
            for _ in range(acquired):
                routes._COMPUTE_SLOTS.release()


# ---------------------------------------------------------------------------
# A13: network-bound endpoints are bounded
# ---------------------------------------------------------------------------

class _FakeRS:
    def __init__(self, row):
        self._row = row
        self._done = False
        self.error_code = "0"

    def next(self):
        if self._done:
            return False
        self._done = True
        return True

    def get_row_data(self):
        return self._row


class TestNetworkBounds:
    def test_live_portfolio_refresh_is_single_flight(self, monkeypatch):
        monkeypatch.setattr(routes, "_holdings_data", {
            "holdings": [{"code": "600519", "hold_vol": 100}],
        })
        assert routes._live_portfolio_slots.acquire(blocking=False)
        try:
            with pytest.raises(HTTPException) as exc:
                routes.get_live_portfolio()
        finally:
            routes._live_portfolio_slots.release()

        assert exc.value.status_code == 503
        assert exc.value.headers.get("Retry-After") == "10"

    def test_live_portfolio_stops_at_the_deadline_and_reports_partial(
        self, monkeypatch,
    ):
        def fake_query(*args, **kwargs):
            time.sleep(0.03)
            return _FakeRS(["2026-10-20", "10.0", "9.5"])

        fake_bs = types.ModuleType("baostock")
        fake_bs.login = lambda: SimpleNamespace(error_code="0")
        fake_bs.logout = lambda: None
        fake_bs.query_history_k_data_plus = fake_query
        monkeypatch.setitem(sys.modules, "baostock", fake_bs)

        holdings = [
            {"code": f"60000{i}", "hold_vol": 100} for i in range(5)
        ]
        monkeypatch.setattr(routes, "_holdings_data", {"holdings": holdings})
        monkeypatch.setattr(routes, "_LIVE_PORTFOLIO_DEADLINE", 0.05)

        data = routes.get_live_portfolio()

        assert data["partial"] is True
        assert data["n_with_price"] == 2
        assert data["n_no_price"] == 3

    def test_fundamentals_bulk_is_single_flight(self, client):
        assert routes._fundamental_fetch_slots.acquire(blocking=False)
        try:
            resp = client.post("/api/fundamentals/bulk", json=["600519"])
        finally:
            routes._fundamental_fetch_slots.release()

        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "15"
