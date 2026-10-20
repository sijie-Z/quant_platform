"""Regression tests for defects found by the full endpoint-contract sweep."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from quant_platform.api import routes


@pytest.fixture
def app_client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _call(coro):
    return asyncio.run(coro)


# ── OMS / execution input validation ──


def test_oms_order_requires_a_ticker(app_client):
    resp = app_client.post("/api/oms/order", json={})
    assert resp.status_code == 422
    assert "ticker" in resp.json()["detail"]


def test_oms_order_rejects_a_negative_quantity(app_client):
    resp = app_client.post(
        "/api/oms/order",
        json={"ticker": "600519", "side": "buy", "quantity": -100},
    )
    assert resp.status_code == 422


def test_smart_route_requires_a_ticker(app_client):
    resp = app_client.post("/api/execution/smart-route", json={})
    assert resp.status_code == 422
    assert "ticker" in resp.json()["detail"]


# ── Trading engine stopped ──


def test_run_once_rejects_a_stopped_engine(monkeypatch):
    class StoppedEngine:
        _running = False

        def run_once(self):  # pragma: no cover - must not run
            raise AssertionError("run_once called on a stopped engine")

    monkeypatch.setattr(routes, "_live_engine", StoppedEngine())

    with pytest.raises(HTTPException) as exc:
        routes.trading_run_once()
    assert exc.value.status_code == 400


def test_manual_order_rejects_a_stopped_engine(monkeypatch):
    class StoppedEngine:
        _running = False

        def manual_order(self, **kwargs):  # pragma: no cover
            raise AssertionError("manual_order called on a stopped engine")

    monkeypatch.setattr(routes, "_live_engine", StoppedEngine())

    with pytest.raises(HTTPException) as exc:
        routes.trading_manual_order({"code": "600519"})
    assert exc.value.status_code == 400


# ── Multi-strategy allocation ──


def test_reallocation_resets_the_pnl_baseline():
    from quant_platform.strategy.multi_strategy import (
        MultiStrategyManager,
        StrategyConfig,
    )

    mgr = MultiStrategyManager(total_capital=100_000_000)
    sid = mgr.add_strategy(StrategyConfig(name="s", allocation_pct=0.1))

    mgr.allocate_capital({sid: 1.0})
    mgr.update_strategy_pnl(sid, 0.01)

    state = mgr.states[sid]
    assert state.capital_allocated == pytest.approx(100_000_000)
    assert state.current_value == pytest.approx(101_000_000)
    assert state.total_return == pytest.approx(0.01)


def test_allocate_rejects_unknown_strategy(app_client, monkeypatch):
    from quant_platform.strategy.multi_strategy import MultiStrategyManager

    monkeypatch.setattr(
        routes, "_multi_strategy_mgr", MultiStrategyManager(total_capital=1.0),
    )

    resp = app_client.post("/api/strategy/allocate", json={"weights": {"nope": 1.0}})

    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]


# ── Parallel sweep / regime status codes ──


def test_parallel_sweep_invalid_param_is_422(app_client, monkeypatch):
    from quant_platform.backtest.distributed import ParallelBacktester

    def explode(self, **kwargs):
        raise ValueError("Unknown sweep parameter 'bogus_key'")

    monkeypatch.setattr(ParallelBacktester, "run_sweep", explode)

    resp = app_client.post(
        "/api/parallel/sweep", json={"param_grid": {"bogus_key": [1]}},
    )

    assert resp.status_code == 422
    assert "bogus_key" in resp.json()["detail"]


def test_regime_with_short_history_is_409(app_client, monkeypatch):
    routes._run_store["short-run"] = {
        "chart_data": {"equity": [1.0] * 10, "dates": [str(i) for i in range(10)]},
    }
    try:
        resp = app_client.post("/api/regime/detect", json={"run_id": "short-run"})
    finally:
        routes._run_store.pop("short-run", None)

    assert resp.status_code == 409
    assert "at least 252" in resp.json()["detail"]


# ── Screen raw values ──


def test_screen_rejects_unknown_factor(app_client, monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=3)
    raw = {"pe_ratio": pd.DataFrame([[10.0, 40.0]] * 3, index=index,
                                    columns=["A", "B"])}

    monkeypatch.setattr(routes, "load_config", lambda _path=None: SimpleNamespace(
        universe=SimpleNamespace(n_stocks=2),
    ))
    monkeypatch.setattr(
        routes, "_load_data",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), None, None, None, None),
    )
    monkeypatch.setattr(
        routes, "_compute_factors",
        lambda *a, **k: ({}, {}, None, None, raw),
    )

    resp = app_client.post("/api/screen", json={
        "rules": [{"factor": "does_not_exist", "operator": "lt", "value": 1}],
    })

    assert resp.status_code == 422
    assert "does_not_exist" in resp.json()["detail"]


# ── Data quality provenance ──


def test_data_quality_marks_synthetic_demo(monkeypatch):
    class FakeProvider:
        def __init__(self, **kwargs):
            pass

        def get_prices(self, start, end):
            index = pd.bdate_range("2024-01-01", periods=5)
            return pd.DataFrame(
                np.linspace(10, 11, 5)[:, None].repeat(2, axis=1),
                index=index, columns=["A", "B"],
            )

    monkeypatch.setattr(
        "quant_platform.data.providers.synthetic.SyntheticDataProvider",
        FakeProvider,
    )

    report = routes.data_quality_check({})

    assert report["source"] == "synthetic_demo"
    assert report["synthetic"] is True
    assert report["data_date_range"] == ["2024-01-01", "2024-01-05"]
