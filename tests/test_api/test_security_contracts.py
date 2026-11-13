"""Security and resource-boundary contracts for the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from quant_core.store import _clamp_days, _clamp_limit

from quant_platform.api import routes
from quant_platform.api.schemas import ParallelSweepRequest, RunRequest, SweepRequest
from quant_platform.app import create_app


class _RemoteClient:
    """Rewrite the ASGI client address so a TestClient looks remote."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope)
            scope["client"] = ("203.0.113.9", 54321)
        await self.app(scope, receive, send)


@pytest.fixture
def remote_client(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    app = create_app(serve_frontend=False)
    return TestClient(_RemoteClient(app))


@pytest.fixture
def local_client(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    app = create_app(serve_frontend=False)
    return TestClient(app)


def test_tokenless_remote_api_is_refused(remote_client):
    """Only /api/health was public; every other route was readable remotely."""
    assert remote_client.get("/api/health").status_code == 200
    for path in (
        "/api/core/store/orders",
        "/api/core/audit",
        "/api/monitor/risk-overview",
        "/api/strategy/list",
    ):
        assert remote_client.get(path).status_code == 403, path


def test_tokenless_remote_docs_are_refused(remote_client):
    assert remote_client.get("/openapi.json").status_code == 403
    assert remote_client.get("/docs").status_code == 403


def test_token_protects_every_api_route(monkeypatch):
    monkeypatch.setenv("QUANT_API_TOKEN", "s3cret")
    app = create_app(serve_frontend=False)
    client = TestClient(_RemoteClient(app))

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/factors").status_code == 401
    assert client.get("/api/ml/train").status_code in (401, 405)
    assert client.get(
        "/api/factors", headers={"Authorization": "Bearer s3cret"},
    ).status_code == 200


def test_cors_preflight_is_not_blocked(monkeypatch):
    monkeypatch.setenv("QUANT_API_TOKEN", "s3cret")
    client = TestClient(create_app(serve_frontend=False))
    resp = client.options(
        "/api/run",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_screen_config_path_must_stay_inside_config_dir(local_client, tmp_path):
    outside = tmp_path / "evil.yaml"
    outside.write_text("universe: {n_stocks: 1}\n", encoding="utf-8")

    resp = local_client.post("/api/screen", json={
        "rules": [{"factor": "pe_ratio", "operator": "lt", "value": 30}],
        "config": str(outside),
    })

    assert resp.status_code == 422
    assert "config/" in resp.json()["detail"]


def test_oms_rejects_nan_limit_price(local_client):
    routes._oms_instance = None
    resp = local_client.post(
        "/api/oms/order",
        content=b'{"ticker":"600519","side":"buy","quantity":100,"limit_price":NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert "limit_price" in resp.json()["detail"]


def test_strategy_rejects_nan_allocation(local_client):
    resp = local_client.post(
        "/api/strategy/add",
        content=b'{"name":"nan","allocation_pct":NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_risk_check_rejects_nan_order(local_client):
    resp = local_client.post(
        "/api/risk/check-order",
        content=b'{"quantity":NaN,"price":NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_run_window_is_bounded():
    with pytest.raises(ValidationError):
        RunRequest(start_date="1900-01-01", end_date="2100-12-31")
    with pytest.raises(ValidationError):
        RunRequest(start_date="2025-02-01", end_date="2025-01-01")


def test_sweep_and_parallel_grid_are_bounded():
    with pytest.raises(ValidationError):
        SweepRequest(n_stocks_list=[10**9])
    with pytest.raises(ValidationError):
        SweepRequest(optimizers=["equal_weight"] * 100_000)
    with pytest.raises(ValidationError):
        SweepRequest(optimizers=[])
    with pytest.raises(ValidationError):
        ParallelSweepRequest(param_grid={"x": list(range(21))})
    with pytest.raises(ValidationError):
        ParallelSweepRequest(param_grid={"a": [1] * 10, "b": [1] * 10, "c": [1, 2]})
    with pytest.raises(ValidationError):
        ParallelSweepRequest(max_workers=100000)


def test_sweep_dedupes_repeated_optimizers():
    req = SweepRequest(
        optimizers=["equal_weight", "equal_weight", "risk_parity"],
    )
    assert req.optimizers == ["equal_weight", "risk_parity"]


def test_parallel_sweep_overrides_are_validated():
    """The audit sent n_stocks=10,000,000 and 1900..2100 straight to workers."""
    grid = {"n_stocks": [100]}

    with pytest.raises(ValidationError):
        ParallelSweepRequest(param_grid={"n_stocks": [10_000_000]})
    with pytest.raises(ValidationError):
        ParallelSweepRequest(
            param_grid=grid, base_overrides={"n_stocks": 10_000_000},
        )
    with pytest.raises(ValidationError):
        ParallelSweepRequest(
            param_grid=grid, base_overrides={"data.provider": "tushare"},
        )
    with pytest.raises(ValidationError):
        ParallelSweepRequest(
            param_grid=grid, base_overrides={"data.start_date": "1900-01-01"},
        )
    with pytest.raises(ValidationError):
        ParallelSweepRequest(param_grid=grid, base_overrides={
            "data.start_date": "2020-01-01", "data.end_date": "2100-01-01",
        })
    # A single one-sided override is checked against the configured window.
    with pytest.raises(ValidationError):
        ParallelSweepRequest(
            param_grid=grid,
            base_overrides={"data.end_date": "2040-01-01"},
        )
    with pytest.raises(ValidationError):
        ParallelSweepRequest(param_grid={"portfolio.optimizer": ["evil"]})
    with pytest.raises(ValidationError):
        ParallelSweepRequest(param_grid={"portfolio.path": ["/etc/passwd"]})
    with pytest.raises(ValidationError):
        ParallelSweepRequest(
            param_grid=grid, base_overrides={"alpha.method": {"nested": 1}},
        )

    ok = ParallelSweepRequest(
        param_grid={"n_stocks": [100, 200]},
        base_overrides={
            "data.start_date": "2022-01-01", "data.end_date": "2024-12-31",
        },
    )
    assert ok.param_grid == {"n_stocks": [100, 200]}


def test_sweep_endpoints_reject_oversized_requests(local_client):
    resp = local_client.post(
        "/api/sweep",
        json={"optimizers": ["equal_weight"] * 50_000},
    )
    assert resp.status_code == 422

    resp = local_client.post(
        "/api/parallel/sweep",
        json={"param_grid": {"n_stocks": [10_000_000]}},
    )
    assert resp.status_code == 422


def test_research_windows_are_bounded(local_client):
    from quant_platform.api.research_context import parse_window

    with pytest.raises(ValueError):
        parse_window("0001-01-01", "9999-12-31")
    with pytest.raises(ValueError):
        parse_window("1900-01-01", "1901-01-01")
    with pytest.raises(ValueError):
        parse_window("2020-01-01", "2100-01-01")

    for params in (
        {"start_date": "0001-01-01", "end_date": "9999-12-31"},
        {"start_date": "1900-01-01", "end_date": "1901-01-01"},
        {"start_date": "2020-01-01", "end_date": "2100-01-01"},
    ):
        resp = local_client.get("/api/analysis/ic-decay", params=params)
        assert resp.status_code == 422, (params, resp.text[:200])


def test_trading_start_rejects_qmt_password_in_body(local_client, monkeypatch):
    """Credentials must come from the server environment, never a body."""
    monkeypatch.setattr(routes, "_live_engine", None)
    resp = local_client.post(
        "/api/trading/start",
        json={"broker": "qmt", "password": "super-secret"},
    )
    assert resp.status_code == 422
    assert "QMT_PASSWORD" in resp.json()["detail"]
    assert "super-secret" not in resp.text


def test_store_limits_are_clamped():
    assert _clamp_limit(-1) == 0
    assert _clamp_limit(10**30) == 1000
    assert _clamp_limit("bogus") == 100
    assert _clamp_days(10**12) == 3650
    assert _clamp_days(-5) == 1


def test_nan_validation_error_is_serialisable(local_client):
    resp = local_client.post(
        "/api/monitor/config",
        content=b'{"max_position_pct":NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    resp.json()  # must not raise on a NaN echo
