"""Tests for optional API token authentication."""

import asyncio

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from quant_platform.app import APITokenMiddleware, create_app


def _request(path: str, host: str) -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": path,
        "client": (host, 12345),
        "headers": [],
        "scheme": "http",
        "server": ("testserver", 80),
        "query_string": b"",
    })


async def _ok(_request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


def test_api_open_when_token_not_configured(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    app = create_app(serve_frontend=False)
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_api_requires_token_when_configured(monkeypatch):
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get(
        "/api/config", headers={"Authorization": "Bearer secret"}
    ).status_code == 200


def test_sensitive_endpoints_are_loopback_only_without_a_token(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    middleware = APITokenMiddleware(app=None)

    remote = asyncio.run(middleware.dispatch(
        _request("/api/trading/status", "10.0.0.5"), _ok,
    ))
    assert remote.status_code == 403

    remote_oms = asyncio.run(middleware.dispatch(
        _request("/api/oms/blotter", "10.0.0.5"), _ok,
    ))
    assert remote_oms.status_code == 403

    local = asyncio.run(middleware.dispatch(
        _request("/api/trading/status", "127.0.0.1"), _ok,
    ))
    assert local.status_code == 200

    # Public read-only endpoints stay open to remote callers.
    health = asyncio.run(middleware.dispatch(
        _request("/api/health", "10.0.0.5"), _ok,
    ))
    assert health.status_code == 200


def test_websocket_requires_the_token_when_configured(monkeypatch):
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws"):
            pass

    with client.websocket_connect("/api/ws?token=secret") as ws:
        ws.send_text("ping")
