"""P0 regression: Host-header path poisoning must not bypass API auth.

On Starlette 0.40 (GHSA-86qp-5c8j-p5mr) ``Host: x/..`` made
``request.url.path`` read ``/../api/config`` while routing still matched
``/api/config``, so ``APITokenMiddleware`` skipped the token/loopback gate.
``TestClient``'s client is loopback, which hides half the bug, so these tests
drive the raw ASGI app with an explicit remote client.

Every bad-Host case here returns 200 on the pre-fix middleware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quant_platform.app import create_app

REMOTE = ("203.0.113.9", 50000)
LOCAL = ("127.0.0.1", 50000)

BAD_HOSTS = [
    "x/..",
    "x/../..",
    "x/%2e%2e",
    "x/../apix",
    "x/..;/api",
    "x\\..\\..",
    "evil.com/y",
    "evil.com:bad",
    "evil.com:99999",
    "user@evil.com",
    "evil.com?x",
    "host with space",
    "evil..com",
    ".evil.com",
    "-bad.example",
]


async def _asgi_call(
    app,
    *,
    host: str | None = None,
    path: str = "/api/config",
    method: str = "GET",
    token: str | None = None,
    root_path: str = "",
    client=REMOTE,
    body: bytes = b"",
    extra_headers: dict[str, str] | None = None,
):
    """Invoke the app as a raw ASGI HTTP request; return (status, body)."""
    headers: list[tuple[bytes, bytes]] = []
    if host is not None:
        headers.append((b"host", host.encode("latin-1")))
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    if body:
        headers.append((b"content-length", str(len(body)).encode()))
    for name, value in (extra_headers or {}).items():
        headers.append((name.encode("latin-1"), value.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": root_path,
        "headers": headers,
        "client": client,
        "server": ("127.0.0.1", 8000),
    }
    sent = {"status": None, "body": b""}
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return sent["status"], sent["body"]


async def _asgi_websocket(app, *, host: str, path: str = "/api/ws", client=REMOTE):
    """Drive a WebSocket handshake; return the close code (None if accepted)."""
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", host.encode("latin-1"))],
        "client": client,
        "server": ("127.0.0.1", 8000),
        "subprotocols": [],
    }
    sent = {"close_code": None, "accepted": False}

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        if message["type"] == "websocket.close":
            sent["close_code"] = message.get("code")
        elif message["type"] == "websocket.accept":
            sent["accepted"] = True

    await app(scope, receive, send)
    return sent


@pytest.mark.parametrize("bad_host", BAD_HOSTS)
def test_bad_host_cannot_bypass_token_auth(monkeypatch, bad_host):
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)

    import asyncio

    status, body = asyncio.run(_asgi_call(app, host=bad_host))
    assert status == 403, (bad_host, status, body[:80])
    assert b"default_config" not in body


@pytest.mark.parametrize(
    "path",
    [
        "/api/config",
        "/api/risk/kill-switch",
        "/api/core/store/orders",
        "/api/core/audit",
    ],
)
def test_bad_host_is_rejected_on_sensitive_routes(monkeypatch, path):
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)

    import asyncio

    status, body = asyncio.run(_asgi_call(app, host="x/..", path=path))
    assert status == 403, (path, status)
    assert b"default_config" not in body


@pytest.mark.parametrize("bad_host", ["x/..", "x/%2e%2e", "x/..;/api"])
def test_bad_host_cannot_bypass_loopback_only_mode(monkeypatch, bad_host):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    app = create_app(serve_frontend=False)

    import asyncio

    status, body = asyncio.run(_asgi_call(app, host=bad_host, client=REMOTE))
    assert status == 403, (bad_host, status)
    assert b"default_config" not in body


def test_root_path_prefixed_deployment_is_still_protected(monkeypatch):
    """`/quant/api/config` with root_path=/quant routes to /api/config."""
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)

    import asyncio

    status, body = asyncio.run(_asgi_call(
        app, host="good.example", path="/quant/api/config",
        root_path="/quant", token=None,
    ))
    assert status == 401
    assert b"default_config" not in body

    status, body = asyncio.run(_asgi_call(
        app, host="good.example", path="/quant/api/config",
        root_path="/quant", token="secret",
    ))
    assert status == 200
    assert b"default_config" in body


def test_root_path_loopback_only_mode(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    app = create_app(serve_frontend=False)

    import asyncio

    status, _ = asyncio.run(_asgi_call(
        app, host="good.example", path="/quant/api/config",
        root_path="/quant", client=REMOTE,
    ))
    assert status == 403


def test_ambiguous_path_is_denied_even_with_valid_token(monkeypatch):
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)

    import asyncio

    for path in (
        "/api/../api/config",
        "/api/%2e%2e/config",
        "/api/..;/config",
        "/api\\config",
    ):
        status, body = asyncio.run(_asgi_call(
            app, host="good.example", path=path, token="secret",
        ))
        assert status in (400, 403), (path, status)
        assert b"default_config" not in body


def test_duplicate_slashes_and_trailing_slash_are_authorized(monkeypatch):
    """Normalization must decide *more* strictly than routing, not less."""
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)

    import asyncio

    for path in ("//api/config", "/api//config/", "/api/./config"):
        status, body = asyncio.run(_asgi_call(
            app, host="good.example", path=path, token=None,
        ))
        assert status == 401, (path, status)
        assert b"default_config" not in body


def test_options_head_docs_and_static_mount_are_guarded(monkeypatch):
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    app = create_app(serve_frontend=False)

    import asyncio

    # OPTIONS is normally passed through for CORS, but a poisoned Host must
    # be rejected before CORS/auth see the request.
    for method in ("OPTIONS", "HEAD"):
        status, _ = asyncio.run(_asgi_call(
            app, host="x/..", method=method,
        ))
        assert status == 403, method

    # Docs need the token, and the static mount ("/") is guarded by Host too.
    for path in ("/docs", "/openapi.json", "/api/docs"):
        status, _ = asyncio.run(_asgi_call(app, host="good.example", path=path))
        assert status == 401, path

    status, _ = asyncio.run(_asgi_call(app, host="x/..", path="/"))
    assert status == 403


def test_websocket_bad_host_is_closed_before_accept():
    app = create_app(serve_frontend=False)

    import asyncio

    outcome = asyncio.run(_asgi_websocket(app, host="x/.."))
    assert outcome["accepted"] is False
    assert outcome["close_code"] == 1008


def test_well_formed_hosts_still_work(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    app = create_app(serve_frontend=False)

    import asyncio

    status, body = asyncio.run(_asgi_call(
        app, host="localhost", path="/api/config", client=LOCAL,
    ))
    assert status == 200
    assert b"default_config" in body


def test_testclient_bad_host_header_is_rejected(monkeypatch):
    """The same bypass through the ordinary TestClient (loopback client).

    Pre-fix this returned 200 with the full config because Starlette's URL
    parser rewrote the path; the Host guard now rejects it first.
    """
    monkeypatch.setenv("QUANT_API_TOKEN", "secret")
    client = TestClient(create_app(serve_frontend=False))

    resp = client.get("/api/config", headers={"Host": "x/.."})

    assert resp.status_code == 403
    assert "default_config" not in resp.text


def test_host_allowlist_env(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    monkeypatch.setenv("QUANT_ALLOWED_HOSTS", "only.example,*.corp.example")
    app = create_app(serve_frontend=False)

    import asyncio

    ok, _ = asyncio.run(_asgi_call(
        app, host="only.example:8000", path="/api/config", client=LOCAL,
    ))
    assert ok == 200

    sub, _ = asyncio.run(_asgi_call(
        app, host="desk.corp.example", path="/api/config", client=LOCAL,
    ))
    assert sub == 200

    denied, _ = asyncio.run(_asgi_call(
        app, host="other.example", path="/api/config", client=LOCAL,
    ))
    assert denied == 403
