"""CSRF/Origin hardening (P2: loopback-mode CSRF, state-changing GETs).

The Origin guard is the outermost middleware, so these tests drive the raw
ASGI app: a loopback client with no token is otherwise fully authorized --
which is exactly the cross-site scenario, because a malicious page opened in
the operator's browser also connects from 127.0.0.1.
"""

from __future__ import annotations

import asyncio

import pytest

from quant_platform.api import routes
from quant_platform.app import create_app

LOCAL = ("127.0.0.1", 50000)
HOST = "127.0.0.1:8000"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    monkeypatch.delenv("QUANT_ALLOWED_ORIGINS", raising=False)
    return create_app(serve_frontend=False)


async def _http(
    app,
    *,
    path: str = "/api/health",
    method: str = "GET",
    host: str | None = HOST,
    scheme: str = "http",
    client=LOCAL,
    origins: list[str] | None = None,
    body: bytes = b"",
    extra_headers: dict[str, str] | None = None,
):
    """Invoke the app as a raw ASGI request; return status/body/headers."""
    headers: list[tuple[bytes, bytes]] = []
    if host is not None:
        headers.append((b"host", host.encode("latin-1")))
    if body:
        headers.append((b"content-length", str(len(body)).encode()))
        headers.append((b"content-type", b"application/json"))
    for value in origins or []:
        headers.append((b"origin", value.encode("latin-1")))
    for name, value in (extra_headers or {}).items():
        headers.append((name.encode("latin-1"), value.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": scheme,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": client,
        "server": ("127.0.0.1", 8000),
    }
    sent = {"status": None, "body": b"", "headers": {}}
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
            sent["headers"] = {
                name.decode("latin-1").lower(): value.decode("latin-1")
                for name, value in message.get("headers", [])
            }
        elif message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return sent


async def _ws(
    app,
    *,
    host: str = HOST,
    client=LOCAL,
    origins: list[str] | None = None,
    subprotocols: list[str] | None = None,
):
    """Drive a WebSocket handshake; the handler exits on the disconnect."""
    headers = [(b"host", host.encode("latin-1"))]
    for value in origins or []:
        headers.append((b"origin", value.encode("latin-1")))
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/api/ws",
        "raw_path": b"/api/ws",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": client,
        "server": ("127.0.0.1", 8000),
        "subprotocols": list(subprotocols or []),
    }
    sent = {"close_code": None, "accepted": False}
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "websocket.connect"}
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        if message["type"] == "websocket.close":
            sent["close_code"] = message.get("code")
        elif message["type"] == "websocket.accept":
            sent["accepted"] = True

    await app(scope, receive, send)
    return sent


# ---------------------------------------------------------------------------
# Cross-origin state changes are refused before the handler
# ---------------------------------------------------------------------------


def test_cross_origin_run_post_is_refused_before_state_change(app):
    before = dict(routes._run_status)
    result = asyncio.run(_http(
        app, path="/api/run", method="POST", body=b"{}",
        origins=["https://evil.example"],
    ))
    assert result["status"] == 403
    assert b"Origin" in result["body"]
    assert routes._run_status == before


class _FakeRiskMonitor:
    def __init__(self):
        self.calls: list[str] = []
        self.kill_switch_active = False

        class _Level:
            value = "GREEN"

        self.risk_level = _Level()

    def activate_kill_switch(self, _reason="", actor="api"):
        self.calls.append("activate")

    def deactivate_kill_switch(self, _reason="", actor="api"):
        self.calls.append("deactivate")

    def get_status(self):
        return {}


def test_cross_origin_kill_switch_post_is_refused_before_state_change(
    app, monkeypatch,
):
    fake = _FakeRiskMonitor()
    monkeypatch.setattr(routes, "_risk_monitor", fake)
    result = asyncio.run(_http(
        app, path="/api/risk/kill-switch", method="POST",
        body=b'{"action": "activate"}', origins=["https://evil.example"],
    ))
    assert result["status"] == 403
    assert fake.calls == []


def test_same_origin_post_reaches_the_handler(app, monkeypatch):
    fake = _FakeRiskMonitor()
    monkeypatch.setattr(routes, "_risk_monitor", fake)
    result = asyncio.run(_http(
        app, path="/api/risk/kill-switch", method="POST",
        body=b'{"action": "activate"}', origins=[f"http://{HOST}"],
    ))
    assert result["status"] == 200
    assert fake.calls == ["activate"]


# ---------------------------------------------------------------------------
# Same-origin / allowlist / absent Origin
# ---------------------------------------------------------------------------


def test_same_origin_origin_passes_loopback_auth(app):
    result = asyncio.run(_http(
        app, path="/api/config", origins=[f"http://{HOST}"],
    ))
    assert result["status"] == 200
    assert b"default_config" in result["body"]


def test_allowlisted_origin_is_allowed(app, monkeypatch):
    # Exact match, case-insensitive, default port and trailing slash normalized;
    # nonsense and wildcard entries never widen the set.
    monkeypatch.setenv(
        "QUANT_ALLOWED_ORIGINS",
        " https://APP.Example.com:443/ , nonsense, * , https://app.example.com ",
    )
    result = asyncio.run(_http(
        app, path="/api/config", origins=["https://app.example.com"],
    ))
    assert result["status"] == 200


def test_absent_origin_is_allowed(app):
    result = asyncio.run(_http(app, path="/api/config"))
    assert result["status"] == 200


@pytest.mark.parametrize(
    ("origins", "host", "scheme", "expected"),
    [
        # Same origin, with normalization that is semantically harmless.
        (["http://127.0.0.1:8000"], HOST, "http", 200),
        (["HTTP://127.0.0.1:8000"], HOST, "http", 200),
        (["http://127.0.0.1:8000/"], HOST, "http", 200),
        (["http://127.0.0.1:80"], "127.0.0.1", "http", 200),
        (["https://127.0.0.1:443"], "127.0.0.1", "https", 200),
        # Scheme downgrade/upgrade, port mismatch, path/query, opaque origin.
        (["https://127.0.0.1:8000"], HOST, "http", 403),
        (["http://127.0.0.1"], HOST, "http", 403),
        (["http://127.0.0.1:8000/api/health"], HOST, "http", 403),
        (["http://127.0.0.1:8000?x=1"], HOST, "http", 403),
        (["http://127.0.0.1:8000#x"], HOST, "http", 403),
        (["http://user@127.0.0.1:8000"], HOST, "http", 403),
        (["http://127.0.0.1%3A8000"], HOST, "http", 403),
        (["null"], HOST, "http", 403),
        ([""], HOST, "http", 403),
        (["https://evil.example"], HOST, "http", 403),
        # A comma-joined list in one header is not a serialized origin.
        (["http://127.0.0.1:8000, https://evil.example"], HOST, "http", 403),
        # Duplicate Origin headers are malformed; never pick one.
        (["http://127.0.0.1:8000", "http://127.0.0.1:8000"], HOST, "http", 403),
    ],
)
def test_origin_parsing_matrix(app, origins, host, scheme, expected):
    result = asyncio.run(_http(
        app, path="/api/health", host=host, scheme=scheme, origins=origins,
    ))
    assert result["status"] == expected, (origins, result["status"])


def test_forwarded_headers_cannot_mint_a_same_origin_match(app):
    result = asyncio.run(_http(
        app, path="/api/health", origins=["https://evil.example"],
        extra_headers={
            "X-Forwarded-Host": "evil.example",
            "X-Forwarded-Proto": "https",
            "X-Real-IP": "127.0.0.1",
        },
    ))
    assert result["status"] == 403


# ---------------------------------------------------------------------------
# WebSocket handshake
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin",
    ["https://evil.example", "null", "https://127.0.0.1:8000"],
)
def test_websocket_bad_origin_closes_1008(app, origin):
    result = asyncio.run(_ws(app, origins=[origin]))
    assert result["close_code"] == 1008
    assert result["accepted"] is False


def test_websocket_same_origin_and_allowlisted_origins_are_accepted(
    app, monkeypatch,
):
    same = asyncio.run(_ws(app, origins=["http://127.0.0.1:8000"]))
    assert same["accepted"] is True

    monkeypatch.setenv("QUANT_ALLOWED_ORIGINS", "https://ui.example.com")
    allowed = asyncio.run(_ws(app, origins=["https://ui.example.com"]))
    assert allowed["accepted"] is True


# ---------------------------------------------------------------------------
# State-changing GETs are POST-only
# ---------------------------------------------------------------------------


class _FakeWsQuoteProvider:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.is_connected = False
        self.stats: dict = {}

    def start(self):
        self.started += 1
        self.is_connected = True

    def stop(self):
        self.stopped += 1
        self.is_connected = False


class _FakeLevel2Provider:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.is_running = False
        self.stats: dict = {}

    def start(self):
        self.started += 1
        self.is_running = True

    def stop(self):
        self.stopped += 1
        self.is_running = False


def test_ws_quotes_start_stop_are_post_only(app, monkeypatch):
    provider = _FakeWsQuoteProvider()
    monkeypatch.setattr(routes, "_ws_provider", provider)

    get_start = asyncio.run(_http(app, path="/api/ws-quotes/start"))
    assert get_start["status"] == 405
    assert get_start["headers"].get("allow") == "POST"
    assert provider.started == 0

    post_start = asyncio.run(_http(
        app, path="/api/ws-quotes/start", method="POST",
    ))
    assert post_start["status"] == 200
    assert b'"started"' in post_start["body"]
    assert provider.started == 1

    get_stop = asyncio.run(_http(app, path="/api/ws-quotes/stop"))
    assert get_stop["status"] == 405
    assert provider.stopped == 0

    post_stop = asyncio.run(_http(
        app, path="/api/ws-quotes/stop", method="POST",
    ))
    assert post_stop["status"] == 200
    assert b'"stopped"' in post_stop["body"]
    assert provider.stopped == 1


def test_l2_start_stop_are_post_only(app, monkeypatch):
    provider = _FakeLevel2Provider()
    monkeypatch.setattr(routes, "_l2_provider", provider)

    assert asyncio.run(_http(app, path="/api/l2/start"))["status"] == 405
    assert asyncio.run(_http(app, path="/api/l2/stop"))["status"] == 405
    assert provider.started == 0
    assert provider.stopped == 0

    assert asyncio.run(
        _http(app, path="/api/l2/start", method="POST"),
    )["status"] == 200
    assert asyncio.run(
        _http(app, path="/api/l2/stop", method="POST"),
    )["status"] == 200
    assert provider.started == 1
    assert provider.stopped == 1
