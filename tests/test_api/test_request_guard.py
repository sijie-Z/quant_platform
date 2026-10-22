"""Request guard: 10 MB body cap and per-client compute rate limiting."""

from __future__ import annotations

import json

from quant_platform.app import _MAX_REQUEST_BODY_BYTES, create_app

REMOTE = ("203.0.113.9", 50000)
LOCAL = ("127.0.0.1", 50000)
OTHER = ("127.0.0.2", 50000)


async def _call(
    app, *, path="/api/health", method="GET", body=b"", client=REMOTE,
    headers: dict[str, str] | None = None, declare_length: bool = True,
):
    raw_headers = [(b"host", b"localhost")]
    if body and declare_length:
        raw_headers.append((b"content-length", str(len(body)).encode()))
    for name, value in (headers or {}).items():
        raw_headers.append((name.encode("latin-1"), value.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": raw_headers,
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


def test_declared_oversized_body_is_rejected_before_parsing():
    import asyncio

    app = create_app(serve_frontend=False)
    too_large = _MAX_REQUEST_BODY_BYTES + 1
    result = asyncio.run(_call(
        app,
        path="/api/config",
        method="POST",
        body=b"",
        headers={"Content-Length": str(too_large)},
        declare_length=False,
    ))
    assert result["status"] == 413
    assert b"exceeds" in result["body"]


def test_streamed_oversized_body_is_rejected():
    import asyncio

    app = create_app(serve_frontend=False)
    payload = b"x" * (_MAX_REQUEST_BODY_BYTES + 1)
    result = asyncio.run(_call(
        app,
        path="/api/screen",
        method="POST",
        body=payload,
        client=LOCAL,
        # Declare a small length: the receive wrapper must still catch it.
        headers={"Content-Length": "10"},
        declare_length=False,
    ))
    assert result["status"] == 413


def test_compute_endpoints_are_rate_limited_per_client():
    import asyncio

    app = create_app(serve_frontend=False)
    payload = json.dumps({"optimizers": []}).encode()
    statuses = []
    for _ in range(21):
        result = asyncio.run(_call(
            app,
            path="/api/sweep",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/json"},
            client=LOCAL,
        ))
        statuses.append(result["status"])

    # The first 20 are admitted (and fail schema validation); the burst is
    # then exhausted and the guard answers 429 before the handler runs.
    assert statuses[:20] == [422] * 20, statuses
    assert statuses[20] == 429
    assert int(result["headers"]["retry-after"]) >= 1

    # Another client has its own bucket (it is not loopback, so auth answers
    # 403 -- the point is that the guard did not answer 429).
    other = asyncio.run(_call(
        app,
        path="/api/sweep",
        method="POST",
        body=payload,
        headers={"Content-Type": "application/json"},
        client=OTHER,
    ))
    assert other["status"] != 429


def test_read_endpoints_are_not_rate_limited():
    import asyncio

    app = create_app(serve_frontend=False)
    for _ in range(40):
        result = asyncio.run(_call(app, path="/api/health"))
        assert result["status"] == 200
