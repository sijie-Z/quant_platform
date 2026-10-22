"""FastAPI application for the A-Share Multi-Factor Quant Platform.

Usage:
    python app.py                    # Start web server (default port 8000)
    python app.py --port 8080        # Custom port
    python app.py --no-frontend      # API only, no static files

    python main.py run               # CLI mode (original)
    python main.py web               # Same as python app.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the parent directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from quant_platform.api.monitor import router as monitor_router
from quant_platform.api.routes import router as api_router

# Determine frontend dist path
_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_DIST_DIR = _FRONTEND_DIR / "dist"

#: Hard cap for request bodies. Parsing happens inside FastAPI/Starlette, so
#: the limit must be enforced before the app sees the stream. 10 MB is well
#: above every documented request (holdings imports and parallel grids are
#: the largest) while bounding memory a single client can pin.
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024

#: Bodies larger than this are authorized *before* buffering so an
#: unauthenticated client cannot make the server allocate 10 MB per request.
_EARLY_AUTH_BYTES = 64 * 1024

#: Protected document routes (kept in sync with APITokenMiddleware).
_PROTECTED_DOCS = frozenset({
    "/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json",
    "/api/docs", "/api/redoc",
})

#: Expensive compute endpoints get a simple per-client token bucket. The
#: analytics pool already rejects concurrent saturation with 429; this stops
#: one client from queueing a rapid-fire series that keeps the pool busy.
_RATE_LIMITED_PATHS = frozenset({
    "/api/run",
    "/api/compare",
    "/api/sweep",
    "/api/parallel/sweep",
    "/api/walkforward",
    "/api/montecarlo",
    "/api/risk/decompose",
    "/api/barra/decompose",
    "/api/barra/covariance",
    "/api/ml/train",
    "/api/ml/predict",
    "/api/ic-monitor/compute",
    "/api/data/quality",
    "/api/report/html",
    "/api/screen",
    "/api/fundamentals/screen",
    "/api/fundamentals/bulk",
    "/api/fundamentals/rank",
    "/api/portfolio/import",
    "/api/oms/order",
    "/api/oms/fill",
    "/api/trading/order",
})
# These GET endpoints build/reuse the heavy research context; without a
# bucket they bypass the compute admission control entirely.
_RATE_LIMITED_GET_PATHS = frozenset({
    "/api/analysis/ic-decay",
    "/api/analysis/correlation",
    "/api/monitor/risk-overview",
    "/api/monitor/tca-summary",
    "/api/monitor/factor-status",
    "/api/monitor/capacity-gauge",
})
_RATE_LIMIT_CAPACITY = 20.0        # burst
_RATE_LIMIT_REFILL_PER_SECOND = 1.0


def _scope_header(scope, name: str) -> str | None:
    """Case-insensitive ASGI header lookup.

    Starlette 0.40's ``Headers.__getitem__`` compares raw header bytes
    case-sensitively, so a hand-built scope (or a server that preserves case)
    can hide ``Content-Length``. The security checks must not depend on that.
    """
    wanted = name.lower().encode("latin-1")
    for header_name, value in scope.get("headers") or []:
        if header_name.lower() == wanted:
            return value.decode("latin-1")
    return None


async def _send_json(
    send, status: int, payload: dict, headers: dict[str, str] | None = None,
) -> None:
    """Send a minimal JSON response from a pure-ASGI middleware."""
    body = json.dumps(payload).encode("utf-8")
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    for name, value in (headers or {}).items():
        raw_headers.append((name.encode("latin-1"), value.encode("latin-1")))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


#: Hostnames are compared as ASCII labels; unicode hosts must arrive as punycode.
_HOST_LABEL_RE = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_-]*[A-Za-z0-9_])?\Z")
_IPV6_LITERAL_RE = re.compile(r"[0-9A-Fa-f:.]+\Z")


def _scope_host(scope) -> str:
    """Raw Host header from the ASGI scope ('' when absent)."""
    for name, value in scope.get("headers") or []:
        if name.lower() == b"host":
            return value.decode("latin-1")
    return ""


def _valid_host_header(value: str) -> bool:
    """Whether ``value`` is a well-formed ``host`` or ``host:port``.

    Starlette 0.40's bad-host parsing (GHSA-86qp-5c8j-p5mr) let
    ``Host: x/..`` rewrite the URL path and strip the auth prefix. Only a
    syntactically valid host reaches the application; anything containing
    path delimiters, credentials, whitespace/control characters, ``..`` or
    malformed ports is rejected before routing.
    """
    if not value or len(value) > 255:
        return False
    if any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in value):
        return False
    if any(ch in value for ch in "/\\@?#,;%"):
        return False
    if ".." in value:
        return False
    port = ""
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return False
        inner = value[1:end]
        rest = value[end + 1:]
        if rest:
            if not rest.startswith(":") or not rest[1:].isdigit():
                return False
            port = rest[1:]
        if not inner or not _IPV6_LITERAL_RE.fullmatch(inner):
            return False
    else:
        host = value
        if host.count(":") > 1:
            return False
        if ":" in host:
            host, _, port = host.partition(":")
        if not host or len(host) > 253:
            return False
        labels = host[:-1].split(".") if host.endswith(".") else host.split(".")
        for label in labels:
            if not label or len(label) > 63 or not _HOST_LABEL_RE.fullmatch(label):
                return False
    if not port:
        return True
    return port.isdigit() and 0 < int(port) <= 65535


def _host_name(value: str) -> str:
    """Host part of a Host header, lowercased, without the port."""
    if value.startswith("["):
        end = value.find("]")
        return value[1:end].lower() if end > 0 else ""
    return (value.rsplit(":", 1)[0] if ":" in value else value).lower()


def _host_is_loopback(value: str) -> bool:
    """Whether a Host header names the local machine.

    DNS rebinding connects to 127.0.0.1 from a page whose Host/Origin is an
    attacker domain, so the token-less loopback bypass must also require a
    loopback Host name (or an explicit QUANT_ALLOWED_HOSTS entry).
    """
    host = _host_name(value)
    if host in {"localhost", "127.0.0.1", "::1", "testserver"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_is_allowed(value: str) -> bool:
    """Optional ``QUANT_ALLOWED_HOSTS`` allowlist (default: any valid host).

    Comma-separated hostnames, case-insensitive; ``*.example.com`` matches
    subdomains and ``*`` disables the allowlist. The port is ignored.
    """
    raw = os.environ.get("QUANT_ALLOWED_HOSTS", "")
    entries = [entry.strip().lower().rstrip(".") for entry in raw.split(",") if entry.strip()]
    if not entries:
        return True
    host = _host_name(value).rstrip(".")
    for entry in entries:
        if entry == "*" or host == entry:
            return True
        if entry.startswith("*.") and host.endswith(entry[1:]):
            return True
    return False


#: Default ports per URL scheme, used to normalize Origin/Host comparisons so
#: ``http://host`` and ``http://host:80`` are the same origin.
_DEFAULT_PORTS = {"http": 80, "https": 443}

#: WebSocket handshake schemes are compared as their HTTP equivalent: a
#: browser's ``Origin`` is always an http(s) origin.
_SCHEME_TO_HTTP = {"http": "http", "https": "https", "ws": "http", "wss": "https"}

#: A serialized origin is exactly ``scheme://authority``; browsers never send
#: a path/query/fragment. One trailing slash is tolerated (a common curl
#: habit) and stripped, but ``/path``, ``?query`` and ``#fragment`` are not.
_ORIGIN_RE = re.compile(r"(?i)\A(https?)://([^/?#]+)/?\Z")


def _split_host_port(value: str, default_port: int) -> tuple[str, int] | None:
    """Lowercased host and numeric port from ``host[:port]``/``[v6][:port]``."""
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            return None
        host = value[1:end]
        rest = value[end + 1:]
        if rest:
            if not rest.startswith(":") or not rest[1:].isdigit():
                return None
            port = int(rest[1:])
        else:
            port = default_port
    else:
        if value.count(":") > 1:
            return None
        if ":" in value:
            host, _, raw_port = value.partition(":")
            if not raw_port.isdigit():
                return None
            port = int(raw_port)
        else:
            host, port = value, default_port
    if not host or not 0 < port <= 65535:
        return None
    # A fully-qualified trailing dot is the same DNS name; normalize it so
    # ``Host: example.com.`` and ``Origin: http://example.com`` compare equal.
    return host.lower().rstrip("."), port


def _parse_origin(value: str) -> tuple[str, str, int] | None:
    """Parse an ``Origin`` header into a normalized ``(scheme, host, port)``.

    ``None`` covers everything that is not a plain serialized http(s) origin:
    ``null`` (opaque origin), malformed authorities, wildcards, userinfo,
    paths and unknown schemes. Such values are never treated as same-origin.
    """
    match = _ORIGIN_RE.fullmatch((value or "").strip())
    if not match:
        return None
    scheme = match.group(1).lower()
    authority = match.group(2)
    if not _valid_host_header(authority):
        return None
    split = _split_host_port(authority, _DEFAULT_PORTS[scheme])
    if split is None:
        return None
    return (scheme, split[0], split[1])


def _allowed_origins() -> set[tuple[str, str, int]]:
    """``QUANT_ALLOWED_ORIGINS`` parsed into normalized origins.

    Comma-separated exact origins, case-insensitive. Wildcards are not
    supported and are ignored (they can never match); malformed entries are
    skipped rather than widening the set.
    """
    raw = os.environ.get("QUANT_ALLOWED_ORIGINS", "")
    allowed: set[tuple[str, str, int]] = set()
    for entry in raw.split(","):
        parsed = _parse_origin(entry)
        if parsed is not None:
            allowed.add(parsed)
    return allowed


def _origin_values(scope) -> list[str]:
    """All ``Origin`` header values, preserving duplicates for rejection."""
    return [
        value.decode("latin-1")
        for name, value in scope.get("headers") or []
        if name.lower() == b"origin"
    ]


def _origin_is_allowed(scope) -> bool:
    """CSRF gate for HTTP requests and WebSocket handshakes.

    Requests without an ``Origin`` (curl, server-to-server, older clients)
    pass. A present ``Origin`` must be same-origin with the validated ``Host``
    (scheme http/https after mapping ws->http/wss->https, host and port
    compared after default-port normalization) or listed verbatim in
    ``QUANT_ALLOWED_ORIGINS``. The decision uses the raw Host header and the
    ASGI scheme only -- ``X-Forwarded-*`` is deliberately ignored, so a forged
    forwarding header cannot manufacture a same-origin match.
    """
    values = _origin_values(scope)
    if not values:
        return True
    # Duplicate Origin headers are malformed (Fetch produces at most one) and
    # could make a proxy and the app disagree; reject rather than guess.
    if len(values) != 1:
        return False
    origin = _parse_origin(values[0])
    if origin is None:
        return False
    if origin in _allowed_origins():
        return True
    scheme = _SCHEME_TO_HTTP.get((scope.get("scheme") or "").lower())
    host = _scope_host(scope)
    if scheme is None or not host:
        return False
    request_origin = _split_host_port(host, _DEFAULT_PORTS[scheme])
    if request_origin is None:
        return False
    return origin == (scheme, request_origin[0], request_origin[1])


def _requires_auth(scope) -> bool:
    """Whether the request targets a protected /api or docs route."""
    path = _normalized_route_path(scope)
    if path is None:
        return True
    if path in _PROTECTED_DOCS:
        return True
    return (
        (path == "/api" or path.startswith("/api/")) and path != "/api/health"
    )


def _quick_authorized(scope) -> bool:
    """Cheap credential check used before buffering a large request body.

    Mirrors APITokenMiddleware (token compares are timing-safe; token-less
    mode additionally requires a loopback Host to defeat DNS rebinding).
    """
    token = os.environ.get("QUANT_API_TOKEN", "")
    if token:
        token_bytes = token.encode("utf-8")
        auth = (_scope_header(scope, "authorization") or "").encode("utf-8")
        key = (_scope_header(scope, "x-api-key") or "").encode("utf-8")
        return secrets.compare_digest(key, token_bytes) or secrets.compare_digest(
            auth, f"Bearer {token}".encode(),
        )
    peer = (scope.get("client") or ("", 0))[0]
    host = _scope_host(scope)
    host_ok = _host_is_loopback(host) or (
        bool(os.environ.get("QUANT_ALLOWED_HOSTS", "").strip())
        and _host_is_allowed(host)
    )
    return (
        peer in ("127.0.0.1", "::1", "localhost", "testclient") and host_ok
    )


def _normalized_route_path(scope) -> str | None:
    """Normalize ``scope["path"]`` for the authorization decision.

    Never use ``request.url.path``: on Starlette 0.40 a crafted Host header
    re-parses the URL and yields ``/../api/config`` while routing still matched
    ``/api/config``. The ASGI ``path`` is what the router matches; it is
    adjusted here for ``root_path`` deployments and collapsed (duplicate
    slashes, ``.`` segments, trailing slash). ``None`` means the path is
    ambiguous (dot-dot segments, percent-encoded or backslash/NUL bytes) and
    the request must be denied.
    """
    path = scope.get("path") or "/"
    root = (scope.get("root_path") or "").rstrip("/")
    if root and path.startswith(root) and (
        len(path) == len(root) or path[len(root):len(root) + 1] == "/"
    ):
        path = path[len(root):] or "/"
    if "%" in path or "\\" in path or "\x00" in path:
        return None
    segments = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment.startswith(".."):
            return None
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in segment):
            return None
        segments.append(segment)
    return "/" + "/".join(segments)


class RequestGuardMiddleware:
    """Outermost, pure-ASGI guard: Host, Origin (CSRF), body cap, rate limit.

    It runs before routing, static files, docs, metrics and authentication, so
    a malformed Host can never reach the router (the P0 was that the auth
    middleware and the router disagreed about the path), a browser request
    from an untrusted Origin is rejected before any state can change, a giant
    body is rejected before it is parsed, and one client cannot hammer
    expensive compute endpoints. Pure ASGI, so it also covers WebSocket
    handshakes, which ``BaseHTTPMiddleware`` never sees.
    """

    def __init__(self, app):
        self.app = app
        self._buckets: dict[str, list[float]] = {}

    def _rate_limited(self, scope, path: str) -> tuple[bool, float]:
        method = scope.get("method")
        applies = (
            (method == "POST" and path in _RATE_LIMITED_PATHS)
            or (method == "GET" and path in _RATE_LIMITED_GET_PATHS)
        )
        if not applies:
            return False, 0.0
        client = scope.get("client")
        key = client[0] if client else "unknown"
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) > 1024:  # bound per-instance state
                self._buckets.clear()
            bucket = [_RATE_LIMIT_CAPACITY, now]
            self._buckets[key] = bucket
        tokens = min(
            _RATE_LIMIT_CAPACITY,
            bucket[0] + (now - bucket[1]) * _RATE_LIMIT_REFILL_PER_SECOND,
        )
        bucket[0], bucket[1] = tokens, now
        if tokens >= 1.0:
            bucket[0] = tokens - 1.0
            return False, 0.0
        return True, (1.0 - tokens) / _RATE_LIMIT_REFILL_PER_SECOND

    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        host = _scope_host(scope)
        host_allowlist_configured = bool(
            os.environ.get("QUANT_ALLOWED_HOSTS", "").strip()
        )
        if (not host and host_allowlist_configured) or (
            host and (not _valid_host_header(host) or not _host_is_allowed(host))
        ):
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await _send_json(
                    send, 403, {"detail": "Invalid or untrusted Host header"},
                )
            return

        # CSRF gate: a browser-supplied Origin must be same-origin with the
        # validated Host or explicitly allowlisted. Runs before routing and
        # authentication, for HTTP and the WebSocket handshake alike. CORS
        # preflights are exempt: CORSMiddleware enforces its own origin
        # allowlist for them and an OPTIONS request never reaches a handler.
        is_preflight = scope_type == "http" and scope.get("method") == "OPTIONS"
        if not is_preflight and not _origin_is_allowed(scope):
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await _send_json(send, 403, {"detail": "Origin not allowed"})
            return

        if scope_type == "websocket":
            await self.app(scope, receive, send)
            return

        path = _normalized_route_path(scope)
        if path is not None:
            limited, retry_after = self._rate_limited(scope, path)
            if limited:
                wait = max(1, int(round(retry_after)))
                await _send_json(
                    send,
                    429,
                    {
                        "detail": "Rate limit exceeded for a compute endpoint",
                        "retry_after_s": wait,
                    },
                    headers={"Retry-After": str(wait)},
                )
                return

        content_length = _scope_header(scope, "content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await _send_json(send, 400, {"detail": "Invalid Content-Length"})
                return
            if declared < 0:
                await _send_json(send, 400, {"detail": "Invalid Content-Length"})
                return
            if declared > _MAX_REQUEST_BODY_BYTES:
                await _send_json(
                    send,
                    413,
                    {"detail": f"Request body exceeds {_MAX_REQUEST_BODY_BYTES} bytes"},
                )
                return
            if (
                declared > _EARLY_AUTH_BYTES
                and _requires_auth(scope)
                and not _quick_authorized(scope)
            ):
                await _send_json(
                    send,
                    401 if os.environ.get("QUANT_API_TOKEN") else 403,
                    {"detail": "Authentication required"},
                )
                return

        # Buffer the body up to the cap, then replay it downstream. A lazy
        # receive-wrapper cannot work here: FastAPI converts any exception
        # raised while parsing the body into a generic 400, so a streaming
        # oversize body would not surface as 413.
        chunks: list[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                break
            chunk = message.get("body", b"")
            received += len(chunk)
            if (
                received > _EARLY_AUTH_BYTES
                and _requires_auth(scope)
                and not _quick_authorized(scope)
            ):
                await _send_json(
                    send,
                    401 if os.environ.get("QUANT_API_TOKEN") else 403,
                    {"detail": "Authentication required"},
                )
                return
            if received > _MAX_REQUEST_BODY_BYTES:
                await _send_json(
                    send,
                    413,
                    {"detail": f"Request body exceeds {_MAX_REQUEST_BODY_BYTES} bytes"},
                )
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        buffered = b"".join(chunks)
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": buffered,
                    "more_body": False,
                }
            return await receive()

        async def send_with_security_headers(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                present = {name.lower() for name, _ in headers}
                for name, value in (
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"same-origin"),
                ):
                    if name not in present:
                        headers.append((name, value))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, replay_receive, send_with_security_headers)


class APITokenMiddleware(BaseHTTPMiddleware):
    """Require a token (or loopback) for every non-health API route.

    When no token is configured, EVERY /api route is loopback-only, not just
    the trading/OMS subset: the shipped default (`main.py web` binds 0.0.0.0)
    let remote anonymous clients read orders, trades, signals, audit rows and
    trigger every compute endpoint. Health stays public for probes.

    The authorization decision uses the ASGI ``scope["path"]`` (normalized for
    ``root_path``), never ``request.url.path``: on Starlette 0.40 a Host header
    containing a path segment (``Host: x/..``, GHSA-86qp-5c8j-p5mr) rewrote the
    parsed URL path while the router still matched ``/api/...``, bypassing this
    entire gate in token and loopback-only modes alike.
    """

    PUBLIC_PATHS = {
        "/api/health",
    }
    DOC_PATHS = {
        "/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json",
        "/api/docs", "/api/redoc",
    }
    LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost", "testclient")

    async def dispatch(self, request: Request, call_next):
        # Let CORS answer preflight; otherwise the browser sees a 401 without
        # CORS headers and the documented dev flow cannot use a token.
        if request.method == "OPTIONS":
            return await call_next(request)

        token = os.environ.get("QUANT_API_TOKEN", "")
        path = _normalized_route_path(request.scope)
        if path is None:
            return JSONResponse(
                status_code=400,
                content={"detail": "Ambiguous request path"},
            )
        protected = (
            (path == "/api" or path.startswith("/api/"))
            and path not in self.PUBLIC_PATHS
        ) or path in self.DOC_PATHS
        if protected:
            if token:
                auth = _scope_header(request.scope, "authorization") or ""
                api_key = _scope_header(request.scope, "x-api-key") or ""
                # Compare bytes: compare_digest(str, str) raises TypeError for
                # non-ASCII tokens, and env tokens are not charset-restricted.
                token_bytes = token.encode("utf-8")
                if not (
                    secrets.compare_digest(api_key.encode("utf-8"), token_bytes)
                    or secrets.compare_digest(
                        auth.encode("utf-8"),
                        f"Bearer {token}".encode(),
                    )
                ):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API token"},
                    )
            else:
                peer = request.client.host if request.client else ""
                host_header = _scope_host(request.scope)
                host_ok = _host_is_loopback(host_header)
                if (
                    not host_ok
                    and os.environ.get("QUANT_ALLOWED_HOSTS", "").strip()
                ):
                    host_ok = _host_is_allowed(host_header)
                if peer not in self.LOOPBACK_HOSTS or not host_ok:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "detail": (
                                "No QUANT_API_TOKEN configured: all API routes "
                                "are loopback-only (and the Host must be a "
                                "loopback name). Set the token to allow "
                                "remote access."
                            )
                        },
                    )
        return await call_next(request)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Run the event bus for the lifetime of the application.

    The bus is a process-wide singleton, but nothing ever started it. Without
    consumer loops every publish fell back to inline synchronous delivery,
    async handlers were skipped, and dead-letter retry never ran.

    Shutdown used to stop only the bus. The live trading engine, the market
    providers, connected WebSocket clients, the non-daemon executor threads
    and the SQLite connections were all abandoned, so an in-flight pipeline
    kept the process alive until it finished (or was SIGKILLed) and the
    engine's final session/NAV writes were lost.
    """
    from quant_platform.api import routes as api_routes
    from quant_platform.core.events import get_event_bus
    from quant_platform.risk.healthcheck import install_alert_handler

    # Executors were shut down by a previous lifespan exit; recreate them and
    # pre-build the critical singletons before any request can race them.
    api_routes.ensure_executors()
    api_routes.initialize_singletons()

    bus = get_event_bus()
    # S4: health-check failures and system alerts get a production handler
    # that logs and forwards them to the alert sink (idempotent per bus).
    install_alert_handler(bus)
    await bus.start()
    try:
        yield
    finally:
        await _shutdown_resources(api_routes, bus)


async def _bounded_await(label: str, awaitable, timeout: float) -> None:
    """Await a shutdown step, logging instead of hanging the process."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError:
        logging.getLogger(__name__).error(
            "%s did not finish within %.0fs; continuing shutdown", label, timeout,
        )
    except Exception:
        logging.getLogger(__name__).exception("%s failed", label)


async def _shutdown_resources(api_routes, bus) -> None:
    log = logging.getLogger(__name__)
    # 1. Engine, market providers, WebSocket clients (bounded inside).
    await _bounded_await(
        "API resource shutdown", api_routes.shutdown_api_resources(), 30.0,
    )
    # 2. Executors: cancel queued work, wait (bounded) for running work.
    #    Python cannot kill a running thread; a multi-minute pipeline keeps
    #    its thread until it returns, but the server no longer waits forever.
    await _bounded_await(
        "Executor shutdown",
        asyncio.to_thread(api_routes.shutdown_executors),
        10.0,
    )
    # 3. Event bus: stop consumer loops. `stop()` cancels consumers and drops
    #    queued events by design (event_bus_v2); the timeout only bounds it.
    await _bounded_await("EventBus stop", bus.stop(), 5.0)
    # 4. Close the process-wide SQLite connections (reopened lazily if the
    #    store is used again in the same process, e.g. repeated lifespans).
    try:
        api_routes._core_store.close()
    except Exception:
        log.exception("Store close failed")
    try:
        from quant_platform.api import monitor as monitor_module
        monitor_store = monitor_module._core_store
        if monitor_store is not None and monitor_store is not api_routes._core_store:
            monitor_store.close()
    except Exception:
        log.exception("Monitor store close failed")


def _json_safe(value):
    """Replace non-finite floats so Starlette's strict encoder accepts them.

    Pydantic v2 puts the original exception object in ``ctx.error`` for
    ``model_validator`` failures; that made the 422 body itself fail to
    serialise (an empty optimizer list returned a 500). Exceptions and any
    other non-JSON object become strings.
    """
    import math

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, BaseException):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def create_app(serve_frontend: bool = True) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="A-Share Multi-Factor Quant Platform",
        description="Multi-factor quantitative research platform with LLM-enhanced stock selection",
        version="1.0.0",
        lifespan=_lifespan,
        docs_url="/api/docs" if serve_frontend else "/docs",
        redoc_url="/api/redoc" if serve_frontend else "/redoc",
    )

    # Pydantic's validation detail echoes the offending input. For NaN/Inf
    # that made the 422 response itself unserialisable, so a bad request
    # became a 500.
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": _json_safe(exc.errors())},
        )

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(APITokenMiddleware)
    # Prometheus request counters/latency. The Grafana dashboard queried
    # api_requests_total/api_duration_seconds, but nothing ever recorded them.
    from quant_platform.utils.metrics import install_http_middleware

    install_http_middleware(app)
    # Outermost guard (Starlette inserts middleware at position 0, so this is
    # the first code to run): Host validation before routing/static/docs, the
    # request-body cap before parsing, and the compute rate limit. Security
    # middleware must run before any handler that could reflect input.
    app.add_middleware(RequestGuardMiddleware)

    # API routes
    app.include_router(api_router)
    app.include_router(monitor_router)

    # Serve frontend static files in production. Warn loudly when the bundle
    # is older than the source: serving a stale dist silently shipped removed
    # client-side fabrications back to users.
    if serve_frontend and _DIST_DIR.exists():
        index = _DIST_DIR / "index.html"
        source_dir = _FRONTEND_DIR / "src"
        if index.exists() and source_dir.exists():
            newest_source = max(
                (path.stat().st_mtime for path in source_dir.rglob("*") if path.is_file()),
                default=0.0,
            )
            if newest_source > index.stat().st_mtime:
                logging.getLogger(__name__).warning(
                    "frontend/dist is older than frontend/src -- serving a "
                    "stale bundle. Run `npm run build` in frontend/.",
                )
        app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Quant Platform Web Server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--no-frontend", action="store_true", help="API only, no static files")

    args = parser.parse_args()

    import uvicorn

    serve_frontend = not args.no_frontend
    global app
    app = create_app(serve_frontend=serve_frontend)

    mode = "API + Frontend" if serve_frontend else "API only"
    print("Quant Platform Web Server starting...")
    print(f"  Mode: {mode}")
    print(f"  API:  http://{args.host}:{args.port}/api/docs")
    if serve_frontend:
        if _DIST_DIR.exists():
            print(f"  UI:   http://{args.host}:{args.port}/")
        else:
            print("  UI:   Frontend not built. Run: cd frontend && npm run build")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
