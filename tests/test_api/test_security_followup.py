"""Follow-up security fixes: rebinding, one-sided windows, early auth, leaks."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from quant_platform.app import create_app


def _raw_scope(path="/api/config", *, host="testserver", method="GET",
               client=("203.0.113.9", 5555), headers=(), content_length=None):
    base = [("host", host)] if host is not None else []
    base.extend(headers)
    if content_length is not None:
        base.append(("content-length", str(content_length)))
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in base],
        "client": client, "server": ("testserver", 80),
    }


async def _drive(app, scope, body=b"", reads=None):
    messages = []
    state = {"done": False}

    async def receive():
        if reads is not None:
            reads.append(1)
        if state["done"]:
            return {"type": "http.disconnect"}
        state["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    status = next(
        (m["status"] for m in messages if m["type"] == "http.response.start"),
        None,
    )
    return status, messages


@pytest.fixture()
def app():
    return create_app(serve_frontend=False)


class TestDnsRebinding:
    def test_rebound_host_is_rejected_in_tokenless_mode(self, app, monkeypatch):
        """Loopback peer + attacker Host/Origin must not reach a handler.

        A page on attacker.example rebound to 127.0.0.1 sends exactly this;
        previously the loopback bypass accepted any Host.
        """
        monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
        monkeypatch.delenv("QUANT_ALLOWED_HOSTS", raising=False)
        scope = _raw_scope(
            host="attacker.example:8000",
            headers=[("origin", "http://attacker.example:8000")],
            client=("127.0.0.1", 4444),
        )
        import asyncio

        status, messages = asyncio.run(_drive(app, scope))
        assert status == 403
        body = b"".join(m.get("body", b"") for m in messages)
        assert b"default_config" not in body

    def test_loopback_host_still_works(self, app, monkeypatch):
        monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
        scope = _raw_scope(host="localhost:8000", client=("127.0.0.1", 4444))
        import asyncio

        status, _ = asyncio.run(_drive(app, scope))
        assert status == 200

    def test_missing_host_rejected_when_allowlist_configured(self, app, monkeypatch):
        monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
        monkeypatch.setenv("QUANT_ALLOWED_HOSTS", "localhost")
        scope = _raw_scope(host=None, client=("127.0.0.1", 4444))
        import asyncio

        status, _ = asyncio.run(_drive(app, scope))
        assert status == 403


class TestOneSidedWindow:
    def test_one_sided_huge_end_is_rejected(self):
        from quant_platform.api.research_context import parse_window

        with pytest.raises(ValueError, match="10 years"):
            parse_window(None, "2100-12-31")

    def test_one_sided_huge_start_is_rejected(self):
        from quant_platform.api.research_context import parse_window

        with pytest.raises(ValueError, match="10 years"):
            parse_window("2000-01-01", None)

    def test_normal_override_still_works(self):
        from quant_platform.api.research_context import parse_window

        start, end = parse_window("2024-01-01", "2024-06-30")
        assert (start, end) == ("2024-01-01", "2024-06-30")


class TestEarlyAuth:
    def test_large_unauthenticated_body_is_rejected_before_read(
        self, app, monkeypatch,
    ):
        monkeypatch.setenv("QUANT_API_TOKEN", "secret-token")
        reads: list[int] = []
        scope = _raw_scope(
            path="/api/run", method="POST", content_length=5 * 1024 * 1024,
            client=("203.0.113.9", 5555),
        )
        import asyncio

        status, _ = asyncio.run(_drive(app, scope, reads=reads))
        assert status == 401
        assert reads == [], "body was read before authentication"


class TestLeaks:
    def test_events_endpoint_hides_audit_payloads(self, app):
        from quant_platform.core.events import get_event_bus

        bus = get_event_bus()
        bus.publish("audit.test", {"tenant_id": "fund_a", "details": {"x": 1}})
        client = TestClient(app)
        payload = client.get("/api/core/events", params={"topic": "audit.test"}).json()
        assert payload["events"] == []

    def test_metrics_method_label_is_bounded(self, app):
        from quant_platform.utils.metrics import get_metrics

        metrics = get_metrics()
        client = TestClient(app)
        metrics.reset()
        for index in range(6):
            client.request(f"X{index:02d}", "/api/health")
        snapshot = metrics.get_snapshot()
        methods = {
            entry.get("labels", {}).get("method")
            for key, entry in snapshot.items()
            if key.startswith("api_requests_total")
        }
        methods.discard(None)
        assert methods == {"OTHER"}

    def test_qmt_server_in_body_is_rejected(self, app):
        client = TestClient(app)
        response = client.post(
            "/api/trading/start",
            json={"broker": "qmt", "server": "169.254.169.254:80"},
        )
        assert response.status_code == 422
        assert "QMT_SERVER" in response.json()["detail"]

    def test_webhook_failure_log_hides_credentials(self, monkeypatch, caplog):
        from urllib.error import URLError

        from quant_platform.risk import healthcheck

        sink = healthcheck.WebhookAlertSink(
            "https://user:topsecret@alerts.example/hook/PATHSECRET?token=abc123",
        )

        def _fail(*args, **kwargs):
            raise URLError("down")

        monkeypatch.setattr(healthcheck.urllib.request, "urlopen", _fail)
        with caplog.at_level(logging.ERROR):
            sink.emit({"severity": "warning", "message": "x"})
        assert "topsecret" not in caplog.text
        assert "abc123" not in caplog.text
        assert "PATHSECRET" not in caplog.text
        assert "alerts.example" in caplog.text


def test_redaction_covers_list_valued_secrets():
    from quant_platform.utils.version_manager import REDACTED, redact_secrets

    redacted = redact_secrets({
        "api_tokens": ["a", "b"],
        "webhooks": ["https://user:pw@example.com/hook"],
        "plain": ["ok"],
    })
    assert redacted["api_tokens"] == REDACTED
    assert redacted["webhooks"] == [REDACTED]
    assert redacted["plain"] == ["ok"]


def test_safety_headers_present(app):
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"


class TestResourceBounds:
    def test_store_stats_does_not_leak_an_absolute_path(self, app):
        client = TestClient(app)
        payload = client.get("/api/core/store/stats").json()
        assert "db_path" not in payload
        assert "/" not in payload.get("db_name", "")
        assert "\\" not in payload.get("db_name", "")

    def test_portfolio_import_is_row_bounded(self, app):
        client = TestClient(app)
        rows = [{"code": f"{index:06d}", "hold_vol": 100} for index in range(10_001)]
        response = client.post("/api/portfolio/import", json={"data": rows})
        assert response.status_code == 413

    def test_l2_reads_do_not_lazily_start_the_provider(self, app, monkeypatch):
        from quant_platform.api import routes

        class _Stopped:
            is_running = False

        monkeypatch.setattr(routes, "_l2_provider", _Stopped())
        client = TestClient(app)
        response = client.get("/api/l2/book/600519")
        assert response.status_code == 503
        assert "POST /api/l2/start" in response.json()["detail"]


def test_redaction_covers_secret_query_parameters():
    from quant_platform.utils.version_manager import REDACTED, redact_secrets

    redacted = redact_secrets({
        "webhook": "https://alerts.example/hook?token=abc123",
        "plain_url": "https://alerts.example/hook?kind=prod",
    })
    assert redacted["webhook"] == REDACTED
    assert redacted["plain_url"] == "https://alerts.example/hook?kind=prod"


async def _drive_ws(app, host, origin=None):
    scope = {
        "type": "websocket", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "scheme": "ws", "path": "/api/ws", "raw_path": b"/api/ws",
        "query_string": b"", "root_path": "",
        "headers": [
            (b"host", host.encode()),
            (b"origin", b"http://" + (origin or host).encode()),
        ],
        "client": ("127.0.0.1", 4567), "server": ("testserver", 80),
        "subprotocols": [],
    }
    sent = []
    queue = [{"type": "websocket.connect"}]

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


class TestWebSocketRebinding:
    def test_rebound_host_is_closed_in_tokenless_mode(self, app, monkeypatch):
        """HTTP required a loopback Host; the WS stream must too."""
        monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
        monkeypatch.delenv("QUANT_ALLOWED_HOSTS", raising=False)
        import asyncio

        sent = asyncio.run(_drive_ws(app, "attacker.example:8000"))
        assert sent and sent[0]["type"] == "websocket.close"
        assert sent[0]["code"] == 1008

    def test_loopback_host_still_accepts(self, app, monkeypatch):
        monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
        import asyncio

        sent = asyncio.run(_drive_ws(app, "localhost:8000"))
        assert sent and sent[0]["type"] == "websocket.accept"

    def test_localhost_origin_may_target_ipv4_loopback(self, app, monkeypatch):
        """The UI opened at localhost must be able to use ws://127.0.0.1."""
        monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
        import asyncio

        sent = asyncio.run(_drive_ws(
            app, "127.0.0.1:8000", origin="localhost:8000",
        ))
        assert sent and sent[0]["type"] == "websocket.accept"


def test_portfolio_import_rejects_extra_columns(app):
    client = TestClient(app)
    rows = [{"code": "600519", "hold_vol": 100, f"junk{i}": i} for i in range(5)]
    response = client.post("/api/portfolio/import", json={"data": rows})
    assert response.status_code == 400
    assert "only 'code' and 'hold_vol'" in response.json()["detail"]


class TestActiveOrderCap:
    def test_active_orders_are_rejected_at_the_cap(self, monkeypatch):
        from quant_platform.execution import oms as oms_module

        monkeypatch.setattr(oms_module, "MAX_ORDERS", 5)
        om = oms_module.OrderManager(initial_cash=1_000_000)
        for index in range(5):
            om.create_order(f"{index:06d}", "buy", 100)
        with pytest.raises(ValueError, match="active-order cap"):
            om.create_order("999999", "buy", 100)


class TestGetRateLimitExemption:
    def test_loopback_ui_is_not_bucket_limited_for_get(self, monkeypatch):
        from quant_platform.app import RequestGuardMiddleware

        guard = RequestGuardMiddleware(app=None)
        scope = _raw_scope(
            path="/api/analysis/ic-decay", client=("127.0.0.1", 4444),
        )
        for _ in range(40):
            limited, _retry = guard._rate_limited(
                scope, "/api/analysis/ic-decay",
            )
            assert limited is False

    def test_remote_get_is_still_bucket_limited(self):
        from quant_platform.app import RequestGuardMiddleware

        guard = RequestGuardMiddleware(app=None)
        scope = _raw_scope(
            path="/api/analysis/ic-decay", client=("203.0.113.9", 4444),
        )
        results = [
            guard._rate_limited(scope, "/api/analysis/ic-decay")[0]
            for _ in range(25)
        ]
        assert results[-1] is True
