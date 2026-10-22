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
            "https://user:topsecret@alerts.example/hook?token=abc123",
        )

        def _fail(*args, **kwargs):
            raise URLError("down")

        monkeypatch.setattr(healthcheck.urllib.request, "urlopen", _fail)
        with caplog.at_level(logging.ERROR):
            sink.emit({"severity": "warning", "message": "x"})
        assert "topsecret" not in caplog.text
        assert "abc123" not in caplog.text
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
