"""Ops wave 2 regressions — alert sinks, dedupe and the healthcheck handler (S4).

Health-check failures used to be an EventBus topic with no production
subscriber, so a failed pre-flight gate or a dead engine was invisible.
"""

from __future__ import annotations

import json
import urllib.error

import pytest
from quant_platform.core.events import EventBus
from quant_platform.risk.healthcheck import (
    AlertSink,
    HealthCheck,
    LoggingAlertSink,
    SystemBlockError,
    WebhookAlertSink,
    build_default_alert_sink,
    emit_alert,
    install_alert_handler,
    reset_alert_dedupe,
    set_alert_sink,
)
from quant_platform.utils.metrics import get_metrics


class RecordingSink(AlertSink):
    def __init__(self):
        self.alerts = []

    def emit(self, alert):
        self.alerts.append(alert)


class FailingBroker:
    def get_account(self):
        raise ConnectionError("broker down")

    def get_positions(self):
        raise ConnectionError("broker down")


@pytest.fixture(autouse=True)
def _clean_alert_state():
    reset_alert_dedupe()
    set_alert_sink(LoggingAlertSink())
    yield
    reset_alert_dedupe()
    set_alert_sink(None)


def test_healthcheck_failure_reaches_the_sink():
    sink = RecordingSink()
    set_alert_sink(sink)
    health = HealthCheck(event_bus=EventBus(), broker=FailingBroker())

    with pytest.raises(SystemBlockError):
        health.run_all_sync()

    messages = [alert["message"] for alert in sink.alerts]
    assert any("healthcheck failed" in message for message in messages)
    assert any(
        alert["severity"] == "critical" for alert in sink.alerts
    )


def test_identical_alerts_are_deduplicated_for_five_minutes():
    sink = RecordingSink()
    set_alert_sink(sink)

    assert emit_alert("critical", "same failure", source="test") is True
    assert emit_alert("critical", "same failure", source="test") is False
    assert emit_alert("critical", "different failure", source="test") is True

    assert len(sink.alerts) == 2

    reset_alert_dedupe()
    assert emit_alert("critical", "same failure", source="test") is True
    assert len(sink.alerts) == 3


def test_installed_handler_forwards_healthcheck_and_system_alerts():
    sink = RecordingSink()
    bus = EventBus()
    assert install_alert_handler(bus, sink=sink) is not None
    # Idempotent per bus.
    assert install_alert_handler(bus, sink=sink) is None

    bus.publish("system.healthcheck.fail", {
        "failed_checks": ["account_balance", "order_routing"],
        "checks": [],
    }, source="healthcheck")
    bus.publish("system.alert", {
        "severity": "critical", "message": "scheduler EOD skipped",
        "source": "scheduler",
    }, source="scheduler")

    messages = [alert["message"] for alert in sink.alerts]
    assert any("account_balance" in message for message in messages)
    assert "scheduler EOD skipped" in messages


def test_a_passing_healthcheck_does_not_alert():
    sink = RecordingSink()
    set_alert_sink(sink)
    bus = EventBus()
    install_alert_handler(bus, sink=sink)

    bus.publish("system.healthcheck.pass", {"checks": []}, source="healthcheck")

    assert sink.alerts == []


def test_webhook_sink_posts_json(monkeypatch):
    posted = {}

    class FakeResponse:
        def read(self, _n=-1):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout=None):
        posted["url"] = request.full_url
        posted["body"] = json.loads(request.data.decode("utf-8"))
        posted["timeout"] = timeout
        posted["content_type"] = request.headers.get("Content-type")
        return FakeResponse()

    monkeypatch.setattr("quant_platform.risk.healthcheck.urllib.request.urlopen",
                        fake_urlopen)
    metrics = get_metrics()
    before = metrics.get_snapshot().get(
        "alerts_webhook_delivered_total", {},
    ).get("value", 0)

    sink = WebhookAlertSink("http://alerts.example/hook", timeout=1.5)
    sink.emit({"severity": "critical", "message": "boom", "source": "test"})

    assert posted["url"] == "http://alerts.example/hook"
    assert posted["body"]["message"] == "boom"
    assert posted["timeout"] == 1.5
    assert posted["content_type"] == "application/json"
    after = metrics.get_snapshot().get(
        "alerts_webhook_delivered_total", {},
    ).get("value", 0)
    assert after == before + 1


def test_webhook_failure_is_logged_and_counted(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("quant_platform.risk.healthcheck.urllib.request.urlopen",
                        fake_urlopen)
    metrics = get_metrics()
    before = metrics.get_snapshot().get(
        "alerts_webhook_failures_total", {},
    ).get("value", 0)

    sink = WebhookAlertSink("http://alerts.example/hook", timeout=0.5)
    sink.emit({"severity": "error", "message": "unreachable", "source": "test"})

    after = metrics.get_snapshot().get(
        "alerts_webhook_failures_total", {},
    ).get("value", 0)
    assert after == before + 1


def test_default_sink_includes_a_webhook_only_when_configured(monkeypatch):
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    assert not any(
        isinstance(sink, WebhookAlertSink)
        for sink in build_default_alert_sink().sinks
    )

    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://alerts.example/hook")
    assert any(
        isinstance(sink, WebhookAlertSink)
        for sink in build_default_alert_sink().sinks
    )
