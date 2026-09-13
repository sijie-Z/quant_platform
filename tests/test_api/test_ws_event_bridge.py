"""Regression tests for the EventBus -> WebSocket bridge (BUG-05).

The engine publishes every trading event from a worker thread
(`LiveTradingEngine._run_loop` starts a `threading.Thread`). `_on_bus_event`
called `asyncio.get_event_loop()`, which raises `RuntimeError` on a thread with
no loop:

    except RuntimeError:
        return

so `order.filled`, `portfolio.snapshot`, `risk.status` and the rest were
dropped and the dashboard's live stream stayed empty *while the engine was
trading*. The only trace was the absence of messages.
"""

import threading
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes


@dataclass
class _Event:
    """Duck-typed stand-in; the bridge only reads .topic and .data."""
    topic: str
    data: dict


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_event_is_delivered_when_published_from_a_worker_thread(client):
    """The exact production shape: the publisher is not on the loop."""
    with client.websocket_connect("/api/ws") as ws:
        result = {}

        def publish():
            try:
                routes._on_bus_event(_Event("order.filled", {"code": "600519"}))
                result["ok"] = True
            except Exception as e:  # pragma: no cover - surfaced via assertion
                result["error"] = e

        t = threading.Thread(target=publish)
        t.start()
        t.join(timeout=10)

        assert result.get("ok"), f"publishing raised: {result.get('error')}"
        msg = ws.receive_text()

    assert "order.filled" in msg
    assert "600519" in msg


def test_the_loop_is_captured_by_the_websocket_connection(client):
    """Worker threads cannot look their own loop up, so it has to be remembered
    from a context that is on it."""
    routes._main_loop = None  # start from a clean slate
    with client.websocket_connect("/api/ws"):
        assert routes._main_loop is not None
        assert routes._main_loop.is_running()


def test_no_clients_means_no_error(client):
    """Publishing with nobody connected must be a no-op, not a crash."""
    routes._on_bus_event(_Event("order.filled", {"code": "600519"}))
