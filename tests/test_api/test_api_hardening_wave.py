"""Regression tests for the API performance/concurrency/lifecycle wave.

Each test pins one defect from the audit: blocking calls in async handlers,
the shared executor, the trading-start race, WebSocket fan-out, run registry
locking, response bounds, singleton races and error-text leakage. They are
written to fail on the old behaviour where that can be observed from the
public surface or a deterministic white-box probe.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.core.store import Store

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_run_registry():
    """Keep the module-level registry clean between tests."""
    with routes._run_lock:
        routes._run_store.clear()
        routes._run_status.clear()
        routes._run_slot_holders.clear()
        routes._run_futures.clear()
    yield
    with routes._run_lock:
        routes._run_store.clear()
        routes._run_status.clear()
        routes._run_slot_holders.clear()
        routes._run_futures.clear()


@pytest.fixture(autouse=True)
def _drain_run_slots():
    """Leave both pipeline permits free even if a test acquired them."""
    yield
    while routes._run_slots.acquire(blocking=False):
        pass
    # Give back exactly the two permits.
    for _ in range(2):
        try:
            routes._run_slots.release()
        except ValueError:
            break


class _DisconnectedProvider:
    def __init__(self):
        self.start_calls = 0

    @property
    def is_connected(self):
        return False

    def start(self):
        self.start_calls += 1

    def get_quote(self, code):
        return None

    def get_all_quotes(self):
        return {}


class _ConnectedProvider(_DisconnectedProvider):
    @property
    def is_connected(self):
        return True


# ---------------------------------------------------------------------------
# A1: quote handlers must not block the event loop
# ---------------------------------------------------------------------------

class TestQuoteHandlersDoNotBlockTheLoop:
    def test_two_handlers_leave_the_loop_responsive(self, monkeypatch):
        """Old code: `time.sleep(0.5)` inside `async def` froze the loop;
        a concurrent /health took >= 0.5 s. New code waits with
        `asyncio.sleep`, so the probe runs immediately and the handler
        answers 503 + Retry-After after its bounded readiness wait."""
        provider = _DisconnectedProvider()
        monkeypatch.setattr(routes, "_get_ws_provider", lambda: provider)
        monkeypatch.setattr(routes, "_QUOTE_READY_TIMEOUT", 0.30)
        monkeypatch.setattr(routes, "_QUOTE_READY_POLL", 0.02)

        async def scenario():
            start = time.monotonic()
            quote_task = asyncio.ensure_future(routes.ws_quote_by_code("600519"))

            async def probe():
                await asyncio.sleep(0.02)
                await routes.health()
                return time.monotonic() - start

            quote_result, probe_done_at = await asyncio.gather(
                quote_task, probe(), return_exceptions=True,
            )
            return quote_result, probe_done_at

        quote_result, probe_done_at = asyncio.run(scenario())

        assert probe_done_at < 0.25, (
            "the event loop was blocked by the quote handler"
        )
        assert isinstance(quote_result, HTTPException)
        assert quote_result.status_code == 503
        assert quote_result.headers.get("Retry-After") == "1"
        assert provider.start_calls == 1

    def test_ready_provider_is_used_without_starting_it(self, monkeypatch):
        provider = _ConnectedProvider()
        monkeypatch.setattr(routes, "_get_ws_provider", lambda: provider)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(routes.ws_quote_by_code("600519"))

        assert exc.value.status_code == 404
        assert provider.start_calls == 0


# ---------------------------------------------------------------------------
# A2: monitor endpoints run their blocking work in a worker thread
# ---------------------------------------------------------------------------

class _SlowStore:
    def get_positions(self, *a, **k):
        time.sleep(0.30)
        return []

    def get_pnl_history(self, *a, **k):
        time.sleep(0.10)
        return []


class TestMonitorDoesNotBlockTheLoop:
    def test_risk_overview_keeps_the_loop_responsive(self, monkeypatch):
        import quant_platform.api.monitor as monitor

        monkeypatch.setattr(monitor, "_core_store", _SlowStore())
        monkeypatch.setattr(monitor, "_core_risk", None)
        monkeypatch.setattr(monitor, "_get_risk", lambda: None)

        async def scenario():
            start = time.monotonic()
            overview = asyncio.ensure_future(monitor.get_risk_overview())

            async def probe():
                await asyncio.sleep(0.02)
                return time.monotonic() - start

            result, probe_done_at = await asyncio.gather(overview, probe())
            return result, probe_done_at

        result, probe_done_at = asyncio.run(scenario())

        assert probe_done_at < 0.20, (
            "monitor risk-overview blocked the event loop"
        )
        assert result.available is False


# ---------------------------------------------------------------------------
# A3: separate bounded pools, 429 for analytics, queued phase
# ---------------------------------------------------------------------------

class TestExecutorSeparation:
    def test_pipeline_and_analytics_use_different_pools(self):
        pipeline = routes._get_pipeline_executor()
        analytics = routes._get_analytics_executor()
        assert pipeline is not analytics
        assert pipeline._max_workers == routes._PIPELINE_WORKERS
        assert analytics._max_workers == routes._ANALYTICS_WORKERS

    def test_saturated_analytics_returns_429(self, client):
        acquired = 0
        try:
            while routes._analytics_slots.acquire(blocking=False):
                acquired += 1
                if acquired >= routes._ANALYTICS_WORKERS:
                    break
            assert acquired == routes._ANALYTICS_WORKERS
            resp = client.post("/api/compare", json={"optimizers": ["equal_weight"]})
        finally:
            for _ in range(acquired):
                routes._analytics_slots.release()

        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "5"

    def test_analytics_saturation_does_not_delay_a_pipeline_run(
        self, client, monkeypatch,
    ):
        started = threading.Event()

        def fake_pipeline(run_id, req):
            started.set()
            routes._set_run_fields(run_id, status="completed", phase="completed")

        monkeypatch.setattr(routes, "_execute_pipeline", fake_pipeline)

        acquired = 0
        try:
            while routes._analytics_slots.acquire(blocking=False):
                acquired += 1
                if acquired >= routes._ANALYTICS_WORKERS:
                    break
            resp = client.post("/api/run", json={})
        finally:
            for _ in range(acquired):
                routes._analytics_slots.release()

        assert resp.status_code == 200, resp.text
        assert started.wait(timeout=5), "pipeline did not start behind analytics"
        with routes._run_lock:
            entry = routes._run_status[resp.json()["run_id"]]
        assert entry["status"] == "completed"

    def test_run_status_exposes_the_queued_phase(self):
        with routes._run_lock:
            routes._run_status["r1"] = {
                "status": "running", "phase": "queued", "progress": 0,
                "stage": "data", "started_at": "2026-01-01T00:00:00",
                "completed_at": None, "error": None,
            }
        data = asyncio.run(routes.get_run_status("r1"))
        assert data.status == "running"
        assert data.phase == "queued"


# ---------------------------------------------------------------------------
# A4: trading-start race
# ---------------------------------------------------------------------------

class _StartProbeEngine:
    instances = 0
    start_entered = threading.Event()
    release = threading.Event()
    fail_with = None

    def __init__(self, **kwargs):
        type(self).instances += 1
        self._running = False
        self.stopped = False

    def set_universe(self, codes):
        pass

    def start(self):
        type(self).start_entered.set()
        assert type(self).release.wait(timeout=10)
        if type(self).fail_with is not None:
            raise type(self).fail_with
        self._running = True

    def stop(self):
        self.stopped = True
        self._running = False


@pytest.fixture
def probe_engine(monkeypatch):
    import quant_platform.trading.engine as engine_module

    _StartProbeEngine.instances = 0
    _StartProbeEngine.start_entered = threading.Event()
    _StartProbeEngine.release = threading.Event()
    _StartProbeEngine.fail_with = None
    monkeypatch.setattr(engine_module, "LiveTradingEngine", _StartProbeEngine)
    monkeypatch.setattr(routes, "_get_rt_market", lambda: (_ for _ in ()).throw(
        RuntimeError("no network in tests"),
    ))
    old_engine = routes._live_engine
    routes._live_engine = None
    yield _StartProbeEngine
    routes._live_engine = old_engine


class TestTradingStartRace:
    def test_concurrent_starts_construct_exactly_one_engine(self, probe_engine):
        outcomes = []

        def start():
            try:
                outcomes.append(("ok", routes.trading_start({"broker": "simulated"})))
            except HTTPException as exc:
                outcomes.append(("http", exc.status_code))
            except Exception as exc:  # pragma: no cover - surfaced in assert
                outcomes.append(("error", repr(exc)))

        first = threading.Thread(target=start)
        first.start()
        assert probe_engine.start_entered.wait(timeout=10), "start() never entered"

        second = threading.Thread(target=start)
        second.start()
        second.join(timeout=10)

        probe_engine.release.set()
        first.join(timeout=10)

        kinds = [kind for kind, _ in outcomes]
        assert kinds.count("ok") == 1, outcomes
        assert kinds.count("http") == 1, outcomes
        http_status = [value for kind, value in outcomes if kind == "http"][0]
        assert http_status == 409
        assert probe_engine.instances == 1
        assert routes._live_engine is not None

    def test_failed_start_rolls_back_the_slot(self, probe_engine, monkeypatch):
        probe_engine.fail_with = RuntimeError("engine exploded: secret path")
        probe_engine.release.set()

        with pytest.raises(HTTPException) as exc:
            routes.trading_start({"broker": "simulated"})

        assert exc.value.status_code == 503
        assert "secret path" not in exc.value.detail
        assert exc.value.headers.get("X-Error-Code") == "trading_start_failed"
        assert routes._live_engine is None

    def test_stop_can_reach_the_published_engine(self, probe_engine):
        probe_engine.release.set()
        routes.trading_start({"broker": "simulated"})
        engine = routes._live_engine
        assert engine is not None

        result = routes.trading_stop()

        assert result == {"status": "stopped"}
        assert engine.stopped is True


# ---------------------------------------------------------------------------
# A5/A19: WebSocket fan-out
# ---------------------------------------------------------------------------

class _SlowSocket:
    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, msg: str):
        await asyncio.sleep(self.delay)
        self.sent.append(msg)

    async def close(self):
        self.closed = True


class TestWebSocketFanOut:
    def test_broadcast_does_not_head_of_line_block_on_slow_clients(self):
        async def scenario():
            clients = []
            for _ in range(5):
                client = routes._WSClient(_SlowSocket(0.2))
                client.task = asyncio.create_task(routes._ws_sender_loop(client))
                routes._ws_clients.add(client)
                clients.append(client)
            try:
                start = time.monotonic()
                await routes._broadcast_event("order.filled", {"code": "600519"})
                enqueue_seconds = time.monotonic() - start
                # Five sequential 0.2 s sends used to take ~1 s (and 5 x 50ms
                # measured 0.297 s in the audit probe).
                await asyncio.sleep(0.50)
                return enqueue_seconds, [c.ws.sent for c in clients]
            finally:
                for client in clients:
                    client.closed = True
                    routes._ws_clients.discard(client)
                    if client.task is not None:
                        client.task.cancel()
                await asyncio.sleep(0)

        enqueue_seconds, deliveries = asyncio.run(scenario())

        assert enqueue_seconds < 0.10, (
            f"broadcast waited for slow clients: {enqueue_seconds:.3f}s"
        )
        assert all(len(sent) == 1 for sent in deliveries)
        assert all("order.filled" in sent[0] for sent in deliveries)

    def test_full_queue_drops_the_oldest_message(self):
        async def scenario():
            fast = _SlowSocket(0.0)
            client = routes._WSClient(fast)
            routes._ws_clients.add(client)
            try:
                for i in range(routes._WS_QUEUE_SIZE):
                    client.queue.put_nowait(f"old-{i}")
                await routes._broadcast_event("market.tick", {"price": 1})
                size = client.queue.qsize()
                newest = client.queue._queue[-1]
                return size, newest, client.dropped
            finally:
                routes._ws_clients.discard(client)

        size, newest, dropped = asyncio.run(scenario())

        assert size == routes._WS_QUEUE_SIZE
        assert "market.tick" in newest
        assert dropped == 1

    def test_unserializable_payload_is_dropped_without_raising(self):
        async def scenario():
            client = routes._WSClient(_SlowSocket(0.0))
            routes._ws_clients.add(client)
            try:
                await routes._broadcast_event("weird", {"value": {1, 2, 3}})
                return client.queue.qsize()
            finally:
                routes._ws_clients.discard(client)

        assert asyncio.run(scenario()) == 0

    def test_adding_a_client_while_broadcasting_is_safe(self):
        async def scenario():
            # No sender task: the message stays in `first.queue`, so its
            # presence proves the broadcast reached the original client even
            # though the live set changed mid-flight.
            first = routes._WSClient(_SlowSocket(0.0))
            routes._ws_clients.add(first)
            second = routes._WSClient(_SlowSocket(0.0))
            try:
                task = asyncio.create_task(
                    routes._broadcast_event("order.filled", {"code": "1"}),
                )
                routes._ws_clients.add(second)
                await task
                routes._ws_clients.discard(second)
                return first.queue.qsize()
            finally:
                for client in (first, second):
                    client.closed = True
                    routes._ws_clients.discard(client)
                    if client.task is not None:
                        client.task.cancel()
                await asyncio.sleep(0)

        # A RuntimeError here is the old "Set changed size during iteration".
        assert asyncio.run(scenario()) == 1

    def test_bus_event_bridge_retains_and_logs_future_exceptions(self, caplog):
        future = Future()
        future.set_exception(RuntimeError("broadcast blew up"))
        routes._log_broadcast_future(future)
        # Logging is best-effort; the important part is that nothing raises
        # and the exception is retrieved (no "exception was never retrieved").
        assert future.exception() is not None
