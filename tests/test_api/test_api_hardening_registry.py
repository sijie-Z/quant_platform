"""Store indexes, store-stats endpoint and run-registry regressions.

Companion to test_api_hardening_wave.py (audit items A6, A8, A9, A10).
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.core.store import Store


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _restore_run_registry():
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
    yield
    while routes._run_slots.acquire(blocking=False):
        pass
    for _ in range(routes._RUN_LIMIT):
        try:
            routes._run_slots.release()
        except ValueError:
            break


# ---------------------------------------------------------------------------
# A6: query-plan indexes
# ---------------------------------------------------------------------------

def _query_plan(store: Store, sql: str, params: tuple = ()) -> list[str]:
    with store._conn() as conn:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return [row[3] for row in rows]


class TestStoreIndexes:
    @pytest.fixture
    def store(self, tmp_path):
        return Store(str(tmp_path / "indexes.db"))

    def test_trades_default_query_sorts_from_an_index(self, store):
        plan = _query_plan(
            store,
            "SELECT * FROM trades WHERE 1=1 "
            "ORDER BY executed_at DESC LIMIT ?", (100,),
        )
        assert not any("TEMP B-TREE" in step for step in plan), plan

    def test_trades_by_code_searches_the_composite_index(self, store):
        plan = _query_plan(
            store,
            "SELECT * FROM trades WHERE 1=1 AND code = ? "
            "ORDER BY executed_at DESC LIMIT ?", ("600519", 100),
        )
        assert any("SEARCH" in step for step in plan), plan
        assert not any("TEMP B-TREE" in step for step in plan), plan

    def test_events_by_topic_does_not_sort_in_a_temp_tree(self, store):
        plan = _query_plan(
            store,
            "SELECT * FROM events WHERE 1=1 AND topic = ? "
            "ORDER BY timestamp DESC LIMIT ?", ("order.filled", 100),
        )
        assert any("SEARCH" in step for step in plan), plan
        assert not any("TEMP B-TREE" in step for step in plan), plan

    def test_signals_default_and_filtered_queries(self, store):
        default_plan = _query_plan(
            store,
            "SELECT * FROM signals WHERE 1=1 "
            "ORDER BY generated_at DESC LIMIT ?", (100,),
        )
        filtered_plan = _query_plan(
            store,
            "SELECT * FROM signals WHERE 1=1 AND consumed = ? "
            "ORDER BY generated_at DESC LIMIT ?", (0, 100),
        )
        assert not any("TEMP B-TREE" in step for step in default_plan)
        assert any("SEARCH" in step for step in filtered_plan)
        assert not any("TEMP B-TREE" in step for step in filtered_plan)

    def test_sessions_use_the_started_at_index(self, store):
        plan = _query_plan(
            store,
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (20,),
        )
        assert not any("TEMP B-TREE" in step for step in plan), plan

    def test_orders_tenant_status_composite(self, store):
        plan = _query_plan(
            store,
            "SELECT * FROM orders WHERE 1=1 AND tenant_id = ? AND status = ? "
            "ORDER BY created_at DESC LIMIT ?", ("default", "filled", 100),
        )
        assert any("SEARCH" in step for step in plan), plan
        assert not any("TEMP B-TREE" in step for step in plan), plan

    def test_indexes_are_created_for_existing_databases_and_data_survives(
        self, tmp_path,
    ):
        db = str(tmp_path / "existing.db")
        first = Store(db)
        first.save_trade({
            "trade_id": "t1", "order_id": "o1", "code": "600519",
            "side": "buy", "quantity": 100, "price": 10,
            "executed_at": "2024-01-01T10:00:00",
        })
        with first._conn() as conn:
            for name in (
                "idx_trades_code_executed", "idx_trades_executed",
                "idx_events_topic_timestamp", "idx_signals_consumed_generated",
                "idx_signals_generated", "idx_sessions_started",
                "idx_orders_tenant_status_created",
            ):
                conn.execute(f"DROP INDEX IF EXISTS {name}")

        second = Store(db)
        names = {
            row[0]
            for row in second._get_connection().execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert {
            "idx_trades_code_executed", "idx_trades_executed",
            "idx_events_topic_timestamp", "idx_signals_consumed_generated",
            "idx_signals_generated", "idx_sessions_started",
            "idx_orders_tenant_status_created",
        } <= names
        assert [r["trade_id"] for r in second.get_trades()] == ["t1"]


# ---------------------------------------------------------------------------
# A8: /api/postgres/stats reuses the process store and caches
# ---------------------------------------------------------------------------

class _CountingStore(Store):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.stats_calls = 0

    def get_stats(self):
        self.stats_calls += 1
        return super().get_stats()


class TestStoreStatsEndpoint:
    @pytest.fixture(autouse=True)
    def _reset_stats_cache(self):
        with routes._stats_cache_lock:
            routes._stats_cache.update(key=None, at=0.0, payload=None)

    def test_no_store_is_constructed_per_request(self, client, tmp_path, monkeypatch):
        store = Store(str(tmp_path / "stats.db"))
        monkeypatch.setattr(routes, "_core_store", store)

        def boom(self, *args, **kwargs):
            raise AssertionError("Store constructed per /postgres/stats request")

        monkeypatch.setattr(Store, "__init__", boom)
        resp = client.get("/api/postgres/stats")

        assert resp.status_code == 200
        assert resp.json()["backend"] == "sqlite"

    def test_stats_are_cached_briefly(self, client, tmp_path, monkeypatch):
        store = _CountingStore(str(tmp_path / "stats2.db"))
        monkeypatch.setattr(routes, "_core_store", store)

        assert client.get("/api/postgres/stats").status_code == 200
        assert client.get("/api/postgres/stats").status_code == 200

        assert store.stats_calls == 1

    def test_backend_is_reported_truthfully(self, client, tmp_path, monkeypatch):
        store = _CountingStore(str(tmp_path / "stats3.db"))
        monkeypatch.setattr(routes, "_core_store", store)
        monkeypatch.setattr(
            store, "get_stats", lambda: {"backend": "postgresql", "orders": 1},
        )

        data = client.get("/api/postgres/stats").json()

        assert data["backend"] == "postgresql"
        assert data["orders"] == 1


# ---------------------------------------------------------------------------
# A9: run registry lock and status-aware cleanup
# ---------------------------------------------------------------------------

class TestRunRegistry:
    def test_cleanup_never_evicts_running_runs(self):
        with routes._run_lock:
            for i in range(120):
                routes._run_status[f"r{i}"] = {
                    "status": "running",
                    "started_at": f"2026-01-01T00:{i:02d}:00",
                }

        routes._cleanup_old_runs()

        with routes._run_lock:
            assert len(routes._run_status) == 120

    def test_cleanup_evicts_old_finished_runs(self):
        now = "2026-01-01T00:00:00"
        with routes._run_lock:
            for i in range(60):
                rid = f"done{i}"
                routes._run_status[rid] = {
                    "status": "completed", "started_at": now,
                    "completed_at": f"2026-01-01T00:{i:02d}:00",
                }
                routes._run_store[rid] = {"performance": {}}

        routes._cleanup_old_runs()

        with routes._run_lock:
            assert len(routes._run_status) <= routes._MAX_RUNS
            assert "done0" not in routes._run_status

    def test_status_endpoint_tolerates_partial_entries(self):
        with routes._run_lock:
            routes._run_status["partial"] = {"status": "running"}

        data = asyncio.run(routes.get_run_status("partial"))

        assert data.status == "running"
        assert data.progress == 0
        assert data.stage == "data"

    def test_latest_run_id_uses_a_snapshot(self):
        with routes._run_lock:
            routes._run_status["old"] = {
                "started_at": "2026-01-01T00:00:00", "status": "completed",
            }
            routes._run_status["new"] = {
                "started_at": "2026-06-01T00:00:00", "status": "completed",
            }
        assert routes._latest_run_id() == "new"


# ---------------------------------------------------------------------------
# A10: retained futures, unexpected-exit handling, guaranteed slot release
# ---------------------------------------------------------------------------

class TestRunFuturesAndSlots:
    def test_done_callback_marks_unexpected_exit_failed(self):
        with routes._run_lock:
            routes._run_status["boom"] = {
                "status": "running", "phase": "executing", "progress": 10,
                "stage": "data", "started_at": "2026-01-01T00:00:00",
            }
        future = Future()
        future.set_exception(RuntimeError("worker blew up"))

        routes._on_run_future_done("boom", future)

        with routes._run_lock:
            entry = routes._run_status["boom"]
        assert entry["status"] == "failed"
        assert entry["phase"] == "failed"
        assert "RuntimeError" in entry["error"]

    def test_completed_run_is_not_overwritten_by_the_callback(self):
        with routes._run_lock:
            routes._run_status["ok"] = {
                "status": "completed", "phase": "completed",
                "started_at": "2026-01-01T00:00:00",
            }
        future = Future()
        future.set_exception(RuntimeError("late error"))

        routes._on_run_future_done("ok", future)

        with routes._run_lock:
            assert routes._run_status["ok"]["status"] == "completed"

    def test_slot_is_released_even_when_the_pipeline_body_fails(
        self, monkeypatch,
    ):
        assert routes._run_slots.acquire(blocking=False)
        with routes._run_lock:
            routes._run_slot_holders.add("slotty")
            routes._run_status["slotty"] = {
                "status": "running", "started_at": "2026-01-01T00:00:00",
            }

        def explode(_overrides):
            raise RuntimeError("pipeline setup failed")

        monkeypatch.setattr(routes, "_get_config_with_overrides", explode)
        try:
            routes._execute_pipeline("slotty", routes.RunRequest())
            with routes._run_lock:
                assert routes._run_status["slotty"]["status"] == "failed"
            assert routes._run_slots.acquire(blocking=False) is True
            routes._run_slots.release()
        finally:
            with routes._run_lock:
                routes._run_slot_holders.discard("slotty")

    def test_third_run_is_rejected_when_both_slots_are_held(self, client):
        acquired = []
        try:
            for _ in range(routes._RUN_LIMIT):
                assert routes._run_slots.acquire(blocking=False)
                acquired.append(True)
            resp = client.post("/api/run", json={})
        finally:
            for _ in acquired:
                routes._run_slots.release()

        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "5"

    def test_run_response_carries_phase_and_frontend_status(self, client, monkeypatch):
        release = threading.Event()

        def fake_pipeline(run_id, req):
            # Hold the worker so the response is built before completion.
            release.wait(timeout=10)
            routes._set_run_fields(
                run_id, status="completed", phase="completed", progress=100,
            )

        monkeypatch.setattr(routes, "_execute_pipeline", fake_pipeline)
        try:
            resp = client.post("/api/run", json={})
            data = resp.json()
        finally:
            release.set()

        assert resp.status_code == 200, resp.text
        assert data["status"] == "running"
        assert data["phase"] in {"queued", "executing"}
