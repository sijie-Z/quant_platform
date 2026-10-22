"""FastAPI routes wrapping the quant platform pipeline."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import threading
import time
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from quant_platform.api.research_context import (
    ANALYSIS_WINDOW_SESSIONS,
    ResearchContext,
    _key_lock_for,
    derived_cache_get,
    derived_cache_put,
    derived_key,
    get_research_context,
    parse_window,
)
from quant_platform.api.schemas import (
    AttributionItem,
    BarraCovarianceResponse,
    BarraDecomposeRequest,
    BarraRiskResponse,
    ChartData,
    CompareRequest,
    CompareResult,
    ConfigInfo,
    DrawdownPeriod,
    ExposureInfo,
    FactorICItem,
    FactorICStatsResponse,
    FactorScatterItem,
    FundamentalMetricsResponse,
    FundamentalRankRequest,
    FundamentalRankResponse,
    FundamentalScreenRequest,
    FundamentalScreenResponse,
    HoldingItem,
    ICMonitorRequest,
    ICMonitorSummary,
    Level2StatsResponse,
    MLPerformanceResponse,
    MLSignalResponse,
    MLTrainRequest,
    OrderBookResponse,
    ParallelSweepRequest,
    ParallelSweepResponse,
    PerformanceMetrics,
    PostgresStoreStatsResponse,
    RealtimeQuoteResponse,
    RiskDecomposeRequest,
    RiskMetrics,
    RunRequest,
    RunResult,
    RunStatus,
    ScreenRequest,
    ScreenResponse,
    ScreenStockInfo,
    StressTestItem,
    SweepRequest,
    SweepResult,
    TickDataResponse,
    TradeFlowResponse,
    TurnoverItem,
    VWAPResponse,
    WalkForwardRequest,
    WebSocketStatsResponse,
)
from quant_platform.core.context import TenantContext
from quant_platform.core.instrument import normalize_symbol
from quant_platform.execution.oms import OrderManager
from quant_platform.main import (
    _compute_factors,
    _execution_gates,
    _generate_signal,
    _load_data,
    _run_backtest,
)
from quant_platform.utils.config import load_config
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Executors: pipeline work and analytics work get separate, bounded pools.
#
# One shared two-worker pool meant a `/compare` request could sit in front of
# an accepted pipeline run, and a burst of analytics requests queued without
# bound. Pipeline runs keep the existing 2-active-run cap; analytics requests
# are rejected with 429 when both analytics workers are busy.
# ---------------------------------------------------------------------------

_PIPELINE_WORKERS = 2
_ANALYTICS_WORKERS = 2
_executors_lock = threading.Lock()
_pipeline_executor: ThreadPoolExecutor | None = None
_analytics_executor: ThreadPoolExecutor | None = None
_analytics_slots = threading.BoundedSemaphore(_ANALYTICS_WORKERS)


def _make_pipeline_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_PIPELINE_WORKERS, thread_name_prefix="pipeline",
    )


def _make_analytics_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_ANALYTICS_WORKERS, thread_name_prefix="analytics",
    )


def _get_pipeline_executor() -> ThreadPoolExecutor:
    global _pipeline_executor
    executor = _pipeline_executor
    if executor is None:
        with _executors_lock:
            if _pipeline_executor is None:
                _pipeline_executor = _make_pipeline_executor()
            executor = _pipeline_executor
    return executor


def _get_analytics_executor() -> ThreadPoolExecutor:
    global _analytics_executor
    executor = _analytics_executor
    if executor is None:
        with _executors_lock:
            if _analytics_executor is None:
                _analytics_executor = _make_analytics_executor()
            executor = _analytics_executor
    return executor


def ensure_executors() -> None:
    """Create the executors if they are absent.

    Called at application startup so a process that shut its executors down
    (lifespan exit) and then started again -- as repeated `TestClient`
    lifespans do -- still has live pools.
    """
    _get_pipeline_executor()
    _get_analytics_executor()


def _shutdown_executor(executor: ThreadPoolExecutor | None) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except Exception:
        logger.exception("Executor shutdown failed")


def shutdown_executors() -> None:
    """Stop both pools. Does not wait past each pool's own in-flight work.

    Python cannot kill a running thread; `cancel_futures=True` only cancels
    work that has not started. A pipeline already inside `_execute_pipeline`
    keeps running until it returns, which is the documented limitation.
    """
    global _pipeline_executor, _analytics_executor
    with _executors_lock:
        pipeline, analytics = _pipeline_executor, _analytics_executor
        _pipeline_executor = None
        _analytics_executor = None
    _shutdown_executor(analytics)
    _shutdown_executor(pipeline)


# In-memory store for run results. Every access goes through `_run_lock`:
# cleanup runs on pipeline worker threads while HTTP handlers iterate the
# same dicts on the loop, and a status-blind cleanup could evict a run that
# was still executing.
_run_lock = threading.RLock()
_run_store: dict[str, dict[str, Any]] = {}
_run_status: dict[str, dict[str, Any]] = {}
_MAX_RUNS = 50  # Keep at most this many finished runs in memory
_RUN_LIMIT = 2
_run_slots = threading.BoundedSemaphore(_RUN_LIMIT)
_run_slot_holders: set[str] = set()
_run_futures: dict[str, Any] = {}


class RunStatusWithPhase(RunStatus):
    """RunStatus plus an additive phase field.

    `status` keeps the values the UI already understands ("running" while a
    run is queued or executing); `phase` distinguishes the two so a client is
    not told a run is executing when it is still waiting for a worker.
    """

    phase: str = "executing"


# WebSocket connection manager for real-time updates.
#
# Each client gets its own bounded queue and sender task. The old broadcast
# iterated `_ws_clients` across `await ws.send_text(...)`, so a connect or
# disconnect during a send raised "Set changed size during iteration" and
# silently dropped the message for every remaining client; a single slow
# client also delayed all the others.
_WS_QUEUE_SIZE = 256
_WS_SEND_TIMEOUT = 5.0
_WS_MAX_CLIENTS = 100
_ws_pending = 0  # handshakes between the cap check and `accept()`
_HIGH_RATE_TOPICS = frozenset({"market.tick"})
_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost", "testclient")


class _WSClient:
    """One connected WebSocket plus its bounded outbound queue."""

    __slots__ = ("ws", "queue", "task", "closed", "dropped")

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_WS_QUEUE_SIZE)
        self.task: asyncio.Task | None = None
        self.closed = False
        self.dropped = 0


_ws_clients: set[_WSClient] = set()

# The server's event loop, captured by the first WebSocket connection. Events
# are published from worker threads, which cannot look their own loop up.
_main_loop: asyncio.AbstractEventLoop | None = None


def _serialize_ws_payload(payload: dict) -> str | None:
    """Serialize a broadcast payload, logging instead of raising.

    `json.dumps` used to sit outside the try block, so one unserializable
    event payload (e.g. a numpy scalar) killed the whole broadcast.
    """
    try:
        return json.dumps(payload)
    except (TypeError, ValueError):
        logger.exception("Dropping WebSocket broadcast: payload is not serializable")
        return None


def _enqueue_ws_message(msg: str, topic: str = "") -> None:
    """Fan a message out to every live client without awaiting.

    Never awaits, so the live set cannot change under it and a slow client
    cannot head-of-line block the rest. When a client's queue is full the
    oldest message is dropped: for high-rate topics like `market.tick` the
    newest tick is worth more than the backlog, and a bounded queue is what
    keeps one stalled browser from growing server memory without limit.
    """
    for client in tuple(_ws_clients):
        if client.closed:
            _ws_clients.discard(client)
            continue
        try:
            client.queue.put_nowait(msg)
            continue
        except asyncio.QueueFull:
            pass
        try:
            client.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            client.queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass
        client.dropped += 1
        if client.dropped == 1 or client.dropped % 1000 == 0:
            logger.warning(
                "WebSocket client queue full (dropped=%d, topic=%s); "
                "dropping the oldest message",
                client.dropped, topic or "status",
            )


async def _broadcast_status(run_id: str, status: dict):
    """Push status update to all connected WebSocket clients."""
    if not _ws_clients:
        return
    msg = _serialize_ws_payload({"type": "status", "run_id": run_id, **status})
    if msg is None:
        return
    _enqueue_ws_message(msg, topic="status")


async def _broadcast_event(event_type: str, data: dict):
    """Push an event to all connected WebSocket clients."""
    if not _ws_clients:
        return
    msg = _serialize_ws_payload({
        "type": "event", "event": event_type, "data": data,
        "ts": datetime.now().isoformat(),
    })
    if msg is None:
        return
    _enqueue_ws_message(
        msg,
        topic=event_type if event_type in _HIGH_RATE_TOPICS else "",
    )


async def _ws_sender_loop(client: _WSClient) -> None:
    """Deliver one client's queued messages until it fails or disconnects."""
    try:
        while True:
            msg = await client.queue.get()
            if client.closed:
                break
            try:
                await asyncio.wait_for(
                    client.ws.send_text(msg), timeout=_WS_SEND_TIMEOUT,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "WebSocket send failed or timed out (%.0fs); closing client: %s",
                    _WS_SEND_TIMEOUT, exc,
                )
                break
    except asyncio.CancelledError:
        pass
    finally:
        client.closed = True
        _ws_clients.discard(client)
        try:
            await asyncio.wait_for(client.ws.close(), timeout=1.0)
        except Exception:
            pass


async def _close_all_ws_clients() -> None:
    """Close every live WebSocket (lifespan shutdown)."""
    for client in tuple(_ws_clients):
        client.closed = True
        _ws_clients.discard(client)
        task = client.task
        if task is not None and not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(client.ws.close(), timeout=1.0)
        except Exception:
            pass


def _log_broadcast_future(future) -> None:
    """Surface exceptions from a fire-and-forget broadcast future.

    The returned future used to be discarded, so a RuntimeError raised by the
    broadcast coroutine was never retrieved or logged.
    """
    try:
        exc = future.exception()
    except Exception:
        return
    if exc is not None:
        logger.error("WebSocket broadcast failed: %s", exc, exc_info=exc)


def _submit_broadcast(coro, loop: asyncio.AbstractEventLoop | None) -> None:
    if loop is None:
        coro.close()
        return
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        coro.close()
        return
    future.add_done_callback(_log_broadcast_future)


def _remember_loop() -> asyncio.AbstractEventLoop | None:
    """Record the server's event loop, from a context that is on it.

    Called from the WebSocket endpoint, which is guaranteed to run on the loop
    before any client could receive an event. Worker threads cannot do this:
    `asyncio.get_running_loop()` raises there, which is the whole problem.
    """
    global _main_loop
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return _main_loop
    if _main_loop is None or _main_loop.is_closed():
        _main_loop = running
    return _main_loop


def _on_bus_event(event):
    """EventBus handler -- bridges internal events to WebSocket clients.

    The trading engine publishes from a worker thread
    (`LiveTradingEngine._run_loop`), and `asyncio.get_event_loop()` raises
    `RuntimeError` there. The old code caught that and returned, so every
    `order.filled` / `portfolio.snapshot` / `risk.status` was dropped and the
    dashboard's live stream stayed empty while the engine traded -- silently,
    since the only trace was the absence of messages.

    Handing the coroutine to the server's loop works from any thread, so the
    event no longer depends on which thread published it. The returned future
    is retained with a done callback that logs, instead of vanishing.
    """
    if not _ws_clients:
        return
    loop = _main_loop
    if loop is None or loop.is_closed() or not loop.is_running():
        return
    _submit_broadcast(_broadcast_event(event.topic, event.data), loop)


def _latest_run_id() -> str:
    """The most recently started run.

    `_run_store` holds only result payloads -- performance, chart_data,
    factors -- and no timestamps at all. The endpoints below compared
    `_run_store[k].get("started_at", "")` for every key, i.e. empty string
    against empty string, so `max` returned the first-inserted run and IC decay
    and factor correlation always served the *oldest* run. `started_at` lives
    in `_run_status`.

    A snapshot is taken under `_run_lock`: cleanup runs on pipeline worker
    threads and mutates the same dict.
    """
    with _run_lock:
        if not _run_status:
            raise HTTPException(400, "No completed run found. Run pipeline first.")
        return max(
            _run_status,
            key=lambda k: (_run_status.get(k) or {}).get("started_at", "") or "",
        )


_TERMINAL_RUN_STATUSES = ("completed", "failed")


def _cleanup_old_runs():
    """Bound both the result store and the status store.

    `_run_store` only receives successful runs, so cleaning it alone let every
    failed run keep a full traceback in `_run_status` for the process
    lifetime.

    Only finished runs are evicted. The old final loop dropped the oldest
    entries regardless of status, so a cleanup triggered by >100 accumulated
    entries could delete a *running* run record (reproduced: 120 running
    entries -> 20 evicted).
    """
    with _run_lock:
        if len(_run_store) <= _MAX_RUNS and len(_run_status) <= _MAX_RUNS:
            return

        finished = [
            (rid, s.get("completed_at") or s.get("started_at") or "")
            for rid, s in _run_status.items()
            if s.get("status") in _TERMINAL_RUN_STATUSES
        ]
        finished.sort(key=lambda x: x[1])
        excess = max(
            len(_run_store) - _MAX_RUNS,
            len(_run_status) - _MAX_RUNS,
            1,
        )
        for rid, _ in finished[:excess]:
            _run_store.pop(rid, None)
            _run_status.pop(rid, None)

        stuck = sum(
            1 for s in _run_status.values()
            if s.get("status") not in _TERMINAL_RUN_STATUSES
        )
        if stuck > _MAX_RUNS * 2:
            # Deliberately no eviction: a running/queued record is the handle
            # a client polls, and deleting it turns a slow run into a 404.
            logger.error(
                "Run registry holds %d non-terminal entries (cap %d); "
                "refusing to evict active runs",
                stuck, _MAX_RUNS * 2,
            )


# ---------------------------------------------------------------------------
# Run registry helpers (all mutations under `_run_lock`)
# ---------------------------------------------------------------------------

def _set_run_fields(run_id: str, **fields: Any) -> dict | None:
    """Update a status entry if it still exists; returns a snapshot."""
    with _run_lock:
        entry = _run_status.get(run_id)
        if entry is None:
            return None
        entry.update(fields)
        return dict(entry)


def _run_status_snapshot(run_id: str) -> dict | None:
    with _run_lock:
        entry = _run_status.get(run_id)
        return dict(entry) if entry is not None else None


def _release_run_slot(run_id: str) -> None:
    """Release a pipeline slot exactly once.

    Called from `_execute_pipeline`'s guaranteed finally *and* from the
    executor future's done callback; the holder set makes the second call a
    no-op, so a failed submit (which never runs the body) still frees its
    slot and a normal completion cannot double-release.
    """
    with _run_lock:
        if run_id not in _run_slot_holders:
            return
        _run_slot_holders.discard(run_id)
    try:
        _run_slots.release()
    except ValueError:
        pass


def _mark_run_failed_unexpected(run_id: str, exc: BaseException) -> None:
    """Mark a run failed when its worker future dies outside the catch path.

    `_execute_pipeline` catches `Exception`, not `BaseException`, and an
    exception raised inside its own except/finally escapes into the discarded
    Future. Without this callback the run stayed "running" forever and held
    its slot, so two such runs made every later POST /run return 429 until
    the process restarted.
    """
    snapshot = None
    with _run_lock:
        entry = _run_status.get(run_id)
        if entry is not None and entry.get("status") == "running":
            entry.update(
                status="failed",
                phase="failed",
                error=f"{type(exc).__name__}: pipeline worker exited unexpectedly"[:500],
                completed_at=datetime.now().isoformat(),
            )
            snapshot = dict(entry)
    logger.error(
        "Pipeline run %s exited unexpectedly: %r", run_id, exc, exc_info=exc,
    )
    if snapshot is not None:
        _submit_broadcast(_broadcast_status(run_id, snapshot), _main_loop)


def _on_run_future_done(run_id: str, future) -> None:
    """Done callback for a pipeline executor future."""
    try:
        exc = future.exception()
    except BaseException as exc:  # noqa: BLE001 - cancelled futures raise here
        _mark_run_failed_unexpected(run_id, exc)
    else:
        if exc is not None:
            _mark_run_failed_unexpected(run_id, exc)
    with _run_lock:
        _run_futures.pop(run_id, None)
    _release_run_slot(run_id)


# ---------------------------------------------------------------------------
# Small bounded primitives shared by several endpoints
# ---------------------------------------------------------------------------

def _bounded_limit(value: int, default: int = 100, maximum: int = 1000) -> int:
    """Clamp a caller-supplied limit (same rule as `store._clamp_limit`)."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0, min(number, maximum))


#: Tenant ids accepted from the API: bounded, path-safe-looking identifiers.
_TENANT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _resolve_tenant(
    request: Request, tenant_id: str = "", all_tenants: bool = False,
) -> str:
    """Resolve the tenant scope of a core-store/audit read.

    Precedence: explicit ``tenant_id`` query/value, ``X-Tenant-Id`` header,
    then the current :class:`TenantContext` (``"default"`` when unset). The
    API deliberately never falls back to "every tenant": a cross-tenant read
    requires ``all_tenants=true`` and a loopback client. The shared
    ``QUANT_API_TOKEN`` authenticates a consumer, it is **not** a tenant
    boundary -- tenants are an explicit, trusted-header convention.

    Returns ``""`` for the explicit cross-tenant ("all") scope.
    """
    header_tenant = (request.headers.get("X-Tenant-Id") or "").strip()
    candidate = (
        (tenant_id or "").strip()
        or header_tenant
        or TenantContext.get_current().tenant_id
        or "default"
    )
    if not _TENANT_ID_RE.fullmatch(candidate):
        raise HTTPException(
            422,
            "tenant_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}",
        )
    if all_tenants:
        host = request.client.host if request.client else ""
        if host not in _LOOPBACK_HOSTS:
            raise HTTPException(
                403,
                "all_tenants=true is loopback-only; pass a tenant_id or "
                "X-Tenant-Id to read a single tenant",
            )
        return ""
    return candidate



def _error_code_header(code: str) -> dict[str, str]:
    """Stable machine-readable code for a 5xx response.

    The HTTP body keeps a generic human message (no exception text, paths or
    SQL); the code lets clients and dashboards distinguish failures without
    parsing prose.
    """
    return {"X-Error-Code": code}


def _internal_error(
    status_code: int, code: str, message: str,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=message,
        headers=_error_code_header(code),
    )


#: Bounded pool for sync endpoints that rerun the full data pipeline
#: (`_load_data` + `_compute_factors`). AnyIO's default limiter is 40 tokens,
#: so without this a burst could run dozens of full pipelines at once.
_COMPUTE_SLOTS = threading.BoundedSemaphore(2)


def _run_bounded_compute(fn, *args):
    """Run a full-pipeline compute handler under the shared 2-slot bound."""
    if not _COMPUTE_SLOTS.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Compute capacity exhausted; retry shortly",
            headers={"Retry-After": "10"},
        )
    try:
        return fn(*args)
    finally:
        _COMPUTE_SLOTS.release()


async def _run_analytics(fn, *args):
    """Run an analytics job on the dedicated executor, 429 when saturated."""
    if not _analytics_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Analytics capacity exhausted; retry shortly",
            headers={"Retry-After": "5"},
        )
    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(_get_analytics_executor(), fn, *args)
    except Exception:
        _analytics_slots.release()
        raise

    def _release(_future) -> None:
        try:
            _analytics_slots.release()
        except ValueError:
            pass

    future.add_done_callback(_release)
    return await future


def _json_float(value: Any, default: Any = None) -> Any:
    """Coerce to a finite float; NaN/inf are not JSON.

    Starlette encodes JSON with ``allow_nan=False``, so an undefined metric
    (e.g. a Sharpe ratio whose return series had zero variance) must become
    ``None`` -- not NaN and not an invented zero.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _research_window(
    start_date: Any, end_date: Any,
) -> tuple[str | None, str | None]:
    """Validate request date overrides, raising a client-safe 422."""
    try:
        return parse_window(start_date, end_date)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def _load_context_or_503(
    start_date: str | None, end_date: str | None,
) -> ResearchContext:
    """Fetch the cached research context; expose load failures as 503."""
    try:
        return await get_research_context(
            start_date=start_date, end_date=end_date,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Research context build failed")
        raise _internal_error(
            503, "research_context_unavailable",
            "Research data unavailable; see server logs",
        ) from exc


def _derived_get(context: ResearchContext, name: str, params: Any) -> Any | None:
    """Return a cached derived result, marking it as served from cache."""
    hit = derived_cache_get(derived_key(context, name, params))
    if hit is None:
        return None
    if isinstance(hit, dict):
        out = dict(hit)
        out["cached"] = True
        return out
    return hit


def _derived_store(context: ResearchContext, name: str, params: Any, value: Any) -> Any:
    derived_cache_put(derived_key(context, name, params), value)
    return value


def _get_config_with_overrides(overrides: dict[str, Any]) -> Any:
    """Load default config and apply overrides."""
    config = load_config(None)
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

_READINESS_SCHEDULER_STALE_SECONDS = 90.0
_READINESS_DATA_MAX_TRADING_DAYS = 2


def _metric_gauge_value(name: str) -> float | None:
    """Read a label-free gauge from the process metrics collector."""
    try:
        from quant_platform.utils.metrics import get_metrics

        entry = get_metrics().get_snapshot().get(name)
        if not entry:
            return None
        value = entry.get("value")
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None


def _trading_days_between(start_ts: float, end_ts: float) -> int:
    """Exchange sessions between two epoch timestamps (Shanghai calendar)."""
    from zoneinfo import ZoneInfo

    from quant_platform.core.exchange_calendar import is_trading_day

    tz = ZoneInfo("Asia/Shanghai")
    start = datetime.fromtimestamp(start_ts, tz).date()
    end = datetime.fromtimestamp(end_ts, tz).date()
    if start >= end:
        return 0
    count = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if is_trading_day(cursor):
            count += 1
        cursor += timedelta(days=1)
    return count


def _readiness_checks() -> tuple[bool, list[dict], dict]:
    """Evaluate liveness of the trading components (S4 dead-man switch).

    Readiness only judges components that are expected to be running: a
    research-only web deployment is ready without an engine, while a running
    engine with a dead loop, a stalled scheduler during the session, or data
    older than two trading days is not.
    """
    now = time.time()
    engine = _live_engine
    engine_running = bool(engine is not None and getattr(engine, "_running", False))
    scheduler = getattr(engine, "_scheduler", None) if engine is not None else None
    scheduler_running = bool(
        scheduler is not None and getattr(scheduler, "_running", False)
    )

    checks: list[dict] = []

    market_open = False
    if scheduler is not None:
        try:
            market_open = bool(scheduler.is_market_open())
        except Exception:
            market_open = False
    # A live broker deliberately skips trading cycles outside the session;
    # paper mode cycles continuously. The engine loop itself still beats
    # every iteration (dead-man), but data is only expected when a fetch is
    # due.
    cycles_outside = getattr(engine, "cycles_outside_session", None) if engine else None
    try:
        expect_data = (
            scheduler is None
            or market_open
            or bool(cycles_outside() if callable(cycles_outside) else False)
        )
    except Exception:
        expect_data = True

    last_cycle = _metric_gauge_value("last_engine_cycle_timestamp")
    if engine_running:
        interval = float(
            getattr(engine, "_rebalance_interval", 300) or 300
        )
        # 2x the interval, with a 60s floor: a full-market fetch can take
        # longer than 2x a 5s interval, and a slow-but-alive cycle must not
        # flap readiness.
        max_age = max(interval * 2.0, 60.0)
        age = None if last_cycle is None else max(0.0, now - last_cycle)
        ok = age is not None and age <= max_age
        checks.append({
            "name": "engine_cycle",
            "ok": ok,
            "last_timestamp": last_cycle,
            "age_seconds": None if age is None else round(age, 1),
            "max_age_seconds": max_age,
            "reason": "" if ok else "engine loop heartbeat older than 2x interval",
        })
    else:
        checks.append({
            "name": "engine_cycle", "ok": True, "status": "not_running",
            "last_timestamp": last_cycle,
        })

    last_tick = _metric_gauge_value("last_scheduler_tick_timestamp")
    if engine_running and not scheduler_running:
        checks.append({
            "name": "scheduler_tick", "ok": False,
            "last_timestamp": last_tick,
            "reason": "engine is running but its scheduler thread is not",
        })
    elif scheduler_running:
        age = None if last_tick is None else max(0.0, now - last_tick)
        ok = age is not None and age <= _READINESS_SCHEDULER_STALE_SECONDS
        checks.append({
            "name": "scheduler_tick",
            "ok": ok,
            "last_timestamp": last_tick,
            "age_seconds": None if age is None else round(age, 1),
            "max_age_seconds": _READINESS_SCHEDULER_STALE_SECONDS,
            "market_open": market_open,
            "reason": "" if ok else "scheduler tick stale > 90s",
        })
    else:
        checks.append({
            "name": "scheduler_tick", "ok": True,
            "status": "not_running", "last_timestamp": last_tick,
        })

    last_data = _metric_gauge_value("last_data_success_timestamp")
    if engine_running and expect_data:
        trading_days = (
            None if last_data is None
            else _trading_days_between(last_data, now)
        )
        ok = (
            last_data is not None
            and trading_days is not None
            and trading_days <= _READINESS_DATA_MAX_TRADING_DAYS
        )
        checks.append({
            "name": "market_data",
            "ok": ok,
            "last_timestamp": last_data,
            "age_trading_days": trading_days,
            "max_age_trading_days": _READINESS_DATA_MAX_TRADING_DAYS,
            "reason": "" if ok else (
                "no successful market-data fetch recorded"
                if last_data is None
                else "market data older than 2 trading days"
            ),
        })
    else:
        checks.append({
            "name": "market_data", "ok": True,
            "status": (
                "market_closed" if engine_running else "engine_not_running"
            ),
            "last_timestamp": last_data,
        })

    ready = all(check.get("ok", False) for check in checks)
    last_seen = {
        "last_engine_cycle": last_cycle,
        "last_scheduler_tick": last_tick,
        "last_data_success": last_data,
        "engine_running": engine_running,
        "scheduler_running": scheduler_running,
    }
    return ready, checks, last_seen


@router.get("/health")
async def health():
    """Liveness probe.

    Deliberately does not fail when the exchange is closed: the Docker
    healthcheck polls this endpoint. ``/api/readiness`` is the dead-man
    switch that judges engine/scheduler/data heartbeats.
    """
    with _run_lock:
        runs_completed = len(_run_store)
        runs_active = sum(
            1 for s in _run_status.values() if s.get("status") == "running"
        )
    last_seen = {
        "last_engine_cycle": _metric_gauge_value("last_engine_cycle_timestamp"),
        "last_scheduler_tick": _metric_gauge_value("last_scheduler_tick_timestamp"),
        "last_data_success": _metric_gauge_value("last_data_success_timestamp"),
    }
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "runs_completed": runs_completed,
        "runs_active": runs_active,
        **last_seen,
    }


@router.get("/readiness")
async def readiness():
    """Dead-man switch: 503 when a running component's heartbeat is stale."""
    from fastapi.responses import JSONResponse

    ready, checks, last_seen = _readiness_checks()
    payload = {
        "status": "ready" if ready else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        **last_seen,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


# ---------------------------------------------------------------------------
# WebSocket — real-time pipeline status
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # BaseHTTPMiddleware never sees WebSocket scopes, so the HTTP token guard
    # does not cover this stream (which carries order.filled and risk events).
    #
    # The token travels only in the `bearer.<token>` subprotocol: browsers
    # cannot set handshake headers, and a `?token=` query string was written
    # verbatim to the uvicorn access log (and every proxy log on the path).
    # The legacy query fallback was removed -- a breaking change documented in
    # the security report; tokens must be RFC 7230 token characters.
    offered = list((getattr(ws, "scope", None) or {}).get("subprotocols") or [])
    bearer = next((p for p in offered if p.startswith("bearer.")), "")
    effective_token = bearer[len("bearer."):] if bearer else ""

    expected = os.environ.get("QUANT_API_TOKEN", "")
    if expected:
        if not effective_token or not secrets.compare_digest(
            effective_token.encode("utf-8"), expected.encode("utf-8"),
        ):
            await ws.close(code=1008)
            return
    else:
        # With no token configured the HTTP guard limits trading/OMS/risk
        # routes to loopback; the event stream carries the same information
        # and must not be open to remote clients either.
        host = ws.client.host if ws.client else ""
        if host not in _LOOPBACK_HOSTS:
            await ws.close(code=1008)
            return
    global _ws_pending
    # Reserve the slot before the first await: checking the set and adding
    # after `accept()` let concurrent handshakes all pass the 100-client cap
    # (99 -> 119 observed).
    if len(_ws_clients) + _ws_pending >= _WS_MAX_CLIENTS:
        await ws.close(code=1013)
        return
    _ws_pending += 1
    try:
        # Capture the loop here: this is the one place guaranteed to be
        # running on it before events start flowing. See _on_bus_event.
        _remember_loop()
        if bearer:
            await ws.accept(subprotocol=bearer)
        else:
            await ws.accept()
        client = _WSClient(ws)
        client.task = asyncio.create_task(_ws_sender_loop(client))
        _ws_clients.add(client)
    finally:
        _ws_pending -= 1
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        client.closed = True
        _ws_clients.discard(client)
        task = client.task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@router.get("/config", response_model=ConfigInfo)
async def get_config():
    # YAML read: never on the event loop.
    config = await run_in_threadpool(load_config, None)
    return ConfigInfo(
        default_config={
            "n_stocks": config.universe.n_stocks,
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
            "optimizer": config.portfolio.optimizer,
            "alpha_method": config.alpha.method,
            "rebalance_frequency": config.backtest.rebalance_frequency,
            "covariance_method": config.portfolio.covariance.method,
            "initial_capital": config.backtest.initial_capital,
        },
        available_options={
            "optimizers": ["equal_weight", "mean_variance", "risk_parity"],
            "alpha_methods": ["equal_weight", "ic_weighted", "icir_weighted"],
            "frequencies": ["daily", "weekly", "monthly"],
            "covariance_methods": ["sample", "ledoit_wolf", "ewma"],
        },
    )


# ---------------------------------------------------------------------------
# Run pipeline (async, non-blocking)
# ---------------------------------------------------------------------------

@router.post("/run", response_model=RunStatusWithPhase)
async def run_pipeline(req: RunRequest):
    """Start a full pipeline run in a background thread.

    Returns immediately with run_id. Poll /run/{run_id}/status for progress.
    The response carries `phase="queued"` until a worker actually picks the
    run up, then `"executing"`; `status` stays "running" throughout so the
    existing UI contract is unchanged. A third concurrent run is rejected
    with 429 (the pipeline pool has two workers).
    """
    if not _run_slots.acquire(blocking=False):
        raise HTTPException(
            429,
            "Two pipeline runs are already queued/running; try again later",
            headers={"Retry-After": "5"},
        )

    run_id = uuid.uuid4().hex[:12]
    try:
        with _run_lock:
            _run_status[run_id] = {
                "status": "running",
                "phase": "queued",
                "progress": 0,
                "stage": "data",
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "error": None,
            }
            _run_slot_holders.add(run_id)

        # Offload to the pipeline-only pool — does NOT block the event loop.
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            _get_pipeline_executor(), _execute_pipeline, run_id, req,
        )
        with _run_lock:
            _run_futures[run_id] = future
        future.add_done_callback(
            lambda done, rid=run_id: _on_run_future_done(rid, done)
        )
    except BaseException:
        with _run_lock:
            _run_status.pop(run_id, None)
            _run_futures.pop(run_id, None)
        _release_run_slot(run_id)
        raise

    snapshot = _run_status_snapshot(run_id) or {}
    return RunStatusWithPhase(
        run_id=run_id,
        status=snapshot.get("status", "running"),
        phase=snapshot.get("phase", "queued"),
        progress=snapshot.get("progress", 0),
        stage=snapshot.get("stage", "data"),
        started_at=snapshot.get("started_at") or datetime.now().isoformat(),
        completed_at=snapshot.get("completed_at"),
        error=snapshot.get("error"),
    )


def _execute_pipeline(run_id: str, req: RunRequest):
    """Run the full quant pipeline (called from thread pool)."""
    _set_run_fields(run_id, phase="executing")
    try:
        config = _get_config_with_overrides({
            "universe.n_stocks": req.n_stocks,
            "data.start_date": req.start_date,
            "data.end_date": req.end_date,
            "portfolio.optimizer": req.optimizer,
            "alpha.method": req.alpha_method,
            "backtest.rebalance_frequency": req.rebalance_frequency,
            "portfolio.covariance.method": req.covariance_method,
        })

        # Stage 1: Data
        _update_status(run_id, 10, "data")
        from quant_platform.utils.metrics import pipeline_stage

        with pipeline_stage("data"):
            prices, returns, benchmark, metadata, financials, turnover = _load_data(
                config, use_tushare=req.use_tushare, use_baostock=req.use_baostock)

        # Compute the quality report while the price panel exists, and keep
        # the small summary with the run. The run record does not retain the
        # panel itself, so `/api/data/quality?run_id=...` used to return 422
        # for every completed run.
        try:
            from quant_platform.data.quality import DataQualityMonitor
            dq = DataQualityMonitor()
            dq.check_prices(prices)
            dq.check_returns(returns)
            data_quality_report = dq.get_report()
        except Exception as exc:
            logger.warning("Data quality check failed for run %s: %s", run_id, exc)
            data_quality_report = {"error": str(exc)}

        # Stage 2: Factors
        _update_status(run_id, 30, "factors")
        with pipeline_stage("factors"):
            processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
                prices, returns, financials, metadata, turnover, config=config)

        # Stage 3: Alpha
        _update_status(run_id, 50, "alpha")
        with pipeline_stage("alpha"):
            signal = _generate_signal(config, processed_factors, returns)

        # Stage 4-5: Backtest
        _update_status(run_id, 70, "backtest")
        with pipeline_stage("backtest"):
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked,
                gates=_execution_gates(metadata),
            )

        # Stage 6: Build response
        _update_status(run_id, 90, "report")

        summary = bt_results["summary"]
        strategy_returns = bt_results["daily_returns"]
        benchmark_returns = bt_results.get("benchmark_returns")
        weights_history = bt_results.get("weights_history", {})

        chart_data = _build_chart_data(
            strategy_returns, benchmark_returns,
            weights_history=weights_history,
            processed_factors=processed_factors,
            signal=signal,
            asset_returns=returns,
        )

        # Risk
        from quant_platform.risk.stress import run_all_stress_tests
        from quant_platform.risk.var import var_summary
        risk = var_summary(
            strategy_returns,
            confidence=config.risk.var.confidence,
            horizon=config.risk.var.horizon,
        )
        stress_df = run_all_stress_tests(strategy_returns)
        stress_tests = [
            StressTestItem(
                scenario=row.name,
                cumulative_return=row["estimated_cumulative_return"],
                max_drawdown=row["estimated_max_drawdown"],
            )
            for _, row in stress_df.iterrows()
        ]

        # Exposure
        weights_history = bt_results.get("weights_history", {})
        if weights_history:
            latest_date = max(weights_history.keys())
            latest_weights = weights_history[latest_date]
            from quant_platform.risk.exposure import exposure_report
            # `metadata` is a DataFrame, and `if metadata` on a DataFrame
            # raises "The truth value of a DataFrame is ambiguous". That made
            # `POST /api/run` fail at this line on every request, before it
            # could store a result. `_compute_factors` (main.py) guards the
            # same expression with `is not None`.
            sector_map_meta = metadata.get("sector", {}) if metadata is not None else {}
            exp = exposure_report(latest_weights, sector_map_meta)
            sectors_dict = exp.get("sector_exposure", {})
            if hasattr(sectors_dict, "to_dict"):
                sectors_dict = sectors_dict.to_dict()

            # Stock-level holdings: top 20 by weight
            top_holdings = []
            if hasattr(latest_weights, 'sort_values'):
                sorted_w = latest_weights.sort_values(ascending=False)
                for ticker, w in sorted_w.head(20).items():
                    if w > 0.001:
                        # `sector_map_meta` is a Series (the metadata frame's
                        # `sector` column), so it cannot be used as a truth
                        # value either.
                        sec = (
                            sector_map_meta.get(str(ticker), "")
                            if sector_map_meta is not None
                            else ""
                        )
                        # Latest daily move as a P&L proxy (the run record
                        # does not persist per-position entry prices).
                        pnl = None
                        if str(ticker) in prices.columns:
                            p = prices[str(ticker)]
                            if len(p) > 1:
                                last = float(p.iloc[-1])
                                prev = float(p.iloc[-2])
                                # A suspension leaves NaN in the panel; NaN is
                                # not JSON-compliant and made the entire
                                # /result endpoint return 500.
                                if (
                                    np.isfinite(last) and np.isfinite(prev)
                                    and prev > 0
                                ):
                                    pnl = float((last / prev - 1) * 100)
                        top_holdings.append(HoldingItem(
                            ticker=str(ticker),
                            weight=round(float(w), 5),
                            sector=str(sec) if sec else None,
                            pnl_pct=round(pnl, 2) if pnl is not None else None,
                        ))

            exposure_info = ExposureInfo(
                n_assets=exp.get("n_assets", 0),
                effective_n=exp.get("effective_n", 0),
                top5_concentration=exp.get("top5_concentration", 0),
                top10_concentration=exp.get("top10_concentration", 0),
                sectors={str(k): float(v) for k, v in dict(sectors_dict).items()},
                top_holdings=top_holdings,
            )
        else:
            exposure_info = ExposureInfo(
                n_assets=0, effective_n=0, top5_concentration=0, top10_concentration=0,
                sectors={},
                top_holdings=[],
            )

        # Factor IC
        factor_items = []
        for name, ic_sum in (ic_results or {}).items():
            try:
                stats = {
                    "mean_ic": float(ic_sum.get("mean_ic")),
                    "std_ic": float(ic_sum.get("std_ic")),
                    "icir": float(ic_sum.get("icir")),
                    "ic_positive_ratio": float(ic_sum.get("ic_positive_ratio")),
                }
            except (TypeError, ValueError):
                stats = {}
            if not stats or not all(np.isfinite(v) for v in stats.values()):
                # Factors with too few evaluable cross-sections carry NaN
                # statistics. Including them made the whole run record
                # unserialisable (GET /result and /factors returned 500);
                # omitting them is honest -- there is no measured IC.
                logger.warning(
                    "Skipping factor %s from the run record: IC statistics "
                    "are not finite (n_obs=%s)",
                    name, ic_sum.get("n_obs"),
                )
                continue
            factor_items.append(FactorICItem(name=name, **stats))
        factor_items.sort(key=lambda x: abs(x.icir), reverse=True)

        # Store plain dicts, not the Pydantic models.
        #
        # Every consumer of `_run_store` treats it as a dict of dicts:
        # `result.get("chart_data", {}).get("equity", [])` in the Monte Carlo,
        # risk-decomposition and regime endpoints, `f.get("name")` over the
        # factor list in the HTML report. Storing models made each of those an
        # `AttributeError: 'ChartData' object has no attribute 'get'`, so the
        # endpoints returned 500 whenever a run had actually succeeded.
        record = {
            "performance": PerformanceMetrics(
                total_return=summary.get("total_return", 0),
                annual_return=summary.get("annual_return", 0),
                annual_volatility=summary.get("annual_volatility", 0),
                sharpe_ratio=summary.get("sharpe_ratio", 0),
                sortino_ratio=summary.get("sortino_ratio", 0),
                calmar_ratio=summary.get("calmar_ratio", 0),
                max_drawdown=summary.get("max_drawdown", 0),
                max_drawdown_peak=summary.get("max_drawdown_peak"),
                max_drawdown_trough=summary.get("max_drawdown_trough"),
                win_rate=summary.get("win_rate", 0),
                profit_loss_ratio=summary.get("profit_loss_ratio"),
                total_days=summary.get("total_days", 0),
                information_ratio=summary.get("information_ratio"),
                tracking_error=summary.get("tracking_error"),
                excess_return=summary.get("excess_return"),
                n_rebalances=summary.get("n_rebalances", 0),
                optimizer=summary.get("optimizer", ""),
                initial_capital=summary.get("initial_capital", 0),
                n_optimizer_fallbacks=summary.get("n_optimizer_fallbacks", 0),
                optimizer_fallback_rate=summary.get("optimizer_fallback_rate", 0.0),
                effective_optimizer=summary.get("effective_optimizer"),
                n_skipped_rebalances=summary.get("n_skipped_rebalances", 0),
            ).model_dump(),
            "risk": RiskMetrics(
                historical_var=risk.get("historical_var", 0),
                parametric_var=risk.get("parametric_var", 0),
                historical_cvar=risk.get("historical_cvar", 0),
            ).model_dump(),
            "data_quality": data_quality_report,
            "stress_tests": [s.model_dump() for s in stress_tests],
            "factors": [f.model_dump() for f in factor_items],
            "exposure": exposure_info.model_dump(),
            "chart_data": chart_data.model_dump(),
        }

        # Starlette's JSON encoder rejects NaN/Inf; a completed run holding
        # one made GET /result and /factors return 500 forever. Fail the run
        # at creation instead of storing an unservable record.
        try:
            json.dumps(record, allow_nan=False)
        except ValueError as exc:
            raise ValueError(
                "Run produced non-JSON-compliant numbers (NaN/Inf); "
                "marking the run failed instead of storing it."
            ) from exc
        with _run_lock:
            _run_store[run_id] = record

        _update_status(run_id, 100, "done")

    except Exception as exc:
        # Keep the full traceback in the server log; return only the exception
        # type/message (with the run id for correlation). The status endpoint
        # previously exposed absolute paths and source lines to any caller.
        full_traceback = traceback.format_exc()
        snapshot = _set_run_fields(
            run_id,
            status="failed",
            phase="failed",
            error=f"{type(exc).__name__}: {exc}"[:500],
            completed_at=datetime.now().isoformat(),
        )
        logger.error("Pipeline %s failed: %s", run_id, full_traceback)
        # Failures were never broadcast, so the frontend's WS failure branch
        # was dead and the run only surfaced through the next poll.
        if snapshot is not None:
            _submit_broadcast(_broadcast_status(run_id, snapshot), _main_loop)

    finally:
        # The slot is released even when the failure path itself raises; the
        # done callback would otherwise be the only release path.
        try:
            _cleanup_old_runs()
        finally:
            _release_run_slot(run_id)


def _update_status(run_id: str, progress: int, stage: str):
    fields: dict[str, Any] = {"progress": progress, "stage": stage}
    if stage == "done":
        fields.update(
            status="completed", phase="completed",
            completed_at=datetime.now().isoformat(),
        )
    snapshot = _set_run_fields(run_id, **fields)
    if snapshot is None:
        logger.warning("Status update for unknown run %s ignored", run_id)
        return
    # Broadcast to WebSocket clients (non-blocking from the worker thread).
    # `asyncio.get_event_loop()` raises on Python 3.12 in a thread with no
    # loop, so progress events were silently dropped; use the loop captured
    # by the WebSocket endpoint instead.
    loop = _main_loop
    if loop is not None and loop.is_running():
        _submit_broadcast(_broadcast_status(run_id, snapshot), loop)


@router.get("/run/{run_id}/status", response_model=RunStatusWithPhase)
async def get_run_status(run_id: str):
    """Poll pipeline run status (including queued-vs-executing phase)."""
    s = _run_status_snapshot(run_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusWithPhase(
        run_id=run_id,
        status=s.get("status", "running"),
        phase=s.get("phase", "executing"),
        progress=s.get("progress", 0),
        stage=s.get("stage", "data"),
        started_at=s.get("started_at") or datetime.now().isoformat(),
        completed_at=s.get("completed_at"),
        error=s.get("error"),
    )


@router.get("/run/{run_id}/result", response_model=RunResult)
async def get_run_result(run_id: str):
    """Get completed run results."""
    with _run_lock:
        data = _run_store.get(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return RunResult(
        run_id=run_id,
        performance=data.get("performance"),
        risk=data.get("risk"),
        stress_tests=data.get("stress_tests"),
        factors=data.get("factors"),
        exposure=data.get("exposure"),
        chart_data=data.get("chart_data"),
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _compute_drawdown_periods(equity: pd.Series, threshold: float = -0.01) -> list[dict]:
    """Extract significant drawdown periods from equity curve."""
    running_max = equity.expanding(min_periods=1).max()
    drawdown = equity / running_max - 1
    periods = []
    in_dd = False
    dd_start = None
    dd_trough_val = 0
    dd_trough_date = None

    for date, dd_val in drawdown.items():
        if dd_val < -threshold and not in_dd:
            in_dd = True
            dd_start = date
            dd_trough_val = dd_val
            dd_trough_date = date
        elif in_dd:
            if dd_val < dd_trough_val:
                dd_trough_val = dd_val
                dd_trough_date = date
            if dd_val >= -threshold * 0.1:  # Recovered
                in_dd = False
                duration = (date - dd_start).days if hasattr(date, 'days') else 0
                try:
                    duration = (pd.Timestamp(date) - pd.Timestamp(dd_start)).days
                except Exception:
                    duration = 0
                periods.append({
                    "start": str(dd_start.date()) if hasattr(dd_start, 'date') else str(dd_start),
                    "end": str(date.date()) if hasattr(date, 'date') else str(date),
                    "trough": str(dd_trough_date.date()) if hasattr(dd_trough_date, 'date') else str(dd_trough_date),
                    "depth": round(float(dd_trough_val), 4),
                    "duration_days": max(duration, 0),
                    "recovered": True,
                    "recovery_date": str(date.date()) if hasattr(date, 'date') else str(date),
                })

    # If still in drawdown at end
    if in_dd:
        last_date = drawdown.index[-1]
        try:
            duration = (pd.Timestamp(last_date) - pd.Timestamp(dd_start)).days
        except Exception:
            duration = 0
        periods.append({
            "start": str(dd_start.date()) if hasattr(dd_start, 'date') else str(dd_start),
            "end": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
            "trough": str(dd_trough_date.date()) if hasattr(dd_trough_date, 'date') else str(dd_trough_date),
            "depth": round(float(dd_trough_val), 4),
            "duration_days": max(duration, 0),
            "recovered": False,
            "recovery_date": None,
        })

    periods.sort(key=lambda x: x["depth"])
    return periods[:10]  # Top 10 worst


def _compute_turnover(weights_history: dict) -> list[dict]:
    """Compute per-period turnover from weights history."""
    if len(weights_history) < 2:
        return []
    sorted_dates = sorted(weights_history.keys())
    items = []
    for i in range(1, len(sorted_dates)):
        prev_w = weights_history[sorted_dates[i - 1]]
        curr_w = weights_history[sorted_dates[i]]
        if hasattr(prev_w, 'values') and hasattr(curr_w, 'values'):
            all_tickers = set(prev_w.index) | set(curr_w.index)
            total_change = 0
            n_trades = 0
            for t in all_tickers:
                pw = float(prev_w.get(t, 0))
                cw = float(curr_w.get(t, 0))
                change = abs(cw - pw)
                total_change += change
                if change > 0.001:
                    n_trades += 1
            items.append({
                # Date-only, matching the demo payload and the chart axis;
                # `str(Timestamp)` included a useless "00:00:00" suffix.
                "date": str(pd.Timestamp(sorted_dates[i]).date()),
                "turnover": round(total_change / 2, 4),  # One-way turnover
                "n_trades": n_trades,
            })
    return items


def _compute_factor_scatter(
    processed_factors: dict,
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    top_n: int = 6,
) -> list[dict]:
    """Compute factor vs forward return scatter data for top factors.

    `returns` must be the asset-level forward-return panel (t -> t+1), the
    same convention the factor evaluator uses. It previously received the
    portfolio-return Series; `returns.shift(-1)` on that Series is also a
    second shift, i.e. a one-day look-ahead when the panel is already
    forward-aligned.
    """
    from scipy import stats

    # Get IC ranking from factors
    factor_ics = {}
    for name, factor_df in processed_factors.items():
        if hasattr(factor_df, 'values'):
            # Compute rank IC with next-period returns
            fwd_ret = returns
            common_dates = factor_df.index.intersection(fwd_ret.dropna().index)
            if len(common_dates) > 50:
                ics = []
                for d in common_dates[:252]:  # Last year of ICs
                    try:
                        f_vals = factor_df.loc[d].dropna()
                        r_vals = fwd_ret.loc[d].dropna()
                        common = f_vals.index.intersection(r_vals.index)
                        if len(common) > 10:
                            ic, _ = stats.spearmanr(f_vals[common], r_vals[common])
                            if not np.isnan(ic):
                                ics.append(ic)
                    except Exception:
                        continue
                if ics:
                    mean_ic = np.mean(ics)
                    std_ic = np.std(ics)
                    icir = mean_ic / std_ic if std_ic > 0 else 0
                    factor_ics[name] = {"mean_ic": mean_ic, "icir": icir}

    # Select top factors by |ICIR|
    sorted_factors = sorted(factor_ics.items(), key=lambda x: abs(x[1]["icir"]), reverse=True)
    top_factors = sorted_factors[:top_n]

    result = []
    for name, ic_info in top_factors:
        factor_df = processed_factors[name]
        fwd_ret = returns
        common_dates = factor_df.index.intersection(fwd_ret.dropna().index)

        # Sample points for scatter (max 500 points)
        points = []
        step = max(1, len(common_dates) // 80)
        for d in common_dates[::step][:500]:
            try:
                f_vals = factor_df.loc[d].dropna()
                r_vals = fwd_ret.loc[d].dropna()
                common = f_vals.index.intersection(r_vals.index)
                if len(common) > 0:
                    for ticker in common[:5]:  # Sample 5 stocks per date
                        points.append({
                            "x": round(float(f_vals[ticker]), 4),
                            "y": round(float(r_vals[ticker]) * 100, 3),
                        })
            except Exception:
                continue

        # Compute t-stat
        t_stat = ic_info["icir"] * np.sqrt(len(common_dates)) if common_dates is not None else 0

        result.append(FactorScatterItem(
            factor_name=name,
            points=points[:300],  # Limit for frontend performance
            ic=round(ic_info["mean_ic"], 4),
            icir=round(ic_info["icir"], 3),
            t_stat=round(float(t_stat), 2),
        ))

    return result


def _compute_attribution(
    processed_factors: dict,
    weights_history: dict,
    returns: pd.DataFrame,
) -> list[dict]:
    """Compute P&L attribution by factor contribution.

    `returns` is the asset-level forward-return panel (t -> t+1). The
    portfolio held at `prev_date` earns that row, so attribution uses
    `prev_date`, not the execution date and not a reindexed Series.
    """
    from scipy import stats

    if len(weights_history) < 2:
        return []

    sorted_dates = sorted(weights_history.keys())
    factor_contribs = {}

    for name, factor_df in processed_factors.items():
        if not hasattr(factor_df, 'values'):
            continue
        contribs = []
        for i in range(1, min(len(sorted_dates), 60)):  # Last 60 periods
            try:
                date = sorted_dates[i]
                prev_date = sorted_dates[i - 1]
                w = weights_history[date]
                # Factor-weighted return
                if prev_date in returns.index and hasattr(w, 'index'):
                    common = w.index.intersection(factor_df.columns if hasattr(factor_df, 'columns') else factor_df.index)
                    if len(common) > 0:
                        # Factor exposure weighted by portfolio weights
                        if hasattr(factor_df, 'loc') and prev_date in factor_df.index:
                            f_vals = factor_df.loc[prev_date].reindex(common).fillna(0)
                            w_vals = w.reindex(common).fillna(0)
                            r_vals = returns.loc[prev_date].reindex(common).fillna(0)
                            # IC contribution
                            ic, _ = stats.spearmanr(f_vals.values, r_vals.values)
                            if not np.isnan(ic):
                                contribs.append(ic * float(w_vals.sum()))
            except Exception:
                continue

        if contribs:
            avg_contrib = np.mean(contribs)
            factor_contribs[name] = avg_contrib

    # Get IC weights for context
    result = []
    total = sum(abs(v) for v in factor_contribs.values()) or 1
    for name, contrib in sorted(factor_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        # Compute average IC
        try:
            fwd_ret = returns
            # `factor_df` here was left over from the loop above, so it held
            # whichever factor that loop happened to end on -- the IC reported
            # for each name was computed from some other factor's dates. Line
            # 729 already used processed_factors[name]; this line did not. If
            # the earlier loop never ran, the name was simply undefined.
            factor_df = processed_factors[name]
            common_dates = factor_df.index.intersection(fwd_ret.dropna().index)
            ics = []
            for d in common_dates[:60]:
                try:
                    f_vals = processed_factors[name].loc[d].dropna()
                    r_vals = fwd_ret.loc[d].dropna()
                    c = f_vals.index.intersection(r_vals.index)
                    if len(c) > 10:
                        ic, _ = stats.spearmanr(f_vals[c], r_vals[c])
                        if not np.isnan(ic):
                            ics.append(ic)
                except Exception:
                    continue
            avg_ic = np.mean(ics) if ics else 0
        except Exception:
            avg_ic = 0

        result.append(AttributionItem(
            factor=name,
            contribution_bps=round(float(contrib) * 10000, 1),
            weight=round(float(abs(contrib) / total), 3),
            avg_ic=round(float(avg_ic), 4),
        ))

    return result


#: Hard bound on any single chart series in a run result. A 20-year daily run
#: is ~5,000 points; beyond this the frontend cannot show more pixels anyway
#: and the JSON payload grows without bound. The most recent point is always
#: retained so the chart still ends at the present.
_MAX_CHART_POINTS = 2000


def _downsample_series(values: list, max_points: int = _MAX_CHART_POINTS) -> list:
    if len(values) <= max_points:
        return values
    step = int(np.ceil(len(values) / max_points))
    sampled = values[::step]
    if sampled[-1] is not values[-1]:
        sampled.append(values[-1])
    return sampled


def _downsample_dates(dates: list[str], values: list,
                      max_points: int = _MAX_CHART_POINTS) -> tuple[list[str], list]:
    if len(values) <= max_points or len(dates) != len(values):
        return dates, values
    step = int(np.ceil(len(values) / max_points))
    sampled_dates = dates[::step]
    sampled_values = values[::step]
    if sampled_values[-1] is not values[-1]:
        sampled_dates.append(dates[-1])
        sampled_values.append(values[-1])
    return sampled_dates, sampled_values


def _build_chart_data(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    weights_history: dict | None = None,
    processed_factors: dict | None = None,
    signal: pd.DataFrame | None = None,
    asset_returns: pd.DataFrame | None = None,
) -> ChartData:
    """Build chart-ready data from returns series.

    `asset_returns` is the date x asset forward-return panel. Scatter and
    attribution need cross-sectional returns; passing the portfolio-return
    Series made both silently produce empty lists on every real run.
    """
    sr = strategy_returns.dropna()
    equity = (1 + sr).cumprod()
    dates = [str(d.date()) for d in equity.index]

    bench_equity = None
    if benchmark_returns is not None:
        br = benchmark_returns.reindex(sr.index).dropna()
        bench_equity = (1 + br).cumprod().tolist()

    # Drawdown from the initial wealth of 1.0; see reporting.performance.
    wealth = np.concatenate([[1.0], equity.to_numpy()])
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth / running_max - 1)[1:].tolist()

    rolling_sharpe = None
    rolling_sharpe_dates = None
    if len(sr) >= 252:
        from quant_platform.backtest.metrics import TRADING_DAYS_PER_YEAR
        rolling_ret = sr.rolling(252).mean() * TRADING_DAYS_PER_YEAR
        rolling_vol = sr.rolling(252).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rs = (rolling_ret / rolling_vol.replace(0, np.nan)).dropna()
        rolling_sharpe = rs.tolist()
        rolling_sharpe_dates = [str(d.date()) for d in rs.index]

    monthly = None
    if len(sr) >= 21:
        # `"M"` was removed in pandas 3; MonthEnd works on 2.x and 3.x.
        monthly_ret = sr.resample(pd.offsets.MonthEnd()).apply(lambda x: (1 + x).prod() - 1)
        monthly_ret.index = monthly_ret.index.to_period("M")
        mdf = monthly_ret.groupby([monthly_ret.index.year, monthly_ret.index.month]).first().unstack()
        # Fill in months with no observation so the array index matches the
        # Jan-Dec axis the frontend and the HTML report draw. Without this a
        # March-May run had its March value plotted under January.
        mdf = mdf.reindex(columns=range(1, 13))
        if not mdf.empty:
            monthly = {
                "years": [str(y) for y in mdf.index.tolist()],
                "months": list(range(1, 13)),
                # NaN is not JSON-compliant; Starlette rejects the response
                # with a 500. A missing cell is null and the heatmap skips it.
                "data": [
                    [None if pd.isna(value) else round(float(value), 6)
                     for value in row]
                    for row in mdf.values.tolist()
                ],
            }

    # Return distribution histogram
    ret_pct = (sr * 100).values
    hist_counts, hist_edges = np.histogram(ret_pct, bins=50)
    distribution = {
        "edges": [round(float(e), 3) for e in hist_edges],
        "counts": [int(c) for c in hist_counts],
        "mean": round(float(np.mean(ret_pct)), 4),
        "std": round(float(np.std(ret_pct)), 4),
        "skew": round(float(pd.Series(ret_pct).skew()), 4),
        "kurtosis": round(float(pd.Series(ret_pct).kurtosis()), 4),
        "min": round(float(np.min(ret_pct)), 3),
        "max": round(float(np.max(ret_pct)), 3),
    }

    # Cumulative excess return
    excess_cumulative = None
    if benchmark_returns is not None:
        br = benchmark_returns.reindex(sr.index).dropna()
        common_idx = sr.index.intersection(br.index)
        if len(common_idx) > 0:
            excess = sr.loc[common_idx] - br.loc[common_idx]
            excess_cumulative = (1 + excess).cumprod().tolist()

    # Drawdown periods
    dd_periods = None
    try:
        dd_periods_raw = _compute_drawdown_periods(equity)
        dd_periods = [DrawdownPeriod(**p) for p in dd_periods_raw]
    except Exception:
        pass

    # Turnover
    turnover = None
    if weights_history and len(weights_history) > 1:
        try:
            turnover_raw = _compute_turnover(weights_history)
            turnover = [TurnoverItem(**t) for t in turnover_raw]
        except Exception:
            pass

    # Factor scatter (top 6 factors by |ICIR|)
    factor_scatter = None
    if processed_factors and signal is not None and asset_returns is not None:
        try:
            factor_scatter = _compute_factor_scatter(
                processed_factors, signal, asset_returns, top_n=6)
        except Exception:
            pass

    # P&L attribution
    attribution = None
    if (
        processed_factors and weights_history and len(weights_history) > 1
        and asset_returns is not None
    ):
        try:
            attribution = _compute_attribution(
                processed_factors, weights_history, asset_returns)
        except Exception:
            pass

    # Bound every long series before it enters the run record. Drawdown
    # periods, turnover, scatter and attribution are already capped above;
    # the equity-aligned series were not, so a long run serialized one point
    # per trading day forever.
    equity_values = equity.tolist()
    dates, equity_values = _downsample_dates(dates, equity_values)
    drawdown = _downsample_series(drawdown)
    if bench_equity is not None:
        bench_equity = _downsample_series(bench_equity)
    if excess_cumulative is not None:
        excess_cumulative = _downsample_series(excess_cumulative)
    if rolling_sharpe is not None and rolling_sharpe_dates is not None:
        rolling_sharpe_dates, rolling_sharpe = _downsample_dates(
            rolling_sharpe_dates, rolling_sharpe,
        )

    return ChartData(
        dates=dates,
        equity=equity_values,
        benchmark=bench_equity,
        drawdown=drawdown,
        rolling_sharpe=rolling_sharpe,
        rolling_sharpe_dates=rolling_sharpe_dates,
        monthly_returns=monthly,
        return_distribution=distribution,
        excess_cumulative=excess_cumulative,
        turnover=turnover,
        drawdown_periods=dd_periods,
        factor_scatter=factor_scatter,
        attribution=attribution,
    )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

@router.get("/demo")
async def get_demo():
    """Return demo data for immediate UI preview without running a pipeline."""
    import random
    rng = random.Random(42)
    n = 1000
    dates = []
    equity = []
    price = 1.0
    import datetime as dt
    d = dt.date(2021, 1, 4)
    for i in range(n):
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        dates.append(str(d))
        ret = rng.gauss(0.0004, 0.015)  # ~10% annual, 24% vol
        price *= (1 + ret)
        equity.append(round(price, 4))
        d += dt.timedelta(days=1)

    # Drawdown (fraction, matching /api/run; the UI multiplies by 100)
    dd = []
    peak = 0
    for v in equity:
        peak = max(peak, v)
        dd.append(round(v / peak - 1, 6))

    # Rolling sharpe (last 252 entries)
    rs_dates = dates[252:]
    rs = []
    rng2 = random.Random(7)
    for _ in rs_dates:
        rs.append(round(rng2.gauss(0.5, 0.3), 2))

    demo_factors = [
        {"name": "momentum_3m", "mean_ic": 0.0312, "std_ic": 0.124, "icir": 0.25, "ic_positive_ratio": 0.58},
        {"name": "roe", "mean_ic": 0.0287, "std_ic": 0.098, "icir": 0.29, "ic_positive_ratio": 0.61},
        {"name": "volatility_60d", "mean_ic": -0.0251, "std_ic": 0.112, "icir": -0.22, "ic_positive_ratio": 0.42},
        {"name": "momentum_1m", "mean_ic": 0.0218, "std_ic": 0.131, "icir": 0.17, "ic_positive_ratio": 0.55},
        {"name": "pb_ratio", "mean_ic": -0.0193, "std_ic": 0.105, "icir": -0.18, "ic_positive_ratio": 0.44},
        {"name": "log_market_cap", "mean_ic": -0.0178, "std_ic": 0.089, "icir": -0.20, "ic_positive_ratio": 0.43},
        {"name": "momentum_6m", "mean_ic": 0.0164, "std_ic": 0.118, "icir": 0.14, "ic_positive_ratio": 0.53},
        {"name": "asset_growth", "mean_ic": -0.0142, "std_ic": 0.096, "icir": -0.15, "ic_positive_ratio": 0.46},
        {"name": "rsi_14d", "mean_ic": 0.0125, "std_ic": 0.087, "icir": 0.14, "ic_positive_ratio": 0.54},
        {"name": "turnover_20d", "mean_ic": -0.0110, "std_ic": 0.092, "icir": -0.12, "ic_positive_ratio": 0.47},
        {"name": "pe_ratio", "mean_ic": -0.0098, "std_ic": 0.078, "icir": -0.13, "ic_positive_ratio": 0.45},
        {"name": "momentum_12m", "mean_ic": 0.0089, "std_ic": 0.108, "icir": 0.08, "ic_positive_ratio": 0.51},
        {"name": "amplitude_20d", "mean_ic": -0.0072, "std_ic": 0.084, "icir": -0.09, "ic_positive_ratio": 0.47},
        {"name": "macd", "mean_ic": 0.0056, "std_ic": 0.075, "icir": 0.07, "ic_positive_ratio": 0.52},
        {"name": "volatility_20d", "mean_ic": -0.0034, "std_ic": 0.081, "icir": -0.04, "ic_positive_ratio": 0.49},
    ]
    demo_factors.sort(key=lambda x: abs(x["icir"]), reverse=True)

    # Return distribution histogram
    ret_series = []
    for i in range(1, len(equity)):
        ret_series.append((equity[i] / equity[i-1] - 1) * 100)
    ret_arr = ret_series
    hist_counts, hist_edges = np.histogram(ret_arr, bins=50)
    distribution = {
        "edges": [round(float(e), 3) for e in hist_edges],
        "counts": [int(c) for c in hist_counts],
        "mean": round(float(np.mean(ret_arr)), 4),
        "std": round(float(np.std(ret_arr)), 4),
        "skew": round(float(pd.Series(ret_arr).skew()), 4),
        "kurtosis": round(float(pd.Series(ret_arr).kurtosis()), 4),
        "min": round(float(np.min(ret_arr)), 3),
        "max": round(float(np.max(ret_arr)), 3),
    }

    # Cumulative excess return
    bench_equity_list = [v * (1 + rng.gauss(0.0002, 0.012)) for v in equity]
    excess_cum = []
    for i in range(len(equity)):
        if bench_equity_list[i] > 0:
            excess_cum.append(round(equity[i] / bench_equity_list[i], 4))
        else:
            excess_cum.append(1.0)

    # Top holdings (demo)
    demo_sectors = {
        "银行": 0.12, "食品饮料": 0.10, "电子": 0.09, "医药生物": 0.08,
        "电力设备": 0.07, "汽车": 0.06, "非银金融": 0.05, "计算机": 0.05,
        "机械设备": 0.04, "化工": 0.04, "有色金属": 0.03, "公用事业": 0.03,
    }
    demo_tickers = [
        ("600519.SH", "食品饮料", 0.048, 1.2),
        ("601318.SH", "非银金融", 0.042, -0.8),
        ("600036.SH", "银行", 0.039, 0.5),
        ("000858.SZ", "食品饮料", 0.036, 2.1),
        ("601166.SH", "银行", 0.033, -0.3),
        ("600276.SH", "医药生物", 0.031, 1.7),
        ("002475.SZ", "电子", 0.028, -1.1),
        ("601012.SH", "电力设备", 0.026, 0.9),
        ("600887.SH", "食品饮料", 0.024, -0.5),
        ("000333.SZ", "机械设备", 0.022, 1.3),
        ("601888.SH", "商贸零售", 0.020, 2.4),
        ("002594.SZ", "汽车", 0.019, -1.8),
        ("600030.SH", "非银金融", 0.018, 0.6),
        ("000568.SZ", "食品饮料", 0.017, 0.2),
        ("601398.SH", "银行", 0.016, -0.1),
    ]
    top_holdings = [
        {"ticker": t[0], "weight": t[2], "sector": t[1], "pnl_pct": t[3]}
        for t in demo_tickers
    ]

    return {
        "performance": {
            "total_return": 0.582,
            "annual_return": 0.124,
            "annual_volatility": 0.231,
            "sharpe_ratio": 0.54,
            "sortino_ratio": 0.82,
            "calmar_ratio": 0.45,
            "max_drawdown": -0.278,
            "max_drawdown_peak": None,
            "max_drawdown_trough": None,
            "win_rate": 0.523,
            "profit_loss_ratio": None,
            "total_days": n,
            "information_ratio": None,
            "tracking_error": None,
            "excess_return": None,
            "n_rebalances": 60,
            "optimizer": "mean_variance",
            "initial_capital": 10_000_000,
        },
        "risk": {
            "historical_var": 0.028,
            "parametric_var": 0.026,
            "historical_cvar": 0.042,
        },
        "stress_tests": [
            {"scenario": "2008 Financial Crisis", "cumulative_return": -0.42, "max_drawdown": -0.55},
            {"scenario": "2015 A-Share Crash", "cumulative_return": -0.38, "max_drawdown": -0.48},
            {"scenario": "2020 COVID-19 Shock", "cumulative_return": 0.12, "max_drawdown": -0.15},
        ],
        "factors": demo_factors,
        "exposure": {
            "n_assets": 85,
            "effective_n": 52.3,
            "top5_concentration": 0.18,
            "top10_concentration": 0.32,
            "sectors": demo_sectors,
            "top_holdings": top_holdings,
        },
        "chart_data": {
            "dates": dates,
            "equity": equity,
            "benchmark": bench_equity_list,
            "drawdown": dd,
            "rolling_sharpe": rs,
            "rolling_sharpe_dates": rs_dates,
            "monthly_returns": {
                "years": ["2021", "2022", "2023", "2024", "2025"],
                "months": list(range(1, 13)),
                "data": [
                    [value / 100 if value is not None else None for value in row]
                    for row in (
                        [3.2, -1.5, 2.8, 0.9, -2.1, 1.4, 3.5, -0.8, 2.1, 1.7, -1.2, 2.9],
                        [-4.1, 1.2, -0.8, 2.3, -1.7, 3.1, -2.5, 0.6, 1.9, -3.2, 2.4, 0.5],
                        [1.8, -0.4, 2.5, 1.1, -0.9, 2.7, 0.3, -1.6, 3.4, 0.8, -2.1, 1.5],
                        [2.1, 1.5, -1.2, 3.8, 0.4, -0.7, 2.9, 1.3, -0.5, 4.1, 0.2, -1.8],
                        [1.2, -0.6, 2.4, None, None, None, None, None, None, None, None, None],
                    )
                ],
            },
            "return_distribution": distribution,
            "excess_cumulative": excess_cum,
            "turnover": [
                {"date": f"2021-{m:02d}-28", "turnover": round(rng.uniform(0.05, 0.25), 3), "n_trades": rng.randint(8, 35)}
                for m in range(1, 13)
            ] + [
                {"date": f"2022-{m:02d}-28", "turnover": round(rng.uniform(0.08, 0.30), 3), "n_trades": rng.randint(10, 40)}
                for m in range(1, 13)
            ] + [
                {"date": f"2023-{m:02d}-28", "turnover": round(rng.uniform(0.06, 0.22), 3), "n_trades": rng.randint(7, 30)}
                for m in range(1, 13)
            ] + [
                {"date": f"2024-{m:02d}-28", "turnover": round(rng.uniform(0.04, 0.18), 3), "n_trades": rng.randint(6, 25)}
                for m in range(1, 13)
            ] + [
                {"date": f"2025-{m:02d}-28", "turnover": round(rng.uniform(0.03, 0.15), 3), "n_trades": rng.randint(5, 20)}
                for m in range(1, 4)
            ],
            "drawdown_periods": [
                {"start": "2021-09-13", "end": "2021-11-02", "trough": "2021-10-12", "depth": -0.128, "duration_days": 50, "recovered": True, "recovery_date": "2021-11-02"},
                {"start": "2022-01-05", "end": "2022-04-27", "trough": "2022-03-15", "depth": -0.278, "duration_days": 112, "recovered": True, "recovery_date": "2022-04-27"},
                {"start": "2022-08-17", "end": "2022-11-01", "trough": "2022-10-12", "depth": -0.156, "duration_days": 76, "recovered": True, "recovery_date": "2022-11-01"},
                {"start": "2023-05-08", "end": "2023-06-28", "trough": "2023-06-01", "depth": -0.089, "duration_days": 51, "recovered": True, "recovery_date": "2023-06-28"},
                {"start": "2024-01-22", "end": "2024-03-15", "trough": "2024-02-05", "depth": -0.145, "duration_days": 53, "recovered": True, "recovery_date": "2024-03-15"},
            ],
            "factor_scatter": [
                {
                    "factor_name": "momentum_3m",
                    "points": [{"x": round(rng.gauss(0, 1), 2), "y": round(rng.gauss(0.5, 3), 2)} for _ in range(80)],
                    "ic": 0.0312, "icir": 0.25, "t_stat": 2.82,
                },
                {
                    "factor_name": "roe",
                    "points": [{"x": round(rng.gauss(0, 1), 2), "y": round(rng.gauss(0.3, 2.5), 2)} for _ in range(80)],
                    "ic": 0.0287, "icir": 0.29, "t_stat": 3.25,
                },
                {
                    "factor_name": "volatility_60d",
                    "points": [{"x": round(rng.gauss(0, 1), 2), "y": round(rng.gauss(-0.2, 2.8), 2)} for _ in range(80)],
                    "ic": -0.0251, "icir": -0.22, "t_stat": -2.47,
                },
                {
                    "factor_name": "pb_ratio",
                    "points": [{"x": round(rng.gauss(0, 1), 2), "y": round(rng.gauss(-0.1, 2.2), 2)} for _ in range(80)],
                    "ic": -0.0193, "icir": -0.18, "t_stat": -2.02,
                },
                {
                    "factor_name": "log_market_cap",
                    "points": [{"x": round(rng.gauss(0, 1), 2), "y": round(rng.gauss(-0.15, 2.0), 2)} for _ in range(80)],
                    "ic": -0.0178, "icir": -0.20, "t_stat": -2.24,
                },
                {
                    "factor_name": "momentum_1m",
                    "points": [{"x": round(rng.gauss(0, 1), 2), "y": round(rng.gauss(0.2, 3.2), 2)} for _ in range(80)],
                    "ic": 0.0218, "icir": 0.17, "t_stat": 1.90,
                },
            ],
            "attribution": [
                {"factor": "momentum_3m", "contribution_bps": 12.4, "weight": 0.22, "avg_ic": 0.0312},
                {"factor": "roe", "contribution_bps": 10.8, "weight": 0.20, "avg_ic": 0.0287},
                {"factor": "volatility_60d", "contribution_bps": -8.3, "weight": 0.15, "avg_ic": -0.0251},
                {"factor": "pb_ratio", "contribution_bps": -5.2, "weight": 0.12, "avg_ic": -0.0193},
                {"factor": "log_market_cap", "contribution_bps": -4.8, "weight": 0.11, "avg_ic": -0.0178},
                {"factor": "momentum_1m", "contribution_bps": 3.6, "weight": 0.08, "avg_ic": 0.0218},
                {"factor": "rsi_14d", "contribution_bps": 2.1, "weight": 0.06, "avg_ic": 0.0125},
                {"factor": "turnover_20d", "contribution_bps": -1.8, "weight": 0.06, "avg_ic": -0.0110},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Factors (latest run)
# ---------------------------------------------------------------------------

@router.get("/factors")
async def get_latest_factors():
    """Return factor IC rankings from the most recent completed run."""
    with _run_lock:
        if not _run_store:
            return {"factors": []}
        status = {rid: dict(s) for rid, s in _run_status.items()}
        store = dict(_run_store)
    latest_rid = max(
        store,
        key=lambda rid: status.get(rid, {}).get("completed_at", ""),
    )
    return {"factors": store[latest_rid].get("factors", [])}


# ---------------------------------------------------------------------------
# Recent runs
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_runs():
    """List recent runs with status."""
    with _run_lock:
        snapshot = [(rid, dict(s)) for rid, s in _run_status.items()]
    runs = []
    for rid, s in sorted(
        snapshot, key=lambda x: x[1].get("started_at", "") or "", reverse=True,
    ):
        runs.append({
            "run_id": rid,
            "status": s.get("status", "running"),
            "phase": s.get("phase", "executing"),
            "progress": s.get("progress", 0),
            "stage": s.get("stage", "data"),
            "started_at": s.get("started_at"),
            "completed_at": s.get("completed_at"),
        })
    return {"runs": runs[:20]}


# ---------------------------------------------------------------------------
# Compare (runs in the analytics pool)
# ---------------------------------------------------------------------------

@router.post("/compare", response_model=CompareResult)
async def compare_strategies(req: CompareRequest):
    """Compare multiple optimizers side by side."""
    return await _run_analytics(_run_compare, req)


def _run_compare(req: CompareRequest) -> CompareResult:
    config = _get_config_with_overrides({"universe.n_stocks": req.n_stocks})

    prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_tushare=False)
    processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover)
    signal = _generate_signal(config, processed_factors, returns)

    table = []
    for opt in req.optimizers:
        try:
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked, optimizer_override=opt,
                gates=_execution_gates(metadata),
            )
            s = bt_results["summary"]
            table.append({
                "Optimizer": opt,
                "Total Return %": round(s.get("total_return", 0) * 100, 2),
                "Ann. Return %": round(s.get("annual_return", 0) * 100, 2),
                "Ann. Vol %": round(s.get("annual_volatility", 0) * 100, 2),
                "Sharpe": round(s.get("sharpe_ratio", 0), 2),
                "Sortino": round(s.get("sortino_ratio", 0), 2),
                "Max DD %": round(s.get("max_drawdown", 0) * 100, 2),
                "Calmar": round(s.get("calmar_ratio", 0), 2),
                "Win Rate %": round(s.get("win_rate", 0) * 100, 1),
                "IR": round(s.get("information_ratio", 0), 2),
            })
        except Exception as e:
            table.append({"Optimizer": opt, "Error": str(e)[:80]})

    return CompareResult(table=table, chart_data=None)


# ---------------------------------------------------------------------------
# Sweep (runs in thread pool)
# ---------------------------------------------------------------------------

@router.post("/sweep", response_model=SweepResult)
async def sweep_parameters(req: SweepRequest):
    """Grid search over parameters."""
    return await _run_analytics(_run_sweep, req)


def _run_sweep(req: SweepRequest) -> SweepResult:
    import itertools

    config = load_config(None)
    table = []

    for opt, freq, n_stocks in itertools.product(
        req.optimizers, req.frequencies, req.n_stocks_list
    ):
        try:
            config.universe.n_stocks = n_stocks
            prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_tushare=False)
            processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
                prices, returns, financials, metadata, turnover)
            signal = _generate_signal(config, processed_factors, returns)
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked,
                optimizer_override=opt, frequency_override=freq,
                gates=_execution_gates(metadata),
            )
            s = bt_results["summary"]
            table.append({
                "Optimizer": opt,
                "Frequency": freq,
                "N Stocks": n_stocks,
                "Sharpe": round(s.get("sharpe_ratio", 0), 2),
                "Ann. Ret %": round(s.get("annual_return", 0) * 100, 1),
                "Max DD %": round(s.get("max_drawdown", 0) * 100, 1),
                "Sortino": round(s.get("sortino_ratio", 0), 2),
                "Win Rate %": round(s.get("win_rate", 0) * 100, 1),
            })
        except Exception as e:
            table.append({
                "Optimizer": opt, "Frequency": freq, "N Stocks": n_stocks,
                "Sharpe": "ERR", "Error": str(e)[:60],
            })

    best = None
    valid = [r for r in table if isinstance(r.get("Sharpe"), (int, float))]
    if valid:
        best = max(valid, key=lambda r: r["Sharpe"])
        best = {
            "optimizer": best["Optimizer"],
            "frequency": best["Frequency"],
            "n_stocks": best["N Stocks"],
            "sharpe": best["Sharpe"],
        }

    return SweepResult(table=table, best_params=best)


# ---------------------------------------------------------------------------
# Live Portfolio — Holdings Import & Tracking
# ---------------------------------------------------------------------------

# Global OMS instance (initialized lazily, guarded by `_singleton_lock`)
_oms_instance = None
_holdings_data: dict[str, Any] = {}

# Bound concurrent network-bound portfolio fetches. `/portfolio/live` is a
# sync handler, so Starlette would otherwise run as many of them as AnyIO's
# default thread limiter allows (40) -- each one issuing up to 100 sequential
# baostock queries against a module-global login session.
_live_portfolio_slots = threading.BoundedSemaphore(1)
_LIVE_PORTFOLIO_DEADLINE = 30.0


def _parse_holdings_rows(rows: list) -> dict[str, Any]:
    """Validate/normalize imported holdings (pandas work, off the loop)."""
    df = pd.DataFrame(rows)
    if "code" not in df.columns or "hold_vol" not in df.columns:
        # Try to infer columns
        if len(df.columns) >= 2:
            df.columns = ["code", "hold_vol"]
        else:
            raise HTTPException(400, "CSV must have 'code' and 'hold_vol' columns")

    # Clean data (vectorized, from reference project)
    df = df.dropna(subset=["code", "hold_vol"])
    df = df.drop_duplicates(subset="code", keep="first")
    df["hold_vol"] = pd.to_numeric(df["hold_vol"], errors="coerce")
    df = df[df["hold_vol"] > 0]
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    # `r"\\d{6}"` is a 6-character pattern -- escaped backslash, then six
    # literal `d`s -- so it matched nothing and only `\dddddd` would have
    # passed. Every import came back {"status": "ok", "n_stocks": 0}.
    before = len(df)
    df = df[df["code"].str.fullmatch(r"\d{6}")]
    if df.empty and before > 0:
        # Not a silent success: the file had rows, none of them usable.
        raise HTTPException(
            400,
            f"None of the {before} rows had a 6-digit code. Note that codes "
            "are zero-padded -- a numeric column read from a spreadsheet "
            "may have lost its leading zeros.",
        )

    return {
        "holdings": df.to_dict(orient="records"),
        "n_stocks": len(df),
        "total_shares": int(df["hold_vol"].sum()),
        "imported_at": datetime.now().isoformat(),
        "sample": df.head(5).to_dict(orient="records"),
    }


@router.post("/portfolio/import")
async def import_holdings(file_content: dict):
    """Import holdings from CSV data (code,hold_vol format)."""
    global _holdings_data
    try:
        rows = file_content.get("data", [])
        if not rows:
            raise HTTPException(400, "No data provided")
        if len(rows) > 10_000:
            raise HTTPException(
                413, "Portfolio import is limited to 10,000 rows",
            )

        # Pandas parsing/cleaning is CPU-bound; keep it off the event loop.
        parsed = await run_in_threadpool(_parse_holdings_rows, rows)
        sample = parsed.pop("sample")
        _holdings_data = parsed

        return {
            "status": "ok",
            "n_stocks": parsed["n_stocks"],
            "total_shares": parsed["total_shares"],
            "sample": sample,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.get("/portfolio/holdings")
async def get_holdings():
    """Get current imported holdings."""
    if not _holdings_data:
        return {"holdings": [], "n_stocks": 0}
    return _holdings_data


@router.get("/portfolio/live")
def get_live_portfolio():
    """Get live portfolio snapshot with current prices from baostock.

    Only one refresh runs at a time (`_live_portfolio_slots`); a second
    concurrent request gets 503 + Retry-After instead of occupying another
    worker thread for up to 100 sequential upstream queries. The sequential
    loop itself is capped by `_LIVE_PORTFOLIO_DEADLINE`: past the deadline the
    partial book is returned with `partial=true`.
    """
    if not _holdings_data or not _holdings_data.get("holdings"):
        # An HTTP error, not a 200 with an `error` key: the frontend treated
        # the latter as a valid empty portfolio and reported success.
        raise HTTPException(409, "No holdings imported. Import CSV first.")

    if not _live_portfolio_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="A live portfolio refresh is already in progress; retry shortly",
            headers={"Retry-After": "10"},
        )
    try:
        return _fetch_live_portfolio()
    finally:
        _live_portfolio_slots.release()


def _fetch_live_portfolio() -> dict:
    holdings = pd.DataFrame(_holdings_data["holdings"])
    codes = holdings["code"].tolist()

    try:
        # Fetch real prices from baostock
        import baostock as bs

        from quant_platform.data.providers.baostock_provider import _to_bs_code

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError("baostock login failed")

        prices = {}
        preclose_prices = {}
        price_dates = {}
        today = datetime.now().strftime("%Y-%m-%d")
        # Daily bars for *today* are not available until after the close
        # (baostock returned nothing intraday, so this endpoint answered 503
        # even though the single-stock endpoint served yesterday's close).
        # Ask for a short window and keep the latest bar, reporting its date.
        lookback = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        deadline = time.monotonic() + _LIVE_PORTFOLIO_DEADLINE
        deadline_hit = False
        for code in codes[:100]:  # Limit to 100 stocks for speed
            if time.monotonic() >= deadline:
                deadline_hit = True
                logger.warning(
                    "Live portfolio price fetch hit the %.0fs deadline after "
                    "%d/%d codes; returning partial data",
                    _LIVE_PORTFOLIO_DEADLINE, len(prices), min(len(codes), 100),
                )
                break
            bs_code = _to_bs_code(code)
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,close,preclose",
                    start_date=lookback, end_date=today,
                    frequency="d", adjustflag="3",
                )
                last_row = None
                while rs.error_code == "0" and rs.next():
                    last_row = rs.get_row_data()
                if last_row and len(last_row) >= 3:
                    close = float(last_row[1]) if last_row[1] else 0
                    preclose = float(last_row[2]) if last_row[2] else 0
                    if close > 0:
                        prices[code] = close
                        preclose_prices[code] = preclose
                        price_dates[code] = last_row[0]
            except Exception:
                continue

        bs.logout()

        if not prices:
            raise HTTPException(
                503,
                "No live prices available from baostock; try again during "
                "the trading session.",
            )

        # Merge holdings with prices (vectorized)
        holdings["close"] = holdings["code"].map(prices)
        holdings["preclose"] = holdings["code"].map(preclose_prices)
        holdings["market_value"] = holdings["hold_vol"] * holdings["close"]
        holdings["prev_value"] = holdings["hold_vol"] * holdings["preclose"]
        holdings["pnl"] = holdings["hold_vol"] * (holdings["close"] - holdings["preclose"])
        holdings["pnl_pct"] = (holdings["close"] / holdings["preclose"] - 1) * 100

        valid = holdings.dropna(subset=["close"])

        total_value = valid["market_value"].sum()
        total_pnl = valid["pnl"].sum()
        total_prev = valid["prev_value"].sum()
        daily_return = (total_pnl / total_prev * 100) if total_prev > 0 else 0

        # Top holdings by market value
        top = valid.nlargest(20, "market_value")
        top = top.assign(price_basis="raw")
        data_date = max(price_dates.values()) if price_dates else None

        return {
            "total_value": round(float(total_value), 2),
            "total_pnl": round(float(total_pnl), 2),
            "daily_return_pct": round(float(daily_return), 4),
            "n_positions": len(valid),
            "n_with_price": len(prices),
            "n_no_price": len(codes) - len(prices),
            "data_date": data_date,
            "stale": data_date != today,
            "partial": deadline_hit,
            # These marks come from baostock adjustflag="3" (unadjusted), i.e.
            # the traded CNY price -- never an hfq/qfq level (D15/D41).
            "price_basis": "raw",
            "price_source": "baostock:adjustflag=3",
            "holdings": top[["code", "hold_vol", "close", "preclose",
                             "market_value", "pnl", "pnl_pct",
                             "price_basis"]].to_dict(orient="records"),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        # 200 + {"error": ...} looked like a successful empty portfolio; raw
        # exception text leaked internals. Log with context, answer generically.
        logger.exception("Live portfolio fetch failed")
        raise _internal_error(
            503, "live_portfolio_unavailable",
            "Live portfolio unavailable; see server logs",
        ) from e


# ---------------------------------------------------------------------------
# Order Management System
# ---------------------------------------------------------------------------

# One lock for every lazy singleton below. The getters are reachable both from
# the event loop (async routes) and from AnyIO worker threads (sync routes);
# an unguarded `if x is None: x = construct()` could build two OMS books or
# two providers, the second overwriting and leaking the first.
_singleton_lock = threading.Lock()


def _get_oms() -> OrderManager:
    """Get or create the global OMS instance."""
    global _oms_instance
    instance = _oms_instance
    if instance is None:
        with _singleton_lock:
            if _oms_instance is None:
                from quant_platform.execution.oms import OrderManager
                _oms_instance = OrderManager(initial_cash=10_000_000.0)
            instance = _oms_instance
    return instance


_rt_market_instance = None


def _get_rt_market():
    """Shared RealTimeMarket.

    Each request used to construct a fresh instance, so its 10-second quote
    cache was always cold and every call triggered a full market scrape.
    """
    global _rt_market_instance
    instance = _rt_market_instance
    if instance is None:
        with _singleton_lock:
            if _rt_market_instance is None:
                from quant_platform.trading.realtime import RealTimeMarket
                _rt_market_instance = RealTimeMarket()
            instance = _rt_market_instance
    return instance


@router.post("/oms/order")
async def create_order(order_req: dict):
    """Create a new order."""
    # Validate before the broad `except Exception`: a missing ticker used to
    # create and submit an order with ticker "", which could then be filled
    # and turn into a position with no instrument.
    ticker = str(order_req.get("ticker", "") or "").strip()
    if not ticker:
        raise HTTPException(422, "ticker is required")
    # Canonical identity: `600519` and `600519.SH` must be one position.
    ticker = normalize_symbol(ticker)
    try:
        quantity = int(order_req.get("quantity", 100))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "quantity must be an integer") from exc
    if quantity <= 0:
        raise HTTPException(422, "quantity must be positive")
    side = order_req.get("side", "buy")
    if side not in ("buy", "sell"):
        raise HTTPException(422, "side must be 'buy' or 'sell'")
    limit_price = order_req.get("limit_price")
    if limit_price is not None:
        try:
            limit_price = float(limit_price)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "limit_price must be a number") from exc
        # NaN/Infinity made the OMS store NaN slippage, after which the JSON
        # blotter and TCA endpoints returned 500 until process restart.
        if not np.isfinite(limit_price) or limit_price <= 0:
            raise HTTPException(422, "limit_price must be a finite positive number")

    try:
        oms = _get_oms()
        order = oms.create_order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=order_req.get("order_type") or order_req.get("type", "market"),
            limit_price=limit_price,
            strategy=order_req.get("strategy", ""),
        )
        # Auto-submit for market orders
        if order_req.get("auto_submit", True):
            oms.submit_order(order.order_id)

        return {
            "order_id": order.order_id,
            "ticker": order.ticker,
            "side": order.side.value,
            "quantity": order.quantity,
            "status": order.status.value,
        }
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/oms/fill")
async def fill_order(fill_req: dict):
    """Process a fill for an order."""
    try:
        price = float(fill_req.get("price", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "price must be a number") from exc
    if price <= 0:
        # A missing price used to default to 0; the UI papered over it with a
        # random 100-150 quote. Refuse rather than invent a fill price.
        raise HTTPException(422, "price is required for a fill")

    try:
        oms = _get_oms()
        order = oms.fill_order(
            order_id=fill_req.get("order_id", ""),
            price=price,
            quantity=fill_req.get("quantity"),
            # Default raw; a caller that declares qfq/hfq is refused by the
            # OMS instead of poisoning cash and P&L (D15).
            price_basis=fill_req.get("price_basis", "raw"),
        )
        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "filled_qty": order.filled_quantity,
            "avg_price": round(order.avg_fill_price, 2),
            "price_basis": oms.price_basis,
        }
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/oms/new-trading-day")
async def oms_new_trading_day():
    """Roll the OMS book: release T+1 locks for a new trading day."""
    oms = _get_oms()
    oms.new_trading_day()
    return {"status": "ok", "n_positions": len(oms.positions)}


@router.get("/oms/blotter")
async def get_blotter(limit: int = Query(200, ge=1, le=2000)):
    """Get order blotter (all orders, newest first, bounded)."""
    oms = _get_oms()
    return {"orders": oms.get_order_blotter(limit=limit)}


@router.get("/oms/positions")
async def get_positions():
    """Get current positions."""
    oms = _get_oms()
    snapshot = oms.get_snapshot()
    return {
        "total_value": snapshot.total_value,
        "cash": snapshot.cash,
        "positions_value": snapshot.positions_value,
        "n_positions": snapshot.n_positions,
        "total_unrealized_pnl": snapshot.total_unrealized_pnl,
        "total_realized_pnl": snapshot.total_realized_pnl,
        # Fills and marks are accepted as raw CNY prices only (adjusted
        # qfq/hfq levels are rejected at the OMS boundary, D15).
        "price_basis": oms.price_basis,
        "positions": [
            {
                "ticker": p.ticker,
                "quantity": p.quantity,
                "avg_cost": round(p.avg_cost, 2),
                "current_price": round(p.current_price, 2),
                "market_value": round(p.market_value, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
                "realized_pnl": round(p.realized_pnl, 2),
                "weight": round(p.weight, 4),
                "price_basis": oms.price_basis,
            }
            for p in snapshot.positions
        ],
    }


@router.get("/oms/tca")
async def get_trade_cost_analysis():
    """Get trade cost analysis."""
    oms = _get_oms()
    return oms.get_trade_cost_analysis()


# ---------------------------------------------------------------------------
# Baostock Real Data
# ---------------------------------------------------------------------------

@router.get("/market/baostock/health")
def baostock_health():
    """Check if baostock is accessible."""
    try:
        import baostock as bs
        lg = bs.login()
        ok = lg.error_code == "0"
        bs.logout()
        return {"status": "ok" if ok else "error", "source": "baostock"}
    except Exception as e:
        # Short reason only: exception text can carry paths and library
        # internals. The detail stays in the server log.
        logger.exception("Baostock health check failed")
        return {
            "status": "error",
            "source": "baostock",
            "error": type(e).__name__,
            "code": "baostock_unavailable",
        }


@router.get("/market/baostock/stock/{code}")
def get_stock_realtime(code: str):
    """Get real-time price for a single stock from baostock."""
    try:
        import baostock as bs

        from quant_platform.data.providers.baostock_provider import _to_bs_code

        bs.login()
        bs_code = _to_bs_code(code.zfill(6))
        today = datetime.now().strftime("%Y-%m-%d")

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=today, end_date=today,
            frequency="d", adjustflag="3",
        )

        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        bs.logout()

        if not rows:
            # Try last trading day
            from datetime import timedelta
            yesterday = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            bs.login()
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,preclose,volume,amount,pctChg",
                start_date=yesterday, end_date=today,
                frequency="d", adjustflag="3",
            )
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            bs.logout()

        if not rows:
            raise HTTPException(404, f"No data found for {code}")

        row = rows[-1]  # Latest
        return {
            "code": code.zfill(6),
            "date": row[0],
            "open": float(row[1]) if row[1] else None,
            "high": float(row[2]) if row[2] else None,
            "low": float(row[3]) if row[3] else None,
            "close": float(row[4]) if row[4] else None,
            "preclose": float(row[5]) if row[5] else None,
            "volume": float(row[6]) if row[6] else None,
            "amount": float(row[7]) if row[7] else None,
            "pctChg": float(row[8]) if row[8] else None,
            "source": "baostock",
            # adjustflag="3" is the unadjusted (traded CNY) series.
            "price_basis": "raw",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Baostock stock quote failed for %s", code)
        raise _internal_error(
            500, "baostock_quote_failed",
            "Market data unavailable; see server logs",
        ) from e


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

@router.post("/walkforward")
async def run_walkforward(req: WalkForwardRequest):
    """Run a real walk-forward validation on the research context.

    The validator recomputes the alpha signal inside each fold from
    train-only data (``factors=processed_factors``), so the numbers are
    genuinely out-of-sample.  Nothing is sliced from a stored run's equity
    curve: ``run_id`` is accepted for compatibility with the dashboard but is
    only checked when supplied.
    """
    run_id = req.run_id
    if run_id is not None and run_id not in _run_store:
        raise HTTPException(400, "No completed run found. Run pipeline first.")

    train_period = req.train_period
    test_period = req.test_period
    folds = req.folds
    mode = req.mode

    start, end = _research_window(req.start_date, req.end_date)
    context = await _load_context_or_503(start, end)

    params = {
        "train_period": train_period,
        "test_period": test_period,
        "mode": mode,
        "folds": folds,
    }
    cached = _derived_get(context, "walkforward", params)
    if cached is not None:
        return cached
    return await _run_analytics(_run_walkforward_cached, context, params)


def _run_walkforward_cached(context: ResearchContext, params: dict) -> dict:
    """Single-flight wrapper: re-check the derived cache inside the worker."""
    key = derived_key(context, "walkforward", params)
    with _key_lock_for(key):
        hit = derived_cache_get(key)
        if hit is not None:
            out = dict(hit)
            out["cached"] = True
            return out
        result = _run_walkforward(context, params)
        return _derived_store(context, "walkforward", params, result)


def _walkforward_unavailable(context: ResearchContext, params: dict, reason: str) -> dict:
    """Shape-preserving unavailable response (WalkForward.vue guards on it)."""
    return {
        "available": False,
        "reason": reason,
        **context.provenance(),
        "n_folds": 0,
        "mode": params.get("mode", "rolling"),
        "train_period": params.get("train_period", 504),
        "test_period": params.get("test_period", 126),
        "fold_details": [],
        "oos_equity": [],
        "oos_dates": [],
        "stability": {
            "mean_sharpe": 0,
            "std_sharpe": 0,
            "min_sharpe": 0,
            "max_sharpe": 0,
            "sharpe_consistency": 0,
            "positive_folds": 0,
            "total_folds": 0,
            "mean_return_pct": 0,
        },
        "aggregate": {},
        "signal_recomputed": False,
    }


def _run_walkforward(context: ResearchContext, params: dict) -> dict:
    """Run ``WalkForwardValidator`` on the research context.

    Each fold refits the alpha combination on train-only data (the validator
    does this when ``factors`` is supplied), so the OOS metrics are not
    in-sample performance wearing a different label.
    """
    from quant_platform.backtest.cost_model import CostModel
    from quant_platform.backtest.walkforward import WalkForwardValidator
    from quant_platform.portfolio.constraints import PortfolioConstraints

    train_period = int(params["train_period"])
    test_period = int(params["test_period"])
    mode = params.get("mode", "rolling")
    folds = params.get("folds")

    n_sessions = int(len(context.prices.index))
    required = train_period + test_period
    if n_sessions < required:
        return _walkforward_unavailable(
            context, params,
            f"Walk-forward needs at least {required} sessions "
            f"(train {train_period} + test {test_period}); the research "
            f"context has {n_sessions}. Shorten the periods or extend the "
            "date range.",
        )
    if context.signal is None or context.signal.dropna(how="all").empty:
        return _walkforward_unavailable(
            context, params,
            "the research context produced an empty alpha signal; walk-forward "
            "has nothing to validate",
        )

    config = context.config
    validator = WalkForwardValidator(
        train_period=train_period,
        test_period=test_period,
        step_size=test_period,
        mode=mode,
        n_folds=folds,
    )
    cost_model = CostModel(
        commission=config.costs.commission,
        stamp_tax=config.costs.stamp_tax,
        slippage=config.costs.slippage,
        slippage_model=config.costs.slippage_model,
    )
    constraints = PortfolioConstraints(
        long_only=config.portfolio.constraints.long_only,
        max_weight=config.portfolio.constraints.max_weight,
        max_sector_exposure=config.portfolio.constraints.max_sector_exposure,
        max_turnover=config.portfolio.constraints.max_turnover,
        lot_size=config.portfolio.constraints.lot_size,
        risk_aversion=config.portfolio.risk_aversion,
    )
    engine_kwargs = {
        "initial_capital": config.backtest.initial_capital,
        "rebalance_frequency": config.backtest.rebalance_frequency,
        "benchmark": config.backtest.benchmark,
    }

    try:
        with warnings.catch_warnings():
            # Fold metrics on constant/empty return slices legitimately warn
            # (mean of an empty slice); the validator handles the NaN and the
            # response reports it as null.
            warnings.simplefilter("ignore")
            results = validator.run(
                signal=context.signal,
                prices=context.prices,
                returns=context.returns,
                benchmark_returns=context.benchmark,
                sector_map=context.sector_map,
                financials=context.fin_unstacked,
                engine_kwargs=engine_kwargs,
                factors=context.processed_factors,
                alpha_kwargs={
                    "method": config.alpha.method,
                    "lookback": config.alpha.lookback,
                    "min_icir": config.alpha.min_icir,
                },
                cost_model=cost_model,
                constraints=constraints,
                optimizer=config.portfolio.optimizer,
                covariance_method=config.portfolio.covariance.method,
                covariance_lookback=config.portfolio.covariance.lookback,
                covariance_ewma_half_life=config.portfolio.covariance.ewma_half_life,
                execution_gates=_execution_gates(context.metadata),
            )
    except ValueError:
        # The validator raises "All walk-forward folds failed" when every
        # fold backtest raises. Say that plainly; the traceback is in the log.
        logger.exception("Walk-forward: every fold failed")
        return _walkforward_unavailable(
            context, params,
            "no walk-forward fold completed successfully; see the server log "
            "for the failing fold's error",
        )

    oos_returns = results.get("oos_returns")
    oos_dates: list[str] = []
    oos_equity: list[Any] = []
    if oos_returns is not None and len(oos_returns) > 0:
        equity = (1.0 + oos_returns.astype(float)).cumprod()
        oos_equity = [_json_float(v) for v in equity.to_list()]
        oos_dates = [
            str(idx.date()) if hasattr(idx, "date") else str(idx)
            for idx in equity.index
        ]

    fold_details = []
    for metric in results.get("fold_metrics", []):
        train_start = metric.get("train_start", "")
        train_end = metric.get("train_end", "")
        test_start = metric.get("test_start", "")
        test_end = metric.get("test_end", "")
        fold_details.append({
            "fold": int(metric.get("fold", len(fold_details))),
            "train": f"{train_start}..{train_end}",
            "test": f"{test_start}..{test_end}",
            "oos_days": int(metric.get("oos_days", 0) or 0),
            "sharpe": _json_float(metric.get("sharpe_ratio")),
            "return_pct": (
                round(float(metric.get("total_return")) * 100.0, 6)
                if _json_float(metric.get("total_return")) is not None else None
            ),
            "max_drawdown_pct": (
                round(float(metric.get("max_drawdown")) * 100.0, 6)
                if _json_float(metric.get("max_drawdown")) is not None else None
            ),
            "sortino": _json_float(metric.get("sortino_ratio")),
        })

    stability_raw = results.get("stability") or {}
    stability = {
        "mean_sharpe": _json_float(stability_raw.get("mean_sharpe")),
        "std_sharpe": _json_float(stability_raw.get("std_sharpe")),
        "min_sharpe": _json_float(stability_raw.get("min_sharpe")),
        "max_sharpe": _json_float(stability_raw.get("max_sharpe")),
        "sharpe_consistency": _json_float(
            stability_raw.get("sharpe_consistency"),
        ),
        "positive_folds": int(stability_raw.get("positive_folds", 0) or 0),
        "total_folds": int(stability_raw.get("total_folds", 0) or 0),
        "mean_return_pct": (
            round(float(stability_raw.get("mean_return")) * 100.0, 6)
            if _json_float(stability_raw.get("mean_return")) is not None else None
        ),
    }

    aggregate_raw = results.get("aggregate_metrics") or {}
    aggregate = {
        key: _json_float(aggregate_raw.get(key))
        for key in (
            "sharpe_ratio", "annual_return", "annual_volatility",
            "total_return", "max_drawdown", "sortino_ratio",
            "calmar_ratio", "win_rate",
        )
        if key in aggregate_raw
    }

    return {
        "available": True,
        "reason": "",
        **context.provenance(),
        # `n_folds` is the number of folds that produced a result (what the
        # panel reports as completed); the validator's planned count is kept
        # separately.
        "n_folds": len(fold_details),
        "planned_folds": int(results.get("n_folds", len(fold_details))),
        "mode": results.get("mode", mode),
        "train_period": int(results.get("train_period", train_period)),
        "test_period": int(results.get("test_period", test_period)),
        "fold_details": fold_details,
        "oos_equity": oos_equity,
        "oos_dates": oos_dates,
        "stability": stability,
        "aggregate": aggregate,
        "signal_recomputed": bool(results.get("signal_recomputed", False)),
    }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

@router.post("/montecarlo")
async def run_monte_carlo(req: dict):
    """Run Monte Carlo simulation on the latest pipeline run."""
    run_id = req.get("run_id")
    if not run_id or run_id not in _run_store:
        raise HTTPException(400, "No completed run found. Run pipeline first.")

    result = _run_store[run_id]

    try:
        n_sims = int(req.get("n_simulations", 1000))
        horizon = int(req.get("horizon_days", 252))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "n_simulations/horizon_days must be integers") from exc
    if not 1 <= n_sims <= 10_000 or not 1 <= horizon <= 2_520:
        raise HTTPException(
            422, "n_simulations must be 1..10000 and horizon_days 1..2520",
        )

    return await _run_analytics(_run_monte_carlo, result, req)


def _run_monte_carlo(result: dict, params: dict) -> dict:
    """Execute Monte Carlo simulation."""
    from quant_platform.risk.monte_carlo import MonteCarloSimulator

    n_sims = params.get("n_simulations", 1000)
    horizon = params.get("horizon_days", 252)
    method = params.get("method", "bootstrap")

    # Reconstruct daily returns from chart data
    chart = result.get("chart_data", {})
    equity = chart.get("equity", [])
    if len(equity) < 2:
        return {"error": "Not enough equity data for simulation"}

    equity_arr = np.array(equity)
    daily_returns = pd.Series(np.diff(equity_arr) / equity_arr[:-1])

    simulator = MonteCarloSimulator(
        n_simulations=n_sims,
        horizon_days=horizon,
    )

    if method == "parametric":
        mc_result = simulator.parametric_simulation(daily_returns)
    else:
        mc_result = simulator.bootstrap_simulation(daily_returns, block_size=21)

    # Convert numpy types for JSON serialization
    terminal_values = np.asarray(mc_result.get("terminal_values", []), dtype=float)
    if terminal_values.size > 2_000:
        step = int(np.ceil(terminal_values.size / 2_000))
        terminal_values = terminal_values[::step]

    return {
        "method": mc_result["method"],
        "n_simulations": mc_result["n_simulations"],
        "horizon_days": mc_result["horizon_days"],
        "terminal_value": mc_result["terminal_value"],
        "annual_return": mc_result["annual_return"],
        "max_drawdown": mc_result["max_drawdown"],
        "sharpe": mc_result.get("sharpe", {}),
        "fitted_distribution": mc_result.get("fitted_distribution"),
        "terminal_values": terminal_values.tolist(),
        "paths": mc_result.get("paths", [])[:30],
    }


# ---------------------------------------------------------------------------
# Factor Risk Decomposition
# ---------------------------------------------------------------------------

@router.post("/risk/decompose")
async def decompose_risk(req: RiskDecomposeRequest):
    """Decompose portfolio risk with the real Barra model.

    Uses the same computation as ``POST /api/barra/decompose``
    (:func:`_barra_decomposition`) on the shared research context instead of
    re-deriving a risk share from stored IC summaries.
    """
    run_id = req.run_id
    if run_id is not None and run_id not in _run_store:
        raise HTTPException(400, "No completed run found. Run pipeline first.")

    half_life = req.half_life
    date_value = req.date
    if date_value:
        try:
            from datetime import date as _date
            _date.fromisoformat(str(date_value))
        except ValueError as exc:
            raise HTTPException(
                422, "date must be an ISO date (YYYY-MM-DD)",
            ) from exc
    shrinkage = req.shrinkage_target

    start, end = _research_window(req.start_date, req.end_date)
    context = await _load_context_or_503(start, end)

    params = {
        "weights": req.weights,
        "date": date_value,
        "half_life": half_life,
        "shrinkage_target": shrinkage,
    }
    # Without explicit weights the book comes from OMS positions, which change
    # between requests; caching that result by the research-context hash would
    # serve a stale book for the TTL. Only explicit-weight results are cached.
    if req.weights:
        cached = _derived_get(context, "risk_decompose", params)
        if cached is not None:
            return cached
        return await _run_analytics(_decompose_risk_cached, context, params)
    return await _run_analytics(_decompose_risk, context, params)


def _decompose_risk_cached(context: ResearchContext, params: dict) -> dict:
    key = derived_key(context, "risk_decompose", params)
    with _key_lock_for(key):
        hit = derived_cache_get(key)
        if hit is not None:
            out = dict(hit)
            out["cached"] = True
            return out
        result = _decompose_risk(context, params)
        return _derived_store(context, "risk_decompose", params, result)


def _risk_decomposition_unavailable(
    context: ResearchContext, reason: str,
) -> dict:
    return {
        "available": False,
        "reason": reason,
        **context.provenance(),
        "factors": [],
        "total_risk_pct": None,
        "factor_risk_pct": None,
        "idiosyncratic_risk_pct": None,
        "r_squared": None,
        "risk_share": {
            "convention": (
                "Euler variance shares: factor variance contribution / total "
                "variance; the alpha row is the specific variance share"
            ),
            "total": None,
            "sums_to_one": False,
        },
        "weights_source": "",
    }


def _barra_decomposition(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark: Any,
    financials: Any,
    processed_factors: dict[str, pd.DataFrame],
    *,
    weights_requested: dict[str, float] | None,
    date: str | None,
    half_life: int,
    shrinkage_target: str,
    raw_prices: pd.DataFrame | None,
):
    """Shared Barra fit/decompose used by both risk endpoints.

    Returns ``(model, decomposition, weights, weights_source)``.
    """
    from quant_platform.risk.barra import BarraModel

    weights, weights_source = _resolve_weights_with_source(
        weights_requested, prices, raw_prices,
    )
    model = BarraModel(
        half_life=half_life,
        shrinkage_target=shrinkage_target,
    )
    factor_exposures = model.build_exposures(
        prices,
        returns,
        benchmark=benchmark,
        financials=financials,
        platform_factors=processed_factors,
    )
    model.fit(factor_exposures, returns)
    decomposition = model.decompose_risk(
        weights, factor_exposures, date=date,
    )
    return model, decomposition, weights, weights_source


def _decompose_risk(context: ResearchContext, params: dict) -> dict:
    """Real Barra risk decomposition on the research context.

    Risk shares are Euler variance contributions: each factor's annualized
    variance contribution divided by the portfolio's total annualized
    variance (``total_risk ** 2``).  The idiosyncratic ("alpha") row is the
    specific variance share, so factor shares plus the alpha row sum to 1.
    """
    raw_prices = None
    attrs = getattr(context.metadata, "attrs", None)
    if isinstance(attrs, dict):
        raw_block = attrs.get("raw_prices")
        if isinstance(raw_block, dict):
            raw_prices = raw_block.get("close")

    requested_date = params.get("date")
    if requested_date:
        try:
            stamp = pd.Timestamp(requested_date)
        except (TypeError, ValueError):
            return _risk_decomposition_unavailable(
                context, f"date {requested_date!r} is not a valid date",
            )
        if stamp not in context.prices.index:
            return _risk_decomposition_unavailable(
                context,
                f"date {requested_date} is not in the research context "
                f"({context.prices.index[0]}..{context.prices.index[-1]})",
            )

    try:
        with warnings.catch_warnings():
            # The cross-sectional regressions emit constant-input warnings on
            # thin dates; they are handled by the model, not endpoint errors.
            warnings.simplefilter("ignore")
            model, decomposition, _weights, weights_source = _barra_decomposition(
                context.prices,
                context.returns,
                context.benchmark,
                context.financials,
                context.processed_factors,
                weights_requested=params.get("weights"),
                date=params.get("date"),
                half_life=int(params.get("half_life", 252)),
                shrinkage_target=str(params.get("shrinkage_target", "identity")),
                raw_prices=raw_prices,
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Risk decomposition failed")
        return _risk_decomposition_unavailable(
            context,
            "the Barra cross-sectional fit failed on this research context; "
            "see the server log",
        )

    total_risk = _json_float(decomposition.total_risk, 0.0) or 0.0
    total_variance = total_risk * total_risk
    if total_variance <= 0:
        return _risk_decomposition_unavailable(
            context,
            "the resolved portfolio has zero measured variance against the "
            "factor model (an empty or degenerate book)",
        )

    factor_returns = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factor_returns = model.get_factor_return_series()
    except Exception:  # statistics are optional; shares are not
        logger.exception("Barra factor-return series unavailable")

    factors: list[dict] = []
    share_sum = 0.0
    non_finite_factors: list[str] = []
    for name, contribution in decomposition.factor_contributions.items():
        contribution_value = _json_float(contribution)
        if contribution_value is None:
            # A factor whose variance contribution is undefined cannot be
            # given a share; record it rather than inventing a number.
            non_finite_factors.append(str(name))
            contribution_value = 0.0
        share = contribution_value / total_variance
        share_sum += share
        annual_bps = None
        t_stat = None
        if factor_returns is not None and name in factor_returns.columns:
            series = factor_returns[name].dropna()
            if len(series) >= 2:
                mean = float(series.mean())
                std = float(series.std())
                annual_bps = _json_float(mean * 252.0 * 1e4)
                if std > 0:
                    t_stat = _json_float(
                        mean / std * math.sqrt(len(series)),
                    )
                if annual_bps is not None:
                    annual_bps = round(annual_bps, 4)
                if t_stat is not None:
                    t_stat = round(t_stat, 4)
        factors.append({
            "factor": str(name),
            "risk_share": share,
            "risk_share_pct": share * 100.0,
            "variance_contribution": contribution_value,
            "annual_return_bps": annual_bps,
            "beta": _json_float(decomposition.factor_exposures.get(name)),
            "t_stat": t_stat,
            "contribution": "factor",
        })

    specific_risk = _json_float(decomposition.specific_risk, 0.0) or 0.0
    idio_share = (specific_risk * specific_risk) / total_variance
    # Floating-point residual goes to the idiosyncratic bucket so the shares
    # add to exactly 1 for the panel's "sums to one" contract.
    idio_share += 1.0 - (share_sum + idio_share)
    share_sum += idio_share
    factors.append({
        "factor": "alpha",
        "risk_share": idio_share,
        "risk_share_pct": idio_share * 100.0,
        "variance_contribution": specific_risk * specific_risk,
        "annual_return_bps": None,
        "beta": None,
        "t_stat": None,
        "contribution": "alpha",
    })

    return {
        "available": True,
        "reason": "",
        **context.provenance(),
        "factors": factors,
        "total_risk_pct": total_risk * 100.0,
        "factor_risk_pct": (_json_float(decomposition.factor_risk, 0.0) or 0.0) * 100.0,
        "idiosyncratic_risk_pct": specific_risk * 100.0,
        "r_squared": _json_float(decomposition.r_squared, 0.0),
        "risk_share": {
            "convention": (
                "Euler variance shares: factor variance contribution / total "
                "variance; the alpha row is the specific variance share"
            ),
            "total": share_sum,
            "sums_to_one": abs(share_sum - 1.0) <= 1e-9,
        },
        "factor_stat_method": (
            "annual_return_bps = mean daily cross-sectional factor return x "
            "252 x 1e4; t_stat = mean/std x sqrt(n); beta = portfolio "
            "exposure to the factor"
        ),
        "non_finite_contributions": non_finite_factors,
        "weights_source": weights_source,
    }


# ---------------------------------------------------------------------------
# Factor Analysis (IC Decay + Correlation Matrix)
# ---------------------------------------------------------------------------

#: Horizons reported by /api/analysis/ic-decay. ``factors.evaluation.ic_decay``
#: computes every horizon from 1 to ``max_periods``; computing all twenty for
#: every factor would be twenty IC passes over the full panel, so the endpoint
#: evaluates ``rank_ic(period=h).mean()`` for the reported horizons only --
#: exactly the quantity ``ic_decay`` returns at those horizons.
IC_DECAY_HORIZONS = (1, 5, 10, 20)


@router.get("/analysis/ic-decay")
async def get_ic_decay(
    start_date: str | None = Query(None, description="ISO start date override"),
    end_date: str | None = Query(None, description="ISO end date override"),
):
    """IC decay at horizons 1/5/10/20 days on the research context."""
    start, end = _research_window(start_date, end_date)
    context = await _load_context_or_503(start, end)

    cached = _derived_get(context, "ic_decay", None)
    if cached is not None:
        return cached
    return await _run_analytics(_compute_ic_decay_cached, context)


def _compute_ic_decay_cached(context: ResearchContext) -> dict:
    key = derived_key(context, "ic_decay", None)
    with _key_lock_for(key):
        hit = derived_cache_get(key)
        if hit is not None:
            out = dict(hit)
            out["cached"] = True
            return out
        return _derived_store(
            context, "ic_decay", None, _compute_ic_decay(context),
        )


def _analysis_window(
    context: ResearchContext,
) -> tuple[dict[str, pd.DataFrame], int, str]:
    """Trailing factor panels used by the factor-statistics endpoints.

    Only the factor panels are truncated; the returns panel stays full so the
    last window dates keep their forward-return horizons.  Panels with no
    value at all in the window (e.g. 12-month momentum on a short sample) are
    omitted -- they cannot contribute a measurement.  ``window_sessions`` is
    the longest tail actually available and ``panel_start`` the earliest
    start, i.e. "up to the last 252 sessions per panel".
    """
    factors: dict[str, pd.DataFrame] = {}
    lengths: list[int] = []
    starts: list[Any] = []
    for name, panel in context.processed_factors.items():
        if panel is None or panel.empty:
            continue
        tail = panel.tail(ANALYSIS_WINDOW_SESSIONS)
        if not tail.notna().to_numpy().any():
            continue
        factors[name] = tail
        lengths.append(len(tail.index))
        starts.append(tail.index[0])
    window_sessions = max(lengths) if lengths else 0
    panel_start = str(min(starts))[:10] if starts else ""
    return factors, window_sessions, panel_start


def _count_factor_assets(factors: dict[str, pd.DataFrame]) -> int:
    widths = [len(panel.columns) for panel in factors.values()]
    return max(widths) if widths else 0


def _analysis_returns(
    context: ResearchContext, factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Returns sliced to the analysis window.

    ``rank_ic`` rebuilds the forward-return target over whatever panel it is
    given, so handing it the full five-year returns frame per factor per
    horizon dominated the endpoint's cost.  Slicing to the window start keeps
    every horizon's target identical (the target for a date only needs the
    following ``h`` returns, all inside the slice) while bounding the work.
    """
    if not factors:
        return context.returns.iloc[0:0]
    starts = [panel.index[0] for panel in factors.values()]
    return context.returns.loc[min(starts):]


def _compute_ic_decay(context: ResearchContext) -> dict:
    from quant_platform.factors.evaluation import rank_ic

    factors, window_sessions, panel_start = _analysis_window(context)
    returns_window = _analysis_returns(context, factors)
    factors_out: list[dict] = []
    with warnings.catch_warnings():
        # Constant cross-sections and short windows legitimately raise
        # numerical warnings inside the IC computation; they are not endpoint
        # errors and must not flood the log (or pytest's warning capture) once
        # per date per factor.
        warnings.simplefilter("ignore")
        for name, panel in factors.items():
            if panel is None or panel.empty:
                continue
            points: list[dict] = []
            for horizon in IC_DECAY_HORIZONS:
                ic_series = rank_ic(panel, returns_window, period=horizon)
                if len(ic_series) == 0:
                    continue
                mean_ic = _json_float(ic_series.mean())
                if mean_ic is None:
                    continue
                points.append({
                    "horizon": int(horizon),
                    "ic": mean_ic,
                    "n": int(ic_series.count()),
                })
            if not points:
                continue
            factors_out.append({
                "factor": str(name),
                # ICDecay.vue reads {factor, lags, ics}; `points` adds the
                # per-horizon observation count.
                "lags": [p["horizon"] for p in points],
                "ics": [p["ic"] for p in points],
                "points": points,
            })

    if not factors_out:
        from quant_platform.factors.evaluation import min_assets_for_ic

        breadth = min_assets_for_ic(_count_factor_assets(factors) or context.n_assets)
        return {
            "available": False,
            "reason": (
                "no processed factor produced an evaluable cross-section at "
                f"the requested horizons; IC needs at least {breadth} assets "
                "on a date"
            ),
            **context.provenance(),
            "horizons": list(IC_DECAY_HORIZONS),
            "window_sessions": window_sessions,
            "panel_start": panel_start,
            "factors": [],
        }

    return {
        "available": True,
        "reason": "",
        **context.provenance(),
        "horizons": list(IC_DECAY_HORIZONS),
        "window_sessions": window_sessions,
        "panel_start": panel_start,
        "ic_definition": (
            "mean cross-sectional Rank IC of the processed factor against the "
            "pipeline's forward returns at each horizon (identical to the "
            "value factors.evaluation.ic_decay reports for that horizon), "
            "over the trailing window"
        ),
        "factors": factors_out,
    }


@router.get("/analysis/correlation")
async def get_factor_correlation(
    start_date: str | None = Query(None, description="ISO start date override"),
    end_date: str | None = Query(None, description="ISO end date override"),
):
    """Average cross-sectional factor correlation matrix on the factor panel."""
    start, end = _research_window(start_date, end_date)
    context = await _load_context_or_503(start, end)

    cached = _derived_get(context, "factor_correlation", None)
    if cached is not None:
        return cached
    return await _run_analytics(_compute_factor_correlation_cached, context)


def _compute_factor_correlation_cached(context: ResearchContext) -> dict:
    key = derived_key(context, "factor_correlation", None)
    with _key_lock_for(key):
        hit = derived_cache_get(key)
        if hit is not None:
            out = dict(hit)
            out["cached"] = True
            return out
        return _derived_store(
            context, "factor_correlation", None,
            _compute_factor_correlation(context),
        )


def _pairwise_factor_correlation(
    factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Pairwise cross-sectional factor correlation.

    ``factors.evaluation.factor_correlation`` requires a date on which *every*
    factor is present before it will correlate the cross-section; a short or
    sparse panel (e.g. a 12-month momentum factor with no history yet) can
    make every date incomplete and the matrix all-NaN.  This fallback keeps
    the same measurement -- per-date Pearson correlation of the processed
    factor values, averaged over dates -- but pairs factors individually and
    only counts a date for a pair when both have at least the IC breadth
    minimum of common assets.  Pairs with no such date stay NaN (reported as
    null).
    """
    from quant_platform.factors.evaluation import min_assets_for_ic

    names = list(factors)
    size = len(names)
    sums = np.zeros((size, size), dtype=float)
    counts = np.zeros((size, size), dtype=float)

    all_dates: set[Any] = set()
    for panel in factors.values():
        all_dates.update(panel.index)

    for day in sorted(all_dates):
        vectors: dict[str, pd.Series] = {}
        for name in names:
            panel = factors[name]
            if day in panel.index:
                vectors[name] = panel.loc[day].dropna()
        if len(vectors) < 2:
            continue
        # Reindex to every factor so the pairwise breadth mask and the
        # correlation matrix stay size x size even when some panel has no
        # observation on this date (missing columns become all-NaN).
        frame = pd.DataFrame(vectors).reindex(columns=names)
        if frame.dropna(how="all").empty:
            continue
        present = frame.notna()
        common = present.T.to_numpy(dtype=float) @ present.to_numpy(dtype=float)
        threshold = min_assets_for_ic(len(frame))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (
                frame.corr()
                .reindex(index=names, columns=names)
                .to_numpy(dtype=float)
            )
        mask = (common >= threshold) & np.isfinite(corr)
        if not mask.any():
            continue
        sums[mask] += corr[mask]
        counts[mask] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(counts > 0, sums / counts, np.nan)
    return pd.DataFrame(averaged, index=names, columns=names)


def _compute_factor_correlation(context: ResearchContext) -> dict:
    from quant_platform.factors.evaluation import factor_correlation

    factors, window_sessions, panel_start = _analysis_window(context)
    if not factors:
        return {
            "available": False,
            "reason": "the research context has no processed factor panels",
            **context.provenance(),
            "window_sessions": window_sessions,
            "panel_start": panel_start,
            "names": [],
            "matrix": [],
        }

    with warnings.catch_warnings():
        # Pairwise correlations over sparse cross-sections emit constant-input
        # warnings per date; they are expected and not endpoint errors.
        warnings.simplefilter("ignore")
        corr = factor_correlation(factors)
        off_diagonal = corr.to_numpy(dtype=float).copy()
        np.fill_diagonal(off_diagonal, np.nan)
        method = "factors.evaluation.factor_correlation"
        if not np.isfinite(off_diagonal).any():
            # No date had every factor present at the breadth minimum. Fall
            # back to the pairwise version so a sparse panel still yields
            # measured correlations instead of an empty matrix.
            corr = _pairwise_factor_correlation(factors)
            method = (
                "pairwise cross-sectional factor correlation "
                "(factor_correlation had no complete cross-section)"
            )

    names = [str(name) for name in corr.columns]
    size = len(names)
    if size == 0:
        return {
            "available": False,
            "reason": (
                "the factor panels share no date with enough assets for a "
                "cross-sectional correlation"
            ),
            **context.provenance(),
            "window_sessions": window_sessions,
            "panel_start": panel_start,
            "names": [],
            "matrix": [],
        }

    matrix: list[list[float | None]] = []
    for i in range(size):
        row: list[float | None] = []
        for j in range(size):
            value = _json_float(corr.iat[i, j])
            if i == j:
                value = 1.0
            elif value is not None:
                value = float(np.clip(value, -1.0, 1.0))
            row.append(value)
        matrix.append(row)

    # Make the matrix exactly symmetric and treat a missing pair as missing
    # on both sides (the pair has no date where both factors are evaluable).
    for i in range(size):
        for j in range(i + 1, size):
            left, right = matrix[i][j], matrix[j][i]
            if left is None or right is None:
                matrix[i][j] = matrix[j][i] = None
            else:
                value = (left + right) / 2.0
                matrix[i][j] = matrix[j][i] = value

    return {
        "available": True,
        "reason": "",
        **context.provenance(),
        "window_sessions": window_sessions,
        "panel_start": panel_start,
        "names": names,
        "matrix": matrix,
        "nan_policy": (
            "per-date cross-sectional Pearson correlations averaged across "
            "dates on which both factors have at least the IC breadth minimum "
            "of common assets; a pair with no such date is null on both "
            "sides. The diagonal is 1.0 by construction."
        ),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------

@router.post("/regime/detect")
async def detect_regime(req: dict):
    """Detect current market regime from pipeline run data."""
    run_id = req.get("run_id")
    if not run_id or run_id not in _run_store:
        raise HTTPException(400, "No completed run found. Run pipeline first.")

    result = _run_store[run_id]
    regime = await _run_analytics(_detect_regime, result)
    if isinstance(regime, dict) and "error" in regime:
        # 200 + {"error": ...} reads as a successful empty result to clients.
        raise HTTPException(409, regime["error"])
    return regime


def _detect_regime(result: dict) -> dict:
    """Run regime detection on stored run data."""
    from quant_platform.risk.regime import CompositeRegimeDetector

    chart = result.get("chart_data", {})
    equity = chart.get("equity", [])
    dates = chart.get("dates", [])

    if len(equity) < 252:
        return {
            "error": (
                "Regime detection needs at least 252 daily observations "
                f"(the volatility lookback); this run has {len(equity)}."
            )
        }

    equity_arr = np.array(equity)
    returns = pd.Series(np.diff(equity_arr) / equity_arr[:-1])
    prices = pd.Series(equity_arr)

    detector = CompositeRegimeDetector()
    regime = detector.detect(returns, prices)
    # A stored run keeps only the portfolio equity curve, not the asset
    # return panel, so a correlation regime cannot be measured. The detector
    # otherwise reports a default "normal_correlation" that looks measured.
    regime["correlation"] = {
        "regime": "unavailable",
        "confidence": 0.0,
        "available": False,
        "reason": (
            "the stored run does not keep the asset return panel needed for "
            "average pairwise correlation"
        ),
    }

    # Generate regime history (monthly snapshots)
    regime_history = []
    for i in range(252, len(returns), 21):
        window_returns = returns.iloc[max(0, i - 252):i]
        window_prices = prices.iloc[max(0, i - 252):i + 1]
        r = detector.vol_detector.detect(window_returns)
        t = detector.trend_detector.detect(window_prices)
        regime_history.append({
            "date": dates[i] if i < len(dates) else dates[-1],
            "volatility": r.get("regime", "medium_volatility"),
            "trend": t.get("regime", "sideways"),
            "vol_percentile": r.get("percentile", 0.5),
        })

    regime["history"] = regime_history[-30:]  # Last 30 months
    return regime


# ---------------------------------------------------------------------------
# Screener: Boolean Factor Screening
# ---------------------------------------------------------------------------

@router.post("/screen", response_model=ScreenResponse)
def screen_stocks(req: ScreenRequest):
    """Screen stocks using boolean factor rules (bounded compute pool)."""
    return _run_bounded_compute(_screen_stocks_impl, req)


def _screen_stocks_impl(req: ScreenRequest) -> ScreenResponse:
    """Screen stocks using boolean factor rules.

    Computes factors for the configured universe then applies user-defined
    rules (e.g. "pe_ratio < 30 AND roe > 0.15"). Returns qualifying stocks
    with their factor values.
    """
    from quant_platform.portfolio.screener import (
        FactorScreener,
        ScreenRule,
    )

    # A request-supplied path was passed straight to `open()`, so an
    # unauthenticated caller could execute an arbitrary local YAML config
    # (path traversal / config injection). Only files under the server's
    # config directory are accepted.
    config_path = req.config
    if config_path:
        from pathlib import Path

        config_root = (Path(__file__).resolve().parent.parent / "config").resolve()
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = (config_root.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_relative_to(config_root) or not candidate.is_file():
            raise HTTPException(
                422,
                "config must name a YAML file inside the server's config/ directory",
            )
        config_path = str(candidate)

    config = load_config(config_path)

    n_stocks = getattr(req, 'n_stocks', None)
    if n_stocks is not None:
        config.universe.n_stocks = n_stocks
    use_baostock = getattr(req, 'use_baostock', False)

    try:
        prices, returns, benchmark, metadata, financials, turnover = _load_data(
            config, use_baostock=use_baostock
        )
    except Exception as e:
        logger.exception("Screen: data load failed")
        raise _internal_error(
            503, "screen_data_unavailable",
            "Data load failed; see server logs for details",
        ) from e

    try:
        _processed_factors, ic_results, sector_map, fin_unstacked, raw_factors = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config,
            return_raw=True,
        )
    except Exception as e:
        logger.exception("Screen: factor computation failed")
        raise _internal_error(
            500, "screen_factor_error",
            "Factor computation failed; see server logs for details",
        ) from e

    if not req.rules:
        raise HTTPException(422, "at least one screen rule is required")
    unknown = sorted({r.factor for r in req.rules} - set(raw_factors))
    if unknown:
        raise HTTPException(
            422,
            f"unknown factor(s): {unknown}; available: {sorted(raw_factors)}",
        )

    rules = []
    for r in req.rules:
        rules.append(ScreenRule(factor=r.factor, operator=r.operator, value=r.value))

    # Screen raw economic values, not the standardized/neutralized panels:
    # `pe_ratio < 30` means a PE of 30, while the processed panel is a
    # cross-sectional z-score (~-3..3) that passes nearly every stock.
    screen_panel = raw_factors

    screener = FactorScreener({
        "enabled": True,
        "rules": [{"factor": r.factor, "operator": r.operator, "value": r.value}
                  for r in rules],
        "logic": req.logic,
        "min_stocks": req.min_stocks,
        "max_stocks": req.max_stocks,
    })
    qualifiers = screener.screen(screen_panel, rules, date=req.date)

    target_date = req.date or str(next(iter(screen_panel.values())).index[-1])
    results = []
    for asset in qualifiers:
        factor_vals = {}
        for r in rules:
            if r.factor in screen_panel:
                df = screen_panel[r.factor]
                try:
                    val = float(df.loc[pd.Timestamp(target_date), asset])
                    factor_vals[r.factor] = round(val, 6)
                except (KeyError, TypeError):
                    pass
        results.append(ScreenStockInfo(code=asset, factors=factor_vals))

    return ScreenResponse(
        date=target_date,
        logic=req.logic,
        num_rules=len(rules),
        total_stocks=len(next(iter(screen_panel.values())).columns),
        qualifying_stocks=len(qualifiers),
        rules=[f"{r.factor} {r.operator} {r.value}" for r in rules],
        results=results,
    )


# ---------------------------------------------------------------------------
# Risk Monitor Status
# ---------------------------------------------------------------------------

# Global risk monitor instance
_risk_monitor = None


def _get_risk_monitor():
    """Return the process-wide RiskMonitor.

    This previously built its own instance with an explicit RiskLimits(...)
    whose five values are identical to the dataclass defaults, so routing it
    through the shared singleton does not change any limit -- it only makes
    /api/risk/* act on the same object the engine checks.
    """
    global _risk_monitor
    instance = _risk_monitor
    if instance is None:
        with _singleton_lock:
            if _risk_monitor is None:
                from quant_platform.risk.circuit_breaker import get_risk_monitor
                _risk_monitor = get_risk_monitor()
            instance = _risk_monitor
    return instance


@router.get("/risk/status")
async def get_risk_status():
    """Get the live risk monitor's current state.

    This used to overwrite the process-wide monitor with the OMS book on
    every GET. The OMS is a different book with a different initial capital,
    and its `daily_pnl` is cumulative rather than daily, so a frontend poll
    could ratchet `peak_value` to the OMS equity and make the next engine
    cycle see a 90% drawdown and trip the kill switch. Risk state is owned by
    the trading engine/runner, which publish their own snapshots.
    """
    monitor = _get_risk_monitor()
    return monitor.get_status()


@router.post("/risk/kill-switch")
async def activate_kill_switch(req: dict):
    """Activate or deactivate the kill switch."""
    monitor = _get_risk_monitor()
    action = req.get("action", "activate")

    if action == "activate":
        monitor.activate_kill_switch(
            req.get("reason", "Manual activation"), actor="api",
        )
        status = monitor.get_status()
        return {
            "status": "activated", "kill_switch": True,
            "persisted": bool(status.get("persisted", False)),
            "persist_error": bool(status.get("persist_error", False)),
        }
    elif action == "deactivate":
        monitor.deactivate_kill_switch(
            reason=req.get("reason", "Manual deactivation"), actor="api",
        )
        status = monitor.get_status()
        return {
            "status": "deactivated", "kill_switch": False,
            "persisted": bool(status.get("persisted", False)),
            "persist_error": bool(status.get("persist_error", False)),
        }
    else:
        raise HTTPException(400, "action must be 'activate' or 'deactivate'")


@router.post("/risk/check-order")
async def check_order_risk(order: dict):
    """Pre-trade risk check on an order."""
    # NaN compares false against every limit, so a NaN order was approved
    # with an empty breach list.
    for key in ("quantity", "price", "limit_price"):
        value = order.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, f"{key} must be a number") from exc
        if not np.isfinite(number) or number <= 0:
            raise HTTPException(422, f"{key} must be a finite positive number")
    monitor = _get_risk_monitor()
    is_approved, breaches = monitor.check_pre_trade(order)
    return {
        "approved": is_approved,
        "breaches": [
            {
                "type": b.breach_type.value,
                "severity": b.severity.value,
                "message": b.message,
                "auto_action": b.auto_action,
            }
            for b in breaches
        ],
    }


# ---------------------------------------------------------------------------
# Execution Algorithms
# ---------------------------------------------------------------------------

@router.post("/execution/smart-route")
def smart_route_order(req: dict):
    """Create an execution plan, and optionally send it to the OMS.

    Set ``execute`` to true to turn every slice into a real OMS order
    (create -> submit -> fill), so positions and cash move with the parent
    order. ``price`` sets the execution price; when omitted the last close
    from the configured data feed is used.
    """
    from quant_platform.execution.algorithms import PlanExecutor, SmartRouter
    from quant_platform.execution.models import Order, OrderSide

    ticker = str(req.get("ticker", "") or "").strip()
    side = req.get("side", "buy")
    quantity = req.get("quantity", 1000)
    adv = req.get("avg_daily_volume", 1_000_000)
    urgency = req.get("urgency", "normal")

    if not ticker:
        raise HTTPException(422, "ticker is required")
    # Canonical CODE.EXCHANGE keeps OMS position identity consistent with the
    # trading runner / live portfolio (D19/D30/D41).
    ticker = normalize_symbol(ticker)
    try:
        quantity = int(quantity)
        adv = float(adv)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "quantity and avg_daily_volume must be numbers") from exc
    if quantity <= 0:
        raise HTTPException(422, "quantity must be positive")
    if adv <= 0:
        raise HTTPException(422, "avg_daily_volume must be positive")

    try:
        order_side = OrderSide(side)
    except ValueError as exc:
        raise HTTPException(422, f"side must be 'buy' or 'sell', got {side!r}") from exc

    order = Order(ticker=ticker, side=order_side, quantity=quantity)
    plan = SmartRouter.execute(order, avg_daily_volume=adv, urgency=urgency)

    response = {
        "plan_id": plan.plan_id,
        "algorithm": plan.algorithm,
        "ticker": plan.ticker,
        "side": plan.side,
        "total_quantity": plan.total_quantity,
        "num_slices": plan.num_slices,
        "start_time": plan.start_time,
        "end_time": plan.end_time,
        "slices": [
            {
                "slice_id": s.slice_id,
                "quantity": s.quantity,
                "target_time": s.target_time,
                "participation_rate": s.participation_rate,
            }
            for s in plan.slices
        ],
    }

    if req.get("execute", False):
        price = req.get("price")
        if price is None:
            # Money path: use the **raw** close, never the adjusted research
            # panel (`_load_data`'s first element).  A provider that cannot
            # supply raw prices must fail the request, not fill at hfq levels
            # (D15).
            try:
                config = load_config()
                _prices, _returns, _benchmark, metadata, *_rest = _load_data(
                    config,
                )
                raw_panel = None
                if metadata is not None:
                    raw_panel = (
                        (getattr(metadata, "attrs", {}) or {})
                        .get("raw_prices", {})
                        .get("close")
                    )
                if raw_panel is None or ticker not in raw_panel.columns:
                    raise RuntimeError(
                        "the active data provider supplied no raw price panel"
                    )
                price = float(raw_panel[ticker].dropna().iloc[-1])
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "execute=true needs a raw CNY price and none could "
                        f"be derived for {ticker}: {e}"
                    ),
                ) from e
        try:
            price = float(price)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "price must be a number") from exc
        if not np.isfinite(price) or price <= 0:
            raise HTTPException(400, "price must be a finite positive number")

        report = PlanExecutor(_get_oms()).execute_plan(plan, price=float(price))
        response["execution"] = report.to_dict()
        response["price_basis"] = "raw"

    return response


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

@router.post("/report/html")
async def generate_report(req: dict):
    """Generate a self-contained HTML backtest report."""
    from quant_platform.reporting.html_report import generate_html_report

    run_id = req.get("run_id", "")
    with _run_lock:
        result = _run_store.get(run_id)
    if not result:
        raise HTTPException(404, f"Run {run_id} not found. Run a pipeline first.")

    # `_build_chart_data` takes a returns Series, not the run record; passing
    # the record made this an AttributeError on `dict.dropna()` and the
    # download bar always returned 500. The chart was already built when the
    # run completed.
    chart = result.get("chart_data", {})
    perf = result.get("performance", {})
    risk_data = result.get("risk", {})
    factors = result.get("factors", [])
    exposure = result.get("exposure", {})
    stress = result.get("stress_tests", [])

    report_input = {
        "performance": perf,
        "risk": risk_data,
        "factors": [
            {"name": f.get("name", f.get("factor", "")),
             "mean_ic": f.get("mean_ic", 0),
             "icir": f.get("icir", 0),
             "ic_positive_ratio": f.get("ic_positive_ratio", 0),
             "std_ic": f.get("std_ic", 0)}
            for f in factors
        ],
        "chart_data": chart,
        "exposure": exposure,
        "stress_tests": stress,
    }

    output_path = f"results/report_{run_id[:8]}.html"
    # HTML generation + file write are blocking; keep them off the loop.
    path = await run_in_threadpool(
        lambda: generate_html_report(report_input, output_path=output_path),
    )
    return {"path": path, "run_id": run_id}


# ---------------------------------------------------------------------------
# Multi-Strategy Manager
# ---------------------------------------------------------------------------

_multi_strategy_mgr = None


def _get_multi_strategy_mgr():
    global _multi_strategy_mgr
    instance = _multi_strategy_mgr
    if instance is None:
        with _singleton_lock:
            if _multi_strategy_mgr is None:
                from quant_platform.strategy.multi_strategy import MultiStrategyManager
                _multi_strategy_mgr = MultiStrategyManager()
            instance = _multi_strategy_mgr
    return instance


_strategy_orchestrator = None


def _get_strategy_orchestrator():
    """Shared orchestrator that owns per-strategy books and cleanup.

    Constructing it registers the manager's removal handler, so deleting a
    strategy cancels its open orders, liquidates/transfers its positions and
    purges its ledgers instead of orphaning capital.
    """
    global _strategy_orchestrator
    instance = _strategy_orchestrator
    if instance is None:
        with _singleton_lock:
            if _strategy_orchestrator is None:
                from quant_platform.strategy.portfolio_orchestrator import PortfolioOrchestrator
                _strategy_orchestrator = PortfolioOrchestrator(_get_multi_strategy_mgr())
            instance = _strategy_orchestrator
    return instance


@router.post("/strategy/add")
async def strategy_add(req: dict):
    """Add a new strategy to the multi-strategy manager."""
    from quant_platform.strategy.multi_strategy import StrategyConfig

    mgr = _get_multi_strategy_mgr()
    try:
        allocation_pct = float(req.get("allocation_pct", 0.0))
        max_drawdown_limit = float(req.get("max_drawdown_limit", 0.15))
        n_stocks = int(req.get("n_stocks", 300))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "allocation_pct/max_drawdown_limit/n_stocks must be numbers") from exc
    # NaN passes every comparison and poisoned /strategy/list and /metrics
    # with a non-JSON-compliant value.
    if not np.isfinite(allocation_pct) or not 0 <= allocation_pct <= 1:
        raise HTTPException(422, "allocation_pct must be a finite fraction in [0, 1]")
    if not np.isfinite(max_drawdown_limit) or not 0 < max_drawdown_limit <= 1:
        raise HTTPException(422, "max_drawdown_limit must be a finite fraction in (0, 1]")
    if not 1 <= n_stocks <= 1000:
        raise HTTPException(422, "n_stocks must be between 1 and 1000")

    config = StrategyConfig(
        name=req.get("name", "Unnamed"),
        description=req.get("description", ""),
        optimizer=req.get("optimizer", "mean_variance"),
        alpha_method=req.get("alpha_method", "icir_weighted"),
        rebalance_frequency=req.get("rebalance_frequency", "monthly"),
        n_stocks=n_stocks,
        allocation_pct=allocation_pct,
        max_drawdown_limit=max_drawdown_limit,
    )
    sid = mgr.add_strategy(config)
    return {"strategy_id": sid, "name": config.name}


@router.post("/strategy/remove")
async def strategy_remove(req: dict):
    """Remove a strategy, cancelling orders and unwinding its book."""
    sid = req.get("strategy_id", "")
    try:
        cleanup = _get_strategy_orchestrator().remove_strategy(sid)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"removed": sid, "cleanup": cleanup}


@router.get("/strategy/list")
async def strategy_list():
    """List all registered strategies."""
    mgr = _get_multi_strategy_mgr()
    result = []
    for sid, config in mgr.strategies.items():
        state = mgr.states.get(sid)
        result.append({
            "strategy_id": sid,
            "name": config.name,
            "description": config.description,
            "optimizer": config.optimizer,
            "alpha_method": config.alpha_method,
            "allocation_pct": config.allocation_pct,
            "is_active": config.is_active,
            "capital_allocated": state.capital_allocated if state else 0,
            "current_value": state.current_value if state else 0,
            "total_pnl": state.total_pnl if state else 0,
            "total_return": state.total_return if state else 0,
            "sharpe_ratio": state.sharpe_ratio if state else 0,
            "max_drawdown": state.max_drawdown if state else 0,
        })
    return {"strategies": result, "total_capital": mgr.total_capital}


@router.post("/strategy/allocate")
async def strategy_allocate(req: dict):
    """Allocate capital across strategies."""
    mgr = _get_multi_strategy_mgr()
    weights = req.get("weights", {})
    if not isinstance(weights, dict) or not weights:
        raise HTTPException(422, "weights must be a non-empty mapping of strategy_id -> weight")
    unknown = sorted(set(weights) - set(mgr.strategies))
    if unknown:
        raise HTTPException(400, f"Unknown strategy_id(s): {unknown}")
    try:
        weights = {sid: float(w) for sid, w in weights.items()}
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "weights must be numeric") from exc
    if any(w < 0 for w in weights.values()):
        raise HTTPException(422, "weights must be non-negative")
    try:
        mgr.allocate_capital(weights)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"status": "ok", "weights": weights}


@router.get("/strategy/metrics")
async def strategy_metrics():
    """Get aggregate multi-strategy metrics."""
    mgr = _get_multi_strategy_mgr()
    return mgr.get_aggregate_metrics()


@router.get("/strategy/alerts")
async def strategy_alerts():
    """Get risk alerts across all strategies."""
    mgr = _get_multi_strategy_mgr()
    return {"alerts": mgr.get_risk_alerts()}


@router.post("/strategy/update-pnl")
async def strategy_update_pnl(req: dict):
    """Update a strategy's daily P&L."""
    mgr = _get_multi_strategy_mgr()
    sid = req.get("strategy_id", "")
    if sid not in mgr.states:
        raise HTTPException(404, f"Unknown strategy_id: {sid}")
    try:
        daily_return = float(req.get("daily_return", 0.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "daily_return must be a number") from exc
    if not np.isfinite(daily_return) or not -1.0 <= daily_return <= 1.0:
        raise HTTPException(422, "daily_return must be a finite fraction in [-1, 1]")
    mgr.update_strategy_pnl(sid, daily_return)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

def _returns_from_quality_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a date x asset return frame from a price panel.

    A `(date, asset)` MultiIndex panel must be unstacked first: `pct_change`
    on the flat frame differences consecutive *rows*, i.e. different stocks
    on the same date, which reported tens of thousands of fake anomalies.
    """
    if isinstance(prices.index, pd.MultiIndex):
        field = "close" if "close" in prices.columns else prices.columns[0]
        wide = prices[field].unstack("asset")
    else:
        wide = prices
    return wide.replace(0, np.nan).pct_change(fill_method=None).dropna()


@router.post("/data/quality")
def data_quality_check(req: dict):
    """Run data quality checks on price data."""
    from quant_platform.data.quality import DataQualityMonitor

    run_id = req.get("run_id", "")
    result = _run_store.get(run_id)

    monitor = DataQualityMonitor(
        max_missing_pct=req.get("max_missing_pct", 0.05),
        max_daily_move=req.get("max_daily_move", 0.22),
        max_stale_days=req.get("max_stale_days", 5),
        min_volume=req.get("min_volume", 1000),
        max_nan_rate=req.get("max_nan_rate", 0.02),
    )

    if result and "data_quality" in result:
        # Computed when the run executed, while the price panel was in memory.
        report = dict(result["data_quality"])
        report["source"] = "run"
        report["synthetic"] = False
        report["run_id"] = run_id
        return report
    if result and "prices" in result:
        prices = result["prices"]
        if isinstance(prices, pd.DataFrame):
            monitor.check_prices(prices)
            monitor.check_returns(_returns_from_quality_prices(prices))
        source = "run_prices"
    elif run_id:
        # The run record does not retain its price panel. Checking an unrelated
        # synthetic sample and presenting it as this run's quality was worse
        # than saying no.
        raise HTTPException(
            422,
            "This run does not retain its price panel, so its quality cannot "
            "be checked. Run the check without run_id for the synthetic demo.",
        )
    else:
        # Generate synthetic data for demo
        from quant_platform.data.providers.synthetic import SyntheticDataProvider
        provider = SyntheticDataProvider(n_stocks=50, start_date="2023-01-01", end_date="2024-12-31")
        prices = provider.get_prices("2023-01-01", "2024-12-31")
        if isinstance(prices, pd.DataFrame):
            monitor.check_prices(prices)
            monitor.check_returns(_returns_from_quality_prices(prices))
        source = "synthetic_demo"

    report = dict(monitor.get_report())
    # Without a provenance marker this report looked like the quality of the
    # live data feed; callers can now tell a demo sample from run data.
    report["source"] = source
    report["synthetic"] = source == "synthetic_demo"
    if run_id:
        report["run_id"] = run_id
    if isinstance(prices, pd.DataFrame) and len(prices.index) > 0:
        try:
            report["data_date_range"] = [
                str(pd.Timestamp(prices.index.min()).date()),
                str(pd.Timestamp(prices.index.max()).date()),
            ]
        except (TypeError, ValueError):
            pass
    return report


# ---------------------------------------------------------------------------
# Core Architecture (shared singletons)
# ---------------------------------------------------------------------------

from quant_platform.core.audit import AuditLog
from quant_platform.core.events import get_event_bus
from quant_platform.core.state_machine import PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor, get_risk_monitor

_core_store = Store()
_core_bus = get_event_bus()
_core_sm = PortfolioStateMachine()
_core_audit = AuditLog(_core_store, _core_bus)
_core_sm.set_transition_callback(
    lambda record: _core_audit.log_state_change(
        record.from_state, record.to_state, record.reason,
    )
)
_core_risk = get_risk_monitor()
# S21: restore the persisted level/breaches/kill switch at startup and keep
# every change durable. The key is tenant-scoped ('risk_state:default' here).
try:
    _core_risk.attach_persistence(
        _core_store, tenant_id="default", audit=_core_audit, restore=True,
    )
except Exception:
    logger.exception("Risk-state persistence setup failed")

# Subscribe EventBus → WebSocket bridge for real-time push
for _topic in ["order.filled", "order.rejected", "portfolio.snapshot",
               "market.tick", "risk.status", "risk.kill_switch", "risk.order_blocked",
               "engine.started", "engine.stopped", "system.error"]:
    _core_bus.subscribe(_topic, _on_bus_event)

# ---------------------------------------------------------------------------
# Live Trading Engine
# ---------------------------------------------------------------------------

_live_engine = None
_trading_start_lock = threading.Lock()


def _get_live_engine():
    global _live_engine
    return _live_engine


def _rollback_engine_start(engine) -> None:
    """Undo the pre-published engine slot after a failed start."""
    global _live_engine
    if _live_engine is engine:
        _live_engine = None
    try:
        engine.stop()
    except Exception:
        logger.exception("Rollback of a failed engine start: stop() failed")


@router.post("/trading/start")
def trading_start(req: dict):
    """Start the live trading engine with full core architecture.

    The whole start path is serialized by a module-level lock. Two concurrent
    calls used to both pass the running check, construct two engines on the
    same broker/store, and leave the first one orphaned (unreachable by
    /trading/stop); now the loser gets 409 before constructing anything.

    `_live_engine` is published *before* `engine.start()` -- broker connect,
    state restore and the health check take a long window -- so a concurrent
    /trading/stop can always reach the engine being started.
    """
    if not _trading_start_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A trading engine start is already in progress.",
        )
    try:
        return _start_live_engine(req)
    finally:
        _trading_start_lock.release()


def _start_live_engine(req: dict) -> dict:
    from quant_platform.trading.broker import SimulatedBroker
    from quant_platform.trading.engine import LiveTradingEngine

    global _live_engine, _core_sm, _core_audit, _core_risk

    # Refuse to start a second engine.
    #
    # `_live_engine` is only ever overwritten, so a second start orphaned the
    # first: its scheduler thread and broker kept running with no handle left
    # to stop them, both engines wrote positions/P&L/orders into the same
    # store, and the orphan's kill switch became unreachable because
    # `_core_risk` was rebound too. LiveTradingEngine.start()'s own
    # `if self._running: return` guard is per-instance and cannot help.
    existing = _live_engine
    if existing is not None and getattr(existing, "_running", False):
        raise HTTPException(
            status_code=409,
            detail=(
                "A trading engine is already running. Call POST /api/trading/stop "
                "before starting another."
            ),
        )

    broker_type = req.get("broker", "simulated")
    initial_cash = req.get("initial_cash", 1_000_000)
    n_stocks = req.get("n_stocks", 30)
    # S20: 0/negative intervals made `Event.wait(timeout=...)` return
    # immediately and hammer the broker; reject them at the edge with 422.
    interval_raw = req.get("rebalance_interval", 300)
    try:
        if isinstance(interval_raw, bool) or not isinstance(
            interval_raw, (int, float)
        ):
            raise ValueError
        if not np.isfinite(float(interval_raw)):
            raise ValueError
        interval = int(interval_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            422, "rebalance_interval must be an integer number of seconds",
        ) from exc
    if interval != interval_raw or interval < 5:
        raise HTTPException(
            422, "rebalance_interval must be an integer >= 5 seconds",
        )

    if broker_type == "simulated":
        broker = SimulatedBroker(initial_cash=initial_cash)
    else:
        from quant_platform.trading.broker import QMTBroker
        # Credentials never travel in request bodies: a body password would be
        # logged, cached and forwarded to the broker by any intermediary.
        # QMTBroker itself reads QMT_PASSWORD from the environment.
        if "password" in req:
            raise HTTPException(
                422,
                "QMT credentials must not be sent in the request body; set "
                "QMT_PASSWORD in the server environment.",
            )
        # A body-supplied `server` let any token holder point the broker
        # connection at an arbitrary internal host:port (SSRF). The QMT
        # terminal address is operator configuration, not request data.
        if "server" in req:
            raise HTTPException(
                422,
                "QMT server must be configured server-side "
                "(QMT_SERVER environment variable).",
            )
        # `QMTBroker.__init__` takes account/server/password/mode/data_server.
        # This passed `qmt_path` and `account_id`, neither of which exists, so
        # selecting the QMT broker raised TypeError and the endpoint returned
        # 500 on every attempt -- the live-trading path could not be started
        # through the API at all.
        try:
            broker = QMTBroker(
                account=req.get("account_id", ""),
                server=os.environ.get("QMT_SERVER", "localhost:58610"),
                password=os.environ.get("QMT_PASSWORD", ""),
                mode=req.get("mode", "sim"),
            )
        except Exception as e:
            # Missing xtquant / unreachable terminal is an availability
            # problem, not an internal fault.
            logger.exception("QMT broker unavailable")
            raise _internal_error(
                503, "qmt_broker_unavailable",
                "QMT broker unavailable; check the xtquant installation and "
                "terminal connection (see server logs)",
            ) from e

    # Fresh state machine per engine start, but NOT a fresh risk monitor:
    # risk state must survive an engine restart, and the Monitor dashboard's
    # kill switch must reach whichever engine is running. See ADR-0003 and
    # risk.circuit_breaker.get_risk_monitor().
    _core_sm = PortfolioStateMachine()
    _core_audit = AuditLog(_core_store, _core_bus)
    # Audit state transitions: the callback is optional and was never wired.
    _core_sm.set_transition_callback(
        lambda record: _core_audit.log_state_change(
            record.from_state, record.to_state, record.reason,
        )
    )
    _core_risk = get_risk_monitor()

    engine = LiveTradingEngine(
        broker=broker,
        store=_core_store,
        bus=_core_bus,
        state_machine=_core_sm,
        audit=_core_audit,
        risk_monitor=_core_risk,
        rebalance_interval=interval,
        n_stocks=n_stocks,
    )

    # Publish the engine before the long start() sequence (broker connect,
    # state restore, health check). The exception path rolls the slot back.
    _live_engine = engine

    try:
        # Set default universe: top stocks by market cap
        try:
            from quant_platform.trading.realtime import RealTimeMarket
            rt = _get_rt_market()
            snapshot = rt.get_market_snapshot()
            # Top N by market cap, filter out ST and new stocks
            snapshot = snapshot[~snapshot['名称'].str.contains('ST|退', na=False)]
            snapshot = snapshot[snapshot['总市值'] > 0]
            top = snapshot.nlargest(n_stocks * 2, '总市值')
            codes = top['代码'].tolist()[:n_stocks]
            engine.set_universe(codes)
        except Exception as e:
            logger.warning("Failed to set universe from market: %s, using defaults", e)
            engine.set_universe([
                '600519', '000001', '601318', '000858', '600036',
                '601166', '600276', '000333', '002415', '600900',
                '601888', '300750', '600030', '601398', '600887',
                '000568', '002304', '600585', '601012', '600809',
                '000651', '601668', '002714', '600309', '601899',
                '002475', '600048', '601601', '000725', '002352',
            ])

        engine.start()
    except HTTPException:
        _rollback_engine_start(engine)
        raise
    except Exception as e:
        logger.exception("Live trading engine failed to start")
        _rollback_engine_start(engine)
        raise _internal_error(
            503, "trading_start_failed",
            "Trading engine failed to start; see server logs",
        ) from e

    return {"status": "started", "broker": broker_type, "n_stocks": n_stocks}


@router.post("/trading/stop")
def trading_stop():
    """Stop the live trading engine."""
    # Serialize with trading_start so a start cannot publish a new engine
    # while stop is tearing the previous one down.
    with _trading_start_lock:
        engine = _live_engine
        if engine:
            engine.stop()
            return {"status": "stopped"}
    return {"status": "no_engine"}


@router.get("/trading/status")
def trading_status():
    """Get live trading engine status."""
    engine = _get_live_engine()
    if not engine:
        return {"status": "no_engine"}
    return engine.get_state()


@router.get("/trading/positions")
def trading_positions():
    """Get live trading positions."""
    engine = _get_live_engine()
    if not engine:
        return {"positions": [], "price_basis": "raw"}
    return {"positions": engine.get_positions(), "price_basis": "raw"}


@router.get("/trading/account")
def trading_account():
    """Get live trading account info."""
    engine = _get_live_engine()
    if not engine:
        return {"account": {}, "price_basis": "raw"}
    return {"account": engine.get_account(), "price_basis": "raw"}


@router.get("/trading/cycles")
def trading_cycles():
    """Get recent trading cycles."""
    engine = _get_live_engine()
    if not engine:
        return {"cycles": []}
    return {"cycles": engine.get_recent_cycles(20)}


@router.post("/trading/order")
def trading_manual_order(req: dict):
    """Place a manual order."""
    engine = _get_live_engine()
    if not engine or not getattr(engine, "_running", False):
        raise HTTPException(400, "Engine not running")
    return engine.manual_order(
        code=req.get("code", ""),
        side=req.get("side", "buy"),
        quantity=req.get("quantity", 100),
        price=req.get("price", 0),
    )


@router.post("/trading/run-once")
def trading_run_once():
    """Run a single trading cycle manually."""
    engine = _get_live_engine()
    # A stopped engine stays in the global slot so status/positions remain
    # inspectable. Checking only `engine is None` let run-once and manual
    # orders keep executing after POST /trading/stop.
    if not engine or not getattr(engine, "_running", False):
        raise HTTPException(400, "Engine not running")
    cycle = engine.run_once()
    return cycle.to_dict()


# ---------------------------------------------------------------------------
# Real-time Market Data
# ---------------------------------------------------------------------------

@router.get("/market/snapshot")
def market_snapshot():
    """Get real-time A-share market snapshot."""
    try:
        from quant_platform.trading.realtime import RealTimeMarket
        rt = _get_rt_market()
        df = rt.get_market_snapshot()
        # Return top 50 by amount for performance
        top = df.nlargest(50, '成交额')
        records = top[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率', '市盈率-动态', '总市值']].to_dict('records')
        # Report when the data was fetched, not when this request was served.
        data_timestamp = rt.snapshot_fetched_at or datetime.now().isoformat()
        return {
            "stocks": records,
            "total": len(df),
            "timestamp": data_timestamp,
            "data_timestamp": data_timestamp,
            "stale": rt.snapshot_is_stale,
        }
    except ImportError as e:
        logger.exception("Market data dependency missing")
        raise _internal_error(
            503, "market_data_dependency_missing",
            "Market data provider dependency is not installed",
        ) from e
    except Exception as e:
        logger.exception("Market data request failed")
        raise _internal_error(
            500, "market_data_failed", "Market data unavailable; see server logs",
        ) from e


@router.get("/market/gainers")
def market_gainers():
    """Get top gainers."""
    try:
        from quant_platform.trading.realtime import RealTimeMarket
        rt = _get_rt_market()
        df = rt.get_top_gainers(20)
        return {"stocks": df.to_dict('records')}
    except ImportError as e:
        logger.exception("Market data dependency missing")
        raise _internal_error(
            503, "market_data_dependency_missing",
            "Market data provider dependency is not installed",
        ) from e
    except Exception as e:
        logger.exception("Market data request failed")
        raise _internal_error(
            500, "market_data_failed", "Market data unavailable; see server logs",
        ) from e


@router.get("/market/losers")
def market_losers():
    """Get top losers."""
    try:
        from quant_platform.trading.realtime import RealTimeMarket
        rt = _get_rt_market()
        df = rt.get_top_losers(20)
        return {"stocks": df.to_dict('records')}
    except ImportError as e:
        logger.exception("Market data dependency missing")
        raise _internal_error(
            503, "market_data_dependency_missing",
            "Market data provider dependency is not installed",
        ) from e
    except Exception as e:
        logger.exception("Market data request failed")
        raise _internal_error(
            500, "market_data_failed", "Market data unavailable; see server logs",
        ) from e


@router.get("/market/sectors")
def market_sectors():
    """Get sector data."""
    try:
        from quant_platform.trading.realtime import RealTimeMarket
        rt = _get_rt_market()
        df = rt.get_sector_data()
        records = df.head(30).to_dict('records')
        return {"sectors": records}
    except ImportError as e:
        logger.exception("Market data dependency missing")
        raise _internal_error(
            503, "market_data_dependency_missing",
            "Market data provider dependency is not installed",
        ) from e
    except Exception as e:
        logger.exception("Market data request failed")
        raise _internal_error(
            500, "market_data_failed", "Market data unavailable; see server logs",
        ) from e


# ---------------------------------------------------------------------------
# Core Architecture Monitoring
# ---------------------------------------------------------------------------

@router.get("/core/events")
async def core_events(topic: str = "", limit: int = 50):
    """Get event bus history (bounded; the ring buffer holds 100k events)."""
    limit = _bounded_limit(limit, default=50, maximum=1000)
    events = await run_in_threadpool(
        _core_bus.get_history, topic=topic, limit=limit,
    )
    # The global ring carries every tenant's audit payloads. Serve those via
    # /api/core/audit (tenant-filtered) instead of leaking them here.
    events = [
        event for event in events
        if not str(event.get("topic", "")).startswith("audit.")
    ]
    return {"events": events}


@router.get("/core/events/metrics")
async def core_events_metrics():
    """Get event bus metrics (publish counts, subscriber counts)."""
    return await run_in_threadpool(_core_bus.get_metrics)


@router.get("/core/store/stats")
async def core_store_stats(
    request: Request, tenant_id: str = "", all_tenants: bool = False,
):
    """Get SQLite store statistics for one tenant (explicit ``all_tenants``
    is loopback-only)."""
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    stats = await run_in_threadpool(_core_store.get_stats, tenant_id=tenant)
    return {**stats, "tenant_id": tenant or "all", "all_tenants": not tenant}


@router.get("/core/store/orders")
async def core_store_orders(
    request: Request, status: str = "", code: str = "", limit: int = 100,
    tenant_id: str = "", all_tenants: bool = False,
):
    """Get orders from the persistent store for one tenant."""
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    return {
        "tenant_id": tenant or "all",
        "all_tenants": not tenant,
        "orders": await run_in_threadpool(
            _core_store.get_orders, status=status, code=code, limit=limit,
            tenant_id=tenant,
        ),
    }


@router.get("/core/store/trades")
async def core_store_trades(
    request: Request, code: str = "", limit: int = 100,
    tenant_id: str = "", all_tenants: bool = False,
):
    """Get trade history from the persistent store for one tenant."""
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    return {
        "tenant_id": tenant or "all",
        "all_tenants": not tenant,
        "trades": await run_in_threadpool(
            _core_store.get_trades, code=code, limit=limit, tenant_id=tenant,
        ),
    }


@router.get("/core/store/pnl")
async def core_store_pnl(
    request: Request, days: int = 30,
    tenant_id: str = "", all_tenants: bool = False,
):
    """Get P&L history from the persistent store for one tenant."""
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    return {
        "tenant_id": tenant or "all",
        "all_tenants": not tenant,
        "history": await run_in_threadpool(
            _core_store.get_pnl_history, days=days, tenant_id=tenant,
        ),
    }


@router.get("/core/store/signals")
async def core_store_signals(
    request: Request, consumed: int = -1, limit: int = 100,
    tenant_id: str = "", all_tenants: bool = False,
):
    """Get signal history from the persistent store for one tenant."""
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    return {
        "tenant_id": tenant or "all",
        "all_tenants": not tenant,
        "signals": await run_in_threadpool(
            _core_store.get_signals, consumed=consumed, limit=limit,
            tenant_id=tenant,
        ),
    }


@router.get("/core/store/sessions")
async def core_store_sessions(
    request: Request, limit: int = 20,
    tenant_id: str = "", all_tenants: bool = False,
):
    """Get trading session history for one tenant."""
    # The store re-clamps too; this makes limit=-1 / 0 / 1e9 explicit at the
    # API boundary (SQLite treats LIMIT -1 as unlimited).
    limit = _bounded_limit(limit, default=20, maximum=1000)
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    return {
        "tenant_id": tenant or "all",
        "all_tenants": not tenant,
        "sessions": await run_in_threadpool(
            _core_store.get_sessions, limit=limit, tenant_id=tenant,
        ),
    }


@router.get("/core/state")
async def core_state():
    """Get state machine current state and history."""
    return await run_in_threadpool(_core_state_payload)


def _core_state_payload() -> dict:
    return {
        "current_state": _core_sm.state_str,
        "state_duration": round(_core_sm.state_duration, 1),
        "history": _core_sm.get_history(limit=50),
    }


@router.get("/core/audit")
async def core_audit(
    request: Request, action: str = "", limit: int = 50,
    tenant_id: str = "", all_tenants: bool = False,
):
    """Get recent audit log entries for one tenant."""
    limit = _bounded_limit(limit, default=50, maximum=1000)
    tenant = _resolve_tenant(request, tenant_id, all_tenants)
    return {
        "tenant_id": tenant or "all",
        "all_tenants": not tenant,
        "events": await run_in_threadpool(
            _core_audit.get_recent, action=action, limit=limit,
            tenant_id=tenant,
        ),
    }


@router.get("/core/risk")
async def core_risk():
    """Get current risk monitor status."""
    return await run_in_threadpool(_core_risk.get_status)


@router.post("/core/risk/kill-switch")
async def core_risk_kill_switch(req: dict):
    """Activate or deactivate the kill switch."""
    activate = req.get("activate", True)
    if activate:
        _core_risk.activate_kill_switch(
            reason=req.get("reason", "Manual API activation"), actor="api",
        )
    else:
        _core_risk.deactivate_kill_switch(
            reason=req.get("reason", "Manual API deactivation"), actor="api",
        )
    status = _core_risk.get_status()
    return {
        "kill_switch_active": _core_risk.kill_switch_active,
        "risk_level": _core_risk.risk_level.value,
        "persisted": bool(status.get("persisted", False)),
        "persist_error": bool(status.get("persist_error", False)),
    }


# ────────────────────────────────────────────────────────────────
# ML Alpha Signal endpoints
# ────────────────────────────────────────────────────────────────

@router.post("/ml/train", response_model=MLPerformanceResponse)
def ml_train(req: MLTrainRequest):
    """Train an ML alpha model (bounded full-pipeline compute)."""
    return _run_bounded_compute(_ml_train_impl, req)


def _ml_train_impl(req: MLTrainRequest) -> MLPerformanceResponse:
    """Train ML model for alpha signal generation."""
    try:
        from quant_platform.alpha.ml_signal import (
            InsufficientTrainingData,
            MLSignalConfig,
            MLSignalGenerator,
        )
        config = load_config()

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        ml_config = MLSignalConfig(
            model_type=req.model_type,
            train_window=req.train_window,
            n_splits=req.n_splits,
            retrain_frequency=req.retrain_frequency,
        )
        generator = MLSignalGenerator(config=ml_config)
        perf = generator.train(processed_factors, returns)

        return MLPerformanceResponse(
            model_type=perf.model_type,
            test_ic=round(perf.test_ic, 6),
            test_icir=round(perf.test_icir, 4),
            train_ic=round(perf.train_ic, 6),
            feature_importance={k: round(v, 6) for k, v in perf.feature_importance.items()},
            n_train_samples=perf.n_train_samples,
            date=perf.date,
        )
    except InsufficientTrainingData as e:
        # The sample is too short for an honest model: a client-input problem,
        # not an internal failure.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("ML train failed")
        raise _internal_error(
            500, "ml_train_failed", "ML training failed; see server logs",
        ) from e


@router.post("/ml/predict", response_model=MLSignalResponse)
def ml_predict(req: MLTrainRequest):
    """Generate ML alpha signals (bounded full-pipeline compute)."""
    return _run_bounded_compute(_ml_predict_impl, req)


#: Upper bound on the dates x assets matrix returned by `/ml/predict`. The
#: default universe x full history is ~630k cells / ~4.7 MiB of JSON, and the
#: serialization alone blocks for ~200 ms. When the matrix exceeds this, the
#: *most recent* dates are kept and older dates are dropped (documented in the
#: endpoint docstring) rather than returning a 413 for the default request.
_MAX_ML_MATRIX_CELLS = 250_000


def _truncate_signal_matrix(signal: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent dates such that dates x assets <= the cap."""
    n_assets = max(1, len(signal.columns))
    max_dates = max(1, _MAX_ML_MATRIX_CELLS // n_assets)
    if len(signal.index) <= max_dates:
        return signal
    logger.warning(
        "Truncating /ml/predict response from %d to %d dates (%d assets) to "
        "stay under %d matrix cells",
        len(signal.index), max_dates, len(signal.columns),
        _MAX_ML_MATRIX_CELLS,
    )
    return signal.iloc[-max_dates:]


def _ml_predict_impl(req: MLTrainRequest) -> MLSignalResponse:
    """Generate ML-based alpha signals.

    The response matrix is capped at `_MAX_ML_MATRIX_CELLS` cells; when the
    signal is larger, only the most recent dates are returned.
    """
    try:
        from quant_platform.alpha.ml_signal import (
            InsufficientTrainingData,
            MLSignalConfig,
            MLSignalGenerator,
        )
        config = load_config()

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        ml_config = MLSignalConfig(
            model_type=req.model_type,
            train_window=req.train_window,
            n_splits=req.n_splits,
        )
        generator = MLSignalGenerator(config=ml_config)
        # `generate()` has no force_retrain parameter; passing it raised
        # TypeError and this endpoint returned 500 on every request.
        signal = generator.generate(processed_factors, returns)

        # Bound the matrix before serializing it.
        signal = _truncate_signal_matrix(signal)

        dates = [str(d) for d in signal.index]
        assets = list(signal.columns)
        signal_data = [
            [None if pd.isna(value) else float(value) for value in row]
            for row in signal.values.tolist()
        ]

        return MLSignalResponse(dates=dates, assets=assets, signal=signal_data)
    except InsufficientTrainingData as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("ML predict failed")
        raise _internal_error(
            500, "ml_predict_failed", "ML prediction failed; see server logs",
        ) from e


# ────────────────────────────────────────────────────────────────
# Factor IC Monitoring endpoints
# ────────────────────────────────────────────────────────────────

@router.post("/ic-monitor/compute", response_model=ICMonitorSummary)
def ic_monitor_compute(req: ICMonitorRequest):
    """Compute IC statistics (bounded full-pipeline compute)."""
    return _run_bounded_compute(_ic_monitor_compute_impl, req)


def _ic_monitor_compute_impl(req: ICMonitorRequest) -> ICMonitorSummary:
    """Compute IC statistics for all factors with decay detection."""
    try:
        from quant_platform.factors.ic_monitor import FactorICMonitor, ICMonitorConfig
        config = load_config()

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        mon_config = ICMonitorConfig(
            rolling_window=req.rolling_window,
            decay_window=req.decay_window,
            significance_threshold=req.significance_threshold,
        )
        monitor = FactorICMonitor(config=mon_config)
        all_stats = monitor.compute_all(processed_factors, returns)

        factor_stats = [
            FactorICStatsResponse(
                name=s.name,
                current_ic=round(s.current_ic, 6),
                rolling_icir=round(s.rolling_icir, 4),
                ic_trend=round(s.ic_trend, 8),
                ic_decay_rate=round(s.ic_decay_rate, 8),
                half_life_days=s.half_life_days,
                ic_positive_ratio=round(s.ic_positive_ratio, 4),
                alert_level=s.alert_level,
            )
            for s in sorted(all_stats.values(), key=lambda x: abs(x.rolling_icir), reverse=True)
        ]

        alerts = monitor.get_alerts()
        weights = monitor.get_adaptive_weights(processed_factors, returns)

        return ICMonitorSummary(factors=factor_stats, alerts=alerts, adaptive_weights=weights)
    except Exception as e:
        logger.exception("IC monitor compute failed")
        raise _internal_error(
            500, "ic_monitor_compute_failed",
            "IC monitor computation failed; see server logs",
        ) from e


@router.get("/ic-monitor/alerts")
def ic_monitor_alerts():
    """Get current IC decay alerts (bounded full-pipeline compute)."""
    return _run_bounded_compute(_ic_monitor_alerts_impl)


def _ic_monitor_alerts_impl():
    """Get current IC decay alerts."""
    try:
        from quant_platform.factors.ic_monitor import FactorICMonitor
        config = load_config()

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        monitor = FactorICMonitor()
        monitor.compute_all(processed_factors, returns)
        return {"alerts": monitor.get_alerts()}
    except Exception as e:
        logger.exception("IC monitor alerts failed")
        raise _internal_error(
            500, "ic_monitor_alerts_failed",
            "IC monitor alerts unavailable; see server logs",
        ) from e


# ────────────────────────────────────────────────────────────────
# Barra Risk Model endpoints
# ────────────────────────────────────────────────────────────────

def _resolve_weights_with_source(
    requested: dict[str, float] | None,
    prices: pd.DataFrame,
    raw_prices: pd.DataFrame | None = None,
) -> tuple[pd.Series, str]:
    """Pick the portfolio to decompose and say where the weights came from.

    Explicit weights win; otherwise the current OMS positions are marked to
    the latest **raw** close (an adjusted research panel would distort the
    weights by the corporate-action factor, D15); otherwise the universe is
    equal-weighted. Equal weight over the whole universe has zero net
    exposure to every cross-sectionally standardized factor by construction,
    so that fallback is only a safe default, not a meaningful risk
    decomposition.
    """
    if requested:
        raw = pd.Series(requested, dtype=float)
        source = "request"
    else:
        marks: dict[str, float] = {}
        mark_panel = raw_prices if raw_prices is not None else prices
        for ticker, position in getattr(_get_oms(), "positions", {}).items():
            key = ticker
            if key not in mark_panel.columns:
                key = normalize_symbol(ticker)
            if key not in mark_panel.columns:
                continue
            series = mark_panel[key].dropna()
            if series.empty:
                continue
            marks[key] = float(position.quantity) * float(series.iloc[-1])
        if marks:
            raw = pd.Series(marks, dtype=float)
            source = "oms_positions"
        else:
            raw = pd.Series(1.0, index=prices.columns)
            source = "equal_weight"

    weights = raw.reindex(prices.columns).fillna(0.0)
    total = float(weights.sum())
    if total <= 0:
        raise HTTPException(status_code=400, detail="Portfolio weights must sum to a positive value")
    return weights / total, source


@router.post("/barra/decompose", response_model=BarraRiskResponse)
def barra_decompose(req: BarraDecomposeRequest):
    """Decompose portfolio risk using Barra 10-factor model (bounded compute)."""
    return _run_bounded_compute(_barra_decompose_impl, req)


def _barra_decompose_impl(req: BarraDecomposeRequest) -> BarraRiskResponse:
    """Decompose portfolio risk using Barra 10-factor model."""
    try:
        config = load_config()

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        raw_prices = (
            (getattr(metadata, "attrs", {}) or {})
            .get("raw_prices", {})
            .get("close")
            if metadata is not None else None
        )

        # Shared with POST /api/risk/decompose -- one model fit, one set of
        # math, so the two endpoints cannot drift apart.
        _model, result, _weights, _source = _barra_decomposition(
            prices,
            returns,
            benchmark,
            financials,
            processed_factors,
            weights_requested=req.weights,
            date=req.date,
            half_life=req.half_life,
            shrinkage_target=req.shrinkage_target,
            raw_prices=raw_prices,
        )

        return BarraRiskResponse(
            total_risk=round(result.total_risk, 6),
            factor_risk=round(result.factor_risk, 6),
            specific_risk=round(result.specific_risk, 6),
            r_squared=round(result.r_squared, 6),
            factor_contributions={k: round(v, 6) for k, v in result.factor_contributions.items()},
            factor_exposures={k: round(v, 6) for k, v in result.factor_exposures.items()},
        )
    except Exception as e:
        logger.exception("Barra decompose failed")
        raise _internal_error(
            500, "barra_decompose_failed",
            "Barra risk decomposition failed; see server logs",
        ) from e


@router.post("/barra/covariance", response_model=BarraCovarianceResponse)
def barra_covariance(req: BarraDecomposeRequest):
    """Get Barra factor covariance matrix (bounded full-pipeline compute)."""
    return _run_bounded_compute(_barra_covariance_impl, req)


def _barra_covariance_impl(req: BarraDecomposeRequest) -> BarraCovarianceResponse:
    """Get Barra factor covariance matrix."""
    try:
        from quant_platform.risk.barra import BarraModel
        config = load_config()

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        model = BarraModel(
            half_life=req.half_life,
            shrinkage_target=req.shrinkage_target,
        )
        factor_exposures = model.build_exposures(
            prices,
            returns,
            benchmark=benchmark,
            financials=financials,
            platform_factors=processed_factors,
        )
        model.fit(factor_exposures, returns)

        cov_df = model.get_factor_covariance_df()
        return BarraCovarianceResponse(
            factors=list(cov_df.columns),
            covariance=cov_df.values.tolist(),
        )
    except Exception as e:
        logger.exception("Barra covariance failed")
        raise _internal_error(
            500, "barra_covariance_failed",
            "Barra covariance computation failed; see server logs",
        ) from e


# ────────────────────────────────────────────────────────────────
# Parallel Backtest endpoints
# ────────────────────────────────────────────────────────────────

@router.post("/parallel/sweep", response_model=ParallelSweepResponse)
def parallel_sweep(req: ParallelSweepRequest):
    """Run a parallel parameter sweep (bounded compute)."""
    return _run_bounded_compute(_parallel_sweep_impl, req)


def _parallel_sweep_impl(req: ParallelSweepRequest) -> ParallelSweepResponse:
    """Run parameter sweep in parallel using multiple processes."""
    try:
        from quant_platform.backtest.distributed import ParallelBacktester

        bt = ParallelBacktester(max_workers=req.max_workers)
        sweep = bt.run_sweep(
            param_grid=req.param_grid,
            base_config_overrides=req.base_overrides,
            metric=req.metric,
        )

        return ParallelSweepResponse(
            results=sweep.summary(),
            best_params=sweep.best_params,
            best_metric=sweep.best_metric,
            total_duration=round(sweep.total_duration, 2),
            n_success=sweep.n_success,
            n_failed=sweep.n_failed,
        )
    except ValueError as e:
        # Invalid parameter keys/values are client errors, not server faults.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Parallel sweep failed")
        raise _internal_error(
            500, "parallel_sweep_failed",
            "Parallel sweep failed; see server logs",
        ) from e


# ────────────────────────────────────────────────────────────────
# Prometheus Metrics endpoint
# ────────────────────────────────────────────────────────────────

@router.get("/metrics")
async def prometheus_metrics():
    """Export system metrics in Prometheus text format."""
    from fastapi.responses import PlainTextResponse

    from quant_platform.utils.metrics import get_metrics
    return PlainTextResponse(
        content=get_metrics().export_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/json")
async def prometheus_metrics_json():
    """Export system metrics as JSON."""
    from quant_platform.utils.metrics import get_metrics
    return get_metrics().get_snapshot()


# ──────────────────────────────────────────────
# Store statistics
# ──────────────────────────────────────────────

_stats_cache_lock = threading.Lock()
_stats_cache: dict[str, Any] = {"key": None, "at": 0.0, "payload": None}
_STATS_CACHE_TTL = 5.0


def _store_backend(store) -> str:
    """The store's real backend (the old endpoint always claimed sqlite)."""
    if "postgres" in type(store).__name__.lower():
        return "postgresql"
    return "sqlite"


@router.get("/postgres/stats", response_model=PostgresStoreStatsResponse)
async def postgres_stats():
    """Get store statistics from the process-wide store.

    This used to construct a brand-new `Store()` per poll on the event loop:
    every request re-ran schema creation/migrations and opened a connection,
    and the response always said "sqlite" because it never looked at the
    configured backend. The process store is reused, `get_stats` runs in a
    worker thread, and the result is cached briefly (an inventory poll does
    not need to COUNT(*) every table on every request).
    """
    store = _core_store
    cache_key = id(store)
    now = time.monotonic()
    with _stats_cache_lock:
        cached = _stats_cache["payload"]
        if (
            cached is not None
            and _stats_cache["key"] == cache_key
            and now - _stats_cache["at"] < _STATS_CACHE_TTL
        ):
            return cached

    stats = await run_in_threadpool(store.get_stats)
    payload = PostgresStoreStatsResponse(
        backend=stats.get("backend") or _store_backend(store),
        orders=stats.get("orders", 0),
        positions=stats.get("positions", 0),
        trades=stats.get("trades", 0),
        pnl_snapshots=stats.get("pnl_snapshots", 0),
        signals=stats.get("signals", 0),
    )
    with _stats_cache_lock:
        _stats_cache.update(
            key=cache_key, at=time.monotonic(), payload=payload,
        )
    return payload


# ──────────────────────────────────────────────
# WebSocket Real-time Quotes
# ──────────────────────────────────────────────

# Global simulated provider instance for demo
_ws_provider = None

#: How long the quote endpoints wait for a just-started provider to report a
#: connection. Waiting uses `asyncio.sleep`; the old code executed a blocking
#: `time.sleep(0.5)` inside an `async def`, freezing the whole event loop for
#: half a second per request.
_QUOTE_READY_TIMEOUT = 2.0
_QUOTE_READY_POLL = 0.05


def _get_ws_provider():
    global _ws_provider
    instance = _ws_provider
    if instance is None:
        with _singleton_lock:
            if _ws_provider is None:
                from quant_platform.data.providers.websocket_provider import SimulatedWebSocketProvider
                _ws_provider = SimulatedWebSocketProvider(
                    codes=["600519", "000001", "300750", "601318", "000858"],
                    update_interval=1.0,
                )
            instance = _ws_provider
    return instance


async def _await_ws_provider_ready(provider) -> None:
    """Wait (without blocking the loop) for a quote provider to connect."""
    if provider.is_connected:
        return
    try:
        provider.start()
    except Exception as e:
        logger.exception("Quote provider failed to start")
        raise _internal_error(
            503, "quote_provider_start_failed",
            "Quote provider unavailable; see server logs",
        ) from e
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _QUOTE_READY_TIMEOUT
    while not provider.is_connected:
        if loop.time() >= deadline:
            raise HTTPException(
                status_code=503,
                detail="Quote provider is still starting; retry shortly",
                headers={"Retry-After": "1"},
            )
        await asyncio.sleep(_QUOTE_READY_POLL)


@router.get("/ws-quotes/start", include_in_schema=False)
def ws_quotes_start_get():
    """State-changing endpoint: GET is rejected so a link/img cannot start it.

    Without this stub the generic ``/ws-quotes/{code}`` route would swallow
    ``GET /ws-quotes/start`` and start the provider while reporting a 404.
    """
    raise HTTPException(
        status_code=405,
        detail="Use POST /api/ws-quotes/start",
        headers={"Allow": "POST"},
    )


@router.post("/ws-quotes/start")
def ws_quotes_start():
    """Start WebSocket quote provider."""
    provider = _get_ws_provider()
    provider.start()
    return {"status": "started", **provider.stats}


@router.get("/ws-quotes/stop", include_in_schema=False)
def ws_quotes_stop_get():
    """State-changing endpoint: GET is rejected (see ``ws_quotes_start_get``)."""
    raise HTTPException(
        status_code=405,
        detail="Use POST /api/ws-quotes/stop",
        headers={"Allow": "POST"},
    )


@router.post("/ws-quotes/stop")
def ws_quotes_stop():
    """Stop WebSocket quote provider."""
    provider = _get_ws_provider()
    provider.stop()
    return {"status": "stopped", **provider.stats}


@router.get("/ws-quotes/stats", response_model=WebSocketStatsResponse)
async def ws_quotes_stats():
    """Get WebSocket provider statistics."""
    provider = _get_ws_provider()
    stats = provider.stats
    return WebSocketStatsResponse(**stats)


@router.get("/ws-quotes/{code}", response_model=RealtimeQuoteResponse)
async def ws_quote_by_code(code: str):
    """Get real-time quote for a single stock."""
    provider = _get_ws_provider()
    await _await_ws_provider_ready(provider)
    quote = provider.get_quote(code)
    if not quote:
        raise HTTPException(404, f"No quote for {code}")
    return RealtimeQuoteResponse(**quote.to_dict())


@router.get("/ws-quotes", response_model=list[RealtimeQuoteResponse])
async def ws_all_quotes():
    """Get all cached real-time quotes."""
    provider = _get_ws_provider()
    await _await_ws_provider_ready(provider)
    quotes = provider.get_all_quotes()
    return [RealtimeQuoteResponse(**q.to_dict()) for q in quotes.values()]


# ──────────────────────────────────────────────
# Level 2 Order Book
# ──────────────────────────────────────────────

_l2_provider = None


def _get_l2_provider():
    global _l2_provider
    instance = _l2_provider
    if instance is None:
        with _singleton_lock:
            if _l2_provider is None:
                from quant_platform.data.providers.level2_provider import Level2DataProvider
                _l2_provider = Level2DataProvider(
                    codes=["600519", "000001", "300750", "601318", "000858"],
                )
            instance = _l2_provider
    return instance


@router.post("/l2/start")
def l2_start():
    """Start Level 2 data provider (POST-only; a GET changed state)."""
    provider = _get_l2_provider()
    provider.start()
    return {"status": "started", **provider.stats}


@router.post("/l2/stop")
def l2_stop():
    """Stop Level 2 data provider (POST-only; a GET changed state)."""
    provider = _get_l2_provider()
    provider.stop()
    return {"status": "stopped"}


@router.get("/l2/stats", response_model=Level2StatsResponse)
async def l2_stats():
    """Get Level 2 provider statistics."""
    provider = _get_l2_provider()
    return Level2StatsResponse(**provider.stats)


@router.get("/l2/book/{code}", response_model=OrderBookResponse)
def l2_order_book(code: str):
    """Get order book for a stock."""
    provider = _get_l2_provider()
    if not provider.is_running:
        raise HTTPException(
            503, "Level 2 provider is not running; POST /api/l2/start first",
        )
    book = provider.get_order_book(code)
    if not book:
        raise HTTPException(404, f"No order book for {code}")
    return OrderBookResponse(**book.to_dict())


@router.get("/l2/ticks/{code}", response_model=list[TickDataResponse])
def l2_ticks(
    code: str,
    limit: int = Query(100, ge=1, le=5000),
):
    """Get recent tick data for a stock (limit 1..5000).

    A raw limit was passed to `ticks[-limit:]`: 0 returned the entire
    10,000-tick buffer and a negative limit dropped the oldest rows instead
    of rejecting the request.
    """
    provider = _get_l2_provider()
    if not provider.is_running:
        raise HTTPException(
            503, "Level 2 provider is not running; POST /api/l2/start first",
        )
    ticks = provider.get_ticks(code, limit=limit)
    return [TickDataResponse(**t.to_dict()) for t in ticks]


@router.get("/l2/vwap/{code}", response_model=VWAPResponse)
def l2_vwap(code: str, n_ticks: int = Query(100, ge=1, le=5000)):
    """Compute VWAP from recent ticks."""
    provider = _get_l2_provider()
    if not provider.is_running:
        raise HTTPException(
            503, "Level 2 provider is not running; POST /api/l2/start first",
        )
    ticks = provider.get_ticks(code, limit=n_ticks)
    if not ticks or sum(t.volume for t in ticks) <= 0:
        # Returning vwap=0.0 with n_ticks=100 presented "no data" as a price.
        raise HTTPException(404, f"No tick data for {code}")
    vwap = provider.compute_vwap(code, n_ticks=n_ticks)
    return VWAPResponse(
        code=code, vwap=round(vwap, 4), n_ticks=len(ticks),
        simulated=True, source="simulated",
    )


@router.get("/l2/flow/{code}", response_model=TradeFlowResponse)
def l2_trade_flow(code: str, n_ticks: int = Query(100, ge=1, le=5000)):
    """Compute buy/sell trade flow imbalance."""
    provider = _get_l2_provider()
    if not provider.is_running:
        raise HTTPException(
            503, "Level 2 provider is not running; POST /api/l2/start first",
        )
    flow = provider.compute_trade_flow(code, n_ticks=n_ticks)
    if flow["buy_volume"] + flow["sell_volume"] <= 0:
        raise HTTPException(404, f"No directional tick data for {code}")
    return TradeFlowResponse(code=code, **flow)


# ──────────────────────────────────────────────
# Real-time Fundamentals
# ──────────────────────────────────────────────

_fundamental_provider = None


def _get_fundamental_provider():
    global _fundamental_provider
    instance = _fundamental_provider
    if instance is None:
        with _singleton_lock:
            if _fundamental_provider is None:
                from quant_platform.data.providers.fundamental_realtime import FundamentalDataProvider
                _fundamental_provider = FundamentalDataProvider()
            instance = _fundamental_provider
    return instance


#: The provider paces outbound requests at 0.5 s per uncached code, so one
#: 100-code call can hold a worker thread for ~50 s. Only one such fetch runs
#: at a time; the rest get 429 rather than piling up on AnyIO's thread pool.
_fundamental_fetch_slots = threading.BoundedSemaphore(1)


def _acquire_fundamental_slot() -> None:
    if not _fundamental_fetch_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="A fundamentals fetch is already in progress; retry shortly",
            headers={"Retry-After": "15"},
        )


@router.get("/fundamentals/stats")
async def fundamental_stats():
    """Get fundamental provider statistics."""
    provider = _get_fundamental_provider()
    return provider.stats

@router.get("/fundamentals/{code}", response_model=FundamentalMetricsResponse)
def get_fundamentals(code: str):
    """Get real-time fundamental metrics for a stock."""
    provider = _get_fundamental_provider()
    metrics = provider.get_fundamentals(code)
    return FundamentalMetricsResponse(**metrics.to_dict())


@router.post("/fundamentals/bulk", response_model=list[FundamentalMetricsResponse])
def get_fundamentals_bulk(codes: list[str]):
    """Get fundamentals for multiple stocks (one paced fetch at a time)."""
    # Each uncached code performs an outbound request with a 0.5s pause; an
    # unbounded list is both a resource and a quota-abuse vector.
    if len(codes) > 100:
        raise HTTPException(422, "at most 100 codes per request")
    _acquire_fundamental_slot()
    try:
        provider = _get_fundamental_provider()
        metrics = provider.get_bulk(codes)
    finally:
        _fundamental_fetch_slots.release()
    return [FundamentalMetricsResponse(**m.to_dict()) for m in metrics.values()]


@router.post("/fundamentals/screen", response_model=FundamentalScreenResponse)
def screen_fundamentals(req: FundamentalScreenRequest):
    """Screen stocks by fundamental criteria (one paced fetch at a time)."""
    from quant_platform.data.providers.fundamental_realtime import FundamentalScreener
    _acquire_fundamental_slot()
    try:
        provider = _get_fundamental_provider()
        screener = FundamentalScreener(provider)
        passed = screener.screen(
            req.codes,
            pe_min=req.pe_min, pe_max=req.pe_max,
            pb_min=req.pb_min, pb_max=req.pb_max,
            roe_min=req.roe_min, roe_max=req.roe_max,
            market_cap_min=req.market_cap_min,
            dividend_yield_min=req.dividend_yield_min,
            debt_ratio_max=req.debt_ratio_max,
        )
    finally:
        _fundamental_fetch_slots.release()
    return FundamentalScreenResponse(
        total=len(req.codes), passed=len(passed), codes=passed,
    )


@router.post("/fundamentals/rank", response_model=FundamentalRankResponse)
def rank_fundamentals(req: FundamentalRankRequest):
    """Rank stocks by a fundamental metric (one paced fetch at a time)."""
    from quant_platform.data.providers.fundamental_realtime import FundamentalScreener
    _acquire_fundamental_slot()
    try:
        provider = _get_fundamental_provider()
        screener = FundamentalScreener(provider)
        ranked = screener.rank_by(
            req.codes, metric=req.metric,
            ascending=req.ascending, top_n=req.top_n,
        )
    finally:
        _fundamental_fetch_slots.release()
    return FundamentalRankResponse(
        metric=req.metric,
        ranked=[{"code": code, "value": round(val, 4)} for code, val in ranked],
    )


# ---------------------------------------------------------------------------
# Application lifecycle helpers (called from app.py's lifespan)
# ---------------------------------------------------------------------------

def initialize_singletons() -> None:
    """Construct the critical lazy singletons at startup, under the lock.

    First-use races used to be possible from both the loop and AnyIO worker
    threads; creating the OMS and risk monitor up front means request handlers
    only ever read an already-initialized object.
    """
    _get_oms()
    _get_risk_monitor()


async def _stop_with_timeout(label: str, fn, timeout: float) -> None:
    try:
        await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout)
    except TimeoutError:
        logger.error("%s did not finish within %.0fs", label, timeout)
    except Exception:
        logger.exception("%s failed", label)


async def shutdown_api_resources() -> None:
    """Stop the live engine, providers and WebSocket clients (bounded).

    Python cannot kill a thread: if `engine.stop()` (or a provider's stop)
    hangs past its timeout, the underlying thread keeps running until the
    process exits. The bounded wait only guarantees shutdown does not block
    the server indefinitely.
    """
    global _live_engine, _ws_provider, _l2_provider, _fundamental_provider

    engine = _live_engine
    if engine is not None:
        await _stop_with_timeout("Live trading engine stop", engine.stop, 15.0)
        _live_engine = None

    for name, provider in (
        ("WebSocket quote provider", _ws_provider),
        ("Level 2 provider", _l2_provider),
        ("Fundamental provider", _fundamental_provider),
    ):
        stop = getattr(provider, "stop", None) if provider is not None else None
        if stop is not None:
            await _stop_with_timeout(f"{name} stop", stop, 5.0)
    _ws_provider = None
    _l2_provider = None
    _fundamental_provider = None

    await _close_all_ws_clients()


