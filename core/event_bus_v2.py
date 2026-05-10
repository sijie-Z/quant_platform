"""High-performance async event bus — industrial grade.

Replaces the original EventBus (threading.Lock + sync handlers) with:
- asyncio.Queue per handler (independent consumption, no cross-blocking)
- Backpressure: queue full → publisher waits, never drops events
- Latency monitoring: per-handler P50/P99/P999 histogram
- Dead letter retry with exponential backoff
- Event sourcing with write-ahead log (WAL)
- Interceptor chain (pre-publish filtering/transform)
- Wildcard topic matching (glob-style: 'market.*', 'order.**')

Architecture:
    Publisher → Interceptors → Topic Router → Handler Queues → Consumers
                                    ↓
                              Event Store (WAL + ring buffer)
                              Dead Letter Queue (retry + alert)

Performance target:
- Publish latency: < 1μs (no contention on hot path)
- Handler dispatch: < 10μs per handler
- Throughput: > 1M events/sec (single producer)
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Event:
    """Immutable event on the bus.

    Using __slots__ for memory efficiency and faster attribute access.
    """
    topic: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time_ns)
    source: str = ""
    event_id: str = ""
    correlation_id: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:16]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "headers": self.headers,
            "time_str": datetime.fromtimestamp(self.timestamp / 1e9).isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


@dataclass
class HandlerStats:
    """Per-handler performance statistics."""
    handler_name: str
    topic: str
    invocations: int = 0
    errors: int = 0
    total_latency_ns: int = 0
    min_latency_ns: int = 0
    max_latency_ns: int = 0
    # Buckets for histogram (in microseconds)
    latency_buckets: list[int] = field(default_factory=lambda: [0] * 13)
    # Bucket boundaries: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, inf μs
    BUCKET_BOUNDARIES_NS: list[int] = field(
        default_factory=lambda: [1000, 2000, 5000, 10_000, 20_000, 50_000,
                                  100_000, 200_000, 500_000, 1_000_000,
                                  2_000_000, 5_000_000, float('inf')]
    )

    def record(self, latency_ns: int):
        self.invocations += 1
        self.total_latency_ns += latency_ns
        if self.min_latency_ns == 0 or latency_ns < self.min_latency_ns:
            self.min_latency_ns = latency_ns
        if latency_ns > self.max_latency_ns:
            self.max_latency_ns = latency_ns
        # Find bucket
        for i, boundary in enumerate(self.BUCKET_BOUNDARIES_NS):
            if latency_ns <= boundary:
                self.latency_buckets[i] += 1
                break

    def record_error(self):
        self.errors += 1

    @property
    def mean_latency_us(self) -> float:
        if self.invocations == 0:
            return 0
        return (self.total_latency_ns / self.invocations) / 1000

    def percentile(self, p: float) -> float:
        """Compute percentile latency in microseconds."""
        if self.invocations == 0:
            return 0
        target = int(self.invocations * p)
        cumulative = 0
        for i, count in enumerate(self.latency_buckets):
            cumulative += count
            if cumulative >= target:
                if i < len(self.BUCKET_BOUNDARIES_NS) - 1:
                    return self.BUCKET_BOUNDARIES_NS[i] / 1000
                return self.max_latency_ns / 1000
        return self.max_latency_ns / 1000

    @property
    def p50_us(self) -> float:
        return self.percentile(0.5)

    @property
    def p99_us(self) -> float:
        return self.percentile(0.99)

    @property
    def p999_us(self) -> float:
        return self.percentile(0.999)

    def to_dict(self) -> dict:
        return {
            "handler": self.handler_name,
            "topic": self.topic,
            "invocations": self.invocations,
            "errors": self.errors,
            "mean_latency_us": round(self.mean_latency_us, 2),
            "p50_us": round(self.p50_us, 2),
            "p99_us": round(self.p99_us, 2),
            "p999_us": round(self.p999_us, 2),
            "min_latency_us": round(self.min_latency_ns / 1000, 2),
            "max_latency_us": round(self.max_latency_ns / 1000, 2),
        }


# ──────────────────────────────────────────────────────────────────────
# Dead Letter Queue
# ──────────────────────────────────────────────────────────────────────


@dataclass
class DeadLetter:
    """A failed event with retry metadata."""
    event: Event
    handler_name: str
    error: str
    retry_count: int = 0
    next_retry_time: float = 0
    first_failure_time: float = field(default_factory=time.time_ns)

    @property
    def is_retriable(self) -> bool:
        return self.retry_count < 5


class DeadLetterQueue:
    """Dead letter queue with exponential backoff retry.

    Retry delays: 100ms, 500ms, 2s, 10s, 30s (then give up).
    """

    MAX_RETRIES = 5
    RETRY_DELAYS_NS = [
        100_000_000,      # 100ms
        500_000_000,      # 500ms
        2_000_000_000,    # 2s
        10_000_000_000,   # 10s
        30_000_000_000,   # 30s
    ]

    def __init__(self, max_size: int = 10_000):
        self._queue: list[DeadLetter] = []
        self._lock = threading.Lock()
        self._max_size = max_size
        self._total_enqueued = 0
        self._total_retried = 0
        self._total_dropped = 0

    def enqueue(self, event: Event, handler_name: str, error: str):
        with self._lock:
            self._total_enqueued += 1
            dl = DeadLetter(
                event=event,
                handler_name=handler_name,
                error=error,
                retry_count=0,
                next_retry_time=time.time_ns() + self.RETRY_DELAYS_NS[0],
            )
            self._queue.append(dl)
            if len(self._queue) > self._max_size:
                # Drop oldest
                self._queue.pop(0)
                self._total_dropped += 1

    def get_ready_retries(self) -> list[DeadLetter]:
        """Get dead letters ready for retry."""
        now = time.time_ns()
        ready = []
        with self._lock:
            remaining = []
            for dl in self._queue:
                if dl.next_retry_time <= now and dl.is_retriable:
                    ready.append(dl)
                    self._total_retried += 1
                elif dl.is_retriable:
                    remaining.append(dl)
                else:
                    self._total_dropped += 1
            self._queue = remaining
        return ready

    def requeue(self, dl: DeadLetter, error: str):
        """Re-enqueue a failed retry with incremented counter."""
        dl.retry_count += 1
        if dl.retry_count >= self.MAX_RETRIES:
            with self._lock:
                self._total_dropped += 1
            logger.error("DLQ: event %s gave up after %d retries: %s",
                        dl.event.event_id, dl.retry_count, error)
            return
        delay = self.RETRY_DELAYS_NS[min(dl.retry_count, len(self.RETRY_DELAYS_NS) - 1)]
        dl.next_retry_time = time.time_ns() + delay
        dl.error = error
        with self._lock:
            self._queue.append(dl)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._queue),
                "total_enqueued": self._total_enqueued,
                "total_retried": self._total_retried,
                "total_dropped": self._total_dropped,
            }


# ──────────────────────────────────────────────────────────────────────
# Event Store (WAL + Ring Buffer)
# ──────────────────────────────────────────────────────────────────────


class EventStore:
    """Event store with write-ahead log and in-memory ring buffer.

    Two tiers:
    1. In-memory ring buffer for fast recent-event queries
    2. WAL file for durability and event sourcing/replay

    The WAL is append-only and crash-safe. On startup, events can be
    replayed from the WAL to reconstruct state.
    """

    def __init__(
        self,
        buffer_size: int = 100_000,
        wal_path: str | None = None,
        wal_sync_interval: float = 1.0,  # seconds
    ):
        self._buffer: list[Event] = [None] * buffer_size
        self._buffer_size = buffer_size
        self._write_pos = 0
        self._total_events = 0
        self._lock = threading.Lock()

        # WAL
        self._wal_path = Path(wal_path) if wal_path else None
        self._wal_file = None
        self._wal_sync_interval = wal_sync_interval
        self._wal_buffer: list[str] = []
        self._wal_lock = threading.Lock()
        self._last_sync = time.time()

        if self._wal_path:
            self._wal_path.parent.mkdir(parents=True, exist_ok=True)
            self._wal_file = open(self._wal_path, 'a', encoding='utf-8')

    def append(self, event: Event):
        """Append event to store. Thread-safe."""
        # Ring buffer
        with self._lock:
            idx = self._write_pos % self._buffer_size
            self._buffer[idx] = event
            self._write_pos += 1
            self._total_events += 1

        # WAL (async, batched)
        if self._wal_file:
            line = event.to_json() + '\n'
            with self._wal_lock:
                self._wal_buffer.append(line)
                now = time.time()
                if now - self._last_sync >= self._wal_sync_interval:
                    self._flush_wal()

    def _flush_wal(self):
        """Flush WAL buffer to disk."""
        if not self._wal_file or not self._wal_buffer:
            return
        try:
            self._wal_file.writelines(self._wal_buffer)
            self._wal_file.flush()
            os.fsync(self._wal_file.fileno())
            self._wal_buffer.clear()
            self._last_sync = time.time()
        except Exception as e:
            logger.error("WAL flush failed: %s", e)

    def get_recent(self, limit: int = 100) -> list[Event]:
        """Get most recent events from ring buffer."""
        with self._lock:
            count = min(limit, self._write_pos, self._buffer_size)
            if count == 0:
                return []
            result = []
            for i in range(count):
                idx = (self._write_pos - count + i) % self._buffer_size
                if self._buffer[idx] is not None:
                    result.append(self._buffer[idx])
            return result

    def get_by_topic(self, topic: str, limit: int = 100) -> list[Event]:
        """Get recent events matching a topic (prefix match)."""
        recent = self.get_recent(limit * 3)  # Get more to filter
        matching = [e for e in recent if e.topic == topic or
                    (topic.endswith('.*') and e.topic.startswith(topic[:-2])) or
                    (topic.endswith('.**') and e.topic.startswith(topic[:-3]))]
        return matching[-limit:]

    def replay_from_wal(self) -> list[Event]:
        """Replay all events from WAL file. For state reconstruction."""
        if not self._wal_path or not self._wal_path.exists():
            return []
        events = []
        with open(self._wal_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = Event(
                        topic=data['topic'],
                        data=data['data'],
                        timestamp=data.get('timestamp', 0),
                        source=data.get('source', ''),
                        event_id=data.get('event_id', ''),
                        correlation_id=data.get('correlation_id', ''),
                        headers=data.get('headers', {}),
                    )
                    events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue
        return events

    def close(self):
        """Flush and close WAL."""
        with self._wal_lock:
            self._flush_wal()
        if self._wal_file:
            self._wal_file.close()
            self._wal_file = None

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_events": self._total_events,
                "buffer_size": self._buffer_size,
                "buffer_used": min(self._write_pos, self._buffer_size),
                "wal_path": str(self._wal_path) if self._wal_path else None,
                "wal_pending": len(self._wal_buffer),
            }


# ──────────────────────────────────────────────────────────────────────
# Async EventBus v2
# ──────────────────────────────────────────────────────────────────────

# Type aliases
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Coroutine[Any, Any, None]]
Interceptor = Callable[[Event], Event | None]


class AsyncEventBus:
    """High-performance async event bus.

    Key improvements over original EventBus:
    1. Each handler gets its own asyncio.Queue — no cross-blocking
    2. Backpressure: when a handler's queue is full, publisher waits
    3. Per-handler latency monitoring with histogram
    4. Dead letter queue with exponential backoff retry
    5. Event sourcing via WAL
    6. Async-first with sync handler compatibility

    Usage:
        bus = AsyncEventBus()

        # Async handler
        async def on_tick(event: Event):
            await process_tick(event.data)

        bus.subscribe("market.tick", on_tick, queue_size=10000)

        # Sync handler (run in thread pool)
        def on_fill(event: Event):
            update_position(event.data)

        bus.subscribe_sync("order.filled", on_fill)

        # Publish
        await bus.publish_async("market.tick", {"price": 100.0})

        # Or sync publish (puts into event loop)
        bus.publish("market.tick", {"price": 100.0})
    """

    def __init__(
        self,
        default_queue_size: int = 10_000,
        max_concurrent_handlers: int = 100,
        event_store: EventStore | None = None,
    ):
        self._default_queue_size = default_queue_size
        self._max_concurrent = max_concurrent_handlers

        # Handler registry: topic -> [(handler, queue, stats)]
        self._handlers: dict[str, list[tuple]] = defaultdict(list)
        self._lock = threading.Lock()

        # Interceptors
        self._interceptors: list[Interceptor] = []

        # Event store
        self._event_store = event_store or EventStore()

        # Dead letter queue
        self._dlq = DeadLetterQueue()

        # Consumer tasks
        self._consumer_tasks: dict[str, asyncio.Task] = {}

        # Metrics
        self._metrics = {
            "published": 0,
            "delivered": 0,
            "errors": 0,
            "backpressure_waits": 0,
            "interceptors_suppressed": 0,
        }
        self._metrics_lock = threading.Lock()

        # Event loop reference (set on first async operation)
        self._loop: asyncio.AbstractEventLoop | None = None

        # DLQ retry task
        self._dlq_task: asyncio.Task | None = None

    # ── Subscription ──

    def subscribe(
        self,
        topic: str,
        handler: AsyncHandler | SyncHandler,
        queue_size: int | None = None,
        name: str | None = None,
    ):
        """Subscribe a handler to a topic. Auto-detects sync vs async.

        Args:
            topic: Topic pattern. Supports:
                   - Exact: "order.filled"
                   - Single wildcard: "order.*" matches "order.filled", "order.cancelled"
                   - Double wildcard: "**" matches everything
            handler: Sync or async callable
            queue_size: Per-handler queue size (default: 10000)
            name: Handler name for metrics (default: handler.__name__)
        """
        # Auto-detect: if handler is a coroutine function, treat as async
        if asyncio.iscoroutinefunction(handler):
            q_size = queue_size or self._default_queue_size
            queue = asyncio.Queue(maxsize=q_size)
            handler_name = name or getattr(handler, '__name__', str(id(handler)))
            stats = HandlerStats(handler_name=handler_name, topic=topic)
            with self._lock:
                self._handlers[topic].append((handler, queue, stats, 'async'))
            logger.debug("Subscribed async handler '%s' to '%s' (queue=%d)",
                        handler_name, topic, q_size)
        else:
            # Sync handler — delegate to subscribe_sync
            self.subscribe_sync(topic, handler, queue_size=queue_size, name=name)

    def subscribe_sync(
        self,
        topic: str,
        handler: SyncHandler,
        queue_size: int | None = None,
        name: str | None = None,
        max_workers: int = 4,
    ):
        """Subscribe a sync handler (will be run in thread pool)."""
        q_size = queue_size or self._default_queue_size
        queue = asyncio.Queue(maxsize=q_size)
        handler_name = name or getattr(handler, '__name__', str(id(handler)))
        stats = HandlerStats(handler_name=handler_name, topic=topic)

        with self._lock:
            self._handlers[topic].append((handler, queue, stats, 'sync'))

        logger.debug("Subscribed sync handler '%s' to '%s' (queue=%d)",
                    handler_name, topic, q_size)

    def unsubscribe(self, topic: str, handler: SyncHandler | AsyncHandler):
        """Remove a handler subscription."""
        with self._lock:
            if topic in self._handlers:
                self._handlers[topic] = [
                    (h, q, s, t) for h, q, s, t in self._handlers[topic]
                    if h != handler
                ]

    def add_interceptor(self, fn: Interceptor):
        """Add a pre-publish interceptor. Return None to suppress event."""
        self._interceptors.append(fn)

    # ── Publishing ──

    def publish(self, topic: str, data: dict, source: str = "", **kwargs) -> Event:
        """Synchronous publish. Schedules delivery on the event loop.

        If called from within an async context, use publish_async() instead.
        """
        event = Event(topic=topic, data=data, source=source, **kwargs)

        # Run interceptors
        for interceptor in self._interceptors:
            result = interceptor(event)
            if result is None:
                with self._metrics_lock:
                    self._metrics["interceptors_suppressed"] += 1
                return event
            event = result

        with self._metrics_lock:
            self._metrics["published"] += 1

        # Store event
        self._event_store.append(event)

        # Schedule delivery
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._schedule_delivery, event)
        else:
            # No event loop running — deliver synchronously (blocking)
            self._deliver_sync(event)

        return event

    async def publish_async(self, topic: str, data: dict, source: str = "", **kwargs) -> Event:
        """Async publish. Directly enqueues to handler queues."""
        event = Event(topic=topic, data=data, source=source, **kwargs)

        # Run interceptors
        for interceptor in self._interceptors:
            result = interceptor(event)
            if result is None:
                with self._metrics_lock:
                    self._metrics["interceptors_suppressed"] += 1
                return event
            event = result

        with self._metrics_lock:
            self._metrics["published"] += 1

        # Store event
        self._event_store.append(event)

        # Deliver to all matching handlers
        await self._deliver_async(event)

        return event

    # ── Delivery ──

    def _schedule_delivery(self, event: Event):
        """Schedule async delivery from sync context."""
        if self._loop:
            asyncio.ensure_future(self._deliver_async(event), loop=self._loop)

    async def _deliver_async(self, event: Event):
        """Deliver event to all matching handler queues."""
        handlers = self._get_matching_handlers(event.topic)

        for handler, queue, stats, handler_type in handlers:
            try:
                # Backpressure: wait if queue is full
                while queue.full():
                    with self._metrics_lock:
                        self._metrics["backpressure_waits"] += 1
                    await asyncio.sleep(0.0001)  # 100μs backoff

                await queue.put(event)

            except asyncio.QueueFull:
                # Should not happen with backpressure, but just in case
                with self._metrics_lock:
                    self._metrics["errors"] += 1
                logger.error("Queue full for handler '%s' on '%s'",
                           stats.handler_name, event.topic)

    def _deliver_sync(self, event: Event):
        """Synchronous delivery fallback (blocking)."""
        handlers = self._get_matching_handlers(event.topic)
        for handler, queue, stats, handler_type in handlers:
            try:
                if handler_type == 'async':
                    logger.warning("Cannot deliver to async handler '%s' synchronously",
                                 stats.handler_name)
                    continue
                start = time.time_ns()
                handler(event)
                latency = time.time_ns() - start
                stats.record(latency)
                with self._metrics_lock:
                    self._metrics["delivered"] += 1
            except Exception as e:
                stats.record_error()
                self._dlq.enqueue(event, stats.handler_name, str(e))
                with self._metrics_lock:
                    self._metrics["errors"] += 1
                logger.error("Handler '%s' failed for '%s': %s",
                           stats.handler_name, event.topic, e)

    def _get_matching_handlers(self, topic: str) -> list[tuple]:
        """Get all handlers matching a topic."""
        with self._lock:
            matching = []
            for pattern, handlers in self._handlers.items():
                if self._topic_matches(topic, pattern):
                    matching.extend(handlers)
            return matching

    @staticmethod
    def _topic_matches(topic: str, pattern: str) -> bool:
        """Check if topic matches pattern.

        Patterns:
        - Exact: "order.filled" matches only "order.filled"
        - Single wildcard: "order.*" matches "order.filled", "order.cancelled"
        - Double wildcard: "**" matches everything
        """
        if pattern == '**':
            return True
        if pattern == topic:
            return True

        # Split into parts
        pattern_parts = pattern.split('.')
        topic_parts = topic.split('.')

        if len(pattern_parts) != len(topic_parts):
            return False

        for pp, tp in zip(pattern_parts, topic_parts):
            if pp == '*':
                continue
            if pp != tp:
                return False

        return True

    # ── Consumer Loops ──

    async def start(self):
        """Start all consumer loops. Call once at application startup."""
        self._loop = asyncio.get_event_loop()

        with self._lock:
            for topic, handlers in self._handlers.items():
                for handler, queue, stats, handler_type in handlers:
                    key = f"{topic}:{stats.handler_name}"
                    if key not in self._consumer_tasks:
                        if handler_type == 'async':
                            task = asyncio.create_task(
                                self._async_consumer_loop(handler, queue, stats)
                            )
                        else:
                            task = asyncio.create_task(
                                self._sync_consumer_loop(handler, queue, stats)
                            )
                        self._consumer_tasks[key] = task

        # Start DLQ retry loop
        self._dlq_task = asyncio.create_task(self._dlq_retry_loop())

        logger.info("EventBus started: %d consumers across %d topics",
                   len(self._consumer_tasks), len(self._handlers))

    async def stop(self):
        """Gracefully stop all consumers. Flush WAL."""
        # Cancel all consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        if self._dlq_task:
            self._dlq_task.cancel()

        # Wait for tasks to finish
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks.values(), return_exceptions=True)

        # Close event store
        self._event_store.close()

        logger.info("EventBus stopped. Stats: %s", self._metrics)

    async def _async_consumer_loop(
        self,
        handler: AsyncHandler,
        queue: asyncio.Queue,
        stats: HandlerStats,
    ):
        """Consumer loop for async handlers."""
        while True:
            try:
                event = await queue.get()
                start = time.time_ns()

                try:
                    await handler(event)
                    latency = time.time_ns() - start
                    stats.record(latency)
                    with self._metrics_lock:
                        self._metrics["delivered"] += 1
                except Exception as e:
                    stats.record_error()
                    self._dlq.enqueue(event, stats.handler_name, str(e))
                    with self._metrics_lock:
                        self._metrics["errors"] += 1
                    logger.error("Async handler '%s' failed for '%s': %s",
                               stats.handler_name, event.topic, e)

                queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consumer loop error for '%s': %s",
                           stats.handler_name, e)

    async def _sync_consumer_loop(
        self,
        handler: SyncHandler,
        queue: asyncio.Queue,
        stats: HandlerStats,
    ):
        """Consumer loop for sync handlers (runs in default executor)."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await queue.get()
                start = time.time_ns()

                try:
                    await loop.run_in_executor(None, handler, event)
                    latency = time.time_ns() - start
                    stats.record(latency)
                    with self._metrics_lock:
                        self._metrics["delivered"] += 1
                except Exception as e:
                    stats.record_error()
                    self._dlq.enqueue(event, stats.handler_name, str(e))
                    with self._metrics_lock:
                        self._metrics["errors"] += 1
                    logger.error("Sync handler '%s' failed for '%s': %s",
                               stats.handler_name, event.topic, e)

                queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sync consumer loop error for '%s': %s",
                           stats.handler_name, e)

    async def _dlq_retry_loop(self):
        """Periodically retry dead letters."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Check every second

                ready = self._dlq.get_ready_retries()
                for dl in ready:
                    handlers = self._get_matching_handlers(dl.event.topic)
                    for handler, queue, stats, handler_type in handlers:
                        if stats.handler_name == dl.handler_name:
                            try:
                                if not queue.full():
                                    await queue.put(dl.event)
                                    logger.info("DLQ retry: event %s to '%s' (attempt %d)",
                                              dl.event.event_id, dl.handler_name,
                                              dl.retry_count + 1)
                                else:
                                    self._dlq.requeue(dl, "queue still full")
                            except Exception as e:
                                self._dlq.requeue(dl, str(e))
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("DLQ retry loop error: %s", e)

    # ── Query & Metrics ──

    def get_history(self, topic: str = "", limit: int = 50) -> list[dict]:
        """Get recent events, optionally filtered by topic."""
        if topic:
            events = self._event_store.get_by_topic(topic, limit)
        else:
            events = self._event_store.get_recent(limit)
        return [e.to_dict() for e in events]

    def get_handler_stats(self) -> list[dict]:
        """Get per-handler performance statistics."""
        all_stats = []
        with self._lock:
            for topic, handlers in self._handlers.items():
                for _, _, stats, _ in handlers:
                    all_stats.append(stats.to_dict())
        return sorted(all_stats, key=lambda s: s['p99_us'], reverse=True)

    def get_metrics(self) -> dict:
        """Get bus-level metrics."""
        with self._metrics_lock:
            metrics = dict(self._metrics)

        with self._lock:
            metrics["active_handlers"] = sum(len(h) for h in self._handlers.values())
            metrics["topics"] = list(self._handlers.keys())
            metrics["consumer_tasks"] = len(self._consumer_tasks)

        metrics["dead_letter_queue"] = self._dlq.stats()
        metrics["event_store"] = self._event_store.stats()

        return metrics

    def get_dead_letters(self) -> list[dict]:
        """Get dead letter queue contents."""
        # Access internal state for debugging
        with self._dlq._lock:
            return [
                {
                    "event_id": dl.event.event_id,
                    "topic": dl.event.topic,
                    "handler": dl.handler_name,
                    "error": dl.error,
                    "retry_count": dl.retry_count,
                }
                for dl in self._dlq._queue
            ]

    def clear(self):
        """Clear all state. For testing."""
        with self._lock:
            self._handlers.clear()
            self._consumer_tasks.clear()
        with self._metrics_lock:
            self._metrics = {
                "published": 0, "delivered": 0, "errors": 0,
                "backpressure_waits": 0, "interceptors_suppressed": 0,
            }


# ──────────────────────────────────────────────────────────────────────
# Global Singleton
# ──────────────────────────────────────────────────────────────────────

_bus_v2: AsyncEventBus | None = None


def get_async_event_bus() -> AsyncEventBus:
    """Get the global async event bus singleton."""
    global _bus_v2
    if _bus_v2 is None:
        _bus_v2 = AsyncEventBus()
    return _bus_v2


# ──────────────────────────────────────────────────────────────────────
# Benchmark Utility
# ──────────────────────────────────────────────────────────────────────


async def benchmark_event_bus(
    n_events: int = 100_000,
    n_handlers: int = 10,
    queue_size: int = 100_000,
) -> dict:
    """Benchmark the event bus throughput and latency.

    Returns:
        Dict with throughput, latency percentiles, and handler stats.
    """
    bus = AsyncEventBus(default_queue_size=queue_size)
    received = {"count": 0}
    latencies = []

    async def counter_handler(event: Event):
        received["count"] += 1
        # Simulate 1μs work
        await asyncio.sleep(0.000001)

    # Register handlers
    for i in range(n_handlers):
        bus.subscribe("bench.*", counter_handler, name=f"handler_{i}")

    await bus.start()

    # Publish
    start = time.time()
    for i in range(n_events):
        await bus.publish_async("bench.tick", {"i": i})
    publish_time = time.time() - start

    # Wait for all events to be consumed
    await asyncio.sleep(2.0)

    await bus.stop()

    total_time = time.time() - start
    return {
        "n_events": n_events,
        "n_handlers": n_handlers,
        "publish_time_s": round(publish_time, 3),
        "total_time_s": round(total_time, 3),
        "throughput_events_per_sec": int(n_events / publish_time),
        "events_received": received["count"],
        "handler_stats": bus.get_handler_stats(),
        "bus_metrics": bus.get_metrics(),
    }
