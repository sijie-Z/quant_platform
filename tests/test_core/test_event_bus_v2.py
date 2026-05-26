"""Tests for the async event bus v2."""

import asyncio
import time

import pytest

from quant_platform.core.event_bus_v2 import (
    AsyncEventBus,
    DeadLetterQueue,
    Event,
    EventStore,
    HandlerStats,
)

# ── Event Tests ──


class TestEvent:
    def test_create_event(self):
        event = Event(topic="test.topic", data={"key": "value"})
        assert event.topic == "test.topic"
        assert event.data == {"key": "value"}
        assert len(event.event_id) == 16
        assert event.timestamp > 0

    def test_event_to_dict(self):
        event = Event(topic="test", data={"a": 1}, source="test_source")
        d = event.to_dict()
        assert d["topic"] == "test"
        assert d["data"] == {"a": 1}
        assert d["source"] == "test_source"

    def test_event_to_json(self):
        event = Event(topic="test", data={"a": 1})
        json_str = event.to_json()
        assert '"topic": "test"' in json_str


# ── HandlerStats Tests ──


class TestHandlerStats:
    def test_record_latency(self):
        stats = HandlerStats(handler_name="test", topic="test")
        stats.record(1000)  # 1μs
        stats.record(5000)  # 5μs
        stats.record(10000)  # 10μs
        assert stats.invocations == 3
        assert stats.mean_latency_us == pytest.approx(5.33, rel=0.1)

    def test_percentiles(self):
        stats = HandlerStats(handler_name="test", topic="test")
        for i in range(100):
            stats.record(i * 1000)  # 0-99μs
        assert stats.p50_us > 0
        assert stats.p99_us > 0

    def test_error_count(self):
        stats = HandlerStats(handler_name="test", topic="test")
        stats.record_error()
        stats.record_error()
        assert stats.errors == 2


# ── DeadLetterQueue Tests ──


class TestDeadLetterQueue:
    def test_enqueue_and_size(self):
        dlq = DeadLetterQueue(max_size=10)
        event = Event(topic="test", data={})
        dlq.enqueue(event, "handler1", "error")
        assert dlq.size == 1

    def test_max_size_eviction(self):
        dlq = DeadLetterQueue(max_size=5)
        for i in range(10):
            dlq.enqueue(Event(topic="test", data={}), "h", "e")
        assert dlq.size == 5

    def test_get_ready_retries(self):
        dlq = DeadLetterQueue()
        event = Event(topic="test", data={})
        dlq.enqueue(event, "handler1", "error")
        # Set retry time to past
        dlq._queue[0].next_retry_time = time.time_ns() - 1000
        ready = dlq.get_ready_retries()
        assert len(ready) == 1

    def test_stats(self):
        dlq = DeadLetterQueue()
        dlq.enqueue(Event(topic="test", data={}), "h", "e")
        stats = dlq.stats()
        assert stats["pending"] == 1
        assert stats["total_enqueued"] == 1


# ── EventStore Tests ──


class TestEventStore:
    def test_append_and_get_recent(self):
        store = EventStore(buffer_size=100)
        for i in range(5):
            store.append(Event(topic=f"test.{i}", data={"i": i}))
        recent = store.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].data["i"] == 4

    def test_ring_buffer_overflow(self):
        store = EventStore(buffer_size=10)
        for i in range(20):
            store.append(Event(topic="test", data={"i": i}))
        recent = store.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].data["i"] == 19

    def test_get_by_topic(self):
        store = EventStore()
        store.append(Event(topic="order.filled", data={}))
        store.append(Event(topic="market.tick", data={}))
        store.append(Event(topic="order.cancelled", data={}))
        orders = store.get_by_topic("order.*")
        # Wildcard matching is done at the bus level, not store level
        # Store does prefix matching
        assert len(orders) >= 1

    def test_stats(self):
        store = EventStore()
        store.append(Event(topic="test", data={}))
        stats = store.stats()
        assert stats["total_events"] == 1


# ── AsyncEventBus Tests ──


class TestAsyncEventBus:
    @pytest.fixture
    def bus(self):
        return AsyncEventBus(default_queue_size=1000)

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("test.topic", handler)
        await bus.start()

        await bus.publish_async("test.topic", {"key": "value"})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].data["key"] == "value"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard_matching(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event.topic)

        bus.subscribe("market.*", handler)
        await bus.start()

        await bus.publish_async("market.tick", {})
        await bus.publish_async("market.snapshot", {})
        await bus.publish_async("order.filled", {})  # Should not match

        await asyncio.sleep(0.1)
        assert "market.tick" in received
        assert "market.snapshot" in received
        assert "order.filled" not in received

        await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, bus):
        count = {"a": 0, "b": 0}

        async def handler_a(event):
            count["a"] += 1

        async def handler_b(event):
            count["b"] += 1

        bus.subscribe("test.*", handler_a, name="handler_a")
        bus.subscribe("test.*", handler_b, name="handler_b")
        await bus.start()

        await bus.publish_async("test.event", {})
        await asyncio.sleep(0.1)

        assert count["a"] == 1
        assert count["b"] == 1

        await bus.stop()

    @pytest.mark.asyncio
    async def test_sync_handler(self, bus):
        received = []

        def sync_handler(event: Event):
            received.append(event.data)

        bus.subscribe_sync("test.sync", sync_handler)
        await bus.start()

        await bus.publish_async("test.sync", {"val": 42})
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0]["val"] == 42

        await bus.stop()

    @pytest.mark.asyncio
    async def test_interceptor(self, bus):
        received = []

        async def handler(event):
            received.append(event)

        def suppress_interceptor(event):
            if event.data.get("suppress"):
                return None
            return event

        bus.add_interceptor(suppress_interceptor)
        bus.subscribe("test.*", handler)
        await bus.start()

        await bus.publish_async("test.ok", {"suppress": False})
        await bus.publish_async("test.blocked", {"suppress": True})
        await asyncio.sleep(0.1)

        assert len(received) == 1

        await bus.stop()

    @pytest.mark.asyncio
    async def test_metrics(self, bus):
        async def handler(event):
            pass

        bus.subscribe("test.*", handler)
        await bus.start()

        await bus.publish_async("test.a", {})
        await bus.publish_async("test.b", {})
        await asyncio.sleep(0.1)

        metrics = bus.get_metrics()
        assert metrics["published"] == 2
        assert metrics["delivered"] >= 2

        await bus.stop()

    @pytest.mark.asyncio
    async def test_handler_stats(self, bus):
        async def handler(event):
            await asyncio.sleep(0.001)

        bus.subscribe("test.*", handler, name="my_handler")
        await bus.start()

        for _ in range(10):
            await bus.publish_async("test.event", {})
        await asyncio.sleep(0.5)

        stats = bus.get_handler_stats()
        assert len(stats) >= 1
        handler_stat = next(s for s in stats if s["handler"] == "my_handler")
        assert handler_stat["invocations"] == 10
        assert handler_stat["p50_us"] > 0

        await bus.stop()


# ── Topic Matching Tests ──


class TestTopicMatching:
    def test_exact_match(self):
        assert AsyncEventBus._topic_matches("order.filled", "order.filled")

    def test_single_wildcard(self):
        assert AsyncEventBus._topic_matches("order.filled", "order.*")
        assert AsyncEventBus._topic_matches("order.cancelled", "order.*")
        assert not AsyncEventBus._topic_matches("market.tick", "order.*")

    def test_double_wildcard(self):
        assert AsyncEventBus._topic_matches("anything.here", "**")

    def test_different_lengths(self):
        assert not AsyncEventBus._topic_matches("a.b.c", "a.*")
        assert not AsyncEventBus._topic_matches("a", "a.*")
