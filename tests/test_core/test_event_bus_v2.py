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

    def test_retry_count_preserved_from_event_header(self):
        dlq = DeadLetterQueue()
        event = Event(topic="test", data={})
        event.headers["_retry_count"] = "3"
        dlq.enqueue(event, "handler1", "error")
        assert dlq._queue[0].retry_count == 3
        dlq._queue[0].next_retry_time = time.time_ns() - 1000
        ready = dlq.get_ready_retries()
        assert len(ready) == 1
        assert ready[0].retry_count == 3


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


class TestTheBusCanBeRestarted:
    """`stop()` left its consumer tasks in `_consumer_tasks`, and `start()`
    skips any key already present (`if key not in self._consumer_tasks`), so a
    stop/start pair produced a bus with zero consumers: every published event
    was queued and never delivered, silently."""

    @pytest.fixture
    def bus(self):
        return AsyncEventBus(default_queue_size=1000)

    @pytest.mark.asyncio
    async def test_stop_clears_the_consumer_registry(self, bus):
        async def handler(event: Event):
            pass

        bus.subscribe("test.topic", handler)
        await bus.start()

        assert bus._consumer_tasks, "start() registered no consumers"

        await bus.stop()

        assert bus._consumer_tasks == {}, (
            "the cancelled tasks survived stop(), so start() will skip them"
        )
        assert bus._dlq_task is None
        assert bus._loop is None

    @pytest.mark.asyncio
    async def test_a_restarted_bus_still_delivers(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("test.topic", handler)

        await bus.start()
        await bus.publish_async("test.topic", {"n": 1})
        await asyncio.sleep(0.1)
        await bus.stop()
        assert len(received) == 1

        await bus.start()
        await bus.publish_async("test.topic", {"n": 2})
        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 2, "the restarted bus never delivered the event"
        assert received[1].data["n"] == 2


class TestProductionStart:
    """The global bus used to be created and never started: every publish
    fell back to inline synchronous delivery, async handlers were skipped,
    and the DLQ retry loop never ran. `start()` is now called from the app
    lifespan, and handlers registered afterwards get a consumer task."""

    @pytest.mark.asyncio
    async def test_is_started_reflects_the_lifecycle(self):
        bus = AsyncEventBus(default_queue_size=100)
        assert bus.is_started is False

        await bus.start()
        assert bus.is_started is True

        await bus.stop()
        assert bus.is_started is False

    @pytest.mark.asyncio
    async def test_handler_subscribed_after_start_is_consumed(self):
        bus = AsyncEventBus(default_queue_size=100)
        received = []

        async def handler(event: Event):
            received.append(event)

        await bus.start()
        bus.subscribe("late.topic", handler)
        await asyncio.sleep(0)  # let the consumer task be scheduled

        await bus.publish_async("late.topic", {"n": 1})
        await asyncio.sleep(0.05)
        await bus.stop()

        assert len(received) == 1, "a handler subscribed after start() was never drained"


class TestConsumerIdentity:
    """Consumer tasks were keyed by handler *name*.

    Two handlers called `on_tick` shared one registry key: the second
    subscription never got a consumer and its events accumulated in an
    undrained queue (eventually blocking publish on a full queue).
    """

    @pytest.mark.asyncio
    async def test_same_named_handlers_both_receive(self):
        from quant_platform.core.event_bus_v2 import AsyncEventBus

        bus = AsyncEventBus(default_queue_size=1000)
        received = []

        class A:
            async def on_tick(self, event):
                received.append(("a", event.data["n"]))

        class B:
            async def on_tick(self, event):
                received.append(("b", event.data["n"]))

        bus.subscribe("market.tick", A().on_tick)
        bus.subscribe("market.tick", B().on_tick)
        await bus.start()

        await bus.publish_async("market.tick", {"n": 1})
        await asyncio.sleep(0.1)

        assert ("a", 1) in received
        assert ("b", 1) in received

        await bus.stop()

    @pytest.mark.asyncio
    async def test_resubscribing_after_unsubscribe_delivers_again(self):
        from quant_platform.core.event_bus_v2 import AsyncEventBus

        bus = AsyncEventBus(default_queue_size=1000)
        received = []

        async def handler(event):
            received.append(event.data["n"])

        bus.subscribe("test.topic", handler)
        await bus.start()
        await bus.publish_async("test.topic", {"n": 1})
        await asyncio.sleep(0.05)

        bus.unsubscribe("test.topic", handler)
        bus.subscribe("test.topic", handler)
        await asyncio.sleep(0.05)
        await bus.publish_async("test.topic", {"n": 2})
        await asyncio.sleep(0.1)

        assert received == [1, 2]

        await bus.stop()


class TestWildcardSemanticsAreShared:
    def test_tail_double_wildcard(self):
        assert AsyncEventBus._topic_matches("order", "order.**")
        assert AsyncEventBus._topic_matches("order.filled", "order.**")
        assert AsyncEventBus._topic_matches("order.filled.partial", "order.**")
        assert not AsyncEventBus._topic_matches("orderx", "order.**")

    def test_single_wildcard_is_segment_aware(self):
        assert AsyncEventBus._topic_matches("market.tick", "market.*")
        assert not AsyncEventBus._topic_matches("market", "market.*")
        assert not AsyncEventBus._topic_matches("market_data.tick", "market.*")

    def test_legacy_bus_delivers_and_filters_the_same_way(self):
        from quant_platform.core.events import LegacyEventBus

        bus = LegacyEventBus()
        seen = []
        bus.subscribe("market.*", lambda event: seen.append(event.topic))

        bus.publish("market.tick", {})
        bus.publish("market", {})
        bus.publish("market_data.tick", {})

        assert seen == ["market.tick"]
        assert [e["topic"] for e in bus.get_history("market.*")] == ["market.tick"]
