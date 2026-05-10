"""Tests for the distributed message bus abstraction."""

import asyncio

import pytest

from quant_platform.core.message_bus import (
    KafkaBus,
    LocalBus,
    Message,
    MessageBus,
    RedisBus,
    ServiceInfo,
    ServiceRegistry,
    create_message_bus,
)


# ── Message Tests ──


class TestMessage:
    def test_create(self):
        msg = Message(topic="test", data={"key": "value"})
        assert msg.topic == "test"
        assert len(msg.message_id) == 16
        assert msg.timestamp > 0

    def test_serialize_deserialize(self):
        msg = Message(topic="test", data={"a": 1, "b": "hello"})
        raw = msg.serialize()
        restored = Message.deserialize(raw)
        assert restored.topic == "test"
        assert restored.data == {"a": 1, "b": "hello"}
        assert restored.message_id == msg.message_id

    def test_with_headers(self):
        msg = Message(topic="test", data={}, headers={"x-trace-id": "abc123"})
        raw = msg.serialize()
        restored = Message.deserialize(raw)
        assert restored.headers["x-trace-id"] == "abc123"


# ── LocalBus Tests ──


class TestLocalBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = LocalBus()
        received = []

        async def handler(msg: Message):
            received.append(msg)

        bus.subscribe("test.topic", handler)
        await bus.start()

        await bus.publish("test.topic", {"key": "value"})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].data["key"] == "value"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard(self):
        bus = LocalBus()
        received = []

        async def handler(msg: Message):
            received.append(msg.topic)

        bus.subscribe("market.*", handler)
        await bus.start()

        await bus.publish("market.tick", {})
        await bus.publish("order.filled", {})
        await asyncio.sleep(0.1)

        assert "market.tick" in received
        assert "order.filled" not in received

        await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = LocalBus()
        count = {"a": 0, "b": 0}

        async def handler_a(msg):
            count["a"] += 1

        async def handler_b(msg):
            count["b"] += 1

        bus.subscribe("test.*", handler_a)
        bus.subscribe("test.*", handler_b)
        await bus.start()

        await bus.publish("test.event", {})
        await asyncio.sleep(0.1)

        assert count["a"] == 1
        assert count["b"] == 1

        await bus.stop()

    @pytest.mark.asyncio
    async def test_health_check(self):
        bus = LocalBus()
        await bus.start()
        health = await bus.health_check()
        assert health["status"] == "healthy"
        assert health["backend"] == "local"
        await bus.stop()

    @pytest.mark.asyncio
    async def test_subscribe_before_start(self):
        """Handlers subscribed before start should still work."""
        bus = LocalBus()
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("test", handler)
        await bus.start()

        await bus.publish("test", {"val": 1})
        await asyncio.sleep(0.1)

        assert len(received) == 1
        await bus.stop()


# ── ServiceRegistry Tests ──


class TestServiceRegistry:
    def test_register_and_discover(self):
        registry = ServiceRegistry()
        registry.register(ServiceInfo(
            name="risk-service", version="1.0",
            host="localhost", port=8001,
        ))
        services = registry.discover("risk-service")
        assert len(services) == 1
        assert services[0].host == "localhost"

    def test_deregister(self):
        registry = ServiceRegistry()
        registry.register(ServiceInfo(
            name="svc", version="1.0",
            host="localhost", port=8001,
        ))
        registry.deregister("svc", "localhost", 8001)
        assert len(registry.discover("svc")) == 0

    def test_heartbeat(self):
        registry = ServiceRegistry()
        registry.register(ServiceInfo(
            name="svc", version="1.0",
            host="localhost", port=8001,
        ))
        registry.heartbeat("svc", "localhost", 8001)
        services = registry.discover("svc")
        assert len(services) == 1

    def test_heartbeat_timeout(self):
        registry = ServiceRegistry(heartbeat_timeout=0.01)
        registry.register(ServiceInfo(
            name="svc", version="1.0",
            host="localhost", port=8001,
        ))
        import time
        time.sleep(0.02)
        services = registry.discover("svc")
        assert len(services) == 0

    def test_get_all(self):
        registry = ServiceRegistry()
        registry.register(ServiceInfo("svc1", "1.0", "h1", 1))
        registry.register(ServiceInfo("svc2", "1.0", "h2", 2))
        all_services = registry.get_all()
        assert "svc1" in all_services
        assert "svc2" in all_services


# ── Factory Tests ──


class TestFactory:
    def test_create_local(self):
        bus = create_message_bus("local")
        assert isinstance(bus, LocalBus)

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            create_message_bus("unknown_backend")
