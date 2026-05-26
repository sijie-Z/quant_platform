"""Distributed message bus abstraction.

Provides a unified interface for inter-service communication with
multiple backend implementations:

1. LocalBus: In-process (for development/testing) — wraps AsyncEventBus
2. RedisBus: Redis Pub/Sub (for single-machine multi-process)
3. KafkaBus: Apache Kafka (for distributed deployment)

The abstraction allows seamless migration from local development to
distributed production without changing application code.

Usage:
    # Development: in-process
    bus = create_message_bus("local")

    # Production: Kafka
    bus = create_message_bus("kafka", brokers=["kafka1:9092", "kafka2:9092"])

    # Publish
    await bus.publish("market.tick", {"symbol": "600519", "price": 1800})

    # Subscribe
    async def on_tick(msg):
        print(msg.data)
    bus.subscribe("market.tick", on_tick)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Message Model
# ──────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Message:
    """A message on the bus. Serializable for cross-process transport."""
    topic: str
    data: dict[str, Any]
    message_id: str = ""
    timestamp: float = 0.0
    source: str = ""
    correlation_id: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.message_id:
            self.message_id = uuid.uuid4().hex[:16]
        if self.timestamp == 0:
            self.timestamp = time.time()

    def serialize(self) -> bytes:
        """Serialize to JSON bytes for transport."""
        return json.dumps({
            "topic": self.topic,
            "data": self.data,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "headers": self.headers,
        }, ensure_ascii=False, default=str).encode('utf-8')

    @classmethod
    def deserialize(cls, raw: bytes) -> Message:
        """Deserialize from JSON bytes."""
        data = json.loads(raw.decode('utf-8'))
        return cls(
            topic=data.get("topic", ""),
            data=data.get("data", {}),
            message_id=data.get("message_id", ""),
            timestamp=data.get("timestamp", 0),
            source=data.get("source", ""),
            correlation_id=data.get("correlation_id", ""),
            headers=data.get("headers", {}),
        )


# Type alias
MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


# ──────────────────────────────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────────────────────────────


class MessageBus(ABC):
    """Abstract message bus interface.

    All implementations must provide:
    - publish: send a message to a topic
    - subscribe: register a handler for a topic
    - start/stop: lifecycle management
    - health_check: connectivity verification
    """

    @abstractmethod
    async def publish(self, topic: str, data: dict, **kwargs) -> Message:
        """Publish a message to a topic."""
        ...

    @abstractmethod
    def subscribe(self, topic: str, handler: MessageHandler, **kwargs):
        """Subscribe a handler to a topic."""
        ...

    @abstractmethod
    async def start(self):
        """Start the message bus."""
        ...

    @abstractmethod
    async def stop(self):
        """Stop the message bus gracefully."""
        ...

    @abstractmethod
    async def health_check(self) -> dict:
        """Check bus connectivity and health."""
        ...

    def get_metrics(self) -> dict:
        """Get bus metrics (override in subclasses)."""
        return {}


# ──────────────────────────────────────────────────────────────────────
# Local Bus (In-Process)
# ──────────────────────────────────────────────────────────────────────


class LocalBus(MessageBus):
    """In-process message bus using asyncio queues.

    For development and testing. No network overhead.
    Wraps the AsyncEventBus pattern with the MessageBus interface.
    """

    def __init__(self):
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._tasks: list[asyncio.Task] = []
        self._published = 0
        self._delivered = 0
        self._errors = 0
        self._started = False

    async def publish(self, topic: str, data: dict, **kwargs) -> Message:
        msg = Message(topic=topic, data=data, **kwargs)
        self._published += 1

        if self._started:
            # Find matching handlers
            for pattern, queues in self._queues.items():
                if self._matches(topic, pattern):
                    for queue in queues:
                        try:
                            await queue.put(msg)
                        except asyncio.QueueFull:
                            self._errors += 1

        return msg

    def subscribe(self, topic: str, handler: MessageHandler, queue_size: int = 10000):
        if topic not in self._handlers:
            self._handlers[topic] = []
            self._queues[topic] = []

        self._handlers[topic].append(handler)
        queue = asyncio.Queue(maxsize=queue_size)
        self._queues[topic].append(queue)

        # If already started, create consumer task
        if self._started:
            task = asyncio.create_task(self._consume(handler, queue, topic))
            self._tasks.append(task)

    async def start(self):
        self._started = True
        # Create consumer tasks for all existing subscriptions
        for topic, handlers in self._handlers.items():
            queues = self._queues.get(topic, [])
            for handler, queue in zip(handlers, queues, strict=False):
                task = asyncio.create_task(self._consume(handler, queue, topic))
                self._tasks.append(task)
        logger.info("LocalBus started with %d subscriptions", sum(len(h) for h in self._handlers.values()))

    async def stop(self):
        self._started = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("LocalBus stopped. Published=%d, Delivered=%d, Errors=%d",
                   self._published, self._delivered, self._errors)

    async def health_check(self) -> dict:
        return {
            "status": "healthy",
            "backend": "local",
            "started": self._started,
            "subscriptions": sum(len(h) for h in self._handlers.values()),
            "published": self._published,
            "delivered": self._delivered,
            "errors": self._errors,
        }

    async def _consume(self, handler: MessageHandler, queue: asyncio.Queue, topic: str):
        while self._started:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                await handler(msg)
                self._delivered += 1
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                logger.error("Handler error on '%s': %s", topic, e)

    @staticmethod
    def _matches(topic: str, pattern: str) -> bool:
        if pattern == '**' or pattern == topic:
            return True
        pattern_parts = pattern.split('.')
        topic_parts = topic.split('.')
        if len(pattern_parts) != len(topic_parts):
            return False
        return all(p == '*' or p == t for p, t in zip(pattern_parts, topic_parts, strict=False))


# ──────────────────────────────────────────────────────────────────────
# Redis Bus (Single-Machine Multi-Process)
# ──────────────────────────────────────────────────────────────────────


class RedisBus(MessageBus):
    """Redis Pub/Sub message bus.

    For single-machine multi-process deployment.
    Requires: pip install redis[hiredis]

    Usage:
        bus = RedisBus(host="localhost", port=6379)
        await bus.start()
        bus.subscribe("market.*", on_tick)
        await bus.publish("market.tick", {"price": 100})
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "quant:",
    ):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._prefix = prefix

        self._redis = None
        self._pubsub = None
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._listen_task: asyncio.Task | None = None
        self._published = 0
        self._received = 0
        self._errors = 0

    async def start(self):
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
                decode_responses=False,
            )
            # Test connection
            await self._redis.ping()
            self._pubsub = self._redis.pubsub()

            # Subscribe existing handlers
            for topic in self._handlers:
                redis_channel = f"{self._prefix}{topic}"
                await self._pubsub.psubscribe(redis_channel)

            # Start listener
            self._listen_task = asyncio.create_task(self._listen_loop())

            logger.info("RedisBus connected to %s:%d", self._host, self._port)

        except ImportError:
            logger.error("redis package not installed: pip install redis[hiredis]")
            raise
        except Exception as e:
            logger.error("RedisBus connection failed: %s", e)
            raise

    async def stop(self):
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("RedisBus stopped")

    async def publish(self, topic: str, data: dict, **kwargs) -> Message:
        if not self._redis:
            raise RuntimeError("RedisBus not started")

        msg = Message(topic=topic, data=data, **kwargs)
        redis_channel = f"{self._prefix}{topic}"

        await self._redis.publish(redis_channel, msg.serialize())
        self._published += 1

        return msg

    def subscribe(self, topic: str, handler: MessageHandler, **kwargs):
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

        # If already connected, subscribe to Redis channel
        if self._pubsub:
            redis_channel = f"{self._prefix}{topic}"
            asyncio.ensure_future(self._pubsub.psubscribe(redis_channel))

    async def health_check(self) -> dict:
        try:
            if self._redis:
                await self._redis.ping()
                return {
                    "status": "healthy",
                    "backend": "redis",
                    "host": self._host,
                    "port": self._port,
                    "published": self._published,
                    "received": self._received,
                    "errors": self._errors,
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

        return {"status": "not_connected"}

    async def _listen_loop(self):
        """Listen for messages and dispatch to handlers."""
        try:
            async for raw_msg in self._pubsub.listen():
                if raw_msg["type"] not in ("message", "pmessage"):
                    continue

                try:
                    msg = Message.deserialize(raw_msg["data"])
                    self._received += 1

                    # Find matching handlers
                    for pattern, handlers in self._handlers.items():
                        if self._matches(msg.topic, pattern):
                            for handler in handlers:
                                try:
                                    await handler(msg)
                                except Exception as e:
                                    self._errors += 1
                                    logger.error("Handler error: %s", e)

                except Exception as e:
                    self._errors += 1
                    logger.error("Message deserialize error: %s", e)

        except asyncio.CancelledError:
            pass

    @staticmethod
    def _matches(topic: str, pattern: str) -> bool:
        if pattern == '**':
            return True
        pattern_parts = pattern.split('.')
        topic_parts = topic.split('.')
        if len(pattern_parts) != len(topic_parts):
            return False
        return all(p == '*' or p == t for p, t in zip(pattern_parts, topic_parts, strict=False))


# ──────────────────────────────────────────────────────────────────────
# Kafka Bus (Distributed)
# ──────────────────────────────────────────────────────────────────────


class KafkaBus(MessageBus):
    """Apache Kafka message bus for distributed deployment.

    For multi-machine, high-throughput, durable messaging.
    Requires: pip install aiokafka

    Usage:
        bus = KafkaBus(brokers=["kafka1:9092", "kafka2:9092"])
        await bus.start()
        bus.subscribe("market.tick", on_tick, group="risk-service")
        await bus.publish("market.tick", {"price": 100})
    """

    def __init__(
        self,
        brokers: list[str] | None = None,
        group_id: str = "quant-platform",
        prefix: str = "quant.",
        auto_offset_reset: str = "latest",
    ):
        self._brokers = brokers or ["localhost:9092"]
        self._group_id = group_id
        self._prefix = prefix
        self._auto_offset_reset = auto_offset_reset

        self._producer = None
        self._consumers: dict[str, Any] = {}
        self._handlers: dict[str, list[tuple[MessageHandler, str]]] = {}  # topic -> [(handler, group)]
        self._consume_tasks: list[asyncio.Task] = []
        self._published = 0
        self._received = 0
        self._errors = 0

    async def start(self):
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._brokers,
                value_serializer=lambda v: v,
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            await self._producer.start()

            # Create consumers for subscriptions
            for topic, handlers in self._handlers.items():
                groups = set(g for _, g in handlers)
                for group in groups:
                    consumer = AIOKafkaConsumer(
                        f"{self._prefix}{topic}",
                        bootstrap_servers=self._brokers,
                        group_id=group or self._group_id,
                        auto_offset_reset=self._auto_offset_reset,
                        value_deserializer=lambda v: v,
                    )
                    await consumer.start()
                    self._consumers[f"{topic}:{group}"] = consumer

                    task = asyncio.create_task(
                        self._consume_loop(consumer, topic, group)
                    )
                    self._consume_tasks.append(task)

            logger.info("KafkaBus connected to %s", self._brokers)

        except ImportError:
            logger.error("aiokafka not installed: pip install aiokafka")
            raise
        except Exception as e:
            logger.error("KafkaBus connection failed: %s", e)
            raise

    async def stop(self):
        for task in self._consume_tasks:
            task.cancel()
        if self._consume_tasks:
            await asyncio.gather(*self._consume_tasks, return_exceptions=True)

        for consumer in self._consumers.values():
            await consumer.stop()

        if self._producer:
            await self._producer.stop()

        logger.info("KafkaBus stopped")

    async def publish(self, topic: str, data: dict, key: str = "", **kwargs) -> Message:
        if not self._producer:
            raise RuntimeError("KafkaBus not started")

        msg = Message(topic=topic, data=data, **kwargs)
        kafka_topic = f"{self._prefix}{topic}"

        await self._producer.send_and_wait(
            kafka_topic,
            value=msg.serialize(),
            key=key or msg.message_id,
        )
        self._published += 1

        return msg

    def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        group: str = "",
        **kwargs,
    ):
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append((handler, group or self._group_id))

    async def health_check(self) -> dict:
        try:
            if self._producer:
                # Try to get cluster metadata
                return {
                    "status": "healthy",
                    "backend": "kafka",
                    "brokers": self._brokers,
                    "published": self._published,
                    "received": self._received,
                    "errors": self._errors,
                    "consumers": len(self._consumers),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

        return {"status": "not_connected"}

    async def _consume_loop(self, consumer, topic: str, group: str):
        """Consume messages from Kafka and dispatch to handlers."""
        try:
            async for raw_msg in consumer:
                try:
                    msg = Message.deserialize(raw_msg.value)
                    self._received += 1

                    handlers = self._handlers.get(topic, [])
                    for handler, handler_group in handlers:
                        if handler_group == group or not group:
                            try:
                                await handler(msg)
                            except Exception as e:
                                self._errors += 1
                                logger.error("Kafka handler error on '%s': %s", topic, e)

                except Exception as e:
                    self._errors += 1
                    logger.error("Kafka message error: %s", e)

        except asyncio.CancelledError:
            pass


# ──────────────────────────────────────────────────────────────────────
# Service Registry
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ServiceInfo:
    """Information about a registered service."""
    name: str
    version: str
    host: str
    port: int
    status: str = "starting"
    last_heartbeat: float = 0.0
    metadata: dict = field(default_factory=dict)


class ServiceRegistry:
    """Simple service registry for service discovery.

    In production, this would be backed by etcd/Consul/ZooKeeper.
    Here we use an in-memory dict for development.

    Usage:
        registry = ServiceRegistry()
        registry.register(ServiceInfo("risk-service", "1.0.0", "localhost", 8001))
        services = registry.discover("risk-service")
    """

    def __init__(self, heartbeat_timeout: float = 30.0):
        self._services: dict[str, list[ServiceInfo]] = {}
        self._heartbeat_timeout = heartbeat_timeout

    def register(self, service: ServiceInfo):
        """Register a service instance."""
        service.status = "healthy"
        service.last_heartbeat = time.time()

        if service.name not in self._services:
            self._services[service.name] = []

        # Update existing or add new
        existing = [
            s for s in self._services[service.name]
            if s.host == service.host and s.port == service.port
        ]
        if existing:
            existing[0].status = service.status
            existing[0].last_heartbeat = service.last_heartbeat
        else:
            self._services[service.name].append(service)

        logger.info("Registered service: %s at %s:%d",
                   service.name, service.host, service.port)

    def deregister(self, name: str, host: str, port: int):
        """Deregister a service instance."""
        if name in self._services:
            self._services[name] = [
                s for s in self._services[name]
                if not (s.host == host and s.port == port)
            ]

    def discover(self, name: str) -> list[ServiceInfo]:
        """Discover healthy instances of a service."""
        services = self._services.get(name, [])
        now = time.time()

        healthy = []
        for s in services:
            if now - s.last_heartbeat > self._heartbeat_timeout:
                s.status = "unhealthy"
            if s.status == "healthy":
                healthy.append(s)

        return healthy

    def heartbeat(self, name: str, host: str, port: int):
        """Update heartbeat for a service instance."""
        services = self._services.get(name, [])
        for s in services:
            if s.host == host and s.port == port:
                s.last_heartbeat = time.time()
                s.status = "healthy"
                return

    def get_all(self) -> dict[str, list[dict]]:
        """Get all registered services."""
        return {
            name: [
                {
                    "host": s.host,
                    "port": s.port,
                    "status": s.status,
                    "version": s.version,
                    "last_heartbeat": s.last_heartbeat,
                }
                for s in instances
            ]
            for name, instances in self._services.items()
        }


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────


def create_message_bus(backend: str = "local", **kwargs) -> MessageBus:
    """Create a message bus instance.

    Args:
        backend: "local", "redis", or "kafka"
        **kwargs: Backend-specific configuration

    Returns:
        MessageBus instance
    """
    if backend == "local":
        return LocalBus()
    elif backend == "redis":
        return RedisBus(**kwargs)
    elif backend == "kafka":
        return KafkaBus(**kwargs)
    else:
        raise ValueError(f"Unknown message bus backend: {backend}")
