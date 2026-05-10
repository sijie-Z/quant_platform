"""Base service class for microservice architecture.

All services follow the same lifecycle:
1. Initialize (config, bus connection)
2. Register with service registry
3. Subscribe to input topics
4. Process messages
5. Publish results to output topics
6. Graceful shutdown

This base class provides:
- Health check endpoint
- Metrics collection
- Graceful shutdown handling
- Service registration/heartbeat
- Error handling and logging
"""

from __future__ import annotations

import asyncio
import signal
import time
from abc import ABC, abstractmethod
from typing import Any

from quant_platform.core.message_bus import (
    Message,
    MessageBus,
    ServiceInfo,
    ServiceRegistry,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class BaseService(ABC):
    """Base class for all microservices.

    Subclasses must implement:
    - name: service name
    - version: service version
    - setup(): initialize resources
    - register_handlers(): subscribe to topics
    - shutdown(): cleanup resources

    Usage:
        class MyService(BaseService):
            name = "my-service"
            version = "1.0.0"

            async def setup(self):
                self.db = await connect_db()

            def register_handlers(self):
                self.bus.subscribe("input.topic", self.handle_input)

            async def handle_input(self, msg: Message):
                result = process(msg.data)
                await self.bus.publish("output.topic", result)

            async def shutdown(self):
                await self.db.close()
    """

    name: str = "base-service"
    version: str = "0.0.0"
    host: str = "localhost"
    port: int = 0

    def __init__(
        self,
        bus: MessageBus,
        registry: ServiceRegistry | None = None,
    ):
        self.bus = bus
        self.registry = registry
        self._running = False
        self._start_time = 0.0

        # Metrics
        self._messages_processed = 0
        self._messages_published = 0
        self._errors = 0
        self._last_error = ""

    # ── Lifecycle ──

    async def run(self):
        """Main entry point. Handles full lifecycle."""
        logger.info("Starting %s v%s", self.name, self.version)
        self._start_time = time.time()

        # Setup
        await self.setup()

        # Start bus
        await self.bus.start()

        # Register handlers
        self.register_handlers()

        # Register with service registry
        if self.registry:
            self.registry.register(ServiceInfo(
                name=self.name,
                version=self.version,
                host=self.host,
                port=self.port,
            ))

        self._running = True

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.stop()))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("%s is running", self.name)

        # Wait until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            heartbeat_task.cancel()
            await self.shutdown()
            await self.bus.stop()

            if self.registry:
                self.registry.deregister(self.name, self.host, self.port)

            logger.info("%s stopped. Processed=%d, Published=%d, Errors=%d",
                       self.name, self._messages_processed,
                       self._messages_published, self._errors)

    async def stop(self):
        """Signal the service to stop."""
        self._running = False

    # ── Abstract Methods ──

    @abstractmethod
    async def setup(self):
        """Initialize resources (database, caches, etc.)."""
        ...

    @abstractmethod
    def register_handlers(self):
        """Subscribe to message bus topics."""
        ...

    @abstractmethod
    async def shutdown(self):
        """Cleanup resources."""
        ...

    # ── Health & Metrics ──

    async def health_check(self) -> dict:
        """Service health check."""
        bus_health = await self.bus.health_check()
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "service": self.name,
            "version": self.version,
            "status": "healthy" if self._running else "stopped",
            "uptime_seconds": round(uptime, 1),
            "messages_processed": self._messages_processed,
            "messages_published": self._messages_published,
            "errors": self._errors,
            "last_error": self._last_error,
            "bus": bus_health,
        }

    def get_metrics(self) -> dict:
        """Get service metrics."""
        return {
            "messages_processed": self._messages_processed,
            "messages_published": self._messages_published,
            "errors": self._errors,
            "uptime": time.time() - self._start_time if self._start_time else 0,
        }

    # ── Helpers ──

    async def _heartbeat_loop(self):
        """Periodic heartbeat to service registry."""
        while self._running:
            try:
                if self.registry:
                    self.registry.heartbeat(self.name, self.host, self.port)
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat error: %s", e)

    def _record_message(self):
        self._messages_processed += 1

    def _record_publish(self):
        self._messages_published += 1

    def _record_error(self, error: str):
        self._errors += 1
        self._last_error = error
