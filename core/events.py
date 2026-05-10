"""Event bus — the nervous system of the platform.

All components communicate through events, not direct calls.
This is how Jane Street/Citadel systems decouple their components.

Provides two implementations:
- LegacyEventBus: Original sync event bus (used by unit tests)
- AsyncEventBus: High-performance async bus with backpressure, latency
  histograms, dead letter retry, and WAL event sourcing (from event_bus_v2)

`get_event_bus()` returns an AsyncEventBus instance — all existing consumers
(engine, audit, scheduler, routes) get the upgraded bus transparently.

Usage:
    bus = get_event_bus()
    bus.subscribe('market.tick', on_tick)          # sync handler auto-detected
    bus.publish('market.tick', {'code': '600519'})

Event types follow domain-driven naming:
    market.tick       — real-time price update
    market.snapshot   — full market snapshot
    signal.generated  — alpha signal produced
    order.submitted   — order sent to broker
    order.filled      — order executed
    order.rejected    — order failed
    position.updated  — position changed
    portfolio.rebalanced — portfolio target reached
    risk.breach       — risk limit violated
    engine.started    — trading engine started
    engine.stopped    — trading engine stopped
    system.error      — system-level error
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Re-export v2 components (primary API)
# ──────────────────────────────────────────────────────────────────────

from quant_platform.core.event_bus_v2 import (  # noqa: E402
    AsyncEventBus,
    Event,
    HandlerStats,
    DeadLetterQueue,
    EventStore,
    get_async_event_bus,
)


# Type alias for event handlers
EventHandler = Callable[[Any], None]


# ──────────────────────────────────────────────────────────────────────
# Legacy EventBus (backward compat — used by unit tests)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LegacyEvent:
    """Legacy event format (time.time() seconds, time_str in to_dict)."""
    topic: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            import uuid
            self.event_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "time_str": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


class LegacyEventBus:
    """Original sync event bus. Kept for backward compatibility with unit tests.

    New code should use AsyncEventBus via get_event_bus().
    """

    def __init__(self, history_size: int = 1000):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[LegacyEvent] = []
        self._history_size = history_size
        self._dead_letters: list[LegacyEvent] = []
        self._lock = threading.Lock()
        self._metrics = {
            "published": 0,
            "delivered": 0,
            "errors": 0,
            "dead_letters": 0,
        }
        self._interceptors: list[Callable[[LegacyEvent], LegacyEvent | None]] = []

    def subscribe(self, topic: str, handler: EventHandler):
        with self._lock:
            self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: EventHandler):
        with self._lock:
            if topic in self._handlers:
                self._handlers[topic] = [h for h in self._handlers[topic] if h != handler]

    def add_interceptor(self, fn: Callable[[LegacyEvent], LegacyEvent | None]):
        self._interceptors.append(fn)

    def publish(self, topic: str, data: dict, source: str = "") -> LegacyEvent:
        event = LegacyEvent(topic=topic, data=data, source=source)
        for interceptor in self._interceptors:
            result = interceptor(event)
            if result is None:
                return event
            event = result

        with self._lock:
            self._metrics["published"] += 1
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        self._deliver(event)
        return event

    def _deliver(self, event: LegacyEvent):
        with self._lock:
            handlers = list(self._handlers.get(event.topic, []))
            for pattern, pattern_handlers in self._handlers.items():
                if pattern.endswith('.*') and event.topic.startswith(pattern[:-2]):
                    handlers.extend(pattern_handlers)
                elif pattern == '*':
                    handlers.extend(pattern_handlers)

        for handler in handlers:
            try:
                handler(event)
                with self._lock:
                    self._metrics["delivered"] += 1
            except Exception as e:
                with self._lock:
                    self._metrics["errors"] += 1
                    self._dead_letters.append(event)
                    if len(self._dead_letters) > 100:
                        self._dead_letters = self._dead_letters[-50:]
                logger.error("Handler %s failed for %s: %s",
                             handler.__name__, event.topic, e)

    def get_history(self, topic: str = "", limit: int = 50) -> list[dict]:
        with self._lock:
            events = self._history
            if topic:
                events = [e for e in events if e.topic == topic or
                          (topic.endswith('.*') and e.topic.startswith(topic[:-2]))]
            return [e.to_dict() for e in events[-limit:]]

    def get_dead_letters(self) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in self._dead_letters]

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                **self._metrics,
                "active_handlers": sum(len(h) for h in self._handlers.values()),
                "topics": list(self._handlers.keys()),
                "history_size": len(self._history),
                "dead_letter_count": len(self._dead_letters),
            }

    def clear(self):
        with self._lock:
            self._handlers.clear()
            self._history.clear()
            self._dead_letters.clear()
            self._metrics = {"published": 0, "delivered": 0, "errors": 0, "dead_letters": 0}


# Backward-compatible alias — tests that create `EventBus()` still work
EventBus = LegacyEventBus


# ──────────────────────────────────────────────────────────────────────
# Global singleton (returns AsyncEventBus)
# ──────────────────────────────────────────────────────────────────────

_bus: AsyncEventBus | None = None


def get_event_bus() -> AsyncEventBus:
    """Get the global event bus singleton.

    Returns AsyncEventBus — all existing consumers get the upgraded bus
    transparently. Sync handlers registered via subscribe() are auto-detected.
    """
    global _bus
    if _bus is None:
        _bus = AsyncEventBus()
    return _bus
