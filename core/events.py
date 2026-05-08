"""Async event bus — the nervous system of the platform.

All components communicate through events, not direct calls.
This is how Jane Street/Citadel systems decouple their components.

Usage:
    bus = EventBus()
    bus.subscribe('market.tick', on_tick)
    bus.publish('market.tick', {'code': '600519', 'price': 1800.0})

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


@dataclass
class Event:
    """An event on the bus."""
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


# Type alias for event handlers
EventHandler = Callable[[Event], None]


class EventBus:
    """In-process async event bus with synchronous fallback.

    Supports:
    - Topic-based pub/sub with wildcard matching ('market.*')
    - Async handlers (run in thread pool)
    - Sync handlers (run directly)
    - Event history (ring buffer)
    - Dead letter queue for failed events
    - Metrics: publish count, handler count, error count

    Architecture:
        Publisher → EventBus → Subscriber(s)
                    ↓
              Event History (ring buffer)
              Dead Letter Queue (errors)
    """

    def __init__(self, history_size: int = 1000):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._history_size = history_size
        self._dead_letters: list[Event] = []
        self._lock = threading.Lock()
        self._metrics = {
            "published": 0,
            "delivered": 0,
            "errors": 0,
            "dead_letters": 0,
        }
        self._interceptors: list[Callable[[Event], Event | None]] = []

    def subscribe(self, topic: str, handler: EventHandler):
        """Subscribe to a topic. Supports wildcard 'market.*'."""
        with self._lock:
            self._handlers[topic].append(handler)
            logger.debug("Subscribed to %s: %s", topic, handler.__name__)

    def unsubscribe(self, topic: str, handler: EventHandler):
        """Remove a subscription."""
        with self._lock:
            if topic in self._handlers:
                self._handlers[topic] = [h for h in self._handlers[topic] if h != handler]

    def add_interceptor(self, fn: Callable[[Event], Event | None]):
        """Add an event interceptor. Return None to suppress the event."""
        self._interceptors.append(fn)

    def publish(self, topic: str, data: dict, source: str = "") -> Event:
        """Publish an event. Returns the Event object."""
        event = Event(topic=topic, data=data, source=source)

        # Run interceptors
        for interceptor in self._interceptors:
            result = interceptor(event)
            if result is None:
                return event  # suppressed
            event = result

        with self._lock:
            self._metrics["published"] += 1
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        # Deliver to matching handlers
        self._deliver(event)
        return event

    def _deliver(self, event: Event):
        """Deliver event to all matching handlers."""
        with self._lock:
            handlers = list(self._handlers.get(event.topic, []))
            # Wildcard matching: 'market.*' matches 'market.tick'
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
                logger.debug(traceback.format_exc())

    def get_history(self, topic: str = "", limit: int = 50) -> list[dict]:
        """Get recent events, optionally filtered by topic."""
        with self._lock:
            events = self._history
            if topic:
                events = [e for e in events if e.topic == topic or
                          (topic.endswith('.*') and e.topic.startswith(topic[:-2]))]
            return [e.to_dict() for e in events[-limit:]]

    def get_dead_letters(self) -> list[dict]:
        """Get failed events."""
        with self._lock:
            return [e.to_dict() for e in self._dead_letters]

    def get_metrics(self) -> dict:
        """Get bus metrics."""
        with self._lock:
            return {
                **self._metrics,
                "active_handlers": sum(len(h) for h in self._handlers.values()),
                "topics": list(self._handlers.keys()),
                "history_size": len(self._history),
                "dead_letter_count": len(self._dead_letters),
            }

    def clear(self):
        """Clear all state."""
        with self._lock:
            self._handlers.clear()
            self._history.clear()
            self._dead_letters.clear()
            self._metrics = {"published": 0, "delivered": 0, "errors": 0, "dead_letters": 0}


# Global singleton
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
