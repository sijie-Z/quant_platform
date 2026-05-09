"""Tests for core.events — EventBus."""

import pytest
from quant_platform.core.events import EventBus, Event, get_event_bus


class TestEventBus:
    def setup_method(self):
        self.bus = EventBus(history_size=100)

    def test_publish_subscribe(self):
        received = []
        self.bus.subscribe("test.topic", lambda e: received.append(e))
        self.bus.publish("test.topic", {"value": 42})
        assert len(received) == 1
        assert received[0].data["value"] == 42
        assert received[0].topic == "test.topic"

    def test_wildcard_subscribe(self):
        received = []
        self.bus.subscribe("market.*", lambda e: received.append(e))
        self.bus.publish("market.tick", {"price": 100})
        self.bus.publish("market.snapshot", {"n": 500})
        self.bus.publish("signal.generated", {"code": "600519"})
        assert len(received) == 2

    def test_global_wildcard(self):
        received = []
        self.bus.subscribe("*", lambda e: received.append(e))
        self.bus.publish("any.topic", {})
        assert len(received) == 1

    def test_unsubscribe(self):
        received = []
        handler = lambda e: received.append(e)
        self.bus.subscribe("test", handler)
        self.bus.publish("test", {})
        assert len(received) == 1
        self.bus.unsubscribe("test", handler)
        self.bus.publish("test", {})
        assert len(received) == 1

    def test_interceptor_suppress(self):
        received = []
        self.bus.subscribe("test", lambda e: received.append(e))
        self.bus.add_interceptor(lambda e: None)  # suppress all
        self.bus.publish("test", {})
        assert len(received) == 0

    def test_interceptor_modify(self):
        received = []
        self.bus.subscribe("test", lambda e: received.append(e))
        self.bus.add_interceptor(lambda e: Event(topic=e.topic, data={"modified": True}))
        self.bus.publish("test", {"original": True})
        assert received[0].data["modified"] is True

    def test_history(self):
        self.bus.publish("a", {"i": 1})
        self.bus.publish("b", {"i": 2})
        self.bus.publish("a", {"i": 3})
        history = self.bus.get_history()
        assert len(history) == 3
        filtered = self.bus.get_history(topic="a")
        assert len(filtered) == 2

    def test_dead_letter_on_handler_error(self):
        def bad_handler(e):
            raise RuntimeError("boom")
        self.bus.subscribe("test", bad_handler)
        self.bus.publish("test", {})
        dl = self.bus.get_dead_letters()
        assert len(dl) == 1

    def test_metrics(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.publish("test", {})
        m = self.bus.get_metrics()
        assert m["published"] == 1
        assert m["delivered"] == 1
        assert m["active_handlers"] == 1

    def test_clear(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.publish("test", {})
        self.bus.clear()
        m = self.bus.get_metrics()
        assert m["published"] == 0
        assert m["active_handlers"] == 0

    def test_event_to_dict(self):
        e = Event(topic="test", data={"x": 1}, source="unit_test")
        d = e.to_dict()
        assert d["topic"] == "test"
        assert d["source"] == "unit_test"
        assert "event_id" in d
        assert "time_str" in d

    def test_event_auto_id(self):
        e1 = Event(topic="a", data={})
        e2 = Event(topic="b", data={})
        assert e1.event_id != e2.event_id

    def test_history_ring_buffer(self):
        bus = EventBus(history_size=5)
        for i in range(10):
            bus.publish("test", {"i": i})
        assert len(bus.get_history()) == 5
