"""Tests for Prometheus-compatible metrics collector."""

import threading
import time

import pytest

from quant_platform.utils.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    Timer,
    get_metrics,
    instrument_pipeline_stage,
)


class TestCounter:
    def test_inc(self):
        c = Counter(name="test")
        c.inc()
        assert c.get() == 1.0

    def test_inc_amount(self):
        c = Counter(name="test")
        c.inc(5.0)
        assert c.get() == 5.0

    def test_labels(self):
        c = Counter(name="test", labels={"env": "prod"})
        assert c.labels == {"env": "prod"}


class TestGauge:
    def test_set(self):
        g = Gauge(name="test")
        g.set(42)
        assert g.get() == 42.0

    def test_inc_dec(self):
        g = Gauge(name="test")
        g.inc(10)
        g.dec(3)
        assert g.get() == 7.0


class TestHistogram:
    def test_observe(self):
        h = Histogram(name="test")
        h.observe(0.05)
        assert h.get_count() == 1
        assert h.get_sum() == pytest.approx(0.05)

    def test_buckets(self):
        h = Histogram(name="test")
        h.observe(0.005)
        h.observe(0.15)
        h.observe(5.0)
        assert h.get_count() == 3

    def test_default_buckets(self):
        h = Histogram(name="test")
        assert len(h.buckets) == 12


class TestMetricsCollector:
    def test_counter_create(self):
        m = MetricsCollector()
        c = m.counter("requests_total", help="Total requests")
        c.inc()
        assert c.get() == 1.0

    def test_gauge_create(self):
        m = MetricsCollector()
        g = m.gauge("queue_depth", help="Queue depth")
        g.set(5)
        assert g.get() == 5.0

    def test_histogram_create(self):
        m = MetricsCollector()
        h = m.histogram("duration", help="Request duration")
        h.observe(0.1)
        assert h.get_count() == 1

    def test_singleton_same(self):
        m = MetricsCollector()
        c1 = m.counter("test")
        c2 = m.counter("test")
        assert c1 is c2

    def test_export_text(self):
        m = MetricsCollector()
        m.counter("test_counter", help="A test counter").inc(3)
        m.gauge("test_gauge", help="A test gauge").set(42)

        text = m.export_text()
        assert "test_counter" in text
        assert "test_gauge" in text
        assert "3" in text
        assert "42" in text
        assert "# HELP test_counter" in text

    def test_export_with_labels(self):
        m = MetricsCollector()
        m.counter("http_requests", labels={"method": "GET"}).inc()
        text = m.export_text()
        assert 'method="GET"' in text

    def test_export_histogram(self):
        m = MetricsCollector()
        h = m.histogram("latency")
        h.observe(0.05)
        text = m.export_text()
        assert "latency_bucket" in text
        assert "latency_sum" in text
        assert "latency_count" in text

    def test_get_snapshot(self):
        m = MetricsCollector()
        m.counter("c1").inc()
        m.gauge("g1").set(10)
        snap = m.get_snapshot()
        assert "c1" in snap
        assert "g1" in snap
        assert snap["c1"]["value"] == 1.0
        assert snap["g1"]["value"] == 10.0

    def test_reset(self):
        m = MetricsCollector()
        m.counter("c1").inc(5)
        m.reset()
        snap = m.get_snapshot()
        assert len(snap) == 0

    def test_thread_safety(self):
        m = MetricsCollector()
        c = m.counter("concurrent")

        def increment():
            for _ in range(1000):
                c.inc()

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert c.get() == 10000.0


class TestTimer:
    def test_timer(self):
        h = Histogram(name="test")
        with Timer(h):
            time.sleep(0.01)
        assert h.get_count() == 1
        assert h.get_sum() >= 0.01


class TestGetMetrics:
    def test_singleton(self):
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2


class TestInstrumentDecorator:
    def test_decorator_success(self):
        m = get_metrics()
        m.reset()

        @instrument_pipeline_stage("test_stage")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

        snap = m.get_snapshot()
        assert "pipeline_stage_total" in snap

    def test_decorator_error(self):
        m = get_metrics()
        m.reset()

        @instrument_pipeline_stage("fail_stage")
        def bad_func():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad_func()

        snap = m.get_snapshot()
        assert "pipeline_stage_errors_total" in snap
