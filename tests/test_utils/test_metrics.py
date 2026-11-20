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
        assert any(key.startswith("pipeline_stage_total") for key in snap)

    def test_decorator_error(self):
        m = get_metrics()
        m.reset()

        @instrument_pipeline_stage("fail_stage")
        def bad_func():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad_func()

        snap = m.get_snapshot()
        assert any(key.startswith("pipeline_stage_errors_total") for key in snap)


class TestConcurrentExport:
    """Export iterated the registry dicts while other threads registered new
    label sets -- "dictionary changed size during iteration" from the live
    /api/metrics endpoint in production."""

    def test_export_while_labels_are_added_does_not_raise(self):
        m = MetricsCollector()
        latency = m.histogram("latency_seconds")
        done = threading.Event()
        errors = []

        def writer():
            try:
                for i in range(2000):
                    m.counter("requests_total", labels={"path": f"/api/{i}"}).inc()
                    latency.observe(i / 1e6)
            finally:
                done.set()

        def reader():
            try:
                while not done.is_set():
                    m.export_text()
                    m.get_snapshot()
                for _ in range(5):
                    m.export_text()
                    m.get_snapshot()
            except Exception as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        writer_thread.start()
        reader_thread.start()
        writer_thread.join(timeout=30)
        reader_thread.join(timeout=30)

        assert not writer_thread.is_alive()
        assert not reader_thread.is_alive()
        assert errors == []

    def test_histogram_snapshot_is_internally_consistent(self):
        h = Histogram(name="h")
        for _ in range(10):
            h.observe(0.004)

        counts, total, count = h.snapshot()

        assert count == 10
        assert total == pytest.approx(0.04)
        assert counts[h.buckets[1]] == 10  # 0.005 bucket includes 0.004
        assert counts[float("inf")] == 10


class TestHTTPMiddleware:
    """No production code path recorded API metrics, so the Grafana API
    panels were always empty. install_http_middleware() is the wiring hook
    (call it from app.py before startup)."""

    def test_requests_and_latency_are_recorded(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from quant_platform.utils.metrics import install_http_middleware

        m = get_metrics()
        m.reset()

        app = FastAPI()

        @app.get("/ping")
        def ping():
            return {"ok": True}

        install_http_middleware(app)

        with TestClient(app) as client:
            assert client.get("/ping").status_code == 200

        text = m.export_text()
        assert "api_requests_total" in text
        assert 'status="200"' in text
        assert "api_duration_seconds_bucket" in text

        snap = m.get_snapshot()
        requests = [
            entry for key, entry in snap.items()
            if key.startswith("api_requests_total")
        ]
        assert len(requests) == 1
        assert requests[0]["value"] == 1.0
        durations = [
            entry for key, entry in snap.items()
            if key.startswith("api_duration_seconds")
        ]
        assert len(durations) == 1
        assert durations[0]["count"] == 1

    def test_route_template_is_used_as_the_label(self):
        """A path parameter must not create one label set per URL."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from quant_platform.utils.metrics import install_http_middleware

        m = get_metrics()
        m.reset()

        app = FastAPI()

        @app.get("/items/{item_id}")
        def item(item_id: int):
            return {"item_id": item_id}

        install_http_middleware(app)

        with TestClient(app) as client:
            client.get("/items/1")
            client.get("/items/2")

        text = m.export_text()
        assert 'path="/items/{item_id}"' in text
        assert 'path="/items/1"' not in text


class TestSnapshotKeyedByLabelSet:
    """S22: two label sets of one metric used to overwrite each other in the
    JSON snapshot, so /api/metrics/json reported only the last series."""

    def test_two_label_sets_both_survive(self):
        m = MetricsCollector()
        m.counter("api_requests_total", labels={"path": "/a"}).inc(2)
        m.counter("api_requests_total", labels={"path": "/b"}).inc(3)

        snap = m.get_snapshot()

        a = snap["api_requests_total{path=/a}"]
        b = snap["api_requests_total{path=/b}"]
        assert a["value"] == 2.0
        assert b["value"] == 3.0
        assert a["name"] == "api_requests_total"
        assert a["labels"] == {"path": "/a"}

    def test_gauges_and_histograms_are_keyed_by_label_set(self):
        m = MetricsCollector()
        m.gauge("risk_level", labels={"source": "a"}).set(1)
        m.gauge("risk_level", labels={"source": "b"}).set(2)
        m.histogram("latency", labels={"path": "/x"}).observe(0.1)

        snap = m.get_snapshot()

        assert snap["risk_level{source=a}"]["value"] == 1.0
        assert snap["risk_level{source=b}"]["value"] == 2.0
        assert snap["latency{path=/x}"]["count"] == 1


class TestExpositionFormat:
    """S22: label values must be escaped and HELP/TYPE emitted once."""

    def test_label_values_are_escaped(self):
        m = MetricsCollector()
        m.counter("weird_total", labels={"msg": 'a"b\\c\nd'}).inc()

        text = m.export_text()

        assert 'msg="a\\"b\\\\c\\nd"' in text

    def test_help_and_type_emitted_once_per_family(self):
        m = MetricsCollector()
        m.counter("requests_total", labels={"path": "/a"}, help="reqs").inc()
        m.counter("requests_total", labels={"path": "/b"}).inc()

        text = m.export_text()

        assert text.count("# TYPE requests_total counter") == 1
        assert text.count("# HELP requests_total reqs") == 1
        assert text.count("requests_total{") == 2
