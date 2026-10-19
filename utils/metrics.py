"""Prometheus-compatible metrics for production monitoring.

Lightweight in-process metrics collector — no external dependency.
Exports in Prometheus text format via /metrics endpoint.

Metric types:
- Counter: monotonically increasing (requests, errors, trades)
- Gauge: current value (positions, latency, factor IC)
- Histogram: distribution (request duration, slippage, P&L)

Usage:
    from quant_platform.utils.metrics import get_metrics

    m = get_metrics()
    m.counter("api_requests_total", labels={"endpoint": "/api/run"}).inc()
    m.gauge("active_positions", labels={"strategy": "main"}).set(15)
    m.histogram("request_duration_seconds").observe(0.25)
    print(m.export_text())
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Counter:
    """Monotonically increasing counter."""
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0):
        with self._lock:
            self.value += amount

    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class Gauge:
    """Gauge that can go up and down."""
    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float):
        with self._lock:
            self.value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self.value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self.value -= amount

    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class Histogram:
    """Histogram for tracking distributions."""
    name: str
    buckets: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    counts: dict[float, int] = field(default_factory=dict)
    _sum: float = 0.0
    _count: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        for b in self.buckets:
            self.counts[b] = 0
        self.counts[float("inf")] = 0

    def observe(self, value: float):
        with self._lock:
            self._sum += value
            self._count += 1
            for b in self.buckets:
                if value <= b:
                    self.counts[b] += 1
            self.counts[float("inf")] += 1

    def get_sum(self) -> float:
        with self._lock:
            return self._sum

    def get_count(self) -> int:
        with self._lock:
            return self._count

    def snapshot(self) -> tuple[dict[float, int], float, int]:
        """Consistent (counts, sum, count) copy taken under the lock.

        Exporting iterated ``counts`` directly while ``observe`` incremented
        it, which could raise "dictionary changed size during iteration" (a
        new bucket appearing mid-loop) or emit torn bucket counts.
        """
        with self._lock:
            return dict(self.counts), self._sum, self._count


class MetricsCollector:
    """Thread-safe Prometheus-compatible metrics collector.

    Usage:
        m = MetricsCollector()
        m.counter("requests").inc()
        m.gauge("queue_depth").set(42)
        m.histogram("latency").observe(0.05)
        print(m.export_text())
    """

    def __init__(self):
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._help: dict[str, str] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, labels: dict[str, str] | None = None, help: str = "") -> Counter:
        """Get or create a counter metric."""
        key = self._make_key(name, labels)
        if key not in self._counters:
            with self._lock:
                if key not in self._counters:
                    self._counters[key] = Counter(name=name, labels=labels or {})
                    if name not in self._help:
                        self._help[name] = help
        return self._counters[key]

    def gauge(self, name: str, labels: dict[str, str] | None = None, help: str = "") -> Gauge:
        """Get or create a gauge metric."""
        key = self._make_key(name, labels)
        if key not in self._gauges:
            with self._lock:
                if key not in self._gauges:
                    self._gauges[key] = Gauge(name=name, labels=labels or {})
                    if name not in self._help:
                        self._help[name] = help
        return self._gauges[key]

    def histogram(self, name: str, labels: dict[str, str] | None = None, help: str = "") -> Histogram:
        """Get or create a histogram metric."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            with self._lock:
                if key not in self._histograms:
                    self._histograms[key] = Histogram(name=name, labels=labels or {})
                    if name not in self._help:
                        self._help[name] = help
        return self._histograms[key]

    def _snapshot_metrics(
        self,
    ) -> tuple[list[Counter], list[Gauge], list[Histogram], dict[str, str]]:
        """Take a consistent copy of the metric registries.

        Metric creation inserts into these dicts under ``_lock``; iterating
        them live meant a thread registering a new label while another
        exported could raise "dictionary changed size during iteration".
        """
        with self._lock:
            counters = list(self._counters.values())
            gauges = list(self._gauges.values())
            histograms = list(self._histograms.values())
            help_map = dict(self._help)
        return counters, gauges, histograms, help_map

    def export_text(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        counters, gauges, histograms, help_map = self._snapshot_metrics()

        # Counters
        for c in counters:
            if c.name in help_map:
                lines.append(f"# HELP {c.name} {help_map[c.name]}")
            lines.append(f"# TYPE {c.name} counter")
            label_str = self._format_labels(c.labels)
            lines.append(f"{c.name}{label_str} {c.get()}")

        # Gauges
        for g in gauges:
            if g.name in help_map:
                lines.append(f"# HELP {g.name} {help_map[g.name]}")
            lines.append(f"# TYPE {g.name} gauge")
            label_str = self._format_labels(g.labels)
            lines.append(f"{g.name}{label_str} {g.get()}")

        # Histograms
        for h in histograms:
            if h.name in help_map:
                lines.append(f"# HELP {h.name} {help_map[h.name]}")
            lines.append(f"# TYPE {h.name} histogram")
            label_str = self._format_labels(h.labels)
            counts, total_sum, total_count = h.snapshot()
            for bucket, count in counts.items():
                le = "+Inf" if bucket == float("inf") else str(bucket)
                bucket_labels = dict(h.labels)
                bucket_labels["le"] = le
                bl = self._format_labels(bucket_labels)
                lines.append(f"{h.name}_bucket{bl} {count}")
            lines.append(f"{h.name}_sum{label_str} {total_sum}")
            lines.append(f"{h.name}_count{label_str} {total_count}")

        return "\n".join(lines) + "\n"

    def get_snapshot(self) -> dict[str, Any]:
        """Get a JSON-friendly snapshot of all metrics."""
        snapshot = {}
        counters, gauges, histograms, _ = self._snapshot_metrics()
        for c in counters:
            snapshot[c.name] = {"type": "counter", "value": c.get(), "labels": dict(c.labels)}
        for g in gauges:
            snapshot[g.name] = {"type": "gauge", "value": g.get(), "labels": dict(g.labels)}
        for h in histograms:
            _, total_sum, total_count = h.snapshot()
            snapshot[h.name] = {
                "type": "histogram",
                "sum": total_sum,
                "count": total_count,
                "labels": dict(h.labels),
            }
        return snapshot

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    @staticmethod
    def _make_key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _format_labels(labels: dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return "{" + pairs + "}"


# Global singleton
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


class Timer:
    """Context manager that times a block and records to a histogram.

    Usage:
        with Timer(metrics.histogram("api_duration"), labels={"endpoint": "/run"}):
            do_work()
    """

    def __init__(self, histogram: Histogram, labels: dict[str, str] | None = None):
        self.histogram = histogram
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.histogram.observe(elapsed)


def instrument_pipeline_stage(stage_name: str):
    """Decorator that instruments a pipeline stage with metrics.

    Records:
    - Counter for invocations
    - Histogram for duration
    - Counter for errors
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            m = get_metrics()
            m.counter("pipeline_stage_total", labels={"stage": stage_name}).inc()
            with Timer(m.histogram("pipeline_stage_duration_seconds", labels={"stage": stage_name})):
                try:
                    result = func(*args, **kwargs)
                    m.counter("pipeline_stage_success_total", labels={"stage": stage_name}).inc()
                    return result
                except Exception as e:
                    m.counter("pipeline_stage_errors_total", labels={"stage": stage_name, "error": type(e).__name__}).inc()
                    raise
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


class HTTPMetricsMiddleware:
    """Pure-ASGI middleware recording request counts and latency.

    Emits the metric names the Grafana dashboard already queries:
    - ``api_requests_total{method,path,status}`` (counter)
    - ``api_errors_total{method,path,error}`` (counter, unhandled exceptions)
    - ``api_duration_seconds{method,path}`` (histogram)

    The ``path`` label uses the matched route template (``/api/run/{id}``)
    when Starlette resolved one, so a path parameter does not create an
    unbounded label set.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_holder = {"status": 0}
        error_name: str | None = None

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 0))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            error_name = type(exc).__name__
            raise
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            path = getattr(route, "path", None) or scope.get("path", "unknown")
            method = scope.get("method", "UNKNOWN")
            status = status_holder["status"] or (500 if error_name else 0)

            m = get_metrics()
            m.counter(
                "api_requests_total",
                labels={"method": method, "path": path, "status": str(status)},
                help="HTTP requests handled by the API",
            ).inc()
            m.histogram(
                "api_duration_seconds",
                labels={"method": method, "path": path},
                help="HTTP request latency in seconds",
            ).observe(elapsed)
            if error_name is not None:
                m.counter(
                    "api_errors_total",
                    labels={"method": method, "path": path, "error": error_name},
                    help="Unhandled HTTP handler exceptions",
                ).inc()


def install_http_middleware(app):
    """Wire HTTP request metrics into a FastAPI/Starlette (or bare ASGI) app.

    This module owns the instrumentation because ``app.py`` and
    ``api/routes.py`` are outside the current change set. Wiring is one line,
    at application construction time (before startup)::

        from quant_platform.utils.metrics import install_http_middleware
        install_http_middleware(app)

    For a Starlette app the middleware is registered in place and the app is
    returned; for a bare ASGI callable it returns the wrapped app.
    """
    add_middleware = getattr(app, "add_middleware", None)
    if callable(add_middleware):
        add_middleware(HTTPMetricsMiddleware)
        return app
    return HTTPMetricsMiddleware(app)


@contextmanager
def pipeline_stage(stage_name: str):
    """Instrument one pipeline stage with the Grafana metric names.

    ``pipeline_stage_total`` / ``pipeline_stage_duration_seconds`` /
    ``pipeline_stage_errors_total`` are what the shipped dashboard queries;
    without a caller they were permanently empty.
    """
    m = get_metrics()
    start = time.perf_counter()
    m.counter("pipeline_stage_total", labels={"stage": stage_name}).inc()
    try:
        yield
        m.counter(
            "pipeline_stage_success_total", labels={"stage": stage_name},
        ).inc()
    except Exception as exc:
        m.counter(
            "pipeline_stage_errors_total",
            labels={"stage": stage_name, "error": type(exc).__name__},
        ).inc()
        raise
    finally:
        m.histogram(
            "pipeline_stage_duration_seconds", labels={"stage": stage_name},
        ).observe(time.perf_counter() - start)
