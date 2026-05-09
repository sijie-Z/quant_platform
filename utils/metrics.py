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
from collections import defaultdict
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

    def export_text(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []

        # Counters
        for key, c in self._counters.items():
            if c.name in self._help:
                lines.append(f"# HELP {c.name} {self._help[c.name]}")
            lines.append(f"# TYPE {c.name} counter")
            label_str = self._format_labels(c.labels)
            lines.append(f"{c.name}{label_str} {c.get()}")

        # Gauges
        for key, g in self._gauges.items():
            if g.name in self._help:
                lines.append(f"# HELP {g.name} {self._help[g.name]}")
            lines.append(f"# TYPE {g.name} gauge")
            label_str = self._format_labels(g.labels)
            lines.append(f"{g.name}{label_str} {g.get()}")

        # Histograms
        for key, h in self._histograms.items():
            if h.name in self._help:
                lines.append(f"# HELP {h.name} {self._help[h.name]}")
            lines.append(f"# TYPE {h.name} histogram")
            label_str = self._format_labels(h.labels)
            for bucket, count in h.counts.items():
                le = "+Inf" if bucket == float("inf") else str(bucket)
                bucket_labels = dict(h.labels)
                bucket_labels["le"] = le
                bl = self._format_labels(bucket_labels)
                lines.append(f"{h.name}_bucket{bl} {count}")
            lines.append(f"{h.name}_sum{label_str} {h.get_sum()}")
            lines.append(f"{h.name}_count{label_str} {h.get_count()}")

        return "\n".join(lines) + "\n"

    def get_snapshot(self) -> dict[str, Any]:
        """Get a JSON-friendly snapshot of all metrics."""
        snapshot = {}
        for key, c in self._counters.items():
            snapshot[c.name] = {"type": "counter", "value": c.get(), "labels": c.labels}
        for key, g in self._gauges.items():
            snapshot[g.name] = {"type": "gauge", "value": g.get(), "labels": g.labels}
        for key, h in self._histograms.items():
            snapshot[h.name] = {
                "type": "histogram",
                "sum": h.get_sum(),
                "count": h.get_count(),
                "labels": h.labels,
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
