"""D14 regression: thread lifecycle on stop/start.

`stop()` joined the loop thread with a hard 10-second timeout and then
returned even if it was still alive; `start()` cleared the *shared* stop
Event without checking, so an immediate restart produced two live loop
threads. The stop path also wrote the "final" session/NAV snapshot without
waiting for a manually triggered cycle that still held the cycle lock.
"""

from __future__ import annotations

import threading
import time

import pytest

from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine


def _make_engine(tmp_path, monkeypatch, rebalance_interval: int = 300):
    engine = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=1_000_000),
        store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
        rebalance_interval=rebalance_interval,
    )
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    return engine


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_start_refuses_while_a_previous_loop_thread_is_alive(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    sleeper = threading.Thread(target=time.sleep, args=(1.2,), daemon=True)
    sleeper.start()
    engine._thread = sleeper

    try:
        with pytest.raises(RuntimeError, match="previous run loop"):
            engine.start()
    finally:
        sleeper.join(timeout=3)

    assert engine._running is False
    assert engine._thread is sleeper


def test_stop_waits_for_an_inflight_manual_cycle(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch, rebalance_interval=300)
    in_cycle = threading.Event()
    release = threading.Event()
    manual = threading.Thread(target=engine.run_once, daemon=True)
    engine.start()
    try:
        # Wait for the loop's first cycle to finish: a manual trigger while it
        # holds the lock is deliberately skipped, not queued.
        _wait_until(
            lambda: engine._cycle_count >= 1 and not engine._cycle_lock.locked()
        )

        def blocking_fetch():
            in_cycle.set()
            release.wait(timeout=10)

        monkeypatch.setattr(engine, "_fetch_prices", blocking_fetch)
        manual.start()
        assert in_cycle.wait(timeout=5), "manual cycle did not start"

        # Release the cycle 0.4s from now; stop() must not complete before
        # the cycle has released the cycle lock.
        threading.Timer(0.4, release.set).start()
        started = time.monotonic()
        engine.stop()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        if engine._thread is not None and engine._thread.is_alive():
            engine.stop()

    manual.join(timeout=5)
    assert not manual.is_alive()
    assert elapsed >= 0.3, (
        f"stop returned after {elapsed:.2f}s while a cycle still held the lock"
    )
    assert engine._thread is None or not engine._thread.is_alive()
    assert not engine._cycle_lock.locked()
