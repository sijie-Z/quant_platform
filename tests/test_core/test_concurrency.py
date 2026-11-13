"""Concurrency regression tests for shared mutable state.

Covers:
- Store: concurrent partial upserts of one order / concurrent fills of one order
- EventBus (the global AsyncEventBus from ``get_event_bus()``): publish while
  subscribe/unsubscribe churn, duplicate/lost delivery and consumer-task leaks
- RiskMonitor: update_portfolio_state + check_pre_trade concurrency
- PipelineCache: concurrent save_stage/load_stage must never expose torn data
- LiveRunner: two runners share the process-wide RiskMonitor kill switch

These tests intentionally use plain OS threads (the production consumers are
threads/executors, not asyncio tasks).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from quant_platform.core.events import get_event_bus
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import (
    RiskLevel,
    RiskLimits,
    RiskMonitor,
)
from quant_platform.utils.cache import PipelineCache

# ── Store.save_order: concurrent partial upserts ──────────────────────────


def test_store_save_order_concurrent_partial_upserts_keep_every_column(tmp_path):
    """8 threads upserting the same order with different column subsets.

    Each thread owns exactly one optional column; the final row must still
    carry every thread's last value. ``INSERT OR REPLACE``-style semantics
    (rebuilding the row from one caller's dict) would silently reset the
    other callers' columns.
    """
    store = Store(str(tmp_path / "orders.db"))
    order_id = "ord-concurrent-upsert"
    base = {
        "order_id": order_id,
        "code": "600519",
        "side": "buy",
        "quantity": 100,
        "price": 10.0,
    }
    owned_fields = {
        "status": "partial",
        "commission": 1.25,
        "tax": 0.25,
        "slippage": 0.5,
        "broker_order_id": "broker-1",
        "error_msg": "retry",
        "strategy_id": "strategy-1",
        "signal_id": "signal-1",
    }
    thread_count = len(owned_fields)
    barrier = threading.Barrier(thread_count)
    errors: list[str] = []

    def worker(field: str, value):
        try:
            barrier.wait(timeout=10)
            for _ in range(25):
                payload = dict(base)
                payload[field] = value
                store.save_order(payload)
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(f"{field}: {exc!r}")

    threads = [
        threading.Thread(target=worker, args=(field, value))
        for field, value in owned_fields.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == [], f"save_order raised under concurrency: {errors}"
    rows = store.get_orders(limit=10)
    assert len(rows) == 1, f"expected exactly one order row, got {len(rows)}"
    row = rows[0]
    assert row["order_id"] == order_id
    for field, value in owned_fields.items():
        assert row[field] == value, (
            f"column {field!r} was lost/reset by a concurrent partial upsert: "
            f"expected {value!r}, row has {row[field]!r}"
        )


# ── Store.save_trade: concurrent fills of one order ───────────────────────


def test_store_save_trade_concurrent_same_order(tmp_path):
    """8 concurrent fills for one order, unique caller-supplied trade_ids."""
    store = Store(str(tmp_path / "trades.db"))
    store.save_order({
        "order_id": "ord-1", "code": "600519", "side": "buy",
        "quantity": 80, "price": 10.0,
    })
    thread_count = 8
    barrier = threading.Barrier(thread_count)
    errors: list[str] = []

    def worker(i: int):
        try:
            barrier.wait(timeout=10)
            store.save_trade({
                "trade_id": f"trade-{i}",
                "order_id": "ord-1",
                "code": "600519",
                "side": "buy",
                "quantity": 10,
                "price": 10.0,
            })
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(f"trade-{i}: {exc!r}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == [], f"save_trade raised under concurrency: {errors}"
    trades = store.get_trades(limit=100)
    assert len(trades) == thread_count, (
        f"expected {thread_count} trade rows, got {len(trades)}"
    )
    assert {t["trade_id"] for t in trades} == {
        f"trade-{i}" for i in range(thread_count)
    }


# ── EventBus: publish while subscribe/unsubscribe churn ───────────────────


async def _noop() -> None:
    """Coroutine used as a loop barrier/flush point."""


def _flush_loop(loop: asyncio.AbstractEventLoop, timeout: float = 10.0) -> None:
    """Wait until already-queued loop callbacks have run.

    ``run_coroutine_threadsafe`` schedules the task creation via
    ``call_soon_threadsafe``, which is FIFO with the subscribe/unsubscribe
    callbacks queued by this same thread.
    """
    asyncio.run_coroutine_threadsafe(_noop(), loop).result(timeout=timeout)


def test_event_bus_publish_during_subscribe_unsubscribe_churn():
    """A stable subscriber must see every event exactly once while a second
    subscription churns, and consumer tasks must return to their baseline."""
    bus = get_event_bus()
    topic = "concurrency.churn"

    received: list[int] = []
    received_lock = threading.Lock()

    def stable_handler(event) -> None:
        with received_lock:
            received.append(event.data["seq"])

    def churn_handler(event) -> None:
        pass

    bus.subscribe(topic, stable_handler, name="stable-concurrency-handler")

    own_loop: asyncio.AbstractEventLoop | None = None
    loop_thread: threading.Thread | None = None
    loop = bus._loop if bus.is_started else None
    if loop is None:
        own_loop = asyncio.new_event_loop()
        loop = own_loop
        started = threading.Event()

        def run_loop() -> None:
            asyncio.set_event_loop(own_loop)
            own_loop.run_until_complete(bus.start())
            started.set()
            own_loop.run_forever()

        loop_thread = threading.Thread(target=run_loop, daemon=True)
        loop_thread.start()
        assert started.wait(timeout=10), "event bus loop failed to start"

    try:
        _flush_loop(loop)
        baseline = len(bus._consumer_tasks)
        assert baseline >= 1, "stable subscriber has no consumer task"

        total = 2000
        publish_errors: list[str] = []

        def publisher(start: int, count: int) -> None:
            try:
                for seq in range(start, start + count):
                    bus.publish(topic, {"seq": seq})
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                publish_errors.append(repr(exc))

        publishers = [
            threading.Thread(target=publisher, args=(0, total // 2)),
            threading.Thread(target=publisher, args=(total // 2, total // 2)),
        ]
        for thread in publishers:
            thread.start()

        # Churn: subscribe + unsubscribe 100 times while publishing.
        for _ in range(100):
            bus.subscribe(topic, churn_handler)
            bus.unsubscribe(topic, churn_handler)

        for thread in publishers:
            thread.join(timeout=30)
        assert not any(t.is_alive() for t in publishers), "publisher thread hung"

        deadline = time.time() + 15
        while time.time() < deadline:
            with received_lock:
                if len(received) >= total:
                    break
            time.sleep(0.01)

        _flush_loop(loop)

        with received_lock:
            got = list(received)

        assert publish_errors == [], (
            f"publish() raised during subscribe/unsubscribe churn: {publish_errors[:5]}"
        )
        assert len(got) == total, (
            f"lost events: {total - len(got)} of {total} never reached the "
            f"stable handler"
        )
        assert len(set(got)) == total, (
            f"duplicate deliveries: {len(got) - len(set(got))} duplicate events"
        )
        assert set(got) == set(range(total))
        assert len(bus._consumer_tasks) == baseline, (
            "consumer tasks leaked across 100 subscribe/unsubscribe cycles: "
            f"{len(bus._consumer_tasks)} vs baseline {baseline}"
        )
    finally:
        bus.unsubscribe(topic, stable_handler)
        bus.unsubscribe(topic, churn_handler)
        if own_loop is not None:
            try:
                if bus.is_started:
                    asyncio.run_coroutine_threadsafe(bus.stop(), own_loop).result(
                        timeout=10
                    )
            finally:
                own_loop.call_soon_threadsafe(own_loop.stop)
                if loop_thread is not None:
                    loop_thread.join(timeout=10)
                own_loop.close()


# ── RiskMonitor: concurrent portfolio updates + pre-trade checks ──────────


def test_risk_monitor_concurrent_state_and_pre_trade_checks():
    """4 threads x 200 iterations of update_portfolio_state + check_pre_trade.

    The portfolio state is constant and breached (5% daily loss), so a sane
    breach log stays at a small constant -- state breaches are supposed to be
    deduplicated ("同状态重复不追加"). Any growth proportional to the 800
    calls means the log grows without bound in production.
    """
    # The monitor logs a warning per breach; silence the flood for this test.
    import quant_platform.risk.circuit_breaker as circuit_breaker

    old_level = circuit_breaker.logger.level
    circuit_breaker.logger.setLevel(logging.CRITICAL)
    try:
        monitor = RiskMonitor(RiskLimits(
            max_orders_per_minute=10**9,
            max_orders_per_second=10**9,
        ))
        portfolio_value = 1_000_000.0
        daily_pnl = -50_000.0  # 5% > 3% daily-loss limit
        state = (portfolio_value, daily_pnl, {}, {})

        # Sanity: repeated identical updates are deduplicated when serial.
        for _ in range(10):
            monitor.update_portfolio_state(*state)
        assert len(monitor.breaches) == 1, (
            f"serial same-state updates appended duplicates: "
            f"{len(monitor.breaches)} breaches"
        )

        thread_count = 4
        iterations = 200
        barrier = threading.Barrier(thread_count)
        errors: list[str] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=10)
                for _ in range(iterations):
                    monitor.update_portfolio_state(*state)
                    monitor.check_pre_trade({
                        "ticker": "600519",
                        "side": "buy",
                        "quantity": 100,
                        "price": 10.0,
                    })
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == [], f"risk monitor raised under concurrency: {errors}"
        # Allow a small constant for the signature-check race between the 4
        # threads; anything proportional to 800 calls is unbounded growth.
        assert len(monitor.breaches) <= 16, (
            f"breach log grew without bound: {len(monitor.breaches)} entries "
            f"after {thread_count * iterations} pre-trade checks in one "
            f"constant breach state (state breach dedup alone keeps this at "
            f"1; update_portfolio_state and check_pre_trade share the list)"
        )
    finally:
        circuit_breaker.logger.setLevel(old_level)


# ── PipelineCache: concurrent save_stage/load_stage ───────────────────────


def test_pipeline_cache_concurrent_save_and_load_never_tears(tmp_path):
    """8 writers + 8 readers on one stage key.

    ``load_stage`` may return ``None`` (miss/corrupt file removed) but must
    never raise and never return a partially-written or two-writers-mixed
    value; each writer's payload is a self-consistent 3-tuple.
    """
    cache = PipelineCache(tmp_path / "cache")
    payload_size = 256 * 1024
    payloads = {
        i: ("payload", i, bytes([i]) * payload_size) for i in range(8)
    }
    thread_count = 16
    barrier = threading.Barrier(thread_count)
    errors: list[str] = []
    malformed: list[str] = []

    def writer(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(15):
                cache.save_stage("data", "same-key", payloads[i])
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(f"save_stage[{i}]: {exc!r}")

    def reader(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(80):
                value = cache.load_stage("data", "same-key")
                if value is None:
                    continue  # documented miss / removed corrupt file
                valid = (
                    isinstance(value, tuple)
                    and len(value) == 3
                    and value[0] == "payload"
                    and value[1] in payloads
                    and value[2] == payloads[value[1]][2]
                )
                if not valid:
                    malformed.append(f"reader[{i}]: {value!r:.160}")
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(f"load_stage[{i}]: {exc!r}")

    threads = (
        [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        + [threading.Thread(target=reader, args=(i,)) for i in range(8)]
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == [], (
        f"cache save/load raised under concurrency: {errors[:5]}"
    )
    assert malformed == [], (
        f"load_stage returned torn/mixed data from a concurrent save: "
        f"{malformed[:5]}"
    )

    # With all writers finished, one last read must be a complete payload
    # (or a documented miss if a racing reader removed the file).
    final = cache.load_stage("data", "same-key")
    if final is not None:
        assert final in payloads.values(), (
            f"final cache entry is torn: {final!r:.160}"
        )


# ── LiveRunner: process-wide RiskMonitor singleton ────────────────────────


def test_live_runners_share_process_wide_kill_switch(monkeypatch, tmp_path):
    """Two LiveRunner instances built without an injected monitor must act on
    the same RiskMonitor: activating the kill switch through one runner blocks
    buys through the other."""
    import quant_platform.trading.live_runner as live_runner_module
    from quant_platform.trading.live_runner import LiveRunner

    # Keep the runners' Store off the repository's data/trading.db.
    monkeypatch.setattr(
        live_runner_module,
        "Store",
        lambda: Store(str(tmp_path / "live.db")),
    )

    # The process-wide monitor must never persist test state into the real
    # runtime DB (data/trading.db) — that is a one-way pollution channel and
    # once armed a KILL switch blocked every later live run. Give this test
    # its own monitor/store and route the module's default resolution to it.
    monitor = RiskMonitor(store=Store(str(tmp_path / "risk_state.db")))
    monkeypatch.setattr(
        live_runner_module, "get_risk_monitor", lambda: monitor,
    )

    runner_a = LiveRunner(broker_type="simulated", dual_track=False)
    runner_b = LiveRunner(broker_type="simulated", dual_track=False)
    assert runner_a is not runner_b
    assert runner_a._risk is runner_b._risk, (
        "runners built without an injected monitor do not share one "
        "RiskMonitor instance"
    )
    assert runner_a._risk is monitor

    runner_a._risk.activate_kill_switch("concurrency test")
    approved, breaches = runner_b._risk.check_pre_trade({
        "ticker": "600519",
        "side": "buy",
        "quantity": 100,
        "price": 10.0,
    })
    assert approved is False, (
        "kill switch activated via runner A did not block a buy through "
        "runner B"
    )
    assert any(b.severity == RiskLevel.KILL for b in breaches), (
        f"expected a KILL-severity breach, got {breaches!r}"
    )
    assert any("KILL SWITCH" in b.message for b in breaches)
