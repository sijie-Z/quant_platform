"""D1 regression: SimulatedBroker's check-then-act sequence must be atomic.

`place_order` validated available shares / cash and then mutated `_cash` and
`_positions` without any lock. Two concurrent full sells both passed the
availability check, credited cash twice and the loser crashed with KeyError
*after* crediting; two concurrent first buys of a symbol both wrote a fresh
`Position` (get()-then-assign), so only the last fill survived. The lock
widening helpers below make the old race deterministic instead of timing
dependent: the patched method sleeps *after* the check and before the
mutation, guaranteeing both threads are inside the critical section.
"""

from __future__ import annotations

import threading
import time

import pytest

from quant_platform.trading.broker import Order, OrderSide, OrderStatus, SimulatedBroker

BARRIER_TIMEOUT = 10.0
WIDEN_SECONDS = 0.15


def _widen_after_call(fn):
    """Call `fn`, then sleep, widening the window before the mutation."""

    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        time.sleep(WIDEN_SECONDS)
        return result

    return wrapped


def _run_concurrently(workers: list) -> tuple[list, list]:
    barrier = threading.Barrier(len(workers))
    results: list = []
    errors: list = []

    def runner(fn):
        try:
            barrier.wait(timeout=BARRIER_TIMEOUT)
            results.append(fn())
        except BaseException as exc:  # noqa: BLE001 - recorded for the assert
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(fn,)) for fn in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=BARRIER_TIMEOUT)
    assert not any(thread.is_alive() for thread in threads), "worker deadlocked"
    return results, errors


def test_concurrent_first_buys_sum_instead_of_overwriting():
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    # Sleep after the affordability check, before the position mutation.
    broker._max_affordable = _widen_after_call(broker._max_affordable)

    def buy():
        return broker.place_order(
            Order(code="600519", side=OrderSide.BUY, quantity=500, price=10.0)
        )

    results, errors = _run_concurrently([buy, buy])

    assert errors == [], f"concurrent buys raised: {errors!r}"
    assert all(r.status == OrderStatus.FILLED for r in results)
    positions = broker.get_positions()
    assert [p.code for p in positions] == ["600519"]
    # Both fills must be present: the old get()-then-assign kept only 500.
    assert positions[0].quantity == 1000
    # Commission: max(500 * 10 * 0.0003, 5) = 5 per order.
    assert broker.get_account()["cash"] == pytest.approx(1_000_000 - 2 * 5005)
    assert broker.get_account()["cash"] >= 0


def test_concurrent_full_sells_credit_cash_once():
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    broker.place_order(
        Order(code="600519", side=OrderSide.BUY, quantity=1000, price=10.0)
    )
    broker.new_trading_day()  # release T+1 so the shares are sellable
    cash_before = broker.get_account()["cash"]

    # Sleep after the book is resolved but before cash/position mutation, so
    # the old unlocked code had both threads past the availability check.
    broker._get_or_create_book = _widen_after_call(broker._get_or_create_book)

    def sell():
        return broker.place_order(
            Order(code="600519", side=OrderSide.SELL, quantity=1000, price=10.0)
        )

    results, errors = _run_concurrently([sell, sell])

    assert errors == [], f"concurrent sells raised: {errors!r}"
    statuses = sorted(r.status.value for r in results)
    assert statuses == ["filled", "rejected"], (
        "exactly one of two full sells may fill; the other must be rejected "
        f"against the reduced availability, got {statuses}"
    )
    # One sale: 10,000 gross - 5 commission - 10 stamp tax.
    assert broker.get_account()["cash"] == pytest.approx(cash_before + 9985)
    assert broker.get_positions() == []
    assert broker.get_account()["cash"] >= 0
