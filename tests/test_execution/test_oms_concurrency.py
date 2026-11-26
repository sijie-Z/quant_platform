"""Concurrency regression tests for execution.oms.OrderManager.

``OrderManager.fill_order`` validates against and then mutates shared state
(order.remaining_quantity, self.cash, self.positions) with no lock, so two
fills on the same order can both pass validation. The test below widens that
check-then-act window at the exact point the production code allocates its
``Fill`` (after quantity clamping and the cash/position validation, before
``order.fills.append``) so the interleaving is deterministic instead of
depending on scheduler luck.
"""

from __future__ import annotations

import threading

import quant_platform.execution.oms as oms_module
from quant_platform.execution.models import Fill
from quant_platform.execution.oms import OrderManager


class _BarrierFill(Fill):
    """Fill whose construction rendezvouses two racing fills mid-flight.

    With a correctly serialized ``fill_order`` only one thread can be at this
    point at a time; the barrier then times out and is ignored.
    """

    barrier: threading.Barrier | None = None

    def __init__(self, **kwargs):
        barrier = type(self).barrier
        if barrier is not None:
            try:
                barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                pass  # serialized implementation: nothing else is in flight
        super().__init__(**kwargs)


def test_concurrent_fills_on_one_order_cannot_overfill_or_overdraw(monkeypatch):
    """Two threads each fill 60 shares of a 100-share order.

    The book starts with 700 RMB cash, enough for exactly one 60-share fill
    (600 + 5 minimum commission). Invariants: filled quantity <= 100,
    cash >= 0, position >= 0.
    """
    _BarrierFill.barrier = threading.Barrier(2)
    monkeypatch.setattr(oms_module, "Fill", _BarrierFill)
    try:
        manager = OrderManager(initial_cash=700.0)
        order = manager.create_order("600519", "buy", 100)
        manager.submit_order(order.order_id)

        errors: list[BaseException] = []

        def fill_once() -> None:
            try:
                manager.fill_order(order.order_id, price=10.0, quantity=60)
            except ValueError as exc:
                # A serialized implementation may legitimately reject the
                # second fill (insufficient cash / no remaining quantity).
                errors.append(exc)
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(exc)

        threads = [threading.Thread(target=fill_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "fill thread hung"

        filled = order.filled_quantity
        cash = manager.cash
        position = manager.positions.get("600519")
        held = position.quantity if position is not None else 0
        fill_sizes = [f.quantity for f in order.fills]

        assert filled <= order.quantity, (
            f"concurrent fills overfilled the order: {filled} > "
            f"{order.quantity} (fill sizes {fill_sizes})"
        )
        assert cash >= 0, (
            f"concurrent fills drove cash negative: {cash:.2f} "
            f"(fill sizes {fill_sizes}, initial cash 700.00)"
        )
        assert held >= 0, f"position went negative: {held} (fill sizes {fill_sizes})"

        unexpected = [e for e in errors if not isinstance(e, ValueError)]
        assert not unexpected, f"unexpected errors under concurrent fills: {unexpected!r}"
    finally:
        _BarrierFill.barrier = None
