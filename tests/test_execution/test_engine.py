"""Tests for ExecutionEngine and Order state machine."""

import pytest
from datetime import datetime

from quant_platform.execution.engine import (
    ExecutionEngine,
    OrderSide,
    OrderStatus,
    validate_order_transition,
    transition_order,
    apply_fill,
    update_position,
)
from quant_platform.execution.models import Fill, Order, Position


class TestOrderFSM:
    def test_valid_transition(self):
        order = Order(status=OrderStatus.PENDING)
        valid, reason = validate_order_transition(order, OrderStatus.SUBMITTED)
        assert valid
        assert reason == ""

    def test_invalid_transition(self):
        order = Order(status=OrderStatus.FILLED)
        valid, reason = validate_order_transition(order, OrderStatus.SUBMITTED)
        assert not valid
        assert "Invalid transition" in reason

    def test_transition_raises_on_invalid(self):
        order = Order(status=OrderStatus.CANCELLED)
        with pytest.raises(ValueError, match="Invalid transition"):
            transition_order(order, OrderStatus.SUBMITTED)

    def test_transition_sets_submitted_at(self):
        order = Order(status=OrderStatus.PENDING)
        transition_order(order, OrderStatus.SUBMITTED)
        assert order.status == OrderStatus.SUBMITTED
        assert order.submitted_at is not None

    def test_terminal_states(self):
        for terminal in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            order = Order(status=terminal)
            valid, _ = validate_order_transition(order, OrderStatus.PENDING)
            assert not valid

    def test_partial_to_filled(self):
        order = Order(status=OrderStatus.PARTIAL)
        transition_order(order, OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None


class TestApplyFill:
    def test_full_fill(self):
        order = Order(quantity=100, status=OrderStatus.SUBMITTED)
        fill = Fill(price=10.0, quantity=100)
        apply_fill(order, fill)
        assert order.status == OrderStatus.FILLED
        assert len(order.fills) == 1
        assert order.filled_quantity == 100

    def test_partial_fill(self):
        order = Order(quantity=100, status=OrderStatus.SUBMITTED)
        fill = Fill(price=10.0, quantity=40)
        apply_fill(order, fill)
        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 40
        assert order.remaining_quantity == 60

    def test_multiple_fills(self):
        order = Order(quantity=100, status=OrderStatus.SUBMITTED)
        apply_fill(order, Fill(price=10.0, quantity=30))
        apply_fill(order, Fill(price=11.0, quantity=70))
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.avg_fill_price == 10.7  # (30*10 + 70*11) / 100


class TestUpdatePosition:
    def test_buy_updates_position(self):
        pos = Position(ticker="600519")
        order = Order(ticker="600519", side=OrderSide.BUY, quantity=100)
        fill = Fill(price=150.0, quantity=100)
        update_position(pos, order, fill)
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0

    def test_sell_realizes_pnl(self):
        pos = Position(ticker="600519", quantity=100, avg_cost=150.0)
        order = Order(ticker="600519", side=OrderSide.SELL, quantity=50)
        fill = Fill(price=170.0, quantity=50)
        update_position(pos, order, fill)
        assert pos.quantity == 50
        assert pos.realized_pnl == 1000.0  # (170-150)*50


class TestExecutionEngine:
    def test_create_order(self):
        engine = ExecutionEngine()
        order = engine.create_order("600519", OrderSide.BUY, 100)
        assert order.ticker == "600519"
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_submit_order(self):
        engine = ExecutionEngine()
        order = engine.create_order("600519", OrderSide.BUY, 100)
        engine.submit_order(order)
        assert order.status == OrderStatus.SUBMITTED

    def test_process_fill_updates_position(self):
        engine = ExecutionEngine()
        order = engine.create_order("600519", OrderSide.BUY, 100)
        engine.submit_order(order)
        engine.process_fill(order, price=150.0, quantity=100)
        assert order.status == OrderStatus.FILLED
        pos = engine.get_position("600519")
        assert pos is not None
        assert pos.quantity == 100

    def test_cancel_order(self):
        engine = ExecutionEngine()
        order = engine.create_order("600519", OrderSide.BUY, 100)
        engine.submit_order(order)
        engine.cancel_order(order)
        assert order.status == OrderStatus.CANCELLED

    def test_portfolio_snapshot(self):
        engine = ExecutionEngine()
        o1 = engine.create_order("600519", OrderSide.BUY, 100)
        engine.submit_order(o1)
        engine.process_fill(o1, price=150.0, quantity=100)

        snapshot = engine.portfolio_snapshot({"600519": 155.0})
        assert snapshot["n_positions"] == 1
        assert snapshot["total_unrealized_pnl"] == 500.0  # (155-150)*100
