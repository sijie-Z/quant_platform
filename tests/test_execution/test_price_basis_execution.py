"""Execution/OMS price inputs must be raw CNY levels (D15).

An adjusted (qfq/hfq) basis is rejected before any cash, share or P&L
mutation, so an accidental research-panel feed cannot poison the book.
"""

from __future__ import annotations

import pytest

from quant_platform.execution.engine import ExecutionEngine
from quant_platform.execution.models import Order, OrderSide
from quant_platform.execution.oms import OrderManager, SimulatedExchange


def test_oms_rejects_adjusted_fill_and_preserves_book():
    oms = OrderManager(initial_cash=10_000.0)
    order = oms.create_order("600519.SH", "buy", 100)
    oms.submit_order(order.order_id)

    with pytest.raises(ValueError, match="raw"):
        oms.fill_order(order.order_id, price=10.0, price_basis="hfq")

    # Nothing was debited and no position was opened.
    assert oms.cash == pytest.approx(10_000.0)
    assert "600519.SH" not in oms.positions
    assert order.status.value == "submitted"

    oms.fill_order(order.order_id, price=10.0, price_basis="raw")
    assert oms.positions["600519.SH"].quantity == 100
    assert oms.price_basis == "raw"


def test_oms_update_prices_rejects_adjusted_marks():
    oms = OrderManager(initial_cash=10_000.0)
    order = oms.create_order("600519.SH", "buy", 100)
    oms.submit_order(order.order_id)
    oms.fill_order(order.order_id, price=10.0)

    with pytest.raises(ValueError, match="raw"):
        oms.update_prices({"600519.SH": 20.0}, price_basis="qfq")
    assert oms.positions["600519.SH"].current_price == pytest.approx(10.0)


def test_simulated_exchange_rejects_adjusted_market_prices():
    exchange = SimulatedExchange()
    exchange.set_order_manager(OrderManager(initial_cash=100_000.0))

    with pytest.raises(ValueError, match="raw"):
        exchange.update_market({"600519.SH": 10.0}, price_basis="hfq")
    with pytest.raises(ValueError, match="raw"):
        exchange.simulate_trading_day({"600519.SH": 10.0}, price_basis="qfq")


def test_execution_engine_rejects_adjusted_fill_and_marks():
    engine = ExecutionEngine(initial_cash=100_000.0)
    order = engine.create_order("600519.SH", OrderSide.BUY, 100)
    engine.submit_order(order)

    with pytest.raises(ValueError, match="raw"):
        engine.process_fill(order, price=10.0, quantity=100, price_basis="hfq")
    assert engine.cash == pytest.approx(100_000.0)
    assert engine.get_position("600519.SH") is None

    engine.process_fill(order, price=10.0, quantity=100)
    assert engine.get_position("600519.SH").quantity == 100

    with pytest.raises(ValueError, match="raw"):
        engine.mark_to_market({"600519.SH": 20.0}, price_basis="qfq")
    with pytest.raises(ValueError, match="raw"):
        engine.portfolio_snapshot({"600519.SH": 20.0}, price_basis="hfq")
    assert engine.get_position("600519.SH").current_price == pytest.approx(10.0)
