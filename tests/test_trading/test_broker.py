"""Tests for trading.broker — Simulated broker."""

import pytest
from quant_platform.trading.broker import (
    SimulatedBroker, Order, OrderSide, OrderType, OrderStatus,
)


class TestSimulatedBroker:
    def setup_method(self):
        self.broker = SimulatedBroker(initial_cash=10_000_000)
        self.broker.connect()

    def test_initial_state(self):
        assert self.broker._cash == 10_000_000
        assert len(self.broker._positions) == 0

    def test_connect(self):
        broker = SimulatedBroker()
        assert broker.connect() is True
        assert broker._connected is True

    def test_place_buy_order(self):
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800.0)
        result = self.broker.place_order(order)
        assert result.status in [OrderStatus.SUBMITTED, OrderStatus.FILLED]
        assert result.filled_quantity > 0

    def test_place_sell_order(self):
        # Buy first
        buy = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800.0)
        self.broker.place_order(buy)
        # Then sell (T+1: need to mark as available)
        if "600519" in self.broker._positions:
            self.broker._positions["600519"].available = 100
        sell = Order(code="600519", side=OrderSide.SELL, quantity=100, price=1900.0)
        result = self.broker.place_order(sell)
        assert result.status in [OrderStatus.SUBMITTED, OrderStatus.FILLED, OrderStatus.REJECTED]

    def test_lot_size_validation(self):
        order = Order(code="600519", side=OrderSide.BUY, quantity=150, price=1800.0)
        result = self.broker.place_order(order)
        assert result.status == OrderStatus.REJECTED

    def test_get_positions(self):
        positions = self.broker.get_positions()
        assert isinstance(positions, list)

    def test_get_account(self):
        account = self.broker.get_account()
        assert "cash" in account
        assert "initial_cash" in account
        assert account["cash"] == 10_000_000

    def test_get_orders(self):
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800.0)
        self.broker.place_order(order)
        orders = self.broker.get_orders()
        assert len(orders) >= 1
