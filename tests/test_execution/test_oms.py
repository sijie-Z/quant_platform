"""Tests for execution.oms — Order Management System."""

import pytest

from quant_platform.execution.models import OrderStatus
from quant_platform.execution.oms import OrderManager, SimulatedExchange


class TestOrderManager:
    def setup_method(self):
        self.om = OrderManager(initial_cash=10_000_000)

    def test_create_order(self):
        order = self.om.create_order("600519", "buy", 100)
        assert order.ticker == "600519"
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_lot_size_rounding(self):
        order = self.om.create_order("600519", "buy", 150)
        assert order.quantity == 100  # rounded down to lot

    def test_invalid_quantity(self):
        with pytest.raises(ValueError):
            self.om.create_order("600519", "buy", 50)

    def test_submit_order(self):
        order = self.om.create_order("600519", "buy", 100)
        submitted = self.om.submit_order(order.order_id)
        assert submitted.status == OrderStatus.SUBMITTED

    def test_fill_order(self):
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        filled = self.om.fill_order(order.order_id, price=1800.0)
        assert filled.status == OrderStatus.FILLED
        assert "600519" in self.om.positions
        assert self.om.positions["600519"].quantity == 100

    def test_buy_updates_cash(self):
        initial_cash = self.om.cash
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1000.0)
        assert self.om.cash < initial_cash

    def test_sell_order(self):
        # Buy first
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1000.0)

        # Then sell
        sell_order = self.om.create_order("600519", "sell", 100)
        self.om.submit_order(sell_order.order_id)
        self.om.fill_order(sell_order.order_id, price=1100.0)
        assert "600519" not in self.om.positions

    def test_sell_insufficient_position(self):
        with pytest.raises(ValueError):
            self.om.create_order("600519", "sell", 100)

    def test_cancel_order(self):
        order = self.om.create_order("600519", "buy", 100)
        cancelled = self.om.cancel_order(order.order_id, reason="user cancel")
        assert cancelled.status == OrderStatus.CANCELLED

    def test_update_prices(self):
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1000.0)
        self.om.update_prices({"600519": 1100.0})
        assert self.om.positions["600519"].market_value == 110_000

    def test_get_snapshot(self):
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1000.0)
        self.om.update_prices({"600519": 1100.0})
        snap = self.om.get_snapshot()
        assert snap.n_positions == 1
        assert snap.total_value > 0

    def test_blotter(self):
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1000.0)
        blotter = self.om.get_order_blotter()
        assert len(blotter) == 1

    def test_tca(self):
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1000.0)
        tca = self.om.get_trade_cost_analysis()
        assert tca["total_orders"] == 1
        assert tca["total_commission"] > 0

    def test_commission_minimum(self):
        order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(order.order_id)
        filled = self.om.fill_order(order.order_id, price=1.0)  # Very cheap stock
        # Commission should be at least 5 RMB
        assert filled.fills[0].commission >= 5.0

    def test_stamp_tax_sell_only(self):
        buy_order = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(buy_order.order_id)
        self.om.fill_order(buy_order.order_id, price=1000.0)
        assert buy_order.fills[0].tax == 0.0

        sell_order = self.om.create_order("600519", "sell", 100)
        self.om.submit_order(sell_order.order_id)
        self.om.fill_order(sell_order.order_id, price=1000.0)
        assert sell_order.fills[0].tax > 0.0


class TestSimulatedExchange:
    def setup_method(self):
        self.exchange = SimulatedExchange()
        self.om = OrderManager(initial_cash=10_000_000)
        self.exchange.set_order_manager(self.om)

    def test_market_order_fill(self):
        order = self.om.create_order("600519", "buy", 100, order_type="market")
        self.om.submit_order(order.order_id)
        self.exchange.update_market({"600519": 1800.0})
        self.exchange.match_orders()
        assert order.status == OrderStatus.FILLED

    def test_limit_order_not_reached(self):
        order = self.om.create_order("600519", "buy", 100, order_type="limit", limit_price=1700.0)
        self.om.submit_order(order.order_id)
        self.exchange.update_market({"600519": 1800.0})
        self.exchange.match_orders()
        assert order.status == OrderStatus.SUBMITTED  # Not filled

    def test_limit_order_reached(self):
        order = self.om.create_order("600519", "buy", 100, order_type="limit", limit_price=1800.0)
        self.om.submit_order(order.order_id)
        self.exchange.update_market({"600519": 1800.0})
        self.exchange.match_orders()
        assert order.status == OrderStatus.FILLED

    def test_no_market_data_rejects(self):
        order = self.om.create_order("600519", "buy", 100, order_type="market")
        self.om.submit_order(order.order_id)
        self.exchange.match_orders()
        assert order.status == OrderStatus.REJECTED

    def test_simulate_trading_day(self):
        order = self.om.create_order("600519", "buy", 100, order_type="market")
        self.om.submit_order(order.order_id)
        self.exchange.simulate_trading_day({"600519": 1800.0})
        assert order.status == OrderStatus.FILLED
        assert "600519" in self.om.positions
