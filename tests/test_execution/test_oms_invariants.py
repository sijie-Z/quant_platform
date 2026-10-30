"""OMS invariants: fills never overfill an order and sells pay their fees.

Regression for two audit survivors:
- ``qty = min(qty, order.remaining_quantity)`` was removed, so one print of
  300 shares completed a 100-share order (position, cash and order state all
  silently wrong).
- The sell ledger credited ``cost + commission + tax`` instead of
  ``cost - commission - tax``, so a same-price round trip made money by
  exactly the round-trip fees.
"""

import pytest

from quant_platform.execution.models import OrderStatus
from quant_platform.execution.oms import OrderManager


class TestASingleFillCannotExceedTheOrder:
    def test_buy_fill_larger_than_remaining_is_clamped(self):
        om = OrderManager(initial_cash=10_000_000)
        order = om.create_order("600519", "buy", 100)
        om.submit_order(order.order_id)
        cash_before = om.cash

        filled = om.fill_order(order.order_id, price=100.0, quantity=300)

        assert filled.fills[0].quantity == 100
        assert filled.filled_quantity == 100
        assert filled.remaining_quantity == 0
        assert filled.status == OrderStatus.FILLED
        assert om.positions["600519"].quantity == 100
        spent = 100 * 100.0 + filled.fills[0].commission + filled.fills[0].tax
        assert om.cash == pytest.approx(cash_before - spent, rel=0, abs=1e-9)

    def test_sell_fill_larger_than_remaining_is_clamped(self):
        om = OrderManager(initial_cash=10_000_000)
        buy = om.create_order("600519", "buy", 300)
        om.submit_order(buy.order_id)
        om.fill_order(buy.order_id, price=10.0)
        om.new_trading_day()  # T+1: all 300 shares become sellable

        sell = om.create_order("600519", "sell", 100)
        om.submit_order(sell.order_id)

        filled = om.fill_order(sell.order_id, price=10.0, quantity=300)

        assert filled.fills[0].quantity == 100
        assert filled.filled_quantity == 100
        assert filled.status == OrderStatus.FILLED
        assert "600519" in om.positions
        assert om.positions["600519"].quantity == 200
        assert om.positions["600519"].available == 200

    def test_a_single_fill_cannot_exceed_a_partially_filled_order(self):
        om = OrderManager(initial_cash=10_000_000)
        order = om.create_order("600519", "buy", 300)
        om.submit_order(order.order_id)
        om.fill_order(order.order_id, price=10.0, quantity=100)

        filled = om.fill_order(order.order_id, price=10.0, quantity=500)

        assert filled.fills[-1].quantity == 200
        assert filled.filled_quantity == 300
        assert om.positions["600519"].quantity == 300


class TestNonPositiveFillQuantityIsRejected:
    @pytest.mark.parametrize("qty", [0, -100])
    def test_zero_and_negative_quantity_do_not_touch_the_book(self, qty):
        om = OrderManager(initial_cash=10_000_000)
        order = om.create_order("600519", "buy", 100)
        om.submit_order(order.order_id)
        cash_before = om.cash

        with pytest.raises(ValueError):
            om.fill_order(order.order_id, price=100.0, quantity=qty)

        assert order.fills == []
        assert om.cash == cash_before
        assert om.positions == {}


class TestSellCashLedger:
    def _bought_book(self, quantity: int = 100, price: float = 10.0) -> OrderManager:
        om = OrderManager(initial_cash=1_000_000)
        buy = om.create_order("600519", "buy", quantity)
        om.submit_order(buy.order_id)
        om.fill_order(buy.order_id, price=price)
        om.new_trading_day()
        return om

    def test_sell_cash_equals_proceeds_minus_commission_and_tax(self):
        om = self._bought_book()
        cash_before = om.cash

        sell = om.create_order("600519", "sell", 100)
        om.submit_order(sell.order_id)
        filled = om.fill_order(sell.order_id, price=10.0)

        fill = filled.fills[0]
        assert fill.commission == pytest.approx(5.0, rel=0, abs=1e-12)
        assert fill.tax == pytest.approx(1.0, rel=0, abs=1e-12)
        assert om.cash == pytest.approx(
            cash_before + 10.0 * 100 - fill.commission - fill.tax,
            rel=0,
            abs=1e-9,
        )
        assert om.cash == pytest.approx(cash_before + 994.0, rel=0, abs=1e-9)

    def test_same_price_round_trip_loses_exactly_the_fees(self):
        om = self._bought_book()
        cash_after_buy = om.cash
        assert cash_after_buy == pytest.approx(1_000_000 - 1000.0 - 5.0)

        sell = om.create_order("600519", "sell", 100)
        om.submit_order(sell.order_id)
        filled = om.fill_order(sell.order_id, price=10.0)

        # Buy commission 5, sell commission 5, sell stamp tax 1: an 11 RMB
        # loss. The mutant credited the sell fees, ending at 1,000,001.
        assert om.cash == pytest.approx(999_989.0, rel=0, abs=1e-9)
        assert om.cash == pytest.approx(
            1_000_000.0
            - 5.0
            - filled.fills[0].commission
            - filled.fills[0].tax,
            rel=0,
            abs=1e-9,
        )
        assert "600519" not in om.positions

    def test_partial_sell_reconciles_cash_and_leaves_the_rest(self):
        om = self._bought_book(quantity=200)
        cash_before_sell = om.cash

        sell = om.create_order("600519", "sell", 100)
        om.submit_order(sell.order_id)
        filled = om.fill_order(sell.order_id, price=10.0)

        fill = filled.fills[0]
        expected = cash_before_sell + 1000.0 - fill.commission - fill.tax
        assert om.cash == pytest.approx(expected, rel=0, abs=1e-9)
        assert om.cash == pytest.approx(998_989.0, rel=0, abs=1e-9)
        assert om.positions["600519"].quantity == 100
        assert om.positions["600519"].available == 100


class TestOrderBookIsBounded:
    def test_completed_orders_are_pruned_but_active_orders_survive(
        self, monkeypatch,
    ):
        """The order book and blotter used to grow for the process lifetime."""
        from quant_platform.execution import oms as oms_module

        monkeypatch.setattr(oms_module, "MAX_ORDERS", 10)
        om = OrderManager(initial_cash=1_000_000)
        for _ in range(30):
            order = om.create_order("600519", "buy", 100)
            om.submit_order(order.order_id)
            om.fill_order(order.order_id, price=10.0)

        assert len(om.orders) <= 10
        assert len(om.blotter) <= 10

        active = om.create_order("000001", "buy", 100)
        assert active.order_id in om.orders
        assert om.orders[active.order_id].status == OrderStatus.PENDING
