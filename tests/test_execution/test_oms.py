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

        # Then sell -- on the *next* session. This test used to sell the same
        # day, which A-share T+1 forbids; the OMS had no `available` concept so
        # it went through. See TestT1Settlement below.
        self.om.new_trading_day()
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

    def test_blotter_includes_open_and_partial_orders(self):
        """Only completed orders were appended to `self.blotter`, so pending,
        submitted and partially filled orders never reached the API."""
        pending = self.om.create_order("600519", "buy", 100)
        submitted = self.om.create_order("000001", "buy", 200)
        self.om.submit_order(submitted.order_id)
        partial = self.om.create_order("300750", "buy", 300)
        self.om.submit_order(partial.order_id)
        self.om.fill_order(partial.order_id, price=100.0, quantity=100)

        blotter = {
            row["order_id"]: row for row in self.om.get_order_blotter()
        }

        assert blotter[pending.order_id]["status"] == "pending"
        assert blotter[submitted.order_id]["status"] == "submitted"
        assert blotter[partial.order_id]["status"] == "partial"
        assert blotter[partial.order_id]["remaining_quantity"] == 200

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

        self.om.new_trading_day()  # T+1: the shares are sellable tomorrow
        sell_order = self.om.create_order("600519", "sell", 100)
        self.om.submit_order(sell_order.order_id)
        self.om.fill_order(sell_order.order_id, price=1000.0)
        assert sell_order.fills[0].tax > 0.0


class TestT1Settlement:
    """A-share T+1: shares bought today cannot be sold today.

    The OMS carried a `_trade_date_offset` field commented "For T+1
    simulation" that nothing ever read, and validated sells against the full
    holding, so a same-day round trip -- which the market forbids -- was booked
    as a profit.
    """

    def setup_method(self):
        self.om = OrderManager(initial_cash=10_000_000)

    def _buy(self, qty=100, price=1000.0):
        order = self.om.create_order("600519", "buy", qty)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=price)

    def test_shares_bought_today_are_not_sellable_today(self):
        self._buy()
        assert self.om.positions["600519"].quantity == 100
        assert self.om.positions["600519"].available == 0

        with pytest.raises(ValueError, match="Insufficient sellable position"):
            self.om.create_order("600519", "sell", 100)

    def test_a_same_day_round_trip_cannot_be_booked(self):
        """Before the fix this buy-then-sell pair completed and pocketed the
        difference as realised P&L."""
        self._buy(price=1000.0)
        with pytest.raises(ValueError):
            self.om.create_order("600519", "sell", 100)

    def test_new_trading_day_releases_overnight_holdings(self):
        self._buy()
        self.om.new_trading_day()
        assert self.om.positions["600519"].available == 100

        order = self.om.create_order("600519", "sell", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1100.0)
        assert "600519" not in self.om.positions

    def test_only_the_old_cohort_is_released(self):
        """A buy after the roll stays locked until the next roll."""
        self._buy(qty=100)
        self.om.new_trading_day()
        self._buy(qty=100)

        pos = self.om.positions["600519"]
        assert pos.quantity == 200
        assert pos.available == 100, "yesterday's 100 only"

        with pytest.raises(ValueError, match="Insufficient sellable position"):
            self.om.create_order("600519", "sell", 200)

        # Selling what is actually available works.
        order = self.om.create_order("600519", "sell", 100)
        self.om.submit_order(order.order_id)
        self.om.fill_order(order.order_id, price=1100.0)
        assert self.om.positions["600519"].quantity == 100


class TestFillValidation:
    """fill_order must reject bad input before touching cash or positions.

    The HTTP layer already refused price <= 0, but internal callers (and the
    simulated exchange) went straight into the OMS, where a NaN price produced
    a NaN commission and a zero/negative quantity could corrupt the book.
    """

    def setup_method(self):
        self.om = OrderManager(initial_cash=10_000_000)

    def _submitted_buy(self, qty: int = 100):
        order = self.om.create_order("600519", "buy", qty)
        self.om.submit_order(order.order_id)
        return order

    @pytest.mark.parametrize(
        "price", [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_invalid_price_raises_before_any_mutation(self, price):
        order = self._submitted_buy()
        cash_before = self.om.cash

        with pytest.raises(ValueError):
            self.om.fill_order(order.order_id, price=price)

        assert order.fills == []
        assert self.om.cash == cash_before
        assert self.om.positions == {}

    @pytest.mark.parametrize("qty", [0, -100])
    def test_non_positive_quantity_raises_before_any_mutation(self, qty):
        order = self._submitted_buy()
        cash_before = self.om.cash

        with pytest.raises(ValueError):
            self.om.fill_order(order.order_id, price=100.0, quantity=qty)

        assert order.fills == []
        assert self.om.cash == cash_before
        assert self.om.positions == {}


class TestMarkToMarketAfterFill:
    """A buy must mark its position at the fill price immediately.

    `_process_fill` updated quantity and avg_cost but left current_price and
    market_value at zero, so `get_snapshot()` right after a buy showed cash
    reduced by the purchase and positions worth nothing -- the whole purchase
    looked like a loss.
    """

    def test_buy_marks_position_at_fill_price(self):
        om = OrderManager(initial_cash=1_000_000)
        order = om.create_order("600519", "buy", 100)
        om.submit_order(order.order_id)
        om.fill_order(order.order_id, price=1000.0)

        pos = om.positions["600519"]
        assert pos.current_price == 1000.0
        assert pos.market_value == 100_000.0

        snap = om.get_snapshot()
        assert snap.positions_value == 100_000.0
        assert snap.total_value == pytest.approx(om.cash + 100_000.0, abs=0.01)
        # Only fees below the initial cash, not the whole purchase.
        assert snap.total_value > 999_000

    def test_partial_sell_marks_the_remaining_position(self):
        om = OrderManager(initial_cash=1_000_000)
        buy = om.create_order("600519", "buy", 200)
        om.submit_order(buy.order_id)
        om.fill_order(buy.order_id, price=10.0)
        om.new_trading_day()

        sell = om.create_order("600519", "sell", 100)
        om.submit_order(sell.order_id)
        om.fill_order(sell.order_id, price=12.0)

        pos = om.positions["600519"]
        assert pos.quantity == 100
        assert pos.current_price == 12.0
        assert pos.market_value == 1200.0

    def test_snapshot_values_a_position_without_a_price_at_cost(self):
        """A position whose price was never marked must not count as zero."""
        from quant_platform.execution.models import Position

        om = OrderManager(initial_cash=1_000_000)
        om.positions["600519"] = Position(
            ticker="600519", quantity=100, avg_cost=50.0,
        )

        snap = om.get_snapshot()
        assert snap.positions_value == 5000.0


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


class TestSimulatedExchangePriceValidation:
    """Bad prints must not become fills, and limits must fill at the better price."""

    def setup_method(self):
        self.exchange = SimulatedExchange()
        self.om = OrderManager(initial_cash=10_000_000)
        self.exchange.set_order_manager(self.om)

    @pytest.mark.parametrize(
        "bad_price", [float("nan"), float("inf"), float("-inf"), 0.0, -5.0],
    )
    def test_invalid_market_price_is_skipped(self, bad_price):
        order = self.om.create_order("600519", "buy", 100, order_type="market")
        self.om.submit_order(order.order_id)

        self.exchange.update_market({"600519": bad_price})
        self.exchange.match_orders()

        assert order.status == OrderStatus.SUBMITTED
        assert order.fills == []
        assert self.om.cash == 10_000_000

    def test_limit_buy_fills_at_the_better_market_price(self):
        order = self.om.create_order(
            "600519", "buy", 100, order_type="limit", limit_price=1800.0,
        )
        self.om.submit_order(order.order_id)

        self.exchange.update_market({"600519": 1770.0})
        self.exchange.match_orders()

        assert order.status == OrderStatus.FILLED
        assert order.fills[0].price == 1770.0  # min(limit, market)

    def test_limit_sell_fills_at_the_better_market_price(self):
        buy = self.om.create_order("600519", "buy", 100)
        self.om.submit_order(buy.order_id)
        self.om.fill_order(buy.order_id, price=1000.0)
        self.om.new_trading_day()

        sell = self.om.create_order(
            "600519", "sell", 100, order_type="limit", limit_price=1700.0,
        )
        self.om.submit_order(sell.order_id)

        self.exchange.update_market({"600519": 1720.0})
        self.exchange.match_orders()

        assert sell.status == OrderStatus.FILLED
        assert sell.fills[0].price == 1720.0  # max(limit, market)


class TestBookConsistency:
    """Fills must not drive cash negative or over-sell a position."""

    def test_an_oversized_buy_is_rejected_before_cash_moves(self):
        om = OrderManager(initial_cash=10_000)
        order = om.create_order("600519", "buy", 100)
        om.submit_order(order.order_id)

        with pytest.raises(ValueError, match="Insufficient cash"):
            om.fill_order(order.order_id, price=100.0)

        assert om.cash == 10_000
        assert "600519" not in om.positions

    def test_two_sells_cannot_share_the_same_available_shares(self):
        om = OrderManager(initial_cash=1_000_000)
        buy = om.create_order("600519", "buy", 100)
        om.submit_order(buy.order_id)
        om.fill_order(buy.order_id, price=10.0)
        om.new_trading_day()  # release T+1

        first = om.create_order("600519", "sell", 100)
        second = om.create_order("600519", "sell", 100)
        om.submit_order(first.order_id)
        om.submit_order(second.order_id)

        om.fill_order(first.order_id, price=11.0)
        with pytest.raises(ValueError, match="Insufficient sellable position"):
            om.fill_order(second.order_id, price=11.0)

        assert "600519" not in om.positions
        assert om.cash > 0

    def test_a_second_partial_fill_cannot_spend_the_same_cash(self):
        """Cash validation must run on every fill, not only the first.

        A 200-share buy with only 15,000 in cash can pay for one 100-share
        fill at 100. The next 100 shares need another ~10,000; without a
        per-fill check the second fill goes through and cash turns negative.
        """
        om = OrderManager(initial_cash=15_000)
        order = om.create_order("600519", "buy", 200)
        om.submit_order(order.order_id)

        first = om.fill_order(order.order_id, price=100.0, quantity=100)
        assert first.status == OrderStatus.PARTIAL
        cash_after_first = om.cash
        assert cash_after_first >= 0

        with pytest.raises(ValueError, match="Insufficient cash"):
            om.fill_order(order.order_id, price=100.0, quantity=100)

        assert om.cash == pytest.approx(cash_after_first)
        assert om.positions["600519"].quantity == 100

    def test_sequential_partial_fills_cannot_exceed_the_order(self):
        om = OrderManager(initial_cash=1_000_000)
        order = om.create_order("600519", "buy", 300)
        om.submit_order(order.order_id)

        for _ in range(3):
            om.fill_order(order.order_id, price=10.0, quantity=100)
        with pytest.raises(ValueError):
            om.fill_order(order.order_id, price=10.0, quantity=100)

        assert om.positions["600519"].quantity == 300

    def test_repeated_buys_never_drive_cash_negative(self):
        om = OrderManager(initial_cash=100_000)
        rejected = 0
        for _ in range(200):
            order = om.create_order("600000", "buy", 100)
            om.submit_order(order.order_id)
            try:
                om.fill_order(order.order_id, price=10.0)
            except ValueError:
                rejected += 1
            assert om.cash >= 0

        assert rejected > 0  # the scenario actually exhausted the cash
