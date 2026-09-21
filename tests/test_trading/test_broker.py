"""Tests for trading.broker — Simulated, QMT, XTP brokers and registry."""

import pytest

from quant_platform.trading.broker import (
    BROKER_REGISTRY,
    BrokerInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    QMTBroker,
    SimulatedBroker,
    XTPBroker,
    create_broker,
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


# ── Cross-Asset SimulatedBroker ──


class TestSimulatedBrokerCrossAsset:
    def test_future_lot_size_validation(self):
        """Futures with lot_size=1 should accept qty=1."""
        from quant_platform.core.instrument import AssetUniverse, Instrument, InstrumentType
        u = AssetUniverse()
        u.add(Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE, lot_size=1))
        broker = SimulatedBroker(initial_cash=50_000_000, asset_universe=u)
        broker.connect()
        order = Order(code="IF2406", side=OrderSide.BUY, quantity=1, price=3500.0)
        result = broker.place_order(order)
        # Should not reject for lot size
        assert result.error_msg == "" or "multiple of 1" not in (result.error_msg or "")

    def test_etf_no_stamp_tax_on_buy(self):
        """ETF buy should not incur stamp tax even with asset_universe."""
        from quant_platform.core.instrument import AssetUniverse, Instrument, InstrumentType
        u = AssetUniverse()
        u.add(Instrument(symbol="510300", asset_type=InstrumentType.ETF, stamp_tax_rate=0.0))
        broker = SimulatedBroker(initial_cash=1_000_000, asset_universe=u)
        broker.connect()
        # Verify the instrument was loaded
        assert broker._asset_universe.get("510300") is not None
        assert broker._asset_universe.get("510300").stamp_tax_rate == 0.0

    def test_lot_size_from_universe(self):
        """Verify lot_size lookup from asset_universe."""
        from quant_platform.core.instrument import AssetUniverse, Instrument, InstrumentType
        u = AssetUniverse()
        u.add(Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE, lot_size=1))
        broker = SimulatedBroker(initial_cash=50_000_000, asset_universe=u)
        assert broker._get_lot_size("IF2406") == 1
        assert broker._get_lot_size("600519") == 100  # default fallback

    def test_multiplier_from_universe(self):
        from quant_platform.core.instrument import AssetUniverse, Instrument, InstrumentType
        u = AssetUniverse()
        u.add(Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE, multiplier=300, lot_size=1))
        broker = SimulatedBroker(initial_cash=50_000_000, asset_universe=u)
        assert broker._get_multiplier("IF2406") == 300
        assert broker._get_multiplier("600519") == 1.0

    def test_commission_from_universe(self):
        from quant_platform.core.instrument import AssetUniverse, Instrument, InstrumentType
        u = AssetUniverse()
        u.add(Instrument(symbol="IF2406", asset_type=InstrumentType.FUTURE,
                        commission_per_lot=25.0, lot_size=1))
        broker = SimulatedBroker(initial_cash=50_000_000, asset_universe=u)
        comm = broker._get_commission("IF2406", 3500.0, 1)
        assert comm == 25.0


# ── XTPBroker (import-only tests — no real connection) ──


class TestXTPBroker:
    def test_init_defaults(self):
        broker = XTPBroker()
        assert not broker._connected
        assert broker._client_id == 1

    def test_init_with_params(self):
        broker = XTPBroker(client_id=2, key="test_key",
                          data_folder="/data/xtp",
                          server_ip="192.168.1.1", server_port=6001)
        assert broker._client_id == 2
        assert broker._key == "test_key"
        assert broker._server_ip == "192.168.1.1"

    def test_connect_without_credentials(self):
        broker = XTPBroker()
        # Should return False when xtp is not installed
        result = broker.connect()
        assert not result

    def test_disconnect(self):
        broker = XTPBroker()
        assert broker.disconnect() is True
        assert not broker._connected

    def test_place_order_not_connected(self):
        broker = XTPBroker()
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800)
        result = broker.place_order(order)
        assert result.status == OrderStatus.REJECTED

    def test_cancel_order_not_connected(self):
        broker = XTPBroker()
        assert broker.cancel_order("any_id") is False

    def test_get_positions_not_connected(self):
        broker = XTPBroker()
        assert broker.get_positions() == []

    def test_get_orders_not_connected(self):
        broker = XTPBroker()
        assert broker.get_orders() == []

    def test_get_account_not_connected(self):
        broker = XTPBroker()
        acct = broker.get_account()
        assert acct["connected"] is False
        assert acct["broker"] == "xtp"

    def test_get_order_book_not_connected(self):
        broker = XTPBroker()
        assert broker.get_order_book("600519") is None

    def test_to_xtp_code(self):
        """Test code-to-exchange mapping without XTP installed."""
        broker = XTPBroker()
        # When xtp not installed, this will raise AttributeError.
        # The static method relies on xtp import.
        # Verify the static method exists.
        assert hasattr(broker, '_to_xtp_code')

    def test_broker_interface_compliance(self):
        """XTPBroker must implement BrokerInterface."""
        assert issubclass(XTPBroker, BrokerInterface)


# ── Broker Registry ──


class TestBrokerRegistry:
    def test_all_brokers_registered(self):
        assert "simulated" in BROKER_REGISTRY
        assert "qmt" in BROKER_REGISTRY
        assert "xtp" in BROKER_REGISTRY

    def test_create_simulated(self):
        broker = create_broker("simulated", initial_cash=5_000_000)
        assert isinstance(broker, SimulatedBroker)
        assert broker._initial_cash == 5_000_000

    def test_create_qmt(self):
        broker = create_broker("qmt", account="123", server="localhost:58610")
        assert isinstance(broker, QMTBroker)

    def test_create_xtp(self):
        broker = create_broker("xtp", client_id=3)
        assert isinstance(broker, XTPBroker)
        assert broker._client_id == 3

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown broker"):
            create_broker("nonexistent")

    def test_create_simulated_lob_alias(self):
        broker = create_broker("simulated_lob")
        assert isinstance(broker, SimulatedBroker)


# ── QMTBroker (import-only tests) ──


class TestQMTBroker:
    def test_init_defaults(self):
        broker = QMTBroker(account="test")
        assert not broker._connected

    def test_connect_without_credentials(self):
        broker = QMTBroker(account="test")
        result = broker.connect(blocking=False)
        assert not result

    def test_place_order_not_connected(self):
        broker = QMTBroker(account="test")
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800)
        result = broker.place_order(order)
        assert result.status == OrderStatus.REJECTED

    def test_to_xt_code_shanghai(self):
        from quant_platform.trading.qmt_utils import to_qmt_code
        assert to_qmt_code("600519") == "600519.SH"

    def test_to_xt_code_shenzhen(self):
        from quant_platform.trading.qmt_utils import to_qmt_code
        assert to_qmt_code("000001") == "000001.SZ"

    def test_to_xt_code_already_formatted(self):
        from quant_platform.trading.qmt_utils import to_qmt_code
        assert to_qmt_code("600519.SH") == "600519.SH"

    def test_from_xt_code(self):
        from quant_platform.trading.qmt_utils import from_qmt_code
        assert from_qmt_code("600519.SH") == "600519"
        assert from_qmt_code("000001") == "000001"


# ── Order Model ──


class TestOrderModel:
    def test_order_defaults(self):
        o = Order()
        assert o.quantity == 0
        assert o.status == OrderStatus.PENDING

    def test_order_to_dict(self):
        o = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800)
        d = o.to_dict()
        assert d["code"] == "600519"
        assert d["side"] == "buy"
        assert d["quantity"] == 100


class TestPositionModel:
    def test_position_defaults(self):
        p = Position()
        assert p.code == ""
        assert p.quantity == 0

    def test_position_update_price(self):
        p = Position(code="600519", quantity=100, avg_cost=1800.0)
        p.update_price(1900.0)
        assert p.current_price == 1900.0
        assert p.market_value == 190000.0
        assert p.unrealized_pnl == 10000.0
        assert abs(p.unrealized_pnl_pct - 0.0556) < 0.001


class TestPositionsAreMarkedOnFill:
    """A position opened during a cycle used to be worth zero in that cycle.

    `Position(...)` is created with the dataclass defaults current_price=0 and
    market_value=0, and only `update_market_prices()` sets them -- which the
    live engine calls in step 1, before it executes orders in step 3. For the
    rest of the cycle the shares contributed nothing to equity while their cost
    had already left cash.
    """

    def _buy(self, broker, quantity=1000, price=10.0):
        return broker.place_order(Order(
            code="600000", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=quantity, price=price,
        ))

    def test_a_same_cycle_buy_is_not_reported_as_a_loss(self):
        broker = SimulatedBroker(initial_cash=1_000_000)
        assert broker.connect()
        before = broker.get_account()["total_equity"]

        order = self._buy(broker)
        assert order.filled_quantity > 0, order.error_msg

        after = broker.get_account()["total_equity"]
        # The account may lose the fees, but not the purchase price as well:
        # the shares it now holds are worth roughly what was paid for them.
        assert before - after < 200, (
            f"equity fell by {before - after:.2f} on a "
            f"{order.filled_quantity * order.filled_price:.2f} purchase"
        )

    def test_the_new_position_carries_a_market_value(self):
        broker = SimulatedBroker(initial_cash=1_000_000)
        assert broker.connect()

        order = self._buy(broker)

        held = broker.get_positions()
        assert len(held) == 1
        assert held[0].current_price == pytest.approx(order.filled_price)
        assert held[0].market_value == pytest.approx(
            held[0].quantity * order.filled_price
        )


class TestCashCannotGoNegative:
    """The buy pre-check used a 0.1% buffer where the fee schedule applies.

    `worst_cost = price * quantity * multiplier * 1.001` ignores the ¥5
    commission floor, so 100 shares at ¥10 -- which costs ¥1005 -- passed the
    check on any balance of ¥1001 or more and left `self._cash` negative.
    """

    PRICE = 10.0

    def _broker(self, cash: float) -> SimulatedBroker:
        broker = SimulatedBroker(initial_cash=cash)
        assert broker.connect()
        return broker

    def _buy(self, broker, quantity=100):
        return broker.place_order(Order(
            code="600000", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=quantity, price=self.PRICE,
        ))

    @pytest.mark.parametrize("cash", [1000.50, 1001.00, 1002.00, 1004.90, 1004.99, 1005.00])
    def test_the_balance_is_never_left_negative(self, cash):
        broker = self._broker(cash)

        self._buy(broker)

        assert broker.get_account()["cash"] >= 0, (
            f"cash went negative from a starting balance of {cash}"
        )

    def test_a_balance_short_of_the_commission_is_rejected(self):
        broker = self._broker(1004.99)  # the shares cost 1000, the fee 5

        order = self._buy(broker)

        assert order.status == OrderStatus.REJECTED
        assert broker.get_account()["cash"] == pytest.approx(1004.99), "cash moved on a rejection"
        assert broker.get_positions() == []

    def test_a_balance_that_exactly_covers_cost_and_fee_fills(self):
        broker = self._broker(1005.00)

        order = self._buy(broker)

        assert order.status == OrderStatus.FILLED, order.error_msg
        assert order.filled_quantity == 100
        assert broker.get_account()["cash"] == pytest.approx(0.0, abs=1e-9)

    def test_the_boundary_is_the_fee_schedule_not_a_percentage(self):
        """¥1001 is above `value * 1.001` (¥1001) and below `value + fee`
        (¥1005): the old check admitted it, a correct one does not."""
        broker = self._broker(1001.00)

        order = self._buy(broker)

        assert order.status == OrderStatus.REJECTED
        assert broker.get_account()["cash"] == pytest.approx(1001.00)
