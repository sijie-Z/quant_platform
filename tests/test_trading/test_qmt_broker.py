"""Tests for QMTBroker — xtquant/miniQMT integration."""

import logging
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from quant_platform.trading.broker import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    QMTBroker,
    create_broker,
)
from quant_platform.trading.qmt_utils import (
    classify_exchange,
    describe_qmt_error,
    from_qmt_code,
    from_qmt_order_type,
    from_qmt_price_type,
    qmt_order_to_dict,
    qmt_position_to_dict,
    qmt_status_to_internal,
    qmt_trade_to_dict,
    to_qmt_code,
    to_qmt_exchange,
    to_qmt_order_type,
    to_qmt_price_type,
)

# ── Symbol mapping ──


class TestSymbolMapping:
    def test_sh_code(self):
        assert classify_exchange("600519") == "SH"
        assert classify_exchange("688981") == "SH"

    def test_sz_code(self):
        assert classify_exchange("000001") == "SZ"
        assert classify_exchange("300750") == "SZ"

    def test_to_qmt_code_bare(self):
        assert to_qmt_code("600519") == "600519.SH"
        assert to_qmt_code("000001") == "000001.SZ"
        assert to_qmt_code("300750") == "300750.SZ"

    def test_to_qmt_code_idempotent(self):
        assert to_qmt_code("600519.SH") == "600519.SH"
        assert to_qmt_code("000001.SZ") == "000001.SZ"

    def test_from_qmt_code(self):
        assert from_qmt_code("600519.SH") == "600519"
        assert from_qmt_code("000001.SZ") == "000001"

    def test_to_qmt_exchange(self):
        assert to_qmt_exchange("600519") == 1  # SH
        assert to_qmt_exchange("000001") == 2  # SZ

    def test_classify_with_dot(self):
        assert classify_exchange("600519.SH") == "SH"
        assert classify_exchange("300750.SZ") == "SZ"


# ── Order type mapping ──


class TestOrderTypeMapping:
    def test_market_to_qmt(self):
        assert to_qmt_price_type(OrderType.MARKET) == 5

    def test_limit_to_qmt(self):
        assert to_qmt_price_type(OrderType.LIMIT) == 11

    def test_buy_to_qmt(self):
        assert to_qmt_order_type(OrderSide.BUY) == 23

    def test_sell_to_qmt(self):
        assert to_qmt_order_type(OrderSide.SELL) == 24


# ── Status mapping ──


class TestStatusMapping:
    """xtquant order_status uses 48-57, not 0-24.

    The previous table mapped 0/1/2/10/11/12/22/23/24/50, values that are
    not XtOrder order_status codes at all, so every real status fell through
    to the PENDING default (and 50 was explicitly wrong as well).
    """

    def test_unreported_and_wait_reporting_are_pending(self):
        assert qmt_status_to_internal(48) == OrderStatus.PENDING
        assert qmt_status_to_internal(49) == OrderStatus.PENDING

    def test_reported_maps_to_submitted(self):
        assert qmt_status_to_internal(50) == OrderStatus.SUBMITTED

    def test_canceled_maps_to_cancelled(self):
        assert qmt_status_to_internal(54) == OrderStatus.CANCELLED

    def test_part_succ_maps_to_partial(self):
        assert qmt_status_to_internal(55) == OrderStatus.PARTIAL

    def test_succeeded_maps_to_filled(self):
        assert qmt_status_to_internal(56) == OrderStatus.FILLED

    def test_junk_maps_to_rejected(self):
        assert qmt_status_to_internal(57) == OrderStatus.REJECTED

    def test_unknown_maps_to_pending(self):
        assert qmt_status_to_internal(999) == OrderStatus.PENDING

    def test_old_non_status_codes_are_not_lifecycle_updates(self):
        # 2/12/22/23 are xtconstant order/side constants, not order states.
        # They must no longer masquerade as SUBMITTED/CANCELLED/etc.
        for bogus in (0, 1, 2, 10, 11, 12, 22, 23, 24):
            assert qmt_status_to_internal(bogus) == OrderStatus.PENDING


# ── Error descriptions ──


class TestErrorDescriptions:
    def test_known_error(self):
        assert "资金不足" in describe_qmt_error(-6)

    def test_unknown_error(self):
        assert "QMT error" in describe_qmt_error(-999)


# ── Conversion helpers ──


class TestTradeConversion:
    def test_dict_passthrough(self):
        d = {"order_id": "123", "price": 10.5}
        assert qmt_trade_to_dict(d) == d

    def test_object_conversion(self):
        obj = MagicMock()
        obj.order_id = "abc"
        obj.stock_code = "600519.SH"
        obj.price = 15.0
        obj.volume = 100
        result = qmt_trade_to_dict(obj)
        assert result["order_id"] == "abc"
        assert result["code"] == "600519.SH"
        assert result["price"] == 15.0

    def test_position_dict_passthrough(self):
        d = {"code": "600519", "volume": 200}
        assert qmt_position_to_dict(d) == d


# ── QMTBroker (mocked xtquant) ──


@pytest.fixture
def mock_xtquant():
    """Mock entire xtquant package for testing without real QMT."""
    mock_xtc = MagicMock()
    mock_xtc.FIX_PRICE = 11
    mock_xtc.LATEST_PRICE = 5
    mock_xtc.STOCK_BUY = 23
    mock_xtc.STOCK_SELL = 24

    mock_xttrader = MagicMock()
    mock_xttrader.XtQuantTraderCallback = type("XtQuantTraderCallback", (), {})

    mock_xttype = MagicMock()
    mock_xttype.StockAccount = MagicMock()

    modules = {
        "xtquant": MagicMock(),
        "xtquant.xtconstant": mock_xtc,
        "xtquant.xttrader": mock_xttrader,
        "xtquant.xttype": mock_xttype,
    }
    with patch.dict(sys.modules, modules):
        yield


class TestQMTBrokerInit:
    def test_init_without_xtquant(self):
        # Ensure xtquant is NOT importable
        with patch.dict(sys.modules, {"xtquant": None}):
            broker = QMTBroker(account="test_acct")
            assert broker._HAS_XTQUANT is False

    def test_init_defaults(self, mock_xtquant):
        broker = QMTBroker(account="sim001")
        assert broker._account_id == "sim001"
        assert broker._server == "localhost:58610"
        assert broker._mode == "sim"
        assert broker._connected is False

    def test_init_explicit_server(self, mock_xtquant):
        broker = QMTBroker(account="sim001", server="192.168.1.1:58610", mode="live")
        assert broker._server == "192.168.1.1:58610"
        assert broker._mode == "live"

    def test_init_reads_password_from_env(self, mock_xtquant):
        with patch.dict(os.environ, {"QMT_PASSWORD": "secret123"}):
            broker = QMTBroker(account="sim001")
            assert broker._password == "secret123"

    def test_init_explicit_password_overrides_env(self, mock_xtquant):
        with patch.dict(os.environ, {"QMT_PASSWORD": "env_secret"}):
            broker = QMTBroker(account="sim001", password="explicit_secret")
            assert broker._password == "explicit_secret"


class TestQMTBrokerConnect:
    def test_connect_no_account_fails(self, mock_xtquant):
        broker = QMTBroker(account="")
        assert broker.connect(blocking=False) is False

    def test_connect_calls_xttrader(self, mock_xtquant):
        broker = QMTBroker(account="sim001", server="localhost:58610")

        # Mock the trader to return success on connect
        with patch.object(broker, "_HAS_XTQUANT", True):
            mock_trader = MagicMock()
            mock_trader.connect.return_value = 0
            broker._XtQuantTrader = MagicMock(return_value=mock_trader)
            broker._StockAccount = MagicMock()
            broker._account = broker._StockAccount.return_value

            result = broker.connect(blocking=False)
            assert result is True
            assert broker._connected is True

    def test_connect_failure_stays_disconnected(self, mock_xtquant):
        broker = QMTBroker(account="sim001", server="localhost:58610")

        with patch.object(broker, "_HAS_XTQUANT", True):
            mock_trader = MagicMock()
            mock_trader.connect.return_value = -1
            broker._XtQuantTrader = MagicMock(return_value=mock_trader)

            result = broker.connect(blocking=False)
            assert result is False


class TestQMTBrokerOrders:
    def setup_method(self):
        broker = QMTBroker(account="sim001")
        broker._HAS_XTQUANT = True
        broker._connected = True
        broker._trader = MagicMock()
        broker._trader.order_stock.return_value = 12345  # broker order ID
        broker._account = MagicMock()
        self.broker = broker

    def test_place_order_buy_limit(self):
        broker = self.broker
        order = Order(code="600519", side=OrderSide.BUY,
                      order_type=OrderType.LIMIT, quantity=100, price=1800.0)
        result = broker.place_order(order)
        assert result.status == OrderStatus.SUBMITTED
        assert result.broker_order_id == "12345"
        broker._trader.order_stock.assert_called_once()

    def test_place_order_while_disconnected_fails_closed(self):
        """No lazy reconnect possible -> raise; order_stock is never called."""
        broker = self.broker
        broker._connected = False
        broker._sleep_fn = lambda _delay: None
        broker.connect = MagicMock(return_value=False)
        order = Order(code="000001", side=OrderSide.BUY, quantity=100, price=15.0)

        with pytest.raises(QMTBroker.BrokerConnectionError):
            broker.place_order(order)

        assert broker.connect.call_count == 3
        broker._trader.order_stock.assert_not_called()

    @pytest.mark.parametrize("reject_value", [0, -1, None])
    def test_order_stock_failure_is_rejected(self, reject_value):
        """order_stock returning <=0 (or None) is a rejection, not a fill."""
        broker = self.broker
        broker._trader.order_stock.return_value = reject_value
        order = Order(code="600519", side=OrderSide.BUY,
                      order_type=OrderType.LIMIT, quantity=100, price=1800.0)

        result = broker.place_order(order)

        assert result.status == OrderStatus.REJECTED
        assert result.broker_order_id == ""
        assert result.error_msg
        broker._trader.order_stock.assert_called_once()

    def test_place_order_market(self):
        broker = self.broker
        order = Order(code="300750", side=OrderSide.SELL,
                      order_type=OrderType.MARKET, quantity=200, price=200.0)
        broker.place_order(order)
        call_args = broker._trader.order_stock.call_args
        assert call_args[1]["price"] == 0.0  # market orders send price=0

    def test_cancel_order_success(self):
        broker = self.broker
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800.0)
        order.broker_order_id = "12345"
        order.status = OrderStatus.SUBMITTED
        broker._orders = [order]

        broker._trader.cancel_order_stock.return_value = 0
        result = broker.cancel_order(order.order_id)
        assert result is True
        broker._trader.cancel_order_stock.assert_called_once()

    def test_cancel_rejected_by_qmt_is_not_reported_as_success(self):
        """cancel_order_stock returns !=0 on failure; the old code returned
        True without ever looking at the result."""
        broker = self.broker
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800.0)
        order.broker_order_id = "12345"
        order.status = OrderStatus.SUBMITTED
        broker._orders = [order]

        broker._trader.cancel_order_stock.return_value = -1
        result = broker.cancel_order(order.order_id)

        assert result is False
        assert order.status == OrderStatus.SUBMITTED  # not confirmed cancelled

    def test_cancel_order_not_found(self):
        broker = self.broker
        result = broker.cancel_order("nonexistent")
        assert result is False

    def test_cancel_order_disconnected_raises(self):
        broker = self.broker
        broker._connected = False
        broker._sleep_fn = lambda _delay: None
        broker.connect = MagicMock(return_value=False)
        with pytest.raises(QMTBroker.BrokerConnectionError):
            broker.cancel_order("any")


class TestQMTBrokerQueries:
    def setup_method(self):
        broker = QMTBroker(account="sim001", initial_cash=5_000_000)
        broker._HAS_XTQUANT = True
        broker._connected = True
        broker._trader = MagicMock()
        broker._account = MagicMock()
        self.broker = broker

    def test_get_account_connected(self):
        broker = self.broker
        mock_asset = MagicMock()
        mock_asset.cash = 3_000_000
        mock_asset.market_value = 2_500_000
        mock_asset.total_asset = 5_500_000
        broker._trader.query_stock_asset.return_value = mock_asset
        broker._trader.query_stock_positions.return_value = []

        acct = broker.get_account()
        assert acct["broker"] == "qmt"
        assert acct["connected"] is True
        assert acct["cash"] == 3_000_000
        assert acct["total_equity"] == 5_500_000

    def test_get_account_disconnected_raises(self):
        """Read paths fail loud when the broker cannot be reached (R10)."""
        broker = self.broker
        broker._connected = False
        broker._sleep_fn = lambda _delay: None
        broker.connect = MagicMock(return_value=False)
        with pytest.raises(QMTBroker.BrokerConnectionError):
            broker.get_account()

    def test_get_positions_empty(self):
        broker = self.broker
        broker._trader.query_stock_positions.return_value = []
        positions = broker.get_positions()
        assert positions == []

    def test_get_positions_with_holdings(self):
        broker = self.broker
        mock_pos = MagicMock()
        mock_pos.stock_code = "600519.SH"
        mock_pos.volume = 100
        mock_pos.can_use_volume = 100
        mock_pos.avg_price = 1750.0
        mock_pos.market_value = 180000.0
        broker._trader.query_stock_positions.return_value = [mock_pos]

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].code == "600519"
        assert positions[0].quantity == 100

    def test_get_orders_filter_by_status(self):
        broker = self.broker
        o1 = Order(code="600519", side=OrderSide.BUY, quantity=100, price=1800.0)
        o1.status = OrderStatus.FILLED
        o2 = Order(code="000001", side=OrderSide.BUY, quantity=200, price=15.0)
        o2.status = OrderStatus.SUBMITTED
        broker._orders = [o1, o2]

        filled = broker.get_orders(status=OrderStatus.FILLED)
        assert len(filled) == 1
        assert filled[0].code == "600519"


class TestQMTBrokerDisconnect:
    def test_disconnect(self, mock_xtquant):
        broker = QMTBroker(account="sim001")
        broker._connected = True
        broker._trader = MagicMock()
        result = broker.disconnect()
        assert result is True
        assert broker._connected is False


# ── Broker Registry ──


class TestBrokerRegistry:
    def test_create_qmt_broker(self, mock_xtquant):
        broker = create_broker("qmt", account="sim001")
        assert isinstance(broker, QMTBroker)
        assert broker._mode == "sim"

    def test_create_qmt_sim_broker(self, mock_xtquant):
        broker = create_broker("qmt_sim", account="sim001")
        assert isinstance(broker, QMTBroker)
        assert broker._mode == "sim"

    def test_create_qmt_live_broker(self, mock_xtquant):
        broker = create_broker("qmt_live", account="live001")
        assert isinstance(broker, QMTBroker)
        assert broker._mode == "live"

    def test_create_paper_broker(self):
        broker = create_broker("paper", initial_cash=1_000_000)
        from quant_platform.trading.broker import SimulatedBroker
        assert isinstance(broker, SimulatedBroker)

    def test_create_unknown_broker_raises(self):
        with pytest.raises(ValueError, match="Unknown broker"):
            create_broker("nonexistent")


class TestCallbackLifecycle:
    """QMT callbacks must update the tracked order from real fill reports."""

    def _broker_with_order(self):
        broker = QMTBroker.__new__(QMTBroker)
        broker._orders = []
        order = Order(
            code="600519", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=10.0,
        )
        order.broker_order_id = "12345"
        order.status = OrderStatus.SUBMITTED
        broker._orders.append(order)
        return broker, order

    def test_trade_report_fills_the_order(self):
        broker, order = self._broker_with_order()

        class XtTrade:
            order_id = 12345
            traded_volume = 100
            traded_price = 10.5

        broker._on_trade(XtTrade())

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.filled_price == 10.5

    def test_order_update_applies_status_and_partial_fill(self):
        broker, order = self._broker_with_order()

        class XtOrder:
            order_id = 12345
            order_status = 55  # ORDER_PART_SUCC
            traded_volume = 40
            traded_price = 10.2

        broker._on_order_update(XtOrder())

        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 40

    def test_order_update_applies_a_filled_status(self):
        broker, order = self._broker_with_order()

        class XtOrder:
            order_id = 12345
            order_status = 56  # ORDER_SUCCEEDED
            traded_volume = 100
            traded_price = 10.5

        broker._on_order_update(XtOrder())

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100

    def test_error_callback_matches_an_integer_id(self):
        broker, order = self._broker_with_order()

        broker._update_order_status(12345, OrderStatus.REJECTED, "boom")

        assert order.status == OrderStatus.REJECTED
        assert order.error_msg == "boom"

    def test_placement_ack_does_not_downgrade_submitted(self):
        broker, order = self._broker_with_order()

        broker._on_async_ack({"order_id": 12345})

        assert order.status == OrderStatus.SUBMITTED

    def test_async_response_without_status_is_ignored(self):
        broker, order = self._broker_with_order()

        broker._on_async_response({"order_id": 12345})

        assert order.status == OrderStatus.SUBMITTED


class TestStaleCallbacksCannotDowngrade:
    """A late lifecycle callback must not undo a terminal state.

    After the status map was fixed, an out-of-order XtOrder snapshot (e.g.
    ORDER_PART_SUCC delivered after ORDER_SUCCEEDED) would still overwrite a
    FILLED/REJECTED/CANCELLED order, because the handler assigned whatever
    the callback carried.
    """

    @staticmethod
    def _broker_with_order(status, filled=0):
        broker = QMTBroker.__new__(QMTBroker)
        broker._orders = []
        order = Order(
            code="600519", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=10.0,
        )
        order.broker_order_id = "12345"
        order.status = status
        order.filled_quantity = filled
        broker._orders.append(order)
        return broker, order

    @pytest.mark.parametrize("terminal", [
        OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED,
    ])
    def test_terminal_states_are_never_downgraded(self, terminal):
        broker, order = self._broker_with_order(terminal)

        class StaleXtOrder:
            order_id = 12345
            order_status = 55  # ORDER_PART_SUCC, stale
            traded_volume = 0
            traded_price = 0.0

        broker._on_order_update(StaleXtOrder())

        assert order.status == terminal

    def test_stale_report_callback_does_not_demote_a_partial_fill(self):
        broker, order = self._broker_with_order(OrderStatus.PARTIAL, filled=40)

        class StaleXtOrder:
            order_id = 12345
            order_status = 50  # ORDER_REPORTED arrives late
            traded_volume = 40
            traded_price = 10.2

        broker._on_order_update(StaleXtOrder())

        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 40


# ── R10: lazy reconnect, broker resync, get_orders TTL ──


def _xt_order(
    order_id,
    *,
    code="600519.SH",
    side=23,
    price_type=11,
    volume=100,
    price=10.0,
    traded_volume=0,
    traded_price=0.0,
    status=50,
    status_msg="",
    order_time=0,
):
    """Shape of an xtquant ``XtOrder`` for reconciliation tests."""
    return SimpleNamespace(
        order_id=order_id,
        stock_code=code,
        order_type=side,
        price_type=price_type,
        order_volume=volume,
        price=price,
        traded_volume=traded_volume,
        traded_price=traded_price,
        order_status=status,
        status_msg=status_msg,
        order_time=order_time,
    )


def _connected_broker(**kwargs):
    """A QMTBroker that believes it is connected, with an empty mock book."""
    broker = QMTBroker(account="sim001", sleep_fn=lambda _delay: None, **kwargs)
    broker._HAS_XTQUANT = True
    broker._connected = True
    broker._trader = MagicMock()
    broker._trader.query_stock_orders.return_value = []
    broker._trader.query_stock_trades.return_value = []
    broker._account = MagicMock()
    return broker


class TestQmtUtilsReconciliationMappings:
    def test_from_qmt_order_type(self):
        assert from_qmt_order_type(23) == OrderSide.BUY
        assert from_qmt_order_type(24) == OrderSide.SELL
        assert from_qmt_order_type(None) == OrderSide.BUY

    def test_from_qmt_price_type(self):
        assert from_qmt_price_type(5) == OrderType.MARKET
        assert from_qmt_price_type(11) == OrderType.LIMIT
        assert from_qmt_price_type(None) == OrderType.LIMIT

    def test_qmt_order_to_dict_reads_xtorder_fields(self):
        d = qmt_order_to_dict(_xt_order("123", traded_volume=40, traded_price=10.2))
        assert d["order_id"] == "123"
        assert d["code"] == "600519.SH"
        assert d["order_volume"] == 100
        assert d["traded_volume"] == 40
        assert d["traded_price"] == 10.2
        assert d["order_status"] == 50

    def test_qmt_trade_to_dict_reads_xttrade_fields(self):
        raw = SimpleNamespace(
            order_id="123", traded_id="T1", stock_code="600519.SH",
            traded_price=10.5, traded_volume=40, traded_amount=420.0,
            trade_time="093000",
        )
        d = qmt_trade_to_dict(raw)
        assert d["order_id"] == "123"
        assert d["trade_id"] == "T1"
        assert d["code"] == "600519.SH"
        assert d["price"] == 10.5
        assert d["quantity"] == 40
        assert d["amount"] == 420.0

    def test_qmt_order_to_dict_normalizes_raw_dict_keys(self):
        d = qmt_order_to_dict({
            "order_id": "1", "stock_code": "600519.SH",
            "order_volume": 100, "order_status": 50,
        })
        assert d["code"] == "600519.SH"
        assert d["stock_code"] == "600519.SH"  # original key preserved

    def test_qmt_trade_to_dict_normalizes_raw_dict_keys(self):
        d = qmt_trade_to_dict({
            "order_id": "1", "traded_volume": 60, "traded_price": 10.1,
        })
        assert d["quantity"] == 60
        assert d["price"] == 10.1


class TestLazyReconnect:
    def test_query_after_disconnect_reconnects(self):
        """on_disconnected only flips state; the next query reconnects."""
        broker = _connected_broker()
        broker._trader.query_stock_positions.return_value = []
        calls = {"n": 0}

        def fake_connect(blocking=True):
            calls["n"] += 1
            broker._connected = True
            return True

        broker.connect = fake_connect
        assert broker.connect() is True          # initial explicit connect
        broker._connected = False                # on_disconnected callback
        positions = broker.get_positions()

        assert positions == []
        assert broker._connected is True
        assert calls["n"] == 2                   # lazy reconnect called again

    def test_reconnect_failure_blocks_submission(self):
        """Fail closed: 3 attempts, exponential backoff, no order_stock call."""
        sleeps = []
        broker = QMTBroker(
            account="sim001",
            max_reconnect_attempts=3,
            reconnect_backoff=1.0,
            sleep_fn=sleeps.append,
        )
        broker._HAS_XTQUANT = True
        broker._connected = False
        broker._trader = MagicMock()
        broker.connect = MagicMock(return_value=False)

        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=10.0)
        with pytest.raises(QMTBroker.BrokerConnectionError):
            broker.place_order(order)

        assert broker.connect.call_count == 3
        assert sleeps == [1.0, 2.0]              # waits before retries 2 and 3
        broker._trader.order_stock.assert_not_called()

    @pytest.mark.parametrize("method", ["get_positions", "get_orders", "get_account"])
    def test_read_paths_fail_loud_when_reconnect_fails(self, method):
        broker = QMTBroker(account="sim001", sleep_fn=lambda _delay: None)
        broker._HAS_XTQUANT = True
        broker._connected = False
        broker._trader = MagicMock()
        broker.connect = MagicMock(return_value=False)

        with pytest.raises(QMTBroker.BrokerConnectionError):
            getattr(broker, method)()

    def test_reconnected_broker_resyncs_immediately(self):
        broker = _connected_broker()
        broker._connected = False
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=50),
        ]

        def fake_connect(blocking=True):
            broker._connected = True
            return True

        broker.connect = fake_connect
        orders = broker.get_orders()

        assert [o.broker_order_id for o in orders] == ["9001"]
        assert broker._trader.query_stock_orders.call_count == 1
        assert broker._last_resync_at is not None


class TestBrokerResync:
    def _broker_with_known_order(self, **kwargs):
        broker = _connected_broker(**kwargs)
        known = Order(
            code="600519", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=10.0,
        )
        known.broker_order_id = "9001"
        known.status = OrderStatus.SUBMITTED
        broker._orders = [known]
        return broker, known

    def test_resync_updates_known_and_recovers_unknown(self):
        broker, known = self._broker_with_known_order(resync_ttl=0.0)
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=56, traded_volume=100, traded_price=10.5),
            _xt_order(
                "9002", code="000001.SZ", side=24, volume=200,
                price=15.0, status=50,
            ),
        ]

        orders = broker.get_orders()

        # Known order: status/fill updated from the broker snapshot.
        assert known.status == OrderStatus.FILLED
        assert known.filled_quantity == 100
        assert known.filled_price == 10.5

        # Unknown order (submitted before restart): recovered, mapped fields.
        recovered = [o for o in orders if o.broker_order_id == "9002"]
        assert len(recovered) == 1
        r = recovered[0]
        assert r.code == "000001"
        assert r.side == OrderSide.SELL
        assert r.order_type == OrderType.LIMIT
        assert r.quantity == 200
        assert r.price == 15.0
        assert r.status == OrderStatus.SUBMITTED
        assert r.error_msg == ""

    def test_resync_is_idempotent(self):
        broker, _ = self._broker_with_known_order(resync_ttl=0.0)
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9002", code="000001.SZ", status=50),
        ]

        broker.get_orders()
        broker.get_orders()
        broker.get_orders()

        assert len([o for o in broker._orders if o.broker_order_id == "9002"]) == 1

    def test_resync_does_not_downgrade_terminal_orders(self):
        broker, known = self._broker_with_known_order(resync_ttl=0.0)
        known.status = OrderStatus.FILLED
        known.filled_quantity = 100
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=50, traded_volume=0),
        ]

        broker.get_orders()

        assert known.status == OrderStatus.FILLED
        assert known.filled_quantity == 100

    def test_trades_cross_check_stale_filled_volume(self):
        broker, known = self._broker_with_known_order(resync_ttl=0.0)
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=50, traded_volume=0),
        ]
        broker._trader.query_stock_trades.return_value = [
            SimpleNamespace(order_id="9001", traded_volume=60, traded_price=10.1),
        ]

        broker.get_orders()

        assert known.filled_quantity == 60
        # Trades never change lifecycle status (order snapshot is the truth).
        assert known.status == OrderStatus.SUBMITTED

    def test_trade_cross_check_never_exceeds_quantity(self):
        broker, known = self._broker_with_known_order(resync_ttl=0.0)
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=50, traded_volume=0),
        ]
        broker._trader.query_stock_trades.return_value = [
            SimpleNamespace(order_id="9001", traded_volume=80),
            SimpleNamespace(order_id="9001", traded_volume=80),
        ]

        broker.get_orders()

        assert known.filled_quantity == 100

    def test_trade_query_failure_warns_but_keeps_order_data(self, caplog):
        broker, known = self._broker_with_known_order(resync_ttl=0.0)
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=55, traded_volume=40, traded_price=10.2),
        ]
        broker._trader.query_stock_trades.side_effect = RuntimeError("no trades")

        with caplog.at_level(logging.WARNING):
            broker.get_orders()

        assert known.status == OrderStatus.PARTIAL
        assert known.filled_quantity == 40
        assert any(
            "query_stock_trades failed" in r.getMessage() for r in caplog.records
        )


class TestResyncFailurePolicy:
    def test_get_orders_degrades_to_memory_on_resync_failure(self, caplog):
        broker = _connected_broker()
        existing = Order(code="600519", side=OrderSide.BUY, quantity=100, price=10.0)
        broker._orders = [existing]
        broker._trader.query_stock_orders.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.WARNING):
            orders = broker.get_orders()

        assert orders == [existing]
        assert any(
            "query_stock_orders failed" in r.getMessage() for r in caplog.records
        )

    def test_submit_proceeds_when_only_resync_fails(self):
        """Reconnect OK + resync broken -> order is still sent (gateway up)."""
        broker = QMTBroker(account="sim001", sleep_fn=lambda _delay: None)
        broker._HAS_XTQUANT = True
        broker._connected = False
        broker._trader = MagicMock()
        broker._trader.order_stock.return_value = 777
        broker._trader.query_stock_orders.side_effect = RuntimeError("query broken")
        broker._account = MagicMock()

        def fake_connect(blocking=True):
            broker._connected = True
            return True

        broker.connect = fake_connect
        order = Order(code="600519", side=OrderSide.BUY, quantity=100, price=10.0)

        result = broker.place_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert result.broker_order_id == "777"
        broker._trader.order_stock.assert_called_once()


class TestGetOrdersTTL:
    def test_two_queries_within_ttl_hit_broker_once(self):
        clock = {"now": 1000.0}
        broker = _connected_broker(resync_ttl=2.0, clock_fn=lambda: clock["now"])

        broker.get_orders()
        assert broker._trader.query_stock_orders.call_count == 1

        clock["now"] += 1.9
        broker.get_orders()
        assert broker._trader.query_stock_orders.call_count == 1

        clock["now"] += 0.2
        broker.get_orders()
        assert broker._trader.query_stock_orders.call_count == 2

    def test_failed_resync_is_rate_limited_by_ttl(self):
        clock = {"now": 1000.0}
        broker = _connected_broker(resync_ttl=2.0, clock_fn=lambda: clock["now"])
        broker._trader.query_stock_orders.side_effect = RuntimeError("boom")

        broker.get_orders()
        broker.get_orders()
        assert broker._trader.query_stock_orders.call_count == 1

        clock["now"] += 2.1
        broker.get_orders()
        assert broker._trader.query_stock_orders.call_count == 2


class TestReconnectReentrancy:
    def test_callback_racing_connect_does_not_deadlock(self):
        """on_disconnected landing inside connect()'s snapshot window must not
        re-enter (and deadlock on) the non-reentrant reconnect lock."""
        broker = QMTBroker(account="sim001", sleep_fn=lambda _delay: None)
        broker._HAS_XTQUANT = True
        broker._trader = MagicMock()
        broker._account = MagicMock()
        inner_errors = []

        def fake_connect(blocking=True):
            broker._connected = True
            broker._connected = False  # on_disconnected fires in this window
            try:
                broker.get_account()   # connect()'s post-connect snapshot path
            except QMTBroker.BrokerConnectionError as exc:
                inner_errors.append(str(exc))
            return True

        broker.connect = fake_connect
        outcome = {}

        def run():
            try:
                broker.place_order(Order(
                    code="600519", side=OrderSide.BUY,
                    order_type=OrderType.LIMIT, quantity=100, price=10.0,
                ))
            except Exception as exc:  # the fail-closed outcome under test
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "lazy reconnect deadlocked on its own lock"
        assert isinstance(outcome.get("error"), QMTBroker.BrokerConnectionError)
        assert inner_errors, "inner get_account never reached the re-entry guard"


class TestConcurrentResyncVsPlaceOrder:
    def test_place_order_does_not_duplicate_resync_recovered_order(self):
        """A resync that recovers the broker order between order_stock() and
        the local append must leave exactly one tracked entry."""
        broker = _connected_broker()
        broker._trader.query_stock_orders.return_value = [
            _xt_order("777", status=50),
        ]

        def order_stock(**_kwargs):
            # Simulate the concurrent resync winning the race.
            broker._resync_from_broker()
            return 777

        broker._trader.order_stock.side_effect = order_stock
        order = Order(
            code="600519", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=10.0,
        )

        result = broker.place_order(order)

        entries = [o for o in broker._orders if o.broker_order_id == "777"]
        assert len(entries) == 1
        assert result is entries[0]


class TestResyncMonotonicStatus:
    def test_stale_unreported_snapshot_does_not_demote_submitted(self):
        broker = _connected_broker(resync_ttl=0.0)
        known = Order(
            code="600519", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=10.0,
        )
        known.broker_order_id = "9001"
        known.status = OrderStatus.SUBMITTED
        broker._orders = [known]
        broker._trader.query_stock_orders.return_value = [
            _xt_order("9001", status=48),  # UNREPORTED, stale
        ]

        broker.get_orders()

        assert known.status == OrderStatus.SUBMITTED


class TestReconnectEdgeCases:
    def test_resync_attempt_after_reconnect_is_not_repeated_in_same_call(self):
        broker = _connected_broker()
        broker._connected = False
        broker._trader.query_stock_orders.side_effect = RuntimeError("boom")

        def fake_connect(blocking=True):
            broker._connected = True
            return True

        broker.connect = fake_connect
        broker.get_orders()

        # One attempt inside the reconnect path, none again via the TTL gate.
        assert broker._trader.query_stock_orders.call_count == 1


class TestConnectHousekeeping:
    def test_connect_stops_previous_trader(self, mock_xtquant):
        broker = QMTBroker(account="sim001")
        broker._connected = True
        old_trader = MagicMock()
        broker._trader = old_trader
        with patch.object(broker, "_HAS_XTQUANT", True):
            new_trader = MagicMock()
            new_trader.connect.return_value = -1
            broker._XtQuantTrader = MagicMock(return_value=new_trader)
            broker._StockAccount = MagicMock()
            assert broker.connect(blocking=False) is False
        old_trader.stop.assert_called_once()
        # The old session is gone; a failed reconnect must not claim to be up.
        assert broker._connected is False

    def test_connect_invalidates_resync_ttl(self, mock_xtquant):
        broker = QMTBroker(account="sim001")
        broker._last_resync_at = 123.0
        with patch.object(broker, "_HAS_XTQUANT", True):
            trader = MagicMock()
            trader.connect.return_value = 0
            broker._XtQuantTrader = MagicMock(return_value=trader)
            broker._StockAccount = MagicMock()
            broker._update_account_snapshot = MagicMock()
            assert broker.connect(blocking=False) is True
        assert broker._last_resync_at is None
