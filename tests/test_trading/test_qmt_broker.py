"""Tests for QMTBroker — xtquant/miniQMT integration."""

import os
import sys
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
    def test_reported_maps_to_submitted(self):
        assert qmt_status_to_internal(2) == OrderStatus.SUBMITTED

    def test_filled_maps_to_filled(self):
        assert qmt_status_to_internal(23) == OrderStatus.FILLED

    def test_canceled_maps_to_cancelled(self):
        assert qmt_status_to_internal(12) == OrderStatus.CANCELLED

    def test_partial_fill(self):
        assert qmt_status_to_internal(22) == OrderStatus.PARTIAL

    def test_unknown_maps_to_pending(self):
        assert qmt_status_to_internal(999) == OrderStatus.PENDING


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

    def test_place_order_while_disconnected(self):
        broker = self.broker
        broker._connected = False
        order = Order(code="000001", side=OrderSide.BUY, quantity=100, price=15.0)
        result = broker.place_order(order)
        assert result.status == OrderStatus.REJECTED
        assert "not connected" in result.error_msg.lower()

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

        broker._trader.cancel_order_stock.return_value = None
        result = broker.cancel_order(order.order_id)
        assert result is True
        broker._trader.cancel_order_stock.assert_called_once()

    def test_cancel_order_not_found(self):
        broker = self.broker
        result = broker.cancel_order("nonexistent")
        assert result is False

    def test_cancel_order_disconnected(self):
        broker = self.broker
        broker._connected = False
        result = broker.cancel_order("any")
        assert result is False


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

    def test_get_account_disconnected(self):
        broker = self.broker
        broker._connected = False
        acct = broker.get_account()
        assert acct["broker"] == "qmt"
        assert acct["connected"] is False

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
