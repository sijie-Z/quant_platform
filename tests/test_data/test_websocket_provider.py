"""Tests for WebSocket real-time quote provider."""

import json
import numpy as np
import pandas as pd
import pytest
import threading
import time
from unittest.mock import patch, MagicMock

from quant_platform.data.providers.websocket_provider import (
    RealtimeQuote,
    WebSocketQuoteProvider,
    SimulatedWebSocketProvider,
)


class TestRealtimeQuote:
    def test_creation(self):
        q = RealtimeQuote(code="600519", price=1800.0, change_pct=2.5)
        assert q.code == "600519"
        assert q.price == 1800.0
        assert q.change_pct == 2.5

    def test_to_dict(self):
        q = RealtimeQuote(code="600519", price=1800.0, bid_prices=[1799, 1798])
        d = q.to_dict()
        assert d["code"] == "600519"
        assert d["bid1_price"] == 1799

    def test_default_timestamp(self):
        q = RealtimeQuote(code="600519")
        assert q.timestamp  # Should have auto-generated timestamp


class TestWebSocketQuoteProvider:
    def test_init_requires_websocket(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", False):
            with pytest.raises(ImportError, match="websocket-client"):
                WebSocketQuoteProvider()

    def test_subscribe(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            ws.subscribe(["600519", "000001"])
            assert "600519" in ws._subscribed_codes
            assert "000001" in ws._subscribed_codes

    def test_unsubscribe(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            ws.subscribe(["600519", "000001"])
            ws.unsubscribe(["600519"])
            assert "600519" not in ws._subscribed_codes
            assert "000001" in ws._subscribed_codes

    def test_get_quote_empty(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            assert ws.get_quote("600519") is None

    def test_get_all_quotes_empty(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            assert ws.get_all_quotes() == {}

    def test_on_quote_callback(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            callback = MagicMock()
            ws.on_quote(callback)
            assert callback in ws._callbacks

    def test_stats(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            stats = ws.stats
            assert "source" in stats
            assert "connected" in stats
            assert "subscribed" in stats
            assert stats["source"] == "eastmoney"

    def test_not_running_by_default(self):
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider()
            assert not ws.is_connected

    def test_parse_eastmoney_message(self):
        """Parse East Money format."""
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider(source="eastmoney")
            data = {
                "data": [{
                    "code": "600519",
                    "name": "贵州茅台",
                    "price": 1800.5,
                    "change_pct": 1.5,
                    "volume": 50000,
                    "amount": 90000000,
                    "high": 1810.0,
                    "low": 1790.0,
                    "open": 1795.0,
                    "prev_close": 1790.0,
                }]
            }
            quotes = ws._parse_message(data)
            assert len(quotes) == 1
            assert quotes[0].code == "600519"
            assert quotes[0].price == 1800.5

    def test_parse_sina_message(self):
        """Parse Sina format."""
        with patch("quant_platform.data.providers.websocket_provider.HAS_WEBSOCKET", True):
            ws = WebSocketQuoteProvider(source="sina")
            data = {
                "data": {
                    "sh600519": [
                        "贵州茅台", 1795.0, 1790.0, 1800.5, 1810.0, 1790.0,
                        1800.0, 1801.0, 50000, 90000000,
                        1799.0, 100, 1798.0, 200, 1797.0, 300, 1796.0, 400, 1795.0, 500,
                        1801.0, 100, 1802.0, 200, 1803.0, 300, 1804.0, 400, 1805.0, 500,
                        "2024-01-01", "15:00:00",
                    ]
                }
            }
            quotes = ws._parse_message(data)
            assert len(quotes) == 1
            assert quotes[0].code == "600519"


class TestSimulatedWebSocketProvider:
    def test_creation(self):
        ws = SimulatedWebSocketProvider(codes=["600519"])
        assert ws._codes == ["600519"]

    def test_subscribe(self):
        ws = SimulatedWebSocketProvider()
        ws.subscribe(["600000"])
        assert "600000" in ws._codes

    def test_start_stop(self):
        ws = SimulatedWebSocketProvider(codes=["600519"], update_interval=0.05)
        ws.start()
        assert ws.is_connected
        time.sleep(0.2)
        ws.stop()
        assert not ws.is_connected

    def test_generates_quotes(self):
        ws = SimulatedWebSocketProvider(codes=["600519"], update_interval=0.05)
        ws.start()
        time.sleep(0.2)
        ws.stop()

        quote = ws.get_quote("600519")
        assert quote is not None
        assert quote.code == "600519"
        assert quote.price > 0

    def test_callback_called(self):
        ws = SimulatedWebSocketProvider(codes=["600519"], update_interval=0.05)
        received = []
        ws.on_quote(lambda q: received.append(q))
        ws.start()
        time.sleep(0.2)
        ws.stop()
        assert len(received) > 0

    def test_get_all_quotes(self):
        ws = SimulatedWebSocketProvider(codes=["600519", "000001"], update_interval=0.05)
        ws.start()
        time.sleep(0.2)
        ws.stop()

        all_quotes = ws.get_all_quotes()
        assert len(all_quotes) == 2

    def test_stats(self):
        ws = SimulatedWebSocketProvider(codes=["600519"], update_interval=0.05)
        ws.start()
        time.sleep(0.2)
        ws.stop()

        stats = ws.stats
        assert stats["source"] == "simulated"
        assert stats["message_count"] > 0

    def test_quote_has_bid_ask(self):
        ws = SimulatedWebSocketProvider(codes=["600519"], update_interval=0.05)
        ws.start()
        time.sleep(0.15)
        ws.stop()

        quote = ws.get_quote("600519")
        assert len(quote.bid_prices) == 5
        assert len(quote.ask_prices) == 5
        assert quote.bid_prices[0] < quote.ask_prices[0]
