"""WebSocket real-time market data provider.

Connects to public WebSocket endpoints for live A-share quotes:
- 东方财富 (East Money) push service
- 新浪财经 (Sina Finance) WebSocket

Replaces AKShare HTTP polling with persistent WebSocket connections
for lower latency and reduced request overhead.

Usage:
    ws = WebSocketQuoteProvider()
    ws.subscribe(["600519", "000001", "300750"])
    ws.start()  # Non-blocking, runs in background thread

    quote = ws.get_quote("600519")
    all_quotes = ws.get_all_quotes()
    ws.stop()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


@dataclass
class RealtimeQuote:
    """Real-time quote from WebSocket stream."""
    code: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    change_amt: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    prev_close: float = 0.0
    bid_prices: list[float] = field(default_factory=list)
    ask_prices: list[float] = field(default_factory=list)
    bid_volumes: list[float] = field(default_factory=list)
    ask_volumes: list[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name, "price": self.price,
            "change_pct": self.change_pct, "change_amt": self.change_amt,
            "volume": self.volume, "amount": self.amount,
            "high": self.high, "low": self.low, "open": self.open,
            "prev_close": self.prev_close,
            "bid1_price": self.bid_prices[0] if self.bid_prices else 0,
            "ask1_price": self.ask_prices[0] if self.ask_prices else 0,
            "timestamp": self.timestamp,
        }


class WebSocketQuoteProvider:
    """WebSocket-based real-time quote provider.

    Connects to public WebSocket endpoints for live A-share data.
    Maintains a local cache of latest quotes.

    Args:
        source: Data source ("eastmoney" or "sina")
        reconnect_interval: Seconds between reconnection attempts
        max_reconnect: Maximum reconnection attempts (0 = unlimited)
    """

    def __init__(
        self,
        source: str = "eastmoney",
        reconnect_interval: float = 5.0,
        max_reconnect: int = 0,
    ):
        if not HAS_WEBSOCKET:
            raise ImportError(
                "websocket-client required. Install with: pip install websocket-client"
            )

        self._source = source
        self._reconnect_interval = reconnect_interval
        self._max_reconnect = max_reconnect

        self._quotes: dict[str, RealtimeQuote] = {}
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._subscribed_codes: set[str] = set()
        self._callbacks: list[Callable[[RealtimeQuote], None]] = []
        self._reconnect_count = 0
        self._last_message_time = 0.0
        self._message_count = 0

    def subscribe(self, codes: list[str]):
        """Subscribe to stock codes.

        Args:
            codes: List of stock codes (e.g., ["600519", "000001"])
        """
        self._subscribed_codes.update(codes)
        if self._ws and self._running:
            self._send_subscribe(codes)

    def unsubscribe(self, codes: list[str]):
        """Unsubscribe from stock codes."""
        for code in codes:
            self._subscribed_codes.discard(code)

    def on_quote(self, callback: Callable[[RealtimeQuote], None]):
        """Register a callback for quote updates."""
        self._callbacks.append(callback)

    def get_quote(self, code: str) -> RealtimeQuote | None:
        """Get latest quote for a stock."""
        with self._lock:
            return self._quotes.get(code)

    def get_all_quotes(self) -> dict[str, RealtimeQuote]:
        """Get all cached quotes."""
        with self._lock:
            return dict(self._quotes)

    def start(self):
        """Start WebSocket connection in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("WebSocket provider started (source=%s)", self._source)

    def stop(self):
        """Stop WebSocket connection."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WebSocket provider stopped")

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    @property
    def stats(self) -> dict:
        return {
            "source": self._source,
            "connected": self.is_connected,
            "subscribed": len(self._subscribed_codes),
            "cached_quotes": len(self._quotes),
            "message_count": self._message_count,
            "last_message_age_s": round(time.time() - self._last_message_time, 1)
            if self._last_message_time else None,
            "reconnect_count": self._reconnect_count,
        }

    # ── Internal ──

    def _run_loop(self):
        """Reconnection loop."""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logger.warning("WebSocket error: %s", e)

            if not self._running:
                break

            self._reconnect_count += 1
            if self._max_reconnect and self._reconnect_count >= self._max_reconnect:
                logger.error("Max reconnection attempts reached")
                break

            logger.info("Reconnecting in %.1fs...", self._reconnect_interval)
            time.sleep(self._reconnect_interval)

    def _connect(self):
        """Create and run WebSocket connection."""
        url = self._get_url()
        logger.info("Connecting to %s", url)

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _get_url(self) -> str:
        """Get WebSocket URL for the selected source."""
        if self._source == "eastmoney":
            return "wss://push2ws.eastmoney.com/api/ws"
        elif self._source == "sina":
            return "wss://ws.sina.com.cn/quotes"
        else:
            raise ValueError(f"Unknown source: {self._source}")

    def _on_open(self, ws):
        """Called when connection opens."""
        logger.info("WebSocket connected to %s", self._source)
        self._reconnect_count = 0
        if self._subscribed_codes:
            self._send_subscribe(list(self._subscribed_codes))

    def _on_message(self, ws, message: str):
        """Called when a message is received."""
        self._last_message_time = time.time()
        self._message_count += 1

        try:
            data = json.loads(message)
            quotes = self._parse_message(data)

            with self._lock:
                for quote in quotes:
                    self._quotes[quote.code] = quote

            for callback in self._callbacks:
                for quote in quotes:
                    try:
                        callback(quote)
                    except Exception as e:
                        logger.warning("Callback error: %s", e)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Parse error: %s", e)

    def _on_error(self, ws, error):
        """Called on WebSocket error."""
        logger.warning("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        """Called when connection closes."""
        logger.info("WebSocket closed (code=%s, msg=%s)", close_status_code, close_msg)

    def _send_subscribe(self, codes: list[str]):
        """Send subscription message."""
        if not self._ws:
            return

        if self._source == "eastmoney":
            # East Money subscription format
            msg = json.dumps({
                "cmd": "subscribe",
                "codes": [f"{c}" for c in codes],
            })
        elif self._source == "sina":
            # Sina subscription format
            msg = json.dumps({
                "action": "subscribe",
                "symbols": [f"sh{c}" if c.startswith("6") else f"sz{c}" for c in codes],
            })
        else:
            return

        try:
            self._ws.send(msg)
            logger.debug("Subscribed to %d stocks", len(codes))
        except Exception as e:
            logger.warning("Subscribe failed: %s", e)

    def _parse_message(self, data: dict) -> list[RealtimeQuote]:
        """Parse WebSocket message into quotes."""
        quotes = []

        if self._source == "eastmoney":
            # East Money push format
            if "data" in data:
                items = data["data"] if isinstance(data["data"], list) else [data["data"]]
                for item in items:
                    if isinstance(item, dict):
                        q = RealtimeQuote(
                            code=str(item.get("code", item.get("f12", ""))),
                            name=item.get("name", item.get("f14", "")),
                            price=float(item.get("price", item.get("f17", 0))),
                            change_pct=float(item.get("change_pct", item.get("f3", 0))),
                            volume=float(item.get("volume", item.get("f5", 0))),
                            amount=float(item.get("amount", item.get("f6", 0))),
                            high=float(item.get("high", item.get("f15", 0))),
                            low=float(item.get("low", item.get("f16", 0))),
                            open=float(item.get("open", item.get("f16", 0))),
                            prev_close=float(item.get("prev_close", item.get("f18", 0))),
                        )
                        quotes.append(q)

        elif self._source == "sina":
            # Sina push format
            if "data" in data and isinstance(data["data"], dict):
                for symbol, fields in data["data"].items():
                    if isinstance(fields, list) and len(fields) >= 32:
                        code = symbol.replace("sh", "").replace("sz", "")
                        q = RealtimeQuote(
                            code=code,
                            name=str(fields[0]),
                            price=float(fields[3] or 0),
                            prev_close=float(fields[2] or 0),
                            open=float(fields[1] or 0),
                            high=float(fields[4] or 0),
                            low=float(fields[5] or 0),
                            volume=float(fields[8] or 0),
                            amount=float(fields[9] or 0),
                            bid_prices=[float(fields[i] or 0) for i in [11, 13, 15, 17, 19]],
                            ask_prices=[float(fields[i] or 0) for i in [21, 23, 25, 27, 29]],
                            bid_volumes=[float(fields[i] or 0) for i in [10, 12, 14, 16, 18]],
                            ask_volumes=[float(fields[i] or 0) for i in [20, 22, 24, 26, 28]],
                        )
                        if q.prev_close > 0:
                            q.change_amt = q.price - q.prev_close
                            q.change_pct = q.change_amt / q.prev_close * 100
                        quotes.append(q)

        return quotes


class SimulatedWebSocketProvider:
    """Simulated WebSocket provider for testing.

    Generates fake real-time quotes without actual network connection.
    Useful for unit tests and development.

    Args:
        codes: Stock codes to simulate
        update_interval: Seconds between quote updates
    """

    def __init__(
        self,
        codes: list[str] | None = None,
        update_interval: float = 1.0,
    ):
        import numpy as np

        self._codes = codes or ["600519", "000001", "300750"]
        self._interval = update_interval
        self._quotes: dict[str, RealtimeQuote] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[RealtimeQuote], None]] = []
        self._message_count = 0
        self._rng = np.random.default_rng(42)

        # Initialize base prices
        self._base_prices = {
            code: 50 + self._rng.random() * 200 for code in self._codes
        }

    def subscribe(self, codes: list[str]):
        self._codes.extend(codes)
        for code in codes:
            if code not in self._base_prices:
                self._base_prices[code] = 50 + self._rng.random() * 200

    def on_quote(self, callback: Callable[[RealtimeQuote], None]):
        self._callbacks.append(callback)

    def get_quote(self, code: str) -> RealtimeQuote | None:
        with self._lock:
            return self._quotes.get(code)

    def get_all_quotes(self) -> dict[str, RealtimeQuote]:
        with self._lock:
            return dict(self._quotes)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_connected(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "source": "simulated",
            "connected": self.is_connected,
            "subscribed": len(self._codes),
            "cached_quotes": len(self._quotes),
            "message_count": self._message_count,
        }

    def _update_loop(self):
        while self._running:
            self._generate_quotes()
            time.sleep(self._interval)

    def _generate_quotes(self):
        for code in self._codes:
            base = self._base_prices.get(code, 100)
            noise = self._rng.normal(0, 0.002)
            price = base * (1 + noise)
            self._base_prices[code] = price

            quote = RealtimeQuote(
                code=code,
                price=round(price, 2),
                change_pct=round(noise * 100, 2),
                change_amt=round(price - base, 2),
                volume=round(self._rng.random() * 100000),
                amount=round(self._rng.random() * 10000000),
                high=round(price * 1.005, 2),
                low=round(price * 0.995, 2),
                open=round(base * (1 + self._rng.normal(0, 0.001)), 2),
                prev_close=round(base, 2),
                bid_prices=[round(price - i * 0.01, 2) for i in range(1, 6)],
                ask_prices=[round(price + i * 0.01, 2) for i in range(1, 6)],
                bid_volumes=[round(self._rng.random() * 10000) for _ in range(5)],
                ask_volumes=[round(self._rng.random() * 10000) for _ in range(5)],
            )

            with self._lock:
                self._quotes[code] = quote
            self._message_count += 1

            for callback in self._callbacks:
                try:
                    callback(quote)
                except Exception:
                    pass
