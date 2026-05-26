"""Data microservice — market data ingestion and distribution.

Subscribes to:
- data.request: Request for historical or real-time data
- data.config: Data source configuration changes

Publishes to:
- market.tick: Real-time market data
- market.snapshot: Periodic full market snapshots
- market.bar: OHLCV bar data
- data.quality: Data quality reports

Usage:
    python -m quant_platform.services.data_service
"""

from __future__ import annotations

import asyncio
import time

from quant_platform.core.message_bus import Message, MessageBus, create_message_bus
from quant_platform.services.base import BaseService
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class DataService(BaseService):
    """Market data ingestion and distribution service.

    Connects to data sources (AKShare, Baostock, WebSocket)
    and publishes normalized market data to the message bus.
    """

    name = "data-service"
    version = "1.0.0"
    port = 8003

    def __init__(self, bus: MessageBus, **kwargs):
        super().__init__(bus, **kwargs)
        self._data_source = None
        self._symbols: list[str] = []
        self._publish_interval = 1.0  # seconds

    async def setup(self):
        logger.info("DataService: initializing data sources")
        # Initialize data source (AKShare, Baostock, etc.)
        # self._data_source = AKShareProvider()

    def register_handlers(self):
        self.bus.subscribe("data.request", self.on_data_request)
        self.bus.subscribe("data.config", self.on_config_change)

    async def run(self):
        """Override to add data publishing loop."""
        await self.setup()
        await self.bus.start()
        self.register_handlers()
        self._running = True

        # Start data publishing loop
        publish_task = asyncio.create_task(self._publish_loop())

        # Run until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            publish_task.cancel()
            await self.shutdown()
            await self.bus.stop()

    async def _publish_loop(self):
        """Periodically publish market data."""
        while self._running:
            try:
                # Fetch latest data
                for symbol in self._symbols:
                    # Simulated tick — replace with real data source
                    tick = {
                        "symbol": symbol,
                        "price": 100.0,  # Would be real price
                        "timestamp_ns": time.time_ns(),
                        "volume": 1000000,
                    }
                    await self.bus.publish("market.tick", tick)
                    self._record_publish()

                await asyncio.sleep(self._publish_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error(str(e))
                logger.error("DataService publish error: %s", e)

    async def on_data_request(self, msg: Message):
        """Handle data requests."""
        self._record_message()
        try:
            request = msg.data
            symbol = request.get("symbol", "")
            request.get("type", "tick")

            # Fetch data
            # data = self._data_source.get_data(symbol, data_type)

            # Publish response
            await self.bus.publish("data.response", {
                "request_id": request.get("request_id", ""),
                "symbol": symbol,
                "data": [],  # Would be real data
            })
            self._record_publish()

        except Exception as e:
            self._record_error(str(e))

    async def on_config_change(self, msg: Message):
        """Handle configuration changes."""
        self._record_message()
        try:
            config = msg.data
            if "symbols" in config:
                self._symbols = config["symbols"]
                logger.info("DataService: updated symbols to %d", len(self._symbols))
            if "publish_interval" in config:
                self._publish_interval = config["publish_interval"]
        except Exception as e:
            self._record_error(str(e))

    async def shutdown(self):
        logger.info("DataService shutting down")


async def main():
    bus = create_message_bus("local")
    service = DataService(bus=bus)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
