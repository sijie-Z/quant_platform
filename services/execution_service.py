"""Execution microservice — order routing and execution as a standalone service.

Subscribes to:
- order.new: New order to execute
- order.cancel: Cancel an order
- market.tick: Market data for execution decisions

Publishes to:
- order.submitted: Order sent to exchange
- order.filled: Order filled
- order.rejected: Order rejected
- execution.tca: Transaction cost analysis

Usage:
    python -m quant_platform.services.execution_service
"""

from __future__ import annotations

import asyncio

from quant_platform.core.message_bus import Message, MessageBus, create_message_bus
from quant_platform.execution.order_book import (
    BookOrder,
    OrderBookManager,
    OrderType,
    Side,
)
from quant_platform.execution.algorithms import SmartRouter
from quant_platform.services.base import BaseService
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionService(BaseService):
    """Order routing and execution service.

    Manages order lifecycle, routes to appropriate execution algorithm,
    and reports fills back to the event bus.
    """

    name = "execution-service"
    version = "1.0.0"
    port = 8002

    def __init__(self, bus: MessageBus, **kwargs):
        super().__init__(bus, **kwargs)
        self.order_manager = OrderBookManager()
        self.smart_router = SmartRouter()
        self._pending_orders: dict[str, BookOrder] = {}

    async def setup(self):
        logger.info("ExecutionService: initializing")

    def register_handlers(self):
        self.bus.subscribe("order.new", self.on_new_order)
        self.bus.subscribe("order.cancel", self.on_cancel_order)
        self.bus.subscribe("market.tick", self.on_tick)

    async def on_new_order(self, msg: Message):
        """Process a new order request."""
        self._record_message()
        try:
            data = msg.data
            order = BookOrder(
                order_id=data.get("order_id", ""),
                symbol=data.get("symbol", ""),
                side=Side(data.get("side", "buy")),
                order_type=OrderType(data.get("order_type", "limit")),
                price=data.get("price", 0),
                quantity=data.get("quantity", 0),
                source=data.get("source", "strategy"),
            )

            # Smart routing
            algo = self.smart_router.select_algorithm(
                quantity=order.quantity,
                symbol=order.symbol,
                urgency=data.get("urgency", "normal"),
            )

            # Submit to order book
            trades = self.order_manager.add_order(order)

            # Publish fills
            for trade in trades:
                await self.bus.publish("order.filled", {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "side": trade.aggressor_side.value,
                    "order_id": order.order_id,
                })
                self._record_publish()

            # Publish order status
            await self.bus.publish("order.submitted", {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
            })
            self._record_publish()

        except Exception as e:
            self._record_error(str(e))
            logger.error("ExecutionService on_new_order error: %s", e)

            # Publish rejection
            await self.bus.publish("order.rejected", {
                "order_id": msg.data.get("order_id", ""),
                "reason": str(e),
            })
            self._record_publish()

    async def on_cancel_order(self, msg: Message):
        """Cancel an order."""
        self._record_message()
        try:
            order_id = msg.data.get("order_id", "")
            symbol = msg.data.get("symbol", "")

            cancelled = self.order_manager.cancel_order(symbol, order_id)
            if cancelled:
                await self.bus.publish("order.cancelled", {
                    "order_id": order_id,
                    "symbol": symbol,
                })
                self._record_publish()

        except Exception as e:
            self._record_error(str(e))

    async def on_tick(self, msg: Message):
        """Update order books with market data."""
        self._record_message()
        try:
            tick = msg.data
            symbol = tick.get("symbol", "")
            if symbol:
                book = self.order_manager.get_or_create(symbol)
                # Update book state from tick
                # (simplified — real impl would process order-by-order)
        except Exception as e:
            self._record_error(str(e))

    async def shutdown(self):
        logger.info("ExecutionService shutting down")


async def main():
    bus = create_message_bus("local")
    service = ExecutionService(bus=bus)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
