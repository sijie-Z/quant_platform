"""Risk microservice — real-time risk computation as a standalone service.

Subscribes to:
- order.filled: Update positions and recompute risk
- market.tick: Update market prices
- portfolio.snapshot: Periodic portfolio state

Publishes to:
- risk.status: Current risk level and Greeks
- risk.breach: Risk limit breaches
- risk.hedge: Hedge order recommendations

Usage:
    python -m quant_platform.services.risk_service
"""

from __future__ import annotations

import asyncio

from quant_platform.core.message_bus import Message, MessageBus, create_message_bus
from quant_platform.risk.realtime_engine import RealTimeRiskEngine
from quant_platform.services.base import BaseService
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class RiskService(BaseService):
    """Real-time risk computation service.

    Runs as a standalone process, communicating via message bus.
    Receives fill events and market data, publishes risk status.
    """

    name = "risk-service"
    version = "1.0.0"
    port = 8001

    def __init__(self, bus: MessageBus, **kwargs):
        super().__init__(bus, **kwargs)
        self.engine = RealTimeRiskEngine(**kwargs)

    async def setup(self):
        """Initialize risk engine."""
        logger.info("RiskService: initializing risk engine")
        self.engine.set_initial_equity(10_000_000)

    def register_handlers(self):
        """Subscribe to relevant topics."""
        self.bus.subscribe("order.filled", self.on_fill)
        self.bus.subscribe("market.tick", self.on_tick)
        self.bus.subscribe("portfolio.snapshot", self.on_snapshot)
        self.bus.subscribe("risk.command", self.on_command)

    async def on_fill(self, msg: Message):
        """Process a fill event: update risk state."""
        self._record_message()
        try:
            fill = msg.data
            update = self.engine.on_fill(fill)

            # Publish risk status
            await self.bus.publish("risk.status", {
                "risk_level": update.risk_level.value,
                "greeks": {
                    "delta": update.greeks.total_delta,
                    "gamma": update.greeks.total_gamma,
                    "vega": update.greeks.total_vega,
                },
                "utilizations": update.limit_utilizations,
                "latency_ns": update.update_latency_ns,
            })
            self._record_publish()

            # Publish breaches
            for breach in update.breaches:
                await self.bus.publish("risk.breach", {
                    "type": breach.limit_type.value,
                    "name": breach.limit_name,
                    "threshold": breach.threshold,
                    "actual": breach.actual_value,
                    "action": breach.action,
                })
                self._record_publish()

            # Publish hedge orders
            for hedge in update.hedge_orders:
                await self.bus.publish("risk.hedge", hedge)
                self._record_publish()

        except Exception as e:
            self._record_error(str(e))
            logger.error("RiskService on_fill error: %s", e)

    async def on_tick(self, msg: Message):
        """Process a market tick: update prices."""
        self._record_message()
        try:
            tick = msg.data
            symbol = tick.get("symbol", "")
            price = tick.get("price", 0)
            if symbol and price:
                self.engine.greeks_calc.update_spot(symbol, price)
        except Exception as e:
            self._record_error(str(e))

    async def on_snapshot(self, msg: Message):
        """Process a portfolio snapshot: update equity."""
        self._record_message()
        try:
            snapshot = msg.data
            equity = snapshot.get("total_value", 0)
            if equity > 0:
                self.engine.set_initial_equity(equity)
        except Exception as e:
            self._record_error(str(e))

    async def on_command(self, msg: Message):
        """Process risk commands (kill switch, limit updates)."""
        self._record_message()
        try:
            cmd = msg.data
            action = cmd.get("action", "")

            if action == "kill_switch":
                activate = cmd.get("activate", True)
                if activate:
                    self.engine.activate_kill_switch(cmd.get("reason", "manual"))
                else:
                    self.engine.deactivate_kill_switch()

                await self.bus.publish("risk.status", {
                    "risk_level": self.engine._risk_level.value,
                    "kill_switch": self.engine._kill_switch_active,
                })
                self._record_publish()

            elif action == "stress_test":
                result = self.engine.run_stress_test()
                await self.bus.publish("risk.stress_result", {
                    "worst_case_pnl": result.worst_case_pnl,
                    "expected_shortfall": result.expected_shortfall,
                    "scenarios_breached": result.scenarios_breached,
                    "run_time_us": result.run_time_us,
                })
                self._record_publish()

        except Exception as e:
            self._record_error(str(e))
            logger.error("RiskService on_command error: %s", e)

    async def shutdown(self):
        """Cleanup."""
        logger.info("RiskService shutting down")


# ──────────────────────────────────────────────────────────────────────
# Standalone Entry Point
# ──────────────────────────────────────────────────────────────────────


async def main():
    """Run risk service as standalone process."""
    bus = create_message_bus("local")  # Use "kafka" in production
    service = RiskService(bus=bus)
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
