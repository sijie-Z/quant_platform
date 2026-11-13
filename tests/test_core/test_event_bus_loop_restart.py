"""The bus must survive a stop/start across event loops.

A repeated application lifespan (two ``TestClient`` contexts, uvicorn reload)
starts the bus on a fresh event loop. Subscription queues created on the
previous loop used to be reused, so every consumer raised "bound to a
different event loop" in a tight loop -- millions of log lines per second and
a hung test process.
"""

import asyncio
import threading

from quant_platform.core.event_bus_v2 import AsyncEventBus


def test_stop_start_on_a_new_event_loop_rebuilds_subscription_queues(caplog):
    bus = AsyncEventBus(default_queue_size=32)
    received: list[int] = []
    seen = threading.Event()

    def handler(event):
        received.append(event.data["i"])
        seen.set()

    bus.subscribe_sync("test.restart", handler)

    async def cycle(value: int) -> None:
        await bus.start()
        bus.publish("test.restart", {"i": value})
        for _ in range(200):
            if len(received) >= value:
                break
            await asyncio.sleep(0.01)
        await bus.stop()

    asyncio.run(cycle(1))
    seen.clear()
    asyncio.run(cycle(2))

    assert received == [1, 2]
    assert not any(
        "bound to a different event loop" in record.message
        for record in caplog.records
    )
