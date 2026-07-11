"""Broker — execution contract. Stable interface over QMT / Paper / future IB.

Per ADR-0003 (Single Live Runner), there is one OMS / one risk engine / one
live path in the Production layer. Broker implementations live in the
Capability layer as drivers; the Production layer consumes exactly one.

Reference Implementations: PaperBroker (simulated), QMTBroker (real exchange).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Broker(Protocol):
    def submit(self, order: dict) -> str:
        """Submit an order; return broker order id."""
        ...

    def cancel(self, order_id: str) -> bool:
        ...

    def positions(self) -> list[dict]:
        ...

    def balance(self) -> dict:
        ...

    def orders(self) -> list[dict]:
        ...
