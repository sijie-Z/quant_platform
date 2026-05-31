"""Connection lifecycle manager for broker connections.

Inspired by Hummingbot's connector base class with explicit network status
management and health checks. This is a standalone utility that existing
brokers can use without changing their inheritance.

Usage:
    from quant_platform.execution.connection import ConnectionManager

    class MyBroker(BrokerInterface):
        def __init__(self):
            self.conn = ConnectionManager(name="QMT")

        def connect(self):
            self.conn.connect()
            try:
                self._do_connect()
                self.conn.mark_connected()
            except:
                self.conn.mark_error()
                raise

        def check_network(self):
            return self.conn.health_check(self._ping)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Callable


class ConnectionStatus(StrEnum):
    """Broker connection state machine.

    INITIALIZED -> CONNECTING -> CONNECTED <-> DISCONNECTING -> DISCONNECTED
                      ↓                                            ↑
                   ERROR ---------------------------------------→ ERROR
    """
    INITIALIZED = "initialized"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ConnectionManager:
    """Manages connection lifecycle with health checks and auto-reconnect.

    Hummingbot-style connection management for broker/source connections.
    """

    def __init__(
        self,
        name: str = "connection",
        max_reconnect_attempts: int = 5,
    ):
        self.name = name
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_attempts = 0
        self._status: ConnectionStatus = ConnectionStatus.INITIALIZED

    @property
    def status(self) -> ConnectionStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status == ConnectionStatus.CONNECTED

    def connect(self) -> None:
        """Initiate connection sequence."""
        self._status = ConnectionStatus.CONNECTING

    def mark_connected(self) -> None:
        """Call after successful connection."""
        self._status = ConnectionStatus.CONNECTED
        self._reconnect_attempts = 0

    def disconnect(self) -> None:
        """Initiate disconnection sequence."""
        self._status = ConnectionStatus.DISCONNECTING

    def mark_disconnected(self) -> None:
        """Call after successful disconnection."""
        self._status = ConnectionStatus.DISCONNECTED

    def mark_error(self) -> None:
        """Call on connection error."""
        self._status = ConnectionStatus.ERROR

    def health_check(self, ping_fn: Callable[[], bool] | None = None) -> bool:
        """Run a health check and update status.

        Args:
            ping_fn: Optional function that returns True if healthy.

        Returns:
            True if the connection is healthy.
        """
        if ping_fn is not None:
            try:
                healthy = ping_fn()
                if healthy:
                    self._status = ConnectionStatus.CONNECTED
                else:
                    self._status = ConnectionStatus.ERROR
                return healthy
            except Exception:
                self._status = ConnectionStatus.ERROR
                return False
        return self._status == ConnectionStatus.CONNECTED

    def try_reconnect(self, reconnect_fn: Callable[[], bool]) -> bool:
        """Attempt reconnection up to max_reconnect_attempts.

        Args:
            reconnect_fn: Function that attempts reconnection.
                         Should return True if successful.

        Returns:
            True if reconnected successfully.
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            return False

        self._reconnect_attempts += 1
        self._status = ConnectionStatus.CONNECTING

        try:
            if reconnect_fn():
                self._status = ConnectionStatus.CONNECTED
                self._reconnect_attempts = 0
                return True
            self._status = ConnectionStatus.ERROR
            return False
        except Exception:
            self._status = ConnectionStatus.ERROR
            return False

    def reset_reconnect_count(self) -> None:
        """Reset the reconnect attempt counter."""
        self._reconnect_attempts = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"status={self._status.value}, "
            f"reconnects={self._reconnect_attempts}/{self._max_reconnect_attempts}"
            f")"
        )
