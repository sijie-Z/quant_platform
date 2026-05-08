"""Portfolio state machine — formal lifecycle management.

States:
    INIT        → starting up, loading config
    READY       → connected to broker, waiting for market
    PRE_MARKET  → market not open yet, preparing signals
    TRADING     → market open, actively trading
    REBALANCING → executing a rebalance cycle
    POST_MARKET → market closed, doing EOD reconciliation
    HALTED      → emergency stop (risk breach or manual)
    ERROR       → unrecoverable error

Transitions are logged and published as events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioState(str, Enum):
    INIT = "init"
    READY = "ready"
    PRE_MARKET = "pre_market"
    TRADING = "trading"
    REBALANCING = "rebalancing"
    POST_MARKET = "post_market"
    HALTED = "halted"
    ERROR = "error"


# Valid state transitions
TRANSITIONS = {
    PortfolioState.INIT: [PortfolioState.READY, PortfolioState.ERROR],
    PortfolioState.READY: [PortfolioState.PRE_MARKET, PortfolioState.TRADING, PortfolioState.HALTED, PortfolioState.ERROR],
    PortfolioState.PRE_MARKET: [PortfolioState.TRADING, PortfolioState.HALTED, PortfolioState.ERROR],
    PortfolioState.TRADING: [PortfolioState.REBALANCING, PortfolioState.POST_MARKET, PortfolioState.HALTED, PortfolioState.ERROR],
    PortfolioState.REBALANCING: [PortfolioState.TRADING, PortfolioState.POST_MARKET, PortfolioState.HALTED, PortfolioState.ERROR],
    PortfolioState.POST_MARKET: [PortfolioState.READY, PortfolioState.PRE_MARKET, PortfolioState.ERROR],
    PortfolioState.HALTED: [PortfolioState.READY, PortfolioState.INIT],
    PortfolioState.ERROR: [PortfolioState.INIT, PortfolioState.READY],
}


@dataclass
class StateTransition:
    """Record of a state change."""
    from_state: str
    to_state: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class PortfolioStateMachine:
    """Formal state machine for portfolio/trading lifecycle.

    Guarantees:
    - Only valid transitions are allowed
    - All transitions are logged
    - Entry/exit hooks fire on state change
    - Current state is always known
    """

    def __init__(self, on_transition: Callable[[StateTransition], None] | None = None):
        self._state = PortfolioState.INIT
        self._history: list[StateTransition] = []
        self._on_transition = on_transition
        self._entry_hooks: dict[PortfolioState, list[Callable]] = {}
        self._exit_hooks: dict[PortfolioState, list[Callable]] = {}
        self._state_since = time.time()

        logger.info("State machine initialized: %s", self._state.value)

    @property
    def state(self) -> PortfolioState:
        return self._state

    @property
    def state_str(self) -> str:
        return self._state.value

    @property
    def state_duration(self) -> float:
        """Seconds in current state."""
        return time.time() - self._state_since

    def transition(self, to_state: PortfolioState, reason: str = "") -> bool:
        """Attempt a state transition. Returns True if successful."""
        valid = TRANSITIONS.get(self._state, [])
        if to_state not in valid:
            logger.warning("Invalid transition: %s → %s (valid: %s)",
                           self._state.value, to_state.value,
                           [s.value for s in valid])
            return False

        from_state = self._state

        # Exit hooks
        for hook in self._exit_hooks.get(from_state, []):
            try:
                hook()
            except Exception as e:
                logger.error("Exit hook failed for %s: %s", from_state.value, e)

        # Transition
        self._state = to_state
        self._state_since = time.time()

        record = StateTransition(
            from_state=from_state.value,
            to_state=to_state.value,
            reason=reason,
        )
        self._history.append(record)
        if len(self._history) > 500:
            self._history = self._history[-200:]

        logger.info("State: %s → %s (%s)", from_state.value, to_state.value, reason)

        # Entry hooks
        for hook in self._entry_hooks.get(to_state, []):
            try:
                hook()
            except Exception as e:
                logger.error("Entry hook failed for %s: %s", to_state.value, e)

        # External callback
        if self._on_transition:
            try:
                self._on_transition(record)
            except Exception as e:
                logger.error("Transition callback failed: %s", e)

        return True

    def on_entry(self, state: PortfolioState, hook: Callable):
        """Register a hook to run when entering a state."""
        self._entry_hooks.setdefault(state, []).append(hook)

    def on_exit(self, state: PortfolioState, hook: Callable):
        """Register a hook to run when exiting a state."""
        self._exit_hooks.setdefault(state, []).append(hook)

    def force_state(self, state: PortfolioState, reason: str = "forced"):
        """Force a state change (bypasses validation). Use for recovery."""
        old = self._state
        self._state = state
        self._state_since = time.time()
        record = StateTransition(from_state=old.value, to_state=state.value, reason=reason)
        self._history.append(record)
        logger.warning("FORCED state: %s → %s (%s)", old.value, state.value, reason)

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get transition history."""
        return [
            {"from": t.from_state, "to": t.to_state, "reason": t.reason, "time": t.timestamp}
            for t in self._history[-limit:]
        ]

    def can_transition(self, to_state: PortfolioState) -> bool:
        """Check if a transition is valid."""
        return to_state in TRANSITIONS.get(self._state, [])
