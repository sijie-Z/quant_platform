"""Trading session scheduler — orchestrates the trading day.

Manages:
- Market hours detection (A-share: 9:30-11:30, 13:00-15:00)
- Pre-market preparation (load data, compute signals)
- Trading session lifecycle
- Post-market reconciliation (save P&L, reconcile positions)
- Rebalance timing

Integrates with EventBus and StateMachine.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from quant_platform.core.events import EventBus, Event, get_event_bus
from quant_platform.core.state_machine import PortfolioStateMachine, PortfolioState
from quant_platform.core.store import Store
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketSession:
    """A-share market session info."""
    date: str = ""
    is_trading_day: bool = True
    market_open: str = "09:30"
    market_close: str = "15:00"
    lunch_start: str = "11:30"
    lunch_end: str = "13:00"
    pre_market: str = "09:15"
    post_market: str = "15:30"
    status: str = "closed"  # pre_market, trading_am, lunch, trading_pm, post_market, closed


class TradingScheduler:
    """Orchestrates trading sessions with market-aware timing.

    Responsibilities:
    - Detect market hours and trading days
    - Trigger pre-market / trading / post-market state transitions
    - Schedule rebalance cycles at configured intervals
    - Handle weekends and holidays
    - Publish timing events on the bus
    """

    def __init__(
        self,
        state_machine: PortfolioStateMachine,
        store: Store,
        bus: EventBus | None = None,
        rebalance_interval: int = 300,  # seconds
    ):
        self._sm = state_machine
        self._store = store
        self._bus = bus or get_event_bus()
        self._rebalance_interval = rebalance_interval

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._session_id = ""
        self._last_rebalance = 0
        self._cycle_count = 0

    def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._session_id = uuid.uuid4().hex[:12]
        self._stop_event.clear()

        # Record session
        self._store.save_session({
            "session_id": self._session_id,
            "status": "active",
            "started_at": datetime.now().isoformat(),
        })

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._bus.publish("scheduler.started", {
            "session_id": self._session_id,
        }, source="scheduler")

        logger.info("Scheduler started. Session: %s", self._session_id)

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

        # Update session
        self._store.save_session({
            "session_id": self._session_id,
            "status": "stopped",
            "stopped_at": datetime.now().isoformat(),
        })

        self._bus.publish("scheduler.stopped", {
            "session_id": self._session_id,
            "cycles": self._cycle_count,
        }, source="scheduler")

        logger.info("Scheduler stopped. Cycles: %d", self._cycle_count)

    def get_market_session(self) -> MarketSession:
        """Get current market session info."""
        now = datetime.now()
        t = now.hour * 100 + now.minute
        weekday = now.weekday()

        session = MarketSession(date=now.strftime("%Y-%m-%d"))

        if weekday >= 5:
            session.is_trading_day = False
            session.status = "closed_weekend"
            return session

        if t < 915:
            session.status = "closed"
        elif 915 <= t < 930:
            session.status = "pre_market"
        elif 930 <= t < 1130:
            session.status = "trading_am"
        elif 1130 <= t < 1300:
            session.status = "lunch"
        elif 1300 <= t < 1457:
            session.status = "trading_pm"
        elif 1457 <= t < 1530:
            session.status = "post_market"
        else:
            session.status = "closed"

        return session

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        s = self.get_market_session()
        return s.status in ("trading_am", "trading_pm")

    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance."""
        if not self.is_market_open():
            return False
        return time.time() - self._last_rebalance >= self._rebalance_interval

    def get_state(self) -> dict:
        """Get scheduler state."""
        ms = self.get_market_session()
        return {
            "running": self._running,
            "session_id": self._session_id,
            "market_status": ms.status,
            "is_trading_day": ms.is_trading_day,
            "is_market_open": self.is_market_open(),
            "cycle_count": self._cycle_count,
            "last_rebalance": self._last_rebalance,
            "rebalance_interval": self._rebalance_interval,
        }

    def _run_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.error("Scheduler tick error: %s", e)

            self._stop_event.wait(timeout=30)  # Check every 30 seconds

    def _tick(self):
        """Single scheduler tick — check market state and trigger transitions."""
        ms = self.get_market_session()

        # State transitions based on market hours
        if ms.status == "pre_market" and self._sm.can_transition(PortfolioState.PRE_MARKET):
            self._sm.transition(PortfolioState.PRE_MARKET, "market pre-open")
            self._bus.publish("market.pre_open", {"date": ms.date}, source="scheduler")

        elif ms.status in ("trading_am", "trading_pm") and self._sm.can_transition(PortfolioState.TRADING):
            if self._sm.state != PortfolioState.TRADING and self._sm.state != PortfolioState.REBALANCING:
                self._sm.transition(PortfolioState.TRADING, "market open")

        elif ms.status in ("post_market", "closed") and self._sm.state == PortfolioState.TRADING:
            self._sm.transition(PortfolioState.POST_MARKET, "market closed")
            self._do_eod_reconciliation()

        elif ms.status in ("post_market", "closed") and self._sm.state == PortfolioState.POST_MARKET:
            # After post-market, go to ready
            self._sm.transition(PortfolioState.READY, "post-market complete")

    def _do_eod_reconciliation(self):
        """End-of-day reconciliation: save P&L snapshot, log events."""
        try:
            positions = self._store.get_positions()
            market_value = sum(p.get('market_value', 0) for p in positions)

            # Get latest account info from store
            pnl_history = self._store.get_pnl_history(days=1)
            last_equity = pnl_history[-1]['total_equity'] if pnl_history else 1_000_000

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "total_equity": last_equity,
                "cash": last_equity - market_value,
                "market_value": market_value,
                "n_positions": len(positions),
            }
            self._store.save_pnl_snapshot(snapshot)

            self._bus.publish("eod.reconciliation", {
                "positions": len(positions),
                "market_value": market_value,
            }, source="scheduler")

            logger.info("EOD reconciliation: %d positions, value=%.2f",
                        len(positions), market_value)
        except Exception as e:
            logger.error("EOD reconciliation failed: %s", e)
