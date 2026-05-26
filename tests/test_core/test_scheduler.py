"""Tests for core.scheduler — Trading scheduler."""

import pytest

from quant_platform.core.events import EventBus
from quant_platform.core.scheduler import TradingScheduler
from quant_platform.core.state_machine import PortfolioStateMachine
from quant_platform.core.store import Store


@pytest.fixture
def scheduler(tmp_path):
    sm = PortfolioStateMachine()
    bus = EventBus()
    store = Store(str(tmp_path / "test.db"))
    return TradingScheduler(state_machine=sm, store=store, bus=bus)


class TestTradingScheduler:
    def test_initialization(self, scheduler):
        assert scheduler is not None

    def test_get_market_session(self, scheduler):
        session = scheduler.get_market_session()
        assert session is not None
        assert hasattr(session, 'status')
