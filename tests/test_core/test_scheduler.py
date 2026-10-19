"""Tests for core.scheduler — Trading scheduler."""

from datetime import UTC

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


class TestMarketClock:
    """`datetime.now()` on a UTC host (the shipped Docker image) put the
    session seven hours off, and every weekday holiday was treated as a
    trading day."""

    def test_a_utc_host_is_read_as_shanghai_time(self, scheduler):
        from datetime import datetime, timezone

        # 01:35 UTC == 09:35 Beijing on a normal Monday.
        session = scheduler.get_market_session(
            now=datetime(2026, 10, 19, 1, 35, tzinfo=UTC)
        )

        assert session.status == "trading_am"
        assert session.is_trading_day is True

    def test_national_day_is_closed(self, scheduler):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        session = scheduler.get_market_session(
            now=datetime(2026, 10, 1, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        )

        assert session.is_trading_day is False
        assert session.status == "closed_holiday"

    def test_spring_festival_is_closed(self, scheduler):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        session = scheduler.get_market_session(
            now=datetime(2026, 2, 17, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        )

        assert session.status == "closed_holiday"

    def test_a_normal_monday_after_the_holiday_is_open(self, scheduler):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        session = scheduler.get_market_session(
            now=datetime(2026, 10, 8, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        )

        assert session.status == "trading_am"

    def test_extra_holidays_can_be_configured(self, scheduler, monkeypatch):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        monkeypatch.setenv("QUANT_EXCHANGE_HOLIDAYS", "2026-11-02")
        session = scheduler.get_market_session(
            now=datetime(2026, 11, 2, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        )

        assert session.status == "closed_holiday"


class TestEodReconciliationIsIdempotent:
    """`_do_eod_reconciliation` had no marker.

    A scheduler restarted after the close ran it again and appended a second
    P&L snapshot for the same day, double-counting that day in history.
    """

    @staticmethod
    def _scheduler(tmp_path):
        from quant_platform.core.scheduler import TradingScheduler
        from quant_platform.core.state_machine import PortfolioStateMachine

        store = Store(str(tmp_path / "eod.db"))
        scheduler = TradingScheduler(
            state_machine=PortfolioStateMachine(), store=store, bus=EventBus()
        )
        return scheduler, store

    def test_two_runs_in_one_day_write_one_snapshot(self, tmp_path, monkeypatch):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        scheduler, store = self._scheduler(tmp_path)
        # Pin the clock: a run straddling exchange-local midnight would
        # otherwise be two different days.
        monkeypatch.setattr(
            scheduler, "_now",
            lambda: datetime(2026, 10, 20, 15, 35, tzinfo=ZoneInfo("Asia/Shanghai")),
        )

        scheduler._do_eod_reconciliation()
        scheduler._do_eod_reconciliation()

        history = store.get_pnl_history(start_date="2026-10-01",
                                        end_date="2026-10-31")
        assert len(history) == 1

    def test_today_is_marked_when_the_run_completes(self, tmp_path, monkeypatch):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        scheduler, store = self._scheduler(tmp_path)
        monkeypatch.setattr(
            scheduler, "_now",
            lambda: datetime(2026, 10, 20, 15, 35, tzinfo=ZoneInfo("Asia/Shanghai")),
        )

        scheduler._do_eod_reconciliation()

        assert store.has_eod_reconciliation("2026-10-20") is True

    def test_a_new_day_gets_its_own_snapshot(self, tmp_path, monkeypatch):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        scheduler, store = self._scheduler(tmp_path)
        tz = ZoneInfo("Asia/Shanghai")
        now = [datetime(2026, 10, 20, 15, 35, tzinfo=tz)]
        # _do_eod_reconciliation calls _now() twice per run; keep it fixed.
        monkeypatch.setattr(scheduler, "_now", lambda: now[0])

        scheduler._do_eod_reconciliation()
        now[0] = datetime(2026, 10, 21, 15, 35, tzinfo=tz)
        scheduler._do_eod_reconciliation()

        history = store.get_pnl_history(start_date="2026-10-01",
                                        end_date="2026-10-31")
        assert len(history) == 2
        assert store.has_eod_reconciliation("2026-10-20") is True
        assert store.has_eod_reconciliation("2026-10-21") is True

