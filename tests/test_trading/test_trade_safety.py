"""S5: the CLI/live-runner trade path must fail closed.

The old path ignored ``connect() == False``, silently replaced QMT with a
simulated broker, fed random-walk prices into real orders, back-dated P&L rows
onto calendar days and exited 0 even when every order was rejected.
"""

from __future__ import annotations

from datetime import date

import pytest

import quant_platform.trading.live_runner as live_runner_module
from quant_platform.core.store import Store
from quant_platform.execution.models import OrderStatus
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import SimulatedBroker
from quant_platform.trading.live_runner import LiveRunner
from quant_platform.utils.exchange_calendar import is_trading_day


class FakeRealBroker:
    """A non-simulated broker (QMT/XTP-shaped) that records orders."""

    def __init__(self, cash: float = 1_000_000.0):
        self.orders = []
        self._cash = cash
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def place_order(self, order):
        order.filled_quantity = order.quantity
        order.filled_price = order.price
        order.status = OrderStatus.FILLED
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self):
        return []

    def get_orders(self, status=None):
        return list(self.orders)

    def get_account(self):
        return {
            "cash": self._cash,
            "total_equity": self._cash,
            "total_pnl": 0.0,
            "n_positions": 0,
            "connected": True,
        }

    def new_trading_day(self):
        pass


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "trade.db"))


def _runner(store, broker, monkeypatch, dual_track=False):
    monkeypatch.setattr(
        live_runner_module, "is_trading_day", lambda day: True,
    )
    return LiveRunner(
        broker=broker,
        store=store,
        risk_monitor=RiskMonitor(),
        dual_track=dual_track,
    )


def test_connect_returning_false_is_a_hard_failure(store):
    class NeverConnects(FakeRealBroker):
        def connect(self):
            return False

    with pytest.raises(RuntimeError, match="failed to connect"):
        LiveRunner(
            broker=NeverConnects(),
            store=store,
            risk_monitor=RiskMonitor(),
            dual_track=False,
        )


def test_qmt_creation_failure_does_not_fall_back_to_simulated(
    store, monkeypatch,
):
    """A missing xtquant used to be swallowed and paper trading continued."""

    def fake_create(name, **kwargs):
        if name.lower() in ("qmt", "qmt_sim", "qmt_live"):
            raise RuntimeError("xtquant not installed")
        return SimulatedBroker(initial_cash=kwargs.get("initial_cash", 1e7))

    monkeypatch.setattr(live_runner_module, "create_broker", fake_create)

    with pytest.raises(RuntimeError, match="xtquant not installed"):
        LiveRunner(
            broker_type="qmt",
            store=store,
            risk_monitor=RiskMonitor(),
            dual_track=False,
        )


def test_real_broker_without_a_price_feed_refuses_to_run(
    store, monkeypatch,
):
    broker = FakeRealBroker()
    runner = _runner(store, broker, monkeypatch)
    runner.set_universe(["600519"])

    with pytest.raises(RuntimeError, match="market-data feed"):
        runner.run(days=1)

    assert broker.orders == []


def test_real_broker_orders_use_fresh_feed_prices(store, monkeypatch):
    broker = FakeRealBroker()
    runner = _runner(store, broker, monkeypatch)
    runner.set_universe(["600519"])
    monkeypatch.setattr(runner, "_generate_signals", lambda: [{
        "code": "600519", "side": "buy", "target_value": 20_000.0,
        "strength": 1.0, "reason": "test",
    }])
    feed_calls = []

    def feed(codes):
        feed_calls.append(list(codes))
        return {"600519": 100.0}

    runner.set_price_provider(feed)
    report = runner.run(days=1)

    assert feed_calls == [["600519.SH"]]
    assert broker.orders, "the signal should have produced an order"
    assert all(order.price == 100.0 for order in broker.orders)
    assert report.total_fills == 1
    assert report.status == "completed"


def test_empty_feed_skips_the_cycle_and_marks_the_session_failed(
    store, monkeypatch,
):
    broker = FakeRealBroker()
    runner = _runner(store, broker, monkeypatch)
    runner.set_universe(["600519"])
    monkeypatch.setattr(runner, "_generate_signals", lambda: [{
        "code": "600519", "side": "buy", "target_value": 20_000.0,
        "strength": 1.0, "reason": "test",
    }])

    runner.set_price_provider(lambda codes: {})
    report = runner.run(days=1)

    assert broker.orders == []
    assert report.failed_cycles == 1
    assert report.total_fills == 0
    assert report.status == "failed"


def test_simulated_runs_iterate_exchange_sessions_not_calendar_days(
    store, monkeypatch,
):
    """Old code back-dated calendar days, including weekends/holidays."""
    runner = LiveRunner(
        broker_type="simulated", store=store,
        risk_monitor=RiskMonitor(), dual_track=False,
    )
    runner.set_universe(["600519", "000001"])

    report = runner.run(days=5, seed=1)

    assert report.days_traded == 5
    for daily in report.daily_reports:
        assert is_trading_day(date.fromisoformat(daily["date"]))


def test_real_broker_multi_day_run_is_rejected(store, monkeypatch):
    runner = _runner(store, FakeRealBroker(), monkeypatch)
    runner.set_universe(["600519"])
    runner.set_price_provider(lambda codes: {"600519": 10.0})

    with pytest.raises(ValueError, match="one exchange day"):
        runner.run(days=2)


def test_zero_fills_mark_the_session_failed(store, monkeypatch):
    """Simulated runs that cannot trade must not report completed."""
    runner = LiveRunner(
        broker_type="simulated", store=store,
        risk_monitor=RiskMonitor(), dual_track=False,
    )
    runner.set_universe(["600519"])
    monkeypatch.setattr(runner, "_generate_signals", lambda: [])

    report = runner.run(days=1, seed=1)

    assert report.total_fills == 0
    assert report.status == "failed"
