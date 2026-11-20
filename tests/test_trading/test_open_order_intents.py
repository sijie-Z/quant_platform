"""Working broker orders must suppress duplicate trading intents."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant_platform.core.store import Store
from quant_platform.execution.models import OrderStatus
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import (
    Order,
    OrderSide,
    Position,
    SimulatedBroker,
)
from quant_platform.trading.engine import CycleResult, LiveTradingEngine
from quant_platform.trading.live_runner import LiveRunner


def _open_order(code: str = "600519", status=OrderStatus.SUBMITTED,
                filled: int = 0) -> Order:
    return Order(
        code=code, side=OrderSide.BUY, quantity=100, price=10.0,
        filled_quantity=filled, status=status,
    )


def _engine(tmp_path, monkeypatch, orders, broker=None):
    broker = broker or SimulatedBroker(initial_cash=1_000_000)
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
    )
    engine._price_history = {"600519": [10.0 + i * 0.01 for i in range(30)]}
    engine._score_prices = lambda _frame: pd.Series({"600519": 0.9})
    engine._price_last_seen = {"600519": time.monotonic()}
    engine._last_fetch_codes = {"600519"}
    monkeypatch.setattr(broker, "get_orders", lambda status=None: orders)
    return engine


@pytest.mark.parametrize("status,filled", [
    (OrderStatus.PENDING, 0),
    (OrderStatus.SUBMITTED, 0),
])
def test_engine_open_buy_suppresses_duplicate(tmp_path, monkeypatch, status, filled):
    engine = _engine(
        tmp_path, monkeypatch, [_open_order(status=status, filled=filled)],
    )
    assert engine._generate_signals(positions=[]) == []


def test_qmt_partial_order_still_suppresses_duplicate(tmp_path, monkeypatch):
    broker = MagicMock()
    broker.get_account.return_value = {"total_equity": 1_000_000.0}
    orders = [_open_order(status=OrderStatus.PARTIAL, filled=40)]
    broker.get_orders_authoritative.return_value = orders
    engine = _engine(tmp_path, monkeypatch, orders, broker=broker)
    assert engine._generate_signals(positions=[]) == []


def test_simulated_ioc_partial_does_not_freeze_symbol(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path, monkeypatch,
        [_open_order(status=OrderStatus.PARTIAL, filled=40)],
    )
    signals = engine._generate_signals(positions=[])
    assert len(signals) == 1
    assert signals[0].side == OrderSide.BUY


def test_engine_filled_order_does_not_block_new_intent(tmp_path, monkeypatch):
    engine = _engine(
        tmp_path, monkeypatch,
        [_open_order(status=OrderStatus.FILLED, filled=100)],
    )
    signals = engine._generate_signals(positions=[])
    assert len(signals) == 1
    assert signals[0].side == OrderSide.BUY


def test_engine_order_query_failure_fails_closed(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=1_000_000)
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
    )
    engine._price_history = {"600519": [10.0] * 30}
    engine._score_prices = lambda _frame: pd.Series({"600519": 0.9})

    def fail_orders():
        raise RuntimeError("broker order query unavailable")

    monkeypatch.setattr(broker, "get_orders", fail_orders)
    assert engine._generate_signals(positions=[]) == []


class _DelayedSellBroker:
    def __init__(self):
        self._position = Position(
            code="600519", quantity=100, available=100, market_value=1000.0,
        )
        self._orders = []
        self.placed = []

    def get_positions(self):
        return [self._position]

    def get_orders_authoritative(self):
        return list(self._orders)

    def place_order(self, order):
        order.status = OrderStatus.SUBMITTED
        self._orders.append(order)
        self.placed.append(order)
        return order


def test_kill_switch_delayed_sell_is_not_repeated(tmp_path):
    broker = _DelayedSellBroker()
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
    )
    cycle = CycleResult(cycle_id=1, timestamp="2026-10-26T10:00:00")

    engine._deleverage(cycle)
    engine._deleverage(cycle)

    assert len(broker.placed) == 1
    assert broker.placed[0].quantity == 100
    assert len(cycle.orders) == 1


def _runner(monkeypatch, orders):
    runner = LiveRunner(broker_type="simulated", dual_track=False)
    runner.set_universe(["600519"])
    runner.set_prices({"600519": 100.0})
    runner._score_prices = lambda: pd.Series({"600519.SH": 0.9})
    monkeypatch.setattr(runner._broker, "get_orders", lambda status=None: orders)
    return runner


def test_live_runner_open_buy_suppresses_duplicate(monkeypatch):
    runner = _runner(monkeypatch, [_open_order()])
    assert runner._generate_signals() == []


def test_live_runner_without_open_order_keeps_shortfall_signal(monkeypatch):
    runner = _runner(monkeypatch, [])
    signals = runner._generate_signals()
    assert len(signals) == 1
    assert signals[0]["side"] == "buy"
    assert signals[0]["target_value"] > 0


def test_live_runner_ioc_partial_allows_new_intent(monkeypatch):
    runner = _runner(
        monkeypatch, [_open_order(status=OrderStatus.PARTIAL, filled=40)],
    )
    signals = runner._generate_signals()
    assert len(signals) == 1
    assert signals[0]["side"] == "buy"


def test_live_runner_order_query_failure_fails_closed(monkeypatch):
    runner = LiveRunner(broker_type="simulated", dual_track=False)
    runner.set_universe(["600519"])
    runner.set_prices({"600519": 100.0})
    runner._score_prices = lambda: pd.Series({"600519.SH": 0.9})

    def fail_orders():
        raise RuntimeError("broker order query unavailable")

    monkeypatch.setattr(runner._broker, "get_orders", fail_orders)
    assert runner._generate_signals() == []
