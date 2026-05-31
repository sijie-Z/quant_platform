"""Tests for PortfolioOrchestrator."""

import pandas as pd
import pytest

from quant_platform.execution.engine import ExecutionEngine, OrderSide
from quant_platform.execution.models import OrderStatus
from quant_platform.strategy.multi_strategy import MultiStrategyManager, StrategyConfig
from quant_platform.strategy.portfolio_orchestrator import PortfolioOrchestrator


@pytest.fixture
def orch():
    ms = MultiStrategyManager(total_capital=1_000_000)
    sid = ms.add_strategy(StrategyConfig(
        name="test", allocation_pct=1.0, is_active=True,
    ))
    engine = ExecutionEngine()
    return PortfolioOrchestrator(ms, exec_engine=engine), sid


class TestPortfolioOrchestrator:
    def test_on_signal_creates_targets(self, orch):
        o, sid = orch
        signal = pd.Series({"A0001": 0.5, "A0002": 0.3, "A0003": 0.1})
        o._last_prices = {"A0001": 100.0, "A0002": 50.0, "A0003": 20.0}
        o.on_signal("2025-01-01", signal, strategy_id=sid)
        assert len(o._targets[sid]) > 0

    def test_rebalance_creates_orders(self, orch):
        o, sid = orch
        signal = pd.Series({"A0001": 0.5, "A0002": 0.3})
        o._last_prices = {"A0001": 100.0, "A0002": 50.0}
        o.on_signal("2025-01-01", signal, strategy_id=sid)
        orders = o.rebalance()
        assert len(orders) > 0
        assert orders[0]["side"] == "buy"

    def test_rebalance_sells_untargeted(self, orch):
        o, sid = orch
        # First establish a position
        signal = pd.Series({"A0001": 0.5})
        o._last_prices = {"A0001": 100.0}
        o.on_signal("2025-01-01", signal, strategy_id=sid)
        o.rebalance()
        o.process_fills({"A0001": 100.0})

        # Now signal changes — exit A0001
        o._targets["test"] = {}
        orders = o.rebalance()
        sells = [o for o in orders if o["side"] == "sell"]
        assert len(sells) > 0
        assert sells[0]["ticker"] == "A0001"

    def test_process_fills_updates_positions(self, orch):
        o, sid = orch
        signal = pd.Series({"A0001": 0.5})
        o._last_prices = {"A0001": 100.0}
        o.on_signal("2025-01-01", signal, strategy_id=sid)
        o.rebalance()
        o.process_fills({"A0001": 100.0})
        pos = o.exec_engine.get_position("A0001")
        assert pos is not None
        assert pos.quantity > 0

    def test_cash_available(self, orch):
        o, _ = orch
        assert o.cash_available == 1_000_000.0

    def test_portfolio_summary(self, orch):
        o, _ = orch
        summary = o.portfolio_summary()
        assert summary["n_positions"] == 0
        assert "cash_available" in summary
