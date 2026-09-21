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


class TestPerStrategyPnlIsNotMisattributed:
    """`_update_strategy_pnl` credited every strategy with the whole book.

    It computed `total_pnl` over `self.exec_engine.positions` -- one shared
    book with no strategy attribution -- divided by a single strategy's
    capital, and pushed that as the strategy's own daily return. With N
    strategies the same book P&L is counted N times.

    An `if state.current_value > 0` guard made it a no-op, until the fix for
    BUG-06 made a funded strategy start at its allocation and the guard began
    to pass.
    """

    @pytest.fixture
    def two_strategies(self):
        ms = MultiStrategyManager(total_capital=1_000_000)
        a = ms.add_strategy(StrategyConfig(name="A", allocation_pct=0.5, is_active=True))
        b = ms.add_strategy(StrategyConfig(name="B", allocation_pct=0.5, is_active=True))
        orch = PortfolioOrchestrator(ms, exec_engine=ExecutionEngine())
        orch._last_prices = {"A0001": 100.0}
        for sid in (a, b):
            orch.on_signal("2025-01-01", pd.Series({"A0001": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"A0001": 100.0})
        # `ExecutionEngine.positions` is a property returning a *copy*, so a
        # test cannot populate the book through it -- the first draft appended
        # to a throwaway list and asserted nothing. Reach the real dict.
        assert orch.exec_engine._positions, "fixture produced no shared position"
        for position in orch.exec_engine._positions.values():
            position.unrealized_pnl = 50_000.0
        return orch, ms, a, b

    def test_the_shared_book_pnl_is_not_credited_to_each_strategy(self, two_strategies):
        orch, ms, a, b = two_strategies

        before = {sid: ms.states[sid].total_pnl for sid in (a, b)}
        orch._update_strategy_pnl()
        after = {sid: ms.states[sid].total_pnl for sid in (a, b)}

        assert after == before, (
            "a strategy was credited with the whole shared book's P&L"
        )
