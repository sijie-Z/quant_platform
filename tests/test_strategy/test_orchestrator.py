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

        # Next session: signal changes — exit A0001. The position was bought
        # today, so T+1 makes it unsellable until the date advances.
        o.on_signal("2025-01-02", pd.Series(dtype=float), strategy_id=sid)
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


class TestPerStrategyPositionAttribution:
    """The orchestrator used to compare each strategy's target with the whole
    shared book.

    `ExecutionEngine.positions` is one book with no strategy attribution.
    With two strategies each targeting 150 shares, the first strategy's buy
    made the second believe it already held the shares; the next rebalance
    then saw 300 shares against a 150-share target and made *both* strategies
    sell 150, oscillating the book. Fills now update a per-strategy ledger
    keyed by `order.strategy`, and P&L follows the lots a strategy owns.
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
        return orch, ms, a, b

    def test_each_strategy_owns_its_own_lots(self, two_strategies):
        orch, _, a, b = two_strategies

        # 500k allocation / 50 positions x 1.5 signal / 100 per share,
        # rounded down to one lot: 100 shares per strategy.
        assert orch.exec_engine.get_position("A0001").quantity == 200
        assert orch._books[a]["A0001"]["qty"] == 100
        assert orch._books[b]["A0001"]["qty"] == 100

    def test_rebalance_does_not_react_to_the_other_strategys_position(
        self, two_strategies,
    ):
        orch, _, _, _ = two_strategies

        assert orch.rebalance() == []

    def test_pnl_follows_the_lots_a_strategy_owns(self, two_strategies):
        orch, ms, a, b = two_strategies

        orch._update_strategy_pnl({"A0001": 110.0})

        # Identical allocations and fills: the two strategies must report the
        # same number, and it must be positive, not the double-credited book.
        assert ms.states[a].total_pnl == pytest.approx(ms.states[b].total_pnl)
        assert ms.states[a].total_pnl > 0
