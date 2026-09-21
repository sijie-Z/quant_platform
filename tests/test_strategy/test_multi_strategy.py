"""Tests for strategy.multi_strategy — Multi-strategy manager."""

import numpy as np
import pytest

from quant_platform.strategy.multi_strategy import (
    MultiStrategyManager,
    StrategyConfig,
)


class TestMultiStrategyManager:
    def setup_method(self):
        self.mgr = MultiStrategyManager(total_capital=100_000_000)

    def test_add_strategy(self):
        config = StrategyConfig(name="momentum", allocation_pct=0.5)
        sid = self.mgr.add_strategy(config)
        assert sid in self.mgr.strategies
        assert self.mgr.states[sid].capital_allocated == 50_000_000

    def test_remove_strategy(self):
        config = StrategyConfig(name="test", allocation_pct=1.0)
        sid = self.mgr.add_strategy(config)
        self.mgr.remove_strategy(sid)
        assert sid not in self.mgr.strategies

    def test_update_pnl(self):
        config = StrategyConfig(name="test", allocation_pct=1.0)
        sid = self.mgr.add_strategy(config)
        self.mgr.update_strategy_pnl(sid, 0.01)  # 1% return
        state = self.mgr.states[sid]
        assert state.daily_pnl > 0
        assert state.total_return > 0

    def test_aggregate_metrics(self):
        c1 = StrategyConfig(name="A", allocation_pct=0.6)
        c2 = StrategyConfig(name="B", allocation_pct=0.4)
        s1 = self.mgr.add_strategy(c1)
        s2 = self.mgr.add_strategy(c2)
        self.mgr.update_strategy_pnl(s1, 0.01)
        self.mgr.update_strategy_pnl(s2, 0.005)
        metrics = self.mgr.get_aggregate_metrics()
        assert metrics["n_strategies"] == 2
        assert metrics["total_capital"] == 100_000_000

    def test_allocate_capital(self):
        c1 = StrategyConfig(name="A", allocation_pct=0.5)
        c2 = StrategyConfig(name="B", allocation_pct=0.5)
        s1 = self.mgr.add_strategy(c1)
        s2 = self.mgr.add_strategy(c2)
        self.mgr.allocate_capital({s1: 0.7, s2: 0.3})
        assert self.mgr.strategies[s1].allocation_pct == 0.7

    def test_risk_alerts_on_drawdown(self):
        config = StrategyConfig(name="test", allocation_pct=1.0, max_drawdown_limit=0.10)
        sid = self.mgr.add_strategy(config)
        # Simulate losses
        for _ in range(20):
            self.mgr.update_strategy_pnl(sid, -0.01)
        alerts = self.mgr.get_risk_alerts()
        assert len(alerts) > 0
        assert any(a["type"] == "drawdown_breach" for a in alerts)

    def test_correlation_matrix(self):
        c1 = StrategyConfig(name="A", allocation_pct=0.5)
        c2 = StrategyConfig(name="B", allocation_pct=0.5)
        s1 = self.mgr.add_strategy(c1)
        s2 = self.mgr.add_strategy(c2)
        np.random.seed(42)
        for _ in range(100):
            self.mgr.update_strategy_pnl(s1, np.random.normal(0, 0.01))
            self.mgr.update_strategy_pnl(s2, np.random.normal(0, 0.01))
        metrics = self.mgr.get_aggregate_metrics()
        assert "correlation_matrix" in metrics


class TestAFundedStrategyStartsAtItsAllocation:
    """`StrategyState.current_value` defaults to 0 and `update_strategy_pnl`
    only ever adds to it, so a funded strategy's first P&L update computed

        total_pnl = 0 + daily_pnl - capital_allocated

    A 100M strategy that earned 1M on its first day therefore reported a -99%
    return, dragged `get_aggregate_metrics()["total_value"]` down to one day's
    P&L, and tripped the `total_return < -0.10` loss alert.

    The previous tests worked around this by assigning `current_value`
    by hand, with a comment saying so -- which is why it went unnoticed."""

    def setup_method(self):
        self.mgr = MultiStrategyManager(total_capital=100_000_000)

    def test_current_value_starts_at_the_allocation(self):
        sid = self.mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.5))

        state = self.mgr.states[sid]
        assert state.current_value == 50_000_000
        assert state.total_pnl == 0
        assert state.total_return == 0

    def test_the_first_day_reports_the_first_day_return(self):
        sid = self.mgr.add_strategy(StrategyConfig(name="A", allocation_pct=1.0))

        self.mgr.update_strategy_pnl(sid, 0.01)

        state = self.mgr.states[sid]
        assert state.current_value == pytest.approx(101_000_000)
        assert state.total_pnl == pytest.approx(1_000_000)
        assert state.total_return == pytest.approx(0.01)

    def test_a_gain_does_not_raise_a_loss_alert(self):
        self.mgr.add_strategy(StrategyConfig(name="A", allocation_pct=1.0))
        sid = list(self.mgr.states)[0]

        self.mgr.update_strategy_pnl(sid, 0.01)

        assert [a for a in self.mgr.get_risk_alerts() if a["type"] == "loss_alert"] == []

    def test_aggregate_value_is_not_just_one_days_pnl(self):
        s1 = self.mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.6))
        s2 = self.mgr.add_strategy(StrategyConfig(name="B", allocation_pct=0.4))

        self.mgr.update_strategy_pnl(s1, 0.01)
        self.mgr.update_strategy_pnl(s2, 0.0)

        metrics = self.mgr.get_aggregate_metrics()
        assert metrics["total_value"] == pytest.approx(100_600_000)
