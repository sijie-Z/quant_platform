"""Tests for strategy.multi_strategy — Multi-strategy manager."""

import numpy as np
import pytest
from quant_platform.strategy.multi_strategy import (
    MultiStrategyManager, StrategyConfig, StrategyState,
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
        # Initialize current_value to capital_allocated
        self.mgr.states[sid].current_value = self.mgr.states[sid].capital_allocated
        self.mgr.update_strategy_pnl(sid, 0.01)  # 1% return
        state = self.mgr.states[sid]
        assert state.daily_pnl > 0
        assert state.total_return > 0

    def test_aggregate_metrics(self):
        c1 = StrategyConfig(name="A", allocation_pct=0.6)
        c2 = StrategyConfig(name="B", allocation_pct=0.4)
        s1 = self.mgr.add_strategy(c1)
        s2 = self.mgr.add_strategy(c2)
        self.mgr.states[s1].current_value = self.mgr.states[s1].capital_allocated
        self.mgr.states[s2].current_value = self.mgr.states[s2].capital_allocated
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
        self.mgr.states[sid].current_value = self.mgr.states[sid].capital_allocated
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
