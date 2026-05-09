"""Tests for risk.circuit_breaker — RiskMonitor and circuit breakers."""

import pytest
from quant_platform.risk.circuit_breaker import (
    RiskMonitor, RiskLimits, RiskLevel, BreachType, RiskBreach,
)


class TestRiskMonitor:
    def setup_method(self):
        self.monitor = RiskMonitor()

    def test_initial_state(self):
        assert self.monitor.risk_level == RiskLevel.GREEN
        assert self.monitor.kill_switch_active is False

    def test_pre_trade_approved(self):
        self.monitor.update_portfolio_state(
            portfolio_value=10_000_000, daily_pnl=0, positions={}, sector_weights={},
        )
        approved, breaches = self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert approved is True
        assert len(breaches) == 0

    def test_position_limit_breach(self):
        self.monitor.update_portfolio_state(
            portfolio_value=10_000_000, daily_pnl=0, positions={}, sector_weights={},
        )
        approved, breaches = self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 6000, "price": 100,
        })
        assert approved is False
        assert any(b.breach_type == BreachType.POSITION_LIMIT for b in breaches)

    def test_daily_loss_breach(self):
        self.monitor.update_portfolio_state(
            portfolio_value=10_000_000, daily_pnl=-400_000, positions={}, sector_weights={},
        )
        approved, breaches = self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert any(b.breach_type == BreachType.DAILY_LOSS for b in breaches)

    def test_drawdown_circuit_breaker(self):
        self.monitor.update_portfolio_state(
            portfolio_value=8_400_000, daily_pnl=0, positions={}, sector_weights={},
        )
        self.monitor.peak_value = 10_000_000  # 16% drawdown > 15% limit
        approved, breaches = self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert any(b.breach_type == BreachType.DRAWDOWN for b in breaches)

    def test_kill_switch_blocks_orders(self):
        self.monitor.activate_kill_switch("test")
        approved, breaches = self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert approved is False
        assert self.monitor.kill_switch_active is True

    def test_deactivate_kill_switch(self):
        self.monitor.activate_kill_switch("test")
        self.monitor.deactivate_kill_switch()
        assert self.monitor.kill_switch_active is False
        assert self.monitor.risk_level == RiskLevel.YELLOW

    def test_kill_drawdown_auto_activates(self):
        self.monitor.update_portfolio_state(
            portfolio_value=7_000_000, daily_pnl=0, positions={}, sector_weights={},
        )
        self.monitor.peak_value = 10_000_000  # 30% drawdown > 25% kill threshold
        self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert self.monitor.kill_switch_active is True
        assert self.monitor.risk_level == RiskLevel.KILL

    def test_sector_limit_breach(self):
        self.monitor.update_portfolio_state(
            portfolio_value=10_000_000, daily_pnl=0,
            positions={"600519": {"value": 500_000, "weight": 0.05, "sector": "白酒"}},
            sector_weights={"白酒": 0.35},
        )
        approved, breaches = self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert any(b.breach_type == BreachType.SECTOR_LIMIT for b in breaches)

    def test_get_status(self):
        self.monitor.update_portfolio_state(
            portfolio_value=10_000_000, daily_pnl=50_000,
            positions={"600519": {"value": 500_000}}, sector_weights={},
        )
        status = self.monitor.get_status()
        assert status["risk_level"] == "green"
        assert status["portfolio_value"] == 10_000_000

    def test_breach_history(self):
        self.monitor.update_portfolio_state(
            portfolio_value=10_000_000, daily_pnl=-400_000, positions={}, sector_weights={},
        )
        self.monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert len(self.monitor.breaches) > 0
