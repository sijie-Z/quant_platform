"""Tests for risk.circuit_breaker — RiskMonitor and circuit breakers."""

import pytest
from quant_platform.risk.circuit_breaker import (
    BreachType,
    RiskLevel,
    RiskMonitor,
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


class TestInstrumentKeyAliases:
    """Regression for BUG-20.

    Position and sector limits read `order["ticker"]`. `trading/live_runner.py`
    passed `{"code": ...}` -- the name `Order`, the positions table and the rest
    of the codebase use -- so the lookup resolved to an empty ticker, found no
    existing position, and re-checked the limit from zero. A held position of
    49% against a 5% single-name limit was approved.
    """

    def _monitor_holding(self):
        monitor = RiskMonitor()
        monitor.update_portfolio_state(
            portfolio_value=1_000_000,
            daily_pnl=0,
            positions={"600519": {"value": 490_000, "sector": "Tech"}},
            sector_weights={"Tech": 0.49},
        )
        return monitor

    @pytest.mark.parametrize("key", ["ticker", "code", "symbol"])
    def test_every_instrument_key_alias_hits_the_limit_check(self, key):
        """A new order of 10k on top of a 49% holding must breach."""
        monitor = self._monitor_holding()
        approved, breaches = monitor.check_pre_trade({
            key: "600519", "side": "buy", "quantity": 100, "price": 100,
        })
        assert approved is False, f"{key!r} bypassed the position limit"
        assert any(b.breach_type == BreachType.POSITION_LIMIT for b in breaches)

    def test_order_without_an_identifier_is_blocked(self):
        """Risk checks fail closed: with no identifier the limits cannot be
        evaluated, so approving would be a silent bypass."""
        monitor = self._monitor_holding()
        approved, breaches = monitor.check_pre_trade({
            "side": "buy", "quantity": 100, "price": 100,
        })
        assert approved is False
        assert breaches

    def test_unknown_identifier_is_still_checked_not_short_circuited(self):
        """An identifier that simply has no position yet is legitimate -- it
        must be checked against the limit, not rejected."""
        monitor = self._monitor_holding()
        approved, breaches = monitor.check_pre_trade({
            "code": "000001", "side": "buy", "quantity": 100, "price": 100,
        })
        assert approved is True
        assert breaches == []


class TestDirectionAwareChecks:
    """Sells reduce exposure and must not be blocked like buys."""

    def _monitor(self, daily_pnl: float = 0.0) -> RiskMonitor:
        monitor = RiskMonitor()
        monitor.update_portfolio_state(
            portfolio_value=10_000_000,
            daily_pnl=daily_pnl,
            positions={
                "600519": {"value": 1_000_000, "weight": 0.10, "sector": "Tech"},
            },
            sector_weights={"Tech": 0.10},
        )
        return monitor

    def test_selling_is_not_blocked_by_the_position_cap(self):
        approved, breaches = self._monitor().check_pre_trade({
            "ticker": "600519", "side": "sell", "quantity": 10_000, "price": 100,
        })

        assert approved is True
        assert not any(b.breach_type == BreachType.POSITION_LIMIT for b in breaches)

    def test_sell_is_allowed_after_a_daily_loss_breach(self):
        approved, _ = self._monitor(daily_pnl=-400_000).check_pre_trade({
            "ticker": "600519", "side": "sell", "quantity": 1_000, "price": 100,
        })

        assert approved is True

    def test_kill_switch_still_allows_a_reducing_sell(self):
        monitor = self._monitor()
        monitor.kill_switch_active = True

        approved, _ = monitor.check_pre_trade({
            "ticker": "600519", "side": "sell", "quantity": 1_000, "price": 100,
        })

        assert approved is True

    def test_sector_limit_includes_the_incoming_order(self):
        monitor = RiskMonitor()
        monitor.update_portfolio_state(
            portfolio_value=10_000_000,
            daily_pnl=0,
            positions={"NEW": {"value": 0, "weight": 0, "sector": "Tech"}},
            sector_weights={"Tech": 0.29},
        )

        approved, breaches = monitor.check_pre_trade({
            "ticker": "NEW", "side": "buy", "quantity": 5_000, "price": 100,
        })

        assert approved is False
        assert any(b.breach_type == BreachType.SECTOR_LIMIT for b in breaches)


class TestPortfolioUpdatesRunTheEquityBreakers:
    """Breakdowns were evaluated only inside `check_pre_trade`.

    A fully invested book can lose 30% without emitting a single order, so
    the drawdown breaker never fired and the risk level stayed GREEN until
    someone tried to trade.
    """

    def test_drawdown_beyond_the_kill_threshold_arms_the_switch(self):
        monitor = RiskMonitor()
        monitor.update_portfolio_state(1_000_000, 0, {}, {})
        monitor.update_portfolio_state(700_000, 0, {}, {})

        assert monitor.kill_switch_active is True
        assert monitor.get_status()["risk_level"] == RiskLevel.KILL.value

    def test_daily_loss_sets_a_red_level(self):
        monitor = RiskMonitor()
        monitor.update_portfolio_state(1_000_000, 0, {}, {})
        monitor.update_portfolio_state(1_000_000, -40_000, {}, {})

        assert monitor.get_status()["risk_level"] == RiskLevel.RED.value
        assert monitor.kill_switch_active is False

    def test_repeated_updates_do_not_flood_the_breach_log(self):
        monitor = RiskMonitor()
        monitor.update_portfolio_state(1_000_000, 0, {}, {})
        monitor.update_portfolio_state(700_000, 0, {}, {})
        count = len(monitor.breaches)

        monitor.update_portfolio_state(700_000, 0, {}, {})

        assert len(monitor.breaches) == count


class TestPendingOrdersShareTheSectorBudget:
    def test_a_batch_of_buys_cannot_all_pass_the_sector_limit(self):
        monitor = RiskMonitor()
        monitor.update_portfolio_state(1_000_000, 0, {}, {})
        codes = [f"S{i}" for i in range(8)]
        monitor.set_sector_map({code: "Tech" for code in codes})

        approved_flags = []
        for code in codes:
            order = {
                "ticker": code, "side": "buy", "quantity": 1_000, "price": 40.0,
            }
            approved, _ = monitor.check_pre_trade(order)
            approved_flags.append(approved)
            if approved:
                monitor.project_order(order)

        # 4% of equity each: seven orders reach 28%, the eighth would be 32%.
        assert approved_flags[:7] == [True] * 7
        assert approved_flags[7] is False

    def test_unknown_sector_comes_from_the_sector_map(self):
        monitor = RiskMonitor()
        monitor.set_sector_map({"600519": "食品饮料"})

        assert monitor._resolve_sector("600519") == "食品饮料"
        assert monitor._resolve_sector("000001") == "Unknown"
