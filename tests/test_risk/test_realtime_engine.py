"""Tests for the real-time risk engine and Greeks calculator."""

import pytest

from quant_platform.risk.greeks import (
    BlackScholesModel,
    GreeksCalculator,
    OptionGreeks,
)
from quant_platform.risk.realtime_engine import (
    PreTradeCheck,
    RealTimeRiskEngine,
    RiskLevel,
    RiskLimit,
    LimitType,
)


# ── Black-Scholes Tests ──


class TestBlackScholes:
    def test_call_price(self):
        # S=100, K=100, T=1, r=5%, vol=20%
        price = BlackScholesModel.price(100, 100, 1.0, 0.05, 0.20, "call")
        assert 10 < price < 15  # Should be ~10.45

    def test_put_price(self):
        price = BlackScholesModel.price(100, 100, 1.0, 0.05, 0.20, "put")
        assert 5 < price < 10  # Should be ~5.57

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        call = BlackScholesModel.price(S, K, T, r, sigma, "call")
        put = BlackScholesModel.price(S, K, T, r, sigma, "put")
        # C - P = S - K * exp(-rT)
        assert abs((call - put) - (S - K * pow(2.71828, -r * T))) < 0.01

    def test_expired_option(self):
        call = BlackScholesModel.price(110, 100, 0, 0.05, 0.20, "call")
        assert call == 10.0  # Intrinsic value
        put = BlackScholesModel.price(90, 100, 0, 0.05, 0.20, "put")
        assert put == 10.0

    def test_greeks_call(self):
        greeks = BlackScholesModel.compute_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert 0.5 < greeks["delta"] < 0.7  # ATM call delta ~0.6
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0

    def test_greeks_put(self):
        greeks = BlackScholesModel.compute_greeks(100, 100, 1.0, 0.05, 0.20, "put")
        assert -0.5 < greeks["delta"] < -0.3  # ATM put delta ~-0.4
        assert greeks["gamma"] > 0  # Same as call

    def test_deep_itm_call(self):
        greeks = BlackScholesModel.compute_greeks(200, 100, 1.0, 0.05, 0.20, "call")
        assert greeks["delta"] > 0.95  # Deep ITM call delta ~1

    def test_deep_otm_call(self):
        greeks = BlackScholesModel.compute_greeks(50, 100, 1.0, 0.05, 0.20, "call")
        assert greeks["delta"] < 0.05  # Deep OTM call delta ~0


# ── GreeksCalculator Tests ──


class TestGreeksCalculator:
    @pytest.fixture
    def calc(self):
        return GreeksCalculator()

    def test_add_position(self, calc):
        pos = OptionGreeks(
            symbol="CALL_100", underlying="600519",
            option_type="call", strike=100, expiry_days=30,
            position=10, spot=100, volatility=0.20,
        )
        calc.add_position(pos)
        assert calc.get_position_count() == 1

    def test_compute_portfolio_greeks(self, calc):
        pos = OptionGreeks(
            symbol="CALL_100", underlying="600519",
            option_type="call", strike=100, expiry_days=30,
            position=10, spot=100, volatility=0.20,
        )
        calc.add_position(pos)
        # Trigger Greeks computation by updating spot
        calc.update_spot("600519", 100)
        portfolio = calc.compute_portfolio_greeks()
        assert portfolio.total_delta != 0
        assert portfolio.total_gamma != 0

    def test_update_spot(self, calc):
        calc.add_position(OptionGreeks(
            symbol="CALL_100", underlying="600519",
            option_type="call", strike=100, expiry_days=30,
            position=10, spot=100, volatility=0.20,
        ))
        portfolio1 = calc.compute_portfolio_greeks()
        calc.update_spot("600519", 110)
        portfolio2 = calc.compute_portfolio_greeks()
        # Delta should change with spot
        assert portfolio2.total_delta != portfolio1.total_delta

    def test_hedge_orders(self, calc):
        calc.add_position(OptionGreeks(
            symbol="CALL_100", underlying="600519",
            option_type="call", strike=100, expiry_days=30,
            position=100, spot=100, volatility=0.20,
        ))
        orders = calc.get_hedge_orders(target_delta=0)
        # Should generate hedge orders to reduce delta
        assert isinstance(orders, list)


# ── RealTimeRiskEngine Tests ──


class TestRealTimeRiskEngine:
    @pytest.fixture
    def engine(self):
        eng = RealTimeRiskEngine(
            max_daily_loss=0.03,
            max_drawdown=0.15,
            auto_hedge=False,
        )
        eng.set_initial_equity(10_000_000)
        return eng

    def test_initial_state(self, engine):
        status = engine.get_risk_status()
        assert status["risk_level"] == "green"
        assert status["kill_switch"] is False

    def test_on_fill(self, engine):
        update = engine.on_fill({
            "symbol": "600519",
            "side": "buy",
            "price": 100.0,
            "quantity": 1000,
        })
        assert update.risk_level in (RiskLevel.GREEN, RiskLevel.YELLOW)
        assert update.update_latency_ns >= 0

    def test_pre_trade_check_approved(self, engine):
        check = engine.pre_trade_check(
            symbol="600519",
            side="buy",
            quantity=100,
            price=100.0,
        )
        assert check.approved is True
        assert check.check_latency_ns >= 0

    def test_pre_trade_check_position_limit(self, engine):
        # Try to buy too much
        check = engine.pre_trade_check(
            symbol="600519",
            side="buy",
            quantity=1000000,  # Way too much
            price=100.0,
        )
        assert check.approved is False

    def test_kill_switch(self, engine):
        engine.activate_kill_switch("test")
        assert engine._kill_switch_active is True
        status = engine.get_risk_status()
        assert status["kill_switch"] is True
        assert status["risk_level"] == "kill"

        # Pre-trade should be rejected
        check = engine.pre_trade_check("SYM", "buy", 100, 10.0)
        assert check.approved is False

    def test_deactivate_kill_switch(self, engine):
        engine.activate_kill_switch("test")
        engine.deactivate_kill_switch()
        assert engine._kill_switch_active is False
        status = engine.get_risk_status()
        assert status["risk_level"] == "green"

    def test_stress_test(self, engine):
        result = engine.run_stress_test()
        assert result.total_scenarios > 0
        assert result.run_time_us >= 0
        assert len(result.scenarios) == result.total_scenarios

    def test_add_limit(self, engine):
        engine.add_limit(RiskLimit(
            limit_type=LimitType.DELTA,
            name="custom_delta",
            threshold=500000,
        ))
        status = engine.get_risk_status()
        assert "custom_delta" in status["limits"]

    def test_breach_history(self, engine):
        # No breaches initially
        history = engine.get_breach_history()
        assert len(history) == 0

    def test_metrics(self, engine):
        engine.pre_trade_check("SYM", "buy", 100, 10.0)
        status = engine.get_risk_status()
        assert status["metrics"]["total_checks"] >= 1

    def test_order_frequency_limit(self, engine):
        # Add many orders quickly
        for _ in range(60):
            engine._order_timestamps.append(
                __import__('time').time_ns()
            )
        check = engine.pre_trade_check("SYM", "buy", 100, 10.0)
        # Should be rejected due to frequency
        assert check.approved is False


# ── Integration Test ──


class TestRiskIntegration:
    def test_fill_then_check(self):
        engine = RealTimeRiskEngine(auto_hedge=False)
        engine.set_initial_equity(10_000_000)

        # Process some fills
        for i in range(5):
            engine.on_fill({
                "symbol": f"SYM{i}",
                "side": "buy",
                "price": 100.0 + i,
                "quantity": 100,
            })

        # Check status
        status = engine.get_risk_status()
        assert status["risk_level"] in ("green", "yellow", "orange")

        # Stress test
        stress = engine.run_stress_test()
        assert stress.total_scenarios > 0
