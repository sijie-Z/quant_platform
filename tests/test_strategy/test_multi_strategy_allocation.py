"""Regression tests for MultiStrategyManager allocation and metrics.

Covers audit findings F6 (absolute weights, omitted ids zeroed, sum > 1
rejected), F7 (reallocation rebases the P&L segment; zero-capital periods
record nothing) and F12 (consistent inactive-strategy policy, value-weighted
aggregate Sharpe/DD).
"""

import numpy as np
import pytest

from quant_platform.strategy.multi_strategy import (
    MultiStrategyManager,
    StrategyConfig,
)


def make(total_capital=100_000_000):
    return MultiStrategyManager(total_capital=total_capital)


# ---------------------------------------------------------------------------
# F6 — allocation semantics
# ---------------------------------------------------------------------------


class TestAllocationSemantics:
    def test_weights_are_absolute_fractions_of_total_capital(self):
        mgr = make()
        sid = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.1))

        mgr.allocate_capital({sid: 0.5})

        # 0.5 means 50% of total capital, not "50% of the provided dict
        # renormalised to 100%" (which used to fund the strategy 100%).
        assert mgr.strategies[sid].allocation_pct == pytest.approx(0.5)
        assert mgr.states[sid].capital_allocated == pytest.approx(50_000_000)

    def test_omitted_strategies_are_zeroed(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.6))
        b = mgr.add_strategy(StrategyConfig(name="B", allocation_pct=0.4))

        mgr.allocate_capital({a: 0.5})

        assert mgr.states[a].capital_allocated == pytest.approx(50_000_000)
        assert mgr.states[b].capital_allocated == 0
        assert mgr.strategies[b].allocation_pct == 0
        metrics = mgr.get_aggregate_metrics()
        assert metrics["total_allocated"] == pytest.approx(50_000_000)
        assert metrics["total_value"] == pytest.approx(50_000_000)

    def test_weights_summing_above_one_are_rejected(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.6))
        b = mgr.add_strategy(StrategyConfig(name="B", allocation_pct=0.4))

        with pytest.raises(ValueError, match="> 1.0"):
            mgr.allocate_capital({a: 0.7, b: 0.4})

        # Nothing was mutated.
        assert mgr.states[a].capital_allocated == pytest.approx(60_000_000)
        assert mgr.states[b].capital_allocated == pytest.approx(40_000_000)

    def test_unknown_negative_and_non_finite_weights_are_rejected(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=1.0))

        with pytest.raises(ValueError, match="Unknown strategy_id"):
            mgr.allocate_capital({a: 0.5, "ghost": 0.5})
        with pytest.raises(ValueError, match="negative"):
            mgr.allocate_capital({a: -0.1})
        with pytest.raises(ValueError, match="not finite"):
            mgr.allocate_capital({a: float("nan")})
        with pytest.raises(ValueError, match="not numeric"):
            mgr.allocate_capital({a: "half"})

    def test_full_coverage_still_works(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.5))
        b = mgr.add_strategy(StrategyConfig(name="B", allocation_pct=0.5))

        mgr.allocate_capital({a: 0.7, b: 0.3})

        assert mgr.states[a].capital_allocated == pytest.approx(70_000_000)
        assert mgr.states[b].capital_allocated == pytest.approx(30_000_000)


# ---------------------------------------------------------------------------
# F7 — reallocation resets the performance segment
# ---------------------------------------------------------------------------


class TestReallocationMetrics:
    def test_reallocation_after_a_loss_rebases_pnl_and_risk_stats(self):
        mgr = make()
        sid = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=1.0))
        for _ in range(20):
            mgr.update_strategy_pnl(sid, -0.01)

        state = mgr.states[sid]
        assert state.total_pnl < 0
        assert state.total_return < -0.15
        assert state.max_drawdown < 0

        mgr.allocate_capital({sid: 1.0})

        assert state.capital_allocated == pytest.approx(100_000_000)
        assert state.current_value == pytest.approx(100_000_000)
        # Mutually consistent again: value - capital == total_pnl == 0.
        assert state.total_pnl == pytest.approx(0)
        assert state.total_return == pytest.approx(0)
        # Risk statistics are since-reallocation, not stale pre-reallocation.
        assert state.sharpe_ratio == 0
        assert state.max_drawdown == 0
        assert mgr.daily_returns[sid] == []

        metrics = mgr.get_aggregate_metrics()
        assert metrics["total_pnl"] == pytest.approx(0)
        assert metrics["total_return"] == pytest.approx(0)
        assert metrics["aggregate_max_dd"] == 0

    def test_zero_capital_periods_record_no_returns(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.6))
        b = mgr.add_strategy(StrategyConfig(name="B", allocation_pct=0.4))
        mgr.allocate_capital({a: 1.0})

        state_b = mgr.states[b]
        assert state_b.capital_allocated == 0

        for _ in range(80):
            mgr.update_strategy_pnl(b, 0.05)      # zero-capital strategy
            mgr.update_strategy_pnl(a, 0.001)

        assert state_b.total_pnl == pytest.approx(0)
        assert state_b.total_return == pytest.approx(0)
        assert state_b.sharpe_ratio == 0
        assert state_b.max_drawdown == 0
        assert mgr.daily_returns[b] == []

        state_a = mgr.states[a]
        metrics = mgr.get_aggregate_metrics()
        assert metrics["total_pnl"] == pytest.approx(state_a.total_pnl)
        assert metrics["total_allocated"] == pytest.approx(100_000_000)
        assert metrics["total_return"] == pytest.approx(state_a.total_return)

    def test_new_pnl_accumulates_cleanly_after_reallocation(self):
        mgr = make()
        sid = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=1.0))
        for _ in range(20):
            mgr.update_strategy_pnl(sid, -0.01)
        mgr.allocate_capital({sid: 0.5})

        mgr.update_strategy_pnl(sid, 0.02)

        state = mgr.states[sid]
        assert state.capital_allocated == pytest.approx(50_000_000)
        assert state.total_pnl == pytest.approx(1_000_000)
        assert state.total_return == pytest.approx(0.02)
        assert state.current_value == pytest.approx(51_000_000)


# ---------------------------------------------------------------------------
# F12 — aggregate metrics inclusion policy and weighting
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_inactive_strategies_are_excluded_from_all_aggregates(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(
            name="A", allocation_pct=0.6, is_active=True,
        ))
        b = mgr.add_strategy(StrategyConfig(
            name="B", allocation_pct=0.4, is_active=False,
        ))
        mgr.update_strategy_pnl(a, 0.01)
        mgr.update_strategy_pnl(b, -0.02)

        metrics = mgr.get_aggregate_metrics()
        state_a = mgr.states[a]

        assert metrics["n_strategies"] == 2
        assert metrics["n_active"] == 1
        assert metrics["total_value"] == pytest.approx(state_a.current_value)
        assert metrics["total_pnl"] == pytest.approx(state_a.total_pnl)
        assert metrics["total_allocated"] == pytest.approx(state_a.capital_allocated)
        assert metrics["total_return"] == pytest.approx(state_a.total_return)

        row_b = next(row for row in metrics["strategies"] if row["id"] == b)
        assert row_b["active"] is False

    def test_aggregate_sharpe_uses_a_value_weighted_portfolio_series(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.6))
        b = mgr.add_strategy(StrategyConfig(name="B", allocation_pct=0.4))

        rets_a = [0.010, -0.015, 0.020, 0.000, -0.005, 0.008, -0.012] * 10
        rets_b = [0.004, 0.006, -0.010, 0.003, -0.002, 0.005, 0.001] * 10
        for ret_a, ret_b in zip(rets_a, rets_b, strict=True):
            mgr.update_strategy_pnl(a, ret_a)
            mgr.update_strategy_pnl(b, ret_b)

        values = np.array([
            mgr.states[a].current_value, mgr.states[b].current_value,
        ])
        weights = values / values.sum()
        assert weights.sum() == pytest.approx(1.0)
        portfolio_rets = (
            weights[0] * np.array(rets_a) + weights[1] * np.array(rets_b)
        )
        expected_sharpe = (
            portfolio_rets.mean() / portfolio_rets.std() * np.sqrt(252)
        )
        cum = np.cumprod(1 + portfolio_rets)
        peak = np.maximum.accumulate(cum)
        expected_dd = ((cum - peak) / peak).min()

        metrics = mgr.get_aggregate_metrics()
        assert metrics["aggregate_sharpe"] == pytest.approx(
            expected_sharpe, rel=1e-9,
        )
        assert metrics["aggregate_max_dd"] == pytest.approx(
            expected_dd, rel=1e-9,
        )

    def test_single_active_strategy_gets_weight_one(self):
        mgr = make()
        a = mgr.add_strategy(StrategyConfig(name="A", allocation_pct=0.6))
        b = mgr.add_strategy(StrategyConfig(
            name="B", allocation_pct=0.4, is_active=False,
        ))
        rets_a = [0.01, -0.02, 0.015] * 10
        for ret in rets_a:
            mgr.update_strategy_pnl(a, ret)
            mgr.update_strategy_pnl(b, -0.03)

        expected = np.mean(rets_a) / np.std(rets_a) * np.sqrt(252)
        metrics = mgr.get_aggregate_metrics()
        assert metrics["aggregate_sharpe"] == pytest.approx(expected, rel=1e-9)
