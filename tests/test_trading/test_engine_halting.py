"""Regression tests for the live engine's halted state (BUG-12).

`PortfolioState.HALTED` was unreachable from production code -- every
transition into it lived in the state-machine tests. The engine's cycle moves
the machine to REBALANCING on entry and back to TRADING at the end, and the
kill switch's early return sat between the two, so while the kill switch was
on the reported state stayed at REBALANCING for the life of the process.
"""

import pytest

from quant_platform.core.state_machine import PortfolioState, PortfolioStateMachine
from quant_platform.core.store import Store
from quant_platform.trading.broker import SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=1_000_000),
        store=Store(str(tmp_path / "engine.db")),
        state_machine=PortfolioStateMachine(),
    )
    # No network in tests; prices are not what these assertions are about.
    monkeypatch.setattr(eng, "_fetch_prices", lambda: None)
    # An engine that has been started, as `start()` leaves it (engine.py:202).
    eng._sm.transition(PortfolioState.READY, "test setup")
    eng._sm.transition(PortfolioState.TRADING, "test setup")
    return eng


def _kill(engine, active: bool) -> None:
    if active:
        engine._risk.activate_kill_switch("test")
    else:
        engine._risk.deactivate_kill_switch()


class TestTheKillSwitchDeRisks:
    """The early return happened before signal generation, so the reducing
    sells that the pre-trade checks explicitly allow were never created: the
    switch halted the engine with the whole book still exposed."""

    def test_available_positions_are_sold(self, engine):
        from types import SimpleNamespace

        from quant_platform.execution.models import OrderStatus
        from quant_platform.trading.broker import OrderSide

        engine._current_prices = {"600519": 10.0}
        engine._broker.get_positions = lambda: [
            SimpleNamespace(code="600519", available=100, quantity=100),
        ]
        placed = []

        class _Result:
            status = OrderStatus.FILLED

            def to_dict(self):
                return {
                    "order_id": "derisk-1", "code": "600519",
                    "side": "sell", "quantity": 100, "price": 10.0,
                    "filled_quantity": 100, "filled_price": 10.0,
                }

        def place(order):
            placed.append(order)
            return _Result()

        engine._broker.place_order = place
        engine._risk.activate_kill_switch("test")

        cycle = engine._execute_cycle()

        assert len(placed) == 1
        assert placed[0].side == OrderSide.SELL
        assert placed[0].quantity == 100
        assert cycle.orders, "the cycle result must report the de-risking order"
        assert engine._sm.state == PortfolioState.HALTED

    def test_t_plus_one_locked_shares_are_left_alone(self, engine):
        from types import SimpleNamespace

        engine._current_prices = {"600519": 10.0}
        engine._broker.get_positions = lambda: [
            SimpleNamespace(code="600519", available=0, quantity=100),
        ]
        engine._broker.place_order = lambda order: pytest.fail(
            "T+1-locked shares must not be sold"
        )
        engine._risk.activate_kill_switch("test")

        engine._execute_cycle()


class TestTheKillSwitchReachesTheStateMachine:
    def test_the_state_does_not_stick_at_rebalancing(self, engine):
        _kill(engine, True)

        engine._execute_cycle()

        assert engine._sm.state != PortfolioState.REBALANCING, (
            "the early return skipped the transition back, leaving the engine "
            "reporting REBALANCING forever"
        )
        assert engine._sm.state == PortfolioState.HALTED

    def test_repeated_cycles_while_halted_do_not_raise(self, engine):
        _kill(engine, True)

        for _ in range(3):
            engine._execute_cycle()

        assert engine._sm.state == PortfolioState.HALTED

    def test_the_halt_is_recorded(self, engine):
        _kill(engine, True)

        engine._execute_cycle()

        reasons = [t["reason"] for t in engine._sm.get_history()]
        assert any("kill switch" in r for r in reasons)

    def test_clearing_the_kill_switch_resumes_trading(self, engine):
        _kill(engine, True)
        engine._execute_cycle()
        assert engine._sm.state == PortfolioState.HALTED

        _kill(engine, False)
        engine._execute_cycle()

        assert engine._sm.state in (PortfolioState.TRADING, PortfolioState.REBALANCING)

    def test_a_normal_cycle_does_not_halt(self, engine):
        engine._execute_cycle()

        assert engine._sm.state != PortfolioState.HALTED


def test_state_transitions_are_written_to_the_audit_log(engine):
    """`AuditLog.log_state_change` existed but nothing called it."""
    events = engine._store.get_events(topic="audit.state_change")
    assert events, "engine state transitions were not audited"
