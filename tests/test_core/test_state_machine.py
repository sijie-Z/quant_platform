"""Tests for core.state_machine — Portfolio state machine."""

import pytest
from quant_platform.core.state_machine import (
    PortfolioStateMachine, PortfolioState, StateTransition,
)


class TestStateMachine:
    def setup_method(self):
        self.sm = PortfolioStateMachine()

    def test_initial_state(self):
        assert self.sm.state == PortfolioState.INIT

    def test_valid_transition(self):
        result = self.sm.transition(PortfolioState.READY, "startup complete")
        assert result is True
        assert self.sm.state == PortfolioState.READY

    def test_invalid_transition(self):
        result = self.sm.transition(PortfolioState.TRADING, "skip ready")
        assert result is False
        assert self.sm.state == PortfolioState.INIT

    def test_full_lifecycle(self):
        self.sm.transition(PortfolioState.READY, "init done")
        self.sm.transition(PortfolioState.PRE_MARKET, "9:15")
        self.sm.transition(PortfolioState.TRADING, "9:30")
        self.sm.transition(PortfolioState.REBALANCING, "signal generated")
        self.sm.transition(PortfolioState.TRADING, "rebalance done")
        self.sm.transition(PortfolioState.POST_MARKET, "15:00")
        self.sm.transition(PortfolioState.READY, "EOD done")
        assert self.sm.state == PortfolioState.READY

    def test_halt_from_any_state(self):
        self.sm.transition(PortfolioState.READY, "")
        self.sm.transition(PortfolioState.TRADING, "")
        assert self.sm.transition(PortfolioState.HALTED, "risk breach") is True
        assert self.sm.state == PortfolioState.HALTED

    def test_recovery_from_halted(self):
        self.sm.force_state(PortfolioState.HALTED)
        assert self.sm.transition(PortfolioState.READY, "manual resume") is True

    def test_error_recovery(self):
        self.sm.force_state(PortfolioState.ERROR)
        assert self.sm.transition(PortfolioState.INIT, "restart") is True

    def test_history_tracking(self):
        self.sm.transition(PortfolioState.READY, "r1")
        self.sm.transition(PortfolioState.PRE_MARKET, "r2")
        history = self.sm.get_history()
        assert len(history) == 2
        assert history[0]["from"] == "init"
        assert history[0]["to"] == "ready"

    def test_can_transition(self):
        assert self.sm.can_transition(PortfolioState.READY) is True
        assert self.sm.can_transition(PortfolioState.TRADING) is False

    def test_force_state(self):
        self.sm.force_state(PortfolioState.TRADING, "forced")
        assert self.sm.state == PortfolioState.TRADING

    def test_entry_hook(self):
        hook_called = []
        self.sm.on_entry(PortfolioState.READY, lambda: hook_called.append(True))
        self.sm.transition(PortfolioState.READY, "")
        assert hook_called == [True]

    def test_exit_hook(self):
        hook_called = []
        self.sm.on_exit(PortfolioState.INIT, lambda: hook_called.append(True))
        self.sm.transition(PortfolioState.READY, "")
        assert hook_called == [True]

    def test_on_transition_callback(self):
        transitions = []
        sm = PortfolioStateMachine(on_transition=lambda t: transitions.append(t))
        sm.transition(PortfolioState.READY, "test")
        assert len(transitions) == 1
        assert transitions[0].from_state == "init"
        assert transitions[0].to_state == "ready"

    def test_state_duration(self):
        import time
        self.sm.transition(PortfolioState.READY, "")
        time.sleep(0.01)
        assert self.sm.state_duration > 0

    def test_state_str(self):
        assert self.sm.state_str == "init"
        self.sm.transition(PortfolioState.READY, "")
        assert self.sm.state_str == "ready"
