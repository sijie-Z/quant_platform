"""Tests for FactorICAutoDecay — automatic factor weight decay on IC deterioration.

Verifies that factors are automatically disabled when their rolling IC
drops below threshold, and re-enabled when IC recovers.
"""


from quant_platform.factors.ic_monitor import FactorICAutoDecay


class TestFactorICAutoDecay:
    """Core auto-decay behavior."""

    def test_init_defaults(self):
        """Default parameters should be set correctly."""
        decay = FactorICAutoDecay()
        assert decay.decay_window == 20
        assert decay.decay_threshold == 0.01
        assert decay.recovery_window == 5
        assert decay.recovery_threshold == 0.02
        assert decay.disabled_factors == set()

    def test_factor_stays_active_with_good_ic(self):
        """Factor with IC above threshold should stay active."""
        decay = FactorICAutoDecay(decay_window=5, decay_threshold=0.01)
        for _ in range(10):
            decay.update("momentum", 0.03)
            decay.check_and_update("momentum")

        assert decay.is_active("momentum")
        assert "momentum" not in decay.disabled_factors

    def test_factor_disabled_on_persistent_low_ic(self):
        """Factor with IC below threshold for decay_window should be disabled."""
        decay = FactorICAutoDecay(decay_window=5, decay_threshold=0.01)

        # Feed low IC values
        for _ in range(6):
            decay.update("weak_factor", 0.005)
            result = decay.check_and_update("weak_factor")

        assert not decay.is_active("weak_factor")
        assert "weak_factor" in decay.disabled_factors
        assert result is False

    def test_factor_recovers_when_ic_improves(self):
        """Disabled factor should recover when IC rises above recovery threshold."""
        decay = FactorICAutoDecay(
            decay_window=5,
            decay_threshold=0.01,
            recovery_window=3,
            recovery_threshold=0.02,
        )

        # Disable the factor
        for _ in range(6):
            decay.update("factor_a", 0.005)
            decay.check_and_update("factor_a")

        assert not decay.is_active("factor_a")

        # Recover with high IC
        for _ in range(4):
            decay.update("factor_a", 0.03)
            result = decay.check_and_update("factor_a")

        assert decay.is_active("factor_a")
        assert result is True

    def test_no_disable_with_insufficient_history(self):
        """Factor should not be disabled before decay_window observations."""
        decay = FactorICAutoDecay(decay_window=20, decay_threshold=0.01)

        # Only 5 observations (less than decay_window)
        for _ in range(5):
            decay.update("new_factor", 0.005)
            decay.check_and_update("new_factor")

        # Should still be active (not enough data)
        assert decay.is_active("new_factor")


class TestGetActiveWeights:
    """Weight adjustment logic."""

    def test_disabled_factor_zeroed(self):
        """Disabled factor should get weight 0, others renormalized."""
        decay = FactorICAutoDecay(decay_window=3, decay_threshold=0.01)

        # Disable factor_b
        for _ in range(4):
            decay.update("factor_b", 0.005)
            decay.check_and_update("factor_b")

        weights = {"factor_a": 0.6, "factor_b": 0.4}
        adjusted = decay.get_active_weights(weights)

        assert adjusted["factor_b"] == 0.0
        assert abs(adjusted["factor_a"] - 1.0) < 1e-10

    def test_all_active_no_change(self):
        """If all factors are active, weights should be renormalized to same ratios."""
        decay = FactorICAutoDecay(decay_window=5, decay_threshold=0.01)

        # Feed good IC to keep all active
        for _ in range(6):
            decay.update("a", 0.03)
            decay.check_and_update("a")
            decay.update("b", 0.03)
            decay.check_and_update("b")

        weights = {"a": 0.6, "b": 0.4}
        adjusted = decay.get_active_weights(weights)

        assert abs(adjusted["a"] - 0.6) < 1e-10
        assert abs(adjusted["b"] - 0.4) < 1e-10

    def test_all_disabled_fallback_to_equal(self):
        """If all factors are disabled, should fallback to equal weight."""
        decay = FactorICAutoDecay(decay_window=3, decay_threshold=0.01)

        for name in ["x", "y", "z"]:
            for _ in range(4):
                decay.update(name, 0.005)
                decay.check_and_update(name)

        weights = {"x": 0.5, "y": 0.3, "z": 0.2}
        adjusted = decay.get_active_weights(weights)

        # All should be equal weight (1/3)
        assert abs(adjusted["x"] - 1/3) < 1e-10
        assert abs(adjusted["y"] - 1/3) < 1e-10
        assert abs(adjusted["z"] - 1/3) < 1e-10

    def test_empty_weights(self):
        """Empty weights dict should return empty."""
        decay = FactorICAutoDecay()
        assert decay.get_active_weights({}) == {}


class TestEventLog:
    """Event logging for disable/recover actions."""

    def test_disable_event_logged(self):
        """Disabling a factor should create an event log entry."""
        decay = FactorICAutoDecay(decay_window=3, decay_threshold=0.01)

        for _ in range(4):
            decay.update("bad_factor", 0.005)
            decay.check_and_update("bad_factor")

        events = decay.events
        assert len(events) >= 1
        assert events[-1]["action"] == "disabled"
        assert events[-1]["factor"] == "bad_factor"

    def test_recover_event_logged(self):
        """Recovering a factor should create an event log entry."""
        decay = FactorICAutoDecay(
            decay_window=3, decay_threshold=0.01,
            recovery_window=2, recovery_threshold=0.02,
        )

        # Disable
        for _ in range(4):
            decay.update("factor_x", 0.005)
            decay.check_and_update("factor_x")

        # Recover
        for _ in range(3):
            decay.update("factor_x", 0.03)
            decay.check_and_update("factor_x")

        events = decay.events
        actions = [e["action"] for e in events]
        assert "disabled" in actions
        assert "recovered" in actions

    def test_reset_clears_state(self):
        """Reset should clear all history, disabled factors, and events."""
        decay = FactorICAutoDecay(decay_window=3, decay_threshold=0.01)

        for _ in range(4):
            decay.update("factor", 0.005)
            decay.check_and_update("factor")

        assert len(decay.disabled_factors) > 0
        assert len(decay.events) > 0

        decay.reset()

        assert decay.disabled_factors == set()
        assert decay.events == []
        assert decay.is_active("factor")
