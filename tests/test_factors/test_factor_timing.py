"""Tests for factor timing — regime-based weight adjustment."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.factors.factor_timing import (
    RegimeBasedTimer,
    map_regime_to_name,
    _FACTOR_CATEGORIES,
    _REGIME_PROFILES,
)


@pytest.fixture
def base_weights():
    """Sample factor weights."""
    return {
        "momentum_3m": 0.25,
        "pb_ratio": 0.25,
        "roe": 0.25,
        "volatility_20d": 0.25,
    }


@pytest.fixture
def timer():
    """Create a RegimeBasedTimer."""
    return RegimeBasedTimer()


class TestRegimeWeights:
    """Test regime-based weight adjustment."""

    def test_normal_returns_base(self, timer, base_weights):
        """Normal regime should return base weights unchanged."""
        result = timer.get_regime_weights(base_weights, "normal")
        assert result == base_weights

    def test_high_vol_quality_up(self, timer, base_weights):
        """High vol: quality and low-vol should get higher weight."""
        result = timer.get_regime_weights(base_weights, "high_vol")

        # roe (quality) and volatility_20d (low_vol) should increase
        assert result["roe"] > base_weights["roe"]
        assert result["volatility_20d"] > base_weights["volatility_20d"]

        # momentum should decrease
        assert result["momentum_3m"] < base_weights["momentum_3m"]

    def test_bull_trend_momentum_up(self, timer, base_weights):
        """Bull trend: momentum should get higher weight."""
        result = timer.get_regime_weights(base_weights, "bull_trend")

        assert result["momentum_3m"] > base_weights["momentum_3m"]
        # pb_ratio (value) should decrease
        assert result["pb_ratio"] < base_weights["pb_ratio"]

    def test_bear_trend_value_up(self, timer, base_weights):
        """Bear trend: value and quality should get higher weight."""
        result = timer.get_regime_weights(base_weights, "bear_trend")

        assert result["pb_ratio"] > base_weights["pb_ratio"]
        assert result["roe"] > base_weights["roe"]
        assert result["momentum_3m"] < base_weights["momentum_3m"]

    def test_weights_sum_to_one(self, timer, base_weights):
        """All regimes should produce weights summing to 1."""
        for regime in ("normal", "high_vol", "bull_trend", "bear_trend"):
            result = timer.get_regime_weights(base_weights, regime)
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01, f"Regime {regime}: sum={total}"

    def test_unknown_regime_returns_base(self, timer, base_weights):
        """Unknown regime should return base weights."""
        result = timer.get_regime_weights(base_weights, "unknown_regime")
        assert result == base_weights

    def test_all_factors_adjusted(self, timer):
        """All known factors should be adjusted."""
        weights = {name: 1.0 / len(_FACTOR_CATEGORIES) for name in _FACTOR_CATEGORIES}
        result = timer.get_regime_weights(weights, "high_vol")

        # All should have non-zero weight
        for name in weights:
            assert result[name] >= 0


class TestSmoothTransition:
    """Test exponential smoothing."""

    def test_no_smoothing_first_time(self, timer):
        """First call with no previous weights should return current."""
        current = {"a": 0.5, "b": 0.5}
        result = timer.smooth_transition(current)
        assert result == current

    def test_smoothing_reduces_change(self, timer):
        """Smoothing should reduce the magnitude of weight changes."""
        prev = {"a": 0.8, "b": 0.2}
        curr = {"a": 0.2, "b": 0.8}

        result = timer.smooth_transition(curr, prev, lambda_=0.5)

        # With lambda=0.5, result should be midpoint
        assert abs(result["a"] - 0.5) < 0.01
        assert abs(result["b"] - 0.5) < 0.01

    def test_smoothing_preserves_sum(self, timer, base_weights):
        """Smoothed weights should sum to ~1."""
        prev = {"momentum_3m": 0.4, "pb_ratio": 0.1, "roe": 0.3, "volatility_20d": 0.2}
        result = timer.smooth_transition(base_weights, prev, lambda_=0.8)

        total = sum(result.values())
        assert abs(total - 1.0) < 0.01

    def test_smoothing_lambda_one_no_change(self, timer):
        """lambda=1 should return current weights exactly."""
        prev = {"a": 0.9, "b": 0.1}
        curr = {"a": 0.1, "b": 0.9}

        result = timer.smooth_transition(curr, prev, lambda_=1.0)
        assert abs(result["a"] - 0.1) < 0.01
        assert abs(result["b"] - 0.9) < 0.01

    def test_smoothing_lambda_zero_full_smoothing(self, timer):
        """lambda=0 should return previous weights."""
        prev = {"a": 0.9, "b": 0.1}
        curr = {"a": 0.1, "b": 0.9}

        result = timer.smooth_transition(curr, prev, lambda_=0.0)
        assert abs(result["a"] - 0.9) < 0.01
        assert abs(result["b"] - 0.1) < 0.01

    def test_stored_previous_weights(self, timer):
        """Timer should store and use previous weights automatically."""
        w1 = {"a": 0.7, "b": 0.3}
        w2 = {"a": 0.3, "b": 0.7}

        timer.smooth_transition(w1)
        result = timer.smooth_transition(w2, lambda_=0.5)

        # Should use stored w1 as previous
        assert abs(result["a"] - 0.5) < 0.01

    def test_reset_clears_stored(self, timer):
        """reset() should clear stored previous weights."""
        timer.smooth_transition({"a": 0.5, "b": 0.5})
        timer.reset()

        # Next call should treat as first time
        result = timer.smooth_transition({"a": 0.1, "b": 0.9})
        assert abs(result["a"] - 0.1) < 0.01


class TestMapRegime:
    """Test regime mapping from detector output."""

    def test_high_vol(self):
        """High volatility regime should map to 'high_vol'."""
        result = {
            "overall_regime": "cautious",
            "volatility": {"regime": "high_volatility"},
            "trend": {"regime": "bull"},
        }
        assert map_regime_to_name(result) == "high_vol"

    def test_extreme_vol(self):
        """Extreme volatility should also map to 'high_vol'."""
        result = {
            "overall_regime": "risk_off",
            "volatility": {"regime": "extreme_volatility"},
            "trend": {"regime": "bear"},
        }
        assert map_regime_to_name(result) == "high_vol"

    def test_bull_trend(self):
        """Bull trend without high vol should map to 'bull_trend'."""
        result = {
            "overall_regime": "risk_on",
            "volatility": {"regime": "medium_volatility"},
            "trend": {"regime": "bull"},
        }
        assert map_regime_to_name(result) == "bull_trend"

    def test_bear_trend(self):
        """Bear trend without high vol should map to 'bear_trend'."""
        result = {
            "overall_regime": "cautious",
            "volatility": {"regime": "low_volatility"},
            "trend": {"regime": "bear"},
        }
        assert map_regime_to_name(result) == "bear_trend"

    def test_normal(self):
        """Medium vol + sideways should map to 'normal'."""
        result = {
            "overall_regime": "neutral",
            "volatility": {"regime": "medium_volatility"},
            "trend": {"regime": "sideways"},
        }
        assert map_regime_to_name(result) == "normal"

    def test_missing_keys(self):
        """Missing keys should default gracefully."""
        result = {"overall_regime": "neutral"}
        assert map_regime_to_name(result) == "normal"


class TestICDecayIntegration:
    """Test factor timing works alongside IC auto-decay."""

    def test_timing_after_decay(self, timer, base_weights):
        """Regime adjustment should work on top of decay-adjusted weights."""
        # Simulate what combination.py does:
        # 1. IC decay zeros out momentum
        decayed = dict(base_weights)
        decayed["momentum_3m"] = 0.0
        total = sum(decayed.values())
        decayed = {k: v / total for k, v in decayed.items()}

        # 2. Regime timing further adjusts
        result = timer.get_regime_weights(decayed, "high_vol")

        # momentum should be zero or very small (was zeroed by decay)
        assert result["momentum_3m"] <= 0.01
        # quality should still be boosted
        assert result["roe"] > decayed["roe"]

    def test_timing_with_custom_category_map(self):
        """Custom category map should be respected."""
        custom_map = {"my_alpha": "momentum", "my_value": "value"}
        custom_timer = RegimeBasedTimer(category_map=custom_map)

        weights = {"my_alpha": 0.5, "my_value": 0.5}
        result = custom_timer.get_regime_weights(weights, "bull_trend")

        assert result["my_alpha"] > weights["my_alpha"]
        assert result["my_value"] < weights["my_value"]
