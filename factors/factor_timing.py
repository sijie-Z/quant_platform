"""Factor timing — regime-based dynamic factor weight adjustment.

Adjusts factor weights based on detected market regime to improve
risk-adjusted returns. Different regimes favor different factors:

- High volatility: quality + low-vol factors outperform (defensive)
- Bull trend: momentum + growth factors outperform (offensive)
- Bear trend: value + quality factors outperform (defensive)
- Normal: no adjustment needed

Also provides exponential smoothing to prevent abrupt weight changes
that would cause excessive turnover.

Reference:
- Asness et al. (2017): "Value and Momentum Everywhere"
- Daniel & Moskowitz (2016): "Momentum Crashes"
"""

from __future__ import annotations

from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Factor category mappings for regime-based adjustment
_FACTOR_CATEGORIES = {
    "momentum": "momentum",
    "momentum_1m": "momentum",
    "momentum_3m": "momentum",
    "momentum_6m": "momentum",
    "momentum_12m": "momentum",
    "growth": "growth",
    "asset_growth": "growth",
    "value": "value",
    "pb_ratio": "value",
    "pe_ratio": "value",
    "quality": "quality",
    "roe": "quality",
    "low_vol": "low_vol",
    "volatility_20d": "low_vol",
    "volatility_60d": "low_vol",
    "liquidity": "neutral",
    "turnover_20d": "neutral",
    "size": "neutral",
    "log_market_cap": "neutral",
    "rsi_14d": "momentum",
    "macd": "momentum",
    "amplitude_20d": "neutral",
}

# Regime multipliers: {category: multiplier}
_REGIME_PROFILES = {
    "high_vol": {
        "quality": 1.5,
        "low_vol": 1.5,
        "value": 1.0,
        "momentum": 0.5,
        "growth": 0.5,
        "neutral": 1.0,
    },
    "bull_trend": {
        "momentum": 1.5,
        "growth": 1.5,
        "value": 0.5,
        "quality": 1.0,
        "low_vol": 0.8,
        "neutral": 1.0,
    },
    "bear_trend": {
        "value": 1.5,
        "quality": 1.5,
        "momentum": 0.5,
        "growth": 0.5,
        "low_vol": 1.2,
        "neutral": 1.0,
    },
    "normal": {
        "momentum": 1.0,
        "growth": 1.0,
        "value": 1.0,
        "quality": 1.0,
        "low_vol": 1.0,
        "neutral": 1.0,
    },
}


class RegimeBasedTimer:
    """Adjust factor weights based on detected market regime.

    Usage:
        from quant_platform.risk.regime import CompositeRegimeDetector
        from quant_platform.factors.factor_timing import RegimeBasedTimer

        detector = CompositeRegimeDetector()
        timer = RegimeBasedTimer(regime_detector=detector)

        # After detecting regime:
        regime_result = detector.detect(returns, prices)
        regime_name = _map_regime(regime_result)  # "high_vol", "bull_trend", etc.
        adjusted = timer.get_regime_weights(base_weights, regime_name)
    """

    def __init__(
        self,
        regime_detector: Any | None = None,
        lookback: int = 60,
        category_map: dict[str, str] | None = None,
        regime_profiles: dict[str, dict[str, float]] | None = None,
    ):
        self.regime_detector = regime_detector
        self.lookback = lookback
        self.category_map = category_map or _FACTOR_CATEGORIES
        self.regime_profiles = regime_profiles or _REGIME_PROFILES
        self._previous_weights: dict[str, float] | None = None

    def get_regime_weights(
        self,
        base_weights: dict[str, float],
        current_regime: str,
    ) -> dict[str, float]:
        """Adjust factor weights based on current regime.

        Args:
            base_weights: Original factor weights (name -> weight).
            current_regime: One of "high_vol", "bull_trend", "bear_trend", "normal".

        Returns:
            Adjusted weights, normalized to sum to 1.
        """
        if current_regime == "normal":
            return dict(base_weights)

        profile = self.regime_profiles.get(current_regime)
        if profile is None:
            logger.warning("Unknown regime '%s', returning base weights", current_regime)
            return dict(base_weights)

        adjusted = {}
        for name, weight in base_weights.items():
            category = self.category_map.get(name, "neutral")
            multiplier = profile.get(category, 1.0)
            adjusted[name] = weight * multiplier

        # Normalize to sum = 1
        total = sum(adjusted.values())
        if total > 1e-10:
            adjusted = {name: w / total for name, w in adjusted.items()}
        else:
            adjusted = dict(base_weights)

        return adjusted

    def smooth_transition(
        self,
        current_weights: dict[str, float],
        previous_weights: dict[str, float] | None = None,
        lambda_: float = 0.8,
    ) -> dict[str, float]:
        """Exponential smoothing to prevent abrupt weight changes.

        new_weight = lambda * current + (1 - lambda) * previous

        Args:
            current_weights: Target weights from regime adjustment.
            previous_weights: Previous period's weights. Uses stored if None.
            lambda_: Smoothing factor (0=full smoothing, 1=no smoothing).

        Returns:
            Smoothed weights, normalized to sum to 1.
        """
        prev = previous_weights if previous_weights is not None else self._previous_weights
        if prev is None:
            self._previous_weights = dict(current_weights)
            return dict(current_weights)

        # Merge keys
        all_names = set(current_weights.keys()) | set(prev.keys())
        smoothed = {}
        for name in all_names:
            curr_w = current_weights.get(name, 0.0)
            prev_w = prev.get(name, 0.0)
            smoothed[name] = lambda_ * curr_w + (1 - lambda_) * prev_w

        # Normalize
        total = sum(smoothed.values())
        if total > 1e-10:
            smoothed = {name: w / total for name, w in smoothed.items()}

        self._previous_weights = dict(smoothed)
        return smoothed

    def reset(self) -> None:
        """Reset stored previous weights."""
        self._previous_weights = None


def map_regime_to_name(regime_result: dict) -> str:
    """Map CompositeRegimeDetector output to regime name for RegimeBasedTimer.

    Args:
        regime_result: Output from CompositeRegimeDetector.detect().

    Returns:
        One of "high_vol", "bull_trend", "bear_trend", "normal".
    """
    overall = regime_result.get("overall_regime", "neutral")
    vol_regime = regime_result.get("volatility", {}).get("regime", "")
    trend_regime = regime_result.get("trend", {}).get("regime", "")

    # High volatility takes priority
    if vol_regime in ("high_volatility", "extreme_volatility"):
        return "high_vol"

    # Trend-based
    if trend_regime == "bear":
        return "bear_trend"
    if trend_regime == "bull":
        return "bull_trend"

    return "normal"
