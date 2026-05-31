"""Per-Stock Profile Classifier with Tradability Gate.

Inspired by KF Timing App's profile-based regime switching system, this
module detects per-stock market regimes (profiles) and computes a
tradability score that can be used to gate alpha signals.

5 Profiles:
  Trend_follower — strong directional trend (high efficiency, high coherence)
  Breakseeker   — breakout ignition (return shock + volume shock)
  Defender      — low tradability, defensive positioning
  Activist      — high transition / curvature, active trading
  All_other     — default moderate profile

The tradability score (0..1) is a composite of efficiency + coherence and
can be applied as a multiplier to alpha signals, suppressing stocks in
noisy / untradable regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILES = ("Trend_follower", "Breakseeker", "Defender", "Activist", "All_other")

_DEFAULT_HYSTERESIS = 0.90      # tolerate deterioration if already in trend
_DEFAULT_TREND_HI = 0.60        # high tradability threshold
_DEFAULT_TREND_LO = 0.30        # low tradability threshold
_DEFAULT_DEFENDER = 0.10        # defender threshold
_DEFAULT_BREAKOUT_MIN = 0.30    # minimum tradability for breakout
_DEFAULT_TRANSITION_Z = 1.5     # curvature z threshold
_DEFAULT_RETURN_Z = 1.5         # return shock z threshold
_DEFAULT_VOL_RATIO = 1.5        # volume ratio threshold
_DEFAULT_CLASSIFIER_WIN = 20    # classifier lookback window


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_efficiency(y: np.ndarray) -> float:
    """Direction-agnostic path efficiency: smooth → high, choppy → low.

    Range [0, 1]: 1 = perfectly efficient (straight line), 0 = random walk.
    """
    total_move = float(np.sum(np.abs(np.diff(y))) + 1e-8)
    return float(abs(y[-1] - y[0]) / total_move)


def compute_coherence_and_curvature(y: np.ndarray) -> tuple[float, float]:
    """Linear fit R² (coherence) + quadratic coefficient (curvature)."""
    x = np.linspace(-1.0, 1.0, len(y))

    p1 = np.polyfit(x, y, 1)
    trend_line = np.polyval(p1, x)
    ss_res = float(np.sum((y - trend_line) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-8)
    coherence = float(1.0 - ss_res / ss_tot)

    p2 = np.polyfit(x, y, 2)
    curvature = float(p2[0])

    return coherence, curvature


def compute_breakout_metrics(
    y: np.ndarray,
    vol: np.ndarray,
    ret_win: int = 3,
    ref_win: int = 20,
    vol_short: int = 3,
    vol_long: int = 20,
    return_z_threshold: float = _DEFAULT_RETURN_Z,
    vol_ratio_threshold: float = _DEFAULT_VOL_RATIO,
) -> tuple[bool, dict]:
    """Detect breakout ignition: return shock + volume shock."""
    min_needed = max(ref_win + ret_win + 1, vol_long, vol_short + 1)
    if len(y) < min_needed or len(vol) < min_needed:
        return False, {
            "return_shock_z": np.nan,
            "volume_shock_ratio": np.nan,
            "ret_shock_on": False,
            "vol_shock_on": False,
        }

    k_ret = y[-1] - y[-1 - ret_win]

    y_ref = y[-(ref_win + ret_win + 1):]
    hist_k = y_ref[ret_win:] - y_ref[:-ret_win]
    hist_base = hist_k[:-1] if len(hist_k) > 1 else hist_k
    hist_mu = float(np.mean(hist_base))
    hist_sd = float(np.std(hist_base, ddof=1)) if len(hist_base) > 1 else 1e-8
    hist_sd = max(hist_sd, 1e-8)

    return_shock_z = float((k_ret - hist_mu) / hist_sd)
    ret_shock_on = abs(return_shock_z) >= return_z_threshold

    recent_vol_short = float(np.mean(vol[-vol_short:]))
    baseline_vol_long = float(np.mean(vol[-vol_long:]) + 1e-8)
    volume_shock_ratio = float(recent_vol_short / baseline_vol_long)
    vol_shock_on = volume_shock_ratio >= vol_ratio_threshold

    breakout = bool(ret_shock_on and vol_shock_on)
    return breakout, {
        "return_shock_z": return_shock_z,
        "volume_shock_ratio": volume_shock_ratio,
        "ret_shock_on": ret_shock_on,
        "vol_shock_on": vol_shock_on,
    }


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------


def compute_classifier_features(
    y: np.ndarray,
    vol: np.ndarray,
    config: dict | None = None,
) -> dict:
    """Compute all classifier features for a single asset.

    Args:
        y: Log price series for the asset.
        vol: Volume series for the asset.
        config: Optional config dict with:
            classifier_window, ret_win, ref_win, vol_short, vol_long,
            return_z_threshold, volume_ratio_threshold.

    Returns:
        Dict of features: efficiency, coherence, curvature, curvature_z,
        tradability, transition, breakout_ignite + breakout metrics.
    """
    cfg = config or {}
    cls_win = int(cfg.get("classifier_window", _DEFAULT_CLASSIFIER_WIN))

    min_needed = max(
        cls_win,
        int(cfg.get("ref_win", 20)) + int(cfg.get("ret_win", 3)) + 1,
        int(cfg.get("vol_long", 20)),
    )

    if len(y) < min_needed or len(vol) < min_needed:
        return {
            "efficiency": np.nan,
            "coherence": np.nan,
            "curvature": np.nan,
            "curvature_z": np.nan,
            "tradability": np.nan,
            "transition": np.nan,
            "breakout_ignite": False,
            "return_shock_z": np.nan,
            "volume_shock_ratio": np.nan,
            "ret_shock_on": False,
            "vol_shock_on": False,
        }

    y_cls = np.asarray(y[-cls_win:], dtype=float)

    efficiency = compute_efficiency(y_cls)
    coherence, curvature = compute_coherence_and_curvature(y_cls)

    local_ret_std = max(float(np.std(np.diff(y_cls), ddof=1)), 1e-8)
    curvature_z = float(abs(curvature) / local_ret_std)

    tradability = float(0.65 * efficiency + 0.35 * coherence)
    transition = curvature_z

    breakout_ignite, breakout_metrics = compute_breakout_metrics(
        y=np.asarray(y, dtype=float),
        vol=np.asarray(vol, dtype=float),
        ret_win=int(cfg.get("ret_win", 3)),
        ref_win=int(cfg.get("ref_win", 20)),
        vol_short=int(cfg.get("vol_short", 3)),
        vol_long=int(cfg.get("vol_long", 20)),
        return_z_threshold=float(cfg.get("return_z_threshold", _DEFAULT_RETURN_Z)),
        vol_ratio_threshold=float(cfg.get("volume_ratio_threshold", _DEFAULT_VOL_RATIO)),
    )

    features = {
        "efficiency": efficiency,
        "coherence": coherence,
        "curvature": curvature,
        "curvature_z": curvature_z,
        "tradability": tradability,
        "transition": transition,
        "breakout_ignite": breakout_ignite,
    }
    features.update(breakout_metrics)
    return features


# ---------------------------------------------------------------------------
# Profile detection
# ---------------------------------------------------------------------------


def detect_profile(
    features: dict,
    prev_profile: str | None = None,
    config: dict | None = None,
) -> str:
    """Determine the trading profile for an asset based on its features.

    Args:
        features: Output of compute_classifier_features().
        prev_profile: Previous profile for hysteresis.
        config: Optional config with threshold overrides.

    Returns:
        Profile name: Trend_follower, Breakseeker, Defender, Activist,
        or All_other.
    """
    cfg = config or {}

    tradability = features.get("tradability")
    transition = features.get("transition")
    breakout = features.get("breakout_ignite", False)

    if not np.isfinite(tradability) or not np.isfinite(transition):
        return "All_other"

    tradability_hi = float(cfg.get("tradability_hi", _DEFAULT_TREND_HI))
    tradability_lo = float(cfg.get("tradability_lo", _DEFAULT_TREND_LO))
    defender_trad = float(cfg.get("defender_tradability", _DEFAULT_DEFENDER))
    breakout_min_trad = float(cfg.get("breakout_min_tradability", _DEFAULT_BREAKOUT_MIN))
    transition_z = float(cfg.get("transition_z_threshold", _DEFAULT_TRANSITION_Z))

    if breakout and tradability >= breakout_min_trad:
        new_profile = "Breakseeker"
    elif tradability >= tradability_hi and transition < transition_z:
        new_profile = "Trend_follower"
    elif tradability >= tradability_lo and transition >= transition_z:
        new_profile = "Activist"
    elif tradability < defender_trad:
        new_profile = "Defender"
    else:
        new_profile = "All_other"

    # Hysteresis: avoid flip-flopping
    if prev_profile is not None and prev_profile in PROFILES:
        hysteresis = float(cfg.get("trend_hysteresis", _DEFAULT_HYSTERESIS))

        if prev_profile == "Trend_follower" and new_profile == "All_other":
            if tradability >= tradability_hi * hysteresis:
                new_profile = prev_profile

        if (
            prev_profile == "Breakseeker"
            and new_profile == "All_other"
            and breakout
        ):
            new_profile = prev_profile

    return new_profile


# ---------------------------------------------------------------------------
# DataFrame-level API (works with our date x asset pipeline)
# ---------------------------------------------------------------------------


class ProfileClassifier:
    """Per-stock profile classifier that works with date × asset DataFrames.

    For each asset in the universe, computes the tradability profile and
    score on the latest date. Can also be applied historically.

    Usage:
        classifier = ProfileClassifier()
        profiles = classifier.classify(prices, volume)
        # Returns dict: {asset: {"profile": str, "tradability": float, ...}}
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def classify(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        prev_profiles: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        """Classify all assets on the latest date.

        Args:
            prices: (date × asset) price DataFrame.
            volume: (date × asset) volume DataFrame. If None, uses
                    absolute price returns as proxy.
            prev_profiles: Previous profiles for hysteresis.

        Returns:
            Dict mapping asset → {
                "profile": str,
                "tradability": float,
                "efficiency": float,
                "coherence": float,
                "breakout_ignite": bool,
                ...
            }
        """
        log_prices = np.log(prices.clip(lower=1e-8))
        if volume is None:
            volume = prices.pct_change(fill_method=None).abs()

        results = {}
        for asset in prices.columns:
            y = log_prices[asset].dropna().values
            v = volume[asset].dropna().values

            if len(y) < 10 or len(v) < 10:
                continue

            features = compute_classifier_features(y, v, self.config)
            prev = prev_profiles.get(asset) if prev_profiles else None
            profile = detect_profile(features, prev, self.config)

            results[asset] = {"profile": profile, **features}

        return results

    def classify_historical(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        window: int = 60,
    ) -> pd.DataFrame:
        """Classify each asset at each date historically.

        Returns:
            DataFrame (date × asset) of tradability scores.
        """
        import warnings

        log_prices = np.log(prices.clip(lower=1e-8))
        if volume is None:
            volume = prices.pct_change(fill_method=None).abs()

        tradability_df = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

        for asset in prices.columns:
            y = log_prices[asset].dropna()
            v = volume[asset].dropna()

            for i in range(window, len(y)):
                y_win = y.iloc[i - window:i].values
                v_win = v.iloc[i - window:i].values

                features = compute_classifier_features(y_win, v_win, self.config)
                tradability = features.get("tradability", np.nan)
                tradability_df.loc[y.index[i], asset] = tradability

        return tradability_df.astype(float)

    def compute_tradability_gate(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        min_tradability: float = 0.3,
    ) -> pd.DataFrame:
        """Compute a tradability multiplier: 1 = tradable, 0 = untradable.

        Returns (date × asset) DataFrame with values in {0, 1}.
        """
        trad = self.classify_historical(prices, volume)
        return (trad >= min_tradability).astype(float)


# ---------------------------------------------------------------------------
# Convenience: apply tradability gate to a signal
# ---------------------------------------------------------------------------


def apply_tradability_gate(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    min_tradability: float = 0.3,
    classifier: ProfileClassifier | None = None,
) -> pd.DataFrame:
    """Multiply alpha signal by tradability gate.

    Stocks below the tradability threshold get zeroed out. This prevents
    the portfolio from taking positions in noisy / choppy stocks where
    the factor signal is unreliable.

    Args:
        signal: (date × asset) alpha signal.
        prices: (date × asset) price data.
        volume: Optional (date × asset) volume data.
        min_tradability: Minimum tradability score (0-1).
        classifier: Reusable classifier instance.

    Returns:
        Gated signal (date × asset).
    """
    if classifier is None:
        classifier = ProfileClassifier()

    gate = classifier.compute_tradability_gate(prices, volume, min_tradability)

    # Align gate with signal
    common_dates = signal.index.intersection(gate.index)
    common_assets = signal.columns.intersection(gate.columns)

    result = signal.loc[common_dates, common_assets].copy()
    result *= gate.loc[common_dates, common_assets]

    logger.info(
        "Tradability gate applied: min_tradability=%.2f, %.1f%% of entries zeroed",
        min_tradability,
        (1 - gate.loc[common_dates, common_assets].mean().mean()) * 100,
    )
    return result
