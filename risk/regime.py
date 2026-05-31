"""Market regime detection.

Identifies the current market regime using multiple approaches:
1. Volatility regime: low/medium/high/extreme based on rolling vol
2. Trend regime: bull/bear/sideways based on moving averages
3. Correlation regime: normal/stressed based on cross-asset correlation
4. Hidden Markov Model: 2-3 state regime switching

Used to dynamically adjust strategy parameters (position sizing,
factor weights, stop-loss levels) based on market conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeType:
    LOW_VOL = "low_volatility"
    MEDIUM_VOL = "medium_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"

    NORMAL = "normal_correlation"
    STRESSED = "stressed_correlation"


class VolatilityRegimeDetector:
    """Detect volatility regime using rolling standard deviation.

    Classifies current volatility relative to historical distribution:
    - Low: < 25th percentile
    - Medium: 25th-75th percentile
    - High: 75th-95th percentile
    - Extreme: > 95th percentile
    """

    def __init__(self, lookback: int = 252, vol_window: int = 21):
        self.lookback = lookback
        self.vol_window = vol_window

    def detect(self, returns: pd.Series) -> dict:
        """Detect current volatility regime."""
        if len(returns) < self.lookback:
            return {"regime": RegimeType.MEDIUM_VOL, "confidence": 0.0}

        # Rolling annualized volatility
        rolling_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 100:
            return {"regime": RegimeType.MEDIUM_VOL, "confidence": 0.0}

        current_vol = rolling_vol.iloc[-1]
        historical_vol = rolling_vol.iloc[:-1]

        # Percentile ranking
        percentile = (historical_vol < current_vol).mean()

        if percentile < 0.25:
            regime = RegimeType.LOW_VOL
        elif percentile < 0.75:
            regime = RegimeType.MEDIUM_VOL
        elif percentile < 0.95:
            regime = RegimeType.HIGH_VOL
        else:
            regime = RegimeType.EXTREME_VOL

        # Vol of vol (regime stability)
        vol_of_vol = historical_vol.rolling(63).std().iloc[-1] if len(historical_vol) > 63 else 0
        confidence = max(0, 1 - vol_of_vol / current_vol) if current_vol > 0 else 0

        return {
            "regime": regime,
            "current_vol": round(float(current_vol), 4),
            "percentile": round(float(percentile), 4),
            "confidence": round(float(confidence), 4),
            "vol_25pct": round(float(historical_vol.quantile(0.25)), 4),
            "vol_75pct": round(float(historical_vol.quantile(0.75)), 4),
            "vol_95pct": round(float(historical_vol.quantile(0.95)), 4),
        }


class TrendRegimeDetector:
    """Detect trend regime using moving average crossover.

    - Bull: short MA > long MA and price > short MA
    - Bear: short MA < long MA and price < short MA
    - Sideways: MAs intertwined or price oscillating around MAs
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def detect(self, prices: pd.Series) -> dict:
        """Detect current trend regime."""
        if len(prices) < self.long_window:
            return {"regime": RegimeType.SIDEWAYS, "confidence": 0.0}

        ma_short = prices.rolling(self.short_window).mean()
        ma_long = prices.rolling(self.long_window).mean()

        current_price = prices.iloc[-1]
        current_short = ma_short.iloc[-1]
        current_long = ma_long.iloc[-1]

        # MA crossover signals
        ma_spread = (current_short - current_long) / current_long
        price_vs_short = (current_price - current_short) / current_short

        # Recent MA crossovers (last 20 days)
        recent_spread = (ma_short.iloc[-20:] - ma_long.iloc[-20:]) / ma_long.iloc[-20:]
        crossovers = ((recent_spread > 0) != (recent_spread.iloc[0] > 0)).sum()

        if ma_spread > 0.02 and price_vs_short > 0:
            regime = RegimeType.BULL
            confidence = min(abs(ma_spread) * 10, 1.0)
        elif ma_spread < -0.02 and price_vs_short < 0:
            regime = RegimeType.BEAR
            confidence = min(abs(ma_spread) * 10, 1.0)
        else:
            regime = RegimeType.SIDEWAYS
            confidence = max(0, 1 - abs(ma_spread) * 20)

        # Lower confidence if many crossovers (choppy market)
        if crossovers > 3:
            confidence *= 0.5

        return {
            "regime": regime,
            "ma_spread": round(float(ma_spread), 4),
            "price_vs_short_ma": round(float(price_vs_short), 4),
            "recent_crossovers": int(crossovers),
            "confidence": round(float(confidence), 4),
            "ma_50": round(float(current_short), 2),
            "ma_200": round(float(current_long), 2),
        }


class CorrelationRegimeDetector:
    """Detect correlation regime (normal vs stressed).

    During market stress, correlations spike (everything falls together).
    Measured by average pairwise correlation of recent returns.
    """

    def __init__(self, lookback: int = 63, threshold: float = 0.5):
        self.lookback = lookback
        self.threshold = threshold

    def detect(self, returns_matrix: pd.DataFrame) -> dict:
        """Detect correlation regime from multi-asset returns."""
        if len(returns_matrix) < self.lookback:
            return {"regime": RegimeType.NORMAL, "confidence": 0.0}

        recent = returns_matrix.iloc[-self.lookback:]
        corr_matrix = recent.corr()

        # Average pairwise correlation (excluding diagonal)
        n = len(corr_matrix)
        if n < 2:
            return {"regime": RegimeType.NORMAL, "confidence": 0.0}

        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr_matrix.values[mask].mean()

        # Historical rolling correlation for comparison
        hist_corrs = []
        for i in range(self.lookback, len(returns_matrix), 21):
            window = returns_matrix.iloc[i - self.lookback:i]
            c = window.corr()
            hist_corrs.append(c.values[mask].mean())

        if hist_corrs:
            percentile = (np.array(hist_corrs) < avg_corr).mean()
        else:
            percentile = 0.5

        regime = RegimeType.STRESSED if avg_corr > self.threshold else RegimeType.NORMAL
        confidence = min(abs(avg_corr - self.threshold) * 5, 1.0)

        return {
            "regime": regime,
            "avg_correlation": round(float(avg_corr), 4),
            "percentile": round(float(percentile), 4),
            "confidence": round(float(confidence), 4),
            "threshold": self.threshold,
            "n_assets": n,
        }


class CompositeRegimeDetector:
    """Combines multiple regime detectors for a holistic view.

    Weights:
    - Volatility: 40% (most reliable)
    - Trend: 35% (directional)
    - Correlation: 25% (stress indicator)
    """

    def __init__(self):
        self.vol_detector = VolatilityRegimeDetector()
        self.trend_detector = TrendRegimeDetector()
        self.corr_detector = CorrelationRegimeDetector()

    def detect(
        self,
        returns: pd.Series,
        prices: pd.Series,
        returns_matrix: pd.DataFrame | None = None,
    ) -> dict:
        """Run all detectors and produce composite regime."""
        vol_result = self.vol_detector.detect(returns)
        trend_result = self.trend_detector.detect(prices)

        corr_result = {"regime": RegimeType.NORMAL, "confidence": 0.0}
        if returns_matrix is not None and len(returns_matrix.columns) > 5:
            corr_result = self.corr_detector.detect(returns_matrix)

        # Composite risk score (0=calm, 1=extreme)
        vol_score = vol_result.get("percentile", 0.5)
        trend_score = 0.8 if trend_result["regime"] == RegimeType.BEAR else 0.2
        corr_score = corr_result.get("avg_correlation", 0)

        composite_risk = 0.4 * vol_score + 0.35 * trend_score + 0.25 * corr_score

        # Determine overall regime
        if composite_risk > 0.75:
            overall = "risk_off"
            recommendation = "Reduce exposure, tighten stops, increase hedging"
        elif composite_risk > 0.5:
            overall = "cautious"
            recommendation = "Normal exposure with tighter risk limits"
        elif composite_risk > 0.25:
            overall = "neutral"
            recommendation = "Standard positioning"
        else:
            overall = "risk_on"
            recommendation = "Full exposure, consider increasing factor bets"

        return {
            "overall_regime": overall,
            "composite_risk_score": round(float(composite_risk), 4),
            "recommendation": recommendation,
            "volatility": vol_result,
            "trend": trend_result,
            "correlation": corr_result,
        }

    def get_execution_params(self, returns: pd.Series | None = None) -> dict:
        """Get regime-adaptive execution parameters.

        Inspired by stock-trader-ai's regime-adaptive stop-loss and
        position sizing. Returns different parameters based on the
        current market regime.

        Returns:
            Dict with keys:
                regime: Current overall regime.
                max_position_pct: Max single position as % of portfolio.
                max_total_positions: Max number of simultaneous positions.
                stop_loss_pct: Stop-loss threshold.
                take_profit_pct: Take-profit threshold.
                max_leverage: Max leverage.
                risk_multiplier: Scale factor [0,1] for position sizing.
        """
        if returns is not None:
            result = self.detect(returns)
        else:
            result = self.detect()

        overall = result["overall_regime"]

        if overall == "risk_off":
            params = {
                "regime": "bear",
                "max_position_pct": 0.03,
                "max_total_positions": 3,
                "stop_loss_pct": -0.04,
                "take_profit_pct": 0.10,
                "max_leverage": 1.0,
                "risk_multiplier": 0.4,
            }
        elif overall == "cautious":
            params = {
                "regime": "range",
                "max_position_pct": 0.05,
                "max_total_positions": 5,
                "stop_loss_pct": -0.07,
                "take_profit_pct": 0.15,
                "max_leverage": 1.0,
                "risk_multiplier": 0.6,
            }
        elif overall == "neutral":
            params = {
                "regime": "range",
                "max_position_pct": 0.07,
                "max_total_positions": 6,
                "stop_loss_pct": -0.08,
                "take_profit_pct": 0.20,
                "max_leverage": 1.2,
                "risk_multiplier": 0.8,
            }
        else:  # risk_on
            params = {
                "regime": "bull",
                "max_position_pct": 0.10,
                "max_total_positions": 8,
                "stop_loss_pct": -0.10,
                "take_profit_pct": 0.25,
                "max_leverage": 1.5,
                "risk_multiplier": 1.0,
            }

        params["composite_risk_score"] = result["composite_risk_score"]
        params["recommendation"] = result["recommendation"]
        return params
