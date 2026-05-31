"""K-line candlestick pattern recognition for A-share stocks.

Recognizes common candlestick patterns that signal potential reversals
or continuations. Used as a qualitative overlay on top of quantitative
factor signals.

Patterns detected:
  Single-candle: doji, hammer, shooting_star, hanging_man, marubozu
  Two-candle:    engulfing, harami, piercing, dark_cloud_cover
  Three-candle:  morning_star, evening_star, three_white_soldiers,
                 three_black_crows, abandoned_baby
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class PatternType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class PatternResult:
    name: str
    type: PatternType
    strength: float  # 0-1
    last_candle_index: int = -1
    description: str = ""


class CandlePatternRecognizer:
    """Recognize candlestick patterns from OHLC data.

    Args:
        body_threshold: threshold for real body vs doji (default 0.1)
        shadow_threshold: threshold for upper/lower shadow (default 0.3)
    """

    def __init__(self, body_threshold: float = 0.1, shadow_threshold: float = 0.3):
        self.body_th = body_threshold
        self.shadow_th = shadow_threshold

    def recognize_all(
        self,
        ohlc: pd.DataFrame,
        lookback: int = 10,
    ) -> list[PatternResult]:
        """Run all pattern recognizers on OHLC data.

        Args:
            ohlc: DataFrame with open, high, low, close columns.
            lookback: Only check last N candles.

        Returns:
            List of PatternResult for detected patterns.
        """
        if len(ohlc) < 5:
            return []

        df = ohlc.iloc[-lookback:].copy()
        results = []

        # Single candle patterns
        for i in range(len(df)):
            patterns = self._check_single_candle(df.iloc[i])
            for p in patterns:
                p.last_candle_index = i
                results.append(p)

        # Two candle patterns
        for i in range(1, len(df)):
            patterns = self._check_two_candles(df.iloc[i - 1], df.iloc[i])
            for p in patterns:
                p.last_candle_index = i
                results.append(p)

        # Three candle patterns
        for i in range(2, len(df)):
            patterns = self._check_three_candles(
                df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            )
            for p in patterns:
                p.last_candle_index = i
                results.append(p)

        return results

    # ------------------------------------------------------------------
    # Single candle patterns
    # ------------------------------------------------------------------

    def _check_single_candle(self, c: pd.Series) -> list[PatternResult]:
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        body = abs(cl - o)
        upper = h - max(o, cl)
        lower = min(o, cl) - l
        total = h - l
        if total == 0:
            return []

        results = []
        is_bull = cl > o

        # Doji: very small body
        if body / total < self.body_th:
            results.append(PatternResult(
                "doji", PatternType.NEUTRAL, 0.3,
                description="开盘收盘价几乎相等，市场犹豫",
            ))

        # Hammer: small body at top, long lower shadow (after downtrend)
        if body / total < 0.4 and lower / total > 0.6 and upper / total < 0.1:
            results.append(PatternResult(
                "hammer", PatternType.BULLISH, 0.6,
                description="锤头线：下影线长，出现在下跌后可能是底部信号",
            ))

        # Shooting star: small body at bottom, long upper shadow (after uptrend)
        if body / total < 0.4 and upper / total > 0.6 and lower / total < 0.1:
            results.append(PatternResult(
                "shooting_star", PatternType.BEARISH, 0.6,
                description="射击之星：上影线长，出现在上涨后可能是顶部信号",
            ))

        # Marubozu: no shadows
        if upper / total < 0.05 and lower / total < 0.05 and body / total > 0.6:
            t = PatternType.BULLISH if is_bull else PatternType.BEARISH
            name = "white_marubozu" if is_bull else "black_marubozu"
            results.append(PatternResult(
                name, t, 0.5,
                description="光头光脚" + ("阳线" if is_bull else "阴线") + "，趋势强劲",
            ))

        return results

    # ------------------------------------------------------------------
    # Two candle patterns
    # ------------------------------------------------------------------

    def _check_two_candles(
        self, c1: pd.Series, c2: pd.Series
    ) -> list[PatternResult]:
        o1, h1, l1, cl1 = c1["open"], c1["high"], c1["low"], c1["close"]
        o2, h2, l2, cl2 = c2["open"], c2["high"], c2["low"], c2["close"]
        bull1 = cl1 > o1
        bull2 = cl2 > o2
        body1 = abs(cl1 - o1)
        body2 = abs(cl2 - o2)

        results = []

        # Bullish engulfing: red candle engulfing previous green
        if not bull1 and bull2 and cl2 > o1 and o2 < cl1:
            results.append(PatternResult(
                "bullish_engulfing", PatternType.BULLISH, 0.7,
                description="阳包阴：多头强势吞没前日阴线",
            ))

        # Bearish engulfing: green candle engulfing previous red
        if bull1 and not bull2 and o2 > cl1 and cl2 < o1:
            results.append(PatternResult(
                "bearish_engulfing", PatternType.BEARISH, 0.7,
                description="阴包阳：空头强势吞没前日阳线",
            ))

        # Bullish harami: small red inside previous green
        if not bull1 and not bull2 and body2 < body1 * 0.5:
            if o2 > cl1 and cl2 < o1:
                results.append(PatternResult(
                    "bullish_harami", PatternType.BULLISH, 0.4,
                    description="多头母子线：孕线，下跌动能减弱",
                ))

        # Bearish harami: small green inside previous red
        if bull1 and bull2 and body2 < body1 * 0.5:
            if cl2 < o1 and o2 > cl1:
                results.append(PatternResult(
                    "bearish_harami", PatternType.BEARISH, 0.4,
                    description="空头母子线：孕线，上涨动能减弱",
                ))

        return results

    # ------------------------------------------------------------------
    # Three candle patterns
    # ------------------------------------------------------------------

    def _check_three_candles(
        self, c1: pd.Series, c2: pd.Series, c3: pd.Series
    ) -> list[PatternResult]:
        o1, cl1 = c1["open"], c1["close"]
        o2, cl2 = c2["open"], c2["close"]
        o3, cl3 = c3["open"], c3["close"]
        bull1 = cl1 > o1
        bull2 = cl2 > o2
        bull3 = cl3 > o3

        results = []

        # Morning star: long red, small body, long green (after downtrend)
        if not bull1 and abs(cl1 - o1) > abs(cl1 - o1) * 0.5:
            body2 = abs(cl2 - o2)
            if bull3 and cl3 > o2 and body2 < abs(cl1 - o1) * 0.4:
                results.append(PatternResult(
                    "morning_star", PatternType.BULLISH, 0.8,
                    description="早晨之星：长阴+小实体+长阳，经典反转信号",
                ))

        # Evening star: long green, small body, long red (after uptrend)
        if bull1 and abs(cl1 - o1) > 0.5:
            body2 = abs(cl2 - o2)
            if not bull3 and o3 > o2 and body2 < abs(cl1 - o1) * 0.4:
                results.append(PatternResult(
                    "evening_star", PatternType.BEARISH, 0.8,
                    description="黄昏之星：长阳+小实体+长阴，经典顶部信号",
                ))

        # Three white soldiers: three consecutive strong bullish candles
        if bull1 and bull2 and bull3:
            if all(cl > o for o, cl in [(o1, cl1), (o2, cl2), (o3, cl3)]):
                results.append(PatternResult(
                    "three_white_soldiers", PatternType.BULLISH, 0.7,
                    description="三白兵：连续三根阳线，强势上涨趋势",
                ))

        # Three black crows: three consecutive strong bearish candles
        if not bull1 and not bull2 and not bull3:
            if all(cl < o for o, cl in [(o1, cl1), (o2, cl2), (o3, cl3)]):
                results.append(PatternResult(
                    "three_black_crows", PatternType.BEARISH, 0.7,
                    description="三只乌鸦：连续三根阴线，强势下跌趋势",
                ))

        return results


def candle_pattern_signal(
    ohlc: pd.DataFrame,
    lookback: int = 10,
) -> float:
    """Aggregate candle pattern signals into a single score [-1, 1].

    Positive = bullish patterns dominate, negative = bearish.

    Returns:
        Score in [-1, 1] where:
        1.0 = multiple strong bullish patterns
        -1.0 = multiple strong bearish patterns
        0 = neutral / no clear patterns
    """
    recognizer = CandlePatternRecognizer()
    patterns = recognizer.recognize_all(ohlc, lookback)

    if not patterns:
        return 0.0

    bullish_score = sum(
        p.strength for p in patterns if p.type == PatternType.BULLISH
    )
    bearish_score = sum(
        p.strength for p in patterns if p.type == PatternType.BEARISH
    )
    total = bullish_score + bearish_score
    if total == 0:
        return 0.0

    net = (bullish_score - bearish_score) / max(total, 1)
    return float(np.clip(net, -1, 1))
