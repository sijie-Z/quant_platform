"""Time-segmented trading rules for A-share intraday sessions.

Inspired by AlphaPilot Pro's time-aware threshold adjustment. A-share
trading sessions have distinct characteristics that affect strategy:

- Opening (09:30-10:00): High volatility, false signals common
- Mid-morning (10:00-11:30): Trend establishment, most reliable
- Lunch break (11:30-13:00): No trading
- Mid-afternoon (13:00-14:30): Continuation, afternoon reversal risk
- Closing (14:30-15:00): Position squaring, volatility spike
- Trailing auction (14:57-15:00): Only limit orders, no cancellation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TimeSegment:
    name: str
    start: str  # HHMM
    end: str
    volatility_risk: str  # low / medium / high
    description: str


SEGMENTS = [
    TimeSegment("opening", "0930", "1000", "high",
                "开盘冲高/回落，假信号多，不做买入"),
    TimeSegment("mid_morning", "1000", "1130", "low",
                "趋势确立，最可靠交易时段"),
    TimeSegment("lunch_break", "1130", "1300", "low",
                "午休，无交易"),
    TimeSegment("mid_afternoon", "1300", "1430", "medium",
                "下午延续，注意午后反转"),
    TimeSegment("closing_auction", "1430", "1457", "high",
                "尾盘博弈，机构调仓，波动大"),
    TimeSegment("final_auction", "1457", "1500", "high",
                "集合竞价，仅限价单，不可撤单"),
]


def get_current_segment(time_str: str | None = None) -> TimeSegment | None:
    """Get the current trading segment."""
    if time_str is None:
        time_str = datetime.now().strftime("%H%M")
    for seg in SEGMENTS:
        if seg.start <= time_str < seg.end:
            return seg
    return None


def in_trading_hours(time_str: str | None = None) -> bool:
    """Check if within A-share trading hours."""
    if time_str is None:
        time_str = datetime.now().strftime("%H%M")
    return ("0930" <= time_str < "1130") or ("1300" <= time_str < "1500")


def adjust_risk_multiplier(
    base: float,
    time_str: str | None = None,
) -> float:
    """Adjust risk multiplier based on time segment.

    High volatility segments → reduce risk.
    Low volatility segments → use base.

    Args:
        base: Base risk multiplier [0, 1].
        time_str: Current time HHMM.

    Returns:
        Adjusted risk multiplier.
    """
    seg = get_current_segment(time_str)
    if seg is None:
        return 0.0  # Out of trading hours

    if seg.volatility_risk == "high":
        return base * 0.6
    elif seg.volatility_risk == "medium":
        return base * 0.8
    return base
