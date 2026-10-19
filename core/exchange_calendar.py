"""A-share exchange trading calendar.

The scheduler used to treat every weekday as a trading day, so it opened the
session and allowed orders on National Day, Spring Festival and every other
weekday exchange holiday. The dates below are the SSE/SZSE closures published
by the exchanges; weekends are handled separately.

Extend `EXCHANGE_HOLIDAYS` when the next year's schedule is published (the
SSE normally announces it in December). Additional one-off closures can be
supplied through `QUANT_EXCHANGE_HOLIDAYS` (comma-separated YYYY-MM-DD).
"""

from __future__ import annotations

import os
from datetime import date

# Weekday closures only; Saturday/Sunday are already non-trading days.
EXCHANGE_HOLIDAYS: frozenset[str] = frozenset({
    # 2024
    "2024-01-01",
    "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14",
    "2024-02-15", "2024-02-16",
    "2024-04-04", "2024-04-05",
    "2024-05-01", "2024-05-02", "2024-05-03",
    "2024-06-10",
    "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07",
    # 2025
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-03", "2025-02-04",
    "2025-04-04",
    "2025-05-01", "2025-05-02", "2025-05-05",
    "2025-06-02",
    "2025-10-01", "2025-10-02", "2025-10-03",
    "2025-10-06", "2025-10-07", "2025-10-08",
    # 2026 (SSE announcement of 2025-12-22)
    "2026-01-01", "2026-01-02",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-23",
    "2026-04-06",
    "2026-05-01", "2026-05-04", "2026-05-05",
    "2026-06-19",
    "2026-09-25",
    "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",
})


def _extra_holidays() -> set[str]:
    raw = os.environ.get("QUANT_EXCHANGE_HOLIDAYS", "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def is_trading_day(day: date) -> bool:
    """Whether the exchange is open on a calendar date."""
    if day.weekday() >= 5:
        return False
    key = day.isoformat()
    if key in EXCHANGE_HOLIDAYS:
        return False
    return key not in _extra_holidays()
