"""A-share exchange trading calendar (canonical implementation).

The scheduler used to treat every weekday as a trading day, so it opened the
session and allowed orders on National Day, Spring Festival and every other
weekday exchange holiday. The dates below are the SSE/SZSE closures published
by the exchanges; weekends are handled separately.

This module lives under ``utils`` (the research core) rather than ``core`` so
that research-side consumers -- the synthetic data provider in particular --
can use the same holiday table as the live scheduler without violating the
core -> live import boundary enforced by
``tests/test_architecture/test_core_boundaries.py``. ``core.exchange_calendar``
re-exports these names for the existing live callers.

Extend `EXCHANGE_HOLIDAYS` when the next year's schedule is published (the
SSE normally announces it in December). Additional one-off closures can be
supplied through `QUANT_EXCHANGE_HOLIDAYS` (comma-separated YYYY-MM-DD).
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

# Weekday closures only; Saturday/Sunday are already non-trading days.
# SSE/SZSE official closures. 2021-2023 were missing from the original table
# even though the default sample starts in 2021: the synthetic provider
# fabricated ~18 sessions per year (fills on days the exchange was closed).
EXCHANGE_HOLIDAYS: frozenset[str] = frozenset({
    # 2021
    "2021-01-01",
    "2021-02-11", "2021-02-12", "2021-02-15", "2021-02-16", "2021-02-17",
    "2021-04-05",
    "2021-05-03", "2021-05-04", "2021-05-05",
    "2021-06-14",
    "2021-09-20", "2021-09-21",
    "2021-10-01", "2021-10-04", "2021-10-05", "2021-10-06", "2021-10-07",
    # 2022
    "2022-01-03",
    "2022-01-31", "2022-02-01", "2022-02-02", "2022-02-03", "2022-02-04",
    "2022-04-04", "2022-04-05",
    "2022-05-02", "2022-05-03", "2022-05-04",
    "2022-06-03",
    "2022-09-12",
    "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06", "2022-10-07",
    # 2023
    "2023-01-02",
    "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
    "2023-04-05",
    "2023-05-01", "2023-05-02", "2023-05-03",
    "2023-06-22", "2023-06-23",
    "2023-09-29",
    "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06",
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


def trading_days(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Exchange sessions in ``[start, end]`` (weekdays minus closures).

    Business days are *not* an acceptable proxy: a five-year A-share sample
    contains roughly 7.5% weekday exchange holidays. Research date grids must
    line up with the sessions the live scheduler will actually trade.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    candidates = pd.bdate_range(start_ts, end_ts, name="date")
    mask = [is_trading_day(ts.date()) for ts in candidates]
    return pd.DatetimeIndex(candidates[mask], name="date")
