"""A-share exchange trading calendar — compatibility re-export.

The canonical implementation moved to ``quant_platform.utils.exchange_calendar``
so that research-core modules (e.g. the synthetic data provider under
``data/``) can share the holiday table without importing ``core/``, which the
core -> live boundary in ``tests/test_architecture/test_core_boundaries.py``
forbids. Live modules may keep importing this path; it re-exports the same
objects, so ``core.exchange_calendar.EXCHANGE_HOLIDAYS`` and
``utils.exchange_calendar.EXCHANGE_HOLIDAYS`` are one and the same frozenset.
"""

from __future__ import annotations

from quant_platform.utils.exchange_calendar import (
    EXCHANGE_HOLIDAYS,
    _extra_holidays,
    is_trading_day,
    trading_days,
)

__all__ = [
    "EXCHANGE_HOLIDAYS",
    "is_trading_day",
    "trading_days",
    "_extra_holidays",
]
