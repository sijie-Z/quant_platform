"""Resolve research date windows instead of hardcoding a stale end date.

Research scripts used to embed ``REAL_END = "2025-12-31"`` (or ``"20260523"``)
as a module constant, so every artifact silently described a window that grew
staler over time and nothing in the output recorded which window was used.

Resolution order:

1. an explicit argument (CLI flag),
2. the ``QUANT_RESEARCH_END_DATE`` environment variable,
3. today's local date.

Both ``YYYY-MM-DD`` and ``YYYYMMDD`` are accepted; the canonical return form is
``YYYY-MM-DD``.
"""

from __future__ import annotations

import os
import re
from datetime import date

ENV_VAR = "QUANT_RESEARCH_END_DATE"

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_COMPACT_RE = re.compile(r"^\d{8}$")


def resolve_end_date(
    value: str | None = None,
    *,
    today: date | None = None,
) -> str:
    """Return the research end date as ``YYYY-MM-DD``.

    Args:
        value: Explicit end date (e.g. from a CLI flag). Takes precedence.
        today: Injectable clock for tests.

    Raises:
        ValueError: if the supplied value is not a valid calendar date.
    """
    raw = value or os.environ.get(ENV_VAR)
    if raw is None or str(raw).strip() == "":
        return (today or date.today()).isoformat()

    raw = str(raw).strip()
    if _COMPACT_RE.match(raw):
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    if not _ISO_RE.match(raw):
        raise ValueError(
            f"invalid research end date {raw!r}; expected YYYY-MM-DD or YYYYMMDD"
        )
    try:
        parsed = date.fromisoformat(raw)  # validates month/day
    except ValueError as exc:
        raise ValueError(
            f"invalid research end date {raw!r}; expected YYYY-MM-DD or YYYYMMDD"
        ) from exc
    return parsed.isoformat()


def compact_date(value: str) -> str:
    """``YYYY-MM-DD`` -> ``YYYYMMDD`` (akshare/tencent interfaces)."""
    return value.replace("-", "")


def year_windows(start_year: int, end_date: str) -> list[tuple[str, str, str]]:
    """Yearly ``(label, start, end)`` windows from ``start_year`` to ``end_date``.

    The final window is clipped at ``end_date`` so a script never requests data
    past its declared as-of date.
    """
    end_year = int(end_date[:4])
    if end_year < start_year:
        return []
    windows: list[tuple[str, str, str]] = []
    for year in range(start_year, end_year + 1):
        end = f"{year}-12-31"
        if year == end_year:
            end = min(end, end_date)
        windows.append((str(year), f"{year}-01-01", end))
    return windows


def stamp_provenance(df, start: str, end: str):
    """Return ``df`` with ``data_start`` / ``data_end`` provenance columns.

    CSV artifacts carried no machine-readable record of the window they were
    computed on. Constant columns are the least invasive way to add it without
    changing the metric columns.
    """
    if df is None or len(df) == 0:
        # Still emit the columns so an empty artifact declares its window.
        if df is not None:
            df = df.copy()
            df["data_start"] = start
            df["data_end"] = end
        return df
    df = df.copy()
    df["data_start"] = start
    df["data_end"] = end
    return df
