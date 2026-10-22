"""Research holdout discipline: a window reserved for final evaluation only.

A *holdout* is a date window that may be used for the final, one-shot
evaluation of an already-selected strategy but must never participate in any
choice: no parameter sweep, no factor screening, no walk-forward training or
testing.  This module is the single place that defines the window and the
rules around it:

  - :class:`HoldoutSpec` — an inclusive ``[start_date, end_date]`` window,
    loaded from settings (``research.holdout``) or from the environment
    (``QUANT_HOLDOUT_START`` / ``QUANT_HOLDOUT_END``).  An empty configuration
    means "disabled" and returns ``None``.
  - :func:`sweep_guard_message` — returns ``None`` when a sweep is allowed, or
    a human-readable refusal when the configured data window overlaps the
    holdout.  Callers turn that message into ``HTTPException(422, ...)`` (API)
    or a non-zero exit (CLI) *before* running anything.
  - :meth:`HoldoutSpec.exclude_dates` / :meth:`HoldoutSpec.mask_frame` —
    drop holdout rows from a panel while preserving index order.

Window semantics (must match the tests): the holdout, like the configured
``data.start_date``/``data.end_date`` data window, is a **closed interval on
both ends**.  A data window ending the day *before* the holdout starts does
not overlap; one ending on the holdout's first day does.

Invalid configuration (start >= end, one date missing, unparsable date) raises
``ValueError`` — never a silent no-op.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

ENV_START = "QUANT_HOLDOUT_START"
ENV_END = "QUANT_HOLDOUT_END"

_EMPTY = (None, "")


def _parse_date(value: Any, *, source: str) -> date:
    """Parse an ISO date (allow a ``date``/``datetime``/``Timestamp``)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} must be an ISO date (YYYY-MM-DD), got {value!r}",
        ) from None


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _section(source: Any, key: str) -> Any:
    """Read a config section from either a mapping or an object."""
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _field(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


@dataclass(frozen=True)
class HoldoutSpec:
    """An inclusive date window reserved for final evaluation only."""

    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        start = _parse_date(self.start_date, source="holdout start_date")
        end = _parse_date(self.end_date, source="holdout end_date")
        # A holdout must be a real window; start == end would silently reserve
        # a single day, which the configuration contract forbids.
        if start >= end:
            raise ValueError(
                "holdout start_date must be strictly before end_date, got "
                f"{start.isoformat()} >= {end.isoformat()}",
            )
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)

    # ── loading ──────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> HoldoutSpec | None:
        """Build a spec from a Config object (or its raw mapping).

        Disabled configuration (no ``research.holdout`` section, or both dates
        empty/null) returns ``None``.  A half-configured window is an error:
        silently treating it as disabled is how a holdout gets forgotten.
        """
        if settings is None:
            return None
        research = _section(settings, "research")
        holdout = _field(research, "holdout")
        start = _clean(_field(holdout, "start_date"))
        end = _clean(_field(holdout, "end_date"))
        if not start and not end:
            return None
        if not start or not end:
            raise ValueError(
                "holdout requires both start_date and end_date (or neither); "
                f"got start={start!r}, end={end!r}",
            )
        return cls(
            _parse_date(start, source="holdout start_date"),
            _parse_date(end, source="holdout end_date"),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HoldoutSpec | None:
        """Build a spec from ``QUANT_HOLDOUT_START`` / ``QUANT_HOLDOUT_END``.

        This is an explicit loader, not an implicit fallback: the API/CLI
        guards read the YAML configuration via :meth:`from_settings`.
        Disabled (both unset/empty) returns ``None``; a half-set pair raises.
        """
        env = os.environ if env is None else env
        start = _clean(env.get(ENV_START))
        end = _clean(env.get(ENV_END))
        if not start and not end:
            return None
        if not start or not end:
            raise ValueError(
                f"holdout environment requires both {ENV_START} and {ENV_END} "
                f"(or neither); got start={start!r}, end={end!r}",
            )
        return cls(
            _parse_date(start, source=ENV_START),
            _parse_date(end, source=ENV_END),
        )

    # ── queries / masking ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, str]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    def overlaps(self, start: Any, end: Any) -> bool:
        """Whether the inclusive ``[start, end]`` window touches the holdout.

        Both ends are inclusive, so a data window ending the day before the
        holdout begins does not overlap; ending on the holdout's first day
        does.  Invalid windows raise ``ValueError``.
        """
        window_start = _parse_date(start, source="data window start")
        window_end = _parse_date(end, source="data window end")
        if window_start > window_end:
            raise ValueError(
                f"data window start ({window_start}) must be <= end "
                f"({window_end})",
            )
        return window_start <= self.end_date and window_end >= self.start_date

    def _row_mask(self, index: pd.Index, level: str | None = None) -> np.ndarray:
        if isinstance(index, pd.MultiIndex):
            dates = pd.to_datetime(index.get_level_values(level or index.names[0]))
        else:
            dates = pd.to_datetime(index)
        return np.asarray(
            (dates >= pd.Timestamp(self.start_date))
            & (dates <= pd.Timestamp(self.end_date)),
            dtype=bool,
        )

    def excluded_days(self, index: pd.Index) -> int:
        """Number of holdout-labelled rows present in ``index``."""
        return int(self._row_mask(index).sum())

    def exclude_dates(self, index: pd.Index) -> pd.Index:
        """Drop holdout rows from an index, preserving the remaining order."""
        return index[~self._row_mask(index)]

    def mask_frame(
        self, frame: pd.DataFrame | pd.Series | None, level: str | None = None,
    ) -> pd.DataFrame | pd.Series | None:
        """Drop holdout rows from a date-indexed frame/series.

        ``level`` selects the date level for a MultiIndex (e.g. financials are
        indexed by ``(date, asset)``).  ``None`` passes through.
        """
        if frame is None:
            return None
        mask = self._row_mask(frame.index, level=level)
        return frame.loc[~mask]


def sweep_guard_message(config: Any) -> str | None:
    """Return a refusal message when a sweep may not touch the holdout.

    ``None`` means the sweep may run unchanged (holdout disabled or the data
    window does not overlap it).  Raises ``ValueError`` for an invalid holdout
    configuration or when the holdout is enabled but the data window is
    incomplete — failing closed instead of running blind.
    """
    spec = HoldoutSpec.from_settings(config)
    if spec is None:
        return None

    data = _section(config, "data")
    start = _clean(_field(data, "start_date"))
    end = _clean(_field(data, "end_date"))
    if not start or not end:
        raise ValueError(
            "holdout is enabled but the data window (data.start_date/"
            "data.end_date) is incomplete; refusing to run a sweep",
        )
    if spec.overlaps(start, end):
        return (
            f"data window {start}..{end} overlaps the research holdout "
            f"[{spec.start_date.isoformat()}..{spec.end_date.isoformat()}]: "
            "parameter selection/tuning must not touch the holdout. Change the "
            "data window or disable the holdout (research.holdout)."
        )
    return None


def load_configured_holdout(config: Any = None) -> HoldoutSpec | None:
    """Resolve the active holdout from the platform configuration.

    The YAML config is the authoritative source.  ``config`` may be passed in
    (e.g. an already-loaded ``Config``); otherwise the default config chain
    (``QUANT_CONFIG`` / ``config/default.yaml``) is loaded.
    """
    if config is None:
        from quant_platform.utils.config import load_config

        config = load_config()
    return HoldoutSpec.from_settings(config)
