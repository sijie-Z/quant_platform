"""UniverseProvider — stock universe contract with explicit honesty metadata.

This is the keystone of Trust First under real-world data friction (see
CONSTITUTION.md Principle 1 and the Builder-mode trust discipline).

The contract does NOT promise a perfect point-in-time universe. Instead it
MANDATES that every implementation declares, in machine-readable form, exactly
how trustworthy it is. The Registry and Report layers consume `trust_meta()`
to (a) record bias warnings per experiment and (b) auto-emit WARNINGs in
reports. This converts a silent approximation into a loud, machine-identifiable
one — preventing the "v0.1 already finished PIT universe" credibility slide.

Why this matters: a wrong universe changes the research question itself. Using
today's CSI300 constituents to backtest 2015 means you already know which
firms survived — that is survivorship bias, not measurement noise. We refuse
to hide it; we refuse to name an approximate provider "PIT" or "Trusted".

Reference Implementations:
    - CurrentConstituentUniverseProvider (approximate, NOT PIT) — v0.1
    - PITConstituentProvider (strict point-in-time)           — v0.2 placeholder
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import pandas as pd

#: Adjustment / honesty tag carried by every universe. Strictly typed strings
#: so the Registry can branch on them without free-form parsing.
UniverseKind = Literal[
    "current_constituent",   # today's index members, forward-applied to history (NOT PIT)
    "pit_constituent",       # strict point-in-time constituents per date (v0.2)
    "static_code_range",     # hardcoded code range (legacy, known-bias; legacy only)
]


@runtime_checkable
class UniverseProvider(Protocol):
    """Provides the tradeable universe per date.

    Implementations MUST call-shot their own trustworthiness via `trust_meta()`.
    The pipeline NEVER assumes PIT; it asks, and records the answer.
    """

    @property
    def kind(self) -> UniverseKind:
        """Categorical identifier of this universe's construction method."""
        ...

    def trust_meta(self) -> dict:
        """Machine-readable honesty declaration.

        Required keys:
            kind: UniverseKind
            pit: bool                  — is this strict point-in-time?
            bias_warning: list[str]    — e.g. ["survivorship_bias_possible"]
            description: str           — one-line human description

        The Registry persists this dict verbatim on every run; the Report
        layer renders WARNINGs from `bias_warning`.
        """
        ...

    def get_universe(self, date: pd.Timestamp) -> list[str]:
        """Return the list of asset codes tradeable on `date`.

        For non-PIT providers this may return the same snapshot regardless of
        date — that is allowed ONLY because `trust_meta()` flags it.
        """
        ...
