"""Factor — cross-sectional factor contract.

Protocol over the existing `factors/` registry (FactorRegistry /
BaseFactor). Reference Implementation wraps the concrete factor classes so the
research loop depends on the Protocol, not the concrete class — allowing
future DSL/AI-generated factors to slot in without changing the loop.

Per Principle 5, factor ENGINE lives in Capability; concrete factor CONTENT
(names, tuned params, results) is Knowledge (lab) / Production (prod).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Factor(Protocol):
    @property
    def name(self) -> str:
        ...

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Cross-sectional factor values indexed by asset for the latest date."""
        ...
