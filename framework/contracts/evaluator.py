"""Evaluator — the trust gate for research conclusions.

Protocol over the existing `research/validation.py` DSR / BH-FDR machinery —
the one genuinely trustworthy piece of the current codebase. This is the
"audit" authority: per Principle 7, NO conclusion (human or AI) is marked
trustworthy without passing through here.

Reference Implementation wraps `research/validation.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        n_trials: int,
    ) -> dict:
        """Return {ic, rank_ic, icir, ic_positive_ratio, dsr, bh_fdr_pass, ...}.

        `n_trials` feeds multiple-testing correction (DSR/BH-FDR). The output
        dict is what the Registry stores and the Report renders.
        """
        ...
