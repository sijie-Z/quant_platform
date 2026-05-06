"""Factor combination methods for alpha signal generation.

Takes multiple processed factor values and combines them into a single
alpha signal. Methods from simple to sophisticated:
- Equal weight: average of all factors
- IC-weighted: weight by historical Rank IC
- ICIR-weighted: weight by IC / IC_std (risk-adjusted predictive power)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def combine_equal_weight(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Simple equal-weighted factor combination.

    All factors contribute equally regardless of their individual
    predictive power. A robust but naive baseline.
    """
    if not factors:
        raise ValueError("No factors provided")

    aligned = _align_factors(factors)
    weights = {name: 1.0 / len(aligned) for name in aligned}
    return _weighted_sum(aligned, weights)


def combine_ic_weighted(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """Weight factors by their mean Rank IC over the lookback period.

    Factors with stronger recent predictive power get higher weight.
    Weights can be negative if a factor's IC is negative.
    """
    if not factors:
        raise ValueError("No factors provided")

    ic_values = {}
    for name, factor in factors.items():
        ic_series = rank_ic(factor, forward_returns)
        if len(ic_series) == 0:
            ic_values[name] = 0.0
            continue
        recent_ic = ic_series.iloc[-lookback:] if len(ic_series) > lookback else ic_series
        ic_values[name] = recent_ic.mean()

    # Normalize absolute values for weighting
    total_abs = sum(abs(v) for v in ic_values.values())
    if total_abs < 1e-10:
        weights = {name: 1.0 / len(factors) for name in factors}
    else:
        weights = {name: v / total_abs for name, v in ic_values.items()}

    aligned = _align_factors(factors)
    return _weighted_sum(aligned, weights)


def combine_icir_weighted(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    lookback: int = 252,
    min_icir: float = 0.0,
) -> pd.DataFrame:
    """Weight factors by Information Coefficient IR (ICIR = mean_IC / std_IC).

    ICIR measures risk-adjusted predictive power. A factor with high
    mean IC but even higher IC volatility gets a lower weight.

    Factors with ICIR below min_icir are excluded entirely.
    """
    if not factors:
        raise ValueError("No factors provided")

    icir_values = {}
    for name, factor in factors.items():
        ic_series = rank_ic(factor, forward_returns)
        if len(ic_series) < 20:
            icir_values[name] = 0.0
            continue
        recent_ic = ic_series.iloc[-lookback:] if len(ic_series) > lookback else ic_series
        summary = ic_summary(recent_ic)
        icir_values[name] = summary["icir"]

    # Filter and weight
    filtered = {name: v for name, v in icir_values.items() if v >= min_icir}
    if not filtered:
        logger.warning("No factors pass min_icir filter, using all with equal weight")
        filtered = {name: max(v, 0.01) for name, v in icir_values.items()}

    total = sum(max(v, 0) for v in filtered.values())
    if total < 1e-10:
        weights = {name: 1.0 / len(filtered) for name in filtered}
    else:
        weights = {name: max(v, 0) / total for name, v in filtered.items()}

    logger.info("ICIR weights: %s",
                 {name: f"{w:.3f}" for name, w in weights.items()})

    aligned = _align_factors(factors)
    aligned = {name: aligned[name] for name in weights if name in aligned}
    return _weighted_sum(aligned, weights)


def _align_factors(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align factor dates and assets."""
    # Find common date range
    common_dates = None
    for df in factors.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No common dates across factors")

    result = {}
    for name, df in factors.items():
        result[name] = df.reindex(common_dates)
    return result


def _weighted_sum(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Compute weighted sum of factor values."""
    result = None
    for name, df in factors.items():
        w = weights.get(name, 0.0)
        if abs(w) < 1e-10:
            continue
        if result is None:
            result = df * w
        else:
            result = result + df * w

    if result is None:
        raise ValueError("No valid factors to combine")

    return result
