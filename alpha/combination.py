"""Factor combination methods for alpha signal generation.

Takes multiple processed factor values and combines them into a single
alpha signal. Methods from simple to sophisticated:
- Equal weight: average of all factors
- IC-weighted: weight by historical Rank IC (point-in-time, no look-ahead)
- ICIR-weighted: weight by IC / IC_std (point-in-time, no look-ahead)

Key design: IC weights are computed point-in-time — at each date, only
data before that date is used. This prevents look-ahead bias and makes
Walk-Forward validation meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.factors.ic_monitor import FactorICAutoDecay
from quant_platform.utils.logging import get_logger

try:
    from quant_platform.factors.factor_timing import RegimeBasedTimer
except ImportError:
    RegimeBasedTimer = None  # type: ignore[assignment,misc]

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
    ic_decay: FactorICAutoDecay | None = None,
    regime_timer: RegimeBasedTimer | None = None,
    current_regime: str = "normal",
) -> pd.DataFrame:
    """Weight factors by mean Rank IC — point-in-time, no look-ahead.

    At each date, only IC observations before that date are used to
    compute weights. This means early-period signals are based on
    less history, but the signal is strictly causal.

    Factors with stronger recent predictive power get higher weight.
    If ic_decay is provided, factors with persistently low IC are
    automatically zeroed out and weights are renormalized.
    """
    if not factors:
        raise ValueError("No factors provided")

    aligned = _align_factors(factors)
    dates = sorted(next(iter(aligned.values())).index)
    factor_names = list(factors.keys())

    # Precompute Rank IC series for each factor (one pass)
    ic_series_dict = {}
    for name in factor_names:
        ic_series_dict[name] = rank_ic(factors[name], forward_returns)

    result_rows = []
    for i, date in enumerate(dates):
        # Point-in-time: only use IC data before this date
        weights = {}
        for name in factor_names:
            ic_s = ic_series_dict[name]
            ic_hist = ic_s[ic_s.index < date]
            if len(ic_hist) < 20:
                weights[name] = 0.0
                continue
            recent = ic_hist.iloc[-lookback:] if len(ic_hist) > lookback else ic_hist
            mean_ic = recent.mean()
            weights[name] = mean_ic

            # Update auto-decay monitor
            if ic_decay is not None:
                ic_decay.update(name, mean_ic)
                ic_decay.check_and_update(name)

        total_abs = sum(abs(v) for v in weights.values())
        if total_abs < 1e-10:
            weights = {name: 1.0 / len(factor_names) for name in factor_names}
        else:
            weights = {name: v / total_abs for name, v in weights.items()}

        # Apply auto-decay: zero out disabled factors and renormalize
        if ic_decay is not None:
            weights = ic_decay.get_active_weights(weights)

        # Apply regime-based factor timing
        if regime_timer is not None:
            weights = regime_timer.get_regime_weights(weights, current_regime)
            weights = regime_timer.smooth_transition(weights)

        row = _build_row(aligned, weights, date)
        if row is not None:
            row.name = date
            result_rows.append(row)

    if not result_rows:
        raise ValueError("No dates with sufficient data for IC weighting")
    return pd.DataFrame(result_rows)


def combine_icir_weighted(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    lookback: int = 252,
    min_icir: float = 0.0,
    ic_decay: FactorICAutoDecay | None = None,
    regime_timer: RegimeBasedTimer | None = None,
    current_regime: str = "normal",
) -> pd.DataFrame:
    """Weight factors by ICIR — point-in-time, no look-ahead.

    ICIR = mean(IC) / std(IC) measures risk-adjusted predictive power.
    Factors with ICIR below min_icir are excluded.

    At each date, only IC history before that date is used.
    If ic_decay is provided, factors with persistently low IC are
    automatically zeroed out and weights are renormalized.
    """
    if not factors:
        raise ValueError("No factors provided")

    aligned = _align_factors(factors)
    dates = sorted(next(iter(aligned.values())).index)
    factor_names = list(factors.keys())

    # Precompute Rank IC series for each factor (one pass)
    ic_series_dict = {}
    for name in factor_names:
        ic_series_dict[name] = rank_ic(factors[name], forward_returns)

    result_rows = []
    for i, date in enumerate(dates):
        # Point-in-time: only use IC data before this date
        icir_values = {}
        for name in factor_names:
            ic_s = ic_series_dict[name]
            ic_hist = ic_s[ic_s.index < date]
            if len(ic_hist) < 20:
                icir_values[name] = 0.0
                continue
            recent = ic_hist.iloc[-lookback:] if len(ic_hist) > lookback else ic_hist
            summary = ic_summary(recent)
            icir_values[name] = summary["icir"]

            # Update auto-decay monitor with mean IC from this window
            if ic_decay is not None:
                ic_decay.update(name, recent.mean())
                ic_decay.check_and_update(name)

        # Filter and weight
        filtered = {name: v for name, v in icir_values.items() if v >= min_icir}
        if not filtered:
            filtered = {name: max(v, 0.01) for name, v in icir_values.items()}

        total = sum(max(v, 0) for v in filtered.values())
        if total < 1e-10:
            weights = {name: 1.0 / len(filtered) for name in filtered}
        else:
            weights = {name: max(v, 0) / total for name, v in filtered.items()}

        # Apply auto-decay: zero out disabled factors and renormalize
        if ic_decay is not None:
            weights = ic_decay.get_active_weights(weights)

        # Apply regime-based factor timing
        if regime_timer is not None:
            weights = regime_timer.get_regime_weights(weights, current_regime)
            weights = regime_timer.smooth_transition(weights)

        row = _build_row(aligned, weights, date)
        if row is not None:
            row.name = date
            result_rows.append(row)

    if not result_rows:
        raise ValueError("No dates with sufficient data for ICIR weighting")
    return pd.DataFrame(result_rows)


def _build_row(
    aligned: dict[str, pd.DataFrame],
    weights: dict[str, float],
    date: pd.Timestamp,
) -> pd.Series | None:
    """Build a single row of weighted factor values for one date."""
    row = None
    for name, factor in aligned.items():
        w = weights.get(name, 0.0)
        if abs(w) < 1e-10:
            continue
        if date not in factor.index:
            continue
        vals = factor.loc[date]
        if row is None:
            row = vals * w
        else:
            row = row + vals * w
    return row


def _align_factors(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align factor dates and assets."""
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
    """Compute weighted sum of factor values (static weights, all dates)."""
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
