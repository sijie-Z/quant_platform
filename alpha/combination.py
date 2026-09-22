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
    # Weights are normalized over the factors that survived alignment: giving
    # a share to a factor that was dropped would scale the row down by it.
    factor_names = list(aligned)

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
    # Weights are normalized over the factors that survived alignment: giving
    # a share to a factor that was dropped would scale the row down by it.
    factor_names = list(aligned)

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

        # Filter and weight (keep sign — negative ICIR means reverse the factor)
        filtered = {name: v for name, v in icir_values.items() if abs(v) >= min_icir}
        if not filtered:
            filtered = dict(icir_values)

        total = sum(abs(v) for v in filtered.values())
        if total < 1e-10:
            weights = {name: 1.0 / len(filtered) for name in filtered}
        else:
            weights = {name: v / total for name, v in filtered.items()}

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
        if not vals.notna().any():
            # A factor with nothing to say about this cross-section used to
            # seed the sum with an all-NaN row, and NaN + x is NaN, so every
            # other factor's values for the date were lost with it. Skipping
            # it keeps the row's meaning: the weighted sum of what the factors
            # actually have. Which factors were summed no longer depends on
            # the order they happen to sit in the dict.
            continue
        if row is None:
            row = vals * w
        else:
            row = row + vals * w
    return row


def _asset_universe(factors: dict[str, pd.DataFrame]) -> pd.Index:
    """The asset universe the factors agree on, by majority.

    Every factor is supposed to span the same assets. One that does not is a
    defect in that factor: taking the union of columns would let its stray
    labels into the signal, and taking the intersection would hand it the
    whole cross-section. The modal column set ignores the odd one out.
    """
    groups: dict[tuple, list[str]] = {}
    for name, df in factors.items():
        groups.setdefault(tuple(df.columns), []).append(name)

    if not groups:
        raise ValueError("No factors provided")
    # Most members wins; ties go to the widest panel, so a one-column factor
    # can never outvote a full asset panel.
    columns = max(groups, key=lambda cols: (len(groups[cols]), len(cols)))
    return pd.Index(columns)


def _align_factors(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align factors to a common date index and asset universe.

    This used to intersect dates only. pandas aligns on labels and fills the
    misses with NaN, so a factor carrying one non-asset column -- or a Series
    wrapped by `process_factor` as a column named 'factor' -- unioned its
    label into every cross-section and emptied the entire signal. A factor
    whose columns are not the universe is now aligned to it, and dropped if
    that leaves nothing, so the damage stops at that factor.
    """
    for name, df in factors.items():
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Factor '{name}' is a {type(df).__name__}, not a (date x asset) "
                "DataFrame. A per-date series has no asset axis to combine on."
            )

    universe = _asset_universe(factors)

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
        if df.columns.equals(universe):
            result[name] = df.reindex(common_dates)
            continue

        stray = [c for c in df.columns if c not in universe]
        logger.warning(
            "Factor '%s' has %d column(s) outside the %d-asset universe (%s) -- "
            "aligning it to the universe",
            name, len(stray), len(universe), stray[:3],
        )
        aligned = df.reindex(index=common_dates, columns=universe)
        if not aligned.notna().any().any():
            logger.warning(
                "Factor '%s' shares no asset with the universe -- dropping it",
                name,
            )
            continue
        result[name] = aligned

    if not result:
        raise ValueError("No factor spans the asset universe; nothing to combine")
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


def combine_vote(
    factors: dict[str, pd.DataFrame],
    long_threshold: float = 0.5,
    short_threshold: float = -0.5,
    tie_breaker: str = "equal_weight",
) -> pd.DataFrame:
    """Ensemble voting combination - each factor votes long/short/pass.

    Inspired by FinRL DRL ensemble strategy. More robust to outlier factors.
    """
    if not factors:
        raise ValueError("No factors provided")

    aligned = _align_factors(factors)
    n_factors = len(aligned)

    votes = None
    for name, df in aligned.items():
        mean = df.mean(axis=1)
        std = df.std(axis=1, ddof=0).replace(0, float("nan"))
        z = df.sub(mean, axis=0).div(std, axis=0)

        vote = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=float)
        vote[z > long_threshold] = 1.0
        vote[z < short_threshold] = -1.0

        if votes is None:
            votes = vote
        else:
            votes = votes + vote

    signal = votes / n_factors
    signal = signal.clip(-1, 1) * 0.5

    logger.info(
        "Vote combination: %d factors, thresholds=(%.2f, %.2f)",
        n_factors, long_threshold, short_threshold,
    )
    return signal
