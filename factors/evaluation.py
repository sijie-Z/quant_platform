"""Factor evaluation: IC analysis, quantile returns, correlation, turnover.

These metrics help assess factor quality before incorporating factors
into an alpha model. A good factor has:
- High |IC| (information coefficient) with future returns
- Monotonic quantile returns (top quintile > bottom quintile)
- Low correlation with other factors (diversification)
- Reasonable turnover (not too unstable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    period: int = 1,
) -> pd.Series:
    """Compute Rank IC (Spearman correlation) between factor and forward returns.

    Uses Numba JIT acceleration when available (3-8x speedup).

    Args:
        factor: (date x asset) factor values.
        forward_returns: (date x asset) forward 1-day returns (return[t] = r_{t→t+1}).
        period: Return horizon (1 = next day, 5 = next week, 21 = next month).

    Returns:
        Series of cross-sectional Rank ICs per date.
    """
    from quant_platform.utils.numba_accelerator import HAS_NUMBA, rank_ic_numba

    # forward_returns[t] is already r_{t→t+1} (shifted in pipeline).
    # Build target: for period=N, compute cumulative return t→t+N.
    if period == 1:
        target = forward_returns
    else:
        # Cumulative forward return over `period` days: (1+r₁)(1+r₂)...(1+r_N) - 1
        target = (1 + forward_returns).rolling(period).apply(
            lambda x: x.prod() - 1, raw=True
        ).shift(-(period - 1))

    if HAS_NUMBA:
        common_dates = factor.index.intersection(target.index)
        f_aligned = factor.loc[common_dates]
        r_aligned = target.loc[common_dates]
        return rank_ic_numba(f_aligned, r_aligned)

    # Pure Pandas fallback
    ic_series = []
    for date in factor.index.intersection(target.index):
        f = factor.loc[date]
        r = target.loc[date]
        common = f.dropna().index.intersection(r.dropna().index)
        if len(common) < 30:
            continue
        ic = f[common].rank().corr(r[common].rank(), method="pearson")
        ic_series.append((date, ic))

    if not ic_series:
        return pd.Series(dtype=float)

    dates, values = zip(*ic_series)
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="rank_ic")


def pearson_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    period: int = 1,
) -> pd.Series:
    """Compute Pearson IC between factor and forward returns."""
    # forward_returns[t] is already r_{t→t+1} (shifted in pipeline).
    if period == 1:
        target = forward_returns
    else:
        target = (1 + forward_returns).rolling(period).apply(
            lambda x: x.prod() - 1, raw=True
        ).shift(-(period - 1))

    ic_series = []
    for date in factor.index.intersection(target.index):
        f = factor.loc[date]
        r = target.loc[date]
        common = f.dropna().index.intersection(r.dropna().index)
        if len(common) < 30:
            continue
        ic = f[common].corr(r[common], method="pearson")
        ic_series.append((date, ic))

    if not ic_series:
        return pd.Series(dtype=float)

    dates, values = zip(*ic_series)
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="pearson_ic")


def ic_summary(ic_series: pd.Series) -> dict:
    """Compute summary statistics for an IC series.

    Returns dict with:
        mean_ic: Average IC
        std_ic: Standard deviation of IC
        icir: Information Coefficient IR = mean / std
        ic_positive_ratio: Proportion of periods with positive IC
    """
    valid = ic_series.dropna()
    if len(valid) < 10:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan, "ic_positive_ratio": np.nan}

    mean_ic = valid.mean()
    std_ic = valid.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    positive_ratio = (valid > 0).mean()

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "ic_positive_ratio": positive_ratio,
    }


def quantile_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
    period: int = 1,
) -> pd.DataFrame:
    """Compute forward returns by factor quantile group.

    Stocks are sorted into n_quantiles based on factor value each period.
    Returns the mean forward return for each quantile group.

    A good factor shows monotonic returns across quantiles:
    Q1 (top) > Q2 > Q3 > Q4 > Q5 (bottom).

    Args:
        factor: (date x asset) factor values.
        forward_returns: (date x asset) forward period returns.
        n_quantiles: Number of quantile groups (default: 5).
        period: Forward return horizon in days.

    Returns:
        DataFrame with columns for each quantile's mean return per date.
    """
    quantile_rets = {f"Q{i+1}": [] for i in range(n_quantiles)}
    dates_list = []

    # forward_returns[t] is already r_{t→t+1} (shifted in pipeline).
    if period == 1:
        target = forward_returns
    else:
        target = (1 + forward_returns).rolling(period).apply(
            lambda x: x.prod() - 1, raw=True
        ).shift(-(period - 1))

    for date in factor.index.intersection(target.index):
        f = factor.loc[date].dropna()
        r = target.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_quantiles * 5:
            continue

        # Assign quantile labels (Q1 = highest factor, Q5 = lowest)
        labels = pd.qcut(f[common], n_quantiles, labels=False, duplicates="drop")
        if labels.nunique() < n_quantiles:
            continue

        dates_list.append(date)
        for qi in range(n_quantiles):
            q_assets = common[labels == qi]
            q_ret = r[q_assets].mean()
            quantile_rets[f"Q{qi+1}"].append(q_ret)

    if not dates_list:
        return pd.DataFrame()

    return pd.DataFrame(quantile_rets, index=pd.DatetimeIndex(dates_list))


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute cross-sectional correlation matrix between factors.

    For each date, computes pairwise correlations between factors,
    then averages across dates for a stable estimate.
    """
    factor_names = list(factors.keys())
    if len(factor_names) < 2:
        return pd.DataFrame([[1.0]], index=factor_names, columns=factor_names)

    # Stack all correlation matrices
    corr_stack = []
    common_dates = factors[factor_names[0]].index
    for name in factor_names[1:]:
        common_dates = common_dates.intersection(factors[name].index)

    for date in common_dates:
        data = {}
        for name in factor_names:
            row = factors[name].loc[date].dropna()
            data[name] = row

        if not data:
            continue

        df = pd.DataFrame(data)
        df = df.dropna()
        if len(df) < 30:
            continue

        corr_stack.append(df.corr().values)

    if not corr_stack:
        return pd.DataFrame(index=factor_names, columns=factor_names)

    avg_corr = np.mean(corr_stack, axis=0)
    return pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)


def factor_turnover(factor: pd.DataFrame) -> pd.Series:
    """Compute factor turnover: proportion of assets changing quantile group.

    High turnover (>50%) indicates the factor is unstable, leading to
    high trading costs. Measured as 1 - rank correlation between
    consecutive periods' factor ranks.
    """
    ranks = factor.rank(axis=1, pct=True)

    turnover_series = []
    dates = ranks.index.sort_values()

    for i in range(1, len(dates)):
        prev = ranks.loc[dates[i - 1]]
        curr = ranks.loc[dates[i]]
        common = prev.dropna().index.intersection(curr.dropna().index)
        if len(common) < 30:
            continue
        rho = prev[common].corr(curr[common], method="spearman")
        turnover_series.append((dates[i], 1 - rho))

    if not turnover_series:
        return pd.Series(dtype=float)

    dates_vals, vals = zip(*turnover_series)
    return pd.Series(vals, index=pd.DatetimeIndex(dates_vals), name="turnover")


def ic_decay(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    max_periods: int = 20,
) -> pd.Series:
    """Compute IC decay: how IC changes with increasing forecast horizon.

    Plots mean IC against forward periods 1..max_periods.
    A good factor's IC decays slowly.
    """
    decay = {}
    for period in range(1, max_periods + 1):
        ic = rank_ic(factor, forward_returns, period=period)
        decay[period] = ic.mean() if len(ic) > 0 else np.nan

    return pd.Series(decay, name="ic_decay")
