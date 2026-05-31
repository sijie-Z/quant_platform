"""Time-series expression functions for the expression factor engine.

Each function takes a DataProxy as the first argument and returns a DataProxy.
Operations are applied per-asset (grouped by column in the date×asset matrix).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from quant_platform.factors.expression_engine import DataProxy


def ts_delay(feature: DataProxy, periods: int) -> DataProxy:
    """Shift data forward by 'periods' (per asset)."""
    return DataProxy(feature.df.shift(periods))


def ts_delta(feature: DataProxy, periods: int) -> DataProxy:
    """Difference between current value and value 'periods' ago."""
    return feature - ts_delay(feature, periods)


def ts_mean(feature: DataProxy, window: int) -> DataProxy:
    """Rolling mean over window (per asset)."""
    return DataProxy(feature.df.T.rolling(window, axis=1).mean().T)


def ts_sum(feature: DataProxy, window: int) -> DataProxy:
    """Rolling sum over window (per asset)."""
    return DataProxy(feature.df.T.rolling(window, axis=1).sum().T)


def ts_std(feature: DataProxy, window: int) -> DataProxy:
    """Rolling standard deviation over window (per asset)."""
    return DataProxy(feature.df.T.rolling(window, axis=1).std(ddof=0).T)


def ts_min(feature: DataProxy, window: int) -> DataProxy:
    """Rolling minimum over window (per asset)."""
    return DataProxy(feature.df.T.rolling(window, axis=1).min().T)


def ts_max(feature: DataProxy, window: int) -> DataProxy:
    """Rolling maximum over window (per asset)."""
    return DataProxy(feature.df.T.rolling(window, axis=1).max().T)


def ts_rank(feature: DataProxy, window: int) -> DataProxy:
    """Percentile rank of current value within rolling window."""
    def _rank(s: np.ndarray) -> float:
        return float(scipy_stats.percentileofscore(s, s[-1]) / 100)

    result = feature.df.T.rolling(window, axis=1).apply(
        _rank, raw=True
    ).T
    return DataProxy(result)


def ts_quantile(feature: DataProxy, window: int, q: float) -> DataProxy:
    """Rolling quantile over window."""
    return DataProxy(feature.df.T.rolling(window, axis=1).quantile(q).T)


def ts_argmax(feature: DataProxy, window: int) -> DataProxy:
    """Index of the maximum value within rolling window."""
    def _argmax(s: np.ndarray) -> float:
        return float(np.argmax(s) + 1)

    result = feature.df.T.rolling(window, axis=1).apply(
        _argmax, raw=True
    ).T
    return DataProxy(result)


def ts_argmin(feature: DataProxy, window: int) -> DataProxy:
    """Index of the minimum value within rolling window."""
    def _argmin(s: np.ndarray) -> float:
        return float(np.argmin(s) + 1)

    result = feature.df.T.rolling(window, axis=1).apply(
        _argmin, raw=True
    ).T
    return DataProxy(result)


def ts_corr(x: DataProxy, y: DataProxy, window: int) -> DataProxy:
    """Rolling correlation between x and y over window (per asset)."""
    result = x.df.T.rolling(window, axis=1).corr(y.df.T).T
    return DataProxy(result)


def ts_cov(x: DataProxy, y: DataProxy, window: int) -> DataProxy:
    """Rolling covariance between x and y over window (per asset)."""
    result = x.df.T.rolling(window, axis=1).cov(y.df.T).T
    return DataProxy(result)


def ts_slope(feature: DataProxy, window: int) -> DataProxy:
    """Linear regression slope over rolling window (per asset)."""
    x_vals = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        if np.any(~np.isfinite(y)):
            return np.nan
        x = x_vals[~np.isnan(y)] if len(y) > 2 else x_vals
        y_clean = y[~np.isnan(y)] if len(y) > 2 else y
        if len(y_clean) < 2:
            return np.nan
        with np.errstate(invalid="ignore"):
            return float(np.polyfit(x[:len(y_clean)], y_clean, 1)[0])

    result = feature.df.T.rolling(window, axis=1).apply(
        _slope, raw=True
    ).T
    return DataProxy(result)


def ts_rsquare(feature: DataProxy, window: int) -> DataProxy:
    """R-squared of linear regression over rolling window (per asset)."""
    x_vals = np.arange(window, dtype=float)

    def _rsq(y: np.ndarray) -> float:
        if np.any(~np.isfinite(y)):
            return np.nan
        x = x_vals[~np.isnan(y)]
        y_clean = y[~np.isnan(y)]
        if len(y_clean) < 2:
            return np.nan
        with np.errstate(invalid="ignore"):
            slope, intercept = np.polyfit(x, y_clean, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((y_clean - pred) ** 2))
            ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    result = feature.df.T.rolling(window, axis=1).apply(
        _rsq, raw=True
    ).T
    return DataProxy(result)


def ts_product(feature: DataProxy, window: int) -> DataProxy:
    """Rolling product over window (per asset)."""
    result = feature.df.T.rolling(window, axis=1).apply(
        lambda x: np.prod(x) if len(x) == window else np.nan,
        raw=True,
    ).T
    return DataProxy(result)


def ts_decay_linear(feature: DataProxy, window: int) -> DataProxy:
    """Linearly weighted moving average: more weight to recent values."""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _decay(s: np.ndarray) -> float:
        if len(s) < window or np.any(~np.isfinite(s)):
            return np.nan
        return float(np.sum(s * weights))

    result = feature.df.T.rolling(window, axis=1).apply(
        _decay, raw=True
    ).T
    return DataProxy(result)


# List of all time-series functions for registration
TS_FUNCTIONS = [
    ts_delay, ts_delta,
    ts_mean, ts_sum, ts_std,
    ts_min, ts_max,
    ts_rank, ts_quantile,
    ts_argmax, ts_argmin,
    ts_corr, ts_cov,
    ts_slope, ts_rsquare,
    ts_product, ts_decay_linear,
]
