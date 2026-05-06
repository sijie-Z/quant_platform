"""Numba JIT-accelerated computation kernels for quant platform.

Provides LLVM-compiled (C-speed) versions of key compute-intensive functions:
- Rolling cumulative returns (momentum factor)
- Max drawdown calculation
- Cross-sectional winsorization
- Rank IC computation
- Covariance shrinkage

Each function has a pure-Pandas and a Numba-JIT version with timing
comparison logged at DEBUG level.

Performance expectation: 5-20x speedup on compute-bound loops.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Check if numba is available
try:
    from numba import jit, prange
    HAS_NUMBA = True
    logger.info("Numba JIT available - accelerated kernels enabled")
except ImportError:
    HAS_NUMBA = False
    # Define no-op jit for graceful fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    logger.warning("Numba not installed - using pure Python/Pandas. "
                   "Install with: pip install numba")


def _time_it(func, *args, **kwargs) -> tuple:
    """Time a function call, return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ======================================================================
# 1. Rolling cumulative return (momentum factor core)
# ======================================================================

@jit(nopython=True, parallel=True, cache=True)
def _rolling_cumret_numba(returns: np.ndarray, period: int) -> np.ndarray:
    """Numba JIT: rolling cumulative product of (1+r) - 1.

    Args:
        returns: (n_dates, n_assets) array of daily returns.
        period: Lookback window size.

    Returns:
        (n_dates, n_assets) momentum factor values.
    """
    n_dates, n_assets = returns.shape
    result = np.full((n_dates, n_assets), np.nan)

    for j in prange(n_assets):
        for t in range(period, n_dates):
            # Check if any NaN in window
            window = returns[t - period:t, j]
            valid = True
            cumret = 1.0
            for k in range(period):
                if np.isnan(window[k]):
                    valid = False
                    break
                cumret *= (1.0 + window[k])
            if valid:
                result[t, j] = cumret - 1.0

    return result


def momentum_factor_pandas(returns: pd.DataFrame, period: int) -> pd.DataFrame:
    """Pandas version: rolling cumulative return."""
    return returns.rolling(period).apply(lambda x: (1 + x).prod() - 1)


def momentum_factor_numba(returns: pd.DataFrame, period: int) -> pd.DataFrame:
    """Numba-accelerated version with auto-fallback."""
    if HAS_NUMBA and returns.shape[1] >= 10:
        arr = returns.values.astype(np.float64)
        result = _rolling_cumret_numba(arr, period)
        return pd.DataFrame(result, index=returns.index, columns=returns.columns)
    return momentum_factor_pandas(returns, period)


# ======================================================================
# 2. Max drawdown (vectorized, numba-optimized)
# ======================================================================

@jit(nopython=True, cache=True)
def _max_drawdown_numba(cumulative: np.ndarray) -> tuple:
    """Numba JIT: max drawdown from cumulative return series.

    Returns (max_dd, peak_idx, trough_idx).
    """
    n = len(cumulative)
    running_max = cumulative[0]
    max_dd = 0.0
    peak_idx = 0
    trough_idx = 0

    for i in range(n):
        if cumulative[i] > running_max:
            running_max = cumulative[i]
        dd = (cumulative[i] - running_max) / running_max
        if dd < max_dd:
            max_dd = dd
            trough_idx = i
            # Find the peak corresponding to this trough
            peak_val = cumulative[0]
            for j in range(i + 1):
                if cumulative[j] > peak_val:
                    peak_val = cumulative[j]
                    peak_idx = j

    return max_dd, peak_idx, trough_idx


def max_drawdown_pandas(returns: pd.Series) -> tuple:
    """Pandas version."""
    cumulative = (1 + returns).cumprod().values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    trough_idx = drawdown.argmin()
    peak_idx = running_max[:trough_idx + 1].argmax()
    return float(max_dd), peak_idx, trough_idx


def max_drawdown_numba(returns: pd.Series) -> tuple:
    """Numba-accelerated version."""
    if HAS_NUMBA:
        cumulative = (1 + returns.dropna()).cumprod().values
        return _max_drawdown_numba(cumulative.astype(np.float64))
    return max_drawdown_pandas(returns)


# ======================================================================
# 3. Cross-sectional winsorization (numba)
# ======================================================================

@jit(nopython=True, parallel=True, cache=True)
def _winsorize_numba(data: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Numba JIT: cross-sectional winsorization.

    For each row (date), clips values at lower/upper quantiles.
    """
    n_rows, n_cols = data.shape
    result = data.copy()

    for i in prange(n_rows):
        row = result[i]
        # Find valid (non-NaN) values
        valid = row[~np.isnan(row)]
        if len(valid) < 10:
            continue

        # Compute quantiles with linear interpolation (matching pandas behavior)
        sorted_valid = np.sort(valid)
        n = len(sorted_valid)
        lo_idx = (n - 1) * lower
        lo_lo = sorted_valid[int(lo_idx)]
        lo_hi = sorted_valid[min(int(lo_idx) + 1, n - 1)]
        lo = lo_lo + (lo_idx - int(lo_idx)) * (lo_hi - lo_lo)

        hi_idx = (n - 1) * upper
        hi_lo = sorted_valid[int(hi_idx)]
        hi_hi = sorted_valid[min(int(hi_idx) + 1, n - 1)]
        hi = hi_lo + (hi_idx - int(hi_idx)) * (hi_hi - hi_lo)

        # Clip
        for j in range(n_cols):
            if not np.isnan(row[j]):
                if row[j] < lo:
                    row[j] = lo
                elif row[j] > hi:
                    row[j] = hi

    return result


def winsorize_pandas(factor: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Pandas version: winsorize per date."""
    result = factor.copy()
    for date in factor.index:
        row = result.loc[date]
        valid = row.dropna()
        if len(valid) < 10:
            continue
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        result.loc[date] = row.clip(lower=lo, upper=hi)
    return result


def winsorize_numba(factor: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Numba-accelerated winsorization."""
    if HAS_NUMBA and factor.shape[1] >= 10:
        arr = factor.values.astype(np.float64)
        result = _winsorize_numba(arr, lower, upper)
        return pd.DataFrame(result, index=factor.index, columns=factor.columns)
    return winsorize_pandas(factor, lower, upper)


# ======================================================================
# 4. Rank IC computation (numba-optimized)
# ======================================================================

@jit(nopython=True, cache=True)
def _spearman_rank_ic(factor_row: np.ndarray, ret_row: np.ndarray) -> float:
    """Numba JIT: Spearman rank correlation for one cross-section."""
    # Find valid pairs
    mask = ~(np.isnan(factor_row) | np.isnan(ret_row))
    n_valid = mask.sum()
    if n_valid < 30:
        return np.nan

    f = factor_row[mask]
    r = ret_row[mask]

    # Rank (use argsort twice for ranking)
    f_rank = np.zeros(n_valid)
    r_rank = np.zeros(n_valid)

    f_order = np.argsort(f)
    r_order = np.argsort(r)

    for i in range(n_valid):
        f_rank[f_order[i]] = i + 1
        r_rank[r_order[i]] = i + 1

    # Pearson correlation of ranks
    f_mean = f_rank.mean()
    r_mean = r_rank.mean()
    f_std = f_rank.std()
    r_std = r_rank.std()

    if f_std < 1e-10 or r_std < 1e-10:
        return np.nan

    cov = ((f_rank - f_mean) * (r_rank - r_mean)).mean()
    return cov / (f_std * r_std)


def rank_ic_numba(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Numba-accelerated Rank IC computation."""
    if not HAS_NUMBA:
        from quant_platform.factors.evaluation import rank_ic
        return rank_ic(factor, forward_returns)

    f_arr = factor.values.astype(np.float64)
    r_arr = forward_returns.values.astype(np.float64)
    n_rows = min(f_arr.shape[0], r_arr.shape[0])

    results = []
    dates = []
    for i in range(n_rows):
        ic = _spearman_rank_ic(f_arr[i], r_arr[i])
        if not np.isnan(ic):
            results.append(ic)
            dates.append(factor.index[i])

    return pd.Series(results, index=pd.DatetimeIndex(dates), name="rank_ic")


# ======================================================================
# 5. Covariance shrinkage (numba-optimized Ledoit-Wolf)
# ======================================================================

@jit(nopython=True, cache=True)
def _ledoit_wolf_shrinkage_numba(returns: np.ndarray) -> np.ndarray:
    """Numba JIT: Ledoit-Wolf covariance shrinkage.

    Shrinks sample covariance toward a structured target (constant correlation).
    This is the simplest Ledoit-Wolf variant - the full version requires
    solving a more complex optimization.
    """
    n, p = returns.shape

    # Sample covariance
    mean_ret = returns.mean(axis=0)
    demeaned = returns - mean_ret
    sample_cov = (demeaned.T @ demeaned) / (n - 1)

    # Target: constant correlation
    stds = np.sqrt(np.diag(sample_cov))
    avg_corr = 0.0
    count = 0
    for i in range(p):
        for j in range(i + 1, p):
            if stds[i] > 1e-10 and stds[j] > 1e-10:
                corr = sample_cov[i, j] / (stds[i] * stds[j])
                avg_corr += corr
                count += 1

    if count > 0:
        avg_corr = avg_corr / count
    else:
        avg_corr = 0.0

    # Build target matrix
    target = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                target[i, j] = stds[i] ** 2
            else:
                target[i, j] = avg_corr * stds[i] * stds[j]

    # Simple shrinkage intensity (can be refined)
    shrinkage = 0.2  # Fixed rate for simplicity

    return (1 - shrinkage) * sample_cov + shrinkage * target


def covariance_numba(returns: pd.DataFrame) -> pd.DataFrame:
    """Numba-accelerated Ledoit-Wolf covariance."""
    if HAS_NUMBA and returns.shape[1] >= 2:
        arr = returns.dropna(axis=1).values.astype(np.float64)
        cov = _ledoit_wolf_shrinkage_numba(arr)
        cols = returns.dropna(axis=1).columns
        return pd.DataFrame(cov, index=cols, columns=cols)

    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    clean = returns.dropna(axis=1).values
    cov = lw.fit(clean).covariance_
    cols = returns.dropna(axis=1).columns
    return pd.DataFrame(cov, index=cols, columns=cols)


# ======================================================================
# 6. Cross-sectional zscore standardization (numba)
# ======================================================================

@jit(nopython=True, parallel=True, cache=True)
def _zscore_numba(data: np.ndarray) -> np.ndarray:
    """Numba JIT: cross-sectional zscore standardization.

    For each row (date), computes (x - mean) / std.
    """
    n_rows, n_cols = data.shape
    result = data.copy()

    for i in prange(n_rows):
        row = result[i]
        valid = row[~np.isnan(row)]
        if len(valid) < 10:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-10:
            for j in range(n_cols):
                if not np.isnan(row[j]):
                    row[j] = 0.0
        else:
            for j in range(n_cols):
                if not np.isnan(row[j]):
                    row[j] = (row[j] - mu) / sigma

    return result


def zscore_numba(factor: pd.DataFrame) -> pd.DataFrame:
    """Numba-accelerated zscore standardization with auto-fallback."""
    if HAS_NUMBA and factor.shape[1] >= 10:
        arr = factor.values.astype(np.float64)
        result = _zscore_numba(arr)
        return pd.DataFrame(result, index=factor.index, columns=factor.columns)

    # Pandas fallback
    result = factor.copy()
    for date in factor.index:
        row = result.loc[date]
        valid = row.dropna()
        if len(valid) < 10:
            continue
        mu = valid.mean()
        sigma = valid.std()
        if sigma < 1e-10:
            result.loc[date] = 0.0
        else:
            result.loc[date] = (row - mu) / sigma
    return result


# ======================================================================
# Benchmark helper
# ======================================================================

def benchmark(func_pandas, func_numba, *args, name: str = "", **kwargs) -> dict:
    """Compare Pandas vs Numba performance.

    Returns dict with timing and speedup info.
    """
    _, pandas_time = _time_it(func_pandas, *args, **kwargs)
    _, numba_time = _time_it(func_numba, *args, **kwargs)

    speedup = pandas_time / numba_time if numba_time > 0 else float("inf")

    logger.info(
        "BENCHMARK [%s]: Pandas=%.4fs  Numba=%.4fs  Speedup=%.1fx",
        name, pandas_time, numba_time, speedup,
    )

    return {
        "name": name,
        "pandas_seconds": pandas_time,
        "numba_seconds": numba_time,
        "speedup": speedup,
    }
