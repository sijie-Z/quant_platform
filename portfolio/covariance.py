"""Covariance matrix estimation for portfolio optimization.

Provides multiple estimation methods:
- Sample covariance: standard historical estimator
- Ledoit-Wolf shrinkage: improves conditioning for high-dimensional problems
- EWMA: exponentially weighted for recent emphasis

Good covariance estimates are critical for mean-variance optimization.
A poorly conditioned covariance matrix leads to extreme/unstable weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    lookback: int = 252,
    ewma_half_life: int = 63,
) -> pd.DataFrame:
    """Estimate covariance matrix from returns.

    Uses Numba JIT acceleration for ledoit_wolf when available.

    Args:
        returns: (date x asset) daily returns.
        method: 'sample', 'ledoit_wolf', or 'ewma'.
        lookback: Number of recent days for estimation.
        ewma_half_life: Half-life for EWMA decay (days).

    Returns:
        Covariance matrix (asset x asset).
    """
    from quant_platform.utils.numba_accelerator import HAS_NUMBA, covariance_numba

    recent = returns.iloc[-lookback:].dropna(axis=1, how="any")

    if recent.shape[1] < 2:
        raise ValueError(f"Need at least 2 assets with full data, got {recent.shape[1]}")

    if method == "sample":
        cov = recent.cov().values

    elif method == "ledoit_wolf":
        if HAS_NUMBA and recent.shape[1] >= 5:
            return covariance_numba(recent)
        lw = LedoitWolf()
        cov = lw.fit(recent.values).covariance_
        logger.debug("Ledoit-Wolf shrinkage applied, shape=%s", cov.shape)

    elif method == "ewma":
        decay = 0.5 ** (1.0 / ewma_half_life)
        cov = _ewm_cov(recent.values, decay)

    else:
        raise ValueError(f"Unknown covariance method: {method}")

    return pd.DataFrame(cov, index=recent.columns, columns=recent.columns)


def _ewm_cov(data: np.ndarray, decay: float) -> np.ndarray:
    """Compute exponentially weighted covariance matrix."""
    n = data.shape[0]
    weights = decay ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()

    # Demean
    weighted_mean = (data * weights[:, np.newaxis]).sum(axis=0)
    demeaned = data - weighted_mean

    # Weighted covariance
    weighted_demeaned = demeaned * np.sqrt(weights[:, np.newaxis])
    return weighted_demeaned.T @ weighted_demeaned
