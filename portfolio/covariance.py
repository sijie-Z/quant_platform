"""Covariance matrix estimation for portfolio optimization.

Provides multiple estimation methods:
- Sample covariance: standard historical estimator
- Ledoit-Wolf shrinkage: improves conditioning for high-dimensional problems
- EWMA: exponentially weighted for recent emphasis

Good covariance estimates are critical for mean-variance optimization.
A poorly conditioned covariance matrix leads to extreme/unstable weights.

Missing-data convention
------------------------
A-share suspensions leave holes in the return panel. Filling them with 0 (the
old behaviour) biased variances and covariances toward zero: an asset with 20%
missing observations had a variance of only ~76% of its complete-data value,
which understated risk for exactly the names most likely to be suspended.

Every estimator here now uses *pairwise-complete* observations:

- ``sample``: pairwise sample covariance (``ddof=1``), nearest-PSD repaired by
  clipping negative eigenvalues to zero. Under missing-at-random the diagonal
  is unbiased; the repair only acts when the pairwise matrix is indefinite.
- ``ledoit_wolf``: one estimator for every universe width (no more
  sklearn scaled-identity below 5 assets / constant-correlation kernel above).
  With complete data this is exactly ``sklearn.covariance.LedoitWolf``. With
  missing data the sample covariance is pairwise-complete (``ddof=0``, the
  same divisor sklearn uses internally) and the shrinkage intensity is
  estimated on mean-imputed data, then applied to that pairwise covariance, so
  variances stay consistent with the complete-data path.
- ``ewma``: exponentially weighted covariance with pairwise-complete
  normalisation; each covariance entry divides by the decay mass of the
  observations where both assets are present.
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

    Args:
        returns: (date x asset) daily returns.
        method: 'sample', 'ledoit_wolf', or 'ewma'.
        lookback: Number of recent days for estimation.
        ewma_half_life: Half-life for EWMA decay (days).

    Returns:
        Covariance matrix (asset x asset), symmetric positive semi-definite.
    """
    window = returns.iloc[-lookback:]
    # Keep assets with at least half the window populated, matching the
    # engine's 50% filter. Requiring 100% (the old `dropna(how="any")`)
    # silently removed every stock that was suspended even once -- and A-share
    # suspensions are common -- which biased optimized portfolios toward
    # continuously traded names and could starve the optimizer into its
    # equal-weight fallback. Missing values are handled per estimator instead
    # of being treated as flat returns.
    coverage = window.notna().mean()
    keep = coverage[coverage >= 0.5].index
    recent = window[keep]

    if recent.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 assets with 50% data coverage, got {recent.shape[1]}"
        )

    if method == "sample":
        cov = _sample_cov(recent)

    elif method == "ledoit_wolf":
        cov = _ledoit_wolf_cov(recent)
        logger.debug("Ledoit-Wolf shrinkage applied, shape=%s", cov.shape)

    elif method == "ewma":
        decay = 0.5 ** (1.0 / ewma_half_life)
        cov = _ewm_cov(recent.to_numpy(dtype=float), decay)

    else:
        raise ValueError(f"Unknown covariance method: {method}")

    return pd.DataFrame(cov, index=recent.columns, columns=recent.columns)


def _sample_cov(recent: pd.DataFrame) -> np.ndarray:
    """Sample covariance; pairwise-complete and PSD-repaired if data holes."""
    if recent.isna().any().any():
        cov = _pairwise_cov(recent, ddof=1)
    else:
        # Keep the exact historical result (np.cov / pandas ddof=1) for the
        # complete-data path.
        cov = recent.cov().to_numpy(dtype=float)
    return cov


def _ledoit_wolf_cov(recent: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled-identity target for all p.

    One estimator for every universe size. The old dispatch used sklearn
    (scaled-identity target) below 5 assets and the numba constant-correlation
    kernel at 5+, so a p=4 model and a p=5 model were different estimators and
    the shrinkage jumped discontinuously at the boundary.
    """
    values = recent.to_numpy(dtype=float)
    if np.isfinite(values).all():
        return LedoitWolf().fit(values).covariance_

    # Missing data: pairwise sample covariance (ddof=0 matches the divisor
    # sklearn uses in its internal empirical covariance) shrunk toward the
    # scaled identity target with the intensity estimated on mean-imputed
    # data. Applying the intensity to the pairwise covariance -- instead of
    # using the imputed sample covariance directly -- keeps each asset's
    # variance consistent with the complete-data estimator.
    sample = _pairwise_cov(recent, ddof=0)
    imputed = recent.fillna(recent.mean()).to_numpy(dtype=float)
    imputed = np.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        delta = float(LedoitWolf().fit(imputed).shrinkage_)
    except Exception:  # degenerate constant columns -- no shrinkage evidence
        delta = 0.0
    p = sample.shape[0]
    mu = float(np.trace(sample)) / p if p else 0.0
    cov = (1.0 - delta) * sample + delta * mu * np.eye(p)
    return _project_psd(cov)


def _pairwise_cov(recent: pd.DataFrame, ddof: int) -> np.ndarray:
    """Pairwise-complete sample covariance with a PSD repair.

    pandas computes each entry from the observations where both assets are
    present, which is unbiased under missing-at-random. Such matrices are not
    guaranteed PSD, so negative eigenvalues are clipped to zero (the repair is
    a no-op whenever the matrix is already PSD). Pairs with fewer than two
    observations fall back to zero covariance.
    """
    cov = recent.cov(ddof=ddof).to_numpy(dtype=float)
    if not np.isfinite(cov).all():
        variances = recent.var(ddof=ddof).to_numpy(dtype=float)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(
            cov, np.where(np.isfinite(variances), variances, 0.0),
        )
    return _project_psd(cov)


def _project_psd(cov: np.ndarray) -> np.ndarray:
    """Symmetrise and clip negative eigenvalues to zero."""
    cov = 0.5 * (cov + cov.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return cov
    if eigvals.min() < 0.0:
        eigvals = np.maximum(eigvals, 0.0)
        cov = (eigvecs * eigvals) @ eigvecs.T
        cov = 0.5 * (cov + cov.T)
    return cov


def _ewm_cov(data: np.ndarray, decay: float) -> np.ndarray:
    """Exponentially weighted covariance, pairwise-complete for NaNs.

    With complete data this is the historical estimator: an ordinary
    weighted covariance with normalised weights. With missing data each
    covariance entry divides by the decay mass of the rows where *both*
    assets are observed, so a suspension no longer drags the entry toward
    zero.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    weights = decay ** np.arange(n - 1, -1, -1)
    observed = np.isfinite(data)
    observed_f = observed.astype(float)

    # Per-asset weighted mean over the observed rows.
    weighted_sum = (np.where(observed, data, 0.0) * weights[:, np.newaxis]).sum(axis=0)
    weight_mass = (observed_f * weights[:, np.newaxis]).sum(axis=0)
    weighted_mean = weighted_sum / np.maximum(weight_mass, 1e-300)

    demeaned = np.where(observed, data - weighted_mean, 0.0)
    weighted_demeaned = demeaned * np.sqrt(weights[:, np.newaxis])

    numerator = weighted_demeaned.T @ weighted_demeaned
    denominator = (observed_f * weights[:, np.newaxis]).T @ observed_f
    cov = np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator), where=denominator > 0,
    )
    return _project_psd(cov)
