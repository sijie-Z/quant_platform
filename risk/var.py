"""Value at Risk (VaR) and Conditional VaR (Expected Shortfall).

Three estimation methods:
- Historical: non-parametric, uses empirical quantiles
- Parametric: assumes normal distribution
- Monte Carlo: simulates from fitted distribution
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """Historical VaR: empirical quantile of historical returns.

    Args:
        returns: Daily return series.
        confidence: Confidence level (0.95 = 95% VaR).
        horizon: Holding period in days.

    Returns:
        VaR (positive value = loss).
    """
    alpha = 1 - confidence
    daily_var = -returns.quantile(alpha)
    return daily_var * np.sqrt(horizon)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """Parametric (normal) VaR.

    Assumes returns are normally distributed with mean mu and std sigma.
    VaR = -(mu + sigma * z_alpha) * sqrt(horizon)

    Simple but underestimates tail risk if returns have fat tails.
    """
    mu = returns.mean()
    sigma = returns.std()
    z_alpha = stats.norm.ppf(1 - confidence)
    daily_var = -(mu + sigma * z_alpha)
    return daily_var * np.sqrt(horizon)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 100_000,
) -> float:
    """Monte Carlo VaR.

    Fits a t-distribution (fatter tails than normal) to returns,
    then simulates n_simulations paths.
    """
    # Fit t-distribution (better captures fat tails)
    params = stats.t.fit(returns.dropna())
    simulated = stats.t.rvs(*params, size=n_simulations)

    alpha = 1 - confidence
    daily_var = -np.percentile(simulated, alpha * 100)
    return daily_var * np.sqrt(horizon)


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """Conditional VaR (Expected Shortfall): mean loss beyond VaR.

    CVaR captures tail risk better than VaR because it accounts for
    the magnitude of extreme losses, not just the cutoff point.
    """
    alpha = 1 - confidence
    var_threshold = returns.quantile(alpha)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return 0.0
    daily_cvar = -tail_losses.mean()
    return daily_cvar * np.sqrt(horizon)


def var_summary(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> dict:
    """Compute all VaR/CVaR measures and return a summary dict."""
    return {
        "historical_var": historical_var(returns, confidence, horizon),
        "parametric_var": parametric_var(returns, confidence, horizon),
        "monte_carlo_var": monte_carlo_var(returns, confidence, horizon),
        "historical_cvar": historical_cvar(returns, confidence, horizon),
        "confidence": confidence,
        "horizon_days": horizon,
    }
