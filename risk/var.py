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
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Monte Carlo VaR.

    Fits a t-distribution (fatter tails than normal) to returns,
    then simulates n_simulations paths.

    Args:
        returns: Daily return series.
        confidence: Confidence level (0.95 = 95% VaR).
        horizon: Holding period in days.
        n_simulations: Number of simulated draws.
        random_state: Seed or ``numpy.random.Generator``. Pass a seed for
            reproducible risk reports and stable tests; ``None`` keeps the
            historical behaviour (fresh entropy on every call, which made two
            runs on identical data differ by ~2e-4).
    """
    # Fit t-distribution (better captures fat tails). The MLE is
    # deterministic, so only the simulation draw needs a controlled RNG.
    params = stats.t.fit(returns.dropna())
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    simulated = stats.t.rvs(*params, size=n_simulations, random_state=rng)

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
    random_state: int | np.random.Generator | None = None,
) -> dict:
    """Compute all VaR/CVaR measures and return a summary dict.

    ``random_state`` is forwarded to :func:`monte_carlo_var` so a whole
    summary can be reproduced from one seed.
    """
    return {
        "historical_var": historical_var(returns, confidence, horizon),
        "parametric_var": parametric_var(returns, confidence, horizon),
        "monte_carlo_var": monte_carlo_var(
            returns, confidence, horizon, random_state=random_state,
        ),
        "historical_cvar": historical_cvar(returns, confidence, horizon),
        "confidence": confidence,
        "horizon_days": horizon,
    }
