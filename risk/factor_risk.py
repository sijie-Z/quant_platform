"""Factor-based risk decomposition.

Decomposes portfolio returns into systematic (factor) and idiosyncratic (alpha) components.
Uses cross-sectional regression to estimate factor betas and attribute risk.

Key concepts:
- Factor exposure: portfolio's sensitivity to each risk factor
- Factor contribution: how much each factor contributes to total risk/return
- Idiosyncratic risk: stock-specific risk not explained by factors
- Factor-adjusted alpha: return after removing factor effects
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def estimate_factor_betas(
    returns: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    window: int = 252,
) -> dict[str, pd.Series]:
    """Estimate time-varying factor betas via rolling cross-sectional regression.

    For each date t, run:
        r_i(t) = sum_j(beta_j * f_j(i,t)) + epsilon_i(t)

    Args:
        returns: (date x asset) daily returns
        factors: dict of factor_name -> (date x asset) factor values
        window: rolling window for beta estimation

    Returns:
        dict of factor_name -> Series of daily betas (time series)
    """
    factor_names = list(factors.keys())
    dates = returns.index

    # Stack factors into a 3D structure: for each date, matrix of (asset x factor)
    betas = {name: [] for name in factor_names}
    beta_dates = []

    for i in range(window, len(dates)):
        date = dates[i]
        # Get cross-sectional returns for this date
        r = returns.loc[date].dropna()

        if len(r) < 30:
            continue

        # Build factor matrix for this date
        X_cols = {}
        for name in factor_names:
            f = factors[name]
            if date in f.index:
                X_cols[name] = f.loc[date].reindex(r.index)

        if len(X_cols) != len(factor_names):
            continue

        X = pd.DataFrame(X_cols).dropna()
        common = X.index.intersection(r.index)
        if len(common) < 30:
            continue

        X = X.loc[common].values
        y = r.loc[common].values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            beta_dates.append(date)
            for j, name in enumerate(factor_names):
                betas[name].append(float(beta[j + 1]))  # Skip intercept
        except np.linalg.LinAlgError:
            continue

    result = {}
    for name in factor_names:
        result[name] = pd.Series(betas[name], index=beta_dates, name=f"beta_{name}")

    return result


def factor_risk_decomposition(
    portfolio_returns: pd.Series,
    factor_betas: dict[str, pd.Series],
    factor_returns: dict[str, pd.Series],
) -> dict:
    """Decompose portfolio returns into factor contributions.

    Portfolio return ≈ alpha + sum(beta_j * r_j)

    Args:
        portfolio_returns: daily portfolio returns
        factor_betas: dict of factor_name -> time series of betas
        factor_returns: dict of factor_name -> time series of factor returns

    Returns:
        dict with:
        - factor_contributions: dict of factor -> contribution series
        - alpha_series: residual (idiosyncratic) returns
        - total_factor_return: sum of all factor contributions
        - factor_risk_share: dict of factor -> fraction of total variance
        - r_squared: model fit quality
    """
    dates = portfolio_returns.index

    # Align all series
    aligned_betas = {}
    aligned_factor_returns = {}
    for name in factor_betas:
        b = factor_betas[name].reindex(dates).ffill()
        fr = factor_returns[name].reindex(dates).fillna(0)
        aligned_betas[name] = b
        aligned_factor_returns[name] = fr

    # Compute factor contributions: beta_j * r_j
    contributions = {}
    for name in factor_betas:
        contributions[name] = aligned_betas[name] * aligned_factor_returns[name]

    total_factor = sum(contributions.values())
    alpha = portfolio_returns - total_factor

    # Variance decomposition
    total_var = portfolio_returns.var()
    factor_variances = {}
    for name in contributions:
        factor_variances[name] = contributions[name].var()

    sum(factor_variances.values())
    alpha_var = alpha.var()

    risk_share = {}
    for name in factor_variances:
        risk_share[name] = float(factor_variances[name] / total_var) if total_var > 0 else 0

    r_squared = float(1 - alpha_var / total_var) if total_var > 0 else 0

    # Annualized contributions
    annual_contributions = {}
    for name in contributions:
        annual_contributions[name] = float(contributions[name].mean() * 252)

    return {
        "factor_contributions": contributions,
        "alpha_series": alpha,
        "total_factor_return": total_factor,
        "factor_risk_share": risk_share,
        "factor_annual_return": annual_contributions,
        "alpha_annual_return": float(alpha.mean() * 252),
        "alpha_volatility": float(alpha.std() * np.sqrt(252)),
        "r_squared": r_squared,
        "total_risk": float(portfolio_returns.std() * np.sqrt(252)),
        "factor_risk": float(total_factor.std() * np.sqrt(252)),
        "idiosyncratic_risk": float(alpha.std() * np.sqrt(252)),
    }


def factor_contribution_summary(
    decomposition: dict,
) -> list[dict]:
    """Format factor risk decomposition as a list of dicts for display."""
    rows = []
    for name in decomposition["factor_risk_share"]:
        rows.append({
            "factor": name,
            "risk_share_pct": round(decomposition["factor_risk_share"][name] * 100, 2),
            "annual_return_bps": round(decomposition["factor_annual_return"][name] * 10000, 1),
            "contribution": "systematic",
        })

    rows.append({
        "factor": "Alpha (idiosyncratic)",
        "risk_share_pct": round((1 - decomposition["r_squared"]) * 100, 2),
        "annual_return_bps": round(decomposition["alpha_annual_return"] * 10000, 1),
        "contribution": "alpha",
    })

    rows.sort(key=lambda x: x["risk_share_pct"], reverse=True)
    return rows
