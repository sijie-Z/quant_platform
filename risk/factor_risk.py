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
    min_history: int = 252,
) -> dict[str, pd.Series]:
    """Estimate time-varying factor betas via daily cross-sectional regression.

    For each date t, run:
        r_i(t) = sum_j(beta_j * f_j(i,t)) + epsilon_i(t)

    Args:
        returns: (date x asset) daily returns
        factors: dict of factor_name -> (date x asset) factor values
        min_history: number of leading dates skipped as a warm-up period. Each
            remaining date's betas are estimated from that date's own
            cross-section only -- this is a daily cross-sectional regression,
            NOT a pooled trailing-window regression. The parameter used to be
            called ``window`` and was documented as a rolling estimation
            window, which it never was.

    Returns:
        dict of factor_name -> Series of daily betas (time series)
    """
    if min_history < 0:
        raise ValueError(f"min_history must be >= 0, got {min_history!r}")
    factor_names = list(factors.keys())
    dates = returns.index

    # Stack factors into a 3D structure: for each date, matrix of (asset x factor)
    betas = {name: [] for name in factor_names}
    beta_dates = []

    for i in range(min_history, len(dates)):
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
        - factor_risk_share: dict of factor -> Euler risk contribution share
        - idiosyncratic_risk_share: Var(alpha) + 2*Cov(TF, alpha), over Var(P)
        - r_squared: model fit quality

    Risk shares use the Euler/covariance decomposition on the sample where
    betas are available (dates before the first beta estimate are excluded):
    with c_j = beta_j * r_j and TF = sum_j c_j,

        RC_j = Cov(c_j, TF) / Var(P)

    and the idiosyncratic bucket holds Var(alpha) + 2*Cov(TF, alpha) over
    Var(P). Because Var(P) = sum_j Cov(c_j, TF) + Var(alpha) + 2*Cov(TF, alpha),
    the factor shares plus the idiosyncratic share sum to exactly 1.
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

    # Variance decomposition (Euler / covariance decomposition).
    #
    # With c_j = beta_j * r_j and TF = sum_j c_j:
    #     Var(P) = sum_j Cov(c_j, TF) + Var(alpha) + 2*Cov(TF, alpha)
    # so Cov(c_j, TF)/Var(P) are the factor contributions and the
    # idiosyncratic bucket holds Var(alpha) + 2*Cov(TF, alpha). The old code
    # used Var(c_j)/Var(P), which omitted every covariance term: on two
    # factors correlated at 0.9 the shares summed to 57% of total variance,
    # and with negatively correlated factors an individual share could exceed
    # 1. The identity above makes factor shares + idiosyncratic share sum to
    # exactly 1.
    # Betas only exist after the ``min_history`` warm-up, so the leading dates
    # have no factor contributions (they stay NaN through ``total_factor`` and
    # ``alpha``). Evaluate every variance and covariance on that common valid
    # sample: using the full portfolio_returns sample as the denominator would
    # cover dates the numerator terms cannot see, and the Euler shares would
    # not sum to 1.
    valid = portfolio_returns.notna() & total_factor.notna() & alpha.notna()
    port_valid = portfolio_returns[valid]
    tf_valid = total_factor[valid]
    alpha_valid = alpha[valid]

    total_var = port_valid.var()
    alpha_var = alpha_valid.var()

    risk_share = {}
    for name, c in contributions.items():
        cov_j = c[valid].cov(tf_valid)
        risk_share[name] = (
            float(cov_j / total_var)
            if total_var > 0 and np.isfinite(total_var) and np.isfinite(cov_j)
            else 0.0
        )

    cov_tf_alpha = tf_valid.cov(alpha_valid)
    idio_share = (
        float((alpha_var + 2.0 * cov_tf_alpha) / total_var)
        if total_var > 0
        and np.isfinite(total_var)
        and np.isfinite(cov_tf_alpha)
        else 0.0
    )

    r_squared = (
        float(1 - alpha_var / total_var)
        if total_var > 0 and np.isfinite(total_var)
        else 0
    )

    # Annualized contributions
    annual_contributions = {}
    for name in contributions:
        annual_contributions[name] = float(contributions[name].mean() * 252)

    return {
        "factor_contributions": contributions,
        "alpha_series": alpha,
        "total_factor_return": total_factor,
        "factor_risk_share": risk_share,
        "idiosyncratic_risk_share": idio_share,
        "factor_annual_return": annual_contributions,
        "alpha_annual_return": float(alpha.mean() * 252),
        "alpha_volatility": float(alpha.std() * np.sqrt(252)),
        "r_squared": r_squared,
        # Risk levels use the same estimated-beta sample as the shares so the
        # reported factor/idiosyncratic risks are mutually consistent.
        "total_risk": float(port_valid.std() * np.sqrt(252)),
        "factor_risk": float(tf_valid.std() * np.sqrt(252)),
        "idiosyncratic_risk": float(alpha_valid.std() * np.sqrt(252)),
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
        # Euler decomposition: this bucket includes Var(alpha) and the
        # 2*Cov(total_factor, alpha) cross term, so factor shares and this
        # row sum to 100%. Fall back to 1 - r_squared for callers holding older
        # decomposition dicts.
        "risk_share_pct": round(
            decomposition.get(
                "idiosyncratic_risk_share",
                1 - decomposition["r_squared"],
            ) * 100,
            2,
        ),
        "annual_return_bps": round(decomposition["alpha_annual_return"] * 10000, 1),
        "contribution": "alpha",
    })

    rows.sort(key=lambda x: x["risk_share_pct"], reverse=True)
    return rows
