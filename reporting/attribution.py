"""Performance attribution analysis.

Decomposes strategy returns into factor-based exposures to understand
the sources of alpha vs. factor beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def factor_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    weights_history: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.DataFrame:
    """Regress strategy returns on factor returns for attribution.

    Args:
        strategy_returns: Daily strategy returns.
        factor_returns: (date x factor) factor returns.
        weights_history: Not used in this simplified version.

    Returns:
        DataFrame with factor betas, t-stats, and contribution.
    """
    from sklearn.linear_model import LinearRegression

    aligned = pd.concat([strategy_returns, factor_returns], axis=1).dropna()
    if len(aligned) < 20:
        return pd.DataFrame()

    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values

    model = LinearRegression()
    model.fit(X, y)

    # Standard errors
    residuals = y - model.predict(X)
    n, k = X.shape
    sigma2 = (residuals ** 2).sum() / (n - k - 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))

    t_stats = model.coef_ / se

    # Factor contributions
    factor_means = X.mean(axis=0)
    contributions = model.coef_ * factor_means

    return pd.DataFrame({
        "beta": model.coef_,
        "t_stat": t_stats,
        "factor_mean_daily": factor_means,
        "contribution_annual": contributions * 252,
        "contribution_pct": contributions / contributions.sum() * 100,
    }, index=factor_returns.columns)


def turnover_analysis(
    weights_history: dict[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    """Analyze turnover across rebalance periods.

    Turnover = sum(|w_new - w_old|) / 2
    One-sided turnover measures the proportion of portfolio replaced.
    """
    dates = sorted(weights_history.keys())
    if len(dates) < 2:
        return pd.DataFrame()

    results = []
    for i in range(1, len(dates)):
        prev = weights_history[dates[i - 1]]
        curr = weights_history[dates[i]]
        common = prev.index.intersection(curr.index)
        if len(common) < 2:
            continue
        turnover = (curr[common] - prev[common]).abs().sum() / 2
        results.append({
            "date": dates[i],
            "turnover": turnover,
            "n_assets": int((curr > 1e-6).sum()),
        })

    return pd.DataFrame(results).set_index("date")
