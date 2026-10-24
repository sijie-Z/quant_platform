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

    from quant_platform.backtest.metrics import infer_periods_per_year

    aligned = pd.concat([strategy_returns, factor_returns], axis=1).dropna()
    if len(aligned) < 20:
        return pd.DataFrame()

    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values

    model = LinearRegression()
    model.fit(X, y)

    # Standard errors. The model includes an intercept (sklearn
    # LinearRegression), so the slope covariance is
    # sigma2 * inv(Xc'Xc) on the *centered* regressors. Using inv(X'X) on the
    # raw regressors attributed the intercept's uncertainty to the slopes and
    # inflated the t-stat of any factor with a nonzero mean (a market/carry
    # factor with mean 0.0099 and sd 0.002 reported t=47 instead of t=8.5).
    residuals = y - model.predict(X)
    n, k = X.shape
    sigma2 = (residuals ** 2).sum() / (n - k - 1) if n - k - 1 > 0 else np.nan
    X_centered = X - X.mean(axis=0)
    XtX_inv = np.linalg.inv(X_centered.T @ X_centered)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    mean_x = X.mean(axis=0)
    se_intercept = np.sqrt(sigma2 * (1.0 / n + float(mean_x @ XtX_inv @ mean_x)))

    t_stats = model.coef_ / se

    # Factor contributions. Annualize with the platform's inferred
    # periods-per-year (A-share sessions are ~242, not 252) instead of a
    # hardcoded 252.
    ppy = infer_periods_per_year(aligned.iloc[:, 0])
    factor_means = X.mean(axis=0)
    contributions = model.coef_ * factor_means
    intercept = float(model.intercept_)

    # The intercept/alpha is part of the mean return; excluding it from the
    # denominator let a single factor's share exceed 100% and let two
    # cancelling factors print +755%/-655%. Near-zero totals report NaN
    # instead of exploding.
    total_daily = intercept + float(contributions.sum())
    # A total mean return that is negligible relative to the gross
    # contributions means the shares are cancellation noise (two offsets
    # cancel): a net below 10% of the gross churn gave +755%/-655% style
    # percentages. Report NaN instead of an unstable share.
    scale = abs(intercept) + float(np.abs(contributions).sum())
    if scale < 1e-12 or abs(total_daily) < 0.1 * scale:
        contribution_pct = np.full(k, np.nan)
        intercept_pct = np.nan
    else:
        contribution_pct = contributions / total_daily * 100
        intercept_pct = intercept / total_daily * 100

    frame = pd.DataFrame({
        "beta": model.coef_,
        "t_stat": t_stats,
        "factor_mean_daily": factor_means,
        "contribution_annual": contributions * ppy,
        "contribution_pct": contribution_pct,
    }, index=factor_returns.columns)
    # Explicit alpha row, so the rows add up to the strategy's mean return.
    frame.loc["intercept"] = {
        "beta": intercept,
        "t_stat": intercept / se_intercept if se_intercept > 0 else np.nan,
        "factor_mean_daily": 1.0,
        "contribution_annual": intercept * ppy,
        "contribution_pct": intercept_pct,
    }
    return frame


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
        # Turnover must count names entering and leaving the portfolio. The
        # old intersection-only diff reported 0.0 for a complete rotation.
        union = prev.index.union(curr.index)
        previous = prev.reindex(union, fill_value=0.0)
        current = curr.reindex(union, fill_value=0.0)
        turnover = (current - previous).abs().sum() / 2
        results.append({
            "date": dates[i],
            "turnover": turnover,
            "n_assets": int((curr > 1e-6).sum()),
        })

    if not results:
        return pd.DataFrame(
            columns=["turnover", "n_assets"], index=pd.Index([], name="date"),
        )
    return pd.DataFrame(results).set_index("date")
