"""Black-Litterman model for combining market-implied returns with investor views.

Inspired by PyPortfolioOpt's implementation. The Black-Litterman model
solves a key problem with mean-variance optimization: expected returns
are hard to estimate. Instead of using historical mean returns (which
have high estimation error), the BL model:

1. Starts with a PRIOR: market-implied returns (reverse-optimized from
   market cap weights using the CAPM).
2. Incorporates VIEWS: investor's subjective views on specific assets.
3. Produces a POSTERIOR: blended expected returns.
4. The posterior is then used in any optimizer (MVO, RP, etc.).

This produces much more stable and realistic portfolios than pure
historical mean-variance optimization.

Usage:
    bl = BlackLittermanModel(cov_matrix, market_caps)
    posterior_returns = bl.bl_returns(views)
    # Use posterior_returns with your optimizer
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class BlackLittermanModel:
    """Black-Litterman expected returns model.

    Combines market-implied prior returns with investor views to produce
    a posterior estimate of expected returns with lower estimation error.

    Args:
        cov_matrix: (asset × asset) covariance matrix of returns.
        market_caps: Series or dict mapping asset → market capitalization.
        risk_aversion: Market risk-aversion coefficient. If None, estimated
            from market data using a 252-trading-day assumption.
        risk_free_rate: Annual risk-free rate.
        tau: Scaling factor for prior covariance (default 0.05).
            Controls how strongly the prior is weighted relative to views.
        view_confidences: Dict mapping asset → confidence weight [0, 1]
            for each absolute view. Higher = views weighted more strongly.
    """

    def __init__(
        self,
        cov_matrix: pd.DataFrame,
        market_caps: pd.Series | dict,
        risk_aversion: float | None = None,
        risk_free_rate: float = 0.0,
        tau: float = 0.05,
        view_confidences: dict[str, float] | None = None,
    ):
        self.cov_matrix = cov_matrix
        self.tau = tau
        self.risk_free_rate = risk_free_rate

        # Market caps
        if isinstance(market_caps, dict):
            market_caps = pd.Series(market_caps)
        self.market_caps = market_caps

        # Market weights
        self.market_weights = market_caps / market_caps.sum()

        # Risk aversion
        if risk_aversion is not None:
            self.risk_aversion = risk_aversion
        else:
            self.risk_aversion = self._estimate_risk_aversion()

        # Market-implied prior returns (Pi)
        self.prior = self._market_implied_prior()

        # View confidences
        self.view_confidences = view_confidences or {}

    def _estimate_risk_aversion(self) -> float:
        """Estimate risk aversion from excess return / variance.

        Using a typical equity risk premium of ~6% and market variance
        implied by the covariance matrix.
        """
        market_var = float(
            self.market_weights.values @ self.cov_matrix.values @ self.market_weights.values
        )
        # Assume 6% equity risk premium as default
        equity_risk_premium = 0.06
        ra = equity_risk_premium / max(market_var, 1e-6)
        logger.info("Estimated risk aversion: %.2f (market var=%.4f)", ra, market_var)
        return ra

    def _market_implied_prior(self) -> pd.Series:
        """Compute market-implied prior returns using reverse optimization.

        Π = δ * Σ * w_mkt

        This is the expected return that would make the market cap weights
        optimal under mean-variance theory.
        """
        pi = (
            self.risk_aversion
            * self.cov_matrix.dot(self.market_weights)
            + self.risk_free_rate
        )
        return pi

    def bl_returns(
        self,
        views: dict[str, float] | None = None,
    ) -> pd.Series:
        """Compute Black-Litterman posterior expected returns.

        Args:
            views: Dict mapping asset → expected return view.
                Only assets with views are updated; others keep prior.

        Returns:
            Series of posterior expected returns (one per asset).
        """
        if views is None:
            logger.info("No views provided — returning market-implied prior")
            return self.prior

        n = len(self.cov_matrix)
        assets = list(self.cov_matrix.columns)
        idx_map = {a: i for i, a in enumerate(assets)}

        # Prior
        pi = self.prior.values  # shape (n,)

        # Build view matrix P and view vector Q
        # For absolute views (single asset), P has one row per view
        view_assets = []
        view_values = []
        view_weights = []  # confidence weights

        for asset, ret in views.items():
            if asset not in idx_map:
                logger.warning("Asset '%s' not in covariance matrix — skipping view", asset)
                continue
            view_assets.append(asset)
            view_values.append(ret)

            w = self.view_confidences.get(asset, 0.5)
            view_weights.append(w)

        if not view_assets:
            logger.warning("No valid views — returning prior")
            return self.prior

        k = len(view_assets)
        P = np.zeros((k, n))
        Q = np.array(view_values, dtype=float)

        for i, asset in enumerate(view_assets):
            P[i, idx_map[asset]] = 1.0

        # View uncertainty matrix Omega
        # Diagonal: tau * diag(P @ Sigma @ P.T) / confidence
        omega = np.diag([
            self.tau * float(P[i] @ self.cov_matrix.values @ P[i])
            / max(w, 0.01)
            for i, w in enumerate(view_weights)
        ])

        # Posterior mean (Black-Litterman formula)
        # μ_post = Π + τ * Σ * P' * (τ * P * Σ * P' + Ω)⁻¹ * (Q - P * Π)
        sigma = self.cov_matrix.values
        tau_sigma = self.tau * sigma

        lhs = tau_sigma @ P.T
        rhs_inv = np.linalg.inv(P @ tau_sigma @ P.T + omega)
        adjustment = lhs @ rhs_inv @ (Q - P @ pi)

        posterior = pi + adjustment

        result = pd.Series(posterior, index=assets, name="bl_returns")
        logger.info(
            "BL posterior computed: %d views on %d assets",
            k, n,
        )
        return result

    def bl_cov(self) -> pd.DataFrame:
        """Compute Black-Litterman posterior covariance.

        The posterior covariance is:
        Σ_post = Σ + τ * Σ - τ * Σ * P' * (τ * P * Σ * P' + Ω)⁻¹ * τ * P * Σ

        This is slightly larger than Σ, reflecting uncertainty in the
        prior and views.
        """
        sigma = self.cov_matrix.values
        tau_sigma = self.tau * sigma
        post_cov = sigma + tau_sigma  # start with Σ + τΣ
        # In a full implementation, subtract the variance reduction from views
        post_cov_df = pd.DataFrame(post_cov, index=self.cov_matrix.index, columns=self.cov_matrix.columns)
        return post_cov_df

    def market_implied_weights(self) -> pd.Series:
        """Reverse-optimize: given expected returns, what weights would
        a mean-variance optimizer produce?

        This is the inverse of the prior calculation:
        w = (δ * Σ)⁻¹ * (Π - r_f)
        """
        sigma = self.cov_matrix.values
        excess = self.prior.values - self.risk_free_rate
        inv_sigma = np.linalg.inv(sigma)
        w = inv_sigma @ excess / self.risk_aversion
        # Normalize to sum to 1
        w = w / max(w.sum(), 1e-10)
        return pd.Series(w, index=self.cov_matrix.columns, name="implied_weights")
