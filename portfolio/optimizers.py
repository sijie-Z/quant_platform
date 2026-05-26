"""Portfolio optimizers: equal weight, mean-variance, risk parity.

Each optimizer takes a signal and optional covariance matrix, then
produces a weight vector satisfying constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer(ABC):
    """Abstract portfolio optimizer."""

    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints

    @abstractmethod
    def optimize(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        prev_weights: pd.Series | None = None,
    ) -> pd.Series:
        """Compute target weights.

        Args:
            signal: Alpha signal per asset (higher = better).
            cov_matrix: (asset x asset) covariance matrix.
            prices: Current close prices per asset.
            prev_weights: Previous period weights for turnover constraint.

        Returns:
            Target weights (sum to 1.0).
        """
        ...


class EqualWeightOptimizer(PortfolioOptimizer):
    """Equal weight to top-N stocks by signal.

    Not as trivial as 1/N — applies constraints:
    - Filter to stocks with positive signal
    - Apply max_weight cap
    - Apply sector limits
    """

    def optimize(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        prev_weights: pd.Series | None = None,
        sector_map: pd.Series | None = None,
    ) -> pd.Series:
        # Select top stocks by signal
        positive = signal[signal > 0].dropna()
        if len(positive) == 0:
            positive = signal.dropna().nlargest(50)

        n = min(len(positive), int(1.0 / self.constraints.max_weight))
        selected = positive.nlargest(n)
        weights = pd.Series(0.0, index=signal.index)

        # Equal weight with max_weight cap
        w = 1.0 / n
        w = min(w, self.constraints.max_weight)

        # Apply weights to selected stocks
        for asset in selected.index:
            weights[asset] = w

        # If sector constraint, scale down overweight sectors
        if sector_map is not None and self.constraints.max_sector_exposure < 1.0:
            for _ in range(3):  # Iterate to converge
                for sector in sector_map.unique():
                    sector_assets = sector_map[sector_map == sector].index
                    sector_assets = sector_assets.intersection(weights.index)
                    sector_weight = weights[sector_assets].sum()
                    if sector_weight > self.constraints.max_sector_exposure:
                        scale = self.constraints.max_sector_exposure / sector_weight
                        weights[sector_assets] *= scale

        # Normalize
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights[:] = 0.0

        # Turnover constraint
        if prev_weights is not None and self.constraints.max_turnover < 1.0:
            turnover = (weights - prev_weights).abs().sum() / 2
            if turnover > self.constraints.max_turnover:
                scale = self.constraints.max_turnover / turnover
                weights = prev_weights + scale * (weights - prev_weights)
                weights = weights / weights.sum()

        return weights


class MeanVarianceOptimizer(PortfolioOptimizer):
    """Mean-variance optimization using cvxpy.

    Maximize: w' * alpha - lambda * w' * Sigma * w
    Subject to: sum(w) = 1, w >= 0, w <= max_weight, sector constraints.
    """

    def optimize(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        prev_weights: pd.Series | None = None,
        sector_map: pd.Series | None = None,
    ) -> pd.Series:
        import cvxpy as cp

        import math
        # Pre-filter: only optimize top-N stocks by signal for numerical stability
        # Must select enough to satisfy sum(w) == 1 with max_weight cap
        min_needed = int(math.ceil(1.0 / max(self.constraints.max_weight, 0.01)))
        max_n = max(
            min_needed,
            min(len(signal.dropna()), 100)
        )

        top_assets = signal.dropna().nlargest(max_n).index
        if cov_matrix is not None:
            top_assets = top_assets.intersection(cov_matrix.index)

        if len(top_assets) < 2:
            return EqualWeightOptimizer(self.constraints).optimize(
                signal, cov_matrix, prices, prev_weights, sector_map
            )

        assets = top_assets.sort_values()
        n = len(assets)

        alpha_vec = signal[assets].values
        if cov_matrix is not None:
            cov = cov_matrix.loc[assets, assets].values
        else:
            cov = np.eye(n)

        # Ensure positive semi-definite
        cov = _ensure_psd(cov)

        w = cp.Variable(n)
        expected_return = alpha_vec @ w
        risk = cp.quad_form(w, cov)
        objective = cp.Maximize(expected_return - self.constraints.risk_aversion * risk)

        constraints_list = [
            cp.sum(w) == 1.0,
            w >= 0 if self.constraints.long_only else w >= -0.1,
            w <= self.constraints.max_weight,
        ]

        # Sector constraint (skip if all assets are in the same sector — infeasible otherwise)
        unique_sectors = sector_map[assets].unique() if sector_map is not None else []
        if len(unique_sectors) > 1 and self.constraints.max_sector_exposure < 1.0:
            for sector in unique_sectors:
                mask = (sector_map[assets] == sector).values.astype(float)
                if mask.sum() > 0:
                    constraints_list.append(mask @ w <= self.constraints.max_sector_exposure)

        # Turnover constraint
        if prev_weights is not None and self.constraints.max_turnover < 1.0:
            prev_w = prev_weights.reindex(assets, fill_value=0.0).values
            constraints_list.append(
                cp.sum(cp.abs(w - prev_w)) <= 2 * self.constraints.max_turnover
            )

        solved = False
        for solver in [cp.ECOS, cp.SCS]:
            try:
                problem = cp.Problem(objective, constraints_list)
                problem.solve(solver=solver, verbose=False)
                if w.value is not None:
                    solved = True
                    break
            except Exception:
                continue

        if not solved:
            logger.warning("MVO solver failed, falling back to equal weight")
            return EqualWeightOptimizer(self.constraints).optimize(
                signal, cov_matrix, prices, prev_weights, sector_map
            )

        weights = pd.Series(w.value, index=assets)
        weights = weights.clip(lower=0)
        total = weights.sum()
        if total > 1e-10:
            weights = weights / total
        else:
            weights = pd.Series(0.0, index=signal.index)

        # Volatility targeting: scale weights down if portfolio vol exceeds target
        if self.constraints.target_volatility > 0 and cov_matrix is not None:
            port_vol = np.sqrt(weights[assets] @ cov @ weights[assets])
            if port_vol > 1e-8:
                scale = min(1.0, self.constraints.target_volatility / port_vol)
                weights = weights * scale
                # sum(weights) < 1 means remaining weight is cash (0% return)

        return weights.reindex(signal.index, fill_value=0.0)


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity: equal risk contribution from each asset.

    Assets contribute equally to portfolio risk.
    Useful when alpha signal is weak or noisy.
    """

    def optimize(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        prev_weights: pd.Series | None = None,
        sector_map: pd.Series | None = None,
    ) -> pd.Series:
        import cvxpy as cp

        if cov_matrix is None:
            return EqualWeightOptimizer(self.constraints).optimize(
                signal, cov_matrix, prices, prev_weights, sector_map
            )

        # Use top assets by signal to narrow universe
        top_n = min(len(signal.dropna()), int(1.0 / self.constraints.max_weight))
        selected = signal.dropna().nlargest(top_n).index
        selected = selected.intersection(cov_matrix.index)
        selected = selected.sort_values()

        if len(selected) < 2:
            return EqualWeightOptimizer(self.constraints).optimize(
                signal, None, prices, prev_weights, sector_map
            )

        cov = cov_matrix.loc[selected, selected].values
        cov = _ensure_psd(cov)

        n = len(selected)
        w = cp.Variable(n)
        portfolio_risk = cp.quad_form(w, cov)

        # Marginal risk contributions
        marginal_risk = cov @ w
        risk_contribution = cp.multiply(w, marginal_risk)

        # Minimize variance of risk contributions (equal risk contribution)
        objective = cp.Minimize(cp.sum_squares(risk_contribution - cp.sum(risk_contribution) / n))

        constraints_list = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= self.constraints.max_weight,
        ]

        try:
            problem = cp.Problem(objective, constraints_list)
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            return EqualWeightOptimizer(self.constraints).optimize(
                signal, cov_matrix, prices, prev_weights, sector_map
            )

        if w.value is None:
            return EqualWeightOptimizer(self.constraints).optimize(
                signal, cov_matrix, prices, prev_weights, sector_map
            )

        weights = pd.Series(w.value, index=selected)
        weights = weights.clip(lower=0)
        weights = weights / weights.sum()
        return weights.reindex(signal.index, fill_value=0.0)


def _ensure_psd(cov: np.ndarray) -> np.ndarray:
    """Ensure matrix is positive semi-definite by fixing eigenvalues.

    Handles NaN/inf gracefully by falling back to identity + small off-diagonal.
    """
    # Check for invalid values
    if np.any(~np.isfinite(cov)):
        n = cov.shape[0]
        return np.eye(n) * 0.9 + 0.1 / n

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    except np.linalg.LinAlgError:
        n = cov.shape[0]
        return np.eye(n) * 0.9 + 0.1 / n
