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


def _limit_turnover(
    weights: pd.Series,
    prev_weights: pd.Series | None,
    max_turnover: float,
    universe: pd.Index | None = None,
) -> pd.Series:
    """Blend a target portfolio toward the previous one until turnover fits.

    Turnover is measured over the *union* of both indices with zero fill, so
    names entering or leaving the portfolio count. The old implementation
    aligned on the intersection, which understated exactly the component an
    equal-weight fallback creates, and then renormalized after blending --
    which could push turnover straight back over the cap.
    """
    if prev_weights is None or max_turnover >= 1.0:
        return weights

    index = weights.index.union(prev_weights.index)
    current = weights.reindex(index, fill_value=0.0)
    previous = prev_weights.reindex(index, fill_value=0.0)
    turnover = float((current - previous).abs().sum() / 2)
    if turnover <= max_turnover:
        return weights

    scale = max_turnover / turnover
    blended = previous + scale * (current - previous)
    if universe is not None:
        blended = blended.reindex(universe, fill_value=0.0)
    return blended


class PortfolioOptimizer(ABC):
    """Abstract portfolio optimizer."""

    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        # Fallback accounting: a silent switch to equal weight changes what
        # the backtest actually held, so the engine must be able to report it.
        self.fallback_count = 0
        self.last_fallback_reason: str | None = None

    def _equal_weight_fallback(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None,
        prices: pd.Series | None,
        prev_weights: pd.Series | None,
        sector_map: pd.Series | None,
        reason: str,
    ) -> pd.Series:
        self.fallback_count += 1
        self.last_fallback_reason = reason
        logger.warning("Optimizer fallback to equal weight: %s", reason)
        return EqualWeightOptimizer(self.constraints).optimize(
            signal, cov_matrix, prices, prev_weights, sector_map,
        )

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

        # Normalize DOWN only. Scaling up used to push single names back above
        # max_weight and sectors back above their cap whenever the candidate
        # set was too narrow; a total below 1 is simply cash.
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

        return _limit_turnover(
            weights, prev_weights, self.constraints.max_turnover, signal.index,
        )


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
        import math

        try:
            import cvxpy as cp
        except ImportError:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "cvxpy is not installed",
            )
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
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "fewer than two optimizable assets",
            )

        assets = top_assets.sort_values()
        n = len(assets)

        # A single inf signal makes the QP unbounded and drives both solvers
        # into the fallback path; treat non-finite alpha as neutral instead.
        alpha_vec = np.nan_to_num(
            signal[assets].values, nan=0.0, posinf=0.0, neginf=0.0,
        )
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

        # Turnover constraint. Holdings dropped from the top-N candidate set
        # are still sold, so their weight belongs on the left-hand side too;
        # reindexing previous weights onto `assets` alone understated it.
        if prev_weights is not None and self.constraints.max_turnover < 1.0:
            prev_w = prev_weights.reindex(assets, fill_value=0.0).values
            dropped = float(
                prev_weights.reindex(
                    prev_weights.index.difference(assets)
                ).abs().sum()
            )
            constraints_list.append(
                cp.sum(cp.abs(w - prev_w)) + dropped <= 2 * self.constraints.max_turnover
            )

        solved = False
        used_solver = "none"
        problem = None
        for solver in [cp.ECOS, cp.SCS]:
            try:
                problem = cp.Problem(objective, constraints_list)
                problem.solve(solver=solver, verbose=False)
                solution = w.value
                if (
                    problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and solution is not None
                    and np.all(np.isfinite(solution))
                ):
                    solved = True
                    used_solver = str(solver)
                    break
            except Exception as e:
                logger.debug("MVO solver %s failed: %s", str(solver), e)
                continue

        if not solved:
            status = problem.status if problem is not None else "N/A"
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                f"MVO solver failed (solver={used_solver}, n={n}, status={status})",
            )

        logger.info("MVO SUCCESS: solver=%s n=%d status=%s w_min=%.4f w_max=%.4f",
                     used_solver, n, problem.status, w.value.min(), w.value.max())

        weights = pd.Series(w.value, index=assets)
        weights = weights.clip(lower=0)
        total = weights.sum()
        if total <= 1e-10:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "MVO returned an all-zero portfolio",
            )
        weights = weights / total

        # Volatility targeting: scale weights down if portfolio vol exceeds target
        if self.constraints.target_volatility > 0 and cov_matrix is not None:
            port_vol = np.sqrt(weights[assets] @ cov @ weights[assets])
            if port_vol > 1e-8:
                scale = min(1.0, self.constraints.target_volatility / port_vol)
                weights = weights * scale
                # sum(weights) < 1 means remaining weight is cash (0% return)

        weights = weights.reindex(signal.index, fill_value=0.0)

        # The target-level turnover cap is a hard constraint, so it is applied
        # last (the solver's version is approximate).
        return _limit_turnover(
            weights, prev_weights, self.constraints.max_turnover, signal.index,
        )


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
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "risk parity needs a covariance matrix",
            )

        # Narrow the universe, but keep more names than the minimum needed to
        # satisfy sum(w)=1 under the cap. Selecting exactly ceil(1/max_weight)
        # names (20 at the default 5% cap) made equal weight the only feasible
        # solution; 2x the minimum leaves room for a real risk solution.
        min_names = int(np.ceil(1.0 / self.constraints.max_weight))
        top_n = min(len(signal.dropna()), max(20, 2 * min_names))
        selected = signal.dropna().nlargest(top_n).index
        selected = selected.intersection(cov_matrix.index)
        selected = selected.sort_values()

        if len(selected) < 2:
            return self._equal_weight_fallback(
                signal, None, prices, prev_weights, sector_map,
                "fewer than two optimizable assets",
            )

        cov = cov_matrix.loc[selected, selected].values
        cov = _ensure_psd(cov)

        n = len(selected)
        w = cp.Variable(n)

        # Convex log-barrier risk-parity formulation:
        #   min 0.5*w'*Sigma*w - (1/n)*sum(log(w))
        #   s.t. sum(w)=1, 0 < w <= cap
        # The old "minimise the variance of risk contributions" objective is
        # not DCP (cvxpy raised DCPError), and it only tried ECOS, which is
        # not installed -- so this optimizer silently always fell back to
        # equal weight.
        objective = cp.Minimize(
            0.5 * cp.quad_form(w, cov) - cp.sum(cp.log(w)) / n
        )
        constraints_list = [
            cp.sum(w) == 1.0,
            w >= 1e-6,
            w <= self.constraints.max_weight,
        ]
        if sector_map is not None and self.constraints.max_sector_exposure < 1.0:
            sectors = sector_map.reindex(selected)
            for sector in sectors.dropna().unique():
                mask = (sectors == sector).values.astype(float)
                if mask.sum() > 0:
                    constraints_list.append(
                        mask @ w <= self.constraints.max_sector_exposure
                    )

        solvers = [
            solver for solver in ("CLARABEL", "SCS", "ECOS")
            if solver in cp.installed_solvers()
        ]
        if not solvers:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "no convex solver is installed",
            )

        solved = False
        last_error = ""
        problem = None
        for solver in solvers:
            try:
                problem = cp.Problem(objective, constraints_list)
                problem.solve(solver=solver, verbose=False)
                if (
                    problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and w.value is not None
                    and np.all(np.isfinite(w.value))
                ):
                    solved = True
                    break
            except Exception as exc:
                last_error = str(exc)
                continue

        if not solved:
            status = problem.status if problem is not None else "N/A"
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                f"risk parity solver failed ({status}): {last_error or 'no optimal solution'}",
            )

        weights = pd.Series(w.value, index=selected)
        weights = weights.clip(lower=0)
        total = weights.sum()
        if total <= 1e-10:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "risk parity returned an all-zero portfolio",
            )
        weights = weights / total
        weights = weights.reindex(signal.index, fill_value=0.0)
        return _limit_turnover(
            weights, prev_weights, self.constraints.max_turnover, signal.index,
        )


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
