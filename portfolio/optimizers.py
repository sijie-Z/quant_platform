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

# Annualization convention for the mean-variance objective.
#
# The engine hands the optimizer a *daily* covariance matrix (estimated from
# daily returns) and a rank alpha whose cross-section is scaled to [-0.5, 0.5]
# by the alpha pipeline (cross-sectional percentile rank minus 0.5).  Mixing a
# unitless rank with a daily variance makes the risk term ~1e-5 of the alpha
# term, so at the configured risk_aversion=1 the "mean-variance" optimizer was
# a constrained alpha ranking that pinned every name at the cap.
#
# The objective is now expressed in annual excess-return and annual-variance
# units:
#     maximise  ALPHA_ANNUAL_SCALE * (rank alpha)'w
#               - risk_aversion * w' (252 * Sigma_daily) w
# ``ALPHA_ANNUAL_SCALE = 0.04`` maps the full rank span [-0.5, 0.5] to
# +/-2% expected annual excess return, a deliberately modest assumption for a
# long-only A-share factor portfolio.  With that convention ``risk_aversion``
# is the coefficient of annual variance (the standard 1/2*gamma convention
# corresponds to gamma = 2 * risk_aversion); the YAML default of 1.0 is a
# balanced trade-off.  Changing this constant changes the effective risk
# aversion one-for-one; it is intentionally a module constant so tests and
# research code can reference the documented convention.
TRADING_DAYS_PER_YEAR = 252
ALPHA_ANNUAL_SCALE = 0.04



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
        sector_map: pd.Series | None = None,
        adv: pd.Series | None = None,
        aum: float | None = None,
    ) -> pd.Series:
        """Compute target weights.

        Args:
            signal: Alpha signal per asset (higher = better).
            cov_matrix: (asset x asset) covariance matrix.
            prices: Current close prices per asset.
            prev_weights: Previous period weights for turnover constraint.
            sector_map: Asset -> sector mapping.
            adv: Optional (asset) average daily traded value in CNY. Used by
                optimizers that implement the R8 capacity cap; ignored
                elsewhere.
            aum: Optional portfolio value in CNY paired with ``adv``.

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
        adv: pd.Series | None = None,
        aum: float | None = None,
    ) -> pd.Series:
        # ``adv`` / ``aum`` are accepted for API parity but not used: equal
        # weight is not an optimization problem, so the R8 cost terms and
        # capacity cap do not apply here (documented scope).
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

    .. math::

        \\max_w \\; s \\, \\alpha' w - \\gamma \\, w' (252 \\Sigma_{daily}) w
                  - c \\sum_i |w_i - w^{prev}_i|
                  - \\lambda \\sum_i (w_i - w^{prev}_i)^2

    where ``alpha`` is the rank signal in ``[-0.5, 0.5]``, ``s`` is
    :data:`ALPHA_ANNUAL_SCALE` (0.04, i.e. the full rank span represents a
    +/-2% annual excess return), ``Sigma_daily`` is the covariance of daily
    returns supplied by the engine (the factor 252 annualizes it), and
    ``gamma`` is ``constraints.risk_aversion``.  The risk aversion is therefore
    expressed in annual-variance units: it equals the standard ``gamma / 2``
    of the textbook ``mu'w - (gamma/2) w'Sigma w`` objective.  Do not pass an
    already-annualized covariance matrix; it would be annualized twice.

    The R8 cost terms use ``c = constraints.transaction_cost_bps / 1e4``
    (one-way cost per unit of two-way turnover) and
    ``\\lambda = constraints.turnover_penalty``.  Each term is added only when
    its coefficient is positive **and** ``prev_weights`` is supplied; with the
    default coefficients of 0 the objective, variables and constraints are
    exactly the pre-R8 problem (bit-identical decision).

    Capacity (R8): when ``constraints.max_participation > 0`` and the caller
    supplies both ``adv`` (per-asset average daily traded value in CNY) and a
    positive ``aum``, every asset's upper bound becomes
    ``min(max_weight, max_participation * adv_i / aum)``.  Assets with a
    missing or non-positive ADV keep the plain ``max_weight`` cap and the
    count is logged -- silently treating a missing quote as zero liquidity
    would make most of the universe untradeable.  If the caps sum to less
    than 1.0 the capacity-constrained problem cannot be funded; the optimizer
    records a warning and falls back to equal weight (which ignores the ADV
    cap; the fallback is counted and the reason is exposed).

    Subject to: sum(w) = 1, long-only (unless ``long_only=False``, in which
    case the solver bound is -0.1), w <= max_weight (or the capacity caps
    above), sector constraints.
    """

    def optimize(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        prev_weights: pd.Series | None = None,
        sector_map: pd.Series | None = None,
        adv: pd.Series | None = None,
        aum: float | None = None,
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

        # Keep existing holdings in the candidate set. When the signal rotates
        # more than the turnover budget out of the top-N, the `dropped` term
        # in the turnover constraint alone exceeds its right-hand side, so no
        # portfolio satisfies it and the solver reports infeasible -- every
        # such rebalance silently fell back to equal weight. With the old
        # holdings available, the optimizer can choose to keep them and trade
        # within the cap.
        if prev_weights is not None and self.constraints.max_turnover < 1.0:
            held = prev_weights[prev_weights.abs() > 1e-12].index
            # A held name can be absent from today's signal (suspended or
            # dropped from the universe) and from the covariance window.
            # Those cannot be optimisation variables; their weight is the
            # `dropped` term instead.
            held = held.intersection(signal.index)
            if cov_matrix is not None:
                held = held.intersection(cov_matrix.index)
            top_assets = top_assets.union(held)

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

        # Annualize the daily covariance inside the objective and scale the
        # rank alpha into annual-return units (see the class docstring and
        # ALPHA_ANNUAL_SCALE). Then both terms are O(1-10%) at the configured
        # risk_aversion, so doubling the covariance genuinely moves weights.
        cov_annual = cov * TRADING_DAYS_PER_YEAR
        w = cp.Variable(n)
        expected_return = ALPHA_ANNUAL_SCALE * (alpha_vec @ w)
        risk = cp.quad_form(w, cov_annual)
        objective_terms = expected_return - self.constraints.risk_aversion * risk

        # R8: transaction-cost / quadratic-turnover terms. They are built only
        # when the coefficient is positive and previous weights exist, so the
        # default (0) configuration constructs exactly the pre-R8 objective.
        # Holdings dropped from the candidate set are constants here: they are
        # not decision variables and their |0 - prev| cost cannot move the
        # argmax, so they stay out of these sums (the turnover constraint
        # below still accounts for them).
        cost_prev = None
        if prev_weights is not None and (
            self.constraints.transaction_cost_bps > 0
            or self.constraints.turnover_penalty > 0
        ):
            cost_prev = prev_weights.reindex(assets, fill_value=0.0).values
        if cost_prev is not None and self.constraints.transaction_cost_bps > 0:
            objective_terms = objective_terms - (
                self.constraints.transaction_cost_bps / 1e4
            ) * cp.sum(cp.abs(w - cost_prev))
        if cost_prev is not None and self.constraints.turnover_penalty > 0:
            objective_terms = objective_terms - (
                self.constraints.turnover_penalty
            ) * cp.sum_squares(w - cost_prev)
        objective = cp.Maximize(objective_terms)

        # R8 capacity cap: w_i <= max_participation * ADV_i / AUM. Engaged
        # only when the participation rate, the ADV series and a positive AUM
        # are all supplied. Missing / non-positive ADV assets keep the plain
        # max_weight cap (and are logged): treating a missing quote as zero
        # liquidity would make most of the universe untradeable, while
        # skipping the cap silently would overstate capacity.
        capacity_caps = None
        if (
            self.constraints.max_participation > 0
            and adv is not None
            and aum is not None
            and float(aum) > 0
        ):
            adv_series = pd.Series(adv, dtype=float).reindex(assets)
            adv_values = adv_series.to_numpy(dtype=float)
            usable = np.isfinite(adv_values) & (adv_values > 0)
            capacity_caps = np.full(n, float(self.constraints.max_weight))
            capacity_caps[usable] = np.minimum(
                capacity_caps[usable],
                float(self.constraints.max_participation)
                * adv_values[usable]
                / float(aum),
            )
            n_unusable = int((~usable).sum())
            if n_unusable:
                logger.warning(
                    "MVO capacity: %d/%d assets have no positive ADV; the "
                    "participation cap is not applied to them "
                    "(max_weight still binds)", n_unusable, n,
                )
            if float(capacity_caps.sum()) < 1.0 - 1e-9:
                return self._equal_weight_fallback(
                    signal, cov_matrix, prices, prev_weights, sector_map,
                    "capacity caps sum to "
                    f"{float(capacity_caps.sum()):.4f} < 1.0 under "
                    f"max_participation={self.constraints.max_participation} "
                    f"and aum={float(aum):.0f}; the capacity-constrained "
                    "problem cannot be funded (equal-weight fallback ignores "
                    "the ADV cap)",
                )
        elif self.constraints.max_participation > 0:
            # Configured but not usable: the caller (BacktestEngine) logs the
            # user-facing warning. Keep a trace for direct API users.
            logger.debug(
                "MVO capacity configured (max_participation=%.3f) but "
                "adv/aum were not supplied (adv=%s, aum=%s); optimizing "
                "without the ADV cap",
                self.constraints.max_participation, adv is not None, aum,
            )

        upper_bound = (
            self.constraints.max_weight
            if capacity_caps is None else capacity_caps
        )
        constraints_list = [
            cp.sum(w) == 1.0,
            w >= 0 if self.constraints.long_only else w >= -0.1,
            w <= upper_bound,
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
            budget = 2 * self.constraints.max_turnover - dropped
            if budget < -1e-9:
                # Only possible when a prior holding is missing from the
                # covariance universe and could not be added above. Selling
                # it alone would exceed the turnover cap, so no solution to
                # the constrained problem exists; the engine clamps realised
                # turnover at execution time.
                logger.warning(
                    "MVO: %.2f of prior holdings are outside the covariance "
                    "universe and exceed the turnover budget; turnover "
                    "constraint skipped for this rebalance", dropped,
                )
            else:
                constraints_list.append(
                    cp.sum(cp.abs(w - prev_w)) <= max(0.0, budget)
                )

        # Try only solvers that are actually installed. This loop used to be
        # `[cp.ECOS, cp.SCS]`; ECOS is absent from cvxpy 1.7 wheels, so the
        # first attempt always failed and every rebalance depended on SCS,
        # which reports `infeasible` on these problems far more often than
        # CLARABEL. The default mean-variance config silently fell back to
        # equal weight on 54 of ~60 rebalances.
        solvers = [
            solver for solver in ("CLARABEL", "ECOS", "SCS")
            if solver in cp.installed_solvers()
        ]
        if not solvers:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "no convex solver is installed",
            )

        solved = False
        used_solver = "none"
        attempts = []
        problem = None
        for solver in solvers:
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
                    used_solver = solver
                    break
                attempts.append(f"{solver}={problem.status}")
            except Exception as e:
                logger.debug("MVO solver %s failed: %s", solver, e)
                attempts.append(f"{solver}=error")
                continue

        if not solved:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                f"MVO solver failed ({', '.join(attempts) or 'no attempt'}, n={n})",
            )

        logger.info("MVO SUCCESS: solver=%s n=%d status=%s w_min=%.4f w_max=%.4f",
                     used_solver, n, problem.status, w.value.min(), w.value.max())

        weights = pd.Series(w.value, index=assets)
        # Shorts are a supported configuration now: the old unconditional
        # `clip(lower=0)` silently turned every long/short solve into a
        # long-only book and renormalized away the solver's solution.
        if self.constraints.long_only:
            weights = weights.clip(lower=0)
        total = weights.sum()
        if total <= 1e-10:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "MVO returned an all-zero portfolio",
            )
        weights = weights / total

        # Volatility targeting: scale weights down if portfolio vol exceeds
        # the target. `target_volatility` is annualized while `cov` is daily,
        # so the daily target is target/sqrt(252). Comparing the annual target
        # directly with a daily vol made the constraint sqrt(252)x too loose
        # (a 10-20% annual target never bound a ~1.5% daily-vol portfolio).
        if self.constraints.target_volatility > 0 and cov_matrix is not None:
            port_vol_daily = np.sqrt(weights[assets] @ cov @ weights[assets])
            target_daily = self.constraints.target_volatility / np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
            if port_vol_daily > 1e-8:
                scale = min(1.0, target_daily / port_vol_daily)
                weights = weights * scale
                # sum(weights) < 1 means remaining weight is cash (0% return)

        weights = weights.reindex(signal.index, fill_value=0.0)

        # The target-level turnover cap is a hard constraint, so it is applied
        # last (the solver's version is approximate). With a capacity cap the
        # blend toward a previous book that itself exceeded the new cap could
        # otherwise push a weight back above it, so clip defensively.
        weights = _limit_turnover(
            weights, prev_weights, self.constraints.max_turnover, signal.index,
        )
        if capacity_caps is not None:
            cap_series = pd.Series(
                capacity_caps, index=assets,
            ).reindex(weights.index)
            above = weights > cap_series + 1e-9
            if bool(above.any()):
                n_above = int(above.sum())
                weights = weights.where(~above, cap_series)
                # Refill the clipped weight out of names that still have
                # headroom under their cap, so a fully fundable book stays
                # fully invested; any residual (caps exhausted) stays cash.
                deficit = 1.0 - float(weights.sum())
                if deficit > 1e-12:
                    headroom = (cap_series - weights).clip(lower=0.0).fillna(
                        0.0
                    )
                    # Sector caps must hold on the refill too: cap each
                    # sector's total headroom at its remaining room and scale
                    # the per-asset headroom inside the sector so the sum
                    # cannot exceed it. Names outside the sector map are not
                    # covered by the sector constraint.
                    if (
                        sector_map is not None
                        and self.constraints.max_sector_exposure < 1.0
                    ):
                        sectors = sector_map.reindex(weights.index)
                        totals = weights.groupby(sectors).sum()
                        sector_room = (
                            float(self.constraints.max_sector_exposure)
                            - totals
                        ).clip(lower=0.0)
                        hr_by_sector = headroom.groupby(sectors).sum()
                        scale = (
                            sector_room
                            / hr_by_sector.where(hr_by_sector > 0.0)
                        ).clip(upper=1.0).fillna(0.0)
                        headroom = headroom * sectors.map(scale).fillna(1.0)
                    room = float(headroom.sum())
                    if room > 0.0:
                        weights = weights + headroom * (
                            min(deficit, room) / room
                        )
                logger.warning(
                    "MVO: the post-turnover blend exceeded the ADV capacity "
                    "cap for %d asset(s); clipped to the cap and refilled "
                    "available headroom (residual cash %.3e). The engine's "
                    "execution-day turnover clamp still applies.",
                    n_above, max(0.0, 1.0 - float(weights.sum())),
                )
        return weights


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity: equal risk contribution from each asset.

    Solves the Spinu (2013) log-barrier problem

    .. math::

        \\min_{v>0} \\; \\tfrac12 v' \\Sigma v - \\frac{1}{n}\\sum_i \\log v_i

    without a budget constraint, then normalizes ``w = v / sum(v)``.  The
    first-order condition ``v_i (Sigma v)_i = 1/n`` makes every asset's risk
    contribution ``w_i (Sigma w)_i / (w' Sigma w)`` equal to ``1/n``, and the
    normalized solution is invariant to multiplying ``Sigma`` by a constant.
    Upper bounds enter in homogeneous form (``v_i <= cap * sum(v)``).  When a
    covariance matrix is unavailable, or the cap is infeasible for the
    selected universe, the optimizer falls back to equal weight and records
    the reason in ``fallback_count`` / ``last_fallback_reason``.
    """

    def optimize(
        self,
        signal: pd.Series,
        cov_matrix: pd.DataFrame | None = None,
        prices: pd.Series | None = None,
        prev_weights: pd.Series | None = None,
        sector_map: pd.Series | None = None,
        adv: pd.Series | None = None,
        aum: float | None = None,
    ) -> pd.Series:
        # ``adv`` / ``aum`` are accepted for API parity but not implemented
        # for risk parity in R8 (documented scope: the cost terms and the
        # capacity cap are MVO-only this round).
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
        v = cp.Variable(n)

        # Spinu (2013) log-barrier risk-parity formulation:
        #   min 0.5*v'*Sigma*v - (1/n)*sum(log(v)),  v > 0
        # with NO sum(v) == 1 constraint.  Under a budget constraint the KKT
        # condition is v_i*(Sigma v)_i = 1/n - lambda*v_i with
        # lambda = 1 - v'Sigma v ~ 1 for a covariance in daily units, so equal
        # weights solve it: the old "risk parity" was an expensive equal-weight
        # optimizer (risk-contribution spread 152% on the default config).
        # Without the budget constraint lambda = 0 and v_i*(Sigma v)_i is
        # equal for every i; normalizing v/sum(v) preserves the equal risk
        # contributions and makes the solution invariant to the scale of
        # Sigma.
        #
        # The per-name cap and sector caps are applied in homogeneous form
        # (v_i <= cap*sum(v), sum_sector v <= cap*sum(v)).  They are invariant
        # to the scaling of v, so they constrain the normalized portfolio
        # exactly while keeping the problem convex.
        objective = cp.Minimize(
            0.5 * cp.quad_form(v, cov) - cp.sum(cp.log(v)) / n
        )
        constraints_list = [v >= 1e-9]
        if self.constraints.max_weight < 1.0:
            constraints_list.append(
                v <= self.constraints.max_weight * cp.sum(v)
            )
        if sector_map is not None and self.constraints.max_sector_exposure < 1.0:
            sectors = sector_map.reindex(selected)
            for sector in sectors.dropna().unique():
                mask = (sectors == sector).values.astype(float)
                if mask.sum() > 0:
                    constraints_list.append(
                        mask @ v <= self.constraints.max_sector_exposure * cp.sum(v)
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
        # The log-barrier solution is only as precise as the solver tolerance.
        # CLARABEL's defaults leave a ~1e-4 relative spread in the risk
        # contributions; tighten the interior-point tolerances so the
        # equal-risk-contribution identity holds to ~1e-10.
        solver_kwargs = {
            "CLARABEL": {
                "tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12,
            },
            "ECOS": {"abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10},
            "SCS": {"eps": 1e-9, "max_iters": 100_000},
        }
        for solver in solvers:
            try:
                problem = cp.Problem(objective, constraints_list)
                problem.solve(
                    solver=solver, verbose=False,
                    **solver_kwargs.get(solver, {}),
                )
                if (
                    problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and v.value is not None
                    and np.all(np.isfinite(v.value))
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

        weights = pd.Series(v.value, index=selected)
        weights = weights.clip(lower=0)
        total = weights.sum()
        if total <= 1e-10:
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "risk parity returned an all-zero portfolio",
            )
        weights = weights / total
        if weights.max() > self.constraints.max_weight + 1e-4:
            # Only reachable when the cap is infeasible for the selected
            # universe (n * max_weight < 1); equal weight respects the cap by
            # construction because it selects at most 1/max_weight names.
            return self._equal_weight_fallback(
                signal, cov_matrix, prices, prev_weights, sector_map,
                "risk parity weights exceed max_weight "
                f"({weights.max():.4f} > {self.constraints.max_weight})",
            )
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
