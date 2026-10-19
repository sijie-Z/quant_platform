"""Tests for portfolio optimizers."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.optimizers import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)


def _make_signal_and_cov(n_assets=50):
    np.random.seed(42)
    assets = [f"{i:06d}.SH" for i in range(n_assets)]

    signal = pd.Series(np.random.randn(n_assets), index=assets)
    signal = signal - signal.min() + 0.1  # Make positive

    # Generate a valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov = A @ A.T + np.eye(n_assets) * 0.01
    cov = pd.DataFrame(cov, index=assets, columns=assets)

    return signal, cov


def test_equal_weight_basic():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(max_weight=0.10)
    optimizer = EqualWeightOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.sum() > 0
    assert weights.max() <= constraints.max_weight + 1e-6


def test_equal_weight_sums_to_one():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints()
    optimizer = EqualWeightOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    # Weights should sum to approximately 1
    assert 0.95 <= weights.sum() <= 1.05


def test_mean_variance_basic():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(risk_aversion=1.0)
    optimizer = MeanVarianceOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.sum() > 0
    assert (weights >= -0.01).all()  # Long only (allow tiny numerical issues)


def test_mean_variance_constraints():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(max_weight=0.10, risk_aversion=0.5)
    optimizer = MeanVarianceOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.max() <= constraints.max_weight + 1e-6


def test_prior_holding_outside_todays_signal_is_not_an_optimization_variable():
    """A suspended/delisted holding cannot be a cvxpy variable.

    Keeping prior holdings in the candidate set must not blindly union names
    that are absent from the signal and covariance; that raised KeyError on
    the first real pipeline run.
    """
    assets = [f"S{i:03d}" for i in range(30)]
    cov = pd.DataFrame(np.eye(30) * 0.04, index=assets, columns=assets)
    signal = pd.Series(1.0, index=assets)
    prev = pd.Series(1.0 / 31, index=assets + ["DELISTED"])

    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.05, max_turnover=0.30, risk_aversion=1.0,
    ))
    weights = optimizer.optimize(signal, cov, prev_weights=prev)

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert abs(float(weights.sum()) - 1.0) < 1e-3


def test_signal_rotation_beyond_turnover_budget_stays_feasible():
    """When the new signal rotates out more than the turnover budget, the
    fixed `dropped` term alone exceeded the constraint's right-hand side.

    No portfolio could satisfy the problem, the solver reported infeasible,
    and every such rebalance silently fell back to equal weight (54 of 60
    rebalances on the default pipeline).
    """
    assets = [f"S{i:03d}" for i in range(200)]
    cov = pd.DataFrame(np.eye(200) * 0.04, index=assets, columns=assets)

    first = pd.Series(0.0, index=assets)
    first.iloc[:100] = 1.0
    second = pd.Series(0.0, index=assets)
    second.iloc[100:] = 1.0

    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.05, max_turnover=0.30, risk_aversion=1.0,
    ))

    w1 = optimizer.optimize(first, cov)
    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason

    w2 = optimizer.optimize(second, cov, prev_weights=w1)

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert abs(float(w2.sum()) - 1.0) < 1e-3


def test_mean_variance_uses_an_installed_solver():
    """The solver loop hardcoded `[cp.ECOS, cp.SCS]`.

    ECOS is not included in cvxpy 1.7 wheels, so the first attempt always
    failed and every solve depended on SCS, which reported `infeasible` on the
    default 100-asset constrained problem. With `n_optimizer_fallbacks`
    reporting the truth, the default mean-variance configuration showed 54
    fallbacks across ~60 rebalances -- it had effectively been equal weight.
    """
    import cvxpy as cp

    signal, cov = _make_signal_and_cov(100)
    sector_map = pd.Series(
        [f"SEC{i % 10}" for i in range(100)], index=signal.index,
    )
    constraints = PortfolioConstraints(
        max_weight=0.05, max_sector_exposure=0.30, max_turnover=0.30,
        risk_aversion=1.0,
    )
    optimizer = MeanVarianceOptimizer(constraints)

    first = optimizer.optimize(signal, cov, sector_map=sector_map)
    optimizer.optimize(
        signal.abs()[::-1], cov, prev_weights=first, sector_map=sector_map,
    )

    assert any(
        solver in cp.installed_solvers()
        for solver in ("CLARABEL", "ECOS", "SCS")
    )
    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason


def test_risk_parity_basic():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints()
    optimizer = RiskParityOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.sum() > 0
    assert (weights >= -0.01).all()


def test_optimizer_with_prev_weights():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(max_turnover=0.30)
    optimizer = EqualWeightOptimizer(constraints)

    prev = pd.Series(0.0, index=signal.index)
    prev.iloc[:10] = 0.1

    weights = optimizer.optimize(signal, cov, prev_weights=prev)
    assert weights.sum() > 0


def test_turnover_counts_names_entering_and_leaving_the_portfolio():
    """The old clamp aligned on the intersection, so a complete swap of the
    portfolio looked like zero turnover."""
    assets = [f"S{i:02d}" for i in range(20)]
    signal = pd.Series(1.0, index=assets)
    constraints = PortfolioConstraints(max_weight=0.10, max_turnover=0.20)
    prev = pd.Series(0.0, index=assets)
    prev.iloc[10:] = 0.10  # hold exactly the names the optimizer will drop

    weights = EqualWeightOptimizer(constraints).optimize(signal, prev_weights=prev)

    index = weights.index.union(prev.index)
    turnover = (
        weights.reindex(index, fill_value=0.0) - prev.reindex(index, fill_value=0.0)
    ).abs().sum() / 2
    assert turnover <= 0.20 + 1e-9, f"turnover {turnover:.3f} exceeded the cap"


def test_equal_weight_never_scales_above_max_weight():
    """Renormalizing a too-narrow candidate set used to undo the cap."""
    assets = [f"S{i}" for i in range(5)]
    signal = pd.Series(1.0, index=assets)
    constraints = PortfolioConstraints(max_weight=0.10)

    weights = EqualWeightOptimizer(constraints).optimize(signal)

    assert weights.max() <= 0.10 + 1e-12
    # Five names cannot fill 100% under a 10% cap; the rest is cash.
    assert weights.sum() <= 0.5 + 1e-12


def test_fallback_is_counted_and_reason_is_recorded():
    signal = pd.Series([1.0], index=["ONLY"])
    optimizer = MeanVarianceOptimizer(PortfolioConstraints(max_weight=1.0))

    weights = optimizer.optimize(signal)

    assert weights.sum() > 0
    assert optimizer.fallback_count == 1
    assert "fewer than two" in (optimizer.last_fallback_reason or "")


def test_solver_failure_is_counted(monkeypatch):
    import cvxpy as cp

    signal, cov = _make_signal_and_cov(20)
    constraints = PortfolioConstraints(max_weight=0.10, max_turnover=0.30)
    optimizer = MeanVarianceOptimizer(constraints)

    def explode(self, *args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(cp.Problem, "solve", explode)

    weights = optimizer.optimize(signal, cov)

    assert weights.sum() > 0
    assert optimizer.fallback_count == 1
    assert "solver" in (optimizer.last_fallback_reason or "")


def test_risk_parity_is_not_silently_equal_weight():
    """The old objective was non-DCP and only tried the missing ECOS solver,
    so this optimizer always fell back to equal weight."""
    signal, cov = _make_signal_and_cov(60)
    constraints = PortfolioConstraints(max_weight=0.10)
    optimizer = RiskParityOptimizer(constraints)

    weights = optimizer.optimize(signal, cov)

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert weights.max() <= constraints.max_weight + 1e-6
    # Equal weights have zero dispersion; a real risk solution does not.
    assert weights.std() > 1e-4
