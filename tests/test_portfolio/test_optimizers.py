"""Tests for portfolio optimizers."""

import numpy as np
import pandas as pd
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
