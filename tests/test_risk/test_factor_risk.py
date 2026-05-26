"""Tests for risk.factor_risk — Factor risk decomposition."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.risk.factor_risk import (
    estimate_factor_betas,
    factor_contribution_summary,
    factor_risk_decomposition,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    assets = [f"stock_{i}" for i in range(50)]

    # Create correlated factor-returns structure
    factor_momentum = pd.DataFrame(
        np.random.normal(0, 1, (300, 50)), index=dates, columns=assets,
    )
    factor_value = pd.DataFrame(
        np.random.normal(0, 1, (300, 50)), index=dates, columns=assets,
    )

    # Returns driven by factors
    noise = np.random.normal(0, 0.01, (300, 50))
    returns = pd.DataFrame(
        0.3 * factor_momentum.values + 0.2 * factor_value.values + noise,
        index=dates, columns=assets,
    )

    factors = {"momentum": factor_momentum, "value": factor_value}
    portfolio_returns = returns.mean(axis=1)
    return returns, factors, portfolio_returns


class TestEstimateFactorBetas:
    def test_returns_betas(self, sample_data):
        returns, factors, _ = sample_data
        betas = estimate_factor_betas(returns, factors, window=100)
        assert "momentum" in betas
        assert "value" in betas
        assert isinstance(betas["momentum"], pd.Series)
        assert len(betas["momentum"]) > 0

    def test_beta_values_reasonable(self, sample_data):
        returns, factors, _ = sample_data
        betas = estimate_factor_betas(returns, factors, window=100)
        for name, beta_series in betas.items():
            assert beta_series.abs().max() < 10


class TestFactorRiskDecomposition:
    def test_decomposition(self, sample_data):
        returns, factors, portfolio_returns = sample_data
        betas = estimate_factor_betas(returns, factors, window=100)
        factor_returns = {name: f.mean(axis=1) for name, f in factors.items()}
        result = factor_risk_decomposition(portfolio_returns, betas, factor_returns)
        assert "factor_risk_share" in result
        assert "r_squared" in result
        # r_squared should be valid (may be NaN if variance is 0)
        if not np.isnan(result["r_squared"]):
            assert 0 <= result["r_squared"] <= 1

    def test_summary(self, sample_data):
        returns, factors, portfolio_returns = sample_data
        betas = estimate_factor_betas(returns, factors, window=100)
        factor_returns = {name: f.mean(axis=1) for name, f in factors.items()}
        result = factor_risk_decomposition(portfolio_returns, betas, factor_returns)
        summary = factor_contribution_summary(result)
        assert isinstance(summary, list)
        assert len(summary) > 0
        assert "factor" in summary[0]
