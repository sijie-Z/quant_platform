"""Tests for risk.monte_carlo — Monte Carlo simulation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.risk.monte_carlo import MonteCarloSimulator


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.02, 500), index=dates)


class TestMonteCarloSimulator:
    def test_bootstrap_simulation(self, sample_returns):
        sim = MonteCarloSimulator(n_simulations=500, horizon_days=252)
        result = sim.bootstrap_simulation(sample_returns, block_size=21)
        assert result["method"] == "bootstrap"
        assert result["n_simulations"] == 500
        assert "terminal_value" in result
        assert "annual_return" in result
        assert "max_drawdown" in result
        assert "sharpe" in result

    def test_bootstrap_confidence_intervals(self, sample_returns):
        sim = MonteCarloSimulator(n_simulations=1000, horizon_days=252)
        result = sim.bootstrap_simulation(sample_returns)
        tv = result["terminal_value"]
        assert tv["p5"] < tv["p95"]
        assert tv["mean"] > 0

    def test_bootstrap_tail_probability(self, sample_returns):
        sim = MonteCarloSimulator(n_simulations=5000, horizon_days=252)
        result = sim.bootstrap_simulation(sample_returns)
        assert "prob_positive" in result["annual_return"]
        assert 0 <= result["annual_return"]["prob_positive"] <= 1

    def test_bootstrap_max_drawdown(self, sample_returns):
        sim = MonteCarloSimulator(n_simulations=500, horizon_days=252)
        result = sim.bootstrap_simulation(sample_returns)
        assert result["max_drawdown"]["mean"] <= 0
        assert result["max_drawdown"]["worst"] <= result["max_drawdown"]["mean"]

    def test_parametric_simulation(self, sample_returns):
        sim = MonteCarloSimulator(n_simulations=500, horizon_days=252)
        result = sim.parametric_simulation(sample_returns)
        assert result["method"] == "parametric"
        assert "fitted_distribution" in result
        assert result["fitted_distribution"]["type"] == "student_t"

    def test_parametric_terminal_value(self, sample_returns):
        sim = MonteCarloSimulator(n_simulations=500, horizon_days=252)
        result = sim.parametric_simulation(sample_returns)
        tv = result["terminal_value"]
        assert tv["p5"] < tv["p95"]
        assert tv["mean"] > 0
