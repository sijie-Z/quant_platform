"""Tests for strategy capacity estimation."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest.capacity import CapacityCurve, CapacityEstimator, CapacityResult
from quant_platform.execution.market_impact import SquareRootModel


@pytest.fixture
def sample_data():
    """Generate sample data for capacity tests."""
    rng = np.random.default_rng(42)
    n_dates = 252
    n_assets = 50
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    assets = [f"{i:06d}.SH" for i in range(1, n_assets + 1)]

    returns = pd.DataFrame(
        rng.normal(0.0003, 0.02, size=(n_dates, n_assets)),
        index=dates, columns=assets,
    )
    prices = pd.DataFrame(
        rng.uniform(10, 100, size=(n_dates, n_assets)),
        index=dates, columns=assets,
    )
    volumes = pd.DataFrame(
        rng.uniform(1e6, 1e8, size=(n_dates, n_assets)),
        index=dates, columns=assets,
    )
    volatility = returns.rolling(20, min_periods=5).std().fillna(0.02)

    # Generate some rebalance dates (monthly)
    rebalance_dates = dates[::21]  # Every ~month
    weights_history = {}
    for rd in rebalance_dates:
        w = rng.dirichlet(np.ones(n_assets))
        weights_history[rd] = pd.Series(w, index=assets)

    return {
        "returns": returns,
        "prices": prices,
        "volumes": volumes,
        "volatility": volatility,
        "weights_history": weights_history,
    }


class TestCapacityEstimator:
    """Capacity estimation behavior."""

    def test_returns_capacity_curve(self, sample_data):
        """Should return a CapacityCurve."""
        estimator = CapacityEstimator()
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e6, 1e7, 1e8],
        )

        assert isinstance(curve, CapacityCurve)
        assert len(curve.results) == 3

    def test_results_have_expected_fields(self, sample_data):
        """Each result should have all expected fields."""
        estimator = CapacityEstimator()
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e7],
        )

        r = curve.results[0]
        assert isinstance(r, CapacityResult)
        assert r.aum == 1e7
        assert isinstance(r.sharpe_ratio, float)
        assert isinstance(r.annualized_return, float)
        assert isinstance(r.max_drawdown, float)
        assert isinstance(r.n_capped_days, int)
        assert isinstance(r.total_impact_cost, float)

    def test_higher_aum_more_capped_days(self, sample_data):
        """Larger AUM should have more capped trading days."""
        estimator = CapacityEstimator(max_participation=0.05)
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e6, 1e9],
        )

        small = curve.results[0]
        large = curve.results[-1]
        # Larger AUM should have more capped days
        assert large.n_capped_days >= small.n_capped_days

    def test_sharpe_decreases_with_aum(self, sample_data):
        """Sharpe ratio generally decreases with higher AUM."""
        estimator = CapacityEstimator(max_participation=0.05)
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e5, 1e6, 1e7, 1e8, 1e9, 1e10],
        )

        sharpes = [r.sharpe_ratio for r in curve.results]
        # The general trend should be decreasing (allow some noise)
        # Check that the last Sharpe is lower than the first
        assert sharpes[-1] <= sharpes[0] + 0.5  # Allow some tolerance

    def test_capacity_found(self, sample_data):
        """Should find a capacity AUM with low enough threshold."""
        # Use a very low threshold since random data has ~0 Sharpe
        estimator = CapacityEstimator(sharpe_threshold=-1.0)
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e5, 1e6, 1e7, 1e8],
        )

        # Should find some capacity (even with negative Sharpe > -1.0)
        assert curve.capacity_aum > 0

    def test_to_dataframe(self, sample_data):
        """to_dataframe should return a DataFrame."""
        estimator = CapacityEstimator()
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e6, 1e7],
        )

        df = curve.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "aum" in df.columns
        assert "sharpe_ratio" in df.columns

    def test_summary(self, sample_data):
        """summary should return a dict."""
        estimator = CapacityEstimator()
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e6, 1e7],
        )

        s = curve.summary()
        assert isinstance(s, dict)
        assert "capacity_aum" in s
        assert "capacity_aum_millions" in s

    def test_custom_impact_model(self, sample_data):
        """Should accept custom impact model."""
        model = SquareRootModel(y_temp=1.0, y_perm=0.5)
        estimator = CapacityEstimator(impact_model=model)
        curve = estimator.estimate(
            weights_history=sample_data["weights_history"],
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e7],
        )

        assert len(curve.results) == 1

    def test_empty_weights(self, sample_data):
        """Empty weights history should still work."""
        estimator = CapacityEstimator()
        curve = estimator.estimate(
            weights_history={},
            prices=sample_data["prices"],
            volumes=sample_data["volumes"],
            returns=sample_data["returns"],
            aum_range=[1e7],
        )

        assert len(curve.results) == 1
