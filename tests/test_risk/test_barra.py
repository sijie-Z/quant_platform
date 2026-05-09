"""Tests for Barra multi-factor risk model."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.risk.barra import (
    BARRA_FACTORS,
    BarraFactorReturn,
    BarraModel,
    BarraRiskDecomposition,
)


@pytest.fixture
def sample_barra_data():
    """Create sample data for Barra model testing."""
    np.random.seed(42)
    n_dates, n_assets = 100, 50
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    assets = [f"stock_{i:03d}" for i in range(n_assets)]

    # Factor exposures
    factor_exposures = {}
    for name in BARRA_FACTORS:
        data = np.random.randn(n_dates, n_assets)
        factor_exposures[name] = pd.DataFrame(data, index=dates, columns=assets)

    # Returns: weighted sum of factor exposures + noise
    weights = np.random.randn(len(BARRA_FACTORS)) * 0.01
    returns_data = np.zeros((n_dates, n_assets))
    for j, name in enumerate(BARRA_FACTORS):
        returns_data += weights[j] * factor_exposures[name].values
    returns_data += np.random.randn(n_dates, n_assets) * 0.02
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)

    return factor_exposures, returns


@pytest.fixture
def portfolio_weights():
    """Create sample portfolio weights."""
    np.random.seed(42)
    assets = [f"stock_{i:03d}" for i in range(50)]
    w = np.random.dirichlet(np.ones(50))
    return pd.Series(w, index=assets)


class TestBarraFactorReturn:
    def test_dataclass(self):
        fr = BarraFactorReturn(
            date="2024-01-01",
            factor_returns={"size": 0.01, "value": -0.005},
            r_squared=0.45,
            n_assets=100,
        )
        assert fr.date == "2024-01-01"
        assert fr.r_squared == 0.45
        assert fr.n_assets == 100


class TestBarraRiskDecomposition:
    def test_dataclass(self):
        rd = BarraRiskDecomposition(
            total_risk=0.15,
            factor_risk=0.12,
            specific_risk=0.08,
            r_squared=0.64,
        )
        assert rd.total_risk == 0.15
        assert rd.r_squared == 0.64


class TestBarraModel:
    def test_default_factors(self):
        model = BarraModel()
        assert len(model.factor_names) == 10
        assert model.factor_names == BARRA_FACTORS

    def test_custom_factors(self):
        model = BarraModel(factor_names=["size", "value"])
        assert model.factor_names == ["size", "value"]

    def test_cross_sectional_regression(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()

        # Build exposure matrix for first date
        date = returns.index[0]
        exposure_rows = {}
        for asset in returns.columns:
            row = []
            for name in BARRA_FACTORS:
                row.append(factor_exposures[name].loc[date, asset])
            exposure_rows[asset] = row

        exposures = pd.DataFrame(
            exposure_rows.values(),
            index=exposure_rows.keys(),
            columns=BARRA_FACTORS,
        )

        daily_returns = returns.loc[date]
        result = model._cross_sectional_regression(daily_returns, exposures)

        assert isinstance(result, BarraFactorReturn)
        assert len(result.factor_returns) == 10
        assert result.n_assets == 50
        assert 0 <= result.r_squared <= 1

    def test_fit(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        assert model._fitted is True
        assert len(model.factor_returns_history) == 100
        assert model.factor_covariance is not None
        assert model.factor_covariance.shape == (10, 10)
        assert model.specific_risk is not None

    def test_factor_covariance_symmetric(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        cov = model.factor_covariance
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_factor_covariance_positive_semidefinite(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        eigenvalues = np.linalg.eigvalsh(model.factor_covariance)
        assert np.all(eigenvalues >= -1e-10)

    def test_decompose_risk(self, sample_barra_data, portfolio_weights):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        rd = model.decompose_risk(portfolio_weights, factor_exposures)

        assert isinstance(rd, BarraRiskDecomposition)
        assert rd.total_risk > 0
        assert rd.factor_risk >= 0
        assert rd.specific_risk >= 0
        assert 0 <= rd.r_squared <= 1
        assert len(rd.factor_contributions) == 10
        assert len(rd.factor_exposures) == 10

    def test_risk_components_add_up(self, sample_barra_data, portfolio_weights):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        rd = model.decompose_risk(portfolio_weights, factor_exposures)
        # total_var = factor_var + specific_var
        # total_risk = sqrt(total_var) * sqrt(252)
        # So total_risk^2/252 should equal factor_var + specific_var
        total_var = rd.total_risk ** 2 / 252
        factor_var = rd.factor_risk ** 2 / 252
        specific_var = rd.specific_risk ** 2 / 252
        np.testing.assert_allclose(total_var, factor_var + specific_var, rtol=1e-6)

    def test_get_factor_return_series(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        series = model.get_factor_return_series()
        assert isinstance(series, pd.DataFrame)
        assert series.shape == (100, 10)
        assert list(series.columns) == BARRA_FACTORS

    def test_get_factor_covariance_df(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        cov_df = model.get_factor_covariance_df()
        assert isinstance(cov_df, pd.DataFrame)
        assert cov_df.shape == (10, 10)
        assert list(cov_df.index) == BARRA_FACTORS

    def test_not_fitted_raises(self, portfolio_weights):
        model = BarraModel()
        with pytest.raises(RuntimeError):
            model.decompose_risk(portfolio_weights, {})

    def test_ledoit_wolf_shrinkage(self, sample_barra_data):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()

        # Build returns matrix
        n = len(returns.index)
        n_factors = len(BARRA_FACTORS)
        returns_matrix = np.random.randn(n, n_factors)
        weights = np.ones(n) / n

        shrinkage = model._ledoit_wolf_shrinkage(returns_matrix, weights)
        assert 0 <= shrinkage <= 1

    def test_empty_history(self):
        model = BarraModel()
        series = model.get_factor_return_series()
        assert series.empty

        cov_df = model.get_factor_covariance_df()
        assert cov_df.empty

    def test_decompose_risk_with_date(self, sample_barra_data, portfolio_weights):
        factor_exposures, returns = sample_barra_data
        model = BarraModel()
        model.fit(factor_exposures, returns)

        date = returns.index[50]
        rd = model.decompose_risk(portfolio_weights, factor_exposures, date=str(date))
        assert isinstance(rd, BarraRiskDecomposition)
        assert rd.total_risk > 0
