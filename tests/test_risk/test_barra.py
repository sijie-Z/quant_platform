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


class TestBarraExposureBuilder:
    """The production endpoint used to pass the platform's factor panel
    straight into `fit()`. None of its names (`momentum_12m`,
    `log_market_cap`, ...) matched the Barra names (`momentum`, `size`, ...),
    so all ten exposures fell back to zero and the decomposition reported
    zero factor risk for every portfolio. These tests pin the translation."""

    @pytest.fixture
    def market_data(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2023-01-02", periods=80)
        assets = [f"S{i:02d}" for i in range(40)]

        returns = pd.DataFrame(
            rng.normal(0.0, 0.01, (len(dates), len(assets))),
            index=dates, columns=assets,
        )
        prices = 10.0 * (1.0 + returns).cumprod()
        benchmark = returns.mean(axis=1)

        platform_factors = {}
        for name in (
            "log_market_cap", "pb_ratio", "momentum_12m", "volatility_60d",
            "roe", "asset_growth", "turnover_20d",
        ):
            platform_factors[name] = pd.DataFrame(
                rng.normal(size=(len(dates), len(assets))),
                index=dates, columns=assets,
            )

        index = pd.MultiIndex.from_product(
            [dates, assets], names=["date", "asset"],
        )
        financials = pd.DataFrame({
            "total_assets": rng.uniform(1e9, 5e9, len(index)),
            "net_assets": rng.uniform(1e8, 1e9, len(index)),
        }, index=index)

        return prices, returns, benchmark, financials, platform_factors

    def test_builds_non_zero_panel_for_every_factor(self, market_data):
        prices, returns, benchmark, financials, platform_factors = market_data
        model = BarraModel()

        exposures = model.build_exposures(
            prices, returns,
            benchmark=benchmark,
            financials=financials,
            platform_factors=platform_factors,
        )

        assert set(exposures) == set(BARRA_FACTORS)
        for name in BARRA_FACTORS:
            panel = exposures[name]
            assert panel.shape == returns.shape
            assert np.isfinite(panel.to_numpy()).all()
            assert panel.abs().to_numpy().sum() > 0, f"{name} exposure is all zero"

    def test_value_proxy_is_negated_price_to_book(self, market_data):
        prices, returns, benchmark, financials, platform_factors = market_data
        model = BarraModel()

        exposures = model.build_exposures(
            prices, returns,
            benchmark=benchmark,
            financials=financials,
            platform_factors=platform_factors,
        )

        correlation = exposures["value"].stack().corr(platform_factors["pb_ratio"].stack())
        assert correlation < 0, "Barra value is book-to-price, pb_ratio is price-to-book"

    def test_fitted_model_reports_non_zero_factor_risk(self, market_data):
        prices, returns, benchmark, financials, platform_factors = market_data
        model = BarraModel()
        exposures = model.build_exposures(
            prices, returns,
            benchmark=benchmark,
            financials=financials,
            platform_factors=platform_factors,
        )
        model.fit(exposures, returns)

        # A concentrated portfolio: equal weight over the whole universe has
        # zero net exposure to every cross-sectionally standardized factor.
        weights = pd.Series(0.0, index=returns.columns)
        weights.iloc[:10] = 0.1
        decomposition = model.decompose_risk(weights, exposures)

        assert decomposition.factor_risk > 0
        assert decomposition.r_squared > 0
        assert sum(abs(v) for v in decomposition.factor_exposures.values()) > 0

    def test_universe_equal_weight_has_no_net_factor_exposure(self, market_data):
        """Documents why a whole-universe equal-weight portfolio decomposes
        to zero factor risk: every exposure is cross-sectionally z-scored to
        zero mean, so the portfolio average is zero by construction."""
        prices, returns, benchmark, financials, platform_factors = market_data
        model = BarraModel()
        exposures = model.build_exposures(
            prices, returns,
            benchmark=benchmark,
            financials=financials,
            platform_factors=platform_factors,
        )
        model.fit(exposures, returns)

        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        decomposition = model.decompose_risk(weights, exposures)
        assert decomposition.factor_risk == pytest.approx(0.0, abs=1e-12)

    def test_specific_risk_is_per_asset(self, market_data):
        prices, returns, benchmark, financials, platform_factors = market_data
        model = BarraModel()
        exposures = model.build_exposures(
            prices, returns,
            benchmark=benchmark,
            financials=financials,
            platform_factors=platform_factors,
        )
        model.fit(exposures, returns)

        specific = model.specific_risk.reindex(returns.columns)
        assert specific.notna().all()
        assert specific.std() > 0, "every asset got the same specific risk"
