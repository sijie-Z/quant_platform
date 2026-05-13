"""Tests for factor orthogonalization."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.factors.orthogonalization import (
    FactorOrthogonalizer,
    OrthMethod,
    OrthResult,
    orthogonalize_factors,
)


@pytest.fixture
def correlated_factors():
    """Generate correlated factors for testing."""
    rng = np.random.default_rng(42)
    n_dates = 100
    n_assets = 50
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    assets = [f"{i:06d}.SH" for i in range(1, n_assets + 1)]

    # Create a common signal
    common = rng.normal(0, 1, size=(n_dates, n_assets))

    # Factor A = common + noise
    factor_a = common + rng.normal(0, 0.3, size=(n_dates, n_assets))
    # Factor B = common + noise (correlated with A)
    factor_b = common + rng.normal(0, 0.3, size=(n_dates, n_assets))
    # Factor C = independent
    factor_c = rng.normal(0, 1, size=(n_dates, n_assets))

    return {
        "momentum": pd.DataFrame(factor_a, index=dates, columns=assets),
        "value": pd.DataFrame(factor_b, index=dates, columns=assets),
        "quality": pd.DataFrame(factor_c, index=dates, columns=assets),
    }


@pytest.fixture
def uncorrelated_factors():
    """Generate uncorrelated factors."""
    rng = np.random.default_rng(123)
    n_dates = 100
    n_assets = 50
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    assets = [f"{i:06d}.SH" for i in range(1, n_assets + 1)]

    return {
        "f1": pd.DataFrame(rng.normal(0, 1, (n_dates, n_assets)), index=dates, columns=assets),
        "f2": pd.DataFrame(rng.normal(0, 1, (n_dates, n_assets)), index=dates, columns=assets),
        "f3": pd.DataFrame(rng.normal(0, 1, (n_dates, n_assets)), index=dates, columns=assets),
    }


class TestGramSchmidt:
    """Test Gram-Schmidt orthogonalization."""

    def test_output_shape(self, correlated_factors):
        """Output should have same shape as input."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.GRAM_SCHMIDT)

        for name in correlated_factors:
            assert result.factors[name].shape == correlated_factors[name].shape

    def test_first_factor_preserved(self, correlated_factors):
        """First factor in priority should be preserved (up to scaling)."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(
            correlated_factors,
            method=OrthMethod.GRAM_SCHMIDT,
            priority_order=["momentum", "value", "quality"],
        )

        # First factor should be correlated with original (rank correlation > 0.9)
        orig = correlated_factors["momentum"].iloc[0].rank()
        orth_f = result.factors["momentum"].iloc[0].rank()
        corr = orig.corr(orth_f)
        assert corr > 0.9

    def test_reduces_correlation(self, correlated_factors):
        """Orthogonalization should reduce correlation between factors."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.GRAM_SCHMIDT)

        # Max off-diagonal correlation should be lower after
        before = result.correlation_before.values
        after = result.correlation_after.values

        n = before.shape[0]
        mask = np.triu_indices(n, k=1)
        max_before = np.max(np.abs(before[mask]))
        max_after = np.max(np.abs(after[mask]))

        assert max_after < max_before

    def test_returns_transform_matrix(self, correlated_factors):
        """Should return a transformation matrix."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.GRAM_SCHMIDT)

        assert result.transform_matrix is not None
        assert result.transform_matrix.shape[0] == result.transform_matrix.shape[1]


class TestPCA:
    """Test PCA orthogonalization."""

    def test_output_shape(self, correlated_factors):
        """Output factors should have same number of dates as input."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(
            correlated_factors, method=OrthMethod.PCA, n_components=3,
        )

        n_dates = len(list(correlated_factors.values())[0])
        for name in result.factors:
            assert len(result.factors[name]) == n_dates

    def test_variance_explained(self, correlated_factors):
        """Should return variance explained per component."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.PCA)

        assert result.variance_explained is not None
        assert len(result.variance_explained) == 3
        # Variance should sum to ~1.0
        assert abs(result.variance_explained.sum() - 1.0) < 0.01

    def test_n_components_with_threshold(self, correlated_factors):
        """Should select components based on variance threshold."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(
            correlated_factors,
            method=OrthMethod.PCA,
            variance_threshold=0.8,
        )

        # With 3 factors and 0.8 threshold, should need 2 components
        assert result.n_components <= 3
        assert result.n_components >= 1

    def test_n_components_fixed(self, correlated_factors):
        """Fixed n_components should override threshold."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(
            correlated_factors,
            method=OrthMethod.PCA,
            n_components=2,
        )

        assert result.n_components == 2

    def test_pca_scores_are_uncorrelated(self, correlated_factors):
        """PCA scores should be uncorrelated."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(
            correlated_factors, method=OrthMethod.PCA, n_components=3,
        )

        # After orthogonalization, off-diagonal correlations should be ~0
        after = result.correlation_after.values
        n = after.shape[0]
        mask = np.triu_indices(n, k=1)
        max_corr = np.max(np.abs(after[mask]))
        assert max_corr < 0.15  # PCA scores are uncorrelated by construction


class TestSymmetric:
    """Test symmetric orthogonalization."""

    def test_output_shape(self, correlated_factors):
        """Output should have same shape as input."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.SYMMETRIC)

        for name in correlated_factors:
            assert result.factors[name].shape == correlated_factors[name].shape

    def test_reduces_correlation(self, correlated_factors):
        """Symmetric orth should reduce correlation."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.SYMMETRIC)

        before = result.correlation_before.values
        after = result.correlation_after.values
        n = before.shape[0]
        mask = np.triu_indices(n, k=1)
        max_before = np.max(np.abs(before[mask]))
        max_after = np.max(np.abs(after[mask]))

        assert max_after < max_before

    def test_preserves_all_factors(self, correlated_factors):
        """All factors should remain present (not dropped)."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors, method=OrthMethod.SYMMETRIC)

        assert set(result.factors.keys()) == set(correlated_factors.keys())

    def test_already_uncorrelated(self, uncorrelated_factors):
        """Uncorrelated factors should stay roughly the same."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(uncorrelated_factors, method=OrthMethod.SYMMETRIC)

        # Correlation should remain low
        after = result.correlation_after.values
        n = after.shape[0]
        mask = np.triu_indices(n, k=1)
        max_corr = np.max(np.abs(after[mask]))
        assert max_corr < 0.3


class TestOrthogonalizeFactors:
    """Test the convenience function."""

    def test_returns_dict(self, correlated_factors):
        """Should return dict of DataFrames."""
        result = orthogonalize_factors(correlated_factors, method="symmetric")
        assert isinstance(result, dict)
        for name, df in result.items():
            assert isinstance(df, pd.DataFrame)

    def test_all_methods_work(self, correlated_factors):
        """All three methods should run without error."""
        for method in ["gram_schmidt", "pca", "symmetric"]:
            result = orthogonalize_factors(correlated_factors, method=method)
            assert len(result) >= 2  # PCA may return fewer factors


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_factor_raises(self):
        """Should raise with only one factor."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=50)
        assets = [f"{i:06d}.SH" for i in range(1, 11)]
        factors = {
            "only": pd.DataFrame(rng.normal(0, 1, (50, 10)), index=dates, columns=assets),
        }

        orth = FactorOrthogonalizer()
        with pytest.raises(ValueError, match="at least 2"):
            orth.orthogonalize(factors)

    def test_two_factors_work(self):
        """Should work with exactly 2 factors."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-01", periods=50)
        assets = [f"{i:06d}.SH" for i in range(1, 11)]
        factors = {
            "a": pd.DataFrame(rng.normal(0, 1, (50, 10)), index=dates, columns=assets),
            "b": pd.DataFrame(rng.normal(0, 1, (50, 10)), index=dates, columns=assets),
        }

        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(factors)
        assert len(result.factors) == 2

    def test_result_has_correlation_matrices(self, correlated_factors):
        """Result should include before/after correlation matrices."""
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(correlated_factors)

        assert isinstance(result.correlation_before, pd.DataFrame)
        assert isinstance(result.correlation_after, pd.DataFrame)
        assert result.correlation_before.shape == (3, 3)
        assert result.correlation_after.shape == (3, 3)
