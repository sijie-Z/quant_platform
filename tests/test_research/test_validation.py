"""Tests for FactorValidator — Deflated Sharpe Ratio and multiple testing correction."""


import numpy as np
import pytest

from quant_platform.research.validation import FactorValidator


class TestDeflatedSharpeTest:
    """Deflated Sharpe Ratio test behavior."""

    def test_high_sharpe_single_trial(self):
        """A high Sharpe from a single trial (N=1) should be significant."""
        validator = FactorValidator(significance_level=0.05)
        rng = np.random.default_rng(42)
        # Generate returns with a clear positive drift
        returns = rng.normal(0.001, 0.01, size=500)  # ~2.5 Sharpe annualized
        result = validator.deflated_sharpe_test(returns, num_trials=1)

        assert result.observed_sharpe > 0
        assert result.significant is True
        assert result.deflated_p_value < 0.05

    def test_high_sharpe_many_trials_not_significant(self):
        """A moderate Sharpe selected from many trials should not be significant."""
        validator = FactorValidator(significance_level=0.05)
        rng = np.random.default_rng(42)
        # Moderate Sharpe (~1.0 annualized)
        returns = rng.normal(0.0004, 0.01, size=500)
        result = validator.deflated_sharpe_test(returns, num_trials=1000)

        # With 1000 trials, E[max SR] ≈ √(2 ln 1000) ≈ 3.72
        # A Sharpe of ~1.0 is not significant against that
        assert result.observed_sharpe < result.expected_max_sharpe
        assert result.significant is False

    def test_zero_returns(self):
        """All-zero returns should give zero Sharpe, not significant."""
        validator = FactorValidator()
        returns = np.zeros(100)
        result = validator.deflated_sharpe_test(returns, num_trials=1)

        assert result.observed_sharpe == 0.0
        assert result.significant is False

    def test_insufficient_data(self):
        """Too few observations should return not significant."""
        validator = FactorValidator()
        returns = np.array([0.01, -0.01, 0.02])
        result = validator.deflated_sharpe_test(returns, num_trials=1)

        assert result.significant is False

    def test_skew_and_kurtosis(self):
        """Should correctly compute higher moments."""
        validator = FactorValidator()
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=1000)
        result = validator.deflated_sharpe_test(returns, num_trials=1)

        # Normal distribution: skew ≈ 0, excess kurtosis ≈ 0
        assert abs(result.skew) < 0.5
        assert abs(result.kurtosis) < 1.0

    def test_expected_max_sharpe_grows_with_trials(self):
        """E[max SR] should increase with more trials."""
        validator = FactorValidator()
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=500)

        r1 = validator.deflated_sharpe_test( returns, num_trials=10)
        r2 = validator.deflated_sharpe_test(returns, num_trials=100)

        assert r2.expected_max_sharpe > r1.expected_max_sharpe

    def test_custom_expected_max_sharpe(self):
        """Can provide custom expected_max_sharpe."""
        validator = FactorValidator()
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, size=500)

        result = validator.deflated_sharpe_test(
            returns, expected_max_sharpe=0.0,
        )
        # With E[max SR] = 0, any positive Sharpe should be significant
        assert result.significant is True


class TestMultipleTestingCorrection:
    """Benjamini-Hochberg and Bonferroni correction."""

    def test_bh_all_significant(self):
        """All small p-values should remain significant under BH."""
        validator = FactorValidator(significance_level=0.05)
        p_values = [0.001, 0.002, 0.003, 0.004]
        result = validator.multiple_testing_correction(p_values, method="bh")

        assert result.all()

    def test_bh_none_significant(self):
        """All large p-values should not be significant under BH."""
        validator = FactorValidator(significance_level=0.05)
        p_values = [0.5, 0.6, 0.7, 0.8]
        result = validator.multiple_testing_correction(p_values, method="bh")

        assert not result.any()

    def test_bh_mixed(self):
        """BH should reject some but not all with mixed p-values."""
        validator = FactorValidator(significance_level=0.05)
        p_values = [0.001, 0.01, 0.03, 0.5, 0.9]
        result = validator.multiple_testing_correction(p_values, method="bh")

        # At least the smallest should be rejected
        assert result[0] is np.True_

    def test_bonferroni_conservative(self):
        """Bonferroni should be more conservative than BH."""
        validator = FactorValidator(significance_level=0.05)
        p_values = [0.01, 0.02, 0.03, 0.04]

        bh = validator.multiple_testing_correction(p_values, method="bh")
        bonf = validator.multiple_testing_correction(p_values, method="bonferroni")

        # Bonferroni rejects fewer (more conservative)
        assert bonf.sum() <= bh.sum()

    def test_bonferroni_threshold(self):
        """Bonferroni threshold is alpha / n."""
        validator = FactorValidator(significance_level=0.05)
        # With 3 tests, adjusted p = p * 3. Reject if p * 3 < 0.05.
        p_values = [0.01, 0.02, 0.1]
        result = validator.multiple_testing_correction(p_values, method="bonferroni")

        # 0.01 * 3 = 0.03 < 0.05 → reject
        # 0.02 * 3 = 0.06 > 0.05 → keep
        # 0.1 * 3 = 0.3 > 0.05 → keep
        assert result[0] is np.True_
        assert result[1] is np.False_

    def test_empty_input(self):
        """Empty input should return empty array."""
        validator = FactorValidator()
        result = validator.multiple_testing_correction([])
        assert len(result) == 0

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        validator = FactorValidator()
        with pytest.raises(ValueError, match="Unknown method"):
            validator.multiple_testing_correction([0.01], method="unknown")


class TestValidateFactorBatch:
    """Batch factor validation with multiple testing correction."""

    def test_batch_returns_results(self):
        """Should return results for all factors."""
        validator = FactorValidator()
        rng = np.random.default_rng(42)
        factor_returns = {
            "momentum": rng.normal(0.001, 0.01, size=500),
            "value": rng.normal(0.0005, 0.01, size=500),
            "noise": rng.normal(0, 0.01, size=500),
        }
        results = validator.validate_factor_batch(factor_returns)

        assert len(results) == 3
        assert all(k in results for k in ["momentum", "value", "noise"])

    def test_batch_multiple_testing_applied(self):
        """Batch validation should apply BH correction."""
        validator = FactorValidator(significance_level=0.05)
        rng = np.random.default_rng(42)
        # Many noise factors — should be filtered out
        factor_returns = {
            f"noise_{i}": rng.normal(0, 0.01, size=200)
            for i in range(20)
        }
        results = validator.validate_factor_batch(factor_returns, num_trials=20)

        # With 20 noise factors and BH correction, most should not be significant
        n_significant = sum(1 for r in results.values() if r.significant)
        assert n_significant < 10  # Much less than 20
