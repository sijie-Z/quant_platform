"""Tests for FactorValidator — Deflated Sharpe Ratio and multiple testing correction."""


import numpy as np
import pandas as pd
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

    def test_annualized_and_per_period_scales_are_not_mixed(self):
        """Regression for the scale bug.

        The old code compared an annualized observed Sharpe with the
        unit-variance order statistic sqrt(2 ln N) -- no sqrt(T) scaling --
        and put the annualized SR into a standard-error formula that expects
        a per-period SR. A genuine annualized SR of 2.0 over 1260 days with
        10 trials was reported as p=0.59.
        """
        validator = FactorValidator(significance_level=0.05)
        rng = np.random.default_rng(0)
        daily_sr = 2.0 / np.sqrt(252)
        # Twenty years: sampling noise on the realized Sharpe is small enough
        # that the test measures the formula, not one draw.
        returns = rng.normal(daily_sr * 0.01, 0.01, size=5040)

        result = validator.deflated_sharpe_test(returns, num_trials=10)

        assert result.significant is True
        assert result.deflated_p_value < 0.01
        # The reported expected max is annualized, like the observed Sharpe.
        assert result.expected_max_sharpe > 0
        assert result.expected_max_sharpe < result.observed_sharpe

    def test_a_short_sample_fluke_is_not_significant(self):
        """High annualized Sharpe over a short sample with many trials.

        The same defect made 5.0 annualized over 126 days with 100 trials
        look significant (p≈0) when the correct DSR is around 0.2-0.3.
        """
        validator = FactorValidator(significance_level=0.05)
        rng = np.random.default_rng(1)
        daily_sr = 5.0 / np.sqrt(252)
        returns = rng.normal(daily_sr * 0.01, 0.01, size=126)

        result = validator.deflated_sharpe_test(returns, num_trials=100)

        assert result.significant is False


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


class TestSignificanceRequiresBothCorrections:
    """R12: significant used to follow the BH mask alone, ignoring the DSR
    deflated p-value. The audit probe had raw_p=0.0319 (passes BH) but
    deflated_p=0.0911 (fails alpha), and both factors were returned as
    significant.
    """

    @staticmethod
    def _fake_result(raw_p, deflated_p):
        from quant_platform.research.validation import DeflatedSharpeResult

        return DeflatedSharpeResult(
            observed_sharpe=1.0, expected_max_sharpe=0.5,
            deflated_p_value=deflated_p, num_trials=2,
            skew=0.0, kurtosis=0.0, significant=True, raw_p_value=raw_p,
        )

    def test_bh_pass_with_dsr_fail_is_not_significant(self, monkeypatch):
        validator = FactorValidator(significance_level=0.05)
        monkeypatch.setattr(
            validator, "deflated_sharpe_test",
            lambda rets, num_trials=1: self._fake_result(0.031874, 0.091072),
        )
        results = validator.validate_factor_batch({"f1": None, "f2": None})
        for result in results.values():
            # BH rejects both raw p-values at m=2 (0.0319 <= 0.05) ...
            assert result.bh_significant is True
            # ... but the deflated p-value does not clear alpha.
            assert result.dsr_significant is False
            assert result.significant is False

    def test_dsr_pass_with_bh_fail_is_not_significant(self, monkeypatch):
        validator = FactorValidator(significance_level=0.05)
        monkeypatch.setattr(
            validator, "deflated_sharpe_test",
            lambda rets, num_trials=1: self._fake_result(0.001, 0.01),
        )
        # Force the BH mask to reject only the first factor.
        monkeypatch.setattr(
            validator, "multiple_testing_correction",
            lambda p, method="bh": np.array([True, False]),
        )
        results = validator.validate_factor_batch({"f1": None, "f2": None})
        assert results["f1"].significant is True
        assert results["f1"].dsr_significant is True
        assert results["f2"].significant is False

    def test_single_factor_uses_dsr_only(self):
        validator = FactorValidator(significance_level=0.05)
        rng = np.random.default_rng(3)
        returns = rng.normal(0.001, 0.01, size=500)
        results = validator.validate_factor_batch({"only": returns})
        result = results["only"]
        assert result.bh_significant is None
        assert result.dsr_significant == result.significant


class TestNonFiniteInputsAndAnnualization:
    """R13: inf returns produced nan/nan silently; annualization was fixed
    at sqrt(252) regardless of the actual return frequency.
    """

    def test_non_finite_returns_are_dropped(self):
        validator = FactorValidator()
        rng = np.random.default_rng(11)
        clean = rng.normal(0.001, 0.01, size=300)
        # Append the non-finite values so filtering restores exactly `clean`.
        with_inf = np.concatenate([clean, [np.inf, -np.inf]])

        base = validator.deflated_sharpe_test(clean, num_trials=1)
        noisy = validator.deflated_sharpe_test(with_inf, num_trials=1)

        assert np.isfinite(noisy.observed_sharpe)
        assert noisy.observed_sharpe == pytest.approx(base.observed_sharpe, abs=1e-9)
        assert noisy.deflated_p_value == pytest.approx(base.deflated_p_value, abs=1e-9)

    def test_periods_per_year_scales_the_annualized_sharpe(self):
        validator = FactorValidator()
        rng = np.random.default_rng(12)
        monthly = rng.normal(0.005, 0.04, size=240)
        daily_result = validator.deflated_sharpe_test(monthly, periods_per_year=252)
        monthly_result = validator.deflated_sharpe_test(monthly, periods_per_year=12)

        per_period = monthly_result.observed_sharpe / np.sqrt(12)
        assert daily_result.observed_sharpe == pytest.approx(
            per_period * np.sqrt(252), abs=1e-5
        )
        # Same per-period Sharpe -> the 12x basis reports 12x/252x lower
        # annualized numbers, not the hardcoded 252 ones.
        assert monthly_result.observed_sharpe < daily_result.observed_sharpe

    def test_invalid_periods_per_year_raises(self):
        validator = FactorValidator()
        for bad in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="periods_per_year"):
                validator.deflated_sharpe_test(np.zeros(100), periods_per_year=bad)


class TestValidateSweepTrials:
    """Sweep-level helper: DSR deflation at N trials + BH on raw p-values."""

    def test_deflated_p_grows_with_num_trials(self):
        from quant_platform.research.validation import validate_sweep_trials

        rng = np.random.default_rng(7)
        returns = rng.normal(0.0008, 0.01, size=500)

        single = validate_sweep_trials({"only": returns})
        many = validate_sweep_trials({f"cfg{i}": returns for i in range(20)})

        assert single["num_trials"] == 1
        assert many["num_trials"] == 20
        assert many["trials"]["cfg0"]["dsr_p"] > single["trials"]["only"]["dsr_p"]
        # The observed Sharpe and the raw p-value do not depend on N; only
        # the E[max SR] deflation changes.
        assert many["trials"]["cfg0"]["observed_sharpe"] == pytest.approx(
            single["trials"]["only"]["observed_sharpe"]
        )
        assert many["trials"]["cfg0"]["raw_p"] == pytest.approx(
            single["trials"]["only"]["raw_p"]
        )

    def test_bh_mask_on_known_p_values_matches_hand_computation(self):
        validator = FactorValidator(significance_level=0.05)

        # m=4, thresholds (rank/m)*alpha = .0125/.025/.0375/.05:
        # .01<=.0125, .02<=.025, .03<=.0375, .2>.05 -> first three rejected.
        mask = validator.multiple_testing_correction(
            [0.01, 0.02, 0.03, 0.2], method="bh",
        )
        assert mask.tolist() == [True, True, True, False]

        # m=3, thresholds .0167/.0333/.05. The third p-value passes and BH's
        # step-up rule therefore also rejects the second (0.04 > 0.0333).
        mask = validator.multiple_testing_correction(
            [0.01, 0.04, 0.045], method="bh",
        )
        assert mask.tolist() == [True, True, True]

    def test_empty_input(self):
        from quant_platform.research.validation import validate_sweep_trials

        result = validate_sweep_trials({})
        assert result == {"num_trials": 0, "trials": {}, "best": None}

    def test_best_is_by_observed_sharpe_and_requires_both_corrections(self):
        from quant_platform.research.validation import validate_sweep_trials

        rng = np.random.default_rng(8)
        strong = rng.normal(0.002, 0.01, size=500)
        weak = rng.normal(0.0001, 0.01, size=500)

        result = validate_sweep_trials({"strong": strong, "weak": weak})

        best = result["best"]
        assert best["config"] == "strong"
        assert best["observed"] == result["trials"]["strong"]["observed_sharpe"]
        assert best["significant"] is True
        assert best["dsr_significant"] is True
        assert best["bh_significant"] is True

    def test_non_finite_and_too_short_series_never_significant(self):
        from quant_platform.research.validation import validate_sweep_trials

        rng = np.random.default_rng(9)
        with_nan = rng.normal(0.0, 0.01, size=100)
        with_nan[::3] = np.nan
        with_inf = np.concatenate([with_nan, [np.inf, -np.inf]])
        too_short = np.array([0.01, -0.02, 0.03, np.nan])

        result = validate_sweep_trials({"nan": with_inf, "short": too_short})

        assert result["num_trials"] == 2
        for name in ("nan", "short"):
            assert result["trials"][name]["dsr_significant"] is False
            assert result["trials"][name]["dsr_p"] > 0.05
            assert np.isfinite(result["trials"][name]["observed_sharpe"])
        # A series with fewer than 20 finite observations is the documented
        # short-circuit: p=1, observed Sharpe reported as 0.
        assert result["trials"]["short"]["dsr_p"] == 1.0
        assert result["best"]["significant"] is False


class TestWalkForwardSelectorLayering:
    """R28: the selector lives in CORE so lab/tools can import it.

    ``tests/test_architecture/test_core_boundaries.py`` forbids lab -> live
    imports, so ``tools/real_data_research.py`` cannot import the strategy
    package. The implementation moved to ``research.validation``; the
    strategy module must keep re-exporting the *same* function object so
    existing callers and monkeypatches do not break.
    """

    def test_strategy_reexports_the_core_function(self):
        from quant_platform.research import validation as research_validation
        from quant_platform.strategy import research_validation as strategy_rv

        assert (
            strategy_rv.select_walk_forward_factors
            is research_validation.select_walk_forward_factors
        )

    def test_core_selector_selects_significant_factors(self):
        from quant_platform.factors.evaluation import ic_summary, rank_ic
        from quant_platform.research.validation import select_walk_forward_factors

        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2022-01-03", periods=160)
        assets = [f"S{i:02d}" for i in range(25)]
        base = rng.normal(size=(160, 25))
        returns = pd.DataFrame(base, index=dates, columns=assets)
        strong = pd.DataFrame(
            base + rng.normal(scale=0.8, size=returns.shape),
            index=dates, columns=assets,
        )
        noise = pd.DataFrame(
            rng.normal(size=returns.shape), index=dates, columns=assets,
        )
        factors = {"strong": strong, "noise": noise}
        ic_results = {
            name: ic_summary(rank_ic(panel, returns))
            for name, panel in factors.items()
        }

        result = select_walk_forward_factors(ic_results, factors, returns)

        assert result["selected"] == ["strong"]
        assert result["fallback"] is False
