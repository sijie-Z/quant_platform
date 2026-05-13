"""Factor and strategy statistical validation.

Provides two core validation tools to prevent false discoveries in
factor research:

1. Deflated Sharpe Ratio (DSR): Given an observed Sharpe ratio from
   a strategy selected among N candidates, computes the probability
   that such a Sharpe would arise by chance. Accounts for multiple
   testing, non-normal returns, and finite sample bias.

2. Multiple Testing Correction: Benjamini-Hochberg FDR control and
   Bonferroni correction for simultaneous factor/parameter tests.

Reference:
- Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"
- Harvey, Liu & Zhu (2016): "...and the Cross-Section of Expected Returns"
- Benjamini & Hochberg (1995): "Controlling the False Discovery Rate"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeflatedSharpeResult:
    """Result of a Deflated Sharpe Ratio test."""
    observed_sharpe: float
    expected_max_sharpe: float   # E[max SR under null]
    deflated_p_value: float      # P(SR >= observed | null)
    num_trials: int              # Number of strategies tested
    skew: float                  # Return skewness
    kurtosis: float              # Return kurtosis (excess)
    significant: bool            # p < significance_level


class FactorValidator:
    """Statistical validation for factor/strategy significance.

    Usage:
        validator = FactorValidator(significance_level=0.05)

        # Test a single strategy
        result = validator.deflated_sharpe_test(strategy_returns, num_trials=100)
        if not result.significant:
            print("Strategy Sharpe is not significant after deflation")

        # Test multiple factors simultaneously
        pvals = [factor_pvalue_1, factor_pvalue_2, ...]
        significant = validator.multiple_testing_correction(pvals, method='bh')
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def deflated_sharpe_test(
        self,
        returns: np.ndarray,
        num_trials: int = 1,
        expected_max_sharpe: float | None = None,
    ) -> DeflatedSharpeResult:
        """Test if an observed Sharpe ratio is significant after accounting
        for multiple testing (the "deflated" Sharpe ratio).

        When a strategy is selected from N candidates (e.g., parameter
        sweep, factor selection), the best-performing strategy has an
        inflated Sharpe. This test corrects for that inflation.

        Two modes:
        1. Provide `num_trials` = number of strategies tested. Uses the
           analytical approximation E[max(SR)] ≈ √(2 ln N) for iid SR.
        2. Provide `expected_max_sharpe` directly (e.g., from bootstrap).

        Args:
            returns: Array of strategy daily returns.
            num_trials: Number of strategies/parameters tested (N).
            expected_max_sharpe: Override for E[max SR] under null.

        Returns:
            DeflatedSharpeResult with p-value and significance flag.
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            logger.warning("Too few observations (%d) for reliable DSR test", len(returns))
            return DeflatedSharpeResult(
                observed_sharpe=0.0,
                expected_max_sharpe=0.0,
                deflated_p_value=1.0,
                num_trials=num_trials,
                skew=0.0,
                kurtosis=0.0,
                significant=False,
            )

        n = len(returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))

        if sigma < 1e-12:
            return DeflatedSharpeResult(
                observed_sharpe=0.0,
                expected_max_sharpe=0.0,
                deflated_p_value=1.0,
                num_trials=num_trials,
                skew=0.0,
                kurtosis=0.0,
                significant=False,
            )

        # Observed annualized Sharpe (assuming daily returns)
        observed_sr = mu / sigma * math.sqrt(252)

        # Higher moments
        skew = float(((returns - mu) ** 3).mean() / (sigma ** 3))
        excess_kurt = float(((returns - mu) ** 4).mean() / (sigma ** 4) - 3)

        # Expected max Sharpe under null (analytical approximation)
        if expected_max_sharpe is None:
            # E[max(SR_1, ..., SR_N)] ≈ √(2 ln N) for iid standard normals
            # More accurate with finite-sample and non-normality correction
            if num_trials <= 1:
                e_max_sr = 0.0
            else:
                e_max_sr = math.sqrt(2 * math.log(num_trials))
                # Finite-sample correction (Bailey & López de Prado)
                euler_mascheroni = 0.5772156649
                e_max_sr -= euler_mascheroni / (2 * math.sqrt(2 * math.log(num_trials))) if num_trials > 1 else 0
        else:
            e_max_sr = expected_max_sharpe

        # Compute deflated p-value: P(SR_observed | H0: SR = E[max SR])
        # Using the Jobson & Korkie (1981) test statistic with
        # non-normality correction (Bailey & López de Prado 2014)
        # SR_hat ~ N(SR_true, (1 + 0.5*SR_true^2 - skew*SR_true + (kurt-3)/4*SR_true^2) / (n-1))
        # Under H0: SR_true = E[max SR]

        # Standard error of Sharpe ratio estimate
        se_sr = math.sqrt(
            (1 + 0.5 * observed_sr ** 2 - skew * observed_sr
             + (excess_kurt) / 4 * observed_sr ** 2) / (n - 1)
        )

        if se_sr < 1e-12:
            deflated_p = 0.0 if observed_sr > e_max_sr else 1.0
        else:
            # Z-test: is observed SR significantly above E[max SR]?
            z = (observed_sr - e_max_sr) / se_sr
            # One-sided p-value (we want SR > E[max SR])
            deflated_p = _norm_sf(z)

        significant = deflated_p < self.significance_level

        logger.info(
            "DSR test: observed_SR=%.3f, E[max_SR]=%.3f (N=%d), p=%.4f, %s",
            observed_sr, e_max_sr, num_trials, deflated_p,
            "SIGNIFICANT" if significant else "NOT SIGNIFICANT",
        )

        return DeflatedSharpeResult(
            observed_sharpe=round(observed_sr, 6),
            expected_max_sharpe=round(e_max_sr, 6),
            deflated_p_value=round(deflated_p, 6),
            num_trials=num_trials,
            skew=round(skew, 6),
            kurtosis=round(excess_kurt, 6),
            significant=significant,
        )

    def multiple_testing_correction(
        self,
        p_values: list[float],
        method: str = "bh",
    ) -> np.ndarray:
        """Apply multiple testing correction to a set of p-values.

        Args:
            p_values: List of raw p-values from individual tests.
            method: 'bh' for Benjamini-Hochberg (FDR control),
                    'bonferroni' for Bonferroni (FWER control).

        Returns:
            Boolean array indicating which tests remain significant.
        """
        p = np.asarray(p_values, dtype=float)
        n = len(p)

        if n == 0:
            return np.array([], dtype=bool)

        if method == "bonferroni":
            # Bonferroni: reject if p < alpha / n
            adjusted = p * n
            adjusted = np.minimum(adjusted, 1.0)
            return adjusted < self.significance_level

        elif method == "bh":
            # Benjamini-Hochberg: control False Discovery Rate
            return _benjamini_hochberg(p, self.significance_level)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'bh' or 'bonferroni'.")

    def validate_factor_batch(
        self,
        factor_returns: dict[str, np.ndarray],
        num_trials: int | None = None,
    ) -> dict[str, DeflatedSharpeResult]:
        """Validate a batch of factors, applying multiple testing correction.

        Args:
            factor_returns: Dict of factor_name -> daily returns array.
            num_trials: Number of trials (defaults to len(factor_returns)).

        Returns:
            Dict of factor_name -> DeflatedSharpeResult. Only factors that
            pass both the DSR test and multiple testing correction are
            marked as significant.
        """
        if num_trials is None:
            num_trials = len(factor_returns)

        results = {}
        raw_p_values = []
        factor_names = []

        for name, rets in factor_returns.items():
            result = self.deflated_sharpe_test(rets, num_trials=num_trials)
            results[name] = result
            raw_p_values.append(result.deflated_p_value)
            factor_names.append(name)

        # Apply multiple testing correction
        if len(raw_p_values) > 1:
            significant_mask = self.multiple_testing_correction(raw_p_values, method="bh")
            for i, name in enumerate(factor_names):
                # Override significance with corrected result
                results[name] = DeflatedSharpeResult(
                    observed_sharpe=results[name].observed_sharpe,
                    expected_max_sharpe=results[name].expected_max_sharpe,
                    deflated_p_value=results[name].deflated_p_value,
                    num_trials=num_trials,
                    skew=results[name].skew,
                    kurtosis=results[name].kurtosis,
                    significant=bool(significant_mask[i]),
                )

        n_significant = sum(1 for r in results.values() if r.significant)
        logger.info(
            "Batch validation: %d/%d factors significant (N=%d trials)",
            n_significant, len(results), num_trials,
        )

        return results


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _norm_sf(x: float) -> float:
    """Survival function (1 - CDF) of standard normal distribution.

    Uses the rational approximation from Abramowitz & Stegun.
    """
    # Use math.erfc for numerical stability
    return 0.5 * math.erfc(x / math.sqrt(2))


def _benjamini_hochberg(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg procedure for FDR control.

    Steps:
    1. Sort p-values
    2. Find largest k such that p_(k) <= (k/m) * alpha
    3. Reject all hypotheses with p <= p_(k)

    Args:
        p_values: Array of raw p-values.
        alpha: Desired FDR level.

    Returns:
        Boolean array (same order as input) indicating rejections.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)

    # Sort p-values, keeping track of original indices
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH thresholds: (rank / m) * alpha
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha

    # Find the largest k where p_(k) <= threshold
    rejections_sorted = sorted_p <= thresholds

    # If any rejections, reject all up to the last True
    if rejections_sorted.any():
        last_reject = np.max(np.where(rejections_sorted)[0])
        rejections_sorted[:last_reject + 1] = True
    else:
        rejections_sorted[:] = False

    # Map back to original order
    result = np.zeros(m, dtype=bool)
    result[sorted_idx] = rejections_sorted

    return result
