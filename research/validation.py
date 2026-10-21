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
from statistics import NormalDist

import numpy as np

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeflatedSharpeResult:
    """Result of a Deflated Sharpe Ratio test."""
    observed_sharpe: float
    expected_max_sharpe: float   # E[max SR under null]
    deflated_p_value: float      # P(SR >= observed | null), deflated for N trials
    num_trials: int              # Number of strategies tested
    skew: float                  # Return skewness
    kurtosis: float              # Return kurtosis (excess)
    significant: bool            # Final decision: DSR *and* BH (batch) — see below
    # One-sided p-value of the observed Sharpe against zero, WITHOUT the
    # E[max SR] deflation. Used as the input to Benjamini-Hochberg in
    # ``validate_factor_batch`` so the multiple-testing correction is applied
    # once, not twice.
    raw_p_value: float = 1.0
    #: Whether the deflated p-value passes the significance level on its own.
    dsr_significant: bool | None = None
    #: Whether the raw p-value passes BH within the batch. ``None`` when no
    #: batch correction was applied (single-factor use).
    bh_significant: bool | None = None


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
        periods_per_year: float = 252.0,
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
            returns: Array of strategy periodic returns (daily by default).
            num_trials: Number of strategies/parameters tested (N).
            expected_max_sharpe: Override for E[max SR] under null. Pass an
                **annualized** Sharpe; it is converted to per-period units
                internally.
            periods_per_year: Annualization basis for the returns. The old
                implementation hardcoded 252, silently mis-annualizing
                weekly or monthly series. Non-finite returns (inf) are
                dropped together with NaN; an inf used to yield a silent
                observed_sharpe=nan / p=1.0.

        Returns:
            DeflatedSharpeResult with p-value and significance flag.
        """
        try:
            periods_per_year = float(periods_per_year)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"periods_per_year must be a positive number, got {periods_per_year!r}"
            ) from exc
        if not math.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be finite and > 0, got {periods_per_year!r}"
            )

        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]

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
                dsr_significant=False,
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
                dsr_significant=False,
            )

        # Work in per-period units. The old code compared an *annualized*
        # Sharpe with the unit-variance order statistic sqrt(2 ln N) and
        # plugged the annualized SR into a standard-error formula that
        # expects a per-period one. The three terms were on different scales:
        # a true annualized SR of 2.0 over 1260 days with N=10 came out at
        # p=0.59, while a short-sample fluke could pass.
        sr = mu / sigma
        observed_sr = sr * math.sqrt(periods_per_year)

        # Higher moments (kurtosis is non-excess in the DSR formula)
        skew = float(((returns - mu) ** 3).mean() / (sigma ** 3))
        excess_kurt = float(((returns - mu) ** 4).mean() / (sigma ** 4) - 3)
        kurt = excess_kurt + 3.0

        # Variance of the per-period Sharpe estimator (Lo 2002 / Bailey &
        # López de Prado 2014, eq. for V[SR]).
        var_sr = max(
            0.0,
            (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(n - 1, 1),
        )
        se_sr = math.sqrt(var_sr)

        if expected_max_sharpe is None:
            if num_trials <= 1:
                e_max_sr = 0.0
            else:
                # E[max(SR_1..SR_N)] with the Gumbel approximation:
                # sqrt(V[SR]) * ((1-gamma)*Phi^-1(1-1/N)
                #                + gamma*Phi^-1(1-1/(N e)))
                euler_mascheroni = 0.5772156649
                nd = NormalDist()
                e_max_sr = math.sqrt(var_sr) * (
                    (1 - euler_mascheroni) * nd.inv_cdf(1 - 1 / num_trials)
                    + euler_mascheroni * nd.inv_cdf(1 - 1 / (num_trials * math.e))
                )
        else:
            # The override is documented as an annualized Sharpe.
            e_max_sr = expected_max_sharpe / math.sqrt(periods_per_year)

        if se_sr < 1e-12:
            raw_p = 0.0 if sr > 0 else 1.0
            deflated_p = 0.0 if sr > e_max_sr else 1.0
        else:
            # Z-test: is the per-period SR above E[max SR]?
            z = (sr - e_max_sr) / se_sr
            deflated_p = _norm_sf(z)
            # Same test against zero, without the multiple-trials deflation.
            raw_p = _norm_sf(sr / se_sr)

        significant = deflated_p < self.significance_level

        logger.info(
            "DSR test: observed_SR=%.3f, E[max_SR]=%.3f (N=%d), p=%.4f, %s",
            observed_sr, e_max_sr, num_trials, deflated_p,
            "SIGNIFICANT" if significant else "NOT SIGNIFICANT",
        )

        return DeflatedSharpeResult(
            observed_sharpe=round(observed_sr, 6),
            expected_max_sharpe=round(e_max_sr * math.sqrt(periods_per_year), 6),
            deflated_p_value=round(deflated_p, 6),
            num_trials=num_trials,
            skew=round(skew, 6),
            kurtosis=round(excess_kurt, 6),
            significant=significant,
            raw_p_value=round(raw_p, 6),
            dsr_significant=significant,
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

        Multiple-testing policy: a factor is reported ``significant`` only
        when **both** tests pass:

        1. the DSR deflated p-value is below ``significance_level`` (the
           effect survives the E[max SR] deflation), and
        2. its raw one-sided p-value survives Benjamini-Hochberg within the
           batch.

        The old implementation set ``significant`` from the BH mask alone and
        ignored the deflated p-value: a factor with raw_p=0.0319 but
        deflated_p=0.0911 was returned as significant. ``deflated_p_value`` is
        still reported as the effect-size adjustment, and the two components
        are exposed separately as ``dsr_significant`` / ``bh_significant``.
        With a single factor BH is a no-op and the DSR decision is used.

        Args:
            factor_returns: Dict of factor_name -> daily returns array.
            num_trials: Number of trials (defaults to len(factor_returns)).

        Returns:
            Dict of factor_name -> DeflatedSharpeResult.
        """
        if num_trials is None:
            num_trials = len(factor_returns)

        results = {}
        raw_p_values = []
        factor_names = []

        for name, rets in factor_returns.items():
            result = self.deflated_sharpe_test(rets, num_trials=num_trials)
            results[name] = result
            raw_p_values.append(result.raw_p_value)
            factor_names.append(name)

        # Apply the multiple-testing correction once, to the raw p-values,
        # and require the deflated p-value to pass as well.
        if len(raw_p_values) > 1:
            significant_mask = self.multiple_testing_correction(raw_p_values, method="bh")
            for i, name in enumerate(factor_names):
                prev = results[name]
                dsr_sig = prev.deflated_p_value < self.significance_level
                bh_sig = bool(significant_mask[i])
                results[name] = DeflatedSharpeResult(
                    observed_sharpe=prev.observed_sharpe,
                    expected_max_sharpe=prev.expected_max_sharpe,
                    deflated_p_value=prev.deflated_p_value,
                    num_trials=num_trials,
                    skew=prev.skew,
                    kurtosis=prev.kurtosis,
                    significant=bh_sig and dsr_sig,
                    raw_p_value=prev.raw_p_value,
                    dsr_significant=dsr_sig,
                    bh_significant=bh_sig,
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
