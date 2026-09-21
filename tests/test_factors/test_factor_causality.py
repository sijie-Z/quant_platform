"""Causality guards for every registered factor.

The no-lookahead contract's first clause -- a factor at date t may only use
data up to and including t -- had no mechanical check. Violations were found by
audit, one at a time: `pure_volatility` fitted its AR coefficients on the whole
sample and `MLSignalGenerator.predict` returned in-sample predictions, and both
were discovered by reading code rather than by a failing test.

This module tests the property directly, for every factor in the registry
including ones added later, without naming any of them:

    compute(panel[:cut])          == compute(panel)[:cut]          (truncation)
    compute(panel with a changed future)[:cut] == compute(panel)[:cut]  (perturbation)

A factor that fits anything on the full sample fails both.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.registry import get_registry
from quant_platform.factors.technical import register_all

CUT = 300
N_DAYS = 420
N_ASSETS = 40

#: Loose enough for last-bit floating-point differences (comparing the same
#: window computed at different array offsets reaches ~1e-18), tight enough
#: that any real leak -- which shows up at the factor's own scale, ~1e-3 --
#: fails loudly.
RTOL = 1e-9
ATOL = 1e-12


def _panel(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)
    cols = [f"S{i:03d}" for i in range(N_ASSETS)]
    steps = rng.normal(0, 0.012, (N_DAYS, N_ASSETS))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=cols
    )


def _registered_factors() -> list[str]:
    register_all()
    return sorted(get_registry().list_all())


def _compute(name: str, panel: pd.DataFrame) -> pd.DataFrame | None:
    """Factor output on `panel`, or None if it needs inputs beyond prices."""
    try:
        out = get_registry().get(name)().compute(panel)
    except Exception:
        return None
    return out if isinstance(out, pd.DataFrame) else None


FACTORS = _registered_factors()

#: Factors whose `compute` needs data this module does not supply (turnover,
#: volume, financials). Named rather than silently skipped, so that a factor
#: which starts failing to compute is noticed.
NEEDS_EXTRA_INPUT = {"turnover_20d", "ma_convergence"}


@pytest.mark.parametrize("name", FACTORS)
def test_a_factor_does_not_change_when_the_future_is_removed(name):
    """`compute(panel[:cut])` must equal `compute(panel)[:cut]`."""
    panel = _panel()
    full = _compute(name, panel)
    if full is None:
        pytest.skip(f"{name} needs inputs beyond prices")

    truncated = _compute(name, panel.iloc[:CUT])
    assert truncated is not None

    pd.testing.assert_frame_equal(
        truncated, full.loc[truncated.index], check_exact=False, rtol=RTOL, atol=ATOL,
    )


@pytest.mark.parametrize("name", FACTORS)
def test_a_factor_does_not_change_when_the_future_is_replaced(name):
    """Changing prices after `cut` must not move values before it. This is the
    check a full-sample fit fails."""
    panel = _panel()
    full = _compute(name, panel)
    if full is None:
        pytest.skip(f"{name} needs inputs beyond prices")

    perturbed = panel.copy()
    perturbed.iloc[CUT:] *= 1.7
    after = _compute(name, perturbed)
    assert after is not None

    pd.testing.assert_frame_equal(
        after.iloc[:CUT], full.iloc[:CUT], check_exact=False, rtol=RTOL, atol=ATOL,
    )


class TestAlphaCombinationIsCausal:
    """The second contract clause: the weights at date t may use only ICs from
    before t. `combine_ic_weighted` slices with a strict `ic_s.index < date`
    (alpha/combination.py:80); this is the check that keeps it that way."""

    @staticmethod
    def _inputs(cut: int | None = None):
        rng = np.random.default_rng(11)
        n_days, n_assets = 400, 40
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        cols = [f"S{i:03d}" for i in range(n_assets)]
        factors = {
            name: pd.DataFrame(rng.normal(size=(n_days, n_assets)), index=dates, columns=cols)
            for name in ("momentum", "value", "quality")
        }
        fwd = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=cols)
        if cut is not None:
            factors = {k: v.iloc[:cut] for k, v in factors.items()}
            fwd = fwd.iloc[:cut]
        return factors, fwd

    @pytest.mark.parametrize("method", ["equal_weight", "ic_weighted", "icir_weighted"])
    def test_weights_do_not_move_when_the_future_is_removed(self, method):
        from quant_platform.alpha.pipeline import AlphaPipeline

        pipeline = AlphaPipeline(method=method, lookback=60)
        full_factors, full_fwd = self._inputs()
        full = pipeline.run(full_factors, full_fwd)

        trunc_factors, trunc_fwd = self._inputs(CUT)
        truncated = pipeline.run(trunc_factors, trunc_fwd)

        pd.testing.assert_frame_equal(
            truncated, full.loc[truncated.index],
            check_exact=False, rtol=RTOL, atol=ATOL,
        )

    @pytest.mark.parametrize("method", ["ic_weighted", "icir_weighted"])
    def test_weights_do_not_move_when_the_future_returns_are_replaced(self, method):
        from quant_platform.alpha.pipeline import AlphaPipeline

        pipeline = AlphaPipeline(method=method, lookback=60)
        full_factors, full_fwd = self._inputs()
        full = pipeline.run(full_factors, full_fwd)

        # Rewrite realised returns after the cut.
        perturbed = {k: v.copy() for k, v in full_factors.items()}
        new_fwd = full_fwd.copy()
        new_fwd.iloc[CUT:] *= -3.0
        after = pipeline.run(perturbed, new_fwd)

        pd.testing.assert_frame_equal(
            after.iloc[:CUT], full.iloc[:CUT],
            check_exact=False, rtol=RTOL, atol=ATOL,
        )


class TestTheHarnessItselfWorks:
    """A causality check that cannot fail is worse than none: the first version
    of the sweep compared with `DataFrame.equals`, which is exact, so last-bit
    floating-point noise read as a violation. These pin down that both
    directions of the check actually discriminate."""

    def test_a_deliberately_non_causal_factor_fails_both(self):
        class FullSampleZScore:
            """Standardises using the whole sample's mean -- a look-ahead."""

            def compute(self, panel):
                return (panel - panel.mean().mean()) / panel.std().std()

        panel = _panel()
        full = FullSampleZScore().compute(panel)

        truncated = FullSampleZScore().compute(panel.iloc[:CUT])
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                truncated, full.loc[truncated.index],
                check_exact=False, rtol=RTOL, atol=ATOL,
            )

        perturbed = panel.copy()
        perturbed.iloc[CUT:] *= 1.7
        after = FullSampleZScore().compute(perturbed)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                after.iloc[:CUT], full.iloc[:CUT],
                check_exact=False, rtol=RTOL, atol=ATOL,
            )

    def test_the_fixture_actually_exercises_the_cut(self):
        panel = _panel()
        assert len(panel) > CUT
        assert not np.allclose(panel.iloc[CUT:].to_numpy(), (panel.iloc[CUT:] * 1.7).to_numpy())

    def test_every_registered_factor_is_covered_or_named(self):
        """No factor may be silently absent from the sweep."""
        covered = {n for n in FACTORS if _compute(n, _panel()) is not None}
        skipped = set(FACTORS) - covered
        assert skipped <= NEEDS_EXTRA_INPUT, (
            f"newly uncomputable factors: {sorted(skipped - NEEDS_EXTRA_INPUT)}"
        )
        assert len(covered) >= 15, f"only {len(covered)} factors were exercised"
