"""Tests for portfolio optimizers."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.optimizers import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)


def _make_signal_and_cov(n_assets=50):
    np.random.seed(42)
    assets = [f"{i:06d}.SH" for i in range(n_assets)]

    signal = pd.Series(np.random.randn(n_assets), index=assets)
    signal = signal - signal.min() + 0.1  # Make positive

    # Generate a valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov = A @ A.T + np.eye(n_assets) * 0.01
    cov = pd.DataFrame(cov, index=assets, columns=assets)

    return signal, cov


def test_equal_weight_basic():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(max_weight=0.10)
    optimizer = EqualWeightOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)

    # 10 names at the 10% cap: the optimum is exactly the top-10 by signal.
    top = signal.nlargest(10).index
    assert weights.sum() == pytest.approx(1.0, rel=1e-9)
    assert set(weights[weights > 0].index) == set(top)
    assert np.allclose(weights[top].to_numpy(), 0.10)
    assert weights.max() <= constraints.max_weight + 1e-6


def test_equal_weight_sums_to_one():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints()
    optimizer = EqualWeightOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.sum() == pytest.approx(1.0, rel=1e-9)
    assert (weights > 0).sum() == 20  # 1 / 5% cap


def test_mean_variance_basic():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(risk_aversion=1.0)
    optimizer = MeanVarianceOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.sum() > 0
    assert (weights >= -0.01).all()  # Long only (allow tiny numerical issues)


def test_mean_variance_constraints():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(max_weight=0.10, risk_aversion=0.5)
    optimizer = MeanVarianceOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.max() <= constraints.max_weight + 1e-6


def test_prior_holding_outside_todays_signal_is_not_an_optimization_variable():
    """A suspended/delisted holding cannot be a cvxpy variable.

    Keeping prior holdings in the candidate set must not blindly union names
    that are absent from the signal and covariance; that raised KeyError on
    the first real pipeline run.
    """
    assets = [f"S{i:03d}" for i in range(30)]
    cov = pd.DataFrame(np.eye(30) * 0.04, index=assets, columns=assets)
    signal = pd.Series(1.0, index=assets)
    prev = pd.Series(1.0 / 31, index=assets + ["DELISTED"])

    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.05, max_turnover=0.30, risk_aversion=1.0,
    ))
    weights = optimizer.optimize(signal, cov, prev_weights=prev)

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert abs(float(weights.sum()) - 1.0) < 1e-3


def test_signal_rotation_beyond_turnover_budget_stays_feasible():
    """When the new signal rotates out more than the turnover budget, the
    fixed `dropped` term alone exceeded the constraint's right-hand side.

    No portfolio could satisfy the problem, the solver reported infeasible,
    and every such rebalance silently fell back to equal weight (54 of 60
    rebalances on the default pipeline).
    """
    assets = [f"S{i:03d}" for i in range(200)]
    cov = pd.DataFrame(np.eye(200) * 0.04, index=assets, columns=assets)

    first = pd.Series(0.0, index=assets)
    first.iloc[:100] = 1.0
    second = pd.Series(0.0, index=assets)
    second.iloc[100:] = 1.0

    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.05, max_turnover=0.30, risk_aversion=1.0,
    ))

    w1 = optimizer.optimize(first, cov)
    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason

    w2 = optimizer.optimize(second, cov, prev_weights=w1)

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert abs(float(w2.sum()) - 1.0) < 1e-3


def test_mean_variance_uses_an_installed_solver():
    """The solver loop hardcoded `[cp.ECOS, cp.SCS]`.

    ECOS is not included in cvxpy 1.7 wheels, so the first attempt always
    failed and every solve depended on SCS, which reported `infeasible` on the
    default 100-asset constrained problem. With `n_optimizer_fallbacks`
    reporting the truth, the default mean-variance configuration showed 54
    fallbacks across ~60 rebalances -- it had effectively been equal weight.
    """
    import cvxpy as cp

    signal, cov = _make_signal_and_cov(100)
    sector_map = pd.Series(
        [f"SEC{i % 10}" for i in range(100)], index=signal.index,
    )
    constraints = PortfolioConstraints(
        max_weight=0.05, max_sector_exposure=0.30, max_turnover=0.30,
        risk_aversion=1.0,
    )
    optimizer = MeanVarianceOptimizer(constraints)

    first = optimizer.optimize(signal, cov, sector_map=sector_map)
    optimizer.optimize(
        signal.abs()[::-1], cov, prev_weights=first, sector_map=sector_map,
    )

    assert any(
        solver in cp.installed_solvers()
        for solver in ("CLARABEL", "ECOS", "SCS")
    )
    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason


def test_risk_parity_basic():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints()
    optimizer = RiskParityOptimizer(constraints)
    weights = optimizer.optimize(signal, cov)
    assert weights.sum() > 0
    assert (weights >= -0.01).all()


def test_optimizer_with_prev_weights():
    signal, cov = _make_signal_and_cov()
    constraints = PortfolioConstraints(max_weight=0.10, max_turnover=0.30)
    optimizer = EqualWeightOptimizer(constraints)

    # The desired portfolio is the top-10. Hold ten *other* names so the
    # unconstrained target would be a complete swap (turnover 100%).
    target = EqualWeightOptimizer(
        PortfolioConstraints(max_weight=0.10, max_turnover=1.0)
    ).optimize(signal, cov)
    outside = signal.index.difference(target[target > 0].index)[:10]
    prev = pd.Series(0.0, index=signal.index)
    prev.loc[outside] = 0.10

    weights = optimizer.optimize(signal, cov, prev_weights=prev)

    index = weights.index.union(prev.index)
    turnover = (
        weights.reindex(index, fill_value=0.0) - prev.reindex(index, fill_value=0.0)
    ).abs().sum() / 2
    assert turnover == pytest.approx(0.30, abs=1e-9)
    # And the clamp actually moved the target rather than being ignored.
    assert not np.allclose(
        weights.reindex(signal.index, fill_value=0.0).to_numpy(),
        target.reindex(signal.index, fill_value=0.0).to_numpy(),
    )


def test_turnover_counts_names_entering_and_leaving_the_portfolio():
    """The old clamp aligned on the intersection, so a complete swap of the
    portfolio looked like zero turnover."""
    assets = [f"S{i:02d}" for i in range(20)]
    signal = pd.Series(1.0, index=assets)
    constraints = PortfolioConstraints(max_weight=0.10, max_turnover=0.20)
    prev = pd.Series(0.0, index=assets)
    prev.iloc[10:] = 0.10  # hold exactly the names the optimizer will drop

    weights = EqualWeightOptimizer(constraints).optimize(signal, prev_weights=prev)

    index = weights.index.union(prev.index)
    turnover = (
        weights.reindex(index, fill_value=0.0) - prev.reindex(index, fill_value=0.0)
    ).abs().sum() / 2
    assert turnover <= 0.20 + 1e-9, f"turnover {turnover:.3f} exceeded the cap"


@pytest.mark.parametrize(
    "optimizer_cls", [MeanVarianceOptimizer, RiskParityOptimizer],
)
def test_the_turnover_cap_holds_for_optimizers_with_prev_weights(optimizer_cls):
    """Only the equal-weight clamp used to be tested.

    Both convex optimizers select a different portfolio than the one held, so
    removing `_limit_turnover` must show up as an over-budget final weight
    vector, not merely as a solver warning.
    """
    signal, cov = _make_signal_and_cov(40)
    constraints = PortfolioConstraints(
        max_weight=0.10, max_turnover=0.10, risk_aversion=1.0,
    )
    prev = pd.Series(0.0, index=signal.index)
    prev.loc[signal.nsmallest(10).index] = 0.10  # names both want to drop

    optimizer = optimizer_cls(constraints)
    weights = optimizer.optimize(signal, cov, prev_weights=prev)

    index = weights.index.union(prev.index)
    turnover = (
        weights.reindex(index, fill_value=0.0) - prev.reindex(index, fill_value=0.0)
    ).abs().sum() / 2
    assert turnover <= 0.10 + 1e-9, f"turnover {turnover:.3f} exceeded the cap"
    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason


def test_equal_weight_never_scales_above_max_weight():
    """Renormalizing a too-narrow candidate set used to undo the cap."""
    assets = [f"S{i}" for i in range(5)]
    signal = pd.Series(1.0, index=assets)
    constraints = PortfolioConstraints(max_weight=0.10)

    weights = EqualWeightOptimizer(constraints).optimize(signal)

    assert weights.max() <= 0.10 + 1e-12
    # Five names cannot fill 100% under a 10% cap; the rest is cash.
    assert weights.sum() <= 0.5 + 1e-12


def test_fallback_is_counted_and_reason_is_recorded():
    signal = pd.Series([1.0], index=["ONLY"])
    optimizer = MeanVarianceOptimizer(PortfolioConstraints(max_weight=1.0))

    weights = optimizer.optimize(signal)

    assert weights.sum() > 0
    assert optimizer.fallback_count == 1
    assert "fewer than two" in (optimizer.last_fallback_reason or "")


def test_solver_failure_is_counted(monkeypatch):
    import cvxpy as cp

    signal, cov = _make_signal_and_cov(20)
    constraints = PortfolioConstraints(max_weight=0.10, max_turnover=0.30)
    optimizer = MeanVarianceOptimizer(constraints)

    def explode(self, *args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(cp.Problem, "solve", explode)

    weights = optimizer.optimize(signal, cov)

    assert weights.sum() > 0
    assert optimizer.fallback_count == 1
    assert "solver" in (optimizer.last_fallback_reason or "")


def test_risk_parity_is_not_silently_equal_weight():
    """The old objective was non-DCP and only tried the missing ECOS solver,
    so this optimizer always fell back to equal weight."""
    signal, cov = _make_signal_and_cov(60)
    constraints = PortfolioConstraints(max_weight=0.10)
    optimizer = RiskParityOptimizer(constraints)

    weights = optimizer.optimize(signal, cov)

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert weights.max() <= constraints.max_weight + 1e-6
    # Equal weights have zero dispersion; a real risk solution does not.
    assert weights.std() > 1e-4


# ---------------------------------------------------------------------------
# R8: cost-aware objective and ADV capacity caps
# ---------------------------------------------------------------------------

# Weights produced by the *pre-R8* MeanVarianceOptimizer on the case built by
# ``_regression_case`` (captured at HEAD 2ccee7b before this change; same
# Python/cvxpy/CLARABEL on this machine). The default configuration must keep
# reproducing this decision.
_BASELINE_NO_PREV = {
    'S00': 0.07483308, 'S01': 0.0, 'S02': 0.06749459, 'S03': 0.01496343,
    'S04': 0.0, 'S05': 0.04153772, 'S06': 0.0, 'S07': 0.03721272,
    'S08': 0.0, 'S09': 0.02165814, 'S10': 0.03179561, 'S11': 0.0416334,
    'S12': 0.06668063, 'S13': 0.05022117, 'S14': 0.0829356, 'S15': 0.0,
    'S16': 0.0, 'S17': 0.05046352, 'S18': 0.07891888, 'S19': 0.0,
    'S20': 0.01963021, 'S21': 0.03082312, 'S22': 0.03885531, 'S23': 0.02517766,
    'S24': 0.08650875, 'S25': 0.09354045, 'S26': 0.0, 'S27': 0.02715692,
    'S28': 0.0062032, 'S29': 0.01175587,
}
_BASELINE_WITH_PREV = {
    'S00': 0.02504663, 'S01': 0.04030593, 'S02': 0.01128309, 'S03': 0.01519727,
    'S04': 0.1, 'S05': 0.01594393, 'S06': 1e-08, 'S07': 0.1,
    'S08': 0.0, 'S09': 0.05363617, 'S10': 0.0, 'S11': 0.0,
    'S12': 0.0, 'S13': 0.0, 'S14': 0.07839839, 'S15': 0.1,
    'S16': 0.0, 'S17': 0.07612203, 'S18': 0.03937641, 'S19': 0.03934194,
    'S20': 0.01232357, 'S21': 0.01399194, 'S22': 0.06267737, 'S23': 0.0,
    'S24': 0.1, 'S25': 0.1, 'S26': 0.0, 'S27': 0.00942568,
    'S28': 0.00692963, 'S29': 0.0,
}


def _regression_case():
    """The fixed inputs used to capture the pre-R8 baseline weights."""
    rng = np.random.default_rng(42)
    n = 30
    assets = [f"S{i:02d}" for i in range(n)]
    signal = pd.Series(rng.normal(size=n), index=assets)
    signal = signal - signal.min() + 0.1
    a = rng.normal(size=(n, n))
    cov = pd.DataFrame(a @ a.T + np.eye(n) * 0.01, index=assets, columns=assets)
    sectors = pd.Series([f"SEC{i % 4}" for i in range(n)], index=assets)
    prev = pd.Series(0.0, index=signal.index)
    prev.loc[signal.nsmallest(10).index] = 0.10
    return signal, cov, sectors, prev


def _regression_constraints(**overrides):
    return PortfolioConstraints(
        max_weight=0.10, max_sector_exposure=0.30, max_turnover=0.30,
        risk_aversion=1.0, **overrides,
    )


def test_default_cost_fields_reproduce_the_pre_r8_decision():
    """With transaction_cost_bps=0 and turnover_penalty=0 the optimizer must
    construct the old problem: same weights as the captured pre-change run
    (within solver tolerance), with and without previous weights."""
    signal, cov, sectors, prev = _regression_case()

    w_no_prev = MeanVarianceOptimizer(_regression_constraints()).optimize(
        signal, cov, sector_map=sectors,
    )
    w_with_prev = MeanVarianceOptimizer(_regression_constraints()).optimize(
        signal, cov, prev_weights=prev, sector_map=sectors,
    )

    expected_no_prev = pd.Series(_BASELINE_NO_PREV)
    expected_with_prev = pd.Series(_BASELINE_WITH_PREV)
    np.testing.assert_allclose(
        w_no_prev.reindex(expected_no_prev.index).to_numpy(),
        expected_no_prev.to_numpy(), atol=1e-6, rtol=0,
    )
    np.testing.assert_allclose(
        w_with_prev.reindex(expected_with_prev.index).to_numpy(),
        expected_with_prev.to_numpy(), atol=1e-6, rtol=0,
    )
    # And the solution is a valid long-only, fully invested portfolio.
    for w in (w_no_prev, w_with_prev):
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.min() >= -1e-9
        assert w.max() <= 0.10 + 1e-6


def test_transaction_cost_without_prev_weights_is_a_no_op_even_with_shorts():
    """Task spec: with ``prev_weights is None`` the cost term is 0.

    Shorting is enabled so ``sum|w| != 1``: if the implementation wrongly
    treated a missing previous book as all-zero weights, the linear cost term
    would penalise gross exposure and change the solution. (In the long-only
    case ``sum|w| == 1`` is a constant and such a test would be vacuous.)
    """
    _, signal, cov, sectors, _ = _cost_aware_case()
    shared = dict(
        max_weight=0.10, max_sector_exposure=0.30, max_turnover=1.0,
        risk_aversion=1.0, long_only=False,
    )
    baseline = MeanVarianceOptimizer(
        PortfolioConstraints(**shared)
    ).optimize(signal, cov, sector_map=sectors)
    costly = MeanVarianceOptimizer(
        PortfolioConstraints(transaction_cost_bps=100.0, **shared)
    ).optimize(signal, cov, sector_map=sectors)

    # The case is not vacuous: the default solution really is short.
    assert (baseline < -1e-6).any()
    assert float(baseline.abs().sum()) > 1.5
    pd.testing.assert_series_equal(
        costly.sort_index(), baseline.sort_index(), atol=1e-12, rtol=0,
    )


def _cost_aware_case(n=40):
    """A well-conditioned case where the cost terms can move the optimum.

    The top-10 names (signal 0.50 -> 0.49) and the held names (0.45 -> 0.44)
    are close enough that the 20bp turnover cost is visible; covariance is
    diagonal so the risk term does not swamp the cost term.
    """
    names = [f"S{i:02d}" for i in range(n)]
    sig = np.zeros(n)
    for i in range(10):
        sig[i] = 0.50 - i * 1e-3
        sig[10 + i] = 0.45 - i * 1e-3
    signal = pd.Series(sig, index=names)
    cov = pd.DataFrame(np.eye(n) * 0.0004, index=names, columns=names)
    sectors = pd.Series([f"SEC{i % 4}" for i in range(n)], index=names)
    prev = pd.Series(0.0, index=names)
    prev.loc[names[10:20]] = 0.10
    return names, signal, cov, sectors, prev


def _cost_aware_constraints(**overrides):
    return PortfolioConstraints(
        max_weight=0.10, max_sector_exposure=0.30, max_turnover=1.0,
        risk_aversion=1.0, **overrides,
    )


def _one_sided_turnover(weights, prev):
    index = weights.index.union(prev.index)
    return float(
        (weights.reindex(index, fill_value=0.0)
         - prev.reindex(index, fill_value=0.0)).abs().sum() / 2
    )


def test_transaction_cost_reduces_turnover_monotonically():
    """b: 20bp must trade less than the baseline, and more cost must trade
    monotonically less (at least two levels)."""
    _, signal, cov, sectors, prev = _cost_aware_case()
    turnovers = []
    for bps in (0.0, 20.0, 50.0):
        optimizer = MeanVarianceOptimizer(
            _cost_aware_constraints(transaction_cost_bps=bps)
        )
        weights = optimizer.optimize(
            signal, cov, prev_weights=prev, sector_map=sectors,
        )
        assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights.max() <= 0.10 + 1e-6
        turnovers.append(_one_sided_turnover(weights, prev))

    assert turnovers[1] < turnovers[0], (
        f"20bp cost did not reduce turnover: {turnovers}"
    )
    assert turnovers[2] < turnovers[1], (
        f"turnover is not monotone in the cost level: {turnovers}"
    )


def test_turnover_penalty_reduces_turnover_and_keeps_constraints():
    """c: the quadratic penalty is monotone and the portfolio stays valid."""
    _, signal, cov, sectors, prev = _cost_aware_case()
    turnovers = []
    for penalty in (0.0, 0.5, 2.0, 10.0):
        optimizer = MeanVarianceOptimizer(
            _cost_aware_constraints(turnover_penalty=penalty)
        )
        weights = optimizer.optimize(
            signal, cov, prev_weights=prev, sector_map=sectors,
        )
        assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights.min() >= -1e-9
        assert weights.max() <= 0.10 + 1e-6
        sector_totals = weights.groupby(sectors).sum()
        assert sector_totals.max() <= 0.30 + 1e-6
        turnovers.append(_one_sided_turnover(weights, prev))

    assert turnovers == sorted(turnovers, reverse=True), (
        f"turnover is not monotone in the penalty: {turnovers}"
    )
    assert turnovers[-1] < turnovers[0]


def test_capacity_caps_weights_when_adv_is_small():
    """d: w_i <= max_participation * adv_i / aum for capped names."""
    names, signal, cov, sectors, prev = _cost_aware_case()
    aum = 1e7
    participation = 0.10
    cap = 0.02  # below max_weight, above zero
    adv = pd.Series(1e9, index=names)
    adv.loc[names[:10]] = cap * aum / participation

    optimizer = MeanVarianceOptimizer(
        _cost_aware_constraints(max_participation=participation)
    )
    weights = optimizer.optimize(
        signal, cov, prev_weights=prev, sector_map=sectors,
        adv=adv, aum=aum,
    )

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert weights.loc[names[:10]].max() <= cap + 1e-6
    # The cap is actually binding: at least one top-signal name sits at it.
    assert weights.loc[names[:10]].max() == pytest.approx(cap, abs=1e-6)
    # The uncapped names absorbed the difference (the problem is feasible).
    assert weights.loc[names[20:]].sum() > 0.0


def test_capacity_is_a_no_op_when_adv_is_ample():
    """d: caps above max_weight leave the unconstrained decision unchanged."""
    names, signal, cov, sectors, prev = _cost_aware_case()
    aum = 1e7
    adv = pd.Series(1e9, index=names)

    unconstrained = MeanVarianceOptimizer(
        _cost_aware_constraints()
    ).optimize(signal, cov, prev_weights=prev, sector_map=sectors)
    optimizer = MeanVarianceOptimizer(
        _cost_aware_constraints(max_participation=0.10)
    )
    capped = optimizer.optimize(
        signal, cov, prev_weights=prev, sector_map=sectors,
        adv=adv, aum=aum,
    )

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    pd.testing.assert_series_equal(
        capped.sort_index(), unconstrained.sort_index(), atol=1e-8, rtol=0,
    )


def test_capacity_requires_adv_and_aum_and_is_otherwise_off():
    """All three inputs are needed; otherwise the default decision stands."""
    _, signal, cov, sectors, prev = _cost_aware_case()
    baseline = MeanVarianceOptimizer(_cost_aware_constraints()).optimize(
        signal, cov, prev_weights=prev, sector_map=sectors,
    )
    for kwargs in (
        {"adv": None, "aum": 1e7},
        {"adv": pd.Series(1e9, index=signal.index), "aum": None},
        {"adv": pd.Series(1e9, index=signal.index), "aum": 0.0},
    ):
        optimizer = MeanVarianceOptimizer(
            _cost_aware_constraints(max_participation=0.10)
        )
        weights = optimizer.optimize(
            signal, cov, prev_weights=prev, sector_map=sectors, **kwargs,
        )
        assert optimizer.fallback_count == 0
        pd.testing.assert_series_equal(
            weights.sort_index(), baseline.sort_index(), atol=1e-12, rtol=0,
        )


def test_missing_adv_assets_keep_max_weight_and_are_logged(caplog):
    """Missing ADV is not treated as zero liquidity: the name keeps the
    max_weight cap (and the count is logged) instead of dropping out."""
    names, signal, cov, sectors, prev = _cost_aware_case()
    aum = 1e7
    participation = 0.10
    adv = pd.Series(1e9, index=names)
    adv.loc[names[:10]] = 0.02 * aum / participation  # 2% cap
    adv.loc[names[5]] = np.nan  # missing quote for a top-signal name

    optimizer = MeanVarianceOptimizer(
        _cost_aware_constraints(max_participation=participation)
    )
    with caplog.at_level("WARNING"):
        weights = optimizer.optimize(
            signal, cov, prev_weights=prev, sector_map=sectors,
            adv=adv, aum=aum,
        )

    assert optimizer.fallback_count == 0, optimizer.last_fallback_reason
    assert any(
        "no positive ADV" in record.getMessage() for record in caplog.records
    ), "the skipped capacity cap was not logged"
    # The missing-ADV name was allowed above its would-be 2% cap (but still
    # under max_weight).
    assert weights.loc[names[5]] > 0.02 + 1e-6
    assert weights.loc[names[5]] <= 0.10 + 1e-6


def test_post_turnover_blend_refills_a_capacity_clipped_book():
    """m1: when the post-solve turnover blend adds back a previous book that
    exceeds the new ADV caps, clipping alone used to leave the target badly
    under-invested (sum << 1). The clipped weight must be refilled from
    remaining headroom -- and that refill must respect the sector caps too
    (the previous book used here is sector-compliant)."""
    n = 40
    names = [f"S{i:02d}" for i in range(n)]
    signal = pd.Series(
        np.r_[np.full(10, 0.5), np.full(20, 0.2), np.zeros(10)], index=names,
    )
    # S39 is deliberately outside the covariance universe (held, dropped).
    cov = pd.DataFrame(
        np.eye(39) * 0.0004, index=names[:39], columns=names[:39],
    )
    # Sector layout: the held names fill SEC0/1/2 to their caps (0.30/0.225/
    # 0.225) and S39 puts SEC3 at 0.25 -- the previous book is compliant.
    sectors = pd.Series([f"SEC{i % 4}" for i in range(n)], index=names)
    sectors.loc[names[10:14]] = "SEC0"
    sectors.loc[names[14:17]] = "SEC1"
    sectors.loc[names[17:20]] = "SEC2"
    sectors.loc[names[39]] = "SEC3"
    prev = pd.Series(0.0, index=names)
    prev.loc[names[10:20]] = 0.075
    prev.loc[names[39]] = 0.25
    prev_sectors = prev.groupby(sectors).sum()
    assert prev_sectors.max() <= 0.30 + 1e-12

    adv = pd.Series(1e9, index=names)
    adv.loc[names[10:20]] = 0.03 * 1e7 / 0.10
    adv.loc[names[39]] = 0.02 * 1e7 / 0.10

    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.10, max_sector_exposure=0.30, max_turnover=0.10,
        risk_aversion=1.0, max_participation=0.10,
    ))
    weights = optimizer.optimize(
        signal, cov, prev_weights=prev, sector_map=sectors,
        adv=adv, aum=1e7,
    )

    assert optimizer.fallback_count == 0
    # Fully funded again, and the old in-universe holdings respect the cap.
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights.loc[names[10:20]].max() <= 0.03 + 1e-9
    # The refill must not push any sector over its cap.
    assert weights.groupby(sectors).sum().max() <= 0.30 + 1e-9


def test_capacity_constraint_is_solved_not_only_post_clipped():
    """The cap must enter the QP, not just be applied after an unconstrained
    solve. A clip-only implementation does not reproduce the constrained
    optimum in this case (it would hold A2/A3/A4 at ~0.098 instead of
    [0.3, 0.3, 0.3, 0.1, 0.0])."""
    import cvxpy as cp

    names = [f"A{i}" for i in range(5)]
    signal = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=names)
    cov = pd.DataFrame(
        np.diag([4e-4, 3e-4, 2e-4, 1e-4, 5e-5]),
        index=names, columns=names,
    )
    aum = 1e7
    participation = 0.10
    caps = np.array([0.30, 0.30, 0.30, 0.30, 0.30])
    adv = pd.Series(caps * aum / participation, index=names)

    weights = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=1.0, max_sector_exposure=1.0, max_turnover=1.0,
        risk_aversion=1.0, max_participation=participation,
    )).optimize(signal, cov, adv=adv, aum=aum)

    w_ref = cp.Variable(len(names))
    problem = cp.Problem(
        cp.Maximize(
            0.04 * (signal.to_numpy() @ w_ref)
            - 1.0 * cp.quad_form(w_ref, cov.to_numpy() * 252)
        ),
        [cp.sum(w_ref) == 1, w_ref >= 0, w_ref <= caps],
    )
    problem.solve(solver="CLARABEL")
    np.testing.assert_allclose(
        weights.to_numpy(), w_ref.value, atol=1e-6, rtol=0,
    )
    # The constrained optimum is not the post-clipped unconstrained one.
    assert weights.loc["A4"] == pytest.approx(0.0, abs=1e-6)


def test_refill_skips_sector_room_for_a_single_sector_universe():
    """The solver deliberately skips the sector constraint when every
    candidate is in one sector; the refill must mirror that guard. Applying
    sector room there would leave needless cash (review observation B)."""
    n = 40
    names = [f"S{i:02d}" for i in range(n)]
    signal = pd.Series(
        np.r_[np.full(10, 0.5), np.full(20, 0.2), np.zeros(10)], index=names,
    )
    cov = pd.DataFrame(
        np.eye(39) * 0.0004, index=names[:39], columns=names[:39],
    )
    sectors = pd.Series("SEC0", index=names)
    prev = pd.Series(0.0, index=names)
    prev.loc[names[10:20]] = 0.075
    prev.loc[names[39]] = 0.25
    adv = pd.Series(1e9, index=names)
    adv.loc[names[10:20]] = 0.03 * 1e7 / 0.10

    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.10, max_sector_exposure=0.30, max_turnover=0.10,
        risk_aversion=1.0, max_participation=0.10,
    ))
    weights = optimizer.optimize(
        signal, cov, prev_weights=prev, sector_map=sectors,
        adv=adv, aum=1e7,
    )

    assert optimizer.fallback_count == 0
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_solver_failure_reason_notes_the_ignored_capacity_cap(monkeypatch):
    """m1 (review 2): every fallback under an active capacity cap must say
    that the equal-weight result ignores the ADV cap."""
    import cvxpy as cp

    signal, cov = _make_signal_and_cov(20)
    adv = pd.Series(1e9, index=signal.index)

    def explode(self, *args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(cp.Problem, "solve", explode)
    optimizer = MeanVarianceOptimizer(PortfolioConstraints(
        max_weight=0.10, max_participation=0.10,
    ))
    weights = optimizer.optimize(signal, cov, adv=adv, aum=1e7)

    assert weights.sum() > 0
    assert optimizer.fallback_count == 1
    assert "ignores the ADV cap" in (optimizer.last_fallback_reason or "")


def test_infeasible_capacity_falls_back_with_a_warning(caplog):
    """e: caps that cannot fund the book must warn and use the existing
    equal-weight fallback, never crash and never silently ignore the cap."""
    names, signal, cov, sectors, prev = _cost_aware_case()
    aum = 1e7
    adv = pd.Series(1e5, index=names)  # cap = 0.001 each, sum 4% << 100%

    optimizer = MeanVarianceOptimizer(
        _cost_aware_constraints(max_participation=0.10)
    )
    with caplog.at_level("WARNING"):
        weights = optimizer.optimize(
            signal, cov, prev_weights=prev, sector_map=sectors,
            adv=adv, aum=aum,
        )

    assert optimizer.fallback_count == 1
    assert "capacity caps sum" in (optimizer.last_fallback_reason or "")
    assert any(
        "Optimizer fallback to equal weight" in record.getMessage()
        for record in caplog.records
    )
    assert weights.sum() > 0.0  # a portfolio is still returned
