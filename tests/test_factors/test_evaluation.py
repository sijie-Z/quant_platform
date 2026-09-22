"""Tests for factor evaluation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.evaluation import (
    factor_correlation,
    factor_turnover,
    ic_decay,
    ic_summary,
    pearson_ic,
    quantile_returns,
    rank_ic,
)
from quant_platform.utils.numba_accelerator import HAS_NUMBA


def _make_factor_and_returns(n_dates=200, n_assets=50):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}.SH" for i in range(n_assets)]

    # Factor with predictive power: add some true signal
    true_signal = np.random.randn(n_assets) * 0.5
    noise = np.random.randn(n_dates, n_assets) * 0.5
    factor = pd.DataFrame(
        true_signal[np.newaxis, :] + noise,
        index=dates, columns=assets,
    )

    # Forward returns partly driven by factor
    fwd_ret = factor.shift(-1) * 0.001 + np.random.randn(n_dates, n_assets) * 0.02
    fwd_ret = pd.DataFrame(fwd_ret, index=dates, columns=assets)

    return factor, fwd_ret


def test_rank_ic():
    factor, fwd_ret = _make_factor_and_returns()
    ic = rank_ic(factor, fwd_ret)
    assert len(ic) > 0
    assert -1 <= ic.mean() <= 1


def test_pearson_ic():
    factor, fwd_ret = _make_factor_and_returns()
    ic = pearson_ic(factor, fwd_ret)
    assert len(ic) > 0


def test_ic_summary():
    factor, fwd_ret = _make_factor_and_returns()
    ic = rank_ic(factor, fwd_ret)
    summary = ic_summary(ic)
    assert "mean_ic" in summary
    assert "icir" in summary
    assert "ic_positive_ratio" in summary


def test_quantile_returns():
    factor, fwd_ret = _make_factor_and_returns()
    qr = quantile_returns(factor, fwd_ret, n_quantiles=5)
    if len(qr) > 0:
        for qi in range(1, 6):
            assert f"Q{qi}" in qr.columns


def test_factor_correlation():
    factor, fwd_ret = _make_factor_and_returns()
    factors = {"f1": factor, "f2": factor.shift(1).fillna(0)}
    corr = factor_correlation(factors)
    assert corr.shape == (2, 2)
    assert abs(corr.iloc[0, 0] - 1.0) < 0.01


def test_factor_turnover():
    factor, _ = _make_factor_and_returns()
    turnover = factor_turnover(factor)
    assert len(turnover) > 0
    # Turnover should be between 0 and 1
    assert (turnover >= 0).all()
    assert (turnover <= 1).all()


def test_ic_decay():
    factor, fwd_ret = _make_factor_and_returns()
    decay = ic_decay(factor, fwd_ret, max_periods=5)
    assert len(decay) == 5


class TestRankIcAlignsTheTwoAxes:
    """`rank_ic_numba` reads `.values`, which is positional: the factor and the
    return panel have to be intersected on dates *and* columns first. They were
    intersected on dates only, so a factor whose columns were not the return
    panel's columns was correlated row-for-row against the wrong assets. The
    one-column frame `MAConvergenceFactor` used to return picked up an IC for
    every date that way -- a number computed from one value against 418."""

    def test_a_factor_without_an_asset_axis_has_no_ic(self):
        factor, fwd_ret = _make_factor_and_returns()
        collapsed = pd.DataFrame({"factor": factor.mean(axis=1)}, index=factor.index)
        assert len(rank_ic(collapsed, fwd_ret)) == 0

    def test_only_the_shared_assets_are_correlated(self):
        factor, fwd_ret = _make_factor_and_returns()
        subset = factor.iloc[:, :40]
        pd.testing.assert_series_equal(
            rank_ic(subset, fwd_ret),
            rank_ic(subset, fwd_ret.iloc[:, :40]),
            check_exact=False, rtol=1e-12, atol=1e-14,
        )

    def test_assets_are_matched_by_label_not_by_position(self):
        factor, fwd_ret = _make_factor_and_returns()
        pd.testing.assert_series_equal(
            rank_ic(factor, fwd_ret.iloc[:, ::-1]),
            rank_ic(factor, fwd_ret),
            check_exact=False, rtol=1e-12, atol=1e-14,
        )

    @pytest.mark.skipif(not HAS_NUMBA, reason="both paths are the same code without numba")
    def test_the_accelerated_path_agrees_with_the_fallback(self, monkeypatch):
        factor, fwd_ret = _make_factor_and_returns()
        accelerated = rank_ic(factor, fwd_ret)

        monkeypatch.setattr("quant_platform.utils.numba_accelerator.HAS_NUMBA", False)
        fallback = rank_ic(factor, fwd_ret)

        assert len(accelerated) > 0
        pd.testing.assert_series_equal(
            accelerated, fallback, check_exact=False, rtol=1e-9, atol=1e-12,
        )
