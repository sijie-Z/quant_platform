"""Tests for factor evaluation."""

import logging

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.evaluation import (
    factor_correlation,
    factor_turnover,
    ic_decay,
    ic_summary,
    min_assets_for_ic,
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


def test_q1_is_the_highest_factor_group():
    """`pd.qcut(labels=False)` numbers the lowest values 0; using it directly
    labeled the bottom quintile Q1, opposite to the documented convention."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    assets = [f"S{i:02d}" for i in range(50)]
    rank = np.tile(np.arange(50, dtype=float), (30, 1))
    factor = pd.DataFrame(rank, index=dates, columns=assets)
    # Forward returns increase with the factor: the top group earns the most.
    fwd = pd.DataFrame(
        rank + np.random.default_rng(0).normal(0, 1, rank.shape),
        index=dates, columns=assets,
    )

    qr = quantile_returns(factor, fwd, n_quantiles=5)

    assert qr["Q1"].mean() > qr["Q5"].mean()


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


def test_ic_summary_reports_the_number_of_observations():
    """The dashboard needs to distinguish "weak factor" from "no data"."""
    few = ic_summary(pd.Series([0.01, -0.02, 0.03, 0.04, -0.01]))

    assert few["n_obs"] == 5
    assert np.isnan(few["mean_ic"])

    many = ic_summary(pd.Series(np.linspace(-0.05, 0.05, 20)))
    assert many["n_obs"] == 20


def _make_sparse_panel(n_dates, n_assets, n_valid_per_row, seed=123):
    """Factor/return panels where each row has exactly ``n_valid_per_row``
    names (the rest are suspended/NaN, as in a real A-share universe)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    assets = [f"{i:06d}.SH" for i in range(n_assets)]
    factor = pd.DataFrame(
        rng.normal(size=(n_dates, n_assets)), index=dates, columns=assets
    )
    fwd_ret = pd.DataFrame(
        rng.normal(0, 0.02, (n_dates, n_assets)), index=dates, columns=assets
    )
    for i in range(n_dates):
        missing = rng.choice(
            n_assets, size=n_assets - n_valid_per_row, replace=False
        )
        factor.iloc[i, missing] = np.nan
        fwd_ret.iloc[i, missing] = np.nan
    return factor, fwd_ret


class TestCrossSectionBreadthRule:
    """A fixed 30-asset floor made every IC NaN in a 30-stock universe.

    The shared rule is ``min(30, max(10, panel_width // 2))``: a 30-asset
    panel with ~27 live names is evaluable, a 5-asset panel is not, and a
    sparse slice of a 500-asset panel is still held to 30.
    """

    def test_threshold_scales_with_the_panel_but_keeps_a_floor(self):
        assert min_assets_for_ic(30) == 15
        assert min_assets_for_ic(500) == 30
        assert min_assets_for_ic(5) == 10
        assert min_assets_for_ic(0) == 10
        assert min_assets_for_ic(60) == 30
        assert min_assets_for_ic(19) == 10

    def test_30_asset_universe_with_27_live_names_is_evaluable(self):
        factor, fwd_ret = _make_sparse_panel(260, 30, 27)

        ic = rank_ic(factor, fwd_ret)
        summary = ic_summary(ic)

        assert len(ic) > 100
        assert np.isfinite(ic).all()
        assert summary["n_obs"] > 100
        assert np.isfinite(summary["mean_ic"])

        p_ic = pearson_ic(factor, fwd_ret)
        assert len(p_ic) > 100
        assert np.isfinite(p_ic).all()

    def test_5_asset_universe_has_no_ic_and_warns(self, caplog):
        factor, fwd_ret = _make_factor_and_returns(n_dates=60, n_assets=5)

        with caplog.at_level(
            logging.WARNING, logger="quant_platform.factors.evaluation"
        ):
            ic = rank_ic(factor, fwd_ret, name="five_name_panel")

        assert len(ic) == 0
        assert np.isnan(ic_summary(ic)["mean_ic"])

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert any(
            "five_name_panel" in message
            and "no cross-section has enough assets for IC" in message
            and "best N=5" in message
            and "required=10" in message
            for message in messages
        )

    def test_sparse_500_asset_universe_is_not_gamed_by_12_names(self, caplog):
        factor, fwd_ret = _make_sparse_panel(40, 500, 12)

        with caplog.at_level(
            logging.WARNING, logger="quant_platform.factors.evaluation"
        ):
            ic = rank_ic(factor, fwd_ret)

        assert len(ic) == 0

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "best N=12, required=30" in message for message in messages
        )
