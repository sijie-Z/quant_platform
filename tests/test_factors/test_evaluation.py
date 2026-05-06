"""Tests for factor evaluation."""

import numpy as np
import pandas as pd

from quant_platform.factors.evaluation import (
    rank_ic,
    pearson_ic,
    ic_summary,
    quantile_returns,
    factor_correlation,
    factor_turnover,
    ic_decay,
)


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
