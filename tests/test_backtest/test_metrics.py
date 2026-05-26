"""Tests for performance metrics."""

import numpy as np
import pandas as pd
from quant_platform.backtest.metrics import (
    all_metrics,
    annualized_return,
    annualized_volatility,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)


def _make_returns(n_days=500, annual_ret=0.10, annual_vol=0.20):
    np.random.seed(42)
    daily_ret = annual_ret / 252 + annual_vol / np.sqrt(252) * np.random.randn(n_days)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    return pd.Series(daily_ret, index=dates)


def test_annualized_return():
    sr = _make_returns()
    ar = annualized_return(sr)
    # Should be around 10% (with noise)
    assert -0.2 < ar < 0.4


def test_annualized_volatility():
    sr = _make_returns(annual_vol=0.20)
    av = annualized_volatility(sr)
    assert 0.10 < av < 0.30


def test_sharpe_ratio():
    sr = _make_returns(annual_ret=0.15, annual_vol=0.20)
    sh = sharpe_ratio(sr, risk_free=0.03)
    # Sharpe = (0.15 - 0.03) / 0.20 = 0.6 (approximate)
    assert 0.2 < sh < 1.2


def test_max_drawdown_negative():
    sr = _make_returns(annual_ret=-0.05)
    mdd, peak, trough = max_drawdown(sr)
    assert mdd < 0  # Negative return -> negative drawdown


def test_win_rate_range():
    sr = _make_returns()
    wr = win_rate(sr)
    assert 0.3 < wr < 0.7


def test_all_metrics_returns_dict():
    sr = _make_returns()
    metrics = all_metrics(sr)
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert metrics["total_days"] == len(sr)


def test_information_ratio():
    sr = _make_returns(annual_ret=0.12)
    bench = _make_returns(annual_ret=0.06, annual_vol=0.15)
    ir = information_ratio(sr, bench)
    assert isinstance(ir, float)
