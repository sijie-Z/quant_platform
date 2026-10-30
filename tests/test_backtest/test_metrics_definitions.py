"""Metric definitions: Sortino downside deviation and inception-based drawdown."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.backtest.metrics import max_drawdown, sortino_ratio


def test_max_drawdown_counts_a_first_day_loss():
    returns = pd.Series(
        [-0.10, 0.0, 0.0], index=pd.bdate_range("2024-01-01", periods=3),
    )

    drawdown, _, _ = max_drawdown(returns)

    assert drawdown == pytest.approx(-0.10)


def test_max_drawdown_compounds_from_inception():
    returns = pd.Series(
        [-0.10, -0.10], index=pd.bdate_range("2024-01-01", periods=2),
    )

    drawdown, _, _ = max_drawdown(returns)

    # 1 -> 0.9 -> 0.81 is a 19% peak-to-trough decline, not 10%.
    assert drawdown == pytest.approx(-0.19)


def test_sortino_uses_downside_deviation_not_loss_dispersion():
    """A profitable series with constant-sized losses must not score 0."""
    values = np.full(300, 0.001)
    values[::5] = -0.002
    returns = pd.Series(
        values, index=pd.bdate_range("2023-01-02", periods=300),
    )

    ratio = sortino_ratio(returns, risk_free=0.0)

    # The old implementation took std() of the negative subset only, which is
    # exactly 0 for constant-sized losses, so it returned 0.0.
    assert np.isfinite(ratio)
    assert ratio > 1.0


def test_sortino_handles_a_single_large_loss():
    values = np.zeros(252)
    values[100] = -0.20
    returns = pd.Series(
        values, index=pd.bdate_range("2023-01-02", periods=252),
    )

    ratio = sortino_ratio(returns, risk_free=0.0)

    assert np.isfinite(ratio)
    assert ratio < 0  # annual return is negative, downside deviation positive
