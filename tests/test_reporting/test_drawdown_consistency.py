"""R88: the saved drawdown chart must be a drawdown of the equity curve.

The old implementation built "wealth" from per-day growth factors without
compounding them, so the plotted series was the dispersion of daily returns
(-0.98% on a strictly rising series, -7.3% on results/daily_returns.csv
instead of -56.5%) rather than a drawdown of the compounded equity curve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest.metrics import max_drawdown
from quant_platform.reporting.performance import plot_drawdown


def _plotted_drawdown(fig: plt.Figure) -> np.ndarray:
    """The underwater line's y-values, converted from percent to fraction."""
    line = fig.axes[0].lines[0]
    return np.asarray(line.get_ydata(), dtype=float) / 100.0


def test_strictly_rising_series_has_zero_drawdown():
    dates = pd.bdate_range("2023-01-02", periods=20)
    returns = pd.Series([0.01, 0.02] * 10, index=dates)

    fig = plot_drawdown(returns)
    try:
        plotted = _plotted_drawdown(fig)
        assert plotted.max() == pytest.approx(0.0, abs=1e-12)
        assert max_drawdown(returns)[0] == pytest.approx(0.0, abs=1e-12)
    finally:
        plt.close(fig)


def test_random_series_matches_metrics_max_drawdown():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=500)
    returns = pd.Series(rng.normal(0.0002, 0.015, 500), index=dates)

    fig = plot_drawdown(returns)
    try:
        plotted_min = float(_plotted_drawdown(fig).min())
        expected, _, _ = max_drawdown(returns)
        assert plotted_min == pytest.approx(expected, abs=1e-12)
        # Guard against the old per-day-growth "drawdown": the two numbers
        # differed by a factor of ~1.5 on this series.
        assert plotted_min < -0.01
    finally:
        plt.close(fig)


def test_first_day_loss_is_visible():
    dates = pd.bdate_range("2023-01-02", periods=3)
    returns = pd.Series([-0.10, 0.005, 0.005], index=dates)

    fig = plot_drawdown(returns)
    try:
        assert float(_plotted_drawdown(fig).min()) == pytest.approx(-0.10, abs=1e-12)
        assert max_drawdown(returns)[0] == pytest.approx(-0.10, abs=1e-12)
    finally:
        plt.close(fig)


def test_nan_returns_are_treated_as_no_move():
    dates = pd.bdate_range("2023-01-02", periods=3)
    returns = pd.Series([np.nan, 0.01, -0.02], index=dates)

    fig = plot_drawdown(returns)
    try:
        plotted = _plotted_drawdown(fig)
        # Dropping a NaN return (metrics) and filling it with 0 (chart) both
        # leave wealth unchanged, so the chart still matches metrics exactly.
        expected, _, _ = max_drawdown(returns)
        assert float(plotted.min()) == pytest.approx(expected, abs=1e-12)
    finally:
        plt.close(fig)
