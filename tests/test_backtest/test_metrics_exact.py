"""Exact metric formulas: Sharpe and Sortino must subtract the risk-free rate.

``test_sharpe_ratio`` only asserted a wide range (0.2 < sh < 1.2), which admits
both the correct value and the one produced when the numerator drops
``risk_free`` entirely. These tests pin the published formulas to 1e-12 on a
constructed series and cover the zero-volatility edge.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.backtest.metrics import (
    annualization_years,
    annualized_return,
    annualized_volatility,
    infer_periods_per_year,
    sharpe_ratio,
    sortino_ratio,
)

TRADING_DAYS = 252
RISK_FREE = 0.03


def _returns(seed: int = 123, n_days: int = 252) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0005, 0.01, n_days)
    return pd.Series(values, index=pd.bdate_range("2023-01-02", periods=n_days))


def _expected_sharpe(series: pd.Series, risk_free: float) -> float:
    values = series.to_numpy()
    years = len(values) / TRADING_DAYS
    ann_ret = (1.0 + values).prod() ** (1.0 / years) - 1.0
    ann_vol = np.std(values, ddof=1) * np.sqrt(TRADING_DAYS)
    return (ann_ret - risk_free) / ann_vol


def _expected_sortino(series: pd.Series, risk_free: float) -> float:
    values = series.to_numpy()
    years = len(values) / TRADING_DAYS
    ann_ret = (1.0 + values).prod() ** (1.0 / years) - 1.0
    daily_mar = risk_free / TRADING_DAYS
    downside = np.minimum(values - daily_mar, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS)
    return (ann_ret - risk_free) / downside_dev


class TestSharpeFormula:
    def test_sharpe_equals_excess_return_over_annualized_volatility(self):
        series = _returns()

        sharpe = sharpe_ratio(
            series, risk_free=RISK_FREE, periods_per_year=TRADING_DAYS,
        )

        assert sharpe == pytest.approx(_expected_sharpe(series, RISK_FREE), rel=1e-12)
        # B12/M15 made the annualization basis calendar-inferred by default;
        # this pins the published formula on the explicit 252-day override.
        assert sharpe == pytest.approx(
            (
                annualized_return(series, periods_per_year=TRADING_DAYS)
                - RISK_FREE
            ) / annualized_volatility(series, periods_per_year=TRADING_DAYS),
            rel=1e-12,
        )

    def test_sharpe_without_the_risk_free_subtraction_is_a_different_number(self):
        """Guard against a tautological test: the mutant's value must differ."""
        series = _returns()

        correct = sharpe_ratio(
            series, risk_free=RISK_FREE, periods_per_year=TRADING_DAYS,
        )
        mutant = (
            annualized_return(series, periods_per_year=TRADING_DAYS)
            / annualized_volatility(series, periods_per_year=TRADING_DAYS)
        )

        assert abs(correct - mutant) > 0.05

    def test_zero_volatility_returns_zero(self):
        flat = pd.Series(
            np.full(TRADING_DAYS, 0.001),
            index=pd.bdate_range("2023-01-02", periods=TRADING_DAYS),
        )
        # Floating-point cancellation leaves ~1e-18, below the 1e-10 guard.
        assert annualized_volatility(flat) < 1e-10
        assert sharpe_ratio(flat, risk_free=RISK_FREE) == 0.0

    def test_zero_returns_and_zero_risk_free_are_not_a_division_by_zero(self):
        flat = pd.Series(
            np.zeros(TRADING_DAYS),
            index=pd.bdate_range("2023-01-02", periods=TRADING_DAYS),
        )
        assert sharpe_ratio(flat, risk_free=0.0) == 0.0


class TestSortinoFormula:
    def test_sortino_equals_excess_return_over_downside_deviation(self):
        series = _returns(seed=7)

        sortino = sortino_ratio(
            series, risk_free=RISK_FREE, periods_per_year=TRADING_DAYS,
        )

        assert sortino == pytest.approx(
            _expected_sortino(series, RISK_FREE), rel=1e-12,
        )

    def test_sortino_without_the_risk_free_subtraction_is_a_different_number(self):
        series = _returns(seed=7)

        correct = sortino_ratio(
            series, risk_free=RISK_FREE, periods_per_year=TRADING_DAYS,
        )
        ann_ret = annualized_return(series, periods_per_year=TRADING_DAYS)
        daily_mar = RISK_FREE / TRADING_DAYS
        downside = np.minimum(series.to_numpy() - daily_mar, 0.0)
        downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS)
        mutant = ann_ret / downside_dev

        assert abs(correct - mutant) > 0.05

    def test_zero_downside_deviation_returns_zero(self):
        flat = pd.Series(
            np.full(TRADING_DAYS, 0.001),
            index=pd.bdate_range("2023-01-02", periods=TRADING_DAYS),
        )
        assert sortino_ratio(flat, risk_free=0.0) == 0.0


class TestCalendarAwareAnnualization:
    """B12/M15: years come from the index span, not ``len / 252``."""

    def test_969_a_share_sessions_span_four_calendar_years(self):
        from quant_platform.utils.exchange_calendar import trading_days

        dates = trading_days("2021-01-04", "2024-12-31")
        assert len(dates) == 969  # 243 + 242 + 242 + 242 sessions
        values = pd.Series(0.0, index=dates)

        years = annualization_years(values)
        assert years == pytest.approx(4.0, abs=0.02)
        assert infer_periods_per_year(values) == pytest.approx(
            len(dates) / 4.0, rel=0.01,
        )

    def test_calendar_daily_365_observations_are_one_year(self):
        dates = pd.date_range("2023-01-01", periods=365, freq="D")
        values = pd.Series(0.0001, index=dates)

        years = annualization_years(values)
        assert years == pytest.approx(1.0, abs=0.01)
        assert infer_periods_per_year(values) == pytest.approx(365.0, rel=0.01)

        # A constant 1 bp/day compounds to ~3.72% over a calendar year; the
        # old 252-observation year reported ~2.55%.
        assert annualized_return(values) == pytest.approx(
            (1.0001 ** 365) - 1, rel=0.01,
        )
