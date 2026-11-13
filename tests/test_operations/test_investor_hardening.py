"""Regression tests for operations.investor (R46-R51).

Covers the common volatility/Sharpe basis, one stated window for every
headline metric, the dormant-fund fallback, JSON serialization, the
documented 10-year store clamp and the tiny-sample annualization guard.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from quant_platform.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
)
from quant_platform.operations.investor import InvestorPortal


class FakeNavStore:
    """Store double with get_nav_history's window semantics."""

    def __init__(self, records):
        self.records = sorted(records, key=lambda r: r["date"])

    def get_nav_history(self, days=365 * 3):
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [r for r in self.records if r["date"] > cutoff]


def _daily_history(n_days=1500):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    records = []
    for i in range(n_days):
        day = today - timedelta(days=n_days - 1 - i)
        nav = 1.0 + 0.00003 * i + 0.01 * math.sin(i / 17.0)
        records.append({
            "date": day.strftime("%Y-%m-%d"),
            "nav_per_unit": nav,
            "nav_total": nav * 10_000_000.0,
            "daily_return": 0.0,
        })
    return records


def _series(records):
    return pd.Series(
        [r["nav_per_unit"] for r in records],
        index=pd.to_datetime([r["date"] for r in records]),
    ).sort_index()


class TestVolatilitySharpeBasis:
    def test_displayed_volatility_is_the_sharpe_denominator(self):
        """R46: sqrt(252) for display vs inferred ~365 for Sharpe made the
        two metrics disagree by ~17% on the same series."""
        records = _daily_history()
        view = InvestorPortal(FakeNavStore(records)).get_investor_view()

        returns = _series(records).pct_change().dropna()

        assert view.volatility == pytest.approx(
            float(annualized_volatility(returns)), rel=1e-12,
        )
        assert view.sharpe_ratio == pytest.approx(
            float(sharpe_ratio(returns)), rel=1e-12,
        )
        # The displayed volatility is exactly the Sharpe denominator.
        assert view.sharpe_ratio == pytest.approx(
            (float(annualized_return(returns)) - 0.03) / view.volatility,
            rel=1e-12,
        )


class TestSingleWindow:
    def test_all_headline_metrics_use_the_same_window(self):
        """R47: cumulative return used to be full-life while Sharpe used the
        3-year lookback."""
        records = _daily_history()
        view = InvestorPortal(FakeNavStore(records)).get_investor_view(
            days=365,
        )

        window_records = [
            r for r in records
            if r["date"] > (
                datetime.now() - timedelta(days=365)
            ).strftime("%Y-%m-%d")
        ]
        series = _series(window_records)

        assert view.window_days == 365
        assert view.cumulative_return == pytest.approx(
            float(series.iloc[-1] / series.iloc[0] - 1), rel=1e-12,
        )
        # Drawdown is computed on the same window, not on older history.
        peak = series.expanding().max()
        assert view.max_drawdown == pytest.approx(
            float(((series - peak) / peak).min()), rel=1e-12,
        )
        returns = series.pct_change().dropna()
        assert view.sharpe_ratio == pytest.approx(
            float(sharpe_ratio(returns)), rel=1e-12,
        )

    def test_default_window_is_the_full_store_history(self):
        records = _daily_history()
        view = InvestorPortal(FakeNavStore(records)).get_investor_view()
        assert view.window_days == InvestorPortal.MAX_HISTORY_DAYS


class TestDormantFundFallback:
    def test_empty_lookback_falls_back_to_full_history(self):
        """R48: a fund dormant for >3 years returned an all-zero view even
        though the store held its history."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = today - timedelta(days=4 * 365)
        records = [
            {"date": start.strftime("%Y-%m-%d"), "nav_per_unit": 1.00},
            {
                "date": (start + timedelta(days=30)).strftime("%Y-%m-%d"),
                "nav_per_unit": 1.05,
            },
        ]

        view = InvestorPortal(FakeNavStore(records)).get_investor_view(
            days=365 * 3,
        )

        assert view.window_days == InvestorPortal.MAX_HISTORY_DAYS
        assert view.cumulative_return == pytest.approx(0.05)
        assert view.latest_nav == pytest.approx(1.05)
        assert not view.nav_curve.empty


class TestJsonSerialization:
    def test_view_is_json_serializable(self):
        """R49: pandas Timestamps and the first NaN daily_return made
        json.dumps(view.to_dict()) raise TypeError."""
        records = _daily_history()
        view = InvestorPortal(FakeNavStore(records)).get_investor_view()

        payload = json.dumps(view.to_dict())
        decoded = json.loads(payload)

        assert all(
            isinstance(row["date"], str) for row in decoded["nav_curve"]
        )
        assert decoded["nav_curve"][0]["daily_return"] is None
        assert decoded["window_days"] == InvestorPortal.MAX_HISTORY_DAYS

    def test_empty_view_is_json_serializable(self):
        view = InvestorPortal(FakeNavStore([])).get_investor_view()
        json.dumps(view.to_dict())


class TestHistoryClamp:
    def test_store_clamp_is_documented_and_applied(self):
        """R50: the store clamps every window to 3650 days; a 12-year-old
        record cannot be labelled inception, and the portal says so."""
        assert InvestorPortal.MAX_HISTORY_DAYS == 3650

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        records = [
            {
                "date": (today - timedelta(days=12 * 365)).strftime("%Y-%m-%d"),
                "nav_per_unit": 0.5,
            },
            {
                "date": (today - timedelta(days=365)).strftime("%Y-%m-%d"),
                "nav_per_unit": 1.0,
            },
            {
                "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "nav_per_unit": 1.1,
            },
        ]

        view = InvestorPortal(FakeNavStore(records)).get_investor_view()

        assert view.inception_date == (
            today - timedelta(days=365)
        ).strftime("%Y-%m-%d")

    def test_requesting_more_than_the_clamp_reports_the_clamp(self):
        view = InvestorPortal(FakeNavStore(_daily_history(30))).get_investor_view(
            days=365 * 30,
        )
        assert view.window_days == InvestorPortal.MAX_HISTORY_DAYS


class TestInsufficientHistory:
    def test_two_records_do_not_annualize_or_report_volatility(self):
        """R51: 2 NAVs and a 1% one-day move once reported 3677%/yr."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        records = [
            {
                "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "nav_per_unit": 1.0,
            },
            {
                "date": today.strftime("%Y-%m-%d"),
                "nav_per_unit": 1.01,
            },
        ]

        view = InvestorPortal(FakeNavStore(records)).get_investor_view()

        assert view.cumulative_return == pytest.approx(0.01)
        assert view.annualized_return == 0.0
        assert view.volatility == 0.0
        assert view.sharpe_ratio == 0.0

    def test_three_records_do_not_report_an_absurd_sharpe(self):
        """The tiny-sample guard must also cover vol/Sharpe, not just the
        annualized return: 3 daily NAVs imply a ~27,000 Sharpe."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        records = [
            {
                "date": (today - timedelta(days=2)).strftime("%Y-%m-%d"),
                "nav_per_unit": 1.00,
            },
            {
                "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "nav_per_unit": 1.01,
            },
            {
                "date": today.strftime("%Y-%m-%d"),
                "nav_per_unit": 1.02,
            },
        ]

        view = InvestorPortal(FakeNavStore(records)).get_investor_view()

        assert view.annualized_return == 0.0
        assert view.volatility == 0.0
        assert view.sharpe_ratio == 0.0


class TestScalarNaNHandling:
    def test_nan_scalars_serialize_as_null(self):
        """A NaN AUM/latest NAV must not emit non-standard JSON tokens."""
        from quant_platform.operations.investor import InvestorView

        view = InvestorView(
            nav_curve=pd.DataFrame(
                columns=["date", "nav_per_unit", "daily_return"]
            ),
            cumulative_return=float("nan"),
            annualized_return=float("nan"),
            max_drawdown=float("nan"),
            sharpe_ratio=float("inf"),
            volatility=float("nan"),
            monthly_returns=pd.DataFrame(),
            latest_nav=float("nan"),
            aum=float("nan"),
        )

        decoded = json.loads(json.dumps(view.to_dict()))

        assert decoded["aum"] is None
        assert decoded["latest_nav"] is None
        assert decoded["cumulative_return"] is None
        assert decoded["sharpe_ratio"] is None
