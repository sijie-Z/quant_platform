"""Tests for operations.investor — investor-facing fund performance portal."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from quant_platform.core.store import Store
from quant_platform.operations.investor import InvestorPortal, InvestorView
from quant_platform.operations.nav import NAVCalculator


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Store(db_path)


@pytest.fixture
def populated_store(store):
    """Store with 60 days of NAV history (recent dates)."""
    calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
    today = datetime.now()
    for i in range(60):
        d = (today - timedelta(days=60 - i)).strftime("%Y-%m-%d")
        mv = 5_000_000 + i * 10_000
        nav = calc.calculate_daily_nav(
            date=d,
            cash=5_000_000,
            market_value=mv,
        )
        calc.save_nav(nav)
    return store


@pytest.fixture
def portal(populated_store):
    return InvestorPortal(populated_store, fund_name="测试基金")


# ── InvestorView Dataclass ──


class TestInvestorView:
    def test_to_dict_keys(self):
        view = InvestorView(
            nav_curve=pd.DataFrame(columns=["date", "nav_per_unit", "daily_return"]),
            cumulative_return=0.05,
            annualized_return=0.10,
            max_drawdown=-0.03,
            sharpe_ratio=1.5,
            volatility=0.15,
            monthly_returns=pd.DataFrame(),
            fund_name="Test Fund",
            inception_date="2024-01-01",
            latest_nav=1.05,
            aum=10_500_000,
        )
        d = view.to_dict()
        assert "fund_name" in d
        assert "cumulative_return" in d
        assert "annualized_return" in d
        assert "max_drawdown" in d
        assert "sharpe_ratio" in d
        assert "nav_curve" in d
        assert "monthly_returns" in d

    def test_to_dict_rounds_values(self):
        view = InvestorView(
            nav_curve=pd.DataFrame(),
            cumulative_return=0.05123456,
            annualized_return=0.10654321,
            max_drawdown=-0.03456789,
            sharpe_ratio=1.56789,
            volatility=0.154321,
            monthly_returns=pd.DataFrame(),
        )
        d = view.to_dict()
        assert d["cumulative_return"] == 0.0512
        assert d["sharpe_ratio"] == 1.5679


# ── InvestorPortal ──


class TestInvestorPortal:
    def test_get_investor_view_returns_view(self, portal):
        view = portal.get_investor_view()
        assert isinstance(view, InvestorView)

    def test_nav_curve_has_columns(self, portal):
        view = portal.get_investor_view()
        assert "date" in view.nav_curve.columns
        assert "nav_per_unit" in view.nav_curve.columns
        assert "daily_return" in view.nav_curve.columns

    def test_cumulative_return_positive(self, portal):
        view = portal.get_investor_view()
        # NAV increases due to market_value growth
        assert view.cumulative_return >= 0

    def test_annualized_return(self, portal):
        view = portal.get_investor_view()
        assert isinstance(view.annualized_return, float)

    def test_max_drawdown_non_negative(self, portal):
        view = portal.get_investor_view()
        # max_drawdown is 0 or negative
        assert view.max_drawdown <= 0

    def test_sharpe_ratio(self, portal):
        view = portal.get_investor_view()
        assert isinstance(view.sharpe_ratio, float)

    def test_volatility_positive(self, portal):
        view = portal.get_investor_view()
        assert view.volatility >= 0

    def test_monthly_returns_shape(self, portal):
        view = portal.get_investor_view()
        assert isinstance(view.monthly_returns, pd.DataFrame)
        # Should have 12 month columns
        if not view.monthly_returns.empty:
            assert len(view.monthly_returns.columns) == 12

    def test_fund_name_set(self, portal):
        view = portal.get_investor_view()
        assert view.fund_name == "测试基金"

    def test_latest_nav(self, portal):
        view = portal.get_investor_view()
        assert view.latest_nav > 0

    def test_inception_date(self, portal):
        view = portal.get_investor_view()
        assert view.inception_date != ""


# ── Empty Data ──


class TestEmptyData:
    def test_empty_store_returns_empty_view(self, store):
        portal = InvestorPortal(store)
        view = portal.get_investor_view()
        assert view.cumulative_return == 0.0
        assert view.max_drawdown == 0.0
        assert view.sharpe_ratio == 0.0
        assert view.nav_curve.empty

    def test_single_nav_record(self, store):
        today = datetime.now().strftime("%Y-%m-%d")
        store.save_nav({
            "date": today, "nav_total": 10_000_000,
            "nav_per_unit": 1.0, "total_units": 10_000_000,
        })
        portal = InvestorPortal(store)
        view = portal.get_investor_view()
        # Need at least 2 records
        assert view.cumulative_return == 0.0


# ── Monthly Returns ──


class TestMonthlyReturns:
    def test_compute_monthly_returns(self, portal):
        view = portal.get_investor_view()
        if not view.monthly_returns.empty:
            # Check index is years
            for year in view.monthly_returns.index:
                assert isinstance(year, int)
            # Check columns are M1..M12
            for col in view.monthly_returns.columns:
                assert col.startswith("M")

    def test_monthly_returns_values(self, portal):
        view = portal.get_investor_view()
        if not view.monthly_returns.empty:
            for col in view.monthly_returns.columns:
                for val in view.monthly_returns[col].dropna():
                    assert isinstance(val, float)


# ── Custom Fund Name ──


class TestCustomFundName:
    def test_custom_name(self, populated_store):
        portal = InvestorPortal(populated_store, fund_name="量化Alpha基金")
        view = portal.get_investor_view()
        assert view.fund_name == "量化Alpha基金"

    def test_default_name(self, populated_store):
        portal = InvestorPortal(populated_store)
        view = portal.get_investor_view()
        assert view.fund_name == "量化多因子基金"
