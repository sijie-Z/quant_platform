"""Tests for operations.nav — NAV calculation with fee accrual."""

from datetime import datetime, timedelta

import pytest

from quant_platform.core.store import Store
from quant_platform.operations.nav import NAV, NAVCalculator

# Use recent dates so get_nav_history() filters work
_TODAY = datetime.now()
_D1 = (_TODAY - timedelta(days=3)).strftime("%Y-%m-%d")
_D2 = (_TODAY - timedelta(days=2)).strftime("%Y-%m-%d")
_D3 = (_TODAY - timedelta(days=1)).strftime("%Y-%m-%d")
_D4 = _TODAY.strftime("%Y-%m-%d")


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Store(db_path)


@pytest.fixture
def calc(store):
    return NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.20)


# ── NAV Dataclass ──


class TestNAVDataclass:
    def test_to_dict_rounds_values(self):
        nav = NAV(
            date="2024-07-15", nav_total=10_050_000.123456,
            nav_per_unit=1.005000123, total_units=10_000_000,
            cash=5_000_000.567, market_value=5_050_000.89,
            mgmt_fee=550.123, perf_fee=1000.456,
            high_water_mark=1.005, daily_return=0.005, cumulative_return=0.005,
        )
        d = nav.to_dict()
        assert d["date"] == "2024-07-15"
        assert d["nav_per_unit"] == 1.005
        assert d["nav_total"] == 10_050_000.1235
        assert d["daily_return"] == 0.005


# ── NAVCalculator Init ──


class TestNAVCalculatorInit:
    def test_default_values(self, store):
        calc = NAVCalculator(store)
        assert calc._annual_mgmt_fee == 0.02
        assert calc._perf_fee_rate == 0.20
        assert calc._total_units == 10_000_000
        assert calc._high_water_mark == 1.0

    def test_custom_values(self, store):
        calc = NAVCalculator(
            store, annual_mgmt_fee=0.015, perf_fee_rate=0.15,
            initial_nav_per_unit=1.0, initial_units=5_000_000,
        )
        assert calc._annual_mgmt_fee == 0.015
        assert calc._total_units == 5_000_000

    def test_restore_state_from_history(self, store):
        store.save_nav({
            "date": _D4, "nav_total": 10_100_000,
            "nav_per_unit": 1.01, "total_units": 10_000_000,
            "high_water_mark": 1.02,
        })
        calc = NAVCalculator(store)
        assert calc._high_water_mark == 1.02


# ── Daily NAV Calculation ──


class TestCalculateDailyNav:
    def test_basic_nav(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        assert nav.date == _D2
        assert nav.nav_per_unit > 0
        assert nav.total_units == 10_000_000

    def test_mgmt_fee_positive(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        assert nav.mgmt_fee > 0
        # Daily mgmt fee = 10M * 0.02 / 252 ≈ 793.65
        assert 700 < nav.mgmt_fee < 900

    def test_perf_fee_zero_when_below_hwm(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        # First call: nav_per_unit < 1.0 (due to mgmt fee) → no perf fee
        assert nav.perf_fee == 0.0

    def test_perf_fee_positive_when_above_hwm(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.20)
        # First call sets HWM
        calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        # Second call with higher value → perf fee
        nav = calc.calculate_daily_nav(
            date=_D3, cash=5_000_000, market_value=5_200_000,
        )
        assert nav.perf_fee > 0

    def test_nav_total_less_than_gross(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        gross = nav.cash + nav.market_value - nav.mgmt_fee
        assert nav.nav_total <= gross

    def test_hwm_updates(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
        nav1 = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        assert calc._high_water_mark == 1.0

        nav2 = calc.calculate_daily_nav(
            date=_D3, cash=5_000_000, market_value=5_100_000,
        )
        assert calc._high_water_mark > 1.0

    def test_daily_return_computed(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
        nav1 = calc.calculate_daily_nav(
            date=_D3, cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(nav1)

        nav2 = calc.calculate_daily_nav(
            date=_D4, cash=5_000_000, market_value=5_100_000,
        )
        assert nav2.daily_return > 0

    def test_cumulative_return(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        # With mgmt fee, cumulative return is slightly negative
        assert isinstance(nav.cumulative_return, float)


# ── Save and Restore ──


class TestSaveNav:
    def test_save_and_retrieve(self, calc, store):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(nav)
        history = store.get_nav_history(days=7)
        assert len(history) == 1
        assert history[0]["date"] == _D2

    def test_update_daily_nav_saves(self, calc, store):
        nav = calc.update_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        history = store.get_nav_history(days=7)
        assert len(history) == 1
        assert history[0]["nav_per_unit"] == pytest.approx(nav.nav_per_unit, abs=1e-4)


# ── Fee Accrual ──


class TestFeeAccrual:
    def test_mgmt_fee_formula(self, calc):
        fee = calc._accrue_mgmt_fee(10_000_000)
        expected = 10_000_000 * 0.02 / 252
        assert fee == pytest.approx(expected, rel=1e-6)

    def test_perf_fee_zero_below_hwm(self, calc):
        calc._high_water_mark = 1.0
        fee = calc._accrue_perf_fee(0.99)
        assert fee == 0.0

    def test_perf_fee_positive_above_hwm(self, calc):
        calc._high_water_mark = 1.0
        fee = calc._accrue_perf_fee(1.05)
        # gain = 0.05, units = 10M, rate = 0.20
        expected = 0.05 * 10_000_000 * 0.20
        assert fee == pytest.approx(expected, rel=1e-6)

    def test_perf_fee_at_hwm_boundary(self, calc):
        calc._high_water_mark = 1.0
        fee = calc._accrue_perf_fee(1.0)
        assert fee == 0.0


# ── Edge Cases ──


class TestNAVCases:
    def test_zero_portfolio_value(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=0, market_value=0,
        )
        assert nav.nav_per_unit >= 0

    def test_large_portfolio(self, store):
        calc = NAVCalculator(store, initial_units=1_000_000_000)
        nav = calc.calculate_daily_nav(
            date=_D2, cash=500_000_000, market_value=500_000_000,
        )
        assert nav.nav_per_unit > 0

    def test_multiple_days_sequence(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
        for i in range(5):
            d = (_TODAY - timedelta(days=5 - i)).strftime("%Y-%m-%d")
            nav = calc.calculate_daily_nav(
                date=d,
                cash=5_000_000,
                market_value=5_000_000 + i * 10_000,
            )
            calc.save_nav(nav)

        history = store.get_nav_history(days=30)
        assert len(history) == 5
