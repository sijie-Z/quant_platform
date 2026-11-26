"""Tests for operations.nav — NAV calculation with fee accrual."""

from datetime import datetime, timedelta

import pandas as pd
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
    def test_to_dict_persists_full_precision(self):
        """R44: persistence must not round before the next day re-reads it.

        nav_per_unit/HWM at 4dp and fees at 2dp produced spurious daily
        returns on unchanged NAVs (4.5e-5) and fee/HWM drift. Rounding is a
        presentation concern only.
        """
        nav = NAV(
            date="2024-07-15", nav_total=10_050_000.123456,
            nav_per_unit=1.005000123, total_units=10_000_000,
            cash=5_000_000.567, market_value=5_050_000.89,
            mgmt_fee=550.123456, perf_fee=1000.456789,
            high_water_mark=1.005000123, daily_return=0.005,
            cumulative_return=0.005,
        )
        d = nav.to_dict()
        assert d["date"] == "2024-07-15"
        assert d["nav_per_unit"] == 1.005000123
        assert d["nav_total"] == 10_050_000.123456
        assert d["mgmt_fee"] == 550.123456
        assert d["perf_fee"] == 1000.456789
        assert d["high_water_mark"] == 1.005000123
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

    def test_restore_state_after_a_date_gap(self, store):
        """Regression: restore asked for a window ending *today*.

        `_restore_state` used `get_nav_history(days=1)`, whose filter is
        `date > (now - 1 day)`. A record written on an earlier calendar day
        does not satisfy a strict `>`, so the lookup only ever found a record
        written today; any restart on a new day reset the high-water mark to
        1.0 and booked a performance fee against investors still below high
        water.

        The existing test above passes today's date, which is exactly the one
        case the old code handled -- so it could never catch this.
        """
        older = (_TODAY - timedelta(days=5)).strftime("%Y-%m-%d")
        store.save_nav({
            "date": older, "nav_total": 10_800_000,
            "nav_per_unit": 1.08, "total_units": 12_000_000,
            "high_water_mark": 1.20,
        })
        calc = NAVCalculator(store)
        assert calc._high_water_mark == 1.20
        assert calc._total_units == 12_000_000

    def test_restore_state_on_empty_store(self, store):
        """No history must leave the defaults alone, not raise."""
        calc = NAVCalculator(store)
        assert calc._high_water_mark == 1.0


# ── Daily NAV Calculation ──


class TestCalculateDailyNav:
    def test_basic_nav(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        assert nav.date == _D2
        assert nav.total_units == 10_000_000
        # Exact arithmetic, not just "positive": the only deduction on the
        # first day is the management fee.
        assert nav.nav_total == pytest.approx(10_000_000 - nav.mgmt_fee, rel=1e-9)
        assert nav.nav_per_unit == pytest.approx(
            nav.nav_total / nav.total_units, rel=1e-12,
        )

    def test_mgmt_fee_positive(self, calc):
        nav = calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        assert nav.mgmt_fee > 0
        # Daily mgmt fee = 10M * 0.02 / 365 ≈ 547.95 (calendar-day accrual)
        assert 500 < nav.mgmt_fee < 600

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
        assert nav.nav_total == pytest.approx(
            10_000_000 - nav.accrued_mgmt_fee - nav.accrued_perf_fee, rel=1e-9,
        )
        assert nav.nav_total <= nav.cash + nav.market_value

    def test_hwm_updates(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
        calc.calculate_daily_nav(
            date=_D2, cash=5_000_000, market_value=5_000_000,
        )
        assert calc._high_water_mark == 1.0

        calc.calculate_daily_nav(
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
        assert nav.cumulative_return == pytest.approx(
            nav.nav_per_unit / 1.0 - 1.0, rel=1e-12,
        )
        assert nav.cumulative_return < 0  # management fee on day one


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
        expected = 10_000_000 * 0.02 / 365
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


class TestFeeAccrualAcrossDays:
    def test_management_fee_is_deducted_and_compounds(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.0)
        dates = pd.bdate_range(end=_TODAY.strftime("%Y-%m-%d"), periods=5)

        navs = []
        for day in dates:
            nav = calc.calculate_daily_nav(
                date=str(day.date()), cash=5_000_000, market_value=5_000_000,
            )
            calc.save_nav(nav)
            navs.append(nav)

        # The fee is a liability: NAV must decline every day. The accrual
        # runs on calendar days (annual rate / 365), so the elapsed gap over
        # a weekend is two extra days.
        assert navs[-1].nav_total < navs[0].nav_total
        base = 10_000_000.0
        prev_day = None
        for day in dates:
            days = 1 if prev_day is None else max(1, (day - prev_day).days)
            base -= base * 0.02 / 365 * days
            prev_day = day
        assert navs[-1].nav_total == pytest.approx(base, rel=1e-6)

    def test_performance_fee_is_not_added_back_on_the_next_day(self, store):
        """The accrued fee is a liability.

        Only the current day's fee was subtracted, so after any
        crystallisation a flat day reported the whole prior fee as new profit
        and NAV/unit jumped.
        """
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.20)
        dates = pd.bdate_range(end=_TODAY.strftime("%Y-%m-%d"), periods=3)

        nav1 = calc.calculate_daily_nav(
            date=str(dates[0].date()), cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(nav1)
        nav2 = calc.calculate_daily_nav(
            date=str(dates[1].date()), cash=5_000_000, market_value=6_000_000,
        )
        calc.save_nav(nav2)

        assert nav2.perf_fee == pytest.approx(200_000, rel=1e-6)

        # Third day: assets unchanged, no new high.
        nav3 = calc.calculate_daily_nav(
            date=str(dates[2].date()), cash=5_000_000, market_value=6_000_000,
        )

        assert nav3.perf_fee == 0.0
        assert nav3.nav_total == pytest.approx(nav2.nav_total, rel=1e-9)
        assert nav3.daily_return == pytest.approx(0.0, abs=1e-12)

    def test_recalculating_the_same_date_does_not_double_charge(self, store):
        """A retry for the same date used to accrue another day of fees and
        overwrite the row with double the liability."""
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.0)
        date = _D2

        first = calc.calculate_daily_nav(
            date=date, cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(first)

        second = calc.calculate_daily_nav(
            date=date, cash=5_000_000, market_value=5_000_000,
        )

        assert second.mgmt_fee == pytest.approx(first.mgmt_fee, rel=1e-9)
        assert second.accrued_mgmt_fee == pytest.approx(
            first.accrued_mgmt_fee, rel=1e-9,
        )
        assert second.nav_total == pytest.approx(first.nav_total, rel=1e-9)

    def test_management_fee_counts_calendar_days(self, store):
        """A Friday-to-Monday gap is three calendar days, not one trading
        day, and the annual rate is divided by 365."""
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.0)
        first = calc.calculate_daily_nav(
            date="2024-01-05", cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(first)

        second = calc.calculate_daily_nav(
            date="2024-01-08", cash=5_000_000, market_value=5_000_000,
        )

        # The fee base is net of fees already accrued.
        expected = (
            second.cash + second.market_value - first.accrued_mgmt_fee
        ) * 0.02 / 365 * 3
        assert second.mgmt_fee == pytest.approx(expected, rel=1e-6)

    def test_performance_fee_is_charged_once_per_new_high(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.20)
        dates = pd.bdate_range(end=_TODAY.strftime("%Y-%m-%d"), periods=3)

        nav1 = calc.calculate_daily_nav(
            date=str(dates[0].date()), cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(nav1)
        nav2 = calc.calculate_daily_nav(
            date=str(dates[1].date()), cash=5_000_000, market_value=6_000_000,
        )
        calc.save_nav(nav2)
        nav3 = calc.calculate_daily_nav(
            date=str(dates[2].date()), cash=5_000_000, market_value=6_880_000,
        )

        total_perf = nav2.perf_fee + nav3.perf_fee
        gross_gain = (nav3.cash + nav3.market_value) - 10_000_000
        assert total_perf == pytest.approx(0.20 * gross_gain, rel=1e-6)

    def test_nav_total_equals_assets_minus_every_accrued_fee(self, store):
        """Exact accounting identity on every day.

        Checking only `nav_total <= gross` lets a regression that subtracts
        just the current day's performance fee -- adding every earlier fee
        liability back to NAV -- pass. This asserts the full equation.
        """
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.20)
        dates = pd.bdate_range(end=_TODAY.strftime("%Y-%m-%d"), periods=4)
        values = [
            (5_000_000, 5_000_000),
            (5_000_000, 5_400_000),
            (5_000_000, 5_600_000),
            (5_000_000, 5_300_000),
        ]

        navs = []
        for day, (cash, market_value) in zip(dates, values, strict=True):
            nav = calc.calculate_daily_nav(
                date=str(day.date()), cash=cash, market_value=market_value,
            )
            calc.save_nav(nav)
            navs.append(nav)

            assert nav.nav_total == pytest.approx(
                cash + market_value - nav.accrued_mgmt_fee - nav.accrued_perf_fee,
                rel=1e-9,
            )
            assert nav.nav_per_unit == pytest.approx(
                nav.nav_total / nav.total_units, rel=1e-12,
            )
            assert nav.accrued_perf_fee == pytest.approx(
                sum(n.perf_fee for n in navs), rel=1e-9,
            )

    def test_a_second_new_high_is_charged_only_on_the_increment(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.20)
        dates = pd.bdate_range(end=_TODAY.strftime("%Y-%m-%d"), periods=4)

        nav1 = calc.calculate_daily_nav(
            date=str(dates[0].date()), cash=5_000_000, market_value=5_000_000,
        )
        calc.save_nav(nav1)
        nav2 = calc.calculate_daily_nav(
            date=str(dates[1].date()), cash=5_000_000, market_value=5_600_000,
        )
        calc.save_nav(nav2)
        nav3 = calc.calculate_daily_nav(
            date=str(dates[2].date()), cash=5_000_000, market_value=5_800_000,
        )
        calc.save_nav(nav3)
        nav4 = calc.calculate_daily_nav(
            date=str(dates[3].date()), cash=5_000_000, market_value=5_700_000,
        )

        assert nav1.perf_fee == 0.0
        assert nav2.perf_fee == pytest.approx(0.20 * 600_000, rel=1e-9)
        assert nav3.perf_fee == pytest.approx(0.20 * 200_000, rel=1e-9)
        assert nav4.perf_fee == 0.0
        assert nav4.nav_total == pytest.approx(5_700_000 + 5_000_000 - 160_000, rel=1e-9)
