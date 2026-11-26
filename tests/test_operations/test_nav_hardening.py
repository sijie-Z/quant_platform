"""Regression tests for the NAV calculator hardening wave (R41-R45).

Covers fail-closed store reads and zero-NAV persistence (R41/R43), backdated
dates and fresh calculator instances (R42), full-precision persistence (R44)
and the single performance-fee accrual path (R45).
"""

from __future__ import annotations

import inspect

import pytest

from quant_platform.core.store import Store
from quant_platform.operations.nav import NAV, NAVCalculator, NAVDataUnavailableError

PERFECT_DAY_FEE = 10_000_000.0 * 0.02 / 365.0


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


class ExplodingStore:
    """Every read raises, like a locked/corrupt SQLite database."""

    def __init__(self):
        self.saved: list[dict] = []

    def get_latest_nav(self):
        raise RuntimeError("sqlite: database is locked")

    def get_nav_history(self, days=30):
        raise RuntimeError("sqlite: database is locked")

    def get_positions(self):
        raise RuntimeError("sqlite: database is locked")

    def get_pnl_history(self, days=30, start_date="", end_date=""):
        raise RuntimeError("sqlite: database is locked")

    def save_nav(self, nav):
        self.saved.append(nav)


class ReadFailingStore:
    """Construction succeeds (no persisted state) but portfolio reads fail."""

    def __init__(self):
        self.saved: list[dict] = []

    def get_latest_nav(self):
        return None

    def get_nav_history(self, days=30):
        return []

    def get_positions(self):
        raise RuntimeError("db read failed")

    def get_pnl_history(self, days=30, start_date="", end_date=""):
        raise RuntimeError("db read failed")

    def save_nav(self, nav):
        self.saved.append(nav)


# ── R41/R43: fail closed, never persist a zero NAV ──────────────────────


class TestFailClosed:
    def test_constructor_refuses_unknown_persisted_state(self):
        """R43: a failing store read must abort, not silently keep HWM=1.0."""
        with pytest.raises(RuntimeError, match="database is locked"):
            NAVCalculator(ExplodingStore())

    def test_store_read_failure_propagates_and_persists_nothing(self):
        broken = ReadFailingStore()
        calc = NAVCalculator(broken)

        with pytest.raises(RuntimeError, match="db read failed"):
            calc.update_daily_nav(date="2024-06-03")

        assert broken.saved == [], "a failed read must not persist a NAV"

    def test_missing_pnl_snapshot_is_refused_not_treated_as_zero_cash(
        self, store,
    ):
        store.save_position({
            "code": "600519", "quantity": 100, "avg_cost": 10.0,
            "current_price": 10.0, "market_value": 1000.0,
        })
        calc = NAVCalculator(store)

        with pytest.raises(NAVDataUnavailableError, match="no P&L snapshot"):
            calc.update_daily_nav(date="2024-06-03")

        assert store.get_nav_history(days=3650) == []

    def test_zero_nav_is_never_persisted(self, store):
        calc = NAVCalculator(store)
        zero = calc.calculate_daily_nav(
            date="2024-06-03", cash=0.0, market_value=0.0,
        )
        assert zero.nav_total == 0.0

        with pytest.raises(NAVDataUnavailableError, match="refusing to persist"):
            calc.save_nav(zero)
        with pytest.raises(NAVDataUnavailableError, match="refusing to persist"):
            calc.update_daily_nav(
                date="2024-06-03", cash=0.0, market_value=0.0,
            )

        assert store.get_nav_history(days=3650) == []

    def test_non_finite_input_is_refused(self, store):
        calc = NAVCalculator(store)
        with pytest.raises(NAVDataUnavailableError, match="non-finite"):
            calc.calculate_daily_nav(
                date="2024-06-03", cash=float("nan"), market_value=0.0,
            )

    def test_unreadable_persisted_row_fails_closed(self, store):
        store.save_nav({
            "date": "2024-06-03", "nav_total": 10_000_000.0,
            "nav_per_unit": 1.0, "total_units": 10_000_000.0,
            "high_water_mark": None,
        })

        with pytest.raises(NAVDataUnavailableError, match="unreadable"):
            NAVCalculator(store)

    def test_manual_nav_with_lax_date_cannot_be_persisted(self, store):
        calc = NAVCalculator(store)
        bad = NAV(
            date="2024-6-3", nav_total=10_000_000.0,
            nav_per_unit=1.0, total_units=10_000_000.0,
            cash=5_000_000.0, market_value=5_000_000.0,
            mgmt_fee=0.0, perf_fee=0.0, high_water_mark=1.0,
        )

        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            calc.save_nav(bad)

        assert store.get_nav_history(days=3650) == []

    def test_invalid_high_water_mark_fails_closed(self, store):
        """HWM=0 would charge 20% of the entire NAV as a performance fee."""
        store.save_nav({
            "date": "2024-06-03", "nav_total": 10_000_000.0,
            "nav_per_unit": 1.0, "total_units": 10_000_000.0,
            "high_water_mark": 0.0,
        })

        with pytest.raises(NAVDataUnavailableError, match="high-water"):
            NAVCalculator(store)

    def test_non_positive_units_fail_closed(self, store):
        store.save_nav({
            "date": "2024-06-03", "nav_total": 10_000_000.0,
            "nav_per_unit": 1.0, "total_units": 0.0,
            "high_water_mark": 1.0,
        })

        with pytest.raises(NAVDataUnavailableError, match="total units"):
            NAVCalculator(store)

    def test_lax_date_shapes_are_rejected(self, store):
        """State selection compares date strings; mixed shapes would corrupt
        ordering and create a second row for the same day."""
        calc = NAVCalculator(store)
        for bad in ("2024-6-3", "2024-06-03T15:00:00", "2024-06-03extra", 20240603):
            with pytest.raises(ValueError, match="YYYY-MM-DD"):
                calc.calculate_daily_nav(
                    date=bad, cash=5_000_000.0, market_value=5_000_000.0,
                )


# ── R42: backdated dates and second instances ───────────────────────────


class TestStateIsReadFromTheDate:
    def test_backdated_date_does_not_inherit_later_fees_or_hwm(self, store):
        store.save_nav({
            "date": "2024-12-31",
            "nav_total": 10_300_000.0,
            "nav_per_unit": 1.03,
            "total_units": 10_000_000.0,
            "cash": 5_000_000.0,
            "market_value": 5_300_000.0,
            "high_water_mark": 1.05,
            "accrued_mgmt_fee": 100_000.0,
            "accrued_perf_fee": 30_000.0,
        })
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.20)

        nav = calc.calculate_daily_nav(
            date="2024-06-03", cash=5_000_000.0, market_value=5_000_000.0,
        )

        # Isolated calculation: only one day of management fee, no December
        # liability, initial high-water mark.
        assert nav.nav_total == pytest.approx(
            10_000_000.0 - PERFECT_DAY_FEE, rel=1e-12,
        )
        assert nav.accrued_mgmt_fee == pytest.approx(PERFECT_DAY_FEE)
        assert nav.accrued_perf_fee == 0.0
        assert nav.high_water_mark == 1.0

    def test_backdated_date_uses_the_row_immediately_before_it(self, store):
        store.save_nav({
            "date": "2024-06-01", "nav_total": 10_000_000.0,
            "nav_per_unit": 1.0, "total_units": 10_000_000.0,
            "high_water_mark": 1.01, "accrued_mgmt_fee": 100.0,
            "accrued_perf_fee": 0.0,
        })
        store.save_nav({
            "date": "2024-12-31", "nav_total": 10_300_000.0,
            "nav_per_unit": 1.03, "total_units": 10_000_000.0,
            "high_water_mark": 1.05, "accrued_mgmt_fee": 100_000.0,
            "accrued_perf_fee": 30_000.0,
        })
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.20)

        nav = calc.calculate_daily_nav(
            date="2024-06-03", cash=5_000_000.0, market_value=5_000_000.0,
        )

        expected_fee = (10_000_000.0 - 100.0) * 0.02 / 365.0 * 2
        assert nav.mgmt_fee == pytest.approx(expected_fee, rel=1e-12)
        assert nav.accrued_mgmt_fee == pytest.approx(100.0 + expected_fee)
        assert nav.high_water_mark == 1.01

    def test_second_calculator_instance_accrues_correctly(self, store):
        calc_a = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.0)
        # Created before day one is persisted: its in-memory state is stale.
        calc_b = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.0)

        nav1 = calc_a.update_daily_nav(
            date="2024-06-03", cash=5_000_000.0, market_value=5_000_000.0,
        )
        nav2 = calc_b.update_daily_nav(
            date="2024-06-04", cash=5_000_000.0, market_value=5_000_000.0,
        )

        fee1 = PERFECT_DAY_FEE
        fee2 = (10_000_000.0 - fee1) * 0.02 / 365.0
        assert nav1.accrued_mgmt_fee == pytest.approx(fee1, rel=1e-12)
        assert nav2.accrued_mgmt_fee == pytest.approx(fee1 + fee2, rel=1e-12)
        assert nav2.mgmt_fee == pytest.approx(fee2, rel=1e-12)
        assert nav2.nav_total == pytest.approx(
            10_000_000.0 - fee1 - fee2, rel=1e-12,
        )

    def test_flagship_sequence_is_unchanged_by_state_reload(self, store):
        """The four-day fee/HWM identity from the audit still holds."""
        calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.20)
        values = [
            (5_000_000.0, 5_000_000.0),
            (5_000_000.0, 5_400_000.0),
            (5_000_000.0, 5_600_000.0),
            (5_000_000.0, 5_300_000.0),
        ]
        navs = []
        for i, (cash, market_value) in enumerate(values):
            nav = calc.update_daily_nav(
                date=f"2024-06-{3 + i:02d}",
                cash=cash, market_value=market_value,
            )
            navs.append(nav)
            assert nav.nav_total == pytest.approx(
                cash + market_value
                - nav.accrued_mgmt_fee
                - nav.accrued_perf_fee,
                rel=1e-9,
            )

        # The cumulative performance fee equals 20% of the peak gross NAV
        # above the initial high-water mark, net of accrued management fees
        # (the audit's four-day identity).
        expected_total_perf = 0.20 * (
            navs[2].cash + navs[2].market_value
            - navs[2].accrued_mgmt_fee - 10_000_000.0
        )
        assert sum(nav.perf_fee for nav in navs) == pytest.approx(
            expected_total_perf, rel=1e-9,
        )
        assert navs[1].perf_fee > 0
        assert navs[2].perf_fee > 0
        assert navs[3].perf_fee == 0.0


# ── R44: full-precision persistence ─────────────────────────────────────


class TestFullPrecisionPersistence:
    def test_persisted_nav_round_trips_without_rounding(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
        nav = calc.calculate_daily_nav(
            date="2024-06-03", cash=5_000_000.0, market_value=5_000_001.0,
        )
        calc.save_nav(nav)

        stored = store.get_latest_nav()
        assert stored["nav_per_unit"] == nav.nav_per_unit
        assert stored["nav_total"] == nav.nav_total
        assert stored["high_water_mark"] == nav.high_water_mark

    def test_next_day_return_is_zero_on_unchanged_nav(self, store):
        calc = NAVCalculator(store, annual_mgmt_fee=0.0, perf_fee_rate=0.0)
        nav1 = calc.update_daily_nav(
            date="2024-06-03", cash=5_000_000.0, market_value=5_000_001.0,
        )
        nav2 = calc.calculate_daily_nav(
            date="2024-06-04", cash=5_000_000.0, market_value=5_000_001.0,
        )

        # 4dp rounding used to leave a spurious 4.5e-5 daily return here.
        assert nav2.nav_per_unit == pytest.approx(
            nav1.nav_per_unit, rel=1e-15,
        )
        assert nav2.daily_return == pytest.approx(0.0, abs=1e-12)


# ── R45: one performance-fee accrual path ───────────────────────────────


def test_performance_fee_helper_is_the_live_path():
    source = inspect.getsource(NAVCalculator.calculate_daily_nav)
    assert "self._accrue_perf_fee(" in source
