"""Regression tests for operations.reconciliation (R53/R54/R55).

A missing position must be valued at the available price (not zero), the
per-position tolerance must be reflected in PositionRow.is_matched, and the
cash/materiality thresholds must be configurable.
"""

from __future__ import annotations

import pytest

from quant_platform.operations.reconciliation import reconcile


class TestMissingLegValuation:
    def test_missing_actual_position_is_valued_at_expected_price(self):
        report = reconcile(
            expected_positions={"600519": (100, 50.0)},
            actual_positions={},
            expected_cash=0.0,
            actual_cash=0.0,
        )

        assert report.mismatched == 1
        assert report.total_mismatch_value == pytest.approx(5_000.0)
        row = report.details[0]
        assert row.diff == 100
        assert row.is_matched is False

    def test_zero_actual_price_uses_expected_price(self):
        report = reconcile(
            expected_positions={"600519": (100, 50.0)},
            actual_positions={"600519": (0, 0.0)},
            expected_cash=1_000_000.0,
            actual_cash=1_000_000.0,
        )

        assert report.total_mismatch_value == pytest.approx(5_000.0)

    def test_extra_actual_position_is_valued_at_actual_price(self):
        report = reconcile(
            expected_positions={},
            actual_positions={"600519": (100, 50.0)},
            expected_cash=0.0,
            actual_cash=0.0,
        )

        assert report.mismatched == 1
        assert report.total_mismatch_value == pytest.approx(5_000.0)

    def test_price_disagreement_uses_the_higher_price(self):
        report = reconcile(
            expected_positions={"600519": (100, 50.0)},
            actual_positions={"600519": (90, 55.0)},
            expected_cash=0.0,
            actual_cash=0.0,
        )

        # 10 shares x max(50, 55) = 550.
        assert report.total_mismatch_value == pytest.approx(550.0)


class TestToleranceReporting:
    def test_position_row_honours_tolerance(self):
        report = reconcile(
            expected_positions={
                "600519": (100, 50.0),
                "000001": (200, 10.0),
            },
            actual_positions={
                "600519": (95, 50.0),   # diff 5, within tolerance
                "000001": (150, 10.0),  # diff 50, breach
            },
            expected_cash=1_000_000.0,
            actual_cash=1_000_000.0,
            tolerance=10,
        )

        assert report.matched == 1
        assert report.mismatched == 1
        by_asset = {row.asset: row for row in report.details}
        assert by_asset["600519"].is_matched is True
        assert by_asset["600519"].tolerance == 10
        assert by_asset["000001"].is_matched is False
        # Only the breaching position contributes to the mismatch value.
        assert report.total_mismatch_value == pytest.approx(50 * 10.0)


class TestCashAndMaterialityThresholds:
    def test_default_absolute_cash_tolerance_is_1000(self):
        within = reconcile(
            expected_positions={}, actual_positions={},
            expected_cash=10_000_000.0, actual_cash=9_999_001.0,
        )
        breach = reconcile(
            expected_positions={}, actual_positions={},
            expected_cash=10_000_000.0, actual_cash=9_998_999.0,
        )
        at_limit = reconcile(
            expected_positions={}, actual_positions={},
            expected_cash=10_000_000.0, actual_cash=9_999_000.0,
        )

        assert within.passes is True
        assert breach.passes is False
        assert at_limit.passes is True  # "allowed" is inclusive

    def test_exact_match_passes_with_zero_tolerance(self):
        report = reconcile(
            expected_positions={}, actual_positions={},
            expected_cash=1_000_000.0, actual_cash=1_000_000.0,
            cash_tolerance=0.0, cash_tolerance_bps=0.0,
        )
        assert report.passes is True

    def test_zero_value_mismatch_is_not_absorbed_by_default(self):
        """An unknown broker position (price 0) still fails: materiality 0
        must mean 'any mismatch fails', not 'value <= 0 is fine'."""
        unexpected = reconcile(
            expected_positions={},
            actual_positions={"UNKNOWN": (100, 0.0)},
            expected_cash=0.0, actual_cash=0.0,
        )
        missing = reconcile(
            expected_positions={"600519": (100, 0.0)},
            actual_positions={},
            expected_cash=0.0, actual_cash=0.0,
        )

        assert unexpected.mismatched == 1
        assert unexpected.total_mismatch_value == 0.0
        assert unexpected.passes is False
        assert missing.passes is False

    def test_bps_tolerance_scales_with_the_cash_balance(self):
        # 10 bps of 10,000,000 = 10,000, replacing the absolute 1000.
        within = reconcile(
            expected_positions={}, actual_positions={},
            expected_cash=10_000_000.0, actual_cash=9_991_000.0,
            cash_tolerance=0.0, cash_tolerance_bps=10.0,
        )
        breach = reconcile(
            expected_positions={}, actual_positions={},
            expected_cash=10_000_000.0, actual_cash=9_988_000.0,
            cash_tolerance=0.0, cash_tolerance_bps=10.0,
        )

        assert within.passes is True
        assert breach.passes is False

    def test_mismatch_materiality_can_absorb_small_breaks(self):
        positions = {"600519": (100, 50.0)}
        actual = {"600519": (90, 50.0)}  # 10 x 50 = 500 mismatch

        absorbed = reconcile(
            positions, actual, 0.0, 0.0, mismatch_materiality=500.0,
        )
        exposed = reconcile(
            positions, actual, 0.0, 0.0, mismatch_materiality=499.0,
        )

        assert absorbed.total_mismatch_value == pytest.approx(500.0)
        assert absorbed.passes is True
        assert exposed.passes is False
