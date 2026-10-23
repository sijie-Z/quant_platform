"""Regression tests for operations.reconciliation (R53/R54/R55).

A missing position must be valued at the available price (not zero), the
per-position tolerance must be reflected in PositionRow.is_matched, and the
cash/materiality thresholds must be configurable.
"""

from __future__ import annotations

import pytest

from quant_platform.operations.reconciliation import (
    positions_from_broker,
    positions_from_store,
    reconcile,
)
from quant_platform.trading.broker import Position


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


class TestBrokerPositionsAdapter:
    """R11: broker Position list -> reconcile() mapping."""

    def test_shares_and_mark_price(self):
        book = positions_from_broker([
            Position(code="600519", quantity=100, current_price=50.0,
                     avg_cost=45.0),
            Position(code="000001", quantity=200, current_price=10.0,
                     avg_cost=11.0),
        ])

        assert book == {"600519": (100, 50.0), "000001": (200, 10.0)}

    def test_a_missing_mark_falls_back_to_cost(self):
        book = positions_from_broker([
            Position(code="600519", quantity=100, current_price=0.0,
                     avg_cost=45.0),
        ])

        assert book == {"600519": (100, 45.0)}

    def test_an_empty_list_is_a_confirmed_flat_book(self):
        assert positions_from_broker([]) == {}

    def test_none_is_unknown_not_flat(self):
        """The broker contract: None means the query failed. Treating it as
        an empty book would silently 'pass' a missing account."""
        with pytest.raises(ValueError, match="unknown"):
            positions_from_broker(None)

    def test_duplicate_assets_merge(self):
        book = positions_from_broker([
            Position(code="600519", quantity=100, current_price=50.0),
            Position(code="600519", quantity=50, current_price=55.0),
        ])

        assert book["600519"] == (150, 55.0)

    def test_a_position_without_a_code_fails_loud(self):
        """Dropping it silently would reconcile a malformed answer as flat."""
        with pytest.raises(ValueError, match="asset code"):
            positions_from_broker([Position(quantity=100, current_price=50.0)])


class TestStorePositionsAdapter:
    """R11: Store.get_positions() rows -> reconcile() mapping."""

    def test_rows_convert_with_the_store_mark(self):
        rows = [
            {"code": "600519", "quantity": 100, "avg_cost": 45.0,
             "current_price": 50.0},
            {"code": "000001", "quantity": 200, "avg_cost": 11.0,
             "current_price": 10.0},
        ]

        assert positions_from_store(rows) == {
            "600519": (100, 50.0), "000001": (200, 10.0),
        }

    def test_a_zero_mark_falls_back_to_cost(self):
        rows = [{"code": "600519", "quantity": 100, "avg_cost": 45.0,
                 "current_price": 0.0}]

        assert positions_from_store(rows) == {"600519": (100, 45.0)}

    def test_none_and_empty_mean_no_rows(self):
        assert positions_from_store(None) == {}
        assert positions_from_store([]) == {}

    def test_duplicate_tenant_rows_merge(self):
        """A tenant-less read merges tenants holding one ticker."""
        rows = [
            {"code": "600519", "quantity": 100, "avg_cost": 45.0},
            {"code": "600519", "quantity": 25, "avg_cost": 48.0},
        ]

        assert positions_from_store(rows) == {"600519": (125, 48.0)}
