"""Regression tests for operations.reconciliation (R53/R54/R55).

A missing position must be valued at the available price (not zero), the
per-position tolerance must be reflected in PositionRow.is_matched, and the
cash/materiality thresholds must be configurable.
"""

from __future__ import annotations

import pytest

from quant_platform.operations.reconciliation import (
    expected_book_from_ledger,
    ledger_from_trades,
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

    def test_a_non_integral_share_count_fails_loud(self):
        """Truncating a malformed count to 0 would shrink the leg silently."""
        with pytest.raises(ValueError, match="share count"):
            positions_from_store([{"code": "600519", "quantity": "abc"}])
        with pytest.raises(ValueError, match="share count"):
            positions_from_store([{"code": "600519", "quantity": 100.7}])

    def test_a_numeric_string_share_count_is_accepted(self):
        assert positions_from_store(
            [{"code": "600519", "quantity": "100"}],
        ) == {"600519": (100, 0.0)}


class TestLedgerFromTrades:
    """R12: expected book derived from a broker baseline + stored fills.

    Cash moves by the full principal plus fees. (The task book's §5a sample
    numbers -- 99,999 / 100,358.5 -- omit the 50 x 10 buy principal and
    contradict its own §2.3 formula; the formula is implemented.)
    """

    @staticmethod
    def _baseline(**overrides):
        baseline = {
            "created_at": "2026-10-20T09:00:00",
            "tenant_id": "default",
            "engine_state_key": "live_engine:default",
            "cash": 100_000.0,
            "positions_json": {"A": {"quantity": 100, "avg_cost": 9.0}},
        }
        baseline.update(overrides)
        return baseline

    @staticmethod
    def _trade(**overrides):
        trade = {
            "trade_id": "t1",
            "tenant_id": "default",
            "code": "A",
            "side": "buy",
            "quantity": 50,
            "price": 10.0,
            "commission": 0.0,
            "tax": 0.0,
            "executed_at": "2026-10-20T09:05:00",
        }
        trade.update(overrides)
        return trade

    def test_buy_then_sell_moves_principal_and_fees(self):
        buy = self._trade(commission=1.0)
        sell = self._trade(
            trade_id="t2", side="sell", quantity=30, price=12.0,
            commission=0.0, tax=0.5, executed_at="2026-10-20T10:00:00",
        )

        after_buy = ledger_from_trades(self._baseline(), [buy])
        assert after_buy["positions"] == {"A": 150}
        assert after_buy["cash"] == pytest.approx(100_000.0 - 501.0)

        after_sell = ledger_from_trades(self._baseline(), [buy, sell])
        assert after_sell["positions"] == {"A": 120}
        assert after_sell["cash"] == pytest.approx(100_000.0 - 501.0 + 359.5)

    def test_multiple_trades_and_assets(self):
        trades = [
            self._trade(trade_id="t1", code="A", quantity=50, price=10.0,
                        commission=1.0, executed_at="2026-10-20T09:05:00"),
            self._trade(trade_id="t2", code="B", side="buy", quantity=20,
                        price=5.0, commission=0.5,
                        executed_at="2026-10-20T09:10:00"),
            self._trade(trade_id="t3", code="A", side="sell", quantity=10,
                        price=11.0, tax=0.2,
                        executed_at="2026-10-20T10:00:00"),
        ]

        ledger = ledger_from_trades(
            self._baseline(positions_json={"A": 100, "B": 0}), trades,
        )

        assert ledger["positions"] == {"A": 140, "B": 20}
        assert ledger["cash"] == pytest.approx(
            100_000.0 - 501.0 - 100.5 + 109.8
        )

    def test_fills_before_the_baseline_are_ignored(self):
        early = self._trade(
            trade_id="old", quantity=999, price=1.0,
            executed_at="2026-10-20T08:59:59",
        )
        ledger = ledger_from_trades(self._baseline(), [early])
        assert ledger["positions"] == {"A": 100}
        assert ledger["cash"] == pytest.approx(100_000.0)

    def test_a_fill_at_the_baseline_instant_is_included(self):
        """The bound is inclusive (>=): the baseline is taken at engine start
        and a fill stamped at the same instant belongs to the ledger."""
        same = self._trade(
            quantity=10, price=10.0, executed_at="2026-10-20T09:00:00",
        )
        ledger = ledger_from_trades(self._baseline(), [same])
        assert ledger["positions"] == {"A": 110}
        assert ledger["cash"] == pytest.approx(99_900.0)

    def test_equal_instants_with_different_precision_are_included(self):
        baseline = self._baseline(created_at="2026-10-20T09:00:00.500000")
        same = self._trade(executed_at="2026-10-20T09:00:00.5")
        just_before = self._trade(
            trade_id="t0", executed_at="2026-10-20T09:00:00.499999",
        )

        ledger = ledger_from_trades(baseline, [same, just_before])

        assert ledger["positions"] == {"A": 150}, (
            "the .5 fill is at the baseline instant; .499999 is before it"
        )

    def test_fills_from_another_tenant_are_ignored(self):
        other = self._trade(tenant_id="other", quantity=1000)
        ledger = ledger_from_trades(self._baseline(), [other])
        assert ledger["positions"] == {"A": 100}
        assert ledger["cash"] == pytest.approx(100_000.0)

    def test_an_unscoped_baseline_does_not_filter_the_tenant(self):
        baseline = self._baseline(tenant_id="")
        ledger = ledger_from_trades(
            baseline, [self._trade(tenant_id="other", quantity=10)],
        )
        assert ledger["positions"] == {"A": 110}

    def test_net_zero_positions_are_dropped(self):
        close = self._trade(side="sell", quantity=100, price=10.0)
        ledger = ledger_from_trades(self._baseline(), [close])
        assert ledger["positions"] == {}
        assert ledger["cash"] == pytest.approx(101_000.0)

    def test_negative_quantity_and_cash_pass_through(self):
        """Short books and cost drift are observations, not errors."""
        sell_short = self._trade(side="sell", quantity=150, price=10.0)
        ledger = ledger_from_trades(
            self._baseline(positions_json={"A": 0}, cash=100.0),
            [sell_short],
        )
        assert ledger["positions"] == {"A": -150}
        assert ledger["cash"] == pytest.approx(1_600.0)

    def test_positions_accept_json_text_and_bare_quantities(self):
        baseline = self._baseline(positions_json='{"A": 100}')
        ledger = ledger_from_trades(baseline, [])
        assert ledger["positions"] == {"A": 100}

    def test_missing_commission_and_tax_default_to_zero(self):
        trade = self._trade()
        del trade["commission"]
        del trade["tax"]
        ledger = ledger_from_trades(self._baseline(), [trade])
        assert ledger["cash"] == pytest.approx(99_500.0)

    def test_a_baseline_without_created_at_fails_loud(self):
        with pytest.raises(ValueError, match="created_at"):
            ledger_from_trades(self._baseline(created_at=""), [])

    def test_a_baseline_without_cash_fails_loud(self):
        baseline = self._baseline()
        del baseline["cash"]
        with pytest.raises(ValueError, match="cash"):
            ledger_from_trades(baseline, [])

    def test_an_unsupported_side_fails_loud(self):
        with pytest.raises(ValueError, match="side"):
            ledger_from_trades(
                self._baseline(), [self._trade(side="short")],
            )

    def test_a_malformed_baseline_position_fails_loud(self):
        with pytest.raises(ValueError, match="quantity"):
            ledger_from_trades(
                self._baseline(positions_json={"A": "many"}), [],
            )


class TestExpectedBookFromLedger:
    """R12: expected prices come from the broker's marks; quantities and
    cash are the only compared values."""

    def test_expected_uses_the_actual_price(self):
        book = expected_book_from_ledger(
            {"600519": 100}, {"600519": (100, 55.0)},
        )
        assert book == {"600519": (100, 55.0)}

    def test_an_expected_only_asset_falls_back_to_its_cost(self):
        book = expected_book_from_ledger(
            {"600519": 100, "000001": 50},
            {},
            {"600519": 45.0},
        )
        assert book == {"600519": (100, 45.0), "000001": (50, 0.0)}

    def test_a_zero_actual_mark_falls_back_to_cost(self):
        book = expected_book_from_ledger(
            {"600519": 100}, {"600519": (100, 0.0)}, {"600519": 45.0},
        )
        assert book == {"600519": (100, 45.0)}

    def test_negative_ledger_quantities_are_kept(self):
        book = expected_book_from_ledger({"600519": -10}, {})
        assert book == {"600519": (-10, 0.0)}
