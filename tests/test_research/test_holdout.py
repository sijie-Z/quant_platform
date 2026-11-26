"""Tests for the research holdout spec, config wiring, and sweep guard."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_platform.research.holdout import (
    ENV_END,
    ENV_START,
    HoldoutSpec,
    load_configured_holdout,
    sweep_guard_message,
)
from quant_platform.utils.config import load_config


def _settings(start=None, end=None):
    return SimpleNamespace(
        research=SimpleNamespace(
            holdout=SimpleNamespace(start_date=start, end_date=end),
        ),
    )


def _raw_config(start="", end="", data_window=("2021-01-01", "2025-12-31")):
    return {
        "data": {"start_date": data_window[0], "end_date": data_window[1]},
        "research": {"holdout": {"start_date": start, "end_date": end}},
    }


class TestHoldoutSpecLoading:
    def test_disabled_when_section_absent(self):
        assert HoldoutSpec.from_settings(SimpleNamespace()) is None
        assert HoldoutSpec.from_settings(None) is None

    def test_disabled_when_both_dates_empty_or_null(self):
        assert HoldoutSpec.from_settings(_settings("", "")) is None
        assert HoldoutSpec.from_settings(_settings(None, None)) is None
        assert HoldoutSpec.from_settings(_settings("  ", "")) is None

    def test_parses_iso_dates(self):
        spec = HoldoutSpec.from_settings(_settings("2024-07-01", "2024-12-31"))

        assert spec == HoldoutSpec(date(2024, 7, 1), date(2024, 12, 31))
        assert spec.to_dict() == {
            "start_date": "2024-07-01", "end_date": "2024-12-31",
        }

    def test_half_configured_window_is_an_error(self):
        with pytest.raises(ValueError, match="both start_date and end_date"):
            HoldoutSpec.from_settings(_settings("2024-07-01", ""))
        with pytest.raises(ValueError, match="both start_date and end_date"):
            HoldoutSpec.from_settings(_settings("", "2024-12-31"))

    def test_unparsable_date_is_an_error(self):
        with pytest.raises(ValueError, match="ISO date"):
            HoldoutSpec.from_settings(_settings("2024/07/01", "2024-12-31"))
        with pytest.raises(ValueError, match="ISO date"):
            HoldoutSpec.from_settings(_settings("2024-02-30", "2024-12-31"))

    def test_start_must_be_strictly_before_end(self):
        with pytest.raises(ValueError, match="strictly before"):
            HoldoutSpec(date(2024, 7, 1), date(2024, 7, 1))
        with pytest.raises(ValueError, match="strictly before"):
            HoldoutSpec(date(2024, 8, 1), date(2024, 7, 1))

    def test_from_env(self):
        assert HoldoutSpec.from_env({}) is None
        assert HoldoutSpec.from_env({ENV_START: "", ENV_END: "  "}) is None

        spec = HoldoutSpec.from_env({
            ENV_START: "2024-07-01", ENV_END: "2024-12-31",
        })
        assert spec == HoldoutSpec(date(2024, 7, 1), date(2024, 12, 31))

        with pytest.raises(ValueError, match="QUANT_HOLDOUT_END"):
            HoldoutSpec.from_env({ENV_START: "2024-07-01"})
        with pytest.raises(ValueError, match="ISO date"):
            HoldoutSpec.from_env({
                ENV_START: "07/01/2024", ENV_END: "2024-12-31",
            })


class TestOverlaps:
    def test_closed_interval_boundaries(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))

        # Ends the day before the holdout starts: contiguous, no overlap.
        assert spec.overlaps("2023-01-01", "2024-02-29") is False
        # Ends on the first holdout day: one shared day.
        assert spec.overlaps("2023-01-01", "2024-03-01") is True
        # Starts the day after the holdout ends: contiguous, no overlap.
        assert spec.overlaps("2024-04-01", "2025-12-31") is False
        # Starts on the last holdout day: one shared day.
        assert spec.overlaps("2024-03-31", "2025-12-31") is True

    def test_containment_and_contained_windows(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        assert spec.overlaps("2023-01-01", "2025-12-31") is True
        assert spec.overlaps("2024-03-10", "2024-03-20") is True
        assert spec.overlaps("2024-03-01", "2024-03-01") is True

    def test_inverted_data_window_raises(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        with pytest.raises(ValueError, match="must be <="):
            spec.overlaps("2024-12-31", "2024-01-01")


class TestMasking:
    def test_exclude_dates_is_inclusive_and_preserves_order(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        index = pd.bdate_range("2024-02-26", "2024-03-05")

        kept = spec.exclude_dates(index)

        assert list(kept) == [
            pd.Timestamp("2024-02-26"), pd.Timestamp("2024-02-27"),
            pd.Timestamp("2024-02-28"), pd.Timestamp("2024-02-29"),
        ]
        # The input index is untouched.
        assert len(index) == 7
        assert spec.excluded_days(index) == 3

    def test_mask_frame_drops_rows_keeps_columns(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        frame = pd.DataFrame(
            {"A": range(5), "B": range(5)},
            index=pd.to_datetime([
                "2024-02-28", "2024-02-29", "2024-03-01",
                "2024-03-02", "2024-04-01",
            ]),
        )

        masked = spec.mask_frame(frame)

        assert list(masked.columns) == ["A", "B"]
        assert list(masked.index) == [
            pd.Timestamp("2024-02-28"), pd.Timestamp("2024-02-29"),
            pd.Timestamp("2024-04-01"),
        ]
        assert list(masked["A"]) == [0, 1, 4]
        assert spec.mask_frame(None) is None

    def test_mask_frame_multiindex_level(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-02-29"), "A"),
                (pd.Timestamp("2024-03-01"), "A"),
                (pd.Timestamp("2024-03-01"), "B"),
                (pd.Timestamp("2024-04-01"), "A"),
            ],
            names=["date", "asset"],
        )
        frame = pd.DataFrame({"market_cap": [1.0, 2.0, 3.0, 4.0]}, index=index)

        masked = spec.mask_frame(frame, level="date")

        assert list(masked.index) == [
            (pd.Timestamp("2024-02-29"), "A"),
            (pd.Timestamp("2024-04-01"), "A"),
        ]

    def test_mask_frame_drop_forward_labels_drops_the_boundary_row(self):
        """returns[t] labels t -> t+1, so the last pre-holdout row carries a
        holdout-day return; the research mask must drop it as well."""
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        index = pd.to_datetime([
            "2024-02-28", "2024-02-29", "2024-03-01", "2024-04-01",
        ])
        frame = pd.DataFrame({"A": range(4)}, index=index)

        default = spec.mask_frame(frame)
        strict = spec.mask_frame(frame, drop_forward_labels=True)

        assert list(default.index) == [
            pd.Timestamp("2024-02-28"), pd.Timestamp("2024-02-29"),
            pd.Timestamp("2024-04-01"),
        ]
        # 2024-02-29's forward label is 2024-03-01, a holdout day.
        assert list(strict.index) == [
            pd.Timestamp("2024-02-28"), pd.Timestamp("2024-04-01"),
        ]

    def test_mask_frame_drop_forward_labels_without_a_prior_row(self):
        spec = HoldoutSpec(date(2024, 3, 1), date(2024, 3, 31))
        index = pd.to_datetime(["2024-03-01", "2024-04-01"])
        frame = pd.DataFrame({"A": [1, 2]}, index=index)

        masked = spec.mask_frame(frame, drop_forward_labels=True)

        assert list(masked.index) == [pd.Timestamp("2024-04-01")]


class TestConfigWiring:
    def test_default_config_has_a_disabled_holdout(self):
        config = load_config()

        assert config.research.holdout.start_date == ""
        assert config.research.holdout.end_date == ""
        assert HoldoutSpec.from_settings(config) is None
        assert load_configured_holdout(config) is None

    def test_yaml_holdout_is_parsed(self):
        config = load_config(raw=_raw_config("2024-07-01", "2024-12-31"))

        spec = HoldoutSpec.from_settings(config)
        assert spec == HoldoutSpec(date(2024, 7, 1), date(2024, 12, 31))

    @pytest.mark.parametrize(
        "start,end",
        [
            ("2024-07-01", "2024-07-01"),   # start == end
            ("2024-08-01", "2024-07-01"),   # start > end
            ("2024-07-01", ""),             # half configured
            ("07/01/2024", "2024-12-31"),   # not ISO
        ],
    )
    def test_invalid_yaml_holdout_raises_value_error(self, start, end):
        with pytest.raises(ValueError):
            load_config(raw=_raw_config(start, end))

    def test_validate_false_skips_holdout_validation_but_spec_still_fails(self):
        """The config system's documented contract: validate=False builds
        synthetic/partial configs without range/enum checks (the generic
        schema round-trip test relies on it). The holdout is still validated
        fail-loud wherever it is consumed, i.e. by HoldoutSpec.from_settings."""
        raw = _raw_config("2024-12-31", "2024-01-01")  # start > end

        config = load_config(raw=raw, validate=False)  # must not raise

        assert config.research.holdout.start_date == "2024-12-31"
        with pytest.raises(ValueError, match="strictly before"):
            HoldoutSpec.from_settings(config)


class TestSweepGuard:
    def test_disabled_holdout_is_a_no_op(self):
        config = load_config(raw=_raw_config("", ""))
        assert sweep_guard_message(config) is None

    def test_non_overlapping_window_is_allowed(self):
        config = load_config(raw=_raw_config(
            "2024-07-01", "2024-12-31",
            data_window=("2021-01-01", "2024-06-30"),
        ))
        assert sweep_guard_message(config) is None

    def test_overlapping_window_is_refused(self):
        config = load_config(raw=_raw_config(
            "2024-07-01", "2024-12-31",
            data_window=("2021-01-01", "2024-07-01"),  # one shared day
        ))

        message = sweep_guard_message(config)

        assert message is not None
        assert "holdout" in message
        assert "must not touch the holdout" in message
        assert "disable the holdout" in message

    def test_enabled_holdout_without_data_window_fails_closed(self):
        config = SimpleNamespace(
            data=None,
            research=SimpleNamespace(
                holdout=SimpleNamespace(
                    start_date="2024-07-01", end_date="2024-12-31",
                ),
            ),
        )
        with pytest.raises(ValueError, match="data window"):
            sweep_guard_message(config)

    def test_guard_accepts_raw_mapping(self):
        raw = _raw_config(
            "2024-07-01", "2024-12-31",
            data_window=("2021-01-01", "2025-12-31"),
        )
        assert "holdout" in sweep_guard_message(raw)
