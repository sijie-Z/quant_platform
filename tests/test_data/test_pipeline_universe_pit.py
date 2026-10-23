"""Index-membership PIT gate (R6a; buy/sell split in R6b).

The provider already builds a per-date constituent mask
(``get_index_membership_mask``), but the pipeline never consumed it: a stock
that joined the index mid-sample was treated as a member over the whole
sample ("样本期内成分并集").  These tests pin the gate:

* per-date membership history feeds a dedicated ``universe_mask``; the
  ``tradable_mask`` stays membership-free so a held position can still be
  sold after its index exit;
* a provider whose constituent data is a single end-date snapshot is refused
  (replaying it as history is the same look-ahead);
* providers without the hook behave exactly as before (universe all True);
* provenance reaches ``metadata.attrs["universe_pit"]`` through
  ``main._load_data`` and the gates carry ``universe_mask``;
* a column mismatch applies the intersection instead of silently excluding
  unmatched assets;
* a mask fetch failure is fatal, not silently all-True/all-False.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import DataContractError, DataQualityError

N_DATES = 90
DATES = pd.bdate_range("2024-01-02", periods=N_DATES, name="date")
ASSETS = ["000001.SZ", "000002.SZ", "000003.SZ"]
#: 000002.SZ joins the index on this business-day offset.
INCLUSION_DAY = 60
#: 000003.SZ leaves the index on this business-day offset (R6b exit test).
EXIT_DAY = 120
N_EXIT_DATES = 180

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _close_panel(seed: int = 17, dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0006, 0.009, (len(dates), len(ASSETS)))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=dates,
        columns=pd.Index(ASSETS, name="asset"),
    )


class _BaseFixtureProvider(DataProvider):
    """Deterministic provider with no membership hook (baseline behaviour)."""

    def __init__(
        self,
        *,
        suspended_window: tuple[str, int, int] | None = None,
        n_dates: int = N_DATES,
    ):
        self._dates = pd.bdate_range("2024-01-02", periods=n_dates, name="date")
        self._close = _close_panel(dates=self._dates)
        self._suspended_window = suspended_window

    # -- provider API ---------------------------------------------------
    def get_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sector": ["Tech"] * len(ASSETS),
                "market_cap_group": ["mid"] * len(ASSETS),
                "is_st": [False] * len(ASSETS),
                "listing_date": [pd.Timestamp("2000-01-01")] * len(ASSETS),
                "delisting_date": [pd.NaT] * len(ASSETS),
            },
            index=pd.Index(ASSETS, name="asset"),
        )

    def get_prices(self, start_date, end_date, fields=None) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [self._dates, ASSETS], names=["date", "asset"]
        )
        close = self._close.stack().reindex(index).to_numpy()
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
                "turnover": 0.02,
                "adj_factor": 1.0,
            },
            index=index,
        )
        suspended = pd.Series(False, index=index)
        if self._suspended_window is not None:
            asset, first, last = self._suspended_window
            window = (self._dates[first], self._dates[last], asset)
            suspended = (
                (index.get_level_values("date") >= window[0])
                & (index.get_level_values("date") <= window[1])
                & (index.get_level_values("asset") == window[2])
            )
        frame["is_suspended"] = np.asarray(suspended, dtype=bool)
        return frame

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [self._dates, ASSETS], names=["date", "asset"]
        )
        return pd.DataFrame(
            {
                "market_cap": 1e9, "total_assets": 2e9, "net_assets": 1e9,
                "revenue": 1e8, "net_profit": 1e7, "roe": 0.10,
                "pb_ratio": 1.5, "pe_ratio": 15.0, "asset_growth": 0.05,
            },
            index=index,
        )

    def get_benchmark(self, start_date, end_date) -> pd.Series:
        return pd.Series(0.0005, index=self._dates, name="benchmark")


class _MembershipProvider(_BaseFixtureProvider):
    """Provider exposing a per-date constituent history.

    ``history_available=False`` models the single end-date snapshot fallback:
    the hook exists but must not be trusted as history.  ``exit_day`` models
    a constituent dropping out of the index at that business-day offset.
    """

    def __init__(
        self,
        *,
        history_available: bool = True,
        drop_mask_columns: list[str] | None = None,
        mask_exception: Exception | None = None,
        index_code: str | None = "000300.SH",
        snapshot_offsets: list[int] | None = None,
        empty_mask: bool = False,
        bad_mask_type: bool = False,
        extra_mask_columns: bool = False,
        outside_window_mask: bool = False,
        inclusion_day: int = INCLUSION_DAY,
        exit_day: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.universe_history_available = history_available
        self.index_code = index_code
        self._drop_mask_columns = list(drop_mask_columns or [])
        self._mask_exception = mask_exception
        self._snapshot_offsets = snapshot_offsets
        self._empty_mask = empty_mask
        self._bad_mask_type = bad_mask_type
        self._extra_mask_columns = extra_mask_columns
        self._outside_window_mask = outside_window_mask
        self._inclusion_day = inclusion_day
        self._exit_day = exit_day

    def get_index_membership_mask(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        if self._mask_exception is not None:
            raise self._mask_exception
        if self._bad_mask_type:
            return [1, 2, 3]  # type: ignore[return-value]
        if self._empty_mask:
            return pd.DataFrame()
        # Bare codes on purpose: the pipeline must normalize mask labels just
        # like every other provider panel.
        bare = [asset.split(".")[0] for asset in ASSETS]
        if self._outside_window_mask:
            # A provider that ignores start/end and returns snapshots only
            # after the price window: two rows, zero usable snapshots.
            future = pd.DatetimeIndex(
                [self._dates[-1] + pd.tseries.offsets.BDay(1),
                 self._dates[-1] + pd.tseries.offsets.BDay(2)],
                name="date",
            )
            return pd.DataFrame(True, index=future, columns=bare)
        mask = pd.DataFrame(True, index=self._dates, columns=bare)
        mask.iloc[:self._inclusion_day, 1] = False  # 000002 joins here
        if self._exit_day is not None:
            mask.iloc[self._exit_day:, 2] = False   # 000003 leaves here
        if self._extra_mask_columns:
            mask["000099"] = True
        if self._snapshot_offsets is not None:
            # Model the real provider: index_weight history is snapshot-dated,
            # not daily.
            mask = mask.iloc[self._snapshot_offsets]
        if self._drop_mask_columns:
            mask = mask.drop(columns=self._drop_mask_columns)
        return mask.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]


class _HookWithoutFlagProvider(_BaseFixtureProvider):
    """Provider exposing the hook but no ``universe_history_available``."""

    def get_index_membership_mask(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        bare = [asset.split(".")[0] for asset in ASSETS]
        mask = pd.DataFrame(True, index=self._dates, columns=bare)
        return mask.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]


def _run(provider: DataProvider) -> DataPipeline:
    dates = getattr(provider, "_dates", DATES)
    pipeline = DataPipeline(
        provider=provider,
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
    )
    pipeline.run()
    return pipeline


def _assert_tradable_is_membership_free(pipeline: DataPipeline) -> None:
    """Membership must never narrow `tradable_mask` (R6b).

    The fixture has no ST/suspension days, so the whole panel stays True.
    """
    expected = pd.DataFrame(
        True, index=pipeline.tradable_mask.index,
        columns=pipeline.tradable_mask.columns,
    )
    pd.testing.assert_frame_equal(
        pipeline.tradable_mask, expected, check_freq=False,
    )


def _assert_universe_all_true(pipeline: DataPipeline) -> None:
    universe = pipeline.universe_mask
    assert universe is not None
    assert universe.dtypes.eq(bool).all()
    assert universe.all().all()
    assert list(universe.columns) == ASSETS


class TestMembershipMaskApplied:
    def test_per_date_history_narrows_universe_not_tradability(self):
        pipeline = _run(_MembershipProvider(history_available=True))

        # R6b: membership lives in its own mask; `tradable_mask` is exactly
        # the ST/listing/suspension mask (all True for this fixture).
        _assert_tradable_is_membership_free(pipeline)

        universe = pipeline.universe_mask
        assert universe is not None
        assert list(universe.columns) == ASSETS
        assert universe.index.equals(DATES)
        assert not universe["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert universe["000002.SZ"].iloc[INCLUSION_DAY:].all()
        # Members throughout are unaffected.
        assert universe["000001.SZ"].all()
        assert universe["000003.SZ"].all()

        pit = pipeline.universe_pit
        assert pit["applied"] is True
        assert pit["source"] == "get_index_membership_mask"
        assert pit["index_code"] == "000300.SH"
        assert pit["reason"] == "per_date_constituent_history"

        # Membership gates buys only: the price history stays intact so
        # factor lookbacks across the inclusion date still work.
        close = pipeline.prices["close"].unstack("asset")
        assert close["000002.SZ"].iloc[:INCLUSION_DAY].notna().all()
        assert close["000002.SZ"].iloc[INCLUSION_DAY:].notna().all()

        # The execution gates carry both panels, separated.
        gates = pipeline.get_execution_gates()
        pd.testing.assert_frame_equal(
            gates["tradable_mask"], pipeline.tradable_mask,
        )
        pd.testing.assert_frame_equal(gates["universe_mask"], universe)

    def test_mask_labels_are_normalized(self):
        pipeline = _run(_MembershipProvider(history_available=True))
        # 000001 (bare) in the mask must line up with 000001.SZ in the panel.
        assert pipeline.universe_mask is not None
        assert "000001.SZ" in pipeline.universe_mask.columns
        assert pipeline.universe_mask["000001.SZ"].all()

    def test_tradable_mask_matches_no_hook_baseline_bit_for_bit(self):
        """R6b: applying membership must not move `tradable_mask` at all."""
        suspended = ("000003.SZ", 10, 15)
        baseline = _run(_BaseFixtureProvider(suspended_window=suspended))
        applied = _run(
            _MembershipProvider(
                history_available=True, suspended_window=suspended,
            )
        )
        pd.testing.assert_frame_equal(
            applied.tradable_mask, baseline.tradable_mask,
        )
        # The suspension is the only difference from all-True, and the
        # membership narrowing is visible in the separate mask.
        assert not applied.tradable_mask.loc[
            DATES[10]:DATES[15], "000003.SZ"
        ].any()
        assert not applied.universe_mask["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert applied.universe_mask["000001.SZ"].all()


class TestMembershipExit:
    """R6b: an index exit must end buy eligibility but not sellability."""

    def test_exit_asset_leaves_universe_but_not_tradability(self):
        provider = _MembershipProvider(
            history_available=True, n_dates=N_EXIT_DATES, exit_day=EXIT_DAY,
        )
        pipeline = _run(provider)
        dates = provider._dates
        universe = pipeline.universe_mask

        # A is a member throughout.
        assert universe["000001.SZ"].all()
        # B joins on day 60 and stays.
        assert not universe["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert universe["000002.SZ"].iloc[INCLUSION_DAY:].all()
        # C is a member until day 120, then exits for good.
        assert universe["000003.SZ"].iloc[:EXIT_DAY].all()
        assert not universe["000003.SZ"].iloc[EXIT_DAY:].any()

        # The exit does not remove C from `tradable_mask` / sellability.
        # `tradable_mask` remains membership-free; the research-input
        # exclusion is applied separately in `get_factor_prices` (R6c; see
        # test_factor_pit_membership.py).
        _assert_tradable_is_membership_free(pipeline)
        assert pipeline.universe_pit["applied"] is True
        assert pipeline.universe_mask.index.equals(dates)


class TestSnapshotDatedMask:
    """Tushare index_weight is monthly: the returned mask is snapshot-dated.

    A snapshot's membership holds until the next snapshot; dates before the
    first snapshot are unknown and must be False (never assumed members).
    """

    def test_snapshot_membership_holds_until_next_snapshot(self):
        provider = _MembershipProvider(
            history_available=True, snapshot_offsets=[10, 30, 60, 80],
        )
        pipeline = _run(provider)
        universe = pipeline.universe_mask

        # Before the first snapshot nothing is known: not members.
        assert not universe["000001.SZ"].iloc[:10].any()
        assert not universe["000002.SZ"].iloc[:10].any()
        # Snapshot at day 10 includes A; membership persists between
        # snapshots (day 15 is not a snapshot date).
        assert universe["000001.SZ"].iloc[10:].all()
        assert universe.loc[DATES[15], "000001.SZ"]
        assert not universe["000002.SZ"].iloc[10:60].any()
        # Snapshot at day 60 includes B; it stays a member to the end.
        assert universe["000002.SZ"].iloc[60:].all()
        assert universe.loc[DATES[65], "000002.SZ"]

        # None of this narrows tradability (R6b).
        _assert_tradable_is_membership_free(pipeline)


class TestSingleSnapshotRefused:
    def test_snapshot_fallback_is_not_applied_and_is_loud(self, caplog):
        provider = _MembershipProvider(history_available=False)
        with caplog.at_level(logging.WARNING):
            pipeline = _run(provider)

        # Nothing changed: 000002 was tradable under the ordinary rules and
        # the universe stays neutral (all True).
        assert pipeline.tradable_mask["000002.SZ"].all()
        _assert_universe_all_true(pipeline)
        assert pipeline.universe_pit["applied"] is False
        assert (
            pipeline.universe_pit["reason"]
            == "single_snapshot_constituent_history"
        )
        assert pipeline.universe_pit["source"] == "get_index_membership_mask"
        assert pipeline.universe_pit["index_code"] == "000300.SH"
        assert any(
            "single end-date snapshot" in record.getMessage()
            for record in caplog.records
        )


class TestNoHookProviderUnchanged:
    def test_baseline_mask_is_bit_identical(self):
        provider = _BaseFixtureProvider(
            suspended_window=("000003.SZ", 10, 15)
        )
        pipeline = _run(provider)

        expected = pd.DataFrame(
            True, index=DATES, columns=pd.Index(ASSETS, name="asset")
        )
        expected.loc[DATES[10]:DATES[15], "000003.SZ"] = False
        pd.testing.assert_frame_equal(
            pipeline.tradable_mask, expected, check_freq=False
        )

        _assert_universe_all_true(pipeline)
        assert pipeline.universe_pit == {
            "applied": False,
            "source": None,
            "reason": "provider_has_no_membership_mask",
        }

    def test_no_hook_prices_are_untouched(self):
        provider = _BaseFixtureProvider()
        pipeline = _run(provider)
        close = pipeline.prices["close"].unstack("asset")
        pd.testing.assert_frame_equal(close, provider._close)

    def test_refused_provider_matches_no_hook_provider(self):
        no_hook = _run(_BaseFixtureProvider())
        refused = _run(_MembershipProvider(history_available=False))
        pd.testing.assert_frame_equal(
            no_hook.tradable_mask, refused.tradable_mask
        )


class TestColumnMismatch:
    def test_missing_mask_column_applies_intersection_and_warns(self, caplog):
        provider = _MembershipProvider(
            history_available=True, drop_mask_columns=["000003"],
        )
        with caplog.at_level(logging.WARNING):
            pipeline = _run(provider)

        universe = pipeline.universe_mask
        # Intersection (A, B) applied...
        assert not universe["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert universe["000002.SZ"].iloc[INCLUSION_DAY:].all()
        assert universe["000001.SZ"].all()
        # ...and the asset with no mask column stays a member, not silently
        # excluded.
        assert universe["000003.SZ"].all()
        assert list(universe.columns) == ASSETS

        assert pipeline.universe_pit["applied"] is True
        _assert_tradable_is_membership_free(pipeline)
        assert any(
            "do not match the price panel" in record.getMessage()
            for record in caplog.records
        )


class TestSingleSnapshotMaskShape:
    """A flag of True cannot turn one snapshot date into per-date history.

    A provider cache may round-trip a single end-date fallback snapshot into
    a "constituents" table and recompute ``universe_history_available`` from
    its length on reload.  Applying that mask would blank every earlier day,
    so the pipeline refuses on the mask's own shape too.
    """

    def test_flag_true_but_one_snapshot_date_is_refused(self, caplog):
        provider = _MembershipProvider(
            history_available=True, snapshot_offsets=[N_DATES - 1],
        )
        with caplog.at_level(logging.WARNING):
            pipeline = _run(provider)

        assert pipeline.tradable_mask.all().all()
        _assert_universe_all_true(pipeline)
        assert pipeline.universe_pit["applied"] is False
        assert (
            pipeline.universe_pit["reason"]
            == "single_snapshot_constituent_history"
        )
        assert any(
            "distinct snapshot" in record.getMessage()
            for record in caplog.records
        )

    def test_missing_availability_attribute_is_refused(self):
        pipeline = _run(_HookWithoutFlagProvider())
        assert pipeline.tradable_mask.all().all()
        _assert_universe_all_true(pipeline)
        assert pipeline.universe_pit["applied"] is False
        assert (
            pipeline.universe_pit["reason"]
            == "single_snapshot_constituent_history"
        )

    def test_snapshots_outside_the_price_window_are_refused(self):
        # Two snapshot rows but zero usable dates in the price window would
        # otherwise ffill to an all-False mask and empty the universe.
        provider = _MembershipProvider(
            history_available=True, outside_window_mask=True,
        )
        pipeline = _run(provider)
        assert pipeline.tradable_mask.all().all()
        _assert_universe_all_true(pipeline)
        assert pipeline.universe_pit["applied"] is False
        assert (
            pipeline.universe_pit["reason"]
            == "single_snapshot_constituent_history"
        )


class TestMaskShapeEdges:
    def test_empty_mask_leaves_tradability_unchanged(self):
        provider = _MembershipProvider(history_available=True, empty_mask=True)
        pipeline = _run(provider)
        assert pipeline.tradable_mask.all().all()
        _assert_universe_all_true(pipeline)
        assert pipeline.universe_pit["applied"] is False
        assert pipeline.universe_pit["reason"] == "empty_membership_mask"

    def test_non_dataframe_mask_is_a_contract_error(self):
        provider = _MembershipProvider(history_available=True, bad_mask_type=True)
        with pytest.raises(DataContractError, match="DataFrame"):
            _run(provider)

    def test_extra_mask_column_is_ignored_with_warning(self, caplog):
        provider = _MembershipProvider(
            history_available=True, extra_mask_columns=True,
        )
        with caplog.at_level(logging.WARNING):
            pipeline = _run(provider)

        assert pipeline.universe_pit["applied"] is True
        assert pipeline.universe_mask is not None
        assert "000099.SZ" not in pipeline.universe_mask.columns
        assert not pipeline.universe_mask["000002.SZ"].iloc[:INCLUSION_DAY].any()
        _assert_tradable_is_membership_free(pipeline)
        assert any(
            "matching no asset" in record.getMessage()
            for record in caplog.records
        )


class TestValidatedWrapper:
    def test_membership_hook_is_read_through_the_wrapper(self):
        from quant_platform.data.providers.validated_provider import (
            ValidatedProvider,
        )

        primary = _MembershipProvider(history_available=True)
        wrapped = ValidatedProvider({"primary": primary}, primary="primary")
        # The wrapper does not implement (or forward) the optional hook...
        assert not hasattr(wrapped, "get_index_membership_mask")

        pipeline = _run(wrapped)
        # ...but the pipeline must still find the primary's membership data.
        assert pipeline.universe_pit["applied"] is True
        assert (
            pipeline.universe_pit["reason"]
            == "per_date_constituent_history"
        )
        assert not pipeline.universe_mask["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert pipeline.universe_mask["000002.SZ"].iloc[INCLUSION_DAY:].all()
        _assert_tradable_is_membership_free(pipeline)


class TestMaskFailureIsFatal:
    def test_fetch_exception_aborts_the_run(self):
        provider = _MembershipProvider(
            history_available=True,
            mask_exception=RuntimeError("network down"),
        )
        with pytest.raises(DataQualityError, match="get_index_membership_mask"):
            _run(provider)


class TestMainProvenanceWiring:
    def test_load_data_puts_universe_pit_on_metadata_attrs(self):
        from quant_platform.main import _load_data
        from quant_platform.utils.config import load_config

        config = load_config(str(DEFAULT_CONFIG))
        config.data.start_date = str(DATES[0].date())
        config.data.end_date = str(DATES[-1].date())
        provider = _MembershipProvider(history_available=True)
        resolution = {
            "provider": provider,
            "provider_name": "fixture_membership",
            "requested_provider": "fixture_membership",
            "effective_provider": "fixture_membership",
        }

        _, _, _, metadata, _, _ = _load_data(config, resolution=resolution)

        pit = metadata.attrs["universe_pit"]
        assert pit["applied"] is True
        assert pit["reason"] == "per_date_constituent_history"
        assert pit["index_code"] == "000300.SH"

        # R6b: the execution gates delivered to the backtest carry the
        # separate universe mask; tradability stays membership-free.
        gates = metadata.attrs["execution_gates"]
        assert "universe_mask" in gates
        assert "tradable_mask" in gates
        assert gates["tradable_mask"].all().all()
        universe = gates["universe_mask"]
        assert not universe["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert universe["000002.SZ"].iloc[INCLUSION_DAY:].all()

    def test_load_data_provenance_records_refusal(self):
        from quant_platform.main import _load_data
        from quant_platform.utils.config import load_config

        config = load_config(str(DEFAULT_CONFIG))
        config.data.start_date = str(DATES[0].date())
        config.data.end_date = str(DATES[-1].date())
        provider = _MembershipProvider(history_available=False)
        resolution = {
            "provider": provider,
            "provider_name": "fixture_snapshot",
            "requested_provider": "fixture_snapshot",
            "effective_provider": "fixture_snapshot",
        }

        _, _, _, metadata, _, _ = _load_data(config, resolution=resolution)

        pit = metadata.attrs["universe_pit"]
        assert pit["applied"] is False
        assert pit["reason"] == "single_snapshot_constituent_history"
