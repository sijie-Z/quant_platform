"""Index-membership PIT gate (R6a).

The provider already builds a per-date constituent mask
(``get_index_membership_mask``), but the pipeline never consumed it: a stock
that joined the index mid-sample was treated as a member over the whole
sample ("样本期内成分并集").  These tests pin the gate:

* per-date membership history is ANDed into ``tradable_mask``;
* a provider whose constituent data is a single end-date snapshot is refused
  (replaying it as history is the same look-ahead);
* providers without the hook behave exactly as before;
* provenance reaches ``metadata.attrs["universe_pit"]`` through
  ``main._load_data``;
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
from quant_platform.data.schema import DataQualityError

N_DATES = 90
DATES = pd.bdate_range("2024-01-02", periods=N_DATES, name="date")
ASSETS = ["000001.SZ", "000002.SZ", "000003.SZ"]
#: 000002.SZ joins the index on this business-day offset.
INCLUSION_DAY = 60

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _close_panel(seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0006, 0.009, (N_DATES, len(ASSETS)))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=DATES,
        columns=pd.Index(ASSETS, name="asset"),
    )


class _BaseFixtureProvider(DataProvider):
    """Deterministic provider with no membership hook (baseline behaviour)."""

    def __init__(self, *, suspended_window: tuple[str, int, int] | None = None):
        self._close = _close_panel()
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

    def get_prices(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [DATES, ASSETS], names=["date", "asset"]
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
            window = (DATES[first], DATES[last], asset)
            suspended = (
                (index.get_level_values("date") >= window[0])
                & (index.get_level_values("date") <= window[1])
                & (index.get_level_values("asset") == window[2])
            )
        frame["is_suspended"] = np.asarray(suspended, dtype=bool)
        return frame

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [DATES, ASSETS], names=["date", "asset"]
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
        return pd.Series(0.0005, index=DATES, name="benchmark")


class _MembershipProvider(_BaseFixtureProvider):
    """Provider exposing a per-date constituent history.

    ``history_available=False`` models the single end-date snapshot fallback:
    the hook exists but must not be trusted as history.
    """

    def __init__(
        self,
        *,
        history_available: bool = True,
        drop_mask_columns: list[str] | None = None,
        mask_exception: Exception | None = None,
        index_code: str | None = "000300.SH",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.universe_history_available = history_available
        self.index_code = index_code
        self._drop_mask_columns = list(drop_mask_columns or [])
        self._mask_exception = mask_exception

    def get_index_membership_mask(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        if self._mask_exception is not None:
            raise self._mask_exception
        # Bare codes on purpose: the pipeline must normalize mask labels just
        # like every other provider panel.
        bare = [asset.split(".")[0] for asset in ASSETS]
        mask = pd.DataFrame(True, index=DATES, columns=bare)
        mask.iloc[:INCLUSION_DAY, 1] = False  # 000002 joins at INCLUSION_DAY
        if self._drop_mask_columns:
            mask = mask.drop(columns=self._drop_mask_columns)
        return mask.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]


def _run(provider: DataProvider) -> DataPipeline:
    pipeline = DataPipeline(
        provider=provider,
        start_date=str(DATES[0].date()),
        end_date=str(DATES[-1].date()),
    )
    pipeline.run()
    return pipeline


class TestMembershipMaskApplied:
    def test_per_date_history_narrows_tradable_mask(self):
        pipeline = _run(_MembershipProvider(history_available=True))

        mask = pipeline.tradable_mask
        assert list(mask.columns) == ASSETS
        assert not mask["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert mask["000002.SZ"].iloc[INCLUSION_DAY:].all()
        # Members throughout are unaffected.
        assert mask["000001.SZ"].all()
        assert mask["000003.SZ"].all()

        pit = pipeline.universe_pit
        assert pit["applied"] is True
        assert pit["source"] == "get_index_membership_mask"
        assert pit["index_code"] == "000300.SH"
        assert pit["reason"] == "per_date_constituent_history"

        assert pipeline.universe_mask is not None
        assert list(pipeline.universe_mask.columns) == ASSETS
        assert pipeline.universe_mask.index.equals(DATES)
        assert not pipeline.universe_mask["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert pipeline.universe_mask["000002.SZ"].iloc[INCLUSION_DAY:].all()

        # Membership gates tradability only: the price history stays intact so
        # factor lookbacks across the inclusion date still work.
        close = pipeline.prices["close"].unstack("asset")
        assert close["000002.SZ"].iloc[:INCLUSION_DAY].notna().all()
        assert close["000002.SZ"].iloc[INCLUSION_DAY:].notna().all()

        # The execution gates consume the same narrowed mask.
        gates = pipeline.get_execution_gates()
        pd.testing.assert_frame_equal(gates["tradable_mask"], mask)

    def test_mask_labels_are_normalized(self):
        pipeline = _run(_MembershipProvider(history_available=True))
        # 000001 (bare) in the mask must line up with 000001.SZ in the panel.
        assert pipeline.universe_mask is not None
        assert "000001.SZ" in pipeline.universe_mask.columns
        assert pipeline.universe_mask["000001.SZ"].all()


class TestSingleSnapshotRefused:
    def test_snapshot_fallback_is_not_applied_and_is_loud(self, caplog):
        provider = _MembershipProvider(history_available=False)
        with caplog.at_level(logging.WARNING):
            pipeline = _run(provider)

        # Nothing changed: 000002 was tradable under the ordinary rules.
        assert pipeline.tradable_mask["000002.SZ"].all()
        assert pipeline.universe_mask is None
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

        assert pipeline.universe_mask is None
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

        mask = pipeline.tradable_mask
        # Intersection (A, B) applied...
        assert not mask["000002.SZ"].iloc[:INCLUSION_DAY].any()
        assert mask["000002.SZ"].iloc[INCLUSION_DAY:].all()
        assert mask["000001.SZ"].all()
        # ...and the asset with no mask column is left unchanged, not silently
        # excluded.
        assert mask["000003.SZ"].all()

        assert pipeline.universe_pit["applied"] is True
        assert pipeline.universe_mask is not None
        assert list(pipeline.universe_mask.columns) == ["000001.SZ", "000002.SZ"]
        assert any(
            "do not match the tradable panel" in record.getMessage()
            for record in caplog.records
        )


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
