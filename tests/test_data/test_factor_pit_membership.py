"""Factor/signal cross-section PIT (R6c): index membership reaches research inputs.

R6b split index membership out of ``tradable_mask`` so an index exit never
traps a held position.  The factor and signal cross-sections were still built
over the whole-sample union of constituents: a name that joined the index on
day 60 received factor values and a signal from day 0.  ``get_factor_prices``
now folds the applied ``universe_mask`` into ``excluded_mask``, which
``main._compute_factors`` and ``main._generate_signal`` re-apply.

These tests pin:

* membership exclusion in ``excluded_mask`` (join blanked before inclusion,
  exit blanked after), as a union with the R6b tradability exclusions;
* a technical factor and the final signal are NaN outside membership while
  the price history itself is not cropped;
* bit-for-bit pre-R6c behaviour when no membership history was applied --
  providers without the hook and the single-snapshot fallback;
* ``get_tradable_mask()`` / execution gates stay membership-free (R6b).
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider
from quant_platform.factors.technical import Momentum1M

N_DATES = 180
INCLUSION_DAY = 60
EXIT_DAY = 120
DATES = pd.bdate_range("2024-01-02", periods=N_DATES, name="date")
ASSETS = ["000001.SZ", "000002.SZ", "000003.SZ"]
#: 000002.SZ joins the index here; 000003.SZ leaves it at EXIT_DAY.
MOMENTUM_WARMUP = 21


def _close_panel(seed: int = 29) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0006, 0.009, (N_DATES, len(ASSETS)))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=DATES,
        columns=pd.Index(ASSETS, name="asset"),
    )


class _BaseFixtureProvider(DataProvider):
    """Deterministic provider with no membership hook (pre-R6c baseline)."""

    def __init__(self, *, suspended_window: tuple[str, int, int] | None = None):
        self._close = _close_panel()
        self._suspended_window = suspended_window

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
        suspended = np.zeros(len(index), dtype=bool)
        if self._suspended_window is not None:
            asset, first, last = self._suspended_window
            dates = index.get_level_values("date")
            assets = index.get_level_values("asset")
            suspended = np.asarray(
                (dates >= DATES[first]) & (dates <= DATES[last]) & (assets == asset),
                dtype=bool,
            )
        frame["is_suspended"] = suspended
        frame = frame.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        if fields:
            frame = frame[fields]
        return frame

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [DATES, ASSETS], names=["date", "asset"]
        )
        return pd.DataFrame(
            {
                "market_cap": 1e9,
                "total_assets": 2e9,
                "net_assets": 1e9,
                "revenue": 1e8,
                "net_profit": 1e7,
                "roe": 0.10,
                "pb_ratio": 1.5,
                "pe_ratio": 15.0,
                "asset_growth": 0.05,
            },
            index=index,
        ).loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

    def get_benchmark(self, start_date, end_date) -> pd.Series:
        return pd.Series(0.0005, index=DATES, name="benchmark").loc[
            pd.Timestamp(start_date):pd.Timestamp(end_date)
        ]


class _MembershipProvider(_BaseFixtureProvider):
    """Provider exposing a per-date constituent history.

    ``history_available=False`` models the single end-date snapshot fallback:
    the hook exists but must not be trusted (and must not reach research
    inputs).
    """

    def __init__(
        self,
        *,
        history_available: bool = True,
        exit_day: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.universe_history_available = history_available
        self.index_code = "000300.SH"
        self._exit_day = exit_day

    def get_index_membership_mask(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        bare = [asset.split(".")[0] for asset in ASSETS]
        mask = pd.DataFrame(True, index=DATES, columns=bare)
        mask.iloc[:INCLUSION_DAY, 1] = False  # 000002 joins here
        if self._exit_day is not None:
            mask.iloc[self._exit_day:, 2] = False  # 000003 leaves here
        return mask.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]


def _run(provider: DataProvider) -> DataPipeline:
    pipeline = DataPipeline(
        provider=provider,
        start_date=str(DATES[0].date()),
        end_date=str(DATES[-1].date()),
    )
    pipeline.run()
    return pipeline


def _factor_config() -> types.SimpleNamespace:
    """Minimal config: one technical factor, all processing disabled."""
    return types.SimpleNamespace(
        factors=types.SimpleNamespace(
            enabled_technicals={"momentum_1m"},
            enabled_fundamentals=set(),
            params={},
            processing={
                "winsorize": {"enabled": False},
                "standardize": {"enabled": False},
                "neutralize": {"enabled": False},
            },
        )
    )


def _alpha_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        alpha=types.SimpleNamespace(
            method="equal_weight",
            lookback=252,
            min_icir=0.0,
            tradability_gate=False,
            min_tradability=0.3,
        )
    )


def _compute_factors(pipeline: DataPipeline) -> pd.DataFrame:
    from quant_platform import main as cli

    prices = pipeline.get_factor_prices()
    processed, _, _, _ = cli._compute_factors(
        prices,
        pipeline.returns,
        pipeline.financials,
        pipeline.metadata,
        turnover=pipeline.get_turnover(),
        config=_factor_config(),
    )
    return processed


def _generate_signal(pipeline: DataPipeline, processed: dict) -> pd.DataFrame:
    from quant_platform import main as cli

    return cli._generate_signal(
        _alpha_config(), processed, pipeline.returns,
        prices=pipeline.get_factor_prices(),
    )


def _expected_excluded(pipeline: DataPipeline) -> pd.DataFrame:
    """Pre-R6c formula: exclusion comes from tradability only."""
    close = pipeline.get_factor_prices()
    return (~pipeline.tradable_mask).reindex(
        index=close.index, columns=close.columns, fill_value=True,
    )


class TestMembershipReachesResearchInputs:
    """R6c: factors and signals are blank outside tradeable-and-member days."""

    @pytest.fixture()
    def pipeline(self) -> DataPipeline:
        return _run(_MembershipProvider(history_available=True, exit_day=EXIT_DAY))

    def test_excluded_mask_is_tradability_union_membership(self, pipeline):
        prices = pipeline.get_factor_prices()
        excluded = prices.attrs["excluded_mask"]
        universe = pipeline.universe_mask

        # Fixture has no ST/suspension days, so tradability alone is all True;
        # every exclusion below is membership-driven.
        assert pipeline.get_tradable_mask().all().all()
        assert universe is not None
        assert excluded.dtypes.eq(bool).all()

        # A is a member throughout.
        assert not excluded["000001.SZ"].any()
        # B has no factor/signal before inclusion, but is unexcluded after.
        assert excluded["000002.SZ"].iloc[:INCLUSION_DAY].all()
        assert not excluded["000002.SZ"].iloc[INCLUSION_DAY:].any()
        # C is unexcluded while a member and blanked from the exit onward.
        assert not excluded["000003.SZ"].iloc[:EXIT_DAY].any()
        assert excluded["000003.SZ"].iloc[EXIT_DAY:].all()

        # Exact combination: (~tradable) | (~universe), aligned to the panel.
        expected = (~pipeline.tradable_mask) | (~universe)
        expected = expected.reindex(
            index=prices.index, columns=prices.columns, fill_value=True,
        )
        pd.testing.assert_frame_equal(excluded, expected, check_freq=False)

        # The price panel itself is not cropped: B's pre-inclusion prices and
        # C's post-exit prices remain so factor lookbacks stay possible.
        close = pipeline.get_close()
        assert close["000002.SZ"].iloc[:INCLUSION_DAY].notna().all()
        assert close["000003.SZ"].iloc[EXIT_DAY:].notna().all()

    def test_excluded_mask_unions_with_suspension_exclusions(self):
        provider = _MembershipProvider(
            history_available=True,
            exit_day=EXIT_DAY,
            suspended_window=("000001.SZ", 10, 15),
        )
        pipeline = _run(provider)
        excluded = pipeline.get_factor_prices().attrs["excluded_mask"]

        # Suspension still excludes through tradability...
        assert excluded.loc[DATES[10]:DATES[15], "000001.SZ"].all()
        # ...and membership still excludes through the universe mask.
        assert excluded["000002.SZ"].iloc[:INCLUSION_DAY].all()

        expected = (
            (~pipeline.tradable_mask) | (~pipeline.universe_mask)
        )
        close = pipeline.get_factor_prices()
        expected = expected.reindex(
            index=close.index, columns=close.columns, fill_value=True,
        )
        pd.testing.assert_frame_equal(excluded, expected, check_freq=False)

    def test_technical_factor_is_blank_outside_membership(self, pipeline):
        processed = _compute_factors(pipeline)
        values = processed["momentum_1m"]

        # A always a member: momentum is live once the lookback warms up.
        assert values["000001.SZ"].iloc[MOMENTUM_WARMUP:].notna().all()

        # B: all NaN before inclusion, live again after the warm-up rebuilt
        # from the (uncropped) earlier price history.
        assert values["000002.SZ"].iloc[:INCLUSION_DAY].isna().all()
        first_live = INCLUSION_DAY + MOMENTUM_WARMUP
        assert values["000002.SZ"].iloc[first_live:].notna().all()

        # C: live while a member, NaN from the exit onward.
        assert values["000003.SZ"].iloc[:EXIT_DAY].iloc[MOMENTUM_WARMUP:].notna().all()
        assert values["000003.SZ"].iloc[EXIT_DAY:].isna().all()

    def test_signal_is_blank_outside_membership(self, pipeline):
        processed = _compute_factors(pipeline)
        signal = _generate_signal(pipeline, processed)

        # The alpha combination fills NaN factors with 0, so only the final
        # re-mask keeps non-members out of the signal cross-section.
        assert signal["000002.SZ"].iloc[:INCLUSION_DAY].isna().all()
        assert signal["000002.SZ"].iloc[INCLUSION_DAY:].notna().all()
        assert signal["000003.SZ"].iloc[:EXIT_DAY].notna().all()
        assert signal["000003.SZ"].iloc[EXIT_DAY:].isna().all()
        assert signal["000001.SZ"].iloc[MOMENTUM_WARMUP:].notna().all()

    def test_tradable_mask_and_execution_gates_stay_membership_free(self, pipeline):
        """R6b execution semantics must not move: membership is research-only."""
        baseline = _run(_BaseFixtureProvider())
        pd.testing.assert_frame_equal(
            pipeline.get_tradable_mask(), baseline.get_tradable_mask(),
            check_freq=False,
        )
        assert pipeline.get_tradable_mask().all().all()

        gates = pipeline.get_execution_gates()
        pd.testing.assert_frame_equal(
            gates["tradable_mask"], pipeline.get_tradable_mask(),
        )
        pd.testing.assert_frame_equal(
            gates["universe_mask"], pipeline.universe_mask,
        )
        assert gates["tradable_mask"].all().all()
        assert not gates["universe_mask"]["000002.SZ"].iloc[:INCLUSION_DAY].any()
        # Membership did reach research inputs (the point of R6c).
        assert (
            pipeline.get_factor_prices().attrs["excluded_mask"]
            != _expected_excluded(pipeline)
        ).any().any()


class TestNoMembershipHistoryUnchanged:
    """R6c must be gated on `universe_pit["applied"]`, not on mask presence."""

    def test_no_hook_excluded_mask_is_pre_r6c_formula(self):
        pipeline = _run(_BaseFixtureProvider())
        prices = pipeline.get_factor_prices()
        excluded = prices.attrs["excluded_mask"]

        pd.testing.assert_frame_equal(
            excluded, _expected_excluded(pipeline), check_freq=False,
        )
        # Provenance: the pipeline did not apply a membership mask at all.
        assert pipeline.universe_pit["applied"] is False
        assert pipeline.universe_pit["reason"] == "provider_has_no_membership_mask"

    def test_no_hook_factor_and_signal_match_pre_r6c_panels(self):
        pipeline = _run(_BaseFixtureProvider())
        prices = pipeline.get_factor_prices()

        processed = _compute_factors(pipeline)
        raw = Momentum1M().run(prices).values
        excluded = prices.attrs["excluded_mask"].reindex(
            index=raw.index, columns=raw.columns, fill_value=True,
        )
        pd.testing.assert_frame_equal(
            processed["momentum_1m"], raw.mask(excluded), check_freq=False,
        )

        signal = _generate_signal(pipeline, processed)
        # The only rows dropped from the signal are the tradability exclusions
        # (none in this fixture), never membership.
        assert pd.notna(signal).all().all()

    def test_single_snapshot_fallback_matches_no_hook_bit_for_bit(self):
        baseline = _run(_BaseFixtureProvider())
        refused = _run(_MembershipProvider(history_available=False))

        # The fallback keeps an all-True universe panel available, but R6c
        # must not merge it: only an *applied* per-date history narrows
        # research inputs.
        assert refused.universe_pit["applied"] is False
        assert (
            refused.universe_pit["reason"]
            == "single_snapshot_constituent_history"
        )
        assert refused.universe_mask.all().all()

        pd.testing.assert_frame_equal(
            refused.get_factor_prices().attrs["excluded_mask"],
            baseline.get_factor_prices().attrs["excluded_mask"],
        )
        pd.testing.assert_frame_equal(
            refused.get_tradable_mask(), baseline.get_tradable_mask(),
            check_freq=False,
        )

        refused_processed = _compute_factors(refused)
        baseline_processed = _compute_factors(baseline)
        pd.testing.assert_frame_equal(
            refused_processed["momentum_1m"],
            baseline_processed["momentum_1m"],
        )
        pd.testing.assert_frame_equal(
            _generate_signal(refused, refused_processed),
            _generate_signal(baseline, baseline_processed),
        )

    def test_suspended_no_hook_matches_suspended_fallback(self):
        """Tradability exclusions alone must be identical across both paths."""
        baseline = _run(
            _BaseFixtureProvider(suspended_window=("000003.SZ", 10, 15))
        )
        refused = _run(
            _MembershipProvider(
                history_available=False,
                suspended_window=("000003.SZ", 10, 15),
            )
        )
        pd.testing.assert_frame_equal(
            refused.get_factor_prices().attrs["excluded_mask"],
            baseline.get_factor_prices().attrs["excluded_mask"],
        )
        # Sanity: the suspension rows are excluded through tradability alone.
        assert refused.get_factor_prices().attrs["excluded_mask"].loc[
            DATES[10]:DATES[15], "000003.SZ"
        ].all()
