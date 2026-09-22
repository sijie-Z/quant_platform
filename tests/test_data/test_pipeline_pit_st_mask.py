"""Point-in-time ST tradability mask tests (replaces `test_pipeline_excludes_st`).

The old test asserted ``not meta["is_st"].any()`` after a pipeline run — i.e.
"no stock that is *ever* ST may appear in the universe at all".  For the
synthetic provider ``is_st`` means *ever becomes ST* (the provider draws its
trigger date from the middle 90% of the sample), so that assertion was the
look-ahead written down as a specification: it demanded that stocks be dropped
from day one for ST events years in their future.  It also failed to notice
that the resulting universe was adversely selected.

These tests pin the intended semantics instead:

* a stock ST before the sample starts is excluded throughout;
* a stock that becomes ST on day D is eligible before D, ineligible from its
  announcement on;
* changing an ST event *after* a date must not change the universe *before*
  that date — the property that makes "no look-ahead" machine-checkable.

A fixed synthetic fixture is used rather than the random generator so the
dates under test do not move.
"""

from __future__ import annotations

import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider

ASSETS = ["000001.SH", "000002.SH", "000003.SH"]


class _FixedProvider(DataProvider):
    """Deterministic provider with an explicit ST episode list."""

    def __init__(self, dates, st_episodes):
        self._dates = pd.DatetimeIndex(dates, name="date")
        self._assets = list(ASSETS)
        self._st_episodes = list(st_episodes)

    # -- provider API ---------------------------------------------------
    def get_metadata(self) -> pd.DataFrame:
        ever_st = {ep["asset"] for ep in self._st_episodes}
        return pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Tech"],
                "market_cap_group": ["mid", "mid", "mid"],
                # Mirrors the synthetic provider: "became ST at some point",
                # NOT "is ST at the start of the sample".
                "is_st": [a in ever_st for a in self._assets],
                "listing_date": [pd.Timestamp("2000-01-01")] * 3,
                "delisting_date": [pd.NaT] * 3,
            },
            index=pd.Index(self._assets, name="asset"),
        )

    def get_prices(self, start_date, end_date) -> pd.DataFrame:
        # Level, gently trending series: no NaN, no limit moves.  Tiled across
        # the assets so every asset-day is present.
        values = [10.0, 10.0, 10.5, 11.0, 11.2, 11.5, 12.0, 12.5, 13.0, 13.2]
        level = pd.Series(values, index=self._dates).reindex(
            self._dates
        ).to_numpy()
        tiled = list(level) * len(self._assets)
        index = pd.MultiIndex.from_product(
            [self._dates, self._assets], names=["date", "asset"]
        )
        return pd.DataFrame(
            {
                "open": tiled,
                "high": [v * 1.01 for v in tiled],
                "low": [v * 0.99 for v in tiled],
                "close": tiled,
                "volume": 1_000_000.0,
                "turnover": 0.02,
                "adj_factor": 1.0,
            },
            index=index,
        )

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [self._dates, self._assets], names=["date", "asset"]
        )
        return pd.DataFrame(
            {"market_cap": 1e9, "pb": 1.5, "pe": 15.0}, index=index
        )

    def get_benchmark(self, start_date, end_date) -> pd.Series:
        return pd.Series(0.0005, index=self._dates)

    def get_st_timeseries(self) -> pd.DataFrame:
        if not self._st_episodes:
            return pd.DataFrame(
                columns=["asset", "announce_date", "end_date", "is_st"]
            )
        return pd.DataFrame(self._st_episodes)


def _run(dates, st_episodes, exclude_st=True) -> DataPipeline:
    pipeline = DataPipeline(
        provider=_FixedProvider(dates, st_episodes),
        start_date=str(pd.Timestamp(dates[0]).date()),
        end_date=str(pd.Timestamp(dates[-1]).date()),
        exclude_st=exclude_st,
    )
    pipeline.run()
    return pipeline


DATES = pd.date_range("2024-01-01", periods=10, freq="B")


class TestPITSTExclusion:
    def test_st_before_sample_start_is_excluded_throughout(self):
        """A stock already ST when the sample opens is never eligible."""
        # `announce_date` predates the sample entirely; the episode runs past
        # the last date, so the stock is ST on every date under test.
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000001.SH",
                    "announce_date": pd.Timestamp("2023-06-01"),
                    "end_date": pd.Timestamp("2025-06-01"),
                    "is_st": True,
                }
            ],
        )
        mask = pipeline.st_eligible
        # Ineligible on every date, so it is not part of the universe at all
        # and does not appear in the mask.
        assert "000001.SH" not in mask.columns
        assert "000001.SH" not in pipeline.get_valid_assets()
        # The other two are unaffected and remain eligible everywhere.
        assert mask[["000002.SH", "000003.SH"]].all().all()

    def test_st_from_announce_date_is_eligible_before_and_excluded_after(self):
        """Eligible before the announcement; excluded from it onward."""
        announce = DATES[4]
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000002.SH",
                    "announce_date": announce,
                    "end_date": DATES[-1],
                    "is_st": True,
                }
            ],
        )
        mask = pipeline.st_eligible["000002.SH"]

        before = mask.loc[mask.index < announce]
        after = mask.loc[mask.index >= announce]
        assert before.all()
        assert not after.any()
        assert len(before) == 4 and len(after) == 6

        # The asset is still part of the universe (it is tradable for part of
        # the sample) but its prices and returns are blanked while ST.
        assert "000002.SH" in pipeline.get_valid_assets()
        returns = pipeline.returns["000002.SH"]
        assert returns.loc[returns.index >= announce].isna().all()

    def test_st_episode_ends_and_asset_becomes_eligible_again(self):
        """The mask is per episode: eligibility resumes after `end_date`."""
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000003.SH",
                    "announce_date": DATES[2],
                    "end_date": DATES[5],
                    "is_st": True,
                }
            ],
        )
        mask = pipeline.st_eligible["000003.SH"]
        assert mask.loc[DATES[1]]
        assert not mask.loc[DATES[2]]
        assert not mask.loc[DATES[5]]
        assert mask.loc[DATES[6]]

    def test_two_episodes_with_a_clean_gap(self):
        """Overlapping/adjacent episodes collapse into one ineligible window."""
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000003.SH",
                    "announce_date": DATES[1],
                    "end_date": DATES[2],
                    "is_st": True,
                },
                {
                    "asset": "000003.SH",
                    "announce_date": DATES[5],
                    "end_date": DATES[6],
                    "is_st": True,
                },
            ],
        )
        mask = pipeline.st_eligible["000003.SH"]
        assert [bool(mask.loc[d]) for d in DATES] == [
            True, False, False, True, True, False, False, True, True, True,
        ]


class TestNoLookAhead:
    """The guard: future ST events must not move the past universe."""

    def test_future_st_events_do_not_change_earlier_universe(self):
        """Two worlds identical through day D, divergent after it.

        If the universe before D differed between them, the pipeline would be
        reading events that had not happened yet.  This is the assertion that
        replaces ``assert not meta["is_st"].any()``.
        """
        cutoff = DATES[4]
        before = [
            {
                "asset": "000003.SH",
                "announce_date": DATES[1],
                "end_date": DATES[2],
                "is_st": True,
            }
        ]
        # World A: 000001 becomes ST late in the sample.
        world_a = _run(
            DATES,
            before
            + [
                {
                    "asset": "000001.SH",
                    "announce_date": DATES[7],
                    "end_date": DATES[-1],
                    "is_st": True,
                }
            ],
        )
        # World B: 000001 becomes ST immediately after the cutoff.
        world_b = _run(
            DATES,
            before
            + [
                {
                    "asset": "000001.SH",
                    "announce_date": DATES[5],
                    "end_date": DATES[6],
                    "is_st": True,
                }
            ],
        )

        early_a = world_a.st_eligible.loc[world_a.st_eligible.index <= cutoff]
        early_b = world_b.st_eligible.loc[world_b.st_eligible.index <= cutoff]
        pd.testing.assert_frame_equal(early_a, early_b)

        # Both worlds keep 000001 through the cutoff — the old static filter
        # dropped it from day one because the metadata flag says "ever ST".
        assert "000001.SH" in early_a.columns
        assert early_a["000001.SH"].all()

        # Guard against a vacuous pass: the two worlds must actually differ
        # *after* the cutoff, otherwise the test proves nothing.
        late_a = world_a.st_eligible.loc[world_a.st_eligible.index > cutoff]
        late_b = world_b.st_eligible.loc[world_b.st_eligible.index > cutoff]
        assert not late_a.equals(late_b)

    def test_no_st_events_means_everything_is_eligible_everywhere(self):
        """Empty timeseries is not an error: nobody is ST, nobody is dropped."""
        pipeline = _run(DATES, [])
        assert pipeline.st_eligible.all().all()
        assert list(pipeline.get_valid_assets()) == ASSETS
        assert pipeline.returns.notna().any().all()

    def test_exclude_st_false_disables_the_mask(self):
        """The config flag still decides whether ST names are screened out."""
        episode = [
            {
                "asset": "000001.SH",
                "announce_date": pd.Timestamp("2023-06-01"),
                "end_date": pd.Timestamp("2025-06-01"),
                "is_st": True,
            }
        ]
        pipeline = _run(DATES, episode, exclude_st=False)
        assert pipeline.st_eligible is None
        assert list(pipeline.get_valid_assets()) == ASSETS
        assert pipeline.returns["000001.SH"].notna().any()

    def test_announce_date_is_used_not_trigger_date(self):
        """Eligibility ends at the announcement, never at the trigger."""
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000002.SH",
                    "trigger_date": DATES[1],
                    "announce_date": DATES[3],
                    "end_date": DATES[-1],
                    "is_st": True,
                }
            ],
        )
        mask = pipeline.st_eligible["000002.SH"]
        # Still eligible on the trigger day and the two days before the
        # announcement — the trader did not know yet.
        assert mask.loc[DATES[1]] and mask.loc[DATES[2]]
        assert not mask.loc[DATES[3]]

    def test_universe_stats_report_the_narrowing(self):
        """`universe_stats` exposes the effect of the mask for reporting."""
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000001.SH",
                    "announce_date": pd.Timestamp("2023-06-01"),
                    "end_date": pd.Timestamp("2025-06-01"),
                    "is_st": True,
                },
                {
                    "asset": "000002.SH",
                    "announce_date": DATES[4],
                    "end_date": DATES[-1],
                    "is_st": True,
                },
            ],
        )
        stats = pipeline.universe_stats()
        # 000001 is ST for the whole sample, so it is not in the universe at
        # all; 000002 is eligible early and excluded from its announcement on.
        assert "000001.SH" not in pipeline.get_valid_assets()
        assert stats["n_static_assets"] == 2
        assert stats["n_ever_st"] == 1
        assert stats["n_eligible_at_peak"] == 2
        assert stats["n_eligible_at_trough"] == 1

    def test_prices_are_blanked_while_st(self):
        """Blanked prices are how the mask reaches factors and the backtest."""
        announce = DATES[4]
        pipeline = _run(
            DATES,
            [
                {
                    "asset": "000002.SH",
                    "announce_date": announce,
                    "end_date": DATES[-1],
                    "is_st": True,
                }
            ],
        )
        close = pipeline.get_close()["000002.SH"]
        assert close.loc[close.index < announce].notna().all()
        assert close.loc[close.index >= announce].isna().all()


class TestLegacyProviderFallback:
    """Providers without an ST timeseries keep the old whole-sample filter."""

    def test_ever_st_asset_is_dropped_when_no_timeseries(self):
        class _NoTimeseriesProvider(_FixedProvider):
            def get_st_timeseries(self):
                raise AttributeError("provider has no ST timeseries")

        pipeline = DataPipeline(
            provider=_NoTimeseriesProvider(
                DATES,
                [
                    {
                        "asset": "000001.SH",
                        "announce_date": DATES[4],
                        "end_date": DATES[-1],
                        "is_st": True,
                    }
                ],
            ),
            start_date=str(DATES[0].date()),
            end_date=str(DATES[-1].date()),
            exclude_st=True,
        )
        pipeline.run()

        # No per-date information exists, so the sample-level filter is the
        # only defensible behaviour — and it is documented as such.
        assert "000001.SH" not in pipeline.valid_assets
        assert "000001.SH" not in pipeline.get_valid_assets()
        # What remains is eligible on every date: the mask carries no
        # per-date information, which is exactly the honest answer here.
        assert pipeline.st_eligible.all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
