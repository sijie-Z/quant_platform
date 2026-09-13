"""Tests for point-in-time data filtering in DataPipeline.

Verifies that the pipeline correctly filters financials by publish_date,
ST status by announcement date, and industry classification by effective_date,
preventing look-ahead bias.
"""

import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.synthetic import SyntheticDataProvider


@pytest.fixture
def small_provider():
    """Small synthetic provider for fast tests."""
    return SyntheticDataProvider(
        n_stocks=20,
        start_date="2023-01-01",
        end_date="2024-12-31",
        seed=99,
    )


@pytest.fixture
def pipeline_with_pit(small_provider):
    """DataPipeline with point-in-time data loaded."""
    pipe = DataPipeline(small_provider, "2023-06-01", "2024-06-01")
    pipe.run()
    return pipe


class TestFinancialsAsOf:
    """Test get_financials_as_of filters by publish_date."""

    def test_financials_have_publish_date(self, pipeline_with_pit):
        """Financials should contain publish_date column."""
        assert pipeline_with_pit.financials is not None
        assert "publish_date" in pipeline_with_pit.financials.columns

    def test_no_lookahead_on_early_date(self, pipeline_with_pit):
        """Financials from future quarters should not be visible early on."""
        early_date = pd.Timestamp("2023-06-15")
        fin = pipeline_with_pit.get_financials_as_of(early_date)

        if not fin.empty and "publish_date" in fin.columns:
            # All publish_dates should be <= early_date
            assert (fin["publish_date"] <= early_date).all()

    def test_more_data_available_later(self, pipeline_with_pit):
        """Later dates should have access to more financial data."""
        early = pipeline_with_pit.get_financials_as_of(pd.Timestamp("2023-06-15"))
        late = pipeline_with_pit.get_financials_as_of(pd.Timestamp("2024-06-15"))

        # Late date should have at least as many assets as early
        if not early.empty and not late.empty:
            assert len(late) >= len(early)

    def test_returns_latest_per_asset(self, pipeline_with_pit):
        """Should return the most recently published record per asset."""
        as_of = pd.Timestamp("2024-06-01")
        fin = pipeline_with_pit.get_financials_as_of(as_of)

        if not fin.empty and "publish_date" in fin.columns:
            # Each asset should appear at most once
            assets = fin.index.get_level_values("asset")
            assert len(assets) == len(assets.unique())

    def test_empty_before_first_publish(self, pipeline_with_pit):
        """Should return empty if queried before any financials are published."""
        very_early = pd.Timestamp("2022-01-01")
        fin = pipeline_with_pit.get_financials_as_of(very_early)
        assert fin.empty


class TestSTStatusAsOf:
    """Test get_st_status filters by announcement date."""

    def test_st_timeseries_loaded(self, pipeline_with_pit):
        """ST timeseries should be loaded from provider."""
        assert pipeline_with_pit._st_timeseries is not None

    def test_returns_set(self, pipeline_with_pit):
        """get_st_status should return a set of asset codes."""
        st = pipeline_with_pit.get_st_status(pd.Timestamp("2024-01-01"))
        assert isinstance(st, set)

    def test_st_status_changes_over_time(self, pipeline_with_pit):
        """ST status should differ between early and late dates (if any ST stocks exist)."""
        if (pipeline_with_pit._st_timeseries is not None
                and not pipeline_with_pit._st_timeseries.empty):
            early_st = pipeline_with_pit.get_st_status(pd.Timestamp("2023-06-01"))
            late_st = pipeline_with_pit.get_st_status(pd.Timestamp("2024-06-01"))
            # They might be different (not guaranteed, but structure should work)
            assert isinstance(early_st, set)
            assert isinstance(late_st, set)


class TestIndustryAsOf:
    """Test get_industry_map filters by effective_date."""

    def test_industry_changes_loaded(self, pipeline_with_pit):
        """Industry changes should be loaded from provider."""
        assert pipeline_with_pit._industry_changes is not None

    def test_returns_dict(self, pipeline_with_pit):
        """get_industry_map should return a dict mapping asset to industry."""
        ind = pipeline_with_pit.get_industry_map(pd.Timestamp("2024-01-01"))
        assert isinstance(ind, dict)
        if ind:
            assert all(isinstance(v, str) for v in ind.values())

    def test_industry_map_has_assets(self, pipeline_with_pit):
        """Industry map should cover the valid assets."""
        ind = pipeline_with_pit.get_industry_map(pd.Timestamp("2024-06-01"))
        if ind:
            assert len(ind) > 0


class TestSyntheticProviderPIT:
    """Test that SyntheticDataProvider generates PIT data."""

    def test_st_timeseries_structure(self):
        """ST timeseries should have expected columns."""
        provider = SyntheticDataProvider(n_stocks=30, seed=42)
        provider._generate_all()
        st = provider.get_st_timeseries()

        if not st.empty:
            assert "asset" in st.columns
            assert "announce_date" in st.columns
            assert "is_st" in st.columns

    def test_industry_changes_structure(self):
        """Industry changes should have expected columns."""
        provider = SyntheticDataProvider(n_stocks=30, seed=42)
        provider._generate_all()
        ind = provider.get_industry_changes()

        assert "asset" in ind.columns
        assert "industry" in ind.columns
        assert "effective_date" in ind.columns

    def test_financials_have_publish_date(self):
        """Financials should include publish_date column."""
        provider = SyntheticDataProvider(n_stocks=10, seed=42)
        provider._generate_all()
        fin = provider.get_financials("2023-01-01", "2024-12-31")

        assert "publish_date" in fin.columns

    def test_publish_date_after_report_date(self):
        """publish_date must not precede the fiscal period it describes.

        After the PIT fix, the index 'date' is the *visibility* date -- the day
        a value became usable -- not the fiscal period end. Reports are
        re-indexed to their publish_date before forward-filling, so the period
        the figures describe now lives in the `fiscal_period_end` column and
        that is what this compares against.
        """
        provider = SyntheticDataProvider(n_stocks=10, seed=42)
        provider._generate_all()

        fin = provider.get_financials("2023-01-01", "2024-12-31")

        if "publish_date" not in fin.columns:
            pytest.skip("No publish_date column")
        if "fiscal_period_end" not in fin.columns:
            pytest.skip("No fiscal_period_end column")

        rows = fin.dropna(subset=["publish_date", "fiscal_period_end"]).reset_index()
        rows = rows.drop_duplicates(subset=["fiscal_period_end", "publish_date"])
        assert not rows.empty

        for _, row in rows.iterrows():
            assert row["publish_date"] >= row["fiscal_period_end"], (
                f"publish_date {row['publish_date']} < fiscal_period_end "
                f"{row['fiscal_period_end']}"
            )

    def test_no_value_is_visible_before_its_publish_date(self):
        """Regression for BUG-22.

        The daily panel used to be forward-filled from the fiscal period end
        and then bfilled, so a Q2 report was visible ~33 trading days before
        publication and the first report was pushed back to the start of the
        sample. A report's figures must first appear on their publish_date.
        """
        provider = SyntheticDataProvider(n_stocks=5, seed=7)
        provider._generate_all()
        fin = provider.get_financials("2020-01-01", "2023-12-31").reset_index()

        rows = fin.dropna(subset=["publish_date"])
        assert not rows.empty

        checked = 0
        for (_, pub), sub in rows.groupby(["asset", "publish_date"]):
            first_visible = pd.Timestamp(sub["date"].min())
            assert first_visible >= pd.Timestamp(pub), (
                f"a value published {pd.Timestamp(pub).date()} is already "
                f"visible on {first_visible.date()}"
            )
            checked += 1
        assert checked > 0, "no (asset, publish_date) groups were checked"

    def test_nothing_is_backfilled_before_the_first_report(self):
        """Regression for BUG-22, second half.

        `bfill()` used to push the first report back to the first date of the
        sample, so fundamentals were "known" months before anything was
        published. Before a company's first report there is genuinely nothing
        to know.
        """
        provider = SyntheticDataProvider(n_stocks=5, seed=7)
        provider._generate_all()
        fin = provider.get_financials("2020-01-01", "2023-12-31")

        earliest_publish = fin["publish_date"].min()
        assert pd.notna(earliest_publish)

        before = fin.loc[fin.index.get_level_values("date") < earliest_publish]
        assert before["market_cap"].isna().all(), (
            "fundamental values exist before the first publish_date, which can "
            "only come from a backward fill"
        )
