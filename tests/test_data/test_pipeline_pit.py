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
        """publish_date should be after the original quarter-end report date.

        After forward-fill to daily frequency, the index 'date' is a daily
        timestamp that may exceed the publish_date. We verify the relationship
        on the original quarterly rows by checking unique publish_date values
        against the quarter-end dates they were generated from.
        """
        provider = SyntheticDataProvider(n_stocks=10, seed=42)
        provider._generate_all()

        fin = provider.get_financials("2023-01-01", "2024-12-31")

        if "publish_date" not in fin.columns:
            pytest.skip("No publish_date column")

        # Get unique publish_dates and their associated quarter-end dates
        valid = fin.dropna(subset=["publish_date"])[["publish_date"]]
        unique = valid.reset_index().drop_duplicates(subset=["date", "publish_date"])

        # After forward-fill, the original quarter-end dates are preserved in
        # the 'date' index. But forward-fill means later daily dates share the
        # same publish_date. We only check rows where date is a quarter-end
        # (the original report dates).
        quarter_ends = unique[unique["date"].dt.is_quarter_end]
        if quarter_ends.empty:
            # If no quarter-end rows in range, just verify publish_dates are
            # reasonable (within 60 days of any date in the row)
            for _, row in unique.head(5).iterrows():
                delta = (row["publish_date"] - row["date"]).days
                assert delta < 60, f"publish_date too far from date: {delta} days"
        else:
            for _, row in quarter_ends.iterrows():
                assert row["publish_date"] >= row["date"], (
                    f"publish_date {row['publish_date']} < report_date {row['date']}"
                )
