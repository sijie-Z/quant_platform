"""Tests for synthetic data provider."""

import pandas as pd
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.data.schema import SECTORS


def test_provider_creation():
    provider = SyntheticDataProvider(n_stocks=50, start_date="2023-01-01", end_date="2023-12-31")
    assert provider.n_stocks == 50
    assert provider.start_date == pd.Timestamp("2023-01-01")


def test_get_prices(synthetic_provider):
    prices = synthetic_provider.get_prices("2023-01-01", "2023-06-30")
    assert prices is not None
    assert len(prices) > 0
    assert isinstance(prices.index, pd.MultiIndex)
    assert "close" in prices.columns
    assert "volume" in prices.columns


def test_price_columns(synthetic_provider):
    prices = synthetic_provider.get_prices("2023-01-01", "2024-12-31")
    expected = {"open", "high", "low", "close", "volume", "turnover", "adj_factor", "vwap"}
    assert expected.issubset(set(prices.columns))


def test_get_financials(synthetic_provider):
    fin = synthetic_provider.get_financials("2023-01-01", "2023-12-31")
    assert fin is not None
    assert "market_cap" in fin.columns
    assert "roe" in fin.columns
    assert "pb_ratio" in fin.columns


def test_get_benchmark(synthetic_provider):
    bench = synthetic_provider.get_benchmark("2023-01-01", "2023-12-31")
    assert len(bench) > 0
    assert bench.name == "benchmark"


def test_get_metadata(synthetic_provider):
    meta = synthetic_provider.get_metadata()
    assert len(meta) == 100
    assert "sector" in meta.columns
    assert "market_cap_group" in meta.columns
    assert "is_st" in meta.columns
    # All sectors should be valid
    assert all(s in SECTORS for s in meta["sector"])


def test_prices_no_negative(synthetic_provider):
    prices = synthetic_provider.get_prices("2023-01-01", "2024-12-31")
    for col in ["open", "high", "low", "close"]:
        valid = prices[col].dropna()
        assert (valid > 0).all(), f"{col} has non-positive values"


def test_reproducibility():
    """Same seed should produce identical data."""
    p1 = SyntheticDataProvider(n_stocks=20, seed=42)
    p2 = SyntheticDataProvider(n_stocks=20, seed=42)
    prices1 = p1.get_prices("2023-01-01", "2023-12-31")
    prices2 = p2.get_prices("2023-01-01", "2023-12-31")
    pd.testing.assert_frame_equal(prices1, prices2)


def test_returns_in_range(synthetic_provider):
    """Daily returns should be within price limit bounds (-10%, +10%).

    A resume day's change spans the forward-filled suspension and is not a
    one-day return; the pipeline masks those rows (issue #58), so the test
    must too.
    """
    prices = synthetic_provider.get_prices("2023-01-01", "2024-12-31")
    close = prices["close"].unstack("asset")
    suspended = (
        prices["is_suspended"].unstack("asset").fillna(False).astype(bool)
    )
    returns = close.pct_change(fill_method=None).mask(
        suspended | suspended.shift(1)
    )
    valid = returns.dropna()
    assert (valid >= -0.11).all().all()  # Allow small numerical error
    assert (valid <= 0.11).all().all()


def test_late_listers_trade_after_listing_and_not_before():
    """A NaN previous close made the listing day NaN forever.

    `close[t] = close[t-1] * (1+r)` with `close[t-1]` NaN (pre-listing)
    propagated NaN through the rest of the sample; the pipeline then dropped
    those stocks as "long suspensions".
    """
    provider = SyntheticDataProvider(
        n_stocks=60, start_date="2023-01-01", end_date="2024-12-31", seed=7,
    )
    prices = provider.get_prices("2023-01-01", "2024-12-31")
    close = prices["close"].unstack("asset")
    listing = pd.to_datetime(provider.get_metadata()["listing_date"])

    late = [a for a in close.columns if listing[a] > close.index[0]]
    assert late, "expected some late listers in the sample"

    for asset in late:
        after = close.loc[listing[asset]:, asset]
        assert after.notna().mean() > 0.95, (
            f"{asset} has prices on only {after.notna().mean():.1%} of its "
            "listed days"
        )
        before = close.loc[:listing[asset] - pd.Timedelta(days=1), asset]
        assert before.isna().all(), f"{asset} is priced before it listed"


class TestReportsPublishedOnNonTradingDays:
    """publish_date is a calendar date, the panel only has trading days.

    Reindexing the financial panel straight onto `publish_date` silently
    dropped every report published on a weekend/holiday -- the (date, asset)
    key did not exist -- so a 20-quarter sample kept only 14-16. Each report
    is now mapped to the first trading day on/after its publication.
    """

    START = "2023-01-01"
    END = "2025-05-31"  # every quarter-end in range publishes before this

    @staticmethod
    def _provider():
        provider = SyntheticDataProvider(
            n_stocks=20, start_date="2023-01-01", end_date="2025-05-31", seed=11,
        )
        provider._generate_all()
        return provider

    @staticmethod
    def _reports(provider):
        fin = provider.get_financials("2023-01-01", "2025-05-31")
        # ffill repeats publish_date on every later day; one row per report.
        return fin.reset_index().dropna(subset=["publish_date"]).drop_duplicates(
            subset=["asset", "publish_date"]
        )

    def test_every_generated_report_is_in_the_panel(self):
        provider = self._provider()
        all_dates = pd.bdate_range(self.START, self.END)
        quarter_ends = all_dates[all_dates.is_quarter_end]
        assert len(quarter_ends) > 0

        reports = self._reports(provider)
        per_asset = reports.groupby("asset").size()

        assert (per_asset == len(quarter_ends)).all(), (
            f"reports lost: {per_asset.to_dict()}, expected {len(quarter_ends)} "
            f"per asset"
        )

    def test_reports_published_on_weekends_are_not_dropped(self):
        provider = self._provider()
        reports = self._reports(provider)

        publish = pd.to_datetime(reports["publish_date"])
        weekend = reports[publish.dt.dayofweek >= 5]
        assert not weekend.empty, (
            "seed must produce at least one weekend publication for this test "
            "to exercise the mapping"
        )

        # Every report is visible on the first trading day on/after publication.
        all_dates = pd.bdate_range(self.START, self.END)
        expected_dates = all_dates[
            all_dates.searchsorted(publish.to_numpy(), side="left")
        ]
        assert (pd.to_datetime(reports["date"]).to_numpy() == expected_dates.to_numpy()).all()

    def test_no_report_is_visible_before_it_was_published(self):
        provider = self._provider()
        reports = self._reports(provider)

        visible = pd.to_datetime(reports["date"])
        published = pd.to_datetime(reports["publish_date"])
        assert (visible >= published).all()


def test_industry_changes_are_stable_across_calls_and_instances():
    """`get_industry_changes()` used the shared RNG, so each call (and each
    pipeline run) produced a different classification history."""
    kwargs = dict(n_stocks=30, start_date="2023-01-01", end_date="2024-12-31", seed=5)

    first = SyntheticDataProvider(**kwargs).get_industry_changes()
    provider = SyntheticDataProvider(**kwargs)
    again = provider.get_industry_changes()
    twice = provider.get_industry_changes()

    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(again, twice)
