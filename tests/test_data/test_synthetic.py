"""Tests for synthetic data provider."""

import pandas as pd
import pytest
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
    is now mapped to the first exchange session on/after its publication.
    """

    START = "2023-01-01"
    END = "2025-05-31"  # every full quarter-end in range publishes before this

    @staticmethod
    def _provider():
        # seed 7 publishes a Q4-2023 report on 2024-02-12, a Spring Festival
        # exchange holiday, and several reports on weekends.
        provider = SyntheticDataProvider(
            n_stocks=20, start_date="2023-01-01", end_date="2025-05-31", seed=7,
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

    @staticmethod
    def _quarter_ends_in_sample() -> pd.DatetimeIndex:
        """Last exchange session of every quarter that actually ends in-sample."""
        from quant_platform.utils.exchange_calendar import trading_days

        all_dates = trading_days(
            TestReportsPublishedOnNonTradingDays.START,
            TestReportsPublishedOnNonTradingDays.END,
        )
        quarter_of = all_dates.to_period("Q")
        last_sessions = all_dates[~quarter_of.duplicated(keep="last")]
        period_end = quarter_of[~quarter_of.duplicated(keep="last")].to_timestamp(
            how="end"
        ).normalize()
        return last_sessions[period_end <= pd.Timestamp(
            TestReportsPublishedOnNonTradingDays.END
        )]

    def test_every_generated_report_is_in_the_panel(self):
        provider = self._provider()
        quarter_ends = self._quarter_ends_in_sample()
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

        # Every report is visible on the first exchange session on/after
        # publication -- not merely the first business day.
        from quant_platform.utils.exchange_calendar import trading_days

        sessions = trading_days(self.START, self.END)
        expected_dates = sessions[
            sessions.searchsorted(publish.to_numpy(), side="left").clip(
                max=len(sessions) - 1
            )
        ]
        assert (pd.to_datetime(reports["date"]).to_numpy() == expected_dates.to_numpy()).all()

    def test_holiday_published_reports_map_to_the_next_exchange_session(self):
        """P7: 2024-02-12 is a Spring Festival closure, not a trading day.

        The report must become visible on the next *exchange* session
        (2024-02-19), not on the holiday itself and not merely on the next
        business day.
        """
        from quant_platform.utils.exchange_calendar import (
            EXCHANGE_HOLIDAYS,
            trading_days,
        )

        provider = self._provider()
        reports = self._reports(provider)
        visible = pd.to_datetime(reports["date"])
        published = pd.to_datetime(reports["publish_date"])

        on_holiday = published.dt.strftime("%Y-%m-%d").isin(EXCHANGE_HOLIDAYS)
        assert on_holiday.any(), "seed must publish at least one report on a holiday"
        assert (visible >= published).all()
        assert (visible[on_holiday] > published[on_holiday]).all(), (
            "a holiday-published report is visible on the holiday itself"
        )

        # Explicit date check for the Spring Festival publication.
        spring = published.dt.strftime("%Y-%m-%d") == "2024-02-12"
        assert spring.any()
        assert (visible[spring] == pd.Timestamp("2024-02-19")).all()

        # No report is dated on an exchange holiday anywhere in the panel.
        sessions = set(trading_days(self.START, self.END))
        assert visible.isin(sessions).all()

    def test_no_report_is_visible_before_it_was_published(self):
        provider = self._provider()
        reports = self._reports(provider)

        visible = pd.to_datetime(reports["date"])
        published = pd.to_datetime(reports["publish_date"])
        assert (visible >= published).all()


class TestExchangeCalendar:
    """P7: the synthetic grid must be exchange sessions, not business days."""

    def test_no_exchange_holiday_is_a_trading_day(self):
        from quant_platform.utils.exchange_calendar import EXCHANGE_HOLIDAYS

        provider = SyntheticDataProvider(
            n_stocks=20, start_date="2024-01-01", end_date="2024-12-31", seed=5,
        )
        provider._generate_all()

        dates = pd.DatetimeIndex(provider._dates)
        holidays = {d for d in dates if d.strftime("%Y-%m-%d") in EXCHANGE_HOLIDAYS}
        assert not holidays, f"exchange holidays on the synthetic grid: {sorted(holidays)}"

        prices = provider.get_prices("2024-01-01", "2024-12-31")
        price_dates = prices.index.get_level_values("date").unique()
        assert not any(d.strftime("%Y-%m-%d") in EXCHANGE_HOLIDAYS for d in price_dates)

        benchmark = provider.get_benchmark("2024-01-01", "2024-12-31")
        assert not any(
            d.strftime("%Y-%m-%d") in EXCHANGE_HOLIDAYS for d in benchmark.index
        )
        assert list(benchmark.index) == list(dates)

        # A known closure is absent; the neighbouring sessions are present.
        assert pd.Timestamp("2024-10-01") not in dates
        assert pd.Timestamp("2024-09-30") in dates
        assert pd.Timestamp("2024-10-08") in dates


class TestShortSampleGeneration:
    """P9: short research windows used to crash on the ST/late-lister draws."""

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            ("2024-01-01", "2024-03-31"),
            ("2024-01-01", "2024-04-30"),
            ("2024-06-01", "2024-06-30"),
        ],
    )
    def test_short_sample_generates(self, start, end):
        # 300 names so the 3% ST draw is virtually guaranteed to fire.
        provider = SyntheticDataProvider(n_stocks=300, start_date=start, end_date=end)
        provider._generate_all()

        assert len(provider._dates) > 0
        prices = provider.get_prices(start, end)
        assert not prices.empty
        assert prices["close"].notna().any()
        assert provider.get_benchmark(start, end).notna().any()

        financials = provider.get_financials(start, end)
        assert not financials.empty
        assert "market_cap" in financials.columns

        st = provider.get_st_timeseries()
        assert isinstance(st, pd.DataFrame)

    def test_two_week_sample_generates(self):
        """A ten-session window has no room for an ST episode but must not
        crash the trigger draw."""
        provider = SyntheticDataProvider(
            n_stocks=300, start_date="2024-03-01", end_date="2024-03-14",
        )
        provider._generate_all()
        assert len(provider._dates) > 0
        assert not provider.get_prices("2024-03-01", "2024-03-14").empty


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
