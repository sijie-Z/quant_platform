"""Benchmark convention alignment in ``DataPipeline._align``.

``DataPipeline.returns[t]`` spans t -> t+1 (forward), while every external
provider's benchmark is a trailing t-day return (``P_t / P_{t-1} - 1``).
Reindexing by date alone lined the strategy's t -> t+1 return up against the
benchmark's t-1 -> t return: an index-tracking strategy showed a tracking
error of roughly sqrt(2) times the index volatility (the audit measured
~22%) instead of ~0.

The fix shifts provider benchmarks by one day in ``_align``.  Benchmarks
already in forward convention -- the engine's internal equal-weight
benchmark, or a provider that says so -- must not be shifted again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quant_platform.backtest.metrics import information_ratio, tracking_error
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider

ASSET = "000001.SH"
N_DAYS = 120


class _BenchmarkProvider(DataProvider):
    """One asset; its close path defines both forward returns and benchmark.

    ``convention="trailing"`` returns the provider-style benchmark
    ``close.pct_change()``; ``convention="forward"`` returns the same series
    already in the pipeline's t -> t+1 convention.
    """

    def __init__(self, close: pd.Series, convention: str = "trailing"):
        self._close = close
        self._dates = pd.DatetimeIndex(close.index, name="date")
        self._assets = [ASSET]
        self._convention = convention
        self.benchmark_is_trailing = convention == "trailing"

    def get_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sector": ["Tech"],
                "market_cap_group": ["mid"],
                "is_st": [False],
                "listing_date": [pd.Timestamp("2000-01-01")],
                "delisting_date": [pd.NaT],
            },
            index=pd.Index(self._assets, name="asset"),
        )

    def get_prices(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [self._dates, self._assets], names=["date", "asset"]
        )
        close = self._close.to_numpy()
        return pd.DataFrame(
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

    def get_financials(self, start_date, end_date) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [self._dates, self._assets], names=["date", "asset"]
        )
        return pd.DataFrame(
            {"market_cap": 1e10, "pb": 1.5, "pe": 15.0}, index=index
        )

    def get_benchmark(self, start_date, end_date) -> pd.Series:
        trailing = self._close.pct_change(fill_method=None)
        if self._convention == "forward":
            return trailing.shift(-1)
        return trailing


def _close() -> pd.Series:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=N_DAYS)
    daily = rng.normal(0.0003, 0.012, N_DAYS)
    return pd.Series(100.0 * np.cumprod(1.0 + daily), index=dates)


def _run(close: pd.Series, convention: str = "trailing", **kwargs) -> DataPipeline:
    pipeline = DataPipeline(
        provider=_BenchmarkProvider(close, convention),
        start_date=str(close.index[0].date()),
        end_date=str(close.index[-1].date()),
        **kwargs,
    )
    pipeline.run()
    return pipeline


class TestTrailingBenchmarkAlignment:
    def test_tracking_error_is_zero_when_strategy_equals_the_index(self):
        """The strategy here IS the single asset; its forward returns are the
        benchmark's trailing series shifted one day.  If _align does not
        convert conventions, TE is ~sqrt(2) * index vol (~22% annualised)."""
        close = _close()
        pipeline = _run(close)

        assert pipeline.benchmark_is_trailing is True
        strategy = pipeline.returns[ASSET]
        benchmark = pipeline.benchmark

        # The naive alignment (reindex only) is what used to happen and is
        # what makes this a regression test rather than a property test.
        naive = close.pct_change(fill_method=None).reindex(strategy.index)
        assert tracking_error(strategy, naive) > 0.1

        np.testing.assert_allclose(
            benchmark.dropna().to_numpy(),
            strategy.dropna().to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        assert tracking_error(strategy, benchmark) < 1e-8
        assert np.isfinite(information_ratio(strategy, benchmark))

    def test_first_date_is_populated_and_last_date_is_nan(self):
        """After shift(-1) the last price date has no forward benchmark."""
        pipeline = _run(_close())
        assert np.isfinite(pipeline.benchmark.iloc[0])
        assert np.isnan(pipeline.benchmark.iloc[-1])


class TestForwardBenchmarkIsNotShiftedAgain:
    def test_provider_declaring_forward_convention_is_left_alone(self):
        close = _close()
        pipeline = _run(close, convention="forward")

        assert pipeline.benchmark_is_trailing is False
        expected = close.pct_change(fill_method=None).shift(-1).reindex(
            pipeline.benchmark.index
        )
        pd.testing.assert_series_equal(
            pipeline.benchmark, expected, check_names=False
        )
        # Not shifted a second time: the first date is populated.
        assert np.isfinite(pipeline.benchmark.iloc[0])

    def test_constructor_override_wins_over_provider_default(self):
        """An internally synthesised (already forward) benchmark can opt out
        even when the provider does not advertise a convention."""
        close = _close()
        pipeline = _run(close, benchmark_is_trailing=False)

        assert pipeline.benchmark_is_trailing is False
        trailing = close.pct_change(fill_method=None).reindex(
            pipeline.benchmark.index
        )
        pd.testing.assert_series_equal(
            pipeline.benchmark, trailing, check_names=False
        )
