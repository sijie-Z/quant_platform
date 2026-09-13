"""Tests for backtest.walkforward — Walk-forward validation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.backtest.walkforward import WalkForwardValidator


class TestWalkForwardValidator:
    def test_basic_structure(self):
        """Test that WalkForwardValidator can be instantiated."""
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63, mode="rolling",
        )
        assert validator.train_period == 252
        assert validator.test_period == 63
        assert validator.mode == "rolling"

    def test_expanding_mode(self):
        validator = WalkForwardValidator(mode="expanding")
        assert validator.mode == "expanding"

    def test_insufficient_data_raises(self):
        short_signal = pd.DataFrame(np.random.randn(100, 10))
        short_prices = pd.DataFrame(np.random.randn(100, 10))
        short_returns = pd.DataFrame(np.random.randn(100, 10))
        benchmark = pd.Series(np.random.randn(100))
        sector_map = pd.Series(["A"] * 10)

        validator = WalkForwardValidator(train_period=504, test_period=126)
        with pytest.raises(ValueError, match="Not enough data"):
            validator.run(short_signal, short_prices, short_returns, benchmark, sector_map)


def _panel(n_dates=900, n_assets=30, seed=0):
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"S{i}" for i in range(n_assets)]
    ret = pd.DataFrame(np.random.randn(n_dates, n_assets) * 0.015, index=dates, columns=assets)
    prices = (1 + ret).cumprod() * 100
    factors = {"f": pd.DataFrame(np.random.randn(n_dates, n_assets), index=dates, columns=assets)}
    signal = pd.DataFrame(0.0, index=dates, columns=assets)
    return signal, prices, ret, ret.mean(axis=1), pd.Series("Tech", index=assets), factors


class TestOOSWindow:
    """Regression for BUG-21.

    The engine is handed extra history so its first rebalance has a warm-up,
    and `result["daily_returns"]` covers that whole range. Reporting it whole
    meant each fold's "OOS" metrics spanned train + test: a 63-day test window
    was reported over 315 days, so the headline OOS numbers were mostly
    in-sample performance.

    Note this is a *reporting window* defect, not signal leakage -- the fold
    signal is recomputed from train-only ICs and that part is correct.
    """

    def test_each_fold_reports_only_its_test_window(self):
        signal, prices, ret, bench, sector_map, factors = _panel()
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63, mode="rolling"
        )
        result = validator.run(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map, factors=factors,
        )

        folds = result["fold_metrics"]
        assert folds, "expected folds"

        for fm in folds:
            assert fm["oos_days"] <= 63, (
                f"fold {fm['fold']} reported {fm['oos_days']} OOS days for a "
                f"63-day test window"
            )

    def test_concatenated_oos_is_not_the_whole_sample(self):
        """Every fold's OOS used to span train+test, so the concatenated series
        covered nearly the entire sample instead of the union of the test
        windows."""
        signal, prices, ret, bench, sector_map, factors = _panel()
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63, mode="rolling"
        )
        result = validator.run(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map, factors=factors,
        )

        oos = result["oos_returns"]
        n_folds = len(result["fold_metrics"])
        n_dates = len(prices)

        # Test windows tile forward without overlapping, so the union is about
        # n_folds * test_period. Before the fix each fold returned train+test,
        # which would put this near the full sample length.
        assert len(oos) <= n_folds * 63 + 63, (
            f"concatenated OOS covers {len(oos)} days for {n_folds} folds of 63 "
            f"days -- it is spanning the training periods again"
        )
        assert len(oos) < n_dates, "OOS covers the whole sample"

        # And it must start at the first fold's test window, not at the start
        # of that fold's warm-up.
        first_test_start = pd.Timestamp(result["fold_metrics"][0]["test_start"])
        assert pd.Timestamp(oos.index[0]) >= first_test_start
