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

    def test_n_folds_caps_the_generated_folds(self):
        """`walkforward --folds 1` printed 1 but ran every fold the data allowed."""
        signal, prices, ret, bench, sector_map, _ = _panel()
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63, n_folds=1,
        )

        result = validator.run(
            signal=signal, prices=prices, returns=ret,
            benchmark_returns=bench, sector_map=sector_map,
            engine_kwargs={"initial_capital": 10_000_000},
        )

        assert result["n_folds"] == 1
        assert len(result["fold_metrics"]) == 1

    def test_n_folds_must_be_positive(self):
        with pytest.raises(ValueError, match="n_folds"):
            WalkForwardValidator(n_folds=0)


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


class TestPurgeAndEmbargo:
    """Purge gap and embargo isolation (label-overlap control).

    ``purge_gap`` delays each test window N trading days past the training
    window end; the gap is neither trained nor tested. ``embargo`` withholds
    the N trading days right after an earlier fold's test window from later
    training samples.
    """

    @staticmethod
    def _dates(n_dates=900):
        return pd.bdate_range("2020-01-01", periods=n_dates)

    def test_default_layout_matches_historical_arithmetic(self):
        """purge_gap=0 / embargo=0 must reproduce the pre-change fold layout."""
        dates = self._dates()
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63,
        )
        folds = validator._generate_folds(dates)

        assert len(folds) == 11
        first = folds[0]
        assert first["train_slice"] == slice(0, 252)
        assert first["test_slice"] == slice(252, 315)
        assert first["train_dates"][0] == pd.Timestamp("2020-01-01")
        assert first["train_dates"][1] == pd.Timestamp("2020-12-17")
        assert first["test_dates"][0] == pd.Timestamp("2020-12-18")
        assert first["test_dates"][1] == pd.Timestamp("2021-03-16")

        second = folds[1]
        assert second["train_slice"] == slice(63, 315)
        assert second["test_slice"] == slice(315, 378)
        assert second["train_dates"][0] == pd.Timestamp("2020-03-30")

        last = folds[-1]
        assert last["test_slice"] == slice(882, 900)
        assert last["test_dates"][0] == pd.Timestamp("2023-05-19")
        assert last["test_dates"][1] == pd.Timestamp("2023-06-13")

        for fold in folds:
            assert fold["train_days"] == 252
            assert fold["embargo_ranges"] == []
            assert list(fold["train_positions"]) == list(
                range(fold["train_slice"].start, fold["train_slice"].stop)
            )

    def test_purge_gap_delays_test_start_by_trading_days(self):
        dates = self._dates()
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63, purge_gap=10,
        )
        folds = validator._generate_folds(dates)

        assert folds, "expected folds"
        for fold in folds:
            train_stop = fold["train_slice"].stop
            assert fold["test_slice"].start == train_stop + 10
            # The gap is counted on the trading-day axis, not the calendar.
            assert fold["test_dates"][0] == dates[train_stop + 10]
            assert fold["train_dates"][1] == dates[train_stop - 1]

    def test_purge_gap_is_visible_in_fold_metrics(self):
        signal, prices, ret, bench, sector_map, _ = _panel(
            n_dates=400, n_assets=15, seed=1,
        )
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63,
            purge_gap=5, n_folds=1,
        )
        result = validator.run(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map,
        )

        dates = prices.index
        fold = result["fold_metrics"][0]
        assert fold["train_end"] == str(dates[251])
        assert fold["test_start"] == str(dates[257])  # 252 + 5 trading days
        assert fold["train_days"] == 252

    def test_embargo_removes_days_after_previous_test_window(self):
        dates = self._dates(300)
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20,
            mode="expanding", embargo=10,
        )
        folds = {f["fold"]: f for f in validator._generate_folds(dates)}

        # Fold 0 has no earlier test window: nothing is embargoed.
        assert folds[0]["train_days"] == 100
        assert folds[0]["embargo_ranges"] == []

        # Fold 2 trains on [0, 140); fold 0's test window ended at 120, so
        # [120, 130) -- the ten days right after it -- must be withheld.
        f2 = folds[2]
        assert f2["train_slice"] == slice(0, 140)
        assert f2["embargo_ranges"] == [(120, 130)]
        assert f2["train_days"] == 130
        pos = {int(p) for p in f2["train_positions"]}
        assert not pos.intersection(range(120, 130))
        assert 119 in pos and 130 in pos

        # Fold 3 also loses the days after fold 1's test window (ends at 140).
        f3 = folds[3]
        assert f3["embargo_ranges"] == [(120, 130), (140, 150)]
        assert f3["train_days"] == 140

        # Every embargo range must sit inside the training window and must
        # actually be absent from train_positions.
        for fold in folds.values():
            train = fold["train_slice"]
            pos = {int(p) for p in fold["train_positions"]}
            for lo, hi in fold["embargo_ranges"]:
                assert train.start <= lo < hi <= train.stop
                assert not pos.intersection(range(lo, hi))
            assert fold["train_days"] == train.stop - train.start - sum(
                hi - lo for lo, hi in fold["embargo_ranges"]
            )

    def test_embargo_applies_in_rolling_mode_too(self):
        dates = self._dates(300)
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20,
            mode="rolling", embargo=5,
        )
        folds = {f["fold"]: f for f in validator._generate_folds(dates)}

        # Fold 2 trains on [40, 140); fold 0's test window ended at 120.
        assert folds[2]["train_slice"] == slice(40, 140)
        assert folds[2]["embargo_ranges"] == [(120, 125)]
        assert folds[2]["train_days"] == 95
        assert folds[3]["embargo_ranges"] == [(120, 125), (140, 145)]
        assert folds[3]["train_days"] == 90

    def test_embargo_days_never_enter_the_ic_training_set(self, monkeypatch):
        """An extreme regime right after a prior test window must not feed
        those days into any later fold's IC estimation."""
        import quant_platform.alpha.combination as combination

        captured = []
        real = combination.combine_ic_weighted

        def spy(factors, forward_returns, lookback=252, *args, **kwargs):
            captured.append(pd.DatetimeIndex(forward_returns.index))
            return real(factors, forward_returns, lookback, *args, **kwargs)

        monkeypatch.setattr(combination, "combine_ic_weighted", spy)

        signal, prices, ret, bench, sector_map, factors = _panel(
            n_dates=300, n_assets=15, seed=3,
        )
        ret = ret.copy()
        ret.iloc[120:130] = 0.2  # extreme regime right after fold 0's test end
        prices = (1 + ret).cumprod() * 100
        bench = ret.mean(axis=1)
        embargo_dates = prices.index[120:130]

        def run(embargo):
            captured.clear()
            validator = WalkForwardValidator(
                train_period=100, test_period=20, step_size=20,
                mode="expanding", embargo=embargo,
            )
            result = validator.run(
                signal=signal, prices=prices, returns=ret,
                benchmark_returns=bench, sector_map=sector_map,
                factors=factors,
                alpha_kwargs={"method": "ic_weighted", "lookback": 20},
            )
            return [idx.copy() for idx in captured], result

        embargoed, result = run(10)
        assert embargoed, "IC-weighted signal was never fitted"
        for train_index in embargoed:
            assert not embargo_dates.isin(train_index).any(), (
                "embargoed days leaked into the IC training set"
            )
        # The exposed fold metadata carries the reduced training-day count.
        fold_two = {m["fold"]: m for m in result["fold_metrics"]}[2]
        assert fold_two["train_days"] == 130

        control, _ = run(0)
        assert any(embargo_dates.isin(idx).all() for idx in control), (
            "control run should include the post-test days -- otherwise the "
            "assertion above proves nothing"
        )

    @pytest.mark.parametrize("kwargs", [
        {"purge_gap": -1},
        {"purge_gap": -10},
        {"embargo": -1},
        {"embargo": -5},
    ])
    def test_negative_purge_or_embargo_raises(self, kwargs):
        with pytest.raises(ValueError, match="purge_gap|embargo"):
            WalkForwardValidator(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"purge_gap": 1.5},
        {"purge_gap": True},
        {"embargo": 2.0},
        {"embargo": True},
    ])
    def test_non_integer_purge_or_embargo_raises(self, kwargs):
        with pytest.raises(ValueError, match="purge_gap|embargo"):
            WalkForwardValidator(**kwargs)
