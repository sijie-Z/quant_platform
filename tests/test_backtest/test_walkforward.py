"""Tests for backtest.walkforward — Walk-forward validation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.alpha.combination import combine_equal_weight
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.walkforward import WalkForwardValidator
from quant_platform.factors.evaluation import rank_ic
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.optimizers import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
)


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


def _regime_panel(n_dates=450, n_assets=15, switch=100, seed=11):
    """Factor A predicts returns only before ``switch``; factor B after it.

    Everything else is cross-sectional noise, so a training-window selector
    must flip from A to B as the (expanding) train window grows past the
    regime boundary.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    assets = [f"S{i}" for i in range(n_assets)]
    base = rng.normal(size=(n_dates, n_assets))
    ret = pd.DataFrame(base * 0.01, index=dates, columns=assets)
    prices = (1 + ret).cumprod() * 100

    a_vals = rng.normal(size=(n_dates, n_assets))
    b_vals = rng.normal(size=(n_dates, n_assets))
    a_vals[:switch] = base[:switch] + rng.normal(scale=0.5, size=(switch, n_assets))
    b_vals[switch:] = base[switch:] + rng.normal(
        scale=0.5, size=(n_dates - switch, n_assets),
    )
    factors = {
        "A": pd.DataFrame(a_vals, index=dates, columns=assets),
        "B": pd.DataFrame(b_vals, index=dates, columns=assets),
    }
    signal = pd.DataFrame(0.0, index=dates, columns=assets)
    return signal, prices, ret, ret.mean(axis=1), pd.Series("Tech", index=assets), factors


def _best_icir_selector(fold_info, factors, forward_returns):
    """Toy selector: the factor with the highest train-window |ICIR|."""
    positions = fold_info["train_positions"]
    scores = {}
    for name, panel in factors.items():
        ic = rank_ic(panel.iloc[positions], forward_returns.iloc[positions])
        ic = ic[np.isfinite(np.asarray(ic, dtype=float))]
        std = float(ic.std()) if len(ic) > 1 else 0.0
        scores[name] = abs(float(ic.mean()) / std) if std > 0 else 0.0
    return [max(sorted(scores), key=lambda name: scores[name])]


class TestFactorSelectorContract:
    """R4: per-fold factor selection is confined to the training window."""

    def test_selector_receives_only_train_positions_minus_embargo(self):
        signal, prices, ret, bench, sector_map, factors = _panel(
            n_dates=300, n_assets=12, seed=5,
        )
        factors = {"f1": factors["f"], "f2": factors["f"].copy()}
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20,
            mode="expanding", purge_gap=3, embargo=10,
        )
        folds = {f["fold"]: f for f in validator._generate_folds(prices.index)}
        seen = []

        def spy(fold_info, factor_panels, forward_returns):
            positions = np.array(fold_info["train_positions"])
            seen.append((fold_info["fold"], positions))
            # The label panel sliced by these positions must cover exactly
            # the training rows the selector is allowed to use.
            window = forward_returns.iloc[positions]
            assert len(window) == len(positions)
            # A hostile mutation must not leak into the signal computation:
            # the engine hands over a copy of fold_info/train_positions.
            fold_info["train_positions"][:] = -1
            return ["f1"]

        result = validator.run(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map, factors=factors,
            alpha_kwargs={"method": "equal_weight"},
            factor_selector=spy,
        )

        assert seen, "selector was never called"
        embargoed_somewhere = False
        for fold_id, positions in seen:
            meta = folds[fold_id]
            test = meta["test_slice"]
            assert not ((positions >= test.start) & (positions < test.stop)).any(), (
                f"fold {fold_id}: selector saw test-period positions"
            )
            for lo, hi in meta["embargo_ranges"]:
                embargoed_somewhere = True
                assert not ((positions >= lo) & (positions < hi)).any(), (
                    f"fold {fold_id}: embargo range ({lo}, {hi}) reached the selector"
                )
            assert set(positions.tolist()) == set(meta["train_positions"].tolist())
        assert embargoed_somewhere, "test layout must exercise an embargo range"

        # The hostile mutation of the handed-over metadata did not corrupt
        # the fold bookkeeping.
        for detail in result["fold_details"]:
            assert detail["train_days"] == folds[detail["fold"]]["train_days"]
        assert result["fold_metrics"], "no fold produced OOS metrics"

    def test_selection_flips_with_the_train_window_regime(self):
        signal, prices, ret, bench, sector_map, factors = _regime_panel()
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20, mode="expanding",
        )

        calls = {}
        real = validator._compute_signal_in_sample

        def spy(factors_used, *args, **kwargs):
            calls[len(calls)] = sorted(factors_used)
            return real(factors_used, *args, **kwargs)

        validator._compute_signal_in_sample = spy

        result = validator.run(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map, factors=factors,
            alpha_kwargs={"method": "equal_weight"},
            factor_selector=_best_icir_selector,
        )

        selected = {d["fold"]: d["selected_factors"] for d in result["fold_details"]}
        assert selected, "expected fold details"
        # Early train window: only A has predictive power; late train window:
        # B dominates. The selection must differ across folds.
        assert selected[0] == ["A"]
        later = [sel for fold, sel in selected.items() if fold > 0]
        assert ["B"] in later, f"selection never flipped to B: {selected}"
        assert selected[0] != ["B"]

        # The signal recomputation really consumed the per-fold selection:
        # calls are in fold order and match selected_factors exactly.
        order = [d["fold"] for d in result["fold_details"]]
        used = [calls[i] for i in range(len(order))]
        assert used == [sorted(selected[fold]) for fold in order]

        # And the fold metrics carry the same selection for audit.
        for metric in result["fold_metrics"]:
            assert metric["selected_factors"] == selected[metric["fold"]]

    def test_no_selector_keeps_the_historical_output_exactly(self):
        """Default ``factor_selector=None`` must behave bit for bit as before:
        identical OOS series, fold metrics and dict shape (no new keys)."""
        signal, prices, ret, bench, sector_map, factors = _panel(
            n_dates=500, n_assets=20, seed=7,
        )
        validator = WalkForwardValidator(
            train_period=200, test_period=50, step_size=50, mode="rolling",
        )
        kwargs = dict(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map, factors=factors,
            alpha_kwargs={"method": "ic_weighted", "lookback": 60},
        )

        default_run = validator.run(**kwargs)
        explicit_none = validator.run(**kwargs, factor_selector=None)

        pd.testing.assert_series_equal(
            default_run["oos_returns"], explicit_none["oos_returns"],
        )
        assert default_run["n_folds"] == explicit_none["n_folds"]
        assert default_run["fold_details"] == explicit_none["fold_details"]
        assert default_run["fold_metrics"] == explicit_none["fold_metrics"]
        assert default_run["stability"] == explicit_none["stability"]

        # The historical dict shape is preserved: no selected_factors keys.
        for detail in default_run["fold_details"]:
            assert "selected_factors" not in detail
        for metric in default_run["fold_metrics"]:
            assert "selected_factors" not in metric

        # Selecting every factor must reproduce the default run exactly --
        # the selector is a pure filter, not a different code path.
        all_factors = validator.run(
            **kwargs, factor_selector=lambda fold, panels, fwd: list(panels),
        )
        pd.testing.assert_series_equal(
            default_run["oos_returns"], all_factors["oos_returns"],
        )
        for base_m, sel_m in zip(
            default_run["fold_metrics"], all_factors["fold_metrics"], strict=True,
        ):
            stripped = {k: v for k, v in sel_m.items() if k != "selected_factors"}
            assert stripped == base_m
        for detail in all_factors["fold_details"]:
            assert detail["selected_factors"] == list(factors)

    def test_empty_selection_skips_the_fold_and_records_it(self):
        signal, prices, ret, bench, sector_map, factors = _panel(
            n_dates=300, n_assets=12, seed=9,
        )
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20, mode="rolling",
        )

        def selector(fold_info, panels, fwd):
            return [] if fold_info["fold"] == 1 else list(panels)

        result = validator.run(
            signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
            sector_map=sector_map, factors=factors,
            alpha_kwargs={"method": "equal_weight"},
            factor_selector=selector,
        )

        details = {d["fold"]: d for d in result["fold_details"]}
        assert details[1]["selected_factors"] == []
        assert details[1]["oos_days"] == 0
        assert "selected no factors" in details[1]["error"]
        # The skipped fold contributes no OOS metrics and no fabricated OOS
        # rows; the other folds still complete.
        assert 1 not in {m["fold"] for m in result["fold_metrics"]}
        fold_one = validator._generate_folds(prices.index)[1]
        test_dates = prices.index[fold_one["test_slice"]]
        assert not test_dates.isin(result["oos_returns"].index).any()
        assert result["n_folds"] == 10

    def test_selector_exceptions_abort_the_run(self):
        signal, prices, ret, bench, sector_map, factors = _panel(
            n_dates=300, n_assets=12, seed=13,
        )
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20,
        )

        def broken(fold_info, panels, fwd):
            raise RuntimeError("selector boom")

        with pytest.raises(RuntimeError, match="selector boom"):
            validator.run(
                signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
                sector_map=sector_map, factors=factors,
                alpha_kwargs={"method": "equal_weight"},
                factor_selector=broken,
            )

    def test_selector_requires_factors(self):
        signal, prices, ret, bench, sector_map, _ = _panel(
            n_dates=300, n_assets=12, seed=15,
        )
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20,
        )
        with pytest.raises(ValueError, match="factor_selector requires factors"):
            validator.run(
                signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
                sector_map=sector_map,
                factor_selector=lambda fold, panels, fwd: [],
            )

    def test_unknown_selected_factor_raises(self):
        signal, prices, ret, bench, sector_map, factors = _panel(
            n_dates=300, n_assets=12, seed=17,
        )
        validator = WalkForwardValidator(
            train_period=100, test_period=20, step_size=20,
        )
        with pytest.raises(ValueError, match="unknown factor"):
            validator.run(
                signal=signal, prices=prices, returns=ret, benchmark_returns=bench,
                sector_map=sector_map, factors=factors,
                alpha_kwargs={"method": "equal_weight"},
                factor_selector=lambda fold, panels, fwd: ["does_not_exist"],
            )


def _capture_engine_signals(monkeypatch):
    """Record the signal every ``BacktestEngine.run`` call receives."""
    captured = []
    real_run = BacktestEngine.run

    def spy(self, **kwargs):
        captured.append(kwargs["signal"].copy())
        return real_run(self, **kwargs)

    monkeypatch.setattr(BacktestEngine, "run", spy)
    return captured


def _rank_panel(n_dates=220, n_assets=10, seed=21):
    """Panel whose factor rows are permutations (known ranks, no ties)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    assets = [f"S{i}" for i in range(n_assets)]
    ret = pd.DataFrame(
        rng.normal(0.0, 0.012, (n_dates, n_assets)), index=dates, columns=assets,
    )
    prices = (1.0 + ret).cumprod() * 100.0
    values = np.vstack(
        [rng.permutation(n_assets).astype(float) for _ in range(n_dates)]
    )
    factor = pd.DataFrame(values, index=dates, columns=assets)
    return ret, prices, factor


class TestRankNormalizedSignal:
    """R20: ``rank_normalize_signal`` maps each fold's signal to [-0.5, 0.5].

    The optimizer documents a rank alpha in [-0.5, 0.5] (``ALPHA_ANNUAL_SCALE``
    maps that span to +/-2% annual excess return); feeding it a z-score
    composite silently changed the effective risk aversion. Ranking is
    row-wise, so only the current cross-section is used.
    """

    def test_rows_become_cross_sectional_ranks(self, monkeypatch):
        ret, prices, factor = _rank_panel()
        captured = _capture_engine_signals(monkeypatch)
        validator = WalkForwardValidator(
            train_period=100, test_period=40, step_size=40, n_folds=1,
        )

        validator.run(
            signal=factor, prices=prices, returns=ret,
            benchmark_returns=ret.mean(axis=1),
            sector_map=pd.Series("Tech", index=factor.columns),
            factors={"f": factor},
            alpha_kwargs={"method": "equal_weight"},
            rank_normalize_signal=True,
        )

        assert len(captured) == 1, "one fold must produce exactly one engine run"
        fold_signal = captured[0]
        raw = factor.iloc[:140]  # train [0, 100) + test [100, 140)
        assert fold_signal.index.equals(raw.index)

        values = fold_signal.to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert values.min() >= -0.5 - 1e-12
        assert values.max() <= 0.5 + 1e-12

        # The rank preserves the cross-sectional order of the raw signal.
        np.testing.assert_array_equal(
            np.argsort(values, axis=1), np.argsort(raw.to_numpy(), axis=1),
        )

        # Hand-check one row against the definition rank(pct=True) - 0.5.
        hand = pd.Series(raw.iloc[0].to_numpy()).rank(pct=True).to_numpy() - 0.5
        np.testing.assert_allclose(values[0], hand)

    def test_default_path_keeps_the_historical_signal_bit_for_bit(self, monkeypatch):
        ret, prices, factor = _rank_panel()
        captured = _capture_engine_signals(monkeypatch)
        validator = WalkForwardValidator(
            train_period=100, test_period=40, step_size=40, n_folds=1,
        )

        validator.run(
            signal=factor, prices=prices, returns=ret,
            benchmark_returns=ret.mean(axis=1),
            sector_map=pd.Series("Tech", index=factor.columns),
            factors={"f": factor},
            alpha_kwargs={"method": "equal_weight"},
            rank_normalize_signal=False,
        )

        expected = combine_equal_weight({"f": factor.iloc[:140]})
        assert len(captured) == 1
        pd.testing.assert_frame_equal(captured[0], expected)
        # And the option is off by default: omitting it is the same run.
        captured.clear()
        validator.run(
            signal=factor, prices=prices, returns=ret,
            benchmark_returns=ret.mean(axis=1),
            sector_map=pd.Series("Tech", index=factor.columns),
            factors={"f": factor},
            alpha_kwargs={"method": "equal_weight"},
        )
        pd.testing.assert_frame_equal(captured[0], expected)

    def test_rows_with_fewer_than_two_valid_values_are_left_alone(self):
        raw = pd.DataFrame(
            [[1.0, 2.0, np.nan],
             [5.0, np.nan, np.nan],
             [np.nan, np.nan, np.nan]],
            index=pd.bdate_range("2021-01-01", periods=3),
            columns=["a", "b", "c"],
        )
        out = WalkForwardValidator._rank_normalize_signal(raw)

        np.testing.assert_allclose(
            out.iloc[0].to_numpy(dtype=float), [0.0, 0.5, np.nan],
            equal_nan=True,
        )
        # A single valid value has no meaningful rank: untouched, NaN kept.
        np.testing.assert_allclose(
            out.iloc[1].to_numpy(dtype=float), [5.0, np.nan, np.nan],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            out.iloc[2].to_numpy(dtype=float), [np.nan, np.nan, np.nan],
            equal_nan=True,
        )


def _two_sector_panel(n_dates=260, n_assets=12, seed=31):
    """Six high-signal assets in S1, six zero-signal assets in S2."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    assets = [f"S{i}" for i in range(n_assets)]
    ret = pd.DataFrame(
        rng.normal(0.0, 0.012, (n_dates, n_assets)), index=dates, columns=assets,
    )
    prices = (1.0 + ret).cumprod() * 100.0
    values = np.zeros((n_dates, n_assets))
    values[:, :6] = 3.0
    factor = pd.DataFrame(values, index=dates, columns=assets)
    sectors = pd.Series(["S1"] * 6 + ["S2"] * 6, index=assets)
    return ret, prices, factor, sectors


class TestSectorCapThroughTheWalkForwardChain:
    """R20: the research chain must feed real sectors into the optimizer.

    ``MeanVarianceOptimizer`` deliberately skips sector constraints when the
    candidate set sits in a single sector (the constraint would be either
    redundant or infeasible). The real-data script passed
    ``sector_map = Series("A", index=columns)``, so ``max_sector_exposure``
    was silently skipped for every rebalance. This test drives the whole
    walk-forward chain: the control (single sector) shows the unchecked
    portfolio loading 60% into one sector; the real two-sector map must cap
    every sector at 30%.
    """

    def test_real_sector_map_caps_every_sector_at_30_percent(self, monkeypatch):
        ret, prices, factor, sectors = _two_sector_panel()
        target_weights = []
        real_mvo = MeanVarianceOptimizer.optimize
        real_ew = EqualWeightOptimizer.optimize

        def wrap(original):
            def spy(self, *args, **kwargs):
                out = original(self, *args, **kwargs)
                target_weights.append(out.copy())
                return out
            return spy

        monkeypatch.setattr(MeanVarianceOptimizer, "optimize", wrap(real_mvo))
        monkeypatch.setattr(EqualWeightOptimizer, "optimize", wrap(real_ew))
        constraints = PortfolioConstraints(
            long_only=True, max_weight=0.10, max_sector_exposure=0.30,
            max_turnover=0.30,
        )

        def run_with(sector_map):
            target_weights.clear()
            validator = WalkForwardValidator(
                train_period=100, test_period=40, step_size=40, n_folds=1,
            )
            result = validator.run(
                signal=factor, prices=prices, returns=ret,
                benchmark_returns=ret.mean(axis=1), sector_map=sector_map,
                constraints=constraints, factors={"f": factor},
                alpha_kwargs={"method": "equal_weight"},
                engine_kwargs={"initial_capital": 2_000_000},
            )
            # A fallback records the equal-weight call and the outer MVO call
            # with identical weights; collapse adjacent duplicates so each
            # rebalance contributes exactly one target.
            deduped = []
            for weights in target_weights:
                if not deduped or not deduped[-1].equals(weights):
                    deduped.append(weights)
            return result, deduped

        def sector_totals(weights):
            by_sector = sectors.reindex(weights.index).fillna("Unknown")
            return weights.groupby(by_sector).sum()

        # Control: the old single-sector map disables the cap, and the
        # optimizer really does put 60% into S1 (the test has teeth).
        _, control_targets = run_with(pd.Series("A", index=factor.columns))
        assert control_targets, "control run produced no optimizer targets"
        control_max = max(
            float(sector_totals(weights).max()) for weights in control_targets
        )
        assert control_max > 0.30 + 0.05, (
            "control with a single 'A' sector never exceeded the 30% cap; "
            "the assertions below would not prove anything"
        )

        # Real map: every optimizer target must respect the 30% sector cap.
        result, capped_targets = run_with(sectors)
        assert result["fold_metrics"], "no OOS metrics"
        assert capped_targets, "capped run produced no optimizer targets"
        for weights in capped_targets:
            totals = sector_totals(weights)
            assert float(totals.max()) <= 0.30 + 1e-6, (
                f"sector total {float(totals.max()):.4f} exceeds "
                f"max_sector_exposure=0.30 ({totals.to_dict()})"
            )
            assert float(weights.sum()) <= 1.0 + 1e-9

    def test_solver_cap_binds_when_the_problem_is_feasible(self):
        """With two sectors a 30% cap makes ``sum(w)=1`` infeasible, so the
        two-sector test above ends in the equal-weight fallback (which also
        respects the cap). Four sectors make the problem feasible: the MVO
        solver itself must pin the favoured sector at 30% -- proving the
        constraint reaches the solver, not just the fallback."""
        rng = np.random.default_rng(41)
        n_assets = 24
        names = [f"A{i}" for i in range(n_assets)]
        sectors = pd.Series(
            [f"SEC{i // 6}" for i in range(n_assets)], index=names,
        )
        cov = pd.DataFrame(
            np.diag(rng.uniform(1e-4, 3e-4, n_assets)),
            index=names, columns=names,
        )
        signal = pd.Series(-0.5, index=names)
        signal.iloc[:6] = 0.5  # SEC0 is strongly favoured

        optimizer = MeanVarianceOptimizer(PortfolioConstraints(
            long_only=True, max_weight=0.10, max_sector_exposure=0.30,
            max_turnover=1.0,
        ))
        weights = optimizer.optimize(signal, cov, sector_map=sectors)

        assert optimizer.fallback_count == 0, (
            f"expected a solved MVO, got fallback: {optimizer.last_fallback_reason}"
        )
        totals = weights.groupby(sectors.reindex(weights.index)).sum()
        assert float(totals.max()) <= 0.30 + 1e-6
        assert float(totals.loc["SEC0"]) >= 0.30 - 1e-4, (
            f"the cap did not bind: SEC0={float(totals.loc['SEC0']):.4f}"
        )
        assert abs(float(weights.sum()) - 1.0) <= 1e-6
