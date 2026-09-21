"""Tests for ML-based alpha signal generation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.alpha.ml_signal import (
    HAS_LGB,
    HAS_XGB,
    MLSignalConfig,
    MLSignalGenerator,
    ModelPerformance,
    TimeSeriesCV,
)


@pytest.fixture
def sample_factors():
    """Create sample factor data for testing.

    `n_dates` has to satisfy the configured training window in *days*, not in
    rows. The default `train_window` is 504 (two years), plus the purge gap and
    the test window, so a 300-day panel cannot support it -- the previous 300
    only "worked" because the splitter counted flattened (date, asset) rows and
    so trained on about ten days of data while believing it had 504.
    """
    np.random.seed(42)
    n_dates, n_assets = 800, 50
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    assets = [f"stock_{i:03d}" for i in range(n_assets)]

    factors = {}
    for name in ["momentum", "volatility", "value", "quality", "growth"]:
        data = np.random.randn(n_dates, n_assets)
        factors[name] = pd.DataFrame(data, index=dates, columns=assets)
    return factors


@pytest.fixture
def sample_forward_returns(sample_factors):
    """Create sample forward returns."""
    np.random.seed(123)
    first = list(sample_factors.values())[0]
    n_dates, n_assets = first.shape
    dates = first.index
    assets = first.columns
    data = np.random.randn(n_dates, n_assets) * 0.02
    return pd.DataFrame(data, index=dates, columns=assets)


class TestTimeSeriesCV:
    def test_split_basic(self):
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=10)
        splits = list(cv.split(300))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) >= 100
            assert len(test_idx) >= 30

    def test_no_leakage(self):
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=10)
        for train_idx, test_idx in cv.split(300):
            assert max(train_idx) < min(test_idx) - 10 + 1

    def test_expanding_mode(self):
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=10, mode="expanding")
        splits = list(cv.split(300))
        # In expanding mode, all splits start from 0
        for train_idx, _ in splits:
            assert train_idx[0] == 0

    def test_rolling_mode(self):
        cv = TimeSeriesCV(n_splits=3, train_size=100, test_size=30, gap=10, mode="rolling")
        splits = list(cv.split(300))
        # In rolling mode, train_start may shift
        assert len(splits) >= 2

    def test_insufficient_data(self):
        cv = TimeSeriesCV(n_splits=5, train_size=200, test_size=50, gap=20)
        with pytest.raises(ValueError):
            list(cv.split(100))


class TestMLSignalConfig:
    def test_default_config(self):
        cfg = MLSignalConfig()
        assert cfg.model_type == "lightgbm"
        assert cfg.train_window == 504
        assert cfg.retrain_frequency == 63
        assert cfg.forward_horizon == 21

    def test_custom_config(self):
        cfg = MLSignalConfig(model_type="xgboost", train_window=252)
        assert cfg.model_type == "xgboost"
        assert cfg.train_window == 252


class TestMLSignalGenerator:
    def test_prepare_features(self, sample_factors):
        gen = MLSignalGenerator()
        X, names = gen._prepare_features(sample_factors, date_idx=0)
        assert X.shape == (50, 5)
        assert len(names) == 5

    def test_prepare_dataset(self, sample_factors, sample_forward_returns):
        gen = MLSignalGenerator()
        X, y, names, row_dates = gen._prepare_dataset(
            sample_factors, sample_forward_returns, 0, 200
        )
        assert len(X) > 0
        assert len(y) == len(X)
        assert len(names) == 5
        # One date per row, so a splitter can divide on the date axis.
        assert len(row_dates) == len(X)

    @pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
    def test_train_lightgbm(self, sample_factors, sample_forward_returns):
        cfg = MLSignalConfig(model_type="lightgbm", n_splits=2)
        gen = MLSignalGenerator(config=cfg)
        perf = gen.train(sample_factors, sample_forward_returns)
        assert isinstance(perf, ModelPerformance)
        assert perf.model_type == "lightgbm"
        assert gen.model is not None

    @pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
    def test_train_xgboost(self, sample_factors, sample_forward_returns):
        cfg = MLSignalConfig(model_type="xgboost", n_splits=2)
        gen = MLSignalGenerator(config=cfg)
        perf = gen.train(sample_factors, sample_forward_returns)
        assert isinstance(perf, ModelPerformance)
        assert perf.model_type == "xgboost"

    @pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
    def test_predict(self, sample_factors, sample_forward_returns):
        cfg = MLSignalConfig(model_type="lightgbm", n_splits=2)
        gen = MLSignalGenerator(config=cfg)
        gen.train(sample_factors, sample_forward_returns)
        signal = gen.predict(sample_factors)
        assert isinstance(signal, pd.DataFrame)
        # Derive from the fixture rather than hardcoding, so growing the panel
        # does not silently invalidate this assertion.
        assert signal.shape[0] == sample_factors["momentum"].shape[0]
        assert signal.shape[1] == sample_factors["momentum"].shape[1]
        # Cross-sectional rank should be in [-0.5, 0.5]
        for col in signal.columns:
            valid = signal[col].dropna()
            if len(valid) > 0:
                assert valid.max() <= 0.5
                assert valid.min() >= -0.5

    @pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
    def test_generate(self, sample_factors, sample_forward_returns):
        cfg = MLSignalConfig(model_type="lightgbm", train_window=100, retrain_frequency=50)
        gen = MLSignalGenerator(config=cfg)
        signal = gen.generate(sample_factors, sample_forward_returns)
        assert isinstance(signal, pd.DataFrame)
        assert signal.shape[0] > 0
        assert not signal.isna().all().all()

    @pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
    def test_feature_importance(self, sample_factors, sample_forward_returns):
        cfg = MLSignalConfig(model_type="lightgbm", n_splits=2)
        gen = MLSignalGenerator(config=cfg)
        gen.train(sample_factors, sample_forward_returns)
        assert len(gen.feature_importance) > 0

    def test_predict_without_train_raises(self, sample_factors):
        gen = MLSignalGenerator()
        with pytest.raises(RuntimeError):
            gen.predict(sample_factors)

    @pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
    def test_save_load_model(self, sample_factors, sample_forward_returns, tmp_path):
        cfg = MLSignalConfig(model_type="lightgbm", n_splits=2)
        gen = MLSignalGenerator(config=cfg)
        gen.train(sample_factors, sample_forward_returns)

        model_path = str(tmp_path / "test_model.lgb")
        gen.save_model(model_path)

        # Verify metadata JSON was created
        meta_path = tmp_path / "test_model.json"
        assert meta_path.exists()

        # Load into new generator (model file may or may not exist depending on LGB version)
        gen2 = MLSignalGenerator(config=cfg)
        model_file = tmp_path / "test_model.lgb"
        if model_file.exists():
            gen2.load_model(model_path)
            assert gen2.model is not None
        else:
            # LightGBM sklearn wrapper may not save via save_model — verify metadata instead
            import json
            meta = json.loads(meta_path.read_text())
            assert meta["model_type"] == "lightgbm"
            assert "feature_importance" in meta


class TestModelPerformance:
    def test_dataclass(self):
        perf = ModelPerformance(
            date="2024-01-01",
            model_type="lightgbm",
            train_ic=0.05,
            test_ic=0.03,
            test_icir=1.5,
        )
        assert perf.date == "2024-01-01"
        assert perf.test_ic == 0.03
        assert perf.test_icir == 1.5


class TestTimeSeriesCVDates:
    """Regression for BUG-23.

    `split()` counts rows, but `_prepare_dataset` returns one row per
    (date, asset) pair. Counting rows made `train_size=504` about ten trading
    days rather than two years, and a ten-row purge gap less than a single
    cross-section -- so train and test shared dates and the labels of those
    dates leaked across the split.
    """

    def test_train_and_test_are_disjoint_on_dates(
        self, sample_factors, sample_forward_returns
    ):
        gen = MLSignalGenerator()
        n_dates = len(sample_factors["momentum"].index)
        _, _, _, row_dates = gen._prepare_dataset(
            sample_factors, sample_forward_returns, 0, n_dates
        )
        cv = TimeSeriesCV(n_splits=2, train_size=252, test_size=63, gap=10)

        folds = list(cv.split_by_dates(row_dates))
        assert folds, "expected at least one fold"

        for train_idx, test_idx in folds:
            train_dates = set(row_dates[train_idx])
            test_dates = set(row_dates[test_idx])
            overlap = train_dates & test_dates
            assert not overlap, f"train and test share {len(overlap)} dates"
            assert max(train_dates) < min(test_dates), "train must precede test"

    def test_train_size_is_counted_in_days(self, sample_factors, sample_forward_returns):
        """504 must mean 504 trading days, not 504 flattened rows."""
        gen = MLSignalGenerator()
        n_dates = len(sample_factors["momentum"].index)
        _, _, _, row_dates = gen._prepare_dataset(
            sample_factors, sample_forward_returns, 0, n_dates
        )
        cv = TimeSeriesCV(n_splits=1, train_size=252, test_size=63, gap=10)
        train_idx, _ = next(iter(cv.split_by_dates(row_dates)))
        assert len(set(row_dates[train_idx])) == 252

    @pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
    def test_train_splits_on_dates_not_rows(
        self, sample_factors, sample_forward_returns, monkeypatch
    ):
        """The guard the two tests above cannot provide.

        They exercise `split_by_dates` directly, so they would still pass if
        `train()` went back to calling `split(len(X_full))`. This asserts which
        one the caller actually uses.
        """
        import quant_platform.alpha.ml_signal as ml

        calls = {"by_dates": 0, "by_rows": 0}
        orig_by_dates = ml.TimeSeriesCV.split_by_dates
        orig_by_rows = ml.TimeSeriesCV.split

        def spy_by_dates(self, row_dates):
            calls["by_dates"] += 1
            return orig_by_dates(self, row_dates)

        def spy_by_rows(self, n_samples):
            calls["by_rows"] += 1
            return orig_by_rows(self, n_samples)

        monkeypatch.setattr(ml.TimeSeriesCV, "split_by_dates", spy_by_dates)
        monkeypatch.setattr(ml.TimeSeriesCV, "split", spy_by_rows)

        cfg = MLSignalConfig(model_type="lightgbm", n_splits=2)
        MLSignalGenerator(config=cfg).train(sample_factors, sample_forward_returns)

        assert calls["by_dates"] > 0, "train() did not split on dates"
        assert calls["by_rows"] == 0, "train() fell back to counting rows"


class TestPredictReturnsOnlyOutOfSampleSignals:
    """`train` fits the final model on the whole dataset, labels included.

    `predict` then emitted a signal for *every* date in `factors` -- so each
    one was an in-sample prediction whose realized forward return the model
    had been fitted on. The walk-forward metrics `train` reports are honest
    out-of-sample numbers; the signal a backtest consumed was not.
    """

    def test_the_model_records_where_training_ended(self, sample_factors, sample_forward_returns):
        gen = MLSignalGenerator()
        gen.train(sample_factors, sample_forward_returns)

        assert gen._train_end_date is not None

    def test_no_signal_is_emitted_inside_the_training_window(
        self, sample_factors, sample_forward_returns,
    ):
        gen = MLSignalGenerator()
        gen.train(sample_factors, sample_forward_returns)

        signal = gen.predict(sample_factors)

        train_end = gen._train_end_date
        in_sample = signal.loc[signal.index <= train_end]
        assert in_sample.isna().to_numpy().all(), (
            "an in-sample date came back with a signal"
        )

    def test_dates_after_training_still_get_signals(self, sample_factors):
        """The real pipeline's last row is unobservable (`shift(-1)`), so the
        final fit stops one day short and that day is a genuine out-of-sample
        signal -- which is what a live signal is."""
        factors = sample_factors
        first = factors[list(factors)[0]]
        forward_returns = pd.DataFrame(
            np.random.randn(*first.shape) * 0.01,
            index=first.index, columns=first.columns,
        )
        forward_returns.iloc[-1] = np.nan  # the pipeline's shift(-1) tail

        gen = MLSignalGenerator()
        gen.train(factors, forward_returns)
        signal = gen.predict(factors)

        after = signal.loc[signal.index > gen._train_end_date]
        assert len(after) == 1, "expected exactly the final, unobservable day"
        assert after.notna().to_numpy().any(), "no out-of-sample signal was produced"

    def test_training_through_the_last_date_yields_no_signal_at_all(
        self, sample_factors, sample_forward_returns,
    ):
        """The honest answer when there is nothing left to predict."""
        gen = MLSignalGenerator()
        gen.train(sample_factors, sample_forward_returns)
        # Pretend the fit consumed every date.
        gen._train_end_date = sample_factors[list(sample_factors)[0]].index.max()

        signal = gen.predict(sample_factors)

        assert signal.isna().to_numpy().all()

    def test_predicting_before_training_still_raises(self, sample_factors):
        gen = MLSignalGenerator()

        with pytest.raises(RuntimeError):
            gen.predict(sample_factors)
