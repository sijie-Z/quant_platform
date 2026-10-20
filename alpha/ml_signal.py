"""ML-based alpha signal generation.

Replaces linear ICIR weighting with gradient boosting models (XGBoost/LightGBM).
Captures non-linear factor interactions and time-varying factor premiums.

Architecture:
    Factor Data → Feature Engineering → Walk-Forward Training → Signal Generation
                                                  ↓
                                          Model Registry (XGB/LGB/Ensemble)

Key design choices:
- Walk-forward validation (no look-ahead bias)
- Expanding window training (adapts to regime changes)
- Feature importance + SHAP for interpretability
- Ensemble with existing ICIR model for robustness
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# Documented minimums for an honest walk-forward model. Below either bound the
# generator raises InsufficientTrainingData instead of returning a zero-IC
# performance record (old `train()`) or an all-NaN signal (old `generate()`):
# both used to look like a successful run with no edge.
MIN_TRAIN_SAMPLES = 60   # flattened (date, asset) observations with a label
MIN_TRAIN_FEATURES = 2   # feature columns with >= 1 finite value in the window
MIN_TRAIN_DATES = 30     # labeled dates in the first walk-forward training fold


class InsufficientTrainingData(RuntimeError):  # noqa: N818 -- public API name
    """Raised when the sample cannot support a real walk-forward model.

    The message always reports the available sample count, the usable feature
    count and the minimums enforced by this module (``MIN_TRAIN_SAMPLES``,
    ``MIN_TRAIN_FEATURES``, ``MIN_TRAIN_DATES``). Callers (CLI, API) should
    surface this as an explicit failure and exit/return non-zero: the
    generator does not fabricate a zero-IC performance record or an all-NaN
    signal in its place.
    """


@dataclass
class MLSignalConfig:
    """Configuration for ML signal generation."""
    model_type: str = "lightgbm"       # "xgboost" / "lightgbm" / "ensemble"
    train_window: int = 504            # ~2 years of trading days
    retrain_frequency: int = 63        # Retrain every quarter
    forward_horizon: int = 21          # Predict 1-month forward return
    n_splits: int = 5                  # Time-series CV splits
    top_n_features: int = 15           # Max features to use
    purge_gap: int = 10                # Purge gap between train/test (trading days)
    embargo: int = 0                   # Embargo after test set (trading days)

    # XGBoost params
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 10,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
    })

    # LightGBM params
    lgb_params: dict = field(default_factory=lambda: {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "objective": "regression",
        "metric": "mse",
        "verbose": -1,
        "random_state": 42,
    })


@dataclass
class ModelPerformance:
    """Track model performance across time."""
    date: str
    model_type: str
    train_ic: float = 0.0
    test_ic: float = 0.0
    train_icir: float = 0.0
    test_icir: float = 0.0
    feature_importance: dict = field(default_factory=dict)
    n_train_samples: int = 0
    n_test_samples: int = 0


class TimeSeriesCV:
    """Time-series aware cross-validation with purge gap.

    Generates expanding/rolling train-test splits that respect temporal order.
    No future data leaks into training.

    The purge gap ensures no information leakage between train and test sets.
    The embargo period after the test set prevents the test set's labels from
    bleeding into future training sets (important when labels are computed
    from overlapping windows, e.g., 21-day forward returns).

    Split pattern (expanding):
    [----train----][purge][--test--][embargo]
    [--------train--------][purge][--test--][embargo]
    [------------train------------][purge][--test--][embargo]

    Args:
        n_splits: Number of CV folds.
        train_size: Minimum training set size (samples).
        test_size: Test set size (samples).
        gap: Purge gap between train and test (trading days).
             Removes any overlap from label computation windows.
        embargo: Embargo period after test set (trading days).
                 Prevents test labels from leaking into next train set.
        mode: 'expanding' (growing window) or 'rolling' (fixed window).
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 252,
        test_size: int = 63,
        gap: int = 10,
        embargo: int = 0,
        mode: str = "expanding",
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.embargo = embargo
        self.mode = mode

    def _folds(self, n_units: int):
        """Yield (train_start, train_end, test_start, test_end) in unit terms.

        Split out from `split()` so the same arithmetic can be applied to a
        date axis or to a row axis.

        Test windows are non-overlapping. The step used to be
        `(available - test_size) // (n_splits-1)` with no floor at
        `test_size`, so with the default settings on a 600-day panel each
        test window shared 58 of its 63 days with the next one and the
        fold-level mean/std were computed on overlapping samples. `embargo`
        also had no effect in expanding mode because nothing referenced it in
        the step. When the data cannot support `n_splits` windows, fewer
        folds are produced (the last may be shorter than `test_size`).
        """
        min_size = self.train_size + self.gap + self.test_size
        if n_units < min_size:
            raise ValueError(f"Need at least {min_size} units, got {n_units}")

        available = n_units - self.train_size - self.gap
        step = self.test_size + max(0, self.embargo)
        # Full non-overlapping windows; if even one does not fit, keep a
        # single shorter test window rather than refusing to validate.
        max_folds = available // step if available >= step else 1
        n_folds = min(self.n_splits, max_folds)

        for i in range(n_folds):
            if self.mode == "expanding":
                train_start = 0
            else:  # rolling: fixed-length training window
                train_start = i * step

            train_end = self.train_size + i * step
            test_start = train_end + self.gap  # purge gap
            test_end = min(test_start + self.test_size, n_units)

            if train_end >= n_units or test_start >= n_units:
                break

            yield train_start, min(train_end, n_units), test_start, test_end

    def split(self, n_samples: int):
        """Generate train/test index pairs with purge and embargo.

        Args:
            n_samples: Total number of samples.

        Yields:
            (train_indices, test_indices) tuples.
        """
        for train_start, train_end, test_start, test_end in self._folds(n_samples):
            train_idx = list(range(train_start, train_end))
            test_idx = list(range(test_start, test_end))
            if train_idx and test_idx:
                yield train_idx, test_idx

    def split_by_dates(self, row_dates):
        """Split on the date axis when each row is a (date, asset) pair.

        `split()` counts in rows. On a cross-sectional panel that makes
        `train_size=504` about twelve trading days rather than two years, and a
        ten-row purge gap removes less than a single cross-section -- so train
        and test end up sharing dates, and the labels of those dates leak
        across the split.

        This applies the identical fold arithmetic to the unique date axis and
        returns row indices, so callers do not change.

        Args:
            row_dates: sequence of dates, one per row, in row order.

        Yields:
            (train_row_indices, test_row_indices) tuples.
        """
        dates = pd.Index(pd.unique(pd.Index(row_dates)))
        rows_by_date: dict = {}
        for i, d in enumerate(row_dates):
            rows_by_date.setdefault(d, []).append(i)

        for train_start, train_end, test_start, test_end in self._folds(len(dates)):
            train_dates = dates[train_start:train_end]
            test_dates = dates[test_start:test_end]
            train_idx = [i for d in train_dates for i in rows_by_date.get(d, ())]
            test_idx = [i for d in test_dates for i in rows_by_date.get(d, ())]
            if train_idx and test_idx:
                yield train_idx, test_idx


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    gap: int = 10,
    test_size: int | None = None,
    embargo: int = 0,
    mode: str = "expanding",
) -> list[tuple[list[int], list[int]]]:
    """Generate purged walk-forward train/test splits.

    Convenience function that returns all splits as a list.
    The purge gap ensures no information leakage between train and test.
    The embargo prevents test labels from bleeding into future training.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of folds.
        gap: Purge gap between train and test (default 10 trading days).
        test_size: Test set size (default: n_samples // (n_splits + 1)).
        embargo: Embargo period after test set (default 0).
        mode: 'expanding' or 'rolling'.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    if test_size is None:
        test_size = max(21, n_samples // (n_splits + 1))

    train_size = max(63, n_samples - n_splits * (test_size + gap + embargo) - gap)

    cv = TimeSeriesCV(
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        gap=gap,
        embargo=embargo,
        mode=mode,
    )
    return list(cv.split(n_samples))


class MLSignalGenerator:
    """ML-based alpha signal generator.

    Trains gradient boosting models on factor data using walk-forward validation.
    Produces cross-sectional signals that can replace or augment ICIR weighting.

    Usage:
        gen = MLSignalGenerator(config=MLSignalConfig(model_type="lightgbm"))
        signal = gen.generate(factors, forward_returns)
    """

    def __init__(self, config: MLSignalConfig | None = None):
        self.config = config or MLSignalConfig()
        self.model = None
        self.performance_history: list[ModelPerformance] = []
        self.feature_importance: dict[str, float] = {}
        self._last_train_date = None
        self._train_end_date = None
        self._feature_names: list[str] | None = None

    def _create_model(self):
        """Create the underlying ML model."""
        if self.config.model_type == "xgboost":
            if not HAS_XGB:
                raise ImportError("xgboost not installed: pip install xgboost")
            return xgb.XGBRegressor(**self.config.xgb_params)
        elif self.config.model_type == "lightgbm":
            if not HAS_LGB:
                raise ImportError("lightgbm not installed: pip install lightgbm")
            return lgb.LGBMRegressor(**self.config.lgb_params)
        elif self.config.model_type == "ensemble":
            # There is no ensemble implementation: the factory returned None
            # and training then failed with AttributeError.
            raise ValueError(
                "model_type='ensemble' is not implemented; use 'xgboost' or "
                "'lightgbm'"
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

    def _prepare_features(
        self,
        factors: dict[str, pd.DataFrame],
        date_idx: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract factor values for a cross-section as feature matrix.

        Args:
            factors: dict of factor_name -> (date x asset) DataFrame
            date_idx: index into the date axis

        Returns:
            (n_assets, n_features) array, feature names list
        """
        feature_names = sorted(factors.keys())[:self.config.top_n_features]
        features = []
        for name in feature_names:
            df = factors[name]
            if date_idx < len(df):
                row = df.iloc[date_idx].values
            else:
                row = np.full(len(df.columns), np.nan)
            features.append(row)

        X = np.column_stack(features)
        return X, feature_names

    def _prepare_dataset(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        start_idx: int,
        end_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str], pd.Index]:
        """Prepare training dataset from factor cross-sections.

        Flattens all dates and assets into a single feature matrix.
        Each row = (date, asset) pair.

        Feature columns with no finite observation anywhere in
        ``[start_idx, end_idx)`` are dropped (and logged): keeping them would
        make the per-row ``valid_mask`` below reject every row, which is how a
        short history used to end up with zero samples and a fake zero-IC
        result. Fewer than ``MIN_TRAIN_FEATURES`` surviving columns is an
        explicit ``InsufficientTrainingData`` failure, never a silent
        "no data" result.

        Returns:
            (X, y, feature_names, row_dates) -- `row_dates` carries the date of
            each row, so a splitter can divide on the date axis instead of
            counting rows. Row counts per date vary (NaN assets are dropped),
            so this cannot be reconstructed by the caller.
        """
        candidate_names = sorted(factors.keys())[:self.config.top_n_features]
        if not candidate_names:
            raise InsufficientTrainingData(
                f"Insufficient training data: 0 samples and 0 usable features "
                f"available; at least {MIN_TRAIN_SAMPLES} samples and "
                f"{MIN_TRAIN_FEATURES} features are required."
            )
        first_factor = factors[candidate_names[0]]
        dates = first_factor.index[start_idx:end_idx]
        assets = first_factor.columns

        # Drop columns with no finite observation in this window. A single
        # all-NaN column used to poison the AND-ed valid_mask below and leave
        # zero training rows while the caller reported zeros instead of an
        # error.
        feature_names: list[str] = []
        dropped: list[str] = []
        for name in candidate_names:
            window = factors[name].iloc[start_idx:end_idx]
            if np.isfinite(window.to_numpy(dtype=float)).any():
                feature_names.append(name)
            else:
                dropped.append(name)
                logger.warning(
                    "Dropping feature '%s': no finite observations in the "
                    "training window (%d dates); keeping it would invalidate "
                    "every training row.",
                    name, len(window),
                )

        if len(feature_names) < MIN_TRAIN_FEATURES:
            n_labeled_rows = 0
            for date in dates:
                if date in forward_returns.index:
                    fwd = forward_returns.loc[date].to_numpy(dtype=float)
                    n_labeled_rows += int(np.isfinite(fwd).sum())
            raise InsufficientTrainingData(
                f"Insufficient training data: {n_labeled_rows} labeled samples "
                f"and {len(feature_names)} usable feature(s) after dropping "
                f"{dropped}; at least {MIN_TRAIN_SAMPLES} samples and "
                f"{MIN_TRAIN_FEATURES} features are required."
            )

        X_list = []
        y_list = []
        dates_list = []

        for i, date in enumerate(dates):
            actual_idx = start_idx + i
            if date not in forward_returns.index:
                continue

            fwd = forward_returns.loc[date].values

            features = []
            valid_mask = ~np.isnan(fwd)
            for name in feature_names:
                df = factors[name]
                if actual_idx < len(df) and date in df.index:
                    row = df.loc[date].values
                else:
                    row = np.full(len(assets), np.nan)
                features.append(row)
                valid_mask = valid_mask & ~np.isnan(row)

            X = np.column_stack(features)
            X_valid = X[valid_mask]
            y_valid = fwd[valid_mask]

            if len(X_valid) > 0:
                X_list.append(X_valid)
                y_list.append(y_valid)
                dates_list.extend([date] * len(X_valid))

        if not X_list:
            return np.array([]), np.array([]), feature_names, pd.Index([])

        return (
            np.vstack(X_list),
            np.concatenate(y_list),
            feature_names,
            pd.Index(dates_list),
        )

    @staticmethod
    def _require_min_training_data(
        n_samples: int, n_features: int, context: str
    ) -> None:
        """Raise InsufficientTrainingData below the documented minimums.

        The message always carries the available sample count, the usable
        feature count and the required minimums, so an operator can tell
        whether the history is too short, the factors too sparse, or both.
        """
        if n_samples < MIN_TRAIN_SAMPLES or n_features < MIN_TRAIN_FEATURES:
            raise InsufficientTrainingData(
                f"Insufficient training data for {context}: {n_samples} samples "
                f"and {n_features} features available; at least "
                f"{MIN_TRAIN_SAMPLES} samples and {MIN_TRAIN_FEATURES} features "
                f"are required."
            )

    def train(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> ModelPerformance:
        """Train the ML model with time-series cross-validation.

        The first walk-forward fold is adaptive. With a long history
        (enough labeled dates that the full ``train_window`` still leaves a
        63-date test window) it trains on exactly ``train_window`` dates,
        unchanged from before. With a shorter history it shrinks the first
        training window just enough to keep at least one out-of-sample fold,
        down to ``MIN_TRAIN_DATES``. A sample that cannot support that raises
        ``InsufficientTrainingData`` -- it never returns a zero-IC
        ``ModelPerformance`` for a model that was not fitted.

        Args:
            factors: dict of factor_name -> (date x asset) processed factor values
            forward_returns: (date x asset) forward returns

        Returns:
            ModelPerformance with CV results

        Raises:
            InsufficientTrainingData: when fewer than ``MIN_TRAIN_SAMPLES``
                usable rows or ``MIN_TRAIN_FEATURES`` usable features exist,
                or the labeled date axis cannot hold ``MIN_TRAIN_DATES``
                training dates plus the purge gap plus one test date.
        """
        # Clear any previously fitted state first: a failed training run must
        # not leave a model (or metadata) behind that `save_model` could
        # persist.
        self.model = None
        self._feature_names = None
        self._train_end_date = None
        self._last_train_date = None
        self.feature_importance = {}

        if not factors:
            self._require_min_training_data(0, 0, context="train()")

        first_factor = list(factors.values())[0]
        n_dates = len(first_factor.index)

        # Prepare full dataset for CV
        X_full, y_full, feature_names, row_dates = self._prepare_dataset(
            factors, forward_returns, 0, n_dates
        )
        self._require_min_training_data(
            len(X_full), len(feature_names), context="train()"
        )

        # The splitter divides the *labeled* date axis, so fold bounds must be
        # expressed in labeled dates (the last date often has no observable
        # forward return and is absent from `row_dates`).
        n_obs_dates = int(len(pd.Index(pd.unique(row_dates))))
        gap = self.config.purge_gap
        test_window = 63  # the walk-forward test window this CV always used
        max_train_end = n_obs_dates - gap - 1
        if max_train_end < MIN_TRAIN_DATES:
            raise InsufficientTrainingData(
                f"Insufficient training data for walk-forward CV: "
                f"{n_obs_dates} labeled dates, {len(X_full)} samples and "
                f"{len(feature_names)} features available; at least "
                f"{MIN_TRAIN_DATES} training dates + {gap}-date purge gap + "
                f"1 test date ({MIN_TRAIN_DATES + gap + 1} labeled dates), "
                f"{MIN_TRAIN_SAMPLES} samples and {MIN_TRAIN_FEATURES} features "
                f"are required."
            )

        # Adaptive first fold. min() keeps three invariants: never train on
        # more than `train_window` (long histories unchanged at 504),
        # never train on fewer than MIN_TRAIN_DATES, and always leave at
        # least one labeled test date after the purge gap.
        first_train_end = min(
            self.config.train_window,
            max(MIN_TRAIN_DATES, n_obs_dates - test_window),
            max_train_end,
        )
        test_size = min(test_window, n_obs_dates - gap - first_train_end)

        # Time-series CV with purge gap
        cv = TimeSeriesCV(
            n_splits=self.config.n_splits,
            train_size=first_train_end,
            test_size=test_size,
            gap=gap,
            embargo=self.config.embargo,
        )

        cv_results = []
        fold_test_ic_series: list[pd.Series] = []
        # Split on dates, not rows: X_full has one row per (date, asset), so a
        # row-based split would put the same trading day in both train and test.
        for fold, (train_idx, test_idx) in enumerate(cv.split_by_dates(row_dates)):
            X_train, y_train = X_full[train_idx], y_full[train_idx]
            X_test, y_test = X_full[test_idx], y_full[test_idx]

            model = self._create_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Cross-sectional (rank) IC per date. `np.corrcoef` on the
            # flattened panel mixed time-series and cross-sectional variation:
            # predictions that carried only market timing scored 0.7 pooled
            # correlation with zero stock-selection skill.
            train_ics = self._cross_sectional_ic(
                row_dates[train_idx], y_train, train_pred,
            )
            test_ics = self._cross_sectional_ic(
                row_dates[test_idx], y_test, test_pred,
            )
            train_ic = float(train_ics.mean()) if len(train_ics) else 0.0
            test_ic = float(test_ics.mean()) if len(test_ics) else 0.0
            fold_test_ic_series.append(test_ics)

            cv_results.append({
                "fold": fold,
                "train_ic": train_ic,
                "test_ic": test_ic,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

            logger.info("  Fold %d: train_IC=%.4f, test_IC=%.4f (n_train=%d, n_test=%d)",
                        fold, train_ic, test_ic, len(train_idx), len(test_idx))

        if not cv_results:
            # The adaptive split above should always yield a fold; if a
            # degenerate date axis slips through, fail explicitly instead of
            # averaging over an empty list.
            raise InsufficientTrainingData(
                f"Insufficient training data for walk-forward CV: no fold "
                f"produced from {n_obs_dates} labeled dates, {len(X_full)} "
                f"samples and {len(feature_names)} features "
                f"(first fold: train_size={first_train_end}, gap={gap}, "
                f"test_size={test_size}); at least {MIN_TRAIN_SAMPLES} samples "
                f"and {MIN_TRAIN_FEATURES} features are required."
            )

        # Train final model on all data. Fit into a local first so a failed
        # fit never leaves an unfitted estimator on `self.model` that
        # `save_model` could try to persist.
        final_model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_model.fit(X_full, y_full)
        self.model = final_model

        # Remember which dates the fitted model has seen, so that `predict`
        # does not hand back in-sample predictions as if they were signals.
        self._train_end_date = max(row_dates) if len(row_dates) else None
        self._feature_names = list(feature_names)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importances.tolist(), strict=False))

        # Aggregate CV metrics over the unique test dates, not over the 2-5
        # fold means: the latter was the dispersion of a handful of numbers,
        # not an ICIR.
        if fold_test_ic_series:
            all_test_ics = pd.concat(fold_test_ic_series)
            all_test_ics = all_test_ics[~all_test_ics.index.duplicated()]
        else:
            all_test_ics = pd.Series(dtype=float)
        mean_test_ic = float(all_test_ics.mean()) if len(all_test_ics) else 0.0
        std_test_ic = (
            float(all_test_ics.std(ddof=1)) if len(all_test_ics) > 1 else 0.0
        )
        test_icir = mean_test_ic / std_test_ic if std_test_ic > 0 else 0

        perf = ModelPerformance(
            date=str(first_factor.index[-1]),
            model_type=self.config.model_type,
            train_ic=float(np.mean([r["train_ic"] for r in cv_results])),
            test_ic=mean_test_ic,
            test_icir=test_icir,
            feature_importance=self.feature_importance,
            n_train_samples=len(X_full),
        )
        self.performance_history.append(perf)
        self._last_train_date = first_factor.index[-1]

        logger.info(
            "ML model trained: type=%s, test_IC=%.4f, test_ICIR=%.4f, "
            "features=%d, first_fold_train_dates=%d",
            self.config.model_type, mean_test_ic, test_icir,
            len(feature_names), first_train_end,
        )

        return perf

    @staticmethod
    def _cross_sectional_ic(dates, y_true, y_pred) -> pd.Series:
        """Rank IC per date from flattened (date, asset) rows.

        Dates with fewer than five assets, or with constant predictions or
        labels, are skipped: Spearman is undefined there.
        """
        frame = pd.DataFrame({
            "date": np.asarray(dates),
            "y": np.asarray(y_true, dtype=float),
            "p": np.asarray(y_pred, dtype=float),
        })
        ics: dict[object, float] = {}
        for date, group in frame.groupby("date", sort=True):
            if (
                len(group) < 5
                or group["y"].nunique() < 2
                or group["p"].nunique() < 2
            ):
                continue
            ic = group["y"].corr(group["p"], method="spearman")
            if pd.notna(ic):
                ics[date] = float(ic)
        return pd.Series(ics, dtype=float)

    def predict(self, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate ML signals for dates the fitted model has not seen.

        `train` fits the final model on the whole dataset, labels included, and
        this used to predict for *every* date in `factors` -- so each signal
        was an in-sample prediction whose realized forward return the model had
        been fitted on. The walk-forward metrics `train` reports are honest
        out-of-sample numbers; the signal a backtest consumed was not.

        Dates at or before the last training date now come back NaN, which is
        the honest answer: a model trained through T has nothing out-of-sample
        to say about T. In practice that leaves the most recent date, whose
        forward return is not yet observable -- which is what a live signal is.

        Args:
            factors: dict of factor_name -> (date x asset) processed factor values

        Returns:
            (date x asset) signal DataFrame, cross-sectionally ranked
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        first_factor = list(factors.values())[0]
        dates = first_factor.index
        assets = first_factor.columns
        feature_names = (
            list(self._feature_names)
            if self._feature_names else sorted(factors.keys())[:self.config.top_n_features]
        )
        missing = [name for name in feature_names if name not in factors]
        if missing:
            raise ValueError(
                f"Model was trained on factors missing from the input: {missing}"
            )

        signal_data = np.full((len(dates), len(assets)), np.nan)

        train_end = getattr(self, "_train_end_date", None)
        if train_end is not None and len(dates):
            in_sample = dates <= train_end
            if in_sample.any():
                logger.warning(
                    "predict: %d of %d dates are inside the training window "
                    "(<= %s) and are returned as NaN rather than as in-sample "
                    "predictions.",
                    int(in_sample.sum()), len(dates), str(train_end)[:10],
                )
            if in_sample.all():
                logger.warning(
                    "predict: the model was fitted through the last available "
                    "date, so no out-of-sample signal exists for any date."
                )

        for i, date in enumerate(dates):
            if train_end is not None and date <= train_end:
                continue
            features = []
            valid_mask = np.ones(len(assets), dtype=bool)
            for name in feature_names:
                df = factors[name]
                if date in df.index:
                    row = df.loc[date].values
                else:
                    row = np.full(len(assets), np.nan)
                features.append(row)
                valid_mask = valid_mask & ~np.isnan(row)

            if not valid_mask.any():
                continue

            X = np.column_stack(features)
            X_valid = X[valid_mask]

            pred = self.model.predict(X_valid)
            signal_data[i][valid_mask] = pred

        signal = pd.DataFrame(signal_data, index=dates, columns=assets)

        # Cross-sectional rank normalization to [-0.5, 0.5]
        signal = signal.rank(axis=1, pct=True, na_option="keep") - 0.5

        return signal

    def get_shap_values(self, factors: dict[str, pd.DataFrame], n_samples: int = 500) -> dict:
        """Compute SHAP values for model interpretability.

        Args:
            factors: factor data
            n_samples: number of samples for SHAP computation

        Returns:
            dict with shap_values, feature_names, mean_abs_shap
        """
        if not HAS_SHAP:
            return {"error": "shap not installed"}
        if self.model is None:
            return {"error": "model not trained"}

        feature_names = (
            list(self._feature_names)
            if self._feature_names
            else sorted(factors.keys())[:self.config.top_n_features]
        )
        missing = [name for name in feature_names if name not in factors]
        if missing:
            return {"error": f"missing factors: {missing}"}

        first_factor = factors[feature_names[0]]
        dates = first_factor.index
        assets = first_factor.columns

        # Feature-only matrix. `_prepare_dataset` needs forward returns and
        # was called with an empty frame, so it always returned no rows and
        # this function always answered {"error": "no data"}.
        rows = []
        for date in dates:
            features = []
            valid_mask = np.ones(len(assets), dtype=bool)
            for name in feature_names:
                df = factors[name]
                row = (
                    df.loc[date].values
                    if date in df.index else np.full(len(assets), np.nan)
                )
                features.append(row)
                valid_mask = valid_mask & ~np.isnan(row)
            if valid_mask.any():
                rows.append(np.column_stack(features)[valid_mask])

        if not rows:
            return {"error": "no data"}
        X = np.vstack(rows)

        # Subsample for efficiency
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "mean_abs_shap": dict(zip(feature_names, mean_abs_shap.tolist(), strict=False)),
        }

    def generate(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Walk-forward signal generation — NO look-ahead bias.

        For each date t:
        1. Train on data[0:t] (expanding window)
        2. Predict only on data[t] (current cross-section)
        3. Retrain every retrain_frequency dates

        This is the ONLY correct way to use ML in backtesting.

        The first training fold is adaptive. Long histories keep the old
        behaviour (start at ``train_window``); short histories keep
        ``MIN_TRAIN_DATES`` training dates and hold out the rest, so a
        260-day sample produces real out-of-sample signals instead of an
        all-NaN frame.

        Raises:
            InsufficientTrainingData: when no walk-forward fold can be
                trained. The old code returned an all-NaN signal in that case,
                which a caller could not distinguish from "the model has
                nothing out-of-sample to say".
        """
        if not factors:
            self._require_min_training_data(0, 0, context="generate()")

        first_factor = list(factors.values())[0]
        dates = first_factor.index
        assets = first_factor.columns
        n_dates = len(dates)

        # Adaptive first fold: `min(train_window, max(MIN_TRAIN_DATES,
        # n_dates - test_window))`, clamped so at least one date remains to
        # predict. For long histories this is exactly `train_window`.
        test_window = 63
        if n_dates < MIN_TRAIN_DATES + 1:
            raise InsufficientTrainingData(
                f"Insufficient training data for generate(): {n_dates} dates "
                f"available; at least {MIN_TRAIN_DATES} training dates + 1 "
                f"out-of-sample date ({MIN_TRAIN_DATES + 1} dates) are required, "
                f"plus {MIN_TRAIN_SAMPLES} samples and {MIN_TRAIN_FEATURES} "
                f"features."
            )
        first_train_end = min(
            self.config.train_window,
            max(MIN_TRAIN_DATES, n_dates - test_window),
            n_dates - 1,
        )

        signal_data = np.full((n_dates, len(assets)), np.nan)
        model = None
        feature_names = sorted(factors.keys())[:self.config.top_n_features]
        last_train_idx = -self.config.retrain_frequency  # force first train
        perf_log = []

        for i in range(first_train_end, n_dates):
            # Check if we need to retrain
            if model is None or i - last_train_idx >= self.config.retrain_frequency:
                # Prepare training data: all dates before i
                train_start = max(0, i - self.config.train_window)
                X_train, y_train, fold_features, _ = self._prepare_dataset(
                    factors, forward_returns, train_start, i
                )

                try:
                    self._require_min_training_data(
                        len(X_train), len(fold_features),
                        context=f"generate() retrain at {str(dates[i])[:10]}",
                    )
                except InsufficientTrainingData:
                    if model is None:
                        # No model at all: an all-NaN frame would be a silent
                        # failure. Propagate.
                        raise
                    # A later retrain window is too thin; keep the existing
                    # model rather than replacing it with noise, and say so.
                    logger.warning(
                        "generate(): keeping the model trained at %s; retrain "
                        "at %s has only %d samples (%d required).",
                        str(dates[last_train_idx])[:10] if last_train_idx >= 0 else "n/a",
                        str(dates[i])[:10], len(X_train), MIN_TRAIN_SAMPLES,
                    )
                else:
                    model = self._create_model()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train)

                    feature_names = fold_features
                    last_train_idx = i

                    # Quick in-sample IC check
                    train_pred = model.predict(X_train)
                    train_ic = float(np.corrcoef(y_train, train_pred)[0, 1]) if len(y_train) > 10 else 0
                    perf_log.append({'idx': i, 'date': str(dates[i])[:10], 'train_ic': train_ic,
                                     'n_train': len(X_train)})

            if model is None:
                continue

            # Predict on current date only
            features = []
            valid_mask = np.ones(len(assets), dtype=bool)
            for name in feature_names:
                df = factors[name]
                if dates[i] in df.index:
                    row = df.loc[dates[i]].values
                else:
                    row = np.full(len(assets), np.nan)
                features.append(row)
                valid_mask = valid_mask & ~np.isnan(row)

            if not valid_mask.any():
                continue

            X = np.column_stack(features)
            X_valid = X[valid_mask]

            pred = model.predict(X_valid)
            signal_data[i][valid_mask] = pred

        if model is None:
            # Unreachable while `first_train_end < n_dates`: the first loop
            # iteration either fits a model or raises. Kept as the documented
            # no-model contract: never return an all-NaN frame that looks like
            # a signal.
            raise InsufficientTrainingData(
                f"Insufficient training data for generate(): no walk-forward "
                f"fold produced a model from {n_dates} dates "
                f"({MIN_TRAIN_SAMPLES} samples and {MIN_TRAIN_FEATURES} "
                f"features required)."
            )

        signal = pd.DataFrame(signal_data, index=dates, columns=assets)
        # Cross-sectional rank normalization
        signal = signal.rank(axis=1, pct=True, na_option="keep") - 0.5

        # Log performance
        if perf_log:
            avg_ic = np.mean([p['train_ic'] for p in perf_log])
            logger.info("Walk-forward ML: %d retrain cycles, avg train_IC=%.4f",
                        len(perf_log), avg_ic)

        return signal

    def save_model(self, path: str):
        """Save model and its metadata sidecar to disk.

        The LightGBM sklearn wrapper has no `save_model`; the fitted booster
        does. The old code wrote only the JSON sidecar for LightGBM, so
        `load_model` either raised FileNotFoundError or installed an unfitted
        estimator. The sidecar also carries the feature list and the last
        training date, without which a loaded model silently loses the
        in-sample guard in `predict`.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.model_type == "lightgbm":
            if not HAS_LGB:
                raise ImportError("lightgbm not installed")
            booster = getattr(self.model, "booster_", None)
            if booster is None:
                raise RuntimeError("LightGBM model has no fitted booster_")
            booster.save_model(str(model_path))
        elif self.config.model_type == "xgboost":
            if not HAS_XGB:
                raise ImportError("xgboost not installed")
            self.model.save_model(str(model_path))
        else:
            raise ValueError(
                f"Cannot save model_type={self.config.model_type!r}"
            )

        meta = {
            "model_type": self.config.model_type,
            "feature_names": list(self._feature_names or []),
            "top_n_features": self.config.top_n_features,
            "feature_importance": self.feature_importance,
            "train_end_date": (
                str(self._train_end_date) if self._train_end_date else None
            ),
            "last_train_date": str(self._last_train_date) if self._last_train_date else None,
            "performance_history": [
                {
                    "date": p.date,
                    "model_type": p.model_type,
                    "train_ic": p.train_ic,
                    "test_ic": p.test_ic,
                    "train_icir": p.train_icir,
                    "test_icir": p.test_icir,
                    "n_train_samples": p.n_train_samples,
                    "n_test_samples": p.n_test_samples,
                }
                for p in self.performance_history
            ],
        }
        meta_path = Path(str(model_path) + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def load_model(self, path: str):
        """Load a model and its metadata sidecar from disk."""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        meta_path = Path(str(model_path) + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {meta_path}. Re-save the model with "
                "save_model()."
            )
        meta = json.loads(meta_path.read_text())

        model_type = meta.get("model_type", self.config.model_type)
        if model_type == "lightgbm":
            if not HAS_LGB:
                raise ImportError("lightgbm not installed")
            # A raw Booster predicts on the same feature matrix as the
            # sklearn wrapper.
            self.model = lgb.Booster(model_file=str(model_path))
        elif model_type == "xgboost":
            if not HAS_XGB:
                raise ImportError("xgboost not installed")
            model = xgb.XGBRegressor(**self.config.xgb_params)
            model.load_model(str(model_path))
            self.model = model
        else:
            raise ValueError(f"Cannot load model_type={model_type!r}")

        # The artifact's own type and feature list win over the current
        # config: a model trained as xgboost must not be predicted through a
        # lightgbm configuration.
        self.config.model_type = model_type
        names = meta.get("feature_names")
        self._feature_names = list(names) if names else None

        train_end = meta.get("train_end_date")
        self._train_end_date = pd.Timestamp(train_end) if train_end else None
        last_train = meta.get("last_train_date")
        self._last_train_date = pd.Timestamp(last_train) if last_train else None
        self.feature_importance = meta.get("feature_importance", {})
        self.performance_history = [
            ModelPerformance(**p) for p in meta.get("performance_history", [])
        ]
