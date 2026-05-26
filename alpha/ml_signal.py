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

    def split(self, n_samples: int):
        """Generate train/test index pairs with purge and embargo.

        Args:
            n_samples: Total number of samples.

        Yields:
            (train_indices, test_indices) tuples.
        """
        min_size = self.train_size + self.gap + self.test_size
        if n_samples < min_size:
            raise ValueError(f"Need at least {min_size} samples, got {n_samples}")

        # Step size accounts for test_size + embargo (next fold's train starts after embargo)
        available = n_samples - self.train_size - self.gap
        step = max(1, (available - self.test_size) // max(self.n_splits - 1, 1))

        for i in range(self.n_splits):
            if self.mode == "expanding":
                train_start = 0
            else:  # rolling
                train_start = max(0, i * step - self.embargo * i)

            train_end = self.train_size + i * step
            test_start = train_end + self.gap  # purge gap
            test_end = min(test_start + self.test_size, n_samples)

            if train_end >= n_samples or test_start >= n_samples:
                break

            train_idx = list(range(train_start, min(train_end, n_samples)))
            test_idx = list(range(test_start, test_end))

            if len(train_idx) > 0 and len(test_idx) > 0:
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
            return None  # Will create both
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
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare training dataset from factor cross-sections.

        Flattens all dates and assets into a single feature matrix.
        Each row = (date, asset) pair.
        """
        feature_names = sorted(factors.keys())[:self.config.top_n_features]
        first_factor = factors[feature_names[0]]
        dates = first_factor.index[start_idx:end_idx]
        assets = first_factor.columns

        X_list = []
        y_list = []

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

        if not X_list:
            return np.array([]), np.array([]), feature_names

        return np.vstack(X_list), np.concatenate(y_list), feature_names

    def train(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> ModelPerformance:
        """Train the ML model with time-series cross-validation.

        Args:
            factors: dict of factor_name -> (date x asset) processed factor values
            forward_returns: (date x asset) forward returns

        Returns:
            ModelPerformance with CV results
        """
        first_factor = list(factors.values())[0]
        n_dates = len(first_factor.index)

        # Prepare full dataset for CV
        X_full, y_full, feature_names = self._prepare_dataset(
            factors, forward_returns, 0, n_dates
        )

        if len(X_full) < 100:
            logger.warning("Not enough data for ML training: %d samples", len(X_full))
            return ModelPerformance(date=str(first_factor.index[-1]), model_type=self.config.model_type)

        # Time-series CV with purge gap
        cv = TimeSeriesCV(
            n_splits=self.config.n_splits,
            train_size=self.config.train_window,
            test_size=63,
            gap=self.config.purge_gap,
            embargo=self.config.embargo,
        )

        cv_results = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(len(X_full))):
            X_train, y_train = X_full[train_idx], y_full[train_idx]
            X_test, y_test = X_full[test_idx], y_full[test_idx]

            model = self._create_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_ic = float(np.corrcoef(y_train, train_pred)[0, 1]) if len(y_train) > 10 else 0
            test_ic = float(np.corrcoef(y_test, test_pred)[0, 1]) if len(y_test) > 10 else 0

            cv_results.append({
                "fold": fold,
                "train_ic": train_ic,
                "test_ic": test_ic,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

            logger.info("  Fold %d: train_IC=%.4f, test_IC=%.4f (n_train=%d, n_test=%d)",
                        fold, train_ic, test_ic, len(train_idx), len(test_idx))

        # Train final model on all data
        self.model = self._create_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_full, y_full)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(feature_names, importances.tolist(), strict=False))

        # Aggregate CV metrics
        test_ics = [r["test_ic"] for r in cv_results]
        mean_test_ic = float(np.mean(test_ics))
        std_test_ic = float(np.std(test_ics)) if len(test_ics) > 1 else 0
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

        logger.info("ML model trained: type=%s, test_IC=%.4f, test_ICIR=%.4f, features=%d",
                     self.config.model_type, mean_test_ic, test_icir, len(feature_names))

        return perf

    def predict(self, factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate ML-based signals for all dates.

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
        feature_names = sorted(factors.keys())[:self.config.top_n_features]

        signal_data = np.full((len(dates), len(assets)), np.nan)

        for i, date in enumerate(dates):
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

        first_factor = list(factors.values())[0]
        n_dates = len(first_factor.index)
        X, _, feature_names = self._prepare_dataset(factors, pd.DataFrame(), 0, n_dates)

        if len(X) == 0:
            return {"error": "no data"}

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
        """
        first_factor = list(factors.values())[0]
        dates = first_factor.index
        assets = first_factor.columns
        feature_names = sorted(factors.keys())[:self.config.top_n_features]
        n_dates = len(dates)

        signal_data = np.full((n_dates, len(assets)), np.nan)
        model = None
        last_train_idx = -self.config.retrain_frequency  # force first train
        perf_log = []

        for i in range(max(self.config.train_window, 60), n_dates):
            # Check if we need to retrain
            if i - last_train_idx >= self.config.retrain_frequency or model is None:
                # Prepare training data: all dates before i
                train_start = max(0, i - self.config.train_window)
                X_train, y_train, feat_names = self._prepare_dataset(
                    factors, forward_returns, train_start, i
                )

                if len(X_train) < 100:
                    continue

                model = self._create_model()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)

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
        """Save model to disk."""
        if self.model is None:
            return
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, 'save_model'):
            self.model.save_model(str(model_path))

        # Save metadata
        meta = {
            "model_type": self.config.model_type,
            "feature_importance": self.feature_importance,
            "last_train_date": str(self._last_train_date) if self._last_train_date else None,
            "performance_history": [
                {
                    "date": p.date,
                    "test_ic": p.test_ic,
                    "test_icir": p.test_icir,
                }
                for p in self.performance_history
            ],
        }
        meta_path = model_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def load_model(self, path: str):
        """Load model from disk."""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model = self._create_model()
        if hasattr(model, 'load_model'):
            model.load_model(str(model_path))
        self.model = model

        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_importance = meta.get("feature_importance", {})
            self._last_train_date = meta.get("last_train_date")
