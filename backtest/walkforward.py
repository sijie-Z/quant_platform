"""Walk-Forward Validation engine.

Splits data into rolling train/test windows, trains on in-sample,
evaluates on out-of-sample. The gold standard for avoiding overfitting.

Two modes:
- Rolling: fixed train window slides forward
- Expanding: train window grows from the start

When `factors` dict is provided, signals are recomputed per fold using
only train-period data — eliminating look-ahead from IC/ICIR weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class WalkForwardValidator:
    """Walk-forward validation for backtesting strategies.

    Splits the timeline into multiple train/test folds:
    [----train----][--test--]
           [----train----][--test--]
                  [----train----][--test--]

    Each fold trains on in-sample data and evaluates out-of-sample.
    The concatenated OOS returns give an unbiased estimate of live performance.

    If `factors` dict is provided, signals are recomputed per fold using only
    train-period data — eliminating look-ahead from IC/ICIR weighting and
    making the OOS evaluation truly representative of live performance.
    """

    def __init__(
        self,
        train_period: int = 504,   # ~2 years of trading days
        test_period: int = 126,    # ~6 months
        step_size: int = 126,      # Step forward by test_period
        mode: str = "rolling",     # "rolling" or "expanding"
    ):
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.mode = mode

    def run(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        sector_map: pd.Series,
        financials: pd.DataFrame | None = None,
        engine_kwargs: dict | None = None,
        factors: dict[str, pd.DataFrame] | None = None,
        alpha_kwargs: dict | None = None,
    ) -> dict:
        """Run walk-forward validation.

        Args:
            signal: Pre-computed (date x asset) alpha signal. Used as fallback
                    when `factors` is None (backward compatible).
            factors: If provided, signal is recomputed per fold using only
                     train-period data. Eliminates look-ahead from IC weighting.
            alpha_kwargs: Dict with 'method', 'lookback', 'min_icir' for signal
                          recomputation. Required when factors is provided.

        Returns:
            dict with keys:
            - oos_returns: concatenated out-of-sample daily returns
            - oos_benchmark: concatenated OOS benchmark returns
            - fold_metrics: list of per-fold performance metrics
            - aggregate_metrics: metrics computed on full OOS period
            - fold_details: train/test date ranges per fold
            - n_folds: number of folds used
            - signal_recomputed: True if signal was recomputed per fold
        """
        engine_kwargs = engine_kwargs or {}
        alpha_kwargs = alpha_kwargs or {}
        signal_recomputed = factors is not None

        dates = signal.index
        n_dates = len(dates)

        if n_dates < self.train_period + self.test_period:
            raise ValueError(
                f"Not enough data: {n_dates} days < {self.train_period} train + {self.test_period} test"
            )

        folds = []
        start = 0
        fold_idx = 0

        while True:
            if self.mode == "expanding":
                train_start = 0
            else:
                train_start = start

            train_end = start + self.train_period
            test_start = train_end
            test_end = min(test_start + self.test_period, n_dates)

            if test_start >= n_dates:
                break

            folds.append({
                "fold": fold_idx,
                "train_dates": (dates[train_start], dates[min(train_end - 1, n_dates - 1)]),
                "test_dates": (dates[test_start], dates[min(test_end - 1, n_dates - 1)]),
                "train_slice": slice(train_start, train_end),
                "test_slice": slice(test_start, test_end),
            })

            fold_idx += 1
            start += self.step_size

            if test_end >= n_dates:
                break

        logger.info("Walk-forward: %d folds (%s mode, train=%d, test=%d, step=%d, recomputed=%s)",
                     len(folds), self.mode, self.train_period, self.test_period, self.step_size,
                     signal_recomputed)

        oos_returns_list = []
        oos_benchmark_list = []
        fold_metrics = []
        fold_details = []

        for fold_info in folds:
            fi = fold_info["fold"]
            train_sl = fold_info["train_slice"]
            test_sl = fold_info["test_slice"]

            # --- Build signal for this fold ---
            if factors is not None:
                fold_signal = self._compute_signal_in_sample(
                    factors, returns, train_sl, test_sl, alpha_kwargs,
                )
            else:
                # Backward compatible: use pre-computed signal
                fold_signal = signal.iloc[test_sl]

            test_prices = prices.iloc[test_sl]
            test_returns = returns.iloc[test_sl]
            test_benchmark = benchmark_returns.iloc[test_sl]

            engine = BacktestEngine(**engine_kwargs)

            try:
                full_sl = slice(max(0, train_sl.start - 252), test_sl.stop)
                full_prices = prices.iloc[full_sl]
                full_returns = returns.iloc[full_sl]

                if fold_signal.index[0] in full_prices.index:
                    result = engine.run(
                        signal=fold_signal,
                        prices=full_prices.loc[fold_signal.index[0]:],
                        returns=full_returns.loc[fold_signal.index[0]:],
                        benchmark_returns=test_benchmark,
                        sector_map=sector_map,
                        financials=financials.iloc[full_sl] if financials is not None else None,
                    )
                else:
                    result = engine.run(
                        signal=fold_signal,
                        prices=test_prices,
                        returns=test_returns,
                        benchmark_returns=test_benchmark,
                        sector_map=sector_map,
                        financials=financials,
                    )

                oos_ret = result["daily_returns"]
                oos_bench = result.get("benchmark_returns", test_benchmark)

                oos_returns_list.append(oos_ret)
                oos_benchmark_list.append(oos_bench)

                if len(oos_ret) > 0:
                    m = all_metrics(oos_ret, oos_bench)
                    m["fold"] = fi
                    m["train_start"] = str(fold_info["train_dates"][0])
                    m["train_end"] = str(fold_info["train_dates"][1])
                    m["test_start"] = str(fold_info["test_dates"][0])
                    m["test_end"] = str(fold_info["test_dates"][1])
                    m["oos_days"] = len(oos_ret)
                    fold_metrics.append(m)

                fold_details.append({
                    "fold": fi,
                    "train": f"{fold_info['train_dates'][0]} -> {fold_info['train_dates'][1]}",
                    "test": f"{fold_info['test_dates'][0]} -> {fold_info['test_dates'][1]}",
                    "oos_days": len(fold_signal),
                    "sharpe": fold_metrics[-1].get("sharpe_ratio", 0) if fold_metrics else 0,
                })

                logger.info("  Fold %d: OOS %s -> %s (%d days, Sharpe=%.2f)",
                            fi, fold_info["test_dates"][0], fold_info["test_dates"][1],
                            len(oos_ret), fold_metrics[-1].get("sharpe_ratio", 0))

            except Exception as e:
                logger.warning("  Fold %d failed: %s", fi, e)
                fold_details.append({
                    "fold": fi,
                    "train": f"{fold_info['train_dates'][0]} -> {fold_info['train_dates'][1]}",
                    "test": f"{fold_info['test_dates'][0]} -> {fold_info['test_dates'][1]}",
                    "oos_days": 0,
                    "error": str(e)[:100],
                })

        if not oos_returns_list:
            raise ValueError("All walk-forward folds failed")

        oos_returns = pd.concat(oos_returns_list)[~pd.concat(oos_returns_list).index.duplicated(keep="first")].sort_index()
        oos_benchmark = pd.concat(oos_benchmark_list) if oos_benchmark_list else None
        if oos_benchmark is not None and oos_benchmark.index.duplicated().any():
            oos_benchmark = oos_benchmark[~oos_benchmark.index.duplicated(keep="first")].sort_index()

        aggregate = all_metrics(oos_returns, oos_benchmark)

        # Stability analysis across folds
        fold_sharpes = [m.get("sharpe_ratio", 0) for m in fold_metrics]
        fold_returns = [m.get("total_return", 0) for m in fold_metrics]

        stability = {
            "mean_sharpe": float(np.mean(fold_sharpes)) if fold_sharpes else 0,
            "std_sharpe": float(np.std(fold_sharpes)) if fold_sharpes else 0,
            "min_sharpe": float(np.min(fold_sharpes)) if fold_sharpes else 0,
            "max_sharpe": float(np.max(fold_sharpes)) if fold_sharpes else 0,
            "sharpe_consistency": float(np.mean([1 for s in fold_sharpes if s > 0])) if fold_sharpes else 0,
            "mean_return": float(np.mean(fold_returns)) if fold_returns else 0,
            "return_std": float(np.std(fold_returns)) if fold_returns else 0,
            "positive_folds": sum(1 for r in fold_returns if r > 0),
            "total_folds": len(fold_returns),
        }

        return {
            "oos_returns": oos_returns,
            "oos_benchmark": oos_benchmark,
            "fold_metrics": fold_metrics,
            "aggregate_metrics": aggregate,
            "fold_details": fold_details,
            "stability": stability,
            "n_folds": len(folds),
            "mode": self.mode,
            "train_period": self.train_period,
            "test_period": self.test_period,
            "signal_recomputed": signal_recomputed,
        }

    def _compute_signal_in_sample(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        train_slice: slice,
        test_slice: slice,
        alpha_kwargs: dict,
    ) -> pd.DataFrame:
        """Recompute signal using only train-period data for IC weights.

        Factor values are taken from the test period (those are observable),
        but IC weights are estimated from train-period data only.
        """
        from quant_platform.alpha.combination import (
            combine_equal_weight,
            combine_ic_weighted,
            combine_icir_weighted,
        )

        method = alpha_kwargs.get("method", "equal_weight")
        lookback = alpha_kwargs.get("lookback", 252)
        min_icir = alpha_kwargs.get("min_icir", 0.0)

        # Slice factors to train + test dates (test period has factor values)
        full_slice = slice(train_slice.start, test_slice.stop)
        fold_factors = {}
        for name, df in factors.items():
            fold_factors[name] = df.iloc[full_slice]

        # Slice forward_returns to train period only
        train_returns = forward_returns.iloc[train_slice]

        if method == "equal_weight":
            return combine_equal_weight(fold_factors)

        elif method == "ic_weighted":
            return combine_ic_weighted(fold_factors, train_returns, lookback)

        elif method == "icir_weighted":
            return combine_icir_weighted(fold_factors, train_returns, lookback, min_icir)

        else:
            return combine_equal_weight(fold_factors)
