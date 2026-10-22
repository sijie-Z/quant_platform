"""Walk-Forward Validation engine.

Splits data into rolling train/test windows, trains on in-sample,
evaluates on out-of-sample. The gold standard for avoiding overfitting.

Two modes:
- Rolling: fixed train window slides forward
- Expanding: train window grows from the start

When `factors` dict is provided, signals are recomputed per fold using
only train-period data — eliminating look-ahead from IC/ICIR weighting.

An optional ``factor_selector`` can pick the factor subset *inside each fold*
from that fold's training window only (see :data:`FactorSelector`). The
default (``None``) keeps the historical behaviour bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.metrics import all_metrics
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: Per-fold factor selector.
#:
#: Called once per fold, before that fold's signal is computed, as
#:
#: ``selector(fold_info, factors, forward_returns) -> iterable of factor names``
#:
#: * ``fold_info``: the fold metadata dict. ``fold_info["train_positions"]``
#:   holds the integer positions (into the validator's date axis) usable for
#:   training — the nominal training window minus embargo days. The dict also
#:   carries ``fold``, ``train_slice``, ``test_slice``, ``train_dates``,
#:   ``test_dates``, ``train_days`` and ``embargo_ranges``.
#: * ``factors``: the full ``{name: (date x asset) panel}`` mapping.
#: * ``forward_returns``: the same forward-return panel the in-sample signal
#:   computation uses (``t -> t+1`` labels).
#:
#: The returned names must be existing keys of ``factors``; only those panels
#: feed the fold's :meth:`WalkForwardValidator._compute_signal_in_sample`.
#: Returning an empty iterable skips the fold (recorded in ``fold_details``,
#: never fabricated). Exceptions raised by the selector abort the run.
FactorSelector = Callable[[dict, dict[str, pd.DataFrame], pd.DataFrame], Sequence[str]]



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

    Purge / embargo isolation (same terminology as `alpha.ml_signal`):

    - ``purge_gap``: each fold's test window starts ``purge_gap`` trading days
      after its training window ends. Those days are neither trained nor
      tested, so labels from the end of the training window cannot spill into
      the test window. purge_gap：测试窗起点相对训练窗终点向后推迟
      purge_gap 个交易日，这段空隙既不训练也不测试。
    - ``embargo``: when training any fold, the ``embargo`` trading days that
      immediately follow an earlier fold's test window are removed from the
      training sample (only when that training window covers those days).
      Freshly tested market conditions must not flow straight back into the
      next fit. embargo：此前某个 fold 的测试窗结束后的 embargo 个交易日
      从后续 fold 的训练样本中剔除。

    The rolling/expanding window advancement is unchanged, and the fold count
    is still capped by ``n_folds``.
    """

    def __init__(
        self,
        train_period: int = 504,   # ~2 years of trading days
        test_period: int = 126,    # ~6 months
        step_size: int = 126,      # Step forward by test_period
        mode: str = "rolling",     # "rolling" or "expanding"
        n_folds: int | None = None,
        purge_gap: int = 0,        # trading days between train end and test start
        embargo: int = 0,          # trading days after a prior test window excluded from training
    ):
        # M22: `start += step_size` never advanced for step_size <= 0, so the
        # fold loop spun forever (or sliced invalid windows). Validate the
        # window arithmetic at construction instead of hanging a worker.
        for name, value in (
            ("train_period", train_period),
            ("test_period", test_period),
            ("step_size", step_size),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value!r}")
        if mode not in ("rolling", "expanding"):
            raise ValueError(f"mode must be 'rolling' or 'expanding', got {mode!r}")
        # purge_gap / embargo may be 0 but never negative: both are offsets
        # into the trading-day axis, and a negative gap would silently turn
        # the test window into an in-sample slice.
        for name, value in (("purge_gap", purge_gap), ("embargo", embargo)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.mode = mode
        if n_folds is not None and n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        self.n_folds = n_folds
        self.purge_gap = int(purge_gap)
        self.embargo = int(embargo)

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
        cost_model=None,
        constraints=None,
        optimizer: str | None = None,
        benchmark: str | None = None,
        covariance_method: str | None = None,
        covariance_lookback: int | None = None,
        covariance_ewma_half_life: int | None = None,
        execution_timing: str | None = None,
        risk_free_rate: float | None = None,
        periods_per_year: float | None = None,
        execution_gates: dict | None = None,
        factor_selector: FactorSelector | None = None,
    ) -> dict:
        """Run walk-forward validation.

        Args:
            signal: Pre-computed (date x asset) alpha signal. Used as fallback
                    when `factors` is None (backward compatible).
            factors: If provided, signal is recomputed per fold using only
                     train-period data. Eliminates look-ahead from IC weighting.
            alpha_kwargs: Dict with 'method', 'lookback', 'min_icir' for signal
                          recomputation. Required when factors is provided.
            factor_selector: Optional per-fold factor selector (see
                :data:`FactorSelector`). Called once per fold with that fold's
                ``train_positions`` (embargo already removed), the full factor
                panels and the forward-return panel; the returned names are
                the only factors fed to that fold's signal recomputation. An
                empty selection skips the fold and records it in
                ``fold_details``; selector exceptions abort the run. Default
                ``None`` recomputes every fold from all ``factors`` exactly as
                before.
            engine_kwargs: Raw BacktestEngine keyword arguments.
            cost_model, constraints, optimizer, benchmark, covariance_method,
            covariance_lookback, covariance_ewma_half_life, execution_timing,
            risk_free_rate, periods_per_year:
                First-class execution-model settings forwarded to every fold's
                ``BacktestEngine`` (B3). They are merged over ``engine_kwargs``
                (explicit values win). The old contract only forwarded what the
                CLI happened to put in a three-key dict, so a configured
                CostModel/optimizer/covariance silently reverted to engine
                defaults for every OOS fold.

        Returns:
            dict with keys:
            - oos_returns: concatenated out-of-sample daily returns
            - oos_benchmark: concatenated OOS benchmark returns
            - fold_metrics: list of per-fold performance metrics
            - aggregate_metrics: metrics computed on full OOS period
            - fold_details: train/test date ranges per fold
            - n_folds: number of folds used
            - signal_recomputed: True if signal was recomputed per fold

            When ``factor_selector`` is used, every ``fold_metrics`` /
            ``fold_details`` entry also carries ``selected_factors`` (the list
            that fold's signal was computed from). With the default
            ``factor_selector=None`` these dicts keep their historical shape.
        """
        engine_kwargs = dict(engine_kwargs or {})
        explicit_engine_config = {
            "cost_model": cost_model,
            "constraints": constraints,
            "optimizer": optimizer,
            "benchmark": benchmark,
            "covariance_method": covariance_method,
            "covariance_lookback": covariance_lookback,
            "covariance_ewma_half_life": covariance_ewma_half_life,
            "execution_timing": execution_timing,
            "risk_free_rate": risk_free_rate,
            "periods_per_year": periods_per_year,
        }
        engine_kwargs.update({
            key: value for key, value in explicit_engine_config.items()
            if value is not None
        })
        alpha_kwargs = alpha_kwargs or {}
        signal_recomputed = factors is not None
        if factor_selector is not None and factors is None:
            # Nothing to select from a pre-computed signal; a selector here
            # would be silently ignored, which is worse than a clear error.
            raise ValueError(
                "factor_selector requires factors: there is nothing to select "
                "from a pre-computed signal"
            )

        dates = signal.index
        n_dates = len(dates)

        if n_dates < self.train_period + self.test_period:
            raise ValueError(
                f"Not enough data: {n_dates} days < {self.train_period} train + {self.test_period} test"
            )

        folds = self._generate_folds(dates)

        logger.info("Walk-forward: %d folds (%s mode, train=%d, test=%d, step=%d, "
                     "purge_gap=%d, embargo=%d, recomputed=%s)",
                     len(folds), self.mode, self.train_period, self.test_period,
                     self.step_size, self.purge_gap, self.embargo, signal_recomputed)

        oos_returns_list = []
        oos_benchmark_list = []
        fold_metrics = []
        fold_details = []

        for fold_info in folds:
            fi = fold_info["fold"]
            train_sl = fold_info["train_slice"]
            test_sl = fold_info["test_slice"]

            # --- Build signal for this fold ---
            selected_factors: list[str] | None = None
            if factors is not None:
                if fold_info["train_days"] == 0:
                    # An out-sized embargo can remove every day of a later
                    # fold's training window. There is nothing to fit on --
                    # record the fold instead of fabricating a signal.
                    logger.warning(
                        "  Fold %d skipped: embargo removed every training day", fi,
                    )
                    detail = {
                        "fold": fi,
                        "train": f"{fold_info['train_dates'][0]} -> {fold_info['train_dates'][1]}",
                        "test": f"{fold_info['test_dates'][0]} -> {fold_info['test_dates'][1]}",
                        "train_days": 0,
                        "oos_days": 0,
                        "error": "embargo removed every training day",
                    }
                    if factor_selector is not None:
                        detail["selected_factors"] = []
                    fold_details.append(detail)
                    continue
                fold_factors = factors
                if factor_selector is not None:
                    # Choose this fold's factors from its training window only
                    # (train_positions already excludes embargo days). The
                    # selector must never see the test window as training
                    # input; test-period factor *values* are still observable
                    # and are used by the signal recomputation itself.
                    selected_factors, fold_factors = self._select_fold_factors(
                        factor_selector, fold_info, factors, returns,
                    )
                    if not selected_factors:
                        # Nothing survives this fold's training-window screen.
                        # Record the fold and move on -- never fabricate OOS.
                        logger.warning(
                            "  Fold %d skipped: factor_selector selected no factors", fi,
                        )
                        fold_details.append({
                            "fold": fi,
                            "train": f"{fold_info['train_dates'][0]} -> {fold_info['train_dates'][1]}",
                            "test": f"{fold_info['test_dates'][0]} -> {fold_info['test_dates'][1]}",
                            "train_days": fold_info["train_days"],
                            "oos_days": 0,
                            "selected_factors": [],
                            "error": "factor_selector selected no factors",
                        })
                        continue
                fold_signal = self._compute_signal_in_sample(
                    fold_factors, returns, train_sl, test_sl, alpha_kwargs,
                    train_positions=fold_info["train_positions"],
                )
            else:
                # Backward compatible: use pre-computed signal. The real alpha
                # stays test-only, but the *index* is extended back over the
                # train window with NaN rows: the engine takes its rebalance
                # dates from the signal index, and without those earlier
                # month-ends it sat in cash for the first weeks of every test
                # window (B4). NaN signal rows make the engine fall back to an
                # equal-weight warm-up portfolio, which is held into the test
                # window; no train-period alpha is leaked.
                fold_signal = signal.iloc[test_sl]
                if train_sl.start < test_sl.start:
                    warmup_index = signal.index[train_sl.start:test_sl.start]
                    if len(warmup_index):
                        warmup = pd.DataFrame(
                            np.nan, index=warmup_index, columns=signal.columns,
                        )
                        fold_signal = pd.concat([warmup, fold_signal])

            test_prices = prices.iloc[test_sl]
            test_returns = returns.iloc[test_sl]
            test_benchmark = benchmark_returns.iloc[test_sl]

            engine = BacktestEngine(**engine_kwargs)

            try:
                # Feed the engine the full panel -- including the 252-day
                # warm-up before the train window -- so the first fold
                # rebalance estimates covariance on a full window. The old
                # code sliced `.loc[fold_signal.index[0]:]`, discarding the
                # warm-up it had just computed (B4).
                full_sl = slice(max(0, train_sl.start - 252), test_sl.stop)
                full_prices = prices.iloc[full_sl]
                full_returns = returns.iloc[full_sl]

                if len(fold_signal) and fold_signal.index[0] in full_prices.index:
                    result = engine.run(
                        signal=fold_signal,
                        prices=full_prices,
                        returns=full_returns,
                        benchmark_returns=test_benchmark,
                        sector_map=sector_map,
                        financials=financials.iloc[full_sl] if financials is not None else None,
                        **(execution_gates or {}),
                    )
                else:
                    result = engine.run(
                        signal=fold_signal,
                        prices=test_prices,
                        returns=test_returns,
                        benchmark_returns=test_benchmark,
                        sector_map=sector_map,
                        financials=financials,
                        **(execution_gates or {}),
                    )

                # The engine is handed extra history so its first rebalance has
                # a warm-up, but only the test window is out of sample. Taking
                # `result["daily_returns"]` whole made every fold's reported
                # metrics cover train + test -- a 63-day test window was being
                # reported over 315 days, so the "OOS" numbers were mostly
                # in-sample performance.
                test_start_ts = prices.index[test_sl.start]

                oos_ret = result["daily_returns"]
                if len(oos_ret) > 0:
                    oos_ret = oos_ret.loc[oos_ret.index >= test_start_ts]

                oos_bench = result.get("benchmark_returns", test_benchmark)
                if hasattr(oos_bench, "index") and len(oos_bench) > 0:
                    oos_bench = oos_bench.loc[oos_bench.index >= test_start_ts]

                oos_returns_list.append(oos_ret)
                oos_benchmark_list.append(oos_bench)

                if len(oos_ret) > 0:
                    m = all_metrics(oos_ret, oos_bench)
                    m["fold"] = fi
                    m["train_start"] = str(fold_info["train_dates"][0])
                    m["train_end"] = str(fold_info["train_dates"][1])
                    m["test_start"] = str(fold_info["test_dates"][0])
                    m["test_end"] = str(fold_info["test_dates"][1])
                    # Trading days actually usable for training: the nominal
                    # window minus any embargo days removed from it.
                    m["train_days"] = fold_info["train_days"]
                    m["oos_days"] = len(oos_ret)
                    if selected_factors is not None:
                        m["selected_factors"] = list(selected_factors)
                    fold_metrics.append(m)

                detail = {
                    "fold": fi,
                    "train": f"{fold_info['train_dates'][0]} -> {fold_info['train_dates'][1]}",
                    "test": f"{fold_info['test_dates'][0]} -> {fold_info['test_dates'][1]}",
                    "train_days": fold_info["train_days"],
                    "oos_days": len(oos_ret),
                    "sharpe": fold_metrics[-1].get("sharpe_ratio", 0) if fold_metrics else 0,
                }
                if selected_factors is not None:
                    detail["selected_factors"] = list(selected_factors)
                fold_details.append(detail)

                logger.info("  Fold %d: OOS %s -> %s (%d days, Sharpe=%.2f)",
                            fi, fold_info["test_dates"][0], fold_info["test_dates"][1],
                            len(oos_ret), fold_metrics[-1].get("sharpe_ratio", 0))

            except Exception as e:
                logger.warning("  Fold %d failed: %s", fi, e)
                failure = {
                    "fold": fi,
                    "train": f"{fold_info['train_dates'][0]} -> {fold_info['train_dates'][1]}",
                    "test": f"{fold_info['test_dates'][0]} -> {fold_info['test_dates'][1]}",
                    "train_days": fold_info["train_days"],
                    "oos_days": 0,
                    "error": str(e)[:100],
                }
                if selected_factors is not None:
                    failure["selected_factors"] = list(selected_factors)
                fold_details.append(failure)

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
            "purge_gap": self.purge_gap,
            "embargo": self.embargo,
            "signal_recomputed": signal_recomputed,
        }

    def _generate_folds(self, dates: pd.Index) -> list[dict]:
        """Cut the timeline into train/test folds.

        Window arithmetic is the historical layout plus the optional purge
        gap: the test window starts ``purge_gap`` trading days after the
        training window ends, and the fold count is still capped by
        ``n_folds``. Each fold also carries ``train_positions`` -- the integer
        positions actually usable for training after embargo days are removed
        -- and the removed ranges in ``embargo_ranges``.

        With ``purge_gap=0`` and ``embargo=0`` the boundaries are identical
        to the pre-purge arithmetic.
        """
        n_dates = len(dates)
        folds = []
        start = 0
        fold_idx = 0

        while True:
            if self.mode == "expanding":
                train_start = 0
            else:
                train_start = start

            train_end = start + self.train_period
            test_start = train_end + self.purge_gap
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

        if self.n_folds is not None:
            # The CLI exposes --folds; previously it only printed the value
            # while the loop decided the count, so a request for one fold ran
            # every fold the data allowed.
            folds = folds[:self.n_folds]

        self._apply_embargo(folds, n_dates)
        return folds

    def _apply_embargo(self, folds: list[dict], n_dates: int) -> None:
        """Remove embargo days from each fold's training positions.

        For a fold at index ``i``, every earlier fold's test window end is
        followed by ``embargo`` trading days. When the current *training*
        window covers any of those days, they are withheld from training --
        the market regime that was just evaluated must not flow straight back
        into the next fit. Windows ending at or after the training window end
        leave it untouched; in the common rolling layout (step ==
        test_period) the immediately preceding test window ends exactly at
        the current train end, so only older test windows contribute embargo
        days there.
        """
        for i, fold in enumerate(folds):
            train_sl = fold["train_slice"]
            positions = np.arange(train_sl.start, train_sl.stop, dtype=int)
            removed = np.zeros(len(positions), dtype=bool)
            embargo_ranges: list[tuple[int, int]] = []
            if self.embargo > 0:
                for prev in folds[:i]:
                    lo = prev["test_slice"].stop
                    hi = min(lo + self.embargo, n_dates)
                    lo = max(lo, train_sl.start)
                    hi = min(hi, train_sl.stop)
                    if lo < hi:
                        removed[lo - train_sl.start:hi - train_sl.start] = True
                        embargo_ranges.append((int(lo), int(hi)))
            fold["train_positions"] = positions[~removed]
            fold["train_days"] = len(fold["train_positions"])
            fold["embargo_ranges"] = embargo_ranges

    @staticmethod
    def _select_fold_factors(
        selector: FactorSelector,
        fold_info: dict,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> tuple[list[str], dict[str, pd.DataFrame]]:
        """Run one per-fold selection and validate the returned names.

        The selector receives a shallow copy of ``fold_info`` whose
        ``train_positions`` array is copied, so a selector cannot mutate the
        metadata the signal computation later relies on. Unknown names and
        string returns are contract violations and fail loud here rather than
        silently producing a signal from the wrong panels.
        """
        selector_fold_info = dict(fold_info)
        selector_fold_info["train_positions"] = np.array(
            fold_info["train_positions"], copy=True,
        )
        chosen = selector(selector_fold_info, factors, forward_returns)
        if isinstance(chosen, (str, bytes)):
            raise ValueError(
                "factor_selector must return an iterable of factor names, "
                f"got the string {chosen!r}"
            )
        selected = list(chosen)
        if not all(name in factors for name in selected):
            unknown = [name for name in selected if name not in factors]
            raise ValueError(
                f"factor_selector returned unknown factor(s) {unknown!r}; "
                f"available factors: {sorted(factors)}"
            )
        # Preserve the selector's order but drop duplicates for clarity.
        selected = list(dict.fromkeys(selected))
        return selected, {name: factors[name] for name in selected}

    def _compute_signal_in_sample(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        train_slice: slice,
        test_slice: slice,
        alpha_kwargs: dict,
        train_positions: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Recompute signal using only train-period data for IC weights.

        Factor values are taken from the test period (those are observable),
        but IC weights are estimated from train-period data only. When
        ``train_positions`` is given, only those positions (the nominal train
        window minus embargo days) feed the IC estimation.
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

        # Slice forward_returns to train period only (minus embargo days)
        if train_positions is None:
            train_returns = forward_returns.iloc[train_slice]
        else:
            train_returns = forward_returns.iloc[train_positions]

        if method == "equal_weight":
            return combine_equal_weight(fold_factors)

        elif method == "ic_weighted":
            return combine_ic_weighted(fold_factors, train_returns, lookback)

        elif method == "icir_weighted":
            return combine_icir_weighted(fold_factors, train_returns, lookback, min_icir)

        else:
            return combine_equal_weight(fold_factors)
