"""One causal multi-factor composite signal for every live trading path.

Both the CLI runner (``python main.py trade`` -> :class:`LiveRunner`) and the
web engine (``POST /api/trading/start`` -> :class:`LiveTradingEngine`) score
their universe with :func:`composite_signal`, so a paper run and a live run
over the same price history rank the same symbols the same way.  The factor
set is the one the web engine has always used:

    momentum_1m + momentum_3m + momentum_6m
    + volatility_20d + volatility_60d + rsi_14d + macd

combined with **equal weights** (1/7 each) after the research pipeline's
cross-sectional processing (1%/99% winsorization, then z-score), exactly as
``AlphaPipeline(method="equal_weight")`` does in the backtest/research path.
The implementation reuses those very factor classes and processing helpers --
it is not a second, hand-rolled definition.

Scale
-----
The returned score is a **cross-sectional percentile in [0, 1]**.  On the
decision row the combined factor value is ranked across the universe
(``rank(pct=True)``) and min-max scaled: ``1.0`` is the most attractive
symbol in the cross-section, ``0.0`` the least attractive.  A degenerate
cross-section (every symbol tied) maps to ``0.0`` for every symbol, so it
never clears a positive entry threshold.  Ties share one averaged rank and
therefore one score.

Causality
---------
Only rows at or before the decision row (the last row, or ``as_of``) are
used.  Every factor is a trailing rolling/EWMA statistic and every processing
step is per-row cross-sectional, so truncating or mutating later rows cannot
change an earlier score.  ``history_limit`` additionally bounds the lookback
window (only the trailing ``SIGNAL_HISTORY_SESSIONS`` rows are consumed), which
keeps the window identical for every caller.

Insufficient history
--------------------
A panel shorter than ``MIN_HISTORY`` rows yields an all-NaN Series: those
symbols are excluded from ranking.  When passing a ``{symbol: [prices]}``
mapping, symbols with fewer than ``MIN_HISTORY`` finite observations are NaN
and never enter the cross-section used to score the others; eligible symbols
are left-padded with NaN onto one row axis so each keeps its own full trailing
history (matching an equivalently padded DataFrame).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.alpha.combination import combine_equal_weight
from quant_platform.factors.processing import process_factor
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: Minimum number of price rows before any score is produced.  This is the
#: gate both live paths used before the extraction (20 observations).
MIN_HISTORY = 20

#: Trailing window (in price rows) consumed by the composite.  126 daily
#: returns are required for six-month momentum, which needs 127 closes.  The
#: same cap bounds the engine's quote buffer and the runner's history buffer,
#: so both entry points score an identical window.
SIGNAL_HISTORY_SESSIONS = 127

#: Default entry threshold on the [0, 1] percentile scale.  The web engine has
#: always used 0.3 (``min_signal_score``); the CLI runner adopts it so the two
#: paths select from the same score distribution.
DEFAULT_MIN_SIGNAL_SCORE = 0.3

#: The seven factors of the engine composite, in combination order.  Order is
#: fixed because the equal-weight sum is accumulated in this order (floating
#: point addition is not associative).
SIGNAL_FACTORS: tuple[str, ...] = (
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "volatility_20d",
    "volatility_60d",
    "rsi_14d",
    "macd",
)


def _factor_instances(factor_names: Sequence[str]) -> list[Any]:
    """Instantiate the requested factors from the research registry.

    The registry lookup (rather than a hardcoded factory table) means any
    registered factor can be requested -- ``LiveSignalGenerator``'s
    ``factor_names`` argument keeps working for custom factor sets -- while
    the default seven stay the single definition used by both live paths.
    """
    from quant_platform.factors.registry import get_registry

    registry = get_registry()
    registered_all = False
    factors = []
    for name in factor_names:
        try:
            factor_cls = registry.get(name)
        except KeyError:
            # The registry is a process-wide singleton; if it is empty or was
            # only partially populated, register the technical factors once
            # and retry before giving up on the name.
            if not registered_all:
                from quant_platform.factors.technical import register_all
                register_all()
                registered_all = True
                try:
                    factor_cls = registry.get(name)
                except KeyError:
                    logger.warning("Unknown composite factor '%s' skipped", name)
                    continue
            else:
                logger.warning("Unknown composite factor '%s' skipped", name)
                continue
        try:
            factors.append(factor_cls())
        except Exception as exc:
            logger.warning("Composite factor '%s' could not start: %s", name, exc)
    return factors


def _price_frame(
    price_history: pd.DataFrame | Mapping[str, Sequence[float]],
    min_periods: int,
) -> tuple[list[Any], pd.DataFrame | None]:
    """Normalize supported inputs to a ``(symbols, frame)`` pair.

    * A ``DataFrame`` is used as given (columns are symbols, rows are time).
    * A ``{symbol: prices}`` mapping keeps only symbols with at least
      ``min_periods`` finite observations, then left-pads the eligible
      histories with NaN so every symbol shares one row axis while keeping
      its own full trailing history (the same shape a caller would get from
      an equivalently NaN-padded DataFrame).  Ineligible symbols stay in the
      returned ``symbols`` list so the caller still gets a NaN score for
      them, and they never enter the cross-section used for scoring others.
    """
    if isinstance(price_history, pd.DataFrame):
        return list(price_history.columns), price_history

    if not isinstance(price_history, Mapping):
        raise TypeError(
            "price_history must be a DataFrame or a {symbol: prices} mapping, "
            f"got {type(price_history).__name__}"
        )

    symbols = list(price_history)
    series: dict[Any, np.ndarray] = {}
    for symbol, values in price_history.items():
        if isinstance(values, pd.Series):
            array = values.to_numpy(dtype=float)
        else:
            array = np.asarray(list(values), dtype=float)
        if array.ndim != 1:
            raise ValueError(
                f"Price history for '{symbol}' must be one-dimensional"
            )
        series[symbol] = array

    eligible = {
        symbol: array
        for symbol, array in series.items()
        if int(np.isfinite(array).sum()) >= min_periods
    }
    if not eligible:
        return symbols, None

    common_len = max(array.size for array in eligible.values())
    frame = pd.DataFrame({
        symbol: np.concatenate(
            (np.full(common_len - array.size, np.nan), array),
        )
        for symbol, array in eligible.items()
    })
    return symbols, frame


def composite_signal(
    price_history: pd.DataFrame | Mapping[str, Sequence[float]],
    *,
    factor_names: Sequence[str] | None = None,
    factor_kwargs: Mapping[str, Any] | None = None,
    min_periods: int = MIN_HISTORY,
    history_limit: int | None = SIGNAL_HISTORY_SESSIONS,
    as_of: int | None = None,
) -> pd.Series:
    """Compute the causal 7-factor equal-weight composite score.

    Args:
        price_history: ``(dates x symbols)`` prices, or ``{symbol: prices}``.
        factor_names: Factor subset; defaults to the engine's seven factors.
        factor_kwargs: Extra panel inputs forwarded to every factor's
            ``compute()`` (e.g. ``{"turnover": turnover_panel}`` for
            ``turnover_20d``); factors ignore what they do not need.
        min_periods: Minimum rows before any score is produced.
        history_limit: Use only the trailing N rows (``None`` disables).
        as_of: Optional positional decision row.  Scores are computed using
            rows ``0..as_of`` only; defaults to the last row.  This exists so
            callers and tests can evaluate an earlier decision date without
            handing the function future rows at all.

    Returns:
        ``pd.Series`` indexed by every input symbol, name
        ``"composite_score"``, values on the [0, 1] percentile scale; NaN for
        symbols without enough history or without a usable factor value.
    """
    symbols, prices = _price_frame(price_history, min_periods)
    nan_scores = pd.Series(
        np.nan, index=pd.Index(symbols), dtype=float, name="composite_score",
    )
    if prices is None or prices.empty:
        return nan_scores

    if as_of is not None:
        if as_of < 0:
            raise ValueError(f"as_of must be >= 0, got {as_of}")
        prices = prices.iloc[: as_of + 1]
    if len(prices) < min_periods:
        return nan_scores
    if history_limit is not None:
        if history_limit <= 0:
            raise ValueError(f"history_limit must be positive, got {history_limit}")
        if len(prices) > history_limit:
            prices = prices.iloc[-history_limit:]

    # Enforce "insufficient history -> excluded" on DataFrame inputs too: a
    # column with fewer than min_periods finite observations is left out of
    # the cross-section entirely (so it cannot shift the others' z-scores or
    # ranks) and comes back NaN.
    finite_counts = np.isfinite(
        prices.to_numpy(dtype=float)
    ).sum(axis=0)
    usable = [
        column for column, count in zip(prices.columns, finite_counts, strict=True)
        if int(count) >= min_periods
    ]
    if not usable:
        return nan_scores
    prices = prices[usable]

    processed: dict[str, pd.DataFrame] = {}
    extra_inputs = dict(factor_kwargs or {})
    for factor in _factor_instances(
        SIGNAL_FACTORS if factor_names is None else factor_names
    ):
        try:
            values = factor.compute(prices, **extra_inputs)
            values = values.reindex(index=prices.index, columns=prices.columns)
            processed[factor.name] = process_factor(values)
        except Exception as exc:  # one broken factor must not kill the score
            logger.debug("Composite factor %s failed: %s", factor.name, exc)

    if not processed:
        return nan_scores

    try:
        combined = combine_equal_weight(processed)
    except ValueError as exc:
        logger.warning("Composite combination failed: %s", exc)
        return nan_scores

    # Mirror AlphaPipeline's equal-weight branch: round first so float
    # summation order cannot flip near-tied assets, rank per date, center,
    # then min-max the decision row onto [0, 1].
    combined = combined.round(12)
    ranked = combined.rank(axis=1, pct=True, na_option="keep") - 0.5
    latest = ranked.iloc[-1].dropna()
    if latest.empty:
        return nan_scores

    floor = float(latest.min())
    spread = float(latest.max()) - floor
    scores = (latest - floor) / max(spread, 1e-8)
    scores = scores.reindex(symbols).astype(float)
    scores.name = "composite_score"
    return scores


def rank_symbols(
    scores: pd.Series | Mapping[str, float],
    *,
    limit: int | None = None,
    min_score: float | None = None,
) -> list[tuple[str, float]]:
    """Rank symbols by score descending with a deterministic tie-break.

    Ties are broken by symbol in ascending order, so both entry points pick
    the same symbols in the same order when scores collide.  NaN/non-finite
    scores (insufficient history) are excluded before the threshold and the
    ``limit`` are applied.
    """
    ranked: list[tuple[str, float]] = []
    for symbol, score in scores.items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        ranked.append((str(symbol), value))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    if min_score is not None:
        ranked = [item for item in ranked if item[1] >= min_score]
    if limit is not None:
        ranked = ranked[: max(int(limit), 0)]
    return ranked
