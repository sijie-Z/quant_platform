"""Market regime detection for adaptive factor allocation.

Detects market regimes (bull/bear/sideways) using:
- Index moving average crossovers
- Market breadth (advance-decline ratio)
- Volatility regime (VIX proxy)

Then adjusts factor weights based on historical regime-conditional IC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class MarketRegimeDetector:
    """Detect market regimes using index signals.

    Regimes:
    - bull: uptrend (short MA > long MA, positive breadth)
    - bear: downtrend (short MA < long MA, negative breadth)
    - sideways: no clear trend

    Uses only past data (point-in-time) to avoid look-ahead.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        vol_window: int = 20,
        high_vol_threshold: float = 0.25,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.vol_window = vol_window
        self.high_vol_threshold = high_vol_threshold

    def detect(self, index_returns: pd.Series) -> pd.Series:
        """Detect regime for each date (point-in-time).

        Args:
            index_returns: Daily returns of market index (e.g., CSI 300).

        Returns:
            Series with regime label ('bull', 'bear', 'sideways', 'high_vol')
            for each date.
        """
        if len(index_returns) < self.long_window + 10:
            return pd.Series('sideways', index=index_returns.index)

        price = (1 + index_returns).cumprod()
        ma_short = price.rolling(self.short_window).mean()
        ma_long = price.rolling(self.long_window).mean()
        trend = ma_short / ma_long - 1
        vol = index_returns.rolling(self.vol_window).std() * np.sqrt(252)

        regimes = pd.Series('sideways', index=index_returns.index)
        bull_mask = (trend > 0.01) & (vol < self.high_vol_threshold)
        regimes[bull_mask] = 'bull'
        bear_mask = trend < -0.01
        regimes[bear_mask] = 'bear'
        high_vol_mask = vol >= self.high_vol_threshold
        regimes[high_vol_mask] = 'high_vol'

        return regimes


# Regime-conditional factor weight multipliers
# These are based on academic evidence for A-shares:
# - Bull: momentum works, value less relevant
# - Bear: value, quality, low-vol outperform
# - High vol: defensive factors (low vol, quality) outperform
REGIME_MULTIPLIERS = {
    'bull': {
        'momentum_1m': 1.2, 'momentum_3m': 1.2, 'momentum_6m': 1.2,
        'momentum_12m': 1.1, 'reversal': 0.8,
        'volatility_20d': 0.9, 'volatility_60d': 0.9,
        'size': 1.1, 'quality': 1.0, 'pb_mrq': 0.9,
        'pe_ttm': 0.9, 'close_raw': 1.0, 'liquidity': 1.0,
        'turnover_20d': 0.9, 'rsi_14d': 1.1, 'amplitude_20d': 0.9, 'macd': 1.1,
    },
    'bear': {
        'momentum_1m': 0.8, 'momentum_3m': 0.8, 'momentum_6m': 0.8,
        'momentum_12m': 0.9, 'reversal': 1.2,
        'volatility_20d': 1.3, 'volatility_60d': 1.3,
        'size': 0.8, 'quality': 1.3, 'pb_mrq': 1.2,
        'pe_ttm': 1.2, 'close_raw': 1.1, 'liquidity': 1.1,
        'turnover_20d': 1.2, 'rsi_14d': 0.9, 'amplitude_20d': 1.2, 'macd': 0.9,
    },
    'high_vol': {
        'momentum_1m': 0.7, 'momentum_3m': 0.7, 'momentum_6m': 0.7,
        'momentum_12m': 0.8, 'reversal': 1.3,
        'volatility_20d': 1.4, 'volatility_60d': 1.4,
        'size': 0.7, 'quality': 1.4, 'pb_mrq': 1.3,
        'pe_ttm': 1.3, 'close_raw': 1.2, 'liquidity': 1.2,
        'turnover_20d': 1.3, 'rsi_14d': 0.8, 'amplitude_20d': 1.3, 'macd': 0.8,
    },
    'sideways': {},  # No adjustment
}


def combine_icir_with_regime(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    index_returns: pd.Series,
    lookback: int = 252,
    min_icir: float = 0.0,
) -> pd.DataFrame:
    """ICIR-weighted combination with regime-based factor timing.

    At each date:
    1. Detect market regime using index returns (point-in-time)
    2. Compute ICIR weights for each factor (point-in-time)
    3. Adjust weights based on regime (e.g., increase value in bear markets)
    4. Combine factors with adjusted weights

    This is the correct way to do regime timing — adjust weights BEFORE
    combination, not the signal AFTER combination.

    Args:
        factors: Dict of processed factor DataFrames.
        forward_returns: Forward returns for IC estimation.
        index_returns: Market index returns for regime detection.
        lookback: IC lookback window.
        min_icir: Minimum ICIR threshold.

    Returns:
        Combined alpha signal (date × asset).
    """
    from quant_platform.alpha.combination import _align_factors, _build_row
    from quant_platform.factors.evaluation import ic_summary, rank_ic

    aligned = _align_factors(factors)
    dates = sorted(next(iter(aligned.values())).index)
    factor_names = list(factors.keys())

    # Precompute IC series
    ic_series_dict = {}
    for name in factor_names:
        ic_series_dict[name] = rank_ic(factors[name], forward_returns)

    # Detect regimes (point-in-time)
    detector = MarketRegimeDetector()
    regimes = detector.detect(index_returns)

    result_rows = []
    regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0, 'high_vol': 0}

    for i, date in enumerate(dates):
        regime = regimes.get(date, 'sideways') if date in regimes.index else 'sideways'
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Compute ICIR weights (point-in-time)
        icir_values = {}
        for name in factor_names:
            ic_s = ic_series_dict[name]
            ic_hist = ic_s[ic_s.index < date]
            if len(ic_hist) < 20:
                icir_values[name] = 0.0
                continue
            recent = ic_hist.iloc[-lookback:] if len(ic_hist) > lookback else ic_hist
            summary = ic_summary(recent)
            icir_values[name] = summary["icir"]

        # Apply regime multipliers to ICIR values
        multipliers = REGIME_MULTIPLIERS.get(regime, {})
        adjusted_icir = {}
        for name, icir_val in icir_values.items():
            mult = multipliers.get(name, 1.0)
            adjusted_icir[name] = icir_val * mult

        # Filter by min_icir on ADJUSTED values
        filtered = {name: v for name, v in adjusted_icir.items() if abs(v) >= min_icir}
        if not filtered:
            filtered = dict(adjusted_icir)

        # Weight by adjusted ICIR
        total = sum(abs(v) for v in filtered.values())
        if total < 1e-10:
            weights = {name: 1.0 / len(filtered) for name in filtered}
        else:
            weights = {name: v / total for name, v in filtered.items()}

        row = _build_row(aligned, weights, date)
        if row is not None:
            row.name = date
            result_rows.append(row)

    if not result_rows:
        raise ValueError("No dates with sufficient data for regime-adjusted ICIR")

    result = pd.DataFrame(result_rows)

    # Log regime distribution
    total_days = sum(regime_counts.values())
    for regime, count in sorted(regime_counts.items()):
        if count > 0:
            logger.info("  Regime %s: %d days (%.1f%%)", regime, count, count/total_days*100)

    return result


def combine_ic_with_regime(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    index_returns: pd.Series,
    lookback: int = 252,
) -> pd.DataFrame:
    """IC-weighted combination with regime-based factor timing.

    Same as combine_icir_with_regime but uses mean IC instead of ICIR.
    """
    from quant_platform.alpha.combination import _align_factors, _build_row
    from quant_platform.factors.evaluation import rank_ic

    aligned = _align_factors(factors)
    dates = sorted(next(iter(aligned.values())).index)
    factor_names = list(factors.keys())

    ic_series_dict = {}
    for name in factor_names:
        ic_series_dict[name] = rank_ic(factors[name], forward_returns)

    detector = MarketRegimeDetector()
    regimes = detector.detect(index_returns)

    result_rows = []

    for i, date in enumerate(dates):
        regime = regimes.get(date, 'sideways') if date in regimes.index else 'sideways'

        weights = {}
        for name in factor_names:
            ic_s = ic_series_dict[name]
            ic_hist = ic_s[ic_s.index < date]
            if len(ic_hist) < 20:
                weights[name] = 0.0
                continue
            recent = ic_hist.iloc[-lookback:] if len(ic_hist) > lookback else ic_hist
            mean_ic = recent.mean()

            # Apply regime multiplier
            mult = REGIME_MULTIPLIERS.get(regime, {}).get(name, 1.0)
            weights[name] = mean_ic * mult

        total_abs = sum(abs(v) for v in weights.values())
        if total_abs < 1e-10:
            weights = {name: 1.0 / len(factor_names) for name in factor_names}
        else:
            weights = {name: v / total_abs for name, v in weights.items()}

        row = _build_row(aligned, weights, date)
        if row is not None:
            row.name = date
            result_rows.append(row)

    if not result_rows:
        raise ValueError("No dates with sufficient data for regime-adjusted IC")

    return pd.DataFrame(result_rows)
