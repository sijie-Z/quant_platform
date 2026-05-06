"""Technical factors for A-share multi-factor strategies.

Implements commonly used technical factors:
- Momentum (1M, 3M, 6M, 12M): cumulative return over lookback period
- Volatility (20d, 60d): daily return standard deviation
- Turnover (20d): average daily turnover rate
- RSI (14d): Relative Strength Index
- Amplitude (20d): average daily (high - low) / close
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.factors.base import BaseFactor, FactorCategory
from quant_platform.factors.registry import get_registry


# ---------------------------------------------------------------------------
# Momentum factors
# ---------------------------------------------------------------------------

class MomentumFactor(BaseFactor):
    """Cumulative return over a lookback period, skipping recent days.

    Standard momentum: return from (t - period - skip) to (t - skip).
    Skipping the most recent period avoids the short-term reversal effect.
    """

    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 63, name: str = "momentum", skip: int = 0):
        super().__init__({"period": period, "skip": skip})
        self._period = period
        self._skip = skip
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from quant_platform.utils.numba_accelerator import HAS_NUMBA, momentum_factor_numba

        ret = prices.pct_change(fill_method=None)
        data = ret.shift(self._skip) if self._skip > 0 else ret

        if HAS_NUMBA and data.shape[1] >= 10:
            return momentum_factor_numba(data, self._period)
        return data.rolling(self._period).apply(lambda x: (1 + x).prod() - 1)


class Momentum1M(MomentumFactor):
    def __init__(self):
        super().__init__(period=21, name="momentum_1m", skip=0)


class Momentum3M(MomentumFactor):
    def __init__(self):
        super().__init__(period=63, name="momentum_3m", skip=0)


class Momentum6M(MomentumFactor):
    def __init__(self):
        super().__init__(period=126, name="momentum_6m", skip=0)


class Momentum12M(MomentumFactor):
    def __init__(self):
        super().__init__(period=252, name="momentum_12m", skip=21)


# ---------------------------------------------------------------------------
# Volatility factors
# ---------------------------------------------------------------------------

class VolatilityFactor(BaseFactor):
    """Historical volatility: standard deviation of daily returns."""

    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 20, name: str = "volatility"):
        super().__init__({"period": period})
        self._period = period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        return ret.rolling(self._period).std()


class Volatility20D(VolatilityFactor):
    def __init__(self):
        super().__init__(period=20, name="volatility_20d")


class Volatility60D(VolatilityFactor):
    def __init__(self):
        super().__init__(period=60, name="volatility_60d")


# ---------------------------------------------------------------------------
# Turnover factor
# ---------------------------------------------------------------------------

class TurnoverFactor(BaseFactor):
    """Average daily turnover rate over lookback period.

    Turnover = volume / shares outstanding. High turnover may indicate
    investor attention or liquidity-driven mispricing.
    """

    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 20, name: str = "turnover_20d"):
        super().__init__({"period": period})
        self._period = period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return prices.rolling(self._period).mean()


# ---------------------------------------------------------------------------
# RSI factor
# ---------------------------------------------------------------------------

class RSIFactor(BaseFactor):
    """Relative Strength Index (14-day).

    RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    Values range 0-100. >70 is overbought, <30 is oversold.
    """

    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 14, name: str = "rsi_14d"):
        super().__init__({"period": period})
        self._period = period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(self._period).mean()
        avg_loss = loss.rolling(self._period).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        return rsi


# ---------------------------------------------------------------------------
# Amplitude factor
# ---------------------------------------------------------------------------

class AmplitudeFactor(BaseFactor):
    """Average daily amplitude: (high - low) / close."""

    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 20, name: str = "amplitude_20d"):
        super().__init__({"period": period})
        self._period = period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # prices here should be close prices. If we had high/low we'd use them.
        # For synthetic data compatibility, use daily return magnitude as proxy
        ret = prices.pct_change(fill_method=None).abs()
        return ret.rolling(self._period).mean()


# ---------------------------------------------------------------------------
# MACD factor
# ---------------------------------------------------------------------------

class MACDFactor(BaseFactor):
    """Moving Average Convergence Divergence.

    MACD = EMA(12) - EMA(26). Signal = EMA(9) of MACD.
    Factor value = MACD - Signal (histogram).
    """

    category = FactorCategory.TECHNICAL

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__({"fast": fast, "slow": slow, "signal": signal})
        self._fast = fast
        self._slow = slow
        self._signal = signal

    @property
    def name(self) -> str:
        return "macd"

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ema_fast = prices.ewm(span=self._fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self._slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self._signal, adjust=False).mean()
        return macd_line - signal_line


# ---------------------------------------------------------------------------
# Register all technical factors
# ---------------------------------------------------------------------------

def register_all():
    registry = get_registry()
    for cls in [
        Momentum1M, Momentum3M, Momentum6M, Momentum12M,
        Volatility20D, Volatility60D,
        TurnoverFactor, RSIFactor, AmplitudeFactor, MACDFactor,
    ]:
        registry.register(cls)
