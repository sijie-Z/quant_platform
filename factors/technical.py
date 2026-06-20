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
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


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
        turnover = kwargs.get("turnover")
        if turnover is not None:
            return turnover.rolling(self._period).mean()
        logger.warning("turnover data not provided, using price SMA as proxy")
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
# Efficiency Ratio factor  (inspired by KF Timing App)
# ---------------------------------------------------------------------------

class EfficiencyRatioFactor(BaseFactor):
    """Efficiency Ratio: trend quality measure from KF Timing App.

    ER = |return(period)| / sum(|daily_return|) over period

    Values in [0, 1]:
    - 1.0 = perfect trend (price goes straight up or down)
    - 0.0 = pure noise (price ends where it started, high total path)

    This measures "trend quality" vs our momentum factors which measure
    "trend magnitude". A high-ER stock has smooth persistent movement.
    A low-ER stock is choppy even if total return is similar.
    """

    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 20, name: str = "efficiency_ratio"):
        super().__init__({"period": period})
        self._period = period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        log_prices = np.log(prices.clip(lower=1e-8))

        # Net direction over period
        direction = log_prices.diff(self._period).abs()

        # Total path length (sum of absolute daily differences)
        daily_diff = log_prices.diff().abs()
        volatility = daily_diff.rolling(self._period).sum()

        er = direction / volatility.replace(0, np.nan)
        return er.clip(0, 1)


# ---------------------------------------------------------------------------
# Breakout Ignition factor  (inspired by KF Timing App)
# ---------------------------------------------------------------------------

class BreakoutIgnitionFactor(BaseFactor):
    """Breakout Ignition: simultaneous return shock + volume shock.

    Signals stocks experiencing an abnormal price move accompanied by
    abnormal volume — classic breakout pattern.

    The factor is 1 (ignition detected) when:
    1. Return z-score over past N days > threshold (return shock)
    2. Recent volume / average volume > threshold (volume shock)

    This is a composite signal factor, not a rankable continuous factor.
    """

    category = FactorCategory.TECHNICAL

    def __init__(
        self,
        return_window: int = 3,
        reference_window: int = 20,
        volume_short_window: int = 3,
        volume_long_window: int = 20,
        return_z_threshold: float = 1.5,
        volume_ratio_threshold: float = 1.5,
        name: str = "breakout_ignition",
    ):
        super().__init__({
            "return_window": return_window,
            "reference_window": reference_window,
            "volume_short_window": volume_short_window,
            "volume_long_window": volume_long_window,
            "return_z_threshold": return_z_threshold,
            "volume_ratio_threshold": volume_ratio_threshold,
        })
        self._ret_win = return_window
        self._ref_win = reference_window
        self._vol_short = volume_short_window
        self._vol_long = volume_long_window
        self._ret_z = return_z_threshold
        self._vol_ratio = volume_ratio_threshold
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        turnover_data = kwargs.get("turnover")
        volume: pd.DataFrame | None = None

        if turnover_data is not None:
            volume = turnover_data
        else:
            # Use price change magnitude as volume proxy
            volume = prices.pct_change(fill_method=None).abs()

        # Return shock: z-score of k-period return vs history
        ret = prices.pct_change(fill_method=None)
        k_ret = ret.rolling(self._ret_win).apply(
            lambda x: (1 + x).prod() - 1 if len(x) == self._ret_win else 0
        )
        ref_mean = k_ret.rolling(self._ref_win).mean()
        ref_std = k_ret.rolling(self._ref_win).std().replace(0, np.nan)
        ret_z = (k_ret - ref_mean) / ref_std
        ret_shock = ret_z.abs() >= self._ret_z

        # Volume shock: recent / baseline
        vol_short_ma = volume.rolling(self._vol_short).mean()
        vol_long_ma = volume.rolling(self._vol_long).mean()
        vol_ratio = vol_short_ma / vol_long_ma.replace(0, np.nan)
        vol_shock = vol_ratio >= self._vol_ratio

        # Composite: both conditions must hold
        ignition = (ret_shock & vol_shock).astype(float)
        return ignition


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



# ---------------------------------------------------------------------------
# K-Line candlestick features  (inspired by Qlib Alpha158)
# ---------------------------------------------------------------------------


class CandleMidpointFactor(BaseFactor):
    """KMID: (close - open) / open. Positive = green candle."""
    category = FactorCategory.TECHNICAL

    @property
    def name(self) -> str:
        return "kmid"

    def compute(self, prices, **kwargs):
        o = kwargs.get("open")
        c = prices
        if o is not None:
            return (c - o) / o.replace(0, float("nan"))
        return prices.pct_change(fill_method=None)


class CandleLengthFactor(BaseFactor):
    """KLEN: (high - low) / open. Intraday range."""
    category = FactorCategory.TECHNICAL

    @property
    def name(self) -> str:
        return "klen"

    def compute(self, prices, **kwargs):
        h = kwargs.get("high")
        l = kwargs.get("low")
        o = kwargs.get("open")
        if h is not None and l is not None and o is not None:
            return (h - l) / o.replace(0, float("nan"))
        return prices.pct_change(fill_method=None).abs()


class CandleUpperShadowFactor(BaseFactor):
    """KUP: (high - max(open, close)) / open. Upper wick."""
    category = FactorCategory.TECHNICAL

    @property
    def name(self) -> str:
        return "kup"

    def compute(self, prices, **kwargs):
        h = kwargs.get("high")
        o = kwargs.get("open")
        c = prices
        if h is not None and o is not None:
            import numpy as np
            upper = h - np.maximum(o.values, c.values)
            return pd.DataFrame(upper, index=prices.index, columns=prices.columns) / o.replace(0, float("nan"))
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


class CandleLowerShadowFactor(BaseFactor):
    """KLOW: (min(open, close) - low) / open. Lower wick."""
    category = FactorCategory.TECHNICAL

    @property
    def name(self) -> str:
        return "klow"

    def compute(self, prices, **kwargs):
        l = kwargs.get("low")
        o = kwargs.get("open")
        c = prices
        if l is not None and o is not None:
            import numpy as np
            lower = np.minimum(o.values, c.values) - l.values if hasattr(l, 'values') else (np.minimum(o, c) - l)
            return pd.DataFrame(lower, index=prices.index, columns=prices.columns) / o.replace(0, float("nan"))
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


class CandleSoftnessFactor(BaseFactor):
    """KSFT: (2*close - high - low) / open. Close in range."""
    category = FactorCategory.TECHNICAL

    @property
    def name(self) -> str:
        return "ksft"

    def compute(self, prices, **kwargs):
        h = kwargs.get("high")
        l = kwargs.get("low")
        o = kwargs.get("open")
        c = prices
        if h is not None and l is not None and o is not None:
            return (2 * c - h - l) / o.replace(0, float("nan"))
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


# ---------------------------------------------------------------------------
# MA Convergence factor  (inspired by a-share-hybrid-strategy)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Trend Stage factor  (inspired by a-share-trend-strategy)
# ---------------------------------------------------------------------------


class TrendStageFactor(BaseFactor):
    """Trend stage: where price sits in its historical range.
    Maps to 'fish head/body/tail' trend stages:
      0-30:   Early stage (鱼头)
      30-70:  Mid stage (鱼身)
      70-100: Late stage (鱼尾)
    """
    category = FactorCategory.TECHNICAL
    def __init__(self, period: int = 120, name: str = "trend_stage"):
        super().__init__({'period': period})
        self._period = period
        self._name = name
    @property
    def name(self) -> str:
        return self._name
    def compute(self, prices, **kwargs):
        high = prices.rolling(self._period).max()
        low = prices.rolling(self._period).min()
        rng = high - low
        stage = (prices - low) / rng.replace(0, float("nan")) * 100
        return stage.clip(0, 100)


class MAConvergenceFactor(BaseFactor):
    """MA convergence: how tightly MA5/MA10/MA20 are clustered.

    When moving averages are bunched closely together (<1.5% gap), it
    signals a coiling pattern — the stock is building energy for a
    breakout. High convergence + low volatility = impending move.

    Score [0, 1] where:
      1.0 = extremely tight (all MAs within 1.5%)
      0.0 = wide separation (gap > 5%)
    """
    category = FactorCategory.TECHNICAL

    @property
    def name(self) -> str:
        return "ma_convergence"

    def compute(self, prices, **kwargs):
        ma5 = prices.rolling(5).mean()
        ma10 = prices.rolling(10).mean()
        ma20 = prices.rolling(20).mean()

        gap1 = (ma5 - ma10).abs() / ma10 * 100
        gap2 = (ma10 - ma20).abs() / ma20 * 100
        max_gap = pd.concat([gap1, gap2], axis=1).max(axis=1)

        # Convert gap % to score: 0% gap = 1.0, >=5% gap = 0.0
        score = 1.0 - (max_gap.clip(0, 5) / 5.0)
        return score


class BreakoutProximityFactor(BaseFactor):
    """Breakout proximity: distance from current price to N-day high.

    Measures how close price is to breaking out of its range.
    Value [0, 1] where:
      1.0 = price is AT or above the N-day high (imminent/actual breakout)
      0.5 = price is halfway between the low and high of the range
      0.0 = price is at the N-day low

    When this is high (>0.9) AND MA convergence is also high, it's a
    strong breakout signal.
    """
    category = FactorCategory.TECHNICAL

    def __init__(self, period: int = 20, name: str = "breakout_proximity"):
        super().__init__({'period': period})
        self._period = period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices, **kwargs):
        high = prices.rolling(self._period).max()
        low = prices.rolling(self._period).min()
        rng = high - low
        proximity = (prices - low) / rng.replace(0, float("nan"))
        return proximity.clip(0, 1)



# ---------------------------------------------------------------------------
# Pure Volatility Factor  (inspired by Dongwu Securities research)
# ---------------------------------------------------------------------------


class PureVolatilityFactor(BaseFactor):
    """Pure idiosyncratic volatility: FF3 residuals orthogonalized.

    Methodology (from Dongwu Securities research report):
    1. Rolling 20-day FF3 regression → residual returns
    2. Idiosyncratic vol = std(residual) over 20 days
    3. Orthogonalize against turnover → remove trading-activity noise
    4. AR(30) filter → remove serial correlation in factor values

    The resulting 'pure volatility' factor isolates the fundamental
    component of idiosyncratic risk, removing both trading noise
    and cross-period information leakage.

    Higher values = higher idiosyncratic risk = expected lower returns
    (the IVOL anomaly: high IVOL stocks tend to underperform).
    """
    category = FactorCategory.TECHNICAL

    def __init__(self, window: int = 20, ar_lags: int = 30, name: str = "pure_volatility"):
        super().__init__({'window': window, 'ar_lags': ar_lags})
        self._window = window
        self._ar_lags = ar_lags
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)

        # Rolling 20-day FF3 regression per asset
        # FF3 factors: market (excess return), SMB, HML
        # For simplicity, use market return (equal-weight cross-section)
        # as the single factor. Full FF3 would need market cap data.
        market_ret = ret.mean(axis=1)

        # Rolling regression: ret_i = alpha + beta * market_ret + epsilon
        # Then IVOL = std(epsilon) over the window
        def _rolling_ivol(x):
            y = x.values
            m = market_ret.reindex(x.index).values
            if len(y) < self._window or np.std(m) < 1e-10:
                return np.nan
            # Simple OLS: beta = cov(y,m) / var(m)
            beta = np.cov(y, m)[0, 1] / np.var(m)
            resid = y - beta * m
            return float(np.std(resid, ddof=2))

        ivol = ret.rolling(self._window).apply(_rolling_ivol, raw=False)

        # Orthogonalize against turnover (if available)
        turnover = kwargs.get('turnover')
        if turnover is not None and not turnover.empty:
            # Regress IVOL ~ turnover per date cross-section
            # Return residuals = turnover-orthogonalized IVOL
            aligned = ivol.align(turnover, join='inner')
            ivol_aligned = aligned[0]
            turn_aligned = aligned[1]

            result = ivol_aligned.copy()
            for date in ivol_aligned.index:
                row_ivol = ivol_aligned.loc[date].dropna()
                row_turn = turn_aligned.loc[date].reindex(row_ivol.index).dropna()
                common = row_ivol.index.intersection(row_turn.index)
                if len(common) < 10:
                    continue
                y = row_ivol[common].values
                x = row_turn[common].values
                if np.std(x) < 1e-10:
                    continue
                beta = np.cov(y, x)[0, 1] / np.var(x)
                resid = y - beta * x
                result.loc[date, common] = resid

            ivol = result

        # AR(lags) filter to remove serial correlation
        # Fit AR model per asset and return residuals
        result = ivol.copy()
        for asset in ivol.columns:
            series = ivol[asset].dropna().values
            if len(series) < self._ar_lags + 10:
                continue
            # Simple AR fit: X_t = sum(phi_i * X_{t-i}) + epsilon
            # Using least squares
            T = len(series)
            X = np.column_stack([
                series[self._ar_lags - 1 - i: T - 1 - i]
                for i in range(self._ar_lags)
            ])
            y = series[self._ar_lags:]
            if X.shape[0] < self._ar_lags + 5:
                continue
            try:
                phi = np.linalg.lstsq(X, y, rcond=None)[0]
                pred = X @ phi
                resid_ar = y - pred
                result.loc[ivol.index[ivol.notna().any(axis=1)][self._ar_lags:], asset] = resid_ar
            except np.linalg.LinAlgError:
                continue

        return result



# ---------------------------------------------------------------------------
# Multi-timeframe resonance factor  (inspired by a-share-signal)
# ---------------------------------------------------------------------------


class MultiTimeframeResonanceFactor(BaseFactor):
    """Multi-timeframe resonance: trend alignment across daily/weekly/monthly.

    Checks whether price trends across 3 timeframes are aligned.
    High resonance = daily, weekly, and monthly trends all point same direction.
    Low resonance = timeframes conflict, choppy price action expected.

    Score [0, 1]:
      1.0 = perfect alignment (all 3 timeframes bullish or bearish)
      0.0 = complete conflict (daily and monthly opposite)
    """
    category = FactorCategory.TECHNICAL

    def __init__(self, name: str = "mtf_resonance"):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices, **kwargs):
        # Daily trend: 20-day MA slope
        daily_ma = prices.rolling(20).mean()
        daily_slope = (daily_ma - daily_ma.shift(5)) / daily_ma.shift(5).replace(0, float("nan"))

        # Weekly trend: approximate via 60-day MA slope
        weekly_ma = prices.rolling(60).mean()
        weekly_slope = (weekly_ma - weekly_ma.shift(15)) / weekly_ma.shift(15).replace(0, float("nan"))

        # Monthly trend: approximate via 120-day MA slope
        monthly_ma = prices.rolling(120).mean()
        monthly_slope = (monthly_ma - monthly_ma.shift(30)) / monthly_ma.shift(30).replace(0, float("nan"))

        # Direction: 1 = bullish, -1 = bearish, 0 = flat
        def _direction(slope: pd.DataFrame) -> pd.DataFrame:
            d = pd.DataFrame(0.0, index=slope.index, columns=slope.columns)
            d[slope > 0.001] = 1.0
            d[slope < -0.001] = -1.0
            return d

        d_dir = _direction(daily_slope)
        w_dir = _direction(weekly_slope)
        m_dir = _direction(monthly_slope)

        # Resonance: how many timeframes agree
        agreement = d_dir + w_dir + m_dir

        # Map to [0, 1] score
        # 3 or -3 = all agree = 1.0
        # 1 or -1 = 2 agree, 1 flat = 0.7
        # 0 = conflict = 0.0
        score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        score[agreement.abs() >= 3] = 1.0
        score[agreement.abs() == 2] = 0.7
        score[agreement.abs() == 1] = 0.3

        return score

def register_all():
    registry = get_registry()
    for cls in [
        Momentum1M, Momentum3M, Momentum6M, Momentum12M,
        Volatility20D, Volatility60D,
        TurnoverFactor, RSIFactor, AmplitudeFactor, MACDFactor,
        EfficiencyRatioFactor, BreakoutIgnitionFactor,
        CandleMidpointFactor, CandleLengthFactor,
        CandleUpperShadowFactor, CandleLowerShadowFactor,
        CandleSoftnessFactor,
        TrendStageFactor,
        MAConvergenceFactor,
        BreakoutProximityFactor,
        PureVolatilityFactor,
        MultiTimeframeResonanceFactor,
    ]:
        registry.register(cls)
