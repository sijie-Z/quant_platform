"""Factor IC monitoring and decay detection.

Rolling computation of factor Information Coefficient (IC) and ICIR.
Detects when factors lose predictive power and alerts for rebalancing.

Key metrics:
- Rolling IC: correlation between factor values and forward returns
- ICIR: IC / std(IC), measures consistency of predictive power
- IC decay rate: how fast IC is declining
- Factor half-life: estimated time until IC crosses zero
- Turnover: how much factor weights change over time

Used to:
- Automatically down-weight stale factors
- Trigger factor model retraining
- Generate risk alerts for factor crowding
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from quant_platform.factors.evaluation import rank_ic
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FactorICStats:
    """IC statistics for a single factor."""
    name: str
    current_ic: float = 0.0
    current_icir: float = 0.0
    rolling_mean_ic: float = 0.0
    rolling_std_ic: float = 0.0
    rolling_icir: float = 0.0
    ic_trend: float = 0.0            # Slope of IC over time (negative = decaying)
    ic_decay_rate: float = 0.0       # Rate of IC decline (units/day)
    half_life_days: int = 0          # Estimated days until IC = 0
    ic_positive_ratio: float = 0.0   # Fraction of days with positive IC
    last_significant_date: str = ""  # Last date IC was statistically significant
    alert_level: str = "green"       # green/yellow/red


@dataclass
class ICMonitorConfig:
    """Configuration for IC monitoring."""
    rolling_window: int = 63           # Rolling IC window (~3 months)
    icir_window: int = 252             # ICIR window (~1 year)
    decay_window: int = 126            # Window for decay detection (~6 months)
    significance_threshold: float = 0.03  # IC > this is "significant"
    decay_alert_threshold: float = -0.001  # IC slope < this triggers alert
    half_life_alert: int = 126         # Alert if half-life < this
    min_observations: int = 63         # Minimum data points for valid IC


class FactorICMonitor:
    """Monitors factor IC/ICIR over time and detects decay.

    Usage:
        monitor = FactorICMonitor()
        stats = monitor.compute_all(factors, forward_returns)
        alerts = monitor.get_alerts()
    """

    def __init__(self, config: ICMonitorConfig | None = None):
        self.config = config or ICMonitorConfig()
        self.factor_stats: dict[str, FactorICStats] = {}
        self.ic_history: dict[str, pd.Series] = {}

    def compute_rolling_ic(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: int = 63,
    ) -> pd.Series:
        """Compute rolling Rank IC between factor and forward returns.

        Args:
            factor: (date x asset) factor values
            forward_returns: (date x asset) forward returns
            window: rolling window size

        Returns:
            Series of daily IC values
        """
        common_dates = factor.index.intersection(forward_returns.index)
        if len(common_dates) < window:
            return pd.Series(dtype=float)

        ic_values = []
        ic_dates = []

        for i in range(window, len(common_dates)):
            date = common_dates[i]
            window_dates = common_dates[i - window:i]

            # Get factor values and returns for the window
            factor_window = factor.loc[window_dates]
            returns_window = forward_returns.loc[window_dates]

            # Compute cross-sectional rank IC for each date, then average
            daily_ics = []
            for d in window_dates:
                f_vals = factor.loc[d].dropna()
                r_vals = returns_window.loc[d].dropna()
                common = f_vals.index.intersection(r_vals.index)
                if len(common) > 10:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ic = f_vals[common].corr(r_vals[common], method="spearman")
                    if not np.isnan(ic):
                        daily_ics.append(ic)

            if daily_ics:
                ic_values.append(float(np.mean(daily_ics)))
                ic_dates.append(date)

        return pd.Series(ic_values, index=ic_dates, name="ic")

    def compute_factor_stats(
        self,
        factor_name: str,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> FactorICStats:
        """Compute comprehensive IC statistics for a single factor."""
        ic_series = self.compute_rolling_ic(
            factor, forward_returns, self.config.rolling_window
        )

        if len(ic_series) < self.config.min_observations:
            return FactorICStats(name=factor_name)

        self.ic_history[factor_name] = ic_series

        # Current IC (most recent)
        current_ic = float(ic_series.iloc[-1])

        # Rolling statistics
        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std())
        icir = mean_ic / std_ic if std_ic > 0 else 0

        # IC trend (linear regression slope)
        x = np.arange(len(ic_series))
        if len(x) > 10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope = float(np.polyfit(x, ic_series.values, 1)[0])
        else:
            slope = 0

        # Decay rate (slope of recent IC)
        recent = ic_series.iloc[-self.config.decay_window:]
        if len(recent) > 10:
            x_recent = np.arange(len(recent))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decay_rate = float(np.polyfit(x_recent, recent.values, 1)[0])
        else:
            decay_rate = 0

        # Half-life estimate
        if decay_rate < 0 and current_ic > 0:
            half_life = int(abs(current_ic / decay_rate)) if decay_rate != 0 else 9999
        elif current_ic <= 0:
            half_life = 0  # Already below zero
        else:
            half_life = 9999  # Not decaying

        # IC positive ratio
        positive_ratio = float((ic_series > 0).mean())

        # Last significant date
        significant = ic_series[ic_series.abs() > self.config.significance_threshold]
        last_sig = str(significant.index[-1]) if len(significant) > 0 else ""

        # Alert level
        alert = "green"
        if current_ic < 0 or (decay_rate < self.config.decay_alert_threshold and half_life < self.config.half_life_alert):
            alert = "red"
        elif decay_rate < self.config.decay_alert_threshold / 2:
            alert = "yellow"

        stats = FactorICStats(
            name=factor_name,
            current_ic=current_ic,
            current_icir=icir,
            rolling_mean_ic=mean_ic,
            rolling_std_ic=std_ic,
            rolling_icir=mean_ic / std_ic if std_ic > 0 else 0,
            ic_trend=slope,
            ic_decay_rate=decay_rate,
            half_life_days=half_life,
            ic_positive_ratio=positive_ratio,
            last_significant_date=last_sig,
            alert_level=alert,
        )

        self.factor_stats[factor_name] = stats
        return stats

    def compute_all(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> dict[str, FactorICStats]:
        """Compute IC stats for all factors."""
        self.factor_stats = {}
        for name, factor in factors.items():
            self.compute_factor_stats(name, factor, forward_returns)
        return self.factor_stats

    def get_alerts(self) -> list[dict]:
        """Get factors with IC decay alerts."""
        alerts = []
        for name, stats in self.factor_stats.items():
            if stats.alert_level in ("red", "yellow"):
                alerts.append({
                    "factor": name,
                    "alert_level": stats.alert_level,
                    "current_ic": stats.current_ic,
                    "ic_decay_rate": stats.ic_decay_rate,
                    "half_life_days": stats.half_life_days,
                    "message": (
                        f"{name}: IC={stats.current_ic:.4f}, "
                        f"decay={stats.ic_decay_rate:.6f}/day, "
                        f"half_life={stats.half_life_days}d"
                    ),
                })
        return alerts

    def get_adaptive_weights(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        base_method: str = "icir",
        decay_penalty: float = 0.5,
    ) -> dict[str, float]:
        """Compute adaptive factor weights that penalize decaying factors.

        Combines base ICIR weighting with decay penalty:
        - Healthy factor (no decay): weight ∝ ICIR
        - Decaying factor: weight ∝ ICIR * decay_penalty_factor

        Args:
            factors: factor data
            forward_returns: forward returns
            base_method: "icir" or "equal"
            decay_penalty: how much to penalize decaying factors (0-1)

        Returns:
            dict of factor_name -> weight (normalized to sum to 1)
        """
        # Compute stats if not done
        if not self.factor_stats:
            self.compute_all(factors, forward_returns)

        raw_weights = {}
        for name, stats in self.factor_stats.items():
            if base_method == "icir":
                base_weight = max(0, stats.rolling_icir)
            else:
                base_weight = 1.0

            # Decay penalty
            if stats.alert_level == "red":
                penalty = decay_penalty
            elif stats.alert_level == "yellow":
                penalty = (1 + decay_penalty) / 2
            else:
                penalty = 1.0

            raw_weights[name] = base_weight * penalty

        # Normalize
        total = sum(raw_weights.values())
        if total > 0:
            return {k: v / total for k, v in raw_weights.items()}
        else:
            n = len(raw_weights)
            return {k: 1.0 / n for k in raw_weights}

    def get_summary(self) -> list[dict]:
        """Get summary table of all factor IC stats."""
        return [
            {
                "factor": stats.name,
                "current_ic": round(stats.current_ic, 4),
                "rolling_icir": round(stats.rolling_icir, 3),
                "ic_trend": round(stats.ic_trend, 6),
                "decay_rate": round(stats.ic_decay_rate, 6),
                "half_life": stats.half_life_days,
                "positive_ratio": round(stats.ic_positive_ratio, 3),
                "alert": stats.alert_level,
            }
            for stats in sorted(
                self.factor_stats.values(),
                key=lambda s: abs(s.rolling_icir),
                reverse=True,
            )
        ]
