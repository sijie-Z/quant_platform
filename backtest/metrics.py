"""Performance metrics for strategy evaluation.

Computes standard quant finance metrics:
- Annualized return, volatility, Sharpe ratio
- Maximum drawdown, Calmar ratio
- Sortino ratio (downside deviation)
- Information ratio, tracking error (vs benchmark)
- Win rate, profit-loss ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    """Compound annual growth rate."""
    total_return = (1 + daily_returns).prod() - 1
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years < 1 / 252:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.03) -> float:
    """Sharpe ratio = (annualized return - rf) / annualized vol."""
    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - risk_free) / ann_vol


def sortino_ratio(daily_returns: pd.Series, risk_free: float = 0.03) -> float:
    """Sortino ratio: uses downside deviation instead of total vol."""
    ann_ret = annualized_return(daily_returns)
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std < 1e-10:
        return 0.0
    return (ann_ret - risk_free) / downside_std


def max_drawdown(daily_returns: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Maximum drawdown: largest peak-to-trough decline.

    Uses Numba JIT acceleration when available, falling back to Pandas.

    Returns:
        (max_dd, peak_date, trough_date)
    """
    from quant_platform.utils.numba_accelerator import (
        HAS_NUMBA, max_drawdown_numba, max_drawdown_pandas, benchmark,
    )

    if HAS_NUMBA:
        max_dd, peak_idx, trough_idx = max_drawdown_numba(daily_returns)

        # Convert indices back to dates
        cumulative = (1 + daily_returns.dropna()).cumprod()
        dates = cumulative.index
        peak_date = dates[peak_idx]
        trough_date = dates[trough_idx]
        return max_dd, peak_date, trough_date

    # Original Pandas implementation
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = running_max.loc[:trough_date].idxmax()
    return max_dd, peak_date, trough_date


def calmar_ratio(daily_returns: pd.Series) -> float:
    """Calmar ratio = annualized return / max drawdown (absolute value)."""
    ann_ret = annualized_return(daily_returns)
    mdd, _, _ = max_drawdown(daily_returns)
    if abs(mdd) < 1e-10:
        return 0.0
    return ann_ret / abs(mdd)


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Information ratio = mean excess return / std excess return (annualized)."""
    aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    excess = aligned["strategy"] - aligned["benchmark"]
    if len(excess) < 2:
        return 0.0
    ann_excess = excess.mean() * TRADING_DAYS_PER_YEAR
    ann_tracking_error = excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if ann_tracking_error < 1e-10:
        return 0.0
    return ann_excess / ann_tracking_error


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualized tracking error."""
    aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    excess = aligned["strategy"] - aligned["benchmark"]
    return excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def win_rate(daily_returns: pd.Series) -> float:
    """Proportion of days with positive return."""
    return (daily_returns > 0).mean()


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    """Average gain on winning days / average loss on losing days."""
    gains = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    if len(losses) == 0 or len(gains) == 0:
        return 0.0
    avg_gain = gains.mean()
    avg_loss = abs(losses.mean())
    if avg_loss < 1e-10:
        return 0.0
    return avg_gain / avg_loss


def all_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """Compute all performance metrics at once.

    Returns dict with all metrics as scalar values for easy reporting.
    """
    sr = strategy_returns.dropna()
    if len(sr) < 10:
        return {}

    mdd, peak_dt, trough_dt = max_drawdown(sr)

    metrics = {
        "annual_return": annualized_return(sr),
        "annual_volatility": annualized_volatility(sr),
        "sharpe_ratio": sharpe_ratio(sr),
        "sortino_ratio": sortino_ratio(sr),
        "max_drawdown": mdd,
        "max_drawdown_peak": str(peak_dt.date()) if pd.notna(peak_dt) else None,
        "max_drawdown_trough": str(trough_dt.date()) if pd.notna(trough_dt) else None,
        "calmar_ratio": calmar_ratio(sr),
        "win_rate": win_rate(sr),
        "profit_loss_ratio": profit_loss_ratio(sr),
        "total_return": (1 + sr).prod() - 1,
        "total_days": len(sr),
    }

    if benchmark_returns is not None:
        br = benchmark_returns.reindex(sr.index).dropna()
        metrics["information_ratio"] = information_ratio(sr.loc[br.index], br)
        metrics["tracking_error"] = tracking_error(sr.loc[br.index], br)
        metrics["benchmark_total_return"] = (1 + br).prod() - 1
        metrics["excess_return"] = metrics["total_return"] - metrics["benchmark_total_return"]

    return metrics
