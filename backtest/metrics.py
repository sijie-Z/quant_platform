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
    """Sortino ratio: excess return over downside deviation.

    Downside deviation is ``sqrt(mean(min(r - MAR, 0)^2))`` over *all*
    observations. Taking the standard deviation of only the negative returns
    (the old implementation) measures their dispersion around their own mean,
    returns 0 for constant-sized losses, and ignores how often losses occur.
    """
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return 0.0
    ann_ret = annualized_return(daily_returns)
    daily_mar = risk_free / TRADING_DAYS_PER_YEAR
    downside = np.minimum(daily_returns.to_numpy() - daily_mar, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_dev < 1e-10:
        return 0.0
    return (ann_ret - risk_free) / downside_dev


def max_drawdown(daily_returns: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Maximum drawdown: largest peak-to-trough decline, measured from inception.

    The wealth path is prepended with 1.0 before the peak/trough scan: starting
    at the first *return* made a first-day loss invisible, so a series
    ``[-10%, -10%]`` reported -10% instead of the true -19%.

    Returns:
        (max_dd, peak_date, trough_date)
    """
    from quant_platform.utils.numba_accelerator import (
        HAS_NUMBA,
        max_drawdown_numba,
    )

    values = daily_returns.dropna()
    if len(values) == 0:
        return 0.0, pd.NaT, pd.NaT

    if HAS_NUMBA:
        max_dd, peak_idx, trough_idx = max_drawdown_numba(values)
    else:
        cumulative = np.concatenate(([1.0], (1 + values).cumprod().to_numpy()))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())
        trough_idx = int(drawdown.argmin())
        peak_idx = int(running_max[:trough_idx + 1].argmax())

    # Index 0 is inception; index k (k >= 1) is the k-th return.
    last = len(values) - 1
    peak_date = values.index[min(max(peak_idx - 1, 0), last)]
    trough_date = values.index[min(max(trough_idx - 1, 0), last)]
    return float(max_dd), peak_date, trough_date


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
    # Handle duplicate index labels (can happen from walk-forward fold concat)
    if strategy_returns.index.duplicated().any():
        strategy_returns = strategy_returns.groupby(strategy_returns.index).mean()
    if benchmark_returns.index.duplicated().any():
        benchmark_returns = benchmark_returns.groupby(benchmark_returns.index).mean()
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
    # Handle duplicate index labels (can happen from walk-forward fold concat)
    if strategy_returns.index.duplicated().any():
        strategy_returns = strategy_returns.groupby(strategy_returns.index).mean()
    if benchmark_returns.index.duplicated().any():
        benchmark_returns = benchmark_returns.groupby(benchmark_returns.index).mean()
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

# ---------------------------------------------------------------------------
# Pretty-print summary (inspired by backtrader's cerebro.run output)
# ---------------------------------------------------------------------------


def print_summary(metrics):
    """Print a formatted backtest performance summary, backtrader-style."""
    lines = [
        ('=' * 70, ''),
        ('  BACKTEST PERFORMANCE SUMMARY', ''),
        ('=' * 70, ''),
    ]

    sections = [
        ('Returns', [
            ('Total Return', 'total_return', '{:+.2%}'),
            ('Annual Return', 'annual_return', '{:+.2%}'),
            ('Benchmark Return', 'benchmark_total_return', '{:+.2%}'),
            ('Excess Return', 'excess_return', '{:+.2%}'),
        ]),
        ('Risk', [
            ('Annual Volatility', 'annual_volatility', '{:.2%}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}'),
            ('Tracking Error', 'tracking_error', '{:.2%}'),
        ]),
        ('Risk-Adjusted', [
            ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
            ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
            ('Calmar Ratio', 'calmar_ratio', '{:.3f}'),
            ('Information Ratio', 'information_ratio', '{:.3f}'),
        ]),
        ('Trading Stats', [
            ('Win Rate', 'win_rate', '{:.1%}'),
            ('Profit/Loss Ratio', 'profit_loss_ratio', '{:.3f}'),
            ('Total Days', 'total_days', '{:.0f}'),
            ('Max DD Peak', 'max_drawdown_peak', '{}'),
            ('Max DD Trough', 'max_drawdown_trough', '{}'),
        ]),
    ]

    has_benchmark = 'benchmark_total_return' in metrics

    for section_name, items in sections:
        displayed = [(l, k, f) for l, k, f in items
                     if k in metrics and metrics[k] is not None]
        if not displayed:
            continue
        if not has_benchmark and any('benchmark' in k or 'excess' in k or 'tracking' in k
                                      for _, k, _ in displayed):
            continue

        lines.append((f'  {section_name}', ''))
        for label, key, fmt in displayed:
            val = metrics.get(key)
            if val is None:
                continue
            try:
                formatted = fmt.format(val)
            except (ValueError, TypeError):
                formatted = str(val)
            lines.append((f'    {label:<22}', formatted))

    lines.append(('=' * 70, ''))
    lines.append(('', ''))

    for parts in lines:
        print(''.join(parts))
