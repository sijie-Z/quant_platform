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


def _dropna(values):
    return values.dropna() if isinstance(values, pd.Series) else values


def annualization_years(daily_returns, periods_per_year: float | None = None) -> float:
    """Calendar years covered by a periodic return series.

    The old convention was ``len(returns) / 252``: a fixed 252-day year for
    every sample, whatever the index actually spanned. A-share samples have
    ~242 sessions/year, calendar-daily NAV series have 365 observations/year,
    and walk-forward OOS aggregates have gaps -- so that convention
    mis-annualized all of them (M15). The default here measures the index
    span in calendar time and adds one median observation step, because the
    series is labelled by period *start*: the last return completes one
    period after its label.

    Args:
        daily_returns: Series (or array) of periodic returns.
        periods_per_year: Explicit override. When given, years is
            ``len(returns) / periods_per_year`` and the index is ignored.

    Returns:
        Elapsed years. Falls back to ``len / 252`` when the input has no
        usable DatetimeIndex.
    """
    values = _dropna(daily_returns)
    n = len(values)
    if n == 0:
        return 0.0
    if periods_per_year is not None:
        return n / float(periods_per_year) if periods_per_year > 0 else 0.0

    index = getattr(values, "index", None)
    if isinstance(index, pd.DatetimeIndex) and n >= 2:
        span_days = float((index[-1] - index[0]) / pd.Timedelta(days=1))
        steps = np.diff(index.view("int64")) / (86_400 * 1_000_000_000)
        if len(steps):
            span_days += max(float(np.median(steps)), 0.0)
        if span_days > 0:
            return span_days / 365.25
    return n / TRADING_DAYS_PER_YEAR


def infer_periods_per_year(daily_returns) -> float:
    """Observations per calendar year implied by the sample's index span.

    A daily A-share series returns ~242; calendar-daily NAV data returns
    ~365; weekly data returns ~52. Falls back to 252 when the index carries
    no usable time span.
    """
    values = _dropna(daily_returns)
    years = annualization_years(values)
    if years > 0:
        return len(values) / years
    return float(TRADING_DAYS_PER_YEAR)


def annualized_return(
    daily_returns: pd.Series, periods_per_year: float | None = None,
) -> float:
    """Compound annual growth rate over the sample's calendar span.

    ``periods_per_year`` forces the old ``len / N`` convention; when omitted
    the elapsed calendar time of the index is used (see
    :func:`annualization_years`).
    """
    values = daily_returns.dropna()
    total_return = (1 + values).prod() - 1
    years = annualization_years(values, periods_per_year)
    if years < 1 / 252:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(
    daily_returns: pd.Series, periods_per_year: float | None = None,
) -> float:
    """Annualized standard deviation of periodic returns."""
    values = daily_returns.dropna()
    ppy = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(values)
    )
    return values.std() * np.sqrt(ppy)


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free: float = 0.03,
    periods_per_year: float | None = None,
) -> float:
    """Sharpe ratio = (annualized return - rf) / annualized vol."""
    ann_ret = annualized_return(daily_returns, periods_per_year)
    ann_vol = annualized_volatility(daily_returns, periods_per_year)
    if ann_vol < 1e-10:
        return 0.0
    return (ann_ret - risk_free) / ann_vol


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free: float = 0.03,
    periods_per_year: float | None = None,
) -> float:
    """Sortino ratio: excess return over downside deviation.

    Downside deviation is ``sqrt(mean(min(r - MAR, 0)^2))`` over *all*
    observations. Taking the standard deviation of only the negative returns
    (the old implementation) measures their dispersion around their own mean,
    returns 0 for constant-sized losses, and ignores how often losses occur.

    The daily minimum acceptable return is ``risk_free / periods_per_year``
    with the same annualization basis as the other metrics.
    """
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return 0.0
    ppy = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(daily_returns)
    )
    ann_ret = annualized_return(daily_returns, ppy)
    daily_mar = risk_free / ppy
    downside = np.minimum(daily_returns.to_numpy() - daily_mar, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(ppy)
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

    # The wealth path is indexed by return-period *start*: the row dated t is
    # the return close(t) -> close(t+1) (data/pipeline.py shifts by -1). Wealth
    # index 0 is inception; wealth index k >= 1 is reached at the close of the
    # k-th return, i.e. on the (k+1)-th label. Labelling it with index[k-1]
    # was one trading day early (B10). Clamp the final return's close to the
    # last available label.
    last = len(values) - 1

    def _close_date(wealth_idx: int) -> pd.Timestamp:
        if wealth_idx <= 0:
            return values.index[0]
        return values.index[min(wealth_idx, last)]

    peak_date = _close_date(peak_idx)
    trough_date = _close_date(trough_idx)
    return float(max_dd), peak_date, trough_date


def calmar_ratio(
    daily_returns: pd.Series, periods_per_year: float | None = None,
) -> float:
    """Calmar ratio = annualized return / max drawdown (absolute value)."""
    ann_ret = annualized_return(daily_returns, periods_per_year)
    mdd, _, _ = max_drawdown(daily_returns)
    if abs(mdd) < 1e-10:
        return 0.0
    return ann_ret / abs(mdd)


def _aligned_excess(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """Strategy minus benchmark on the common index (duplicates collapsed)."""
    if strategy_returns.index.duplicated().any():
        strategy_returns = strategy_returns.groupby(strategy_returns.index).mean()
    if benchmark_returns.index.duplicated().any():
        benchmark_returns = benchmark_returns.groupby(benchmark_returns.index).mean()
    aligned = pd.DataFrame({
        "strategy": strategy_returns, "benchmark": benchmark_returns,
    }).dropna()
    return aligned["strategy"] - aligned["benchmark"]


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float | None = None,
) -> float:
    """Information ratio = mean excess return / std excess return (annualized)."""
    excess = _aligned_excess(strategy_returns, benchmark_returns)
    if len(excess) < 2:
        return 0.0
    ppy = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(excess)
    )
    ann_excess = excess.mean() * ppy
    ann_tracking_error = excess.std() * np.sqrt(ppy)
    if ann_tracking_error < 1e-10:
        return 0.0
    return ann_excess / ann_tracking_error


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float | None = None,
) -> float:
    """Annualized tracking error."""
    excess = _aligned_excess(strategy_returns, benchmark_returns)
    ppy = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(excess)
    )
    return excess.std() * np.sqrt(ppy)


def win_rate(daily_returns: pd.Series) -> float:
    """Proportion of days with positive return (NaN observations ignored)."""
    values = daily_returns.dropna()
    if len(values) == 0:
        return 0.0
    return (values > 0).mean()


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    """Average gain on winning days / average loss on losing days (NaN ignored)."""
    values = daily_returns.dropna()
    gains = values[values > 0]
    losses = values[values < 0]
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
    periods_per_year: float | None = None,
    risk_free: float = 0.03,
) -> dict:
    """Compute all performance metrics at once.

    Args:
        strategy_returns: Periodic strategy returns.
        benchmark_returns: Optional benchmark returns on the same clock.
        periods_per_year: Annualization override. When omitted it is inferred
            from the strategy index span (~252 for synthetic business-day
            data, ~242 for real A-share sessions, 365 for calendar-daily NAV
            data). Risk-free is annual, converted to a per-period MAR with
            the same basis.
        risk_free: Annual risk-free rate, default 3%.

    Returns:
        Dict of scalar metrics. When a benchmark is supplied, ``total_return``
        / ``benchmark_total_return`` / ``excess_return`` are all computed on
        the *common* index (M11), so the excess is a like-for-like difference
        even when the benchmark series is shorter than the strategy series.
        ``benchmark_coverage`` reports the fraction of strategy observations
        the benchmark covers.
    """
    sr = strategy_returns.dropna()
    if len(sr) < 10:
        return {}

    ppy = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(sr)
    )

    mdd, peak_dt, trough_dt = max_drawdown(sr)

    metrics = {
        "annual_return": annualized_return(sr, ppy),
        "annual_volatility": annualized_volatility(sr, ppy),
        "sharpe_ratio": sharpe_ratio(sr, risk_free, ppy),
        "sortino_ratio": sortino_ratio(sr, risk_free, ppy),
        "max_drawdown": mdd,
        "max_drawdown_peak": str(peak_dt.date()) if pd.notna(peak_dt) else None,
        "max_drawdown_trough": str(trough_dt.date()) if pd.notna(trough_dt) else None,
        "calmar_ratio": calmar_ratio(sr, ppy),
        "win_rate": win_rate(sr),
        "profit_loss_ratio": profit_loss_ratio(sr),
        "total_return": (1 + sr).prod() - 1,
        "total_days": len(sr),
        "periods_per_year": float(ppy),
        "annualization_years": annualization_years(sr, ppy),
        "risk_free_rate": risk_free,
    }

    if benchmark_returns is not None:
        br = benchmark_returns
        if br.index.duplicated().any():
            br = br.groupby(br.index).mean()
        aligned = pd.DataFrame({
            "strategy": sr, "benchmark": br,
        }).dropna().sort_index()
        if len(aligned):
            metrics["information_ratio"] = information_ratio(
                aligned["strategy"], aligned["benchmark"], ppy,
            )
            metrics["tracking_error"] = tracking_error(
                aligned["strategy"], aligned["benchmark"], ppy,
            )
            metrics["benchmark_total_return"] = (1 + aligned["benchmark"]).prod() - 1
            # M11: both legs on the same window, so the difference is a
            # genuine excess return rather than a 70-day compound minus a
            # 100-day compound.
            metrics["total_return"] = (1 + aligned["strategy"]).prod() - 1
            metrics["excess_return"] = (
                metrics["total_return"] - metrics["benchmark_total_return"]
            )
            metrics["benchmark_days"] = len(aligned)
            metrics["benchmark_coverage"] = len(aligned) / len(sr)
            metrics["strategy_total_return_full_window"] = (
                (1 + sr).prod() - 1
            )

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
