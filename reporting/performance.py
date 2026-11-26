"""Performance visualization: equity curves, drawdowns, rolling metrics.

Generates publication-quality charts for backtest results.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from quant_platform.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def plot_equity_curve(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Strategy Performance",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot cumulative equity curve of strategy vs benchmark."""
    fig, ax = plt.subplots(figsize=(12, 6))

    strategy_curve = (1 + strategy_returns).cumprod()
    ax.plot(strategy_curve.index, strategy_curve.values, label="Strategy", linewidth=1.5, color="#1f77b4")

    if benchmark_returns is not None:
        bench_curve = (1 + benchmark_returns.reindex(strategy_returns.index, fill_value=0)).cumprod()
        ax.plot(bench_curve.index, bench_curve.values, label="Benchmark", linewidth=1.0, color="gray", alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (x)")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    strategy_returns: pd.Series,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot underwater (drawdown) chart."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Include the initial wealth of 1.0 as the first peak, matching
    # backtest.metrics.max_drawdown. Starting from the first *return* made a
    # loss on day one invisible to the running maximum.
    #
    # The per-day returns must be compounded before the underwater scan: the
    # old code treated each daily growth factor (1 + r) as the wealth level
    # itself, so the plotted "drawdown" was the dispersion of daily returns
    # (about -6% on a random series) instead of a drawdown of the equity curve
    # (about -9% to -56% on real results). The chart is now exactly
    # backtest.metrics.max_drawdown in percent.
    growth = (1 + strategy_returns.fillna(0)).cumprod().to_numpy()
    wealth = np.concatenate([[1.0], growth])
    running_max = np.maximum.accumulate(wealth)
    dd_values = (wealth / running_max - 1)[1:] * 100
    drawdown = pd.Series(dd_values, index=strategy_returns.index)

    ax.fill_between(drawdown.index, 0, drawdown.values, color="#d62728", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=0.5)
    ax.set_title("Drawdown (% from Peak)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_sharpe(
    strategy_returns: pd.Series,
    window: int = 252,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio.

    The rolling metric is the zero-risk-free, fixed-252 convention
    ``mean * 252 / (std * sqrt(252))``, not the headline Sharpe (which uses
    the 3% risk-free rate and an inferred periods-per-year). The axis title
    and legend state this so the two numbers are not read as the same metric
    (R89).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    rolling_ann_ret = strategy_returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_ann_vol = strategy_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Constant returns give zero vol; the raw division produced inf and the
    # legend read "Mean: inf".
    rolling_sharpe = (rolling_ann_ret / rolling_ann_vol.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan,
    )

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#2ca02c", linewidth=1.0)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    mean_sharpe = rolling_sharpe.mean()
    if np.isfinite(mean_sharpe):
        ax.axhline(y=mean_sharpe, color="gray", linestyle="--", alpha=0.5,
                   label=f"Mean: {mean_sharpe:.2f}")
    else:
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Mean: n/a")
    if not np.isfinite(rolling_sharpe.to_numpy(dtype=float)).any():
        # An all-NaN series (constant returns / too few observations) used to
        # render an empty plot with no explanation (R105).
        ax.text(
            0.5, 0.5, "Insufficient variation for a rolling Sharpe",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#6b7a8d",
        )
    ax.set_title(
        f"Rolling {window}-Day Sharpe Ratio (rf = 0)", fontsize=13, fontweight="bold",
    )
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _monthly_returns_by_period_end(strategy_returns: pd.Series) -> pd.Series:
    """Compound returns into calendar months keyed by the *period end*.

    The engine labels a return row with the period *start* (row t is the
    close(t) -> close(t+1) return). Bucketing on that label put a Friday
    return into the wrong month (2023-12-29 +10% was plotted as December even
    though it accrued in January). Each return is therefore attributed to the
    next observation's date, and the final row to the following calendar day.
    """
    values = strategy_returns.dropna()
    if len(values) == 0 or not isinstance(values.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    if len(values) >= 2:
        period_ends = pd.DatetimeIndex(
            list(values.index[1:])
            + [values.index[-1] + pd.Timedelta(days=1)]
        )
    else:
        period_ends = pd.DatetimeIndex([values.index[0] + pd.Timedelta(days=1)])
    attributed = pd.Series(values.to_numpy(), index=period_ends)
    return attributed.resample(pd.offsets.MonthEnd()).apply(
        lambda x: (1 + x).prod() - 1 if len(x) else np.nan
    )


def plot_monthly_returns_heatmap(
    strategy_returns: pd.Series,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot monthly returns heatmap."""
    try:
        monthly = _monthly_returns_by_period_end(strategy_returns)
    except (TypeError, ValueError):
        monthly = pd.Series(dtype=float)

    # Build pivot: year x month
    if monthly.empty:
        pivot = pd.DataFrame()
    else:
        df = monthly.to_frame(name="return")
        df["year"] = df.index.year
        df["month"] = df.index.month
        # `aggfunc="sum"` turns an all-NaN month into 0.0 (pandas' sum has
        # min_count=0), plotting a real-looking 0% for a month with no data.
        pivot = df.pivot_table(
            values="return", index="year", columns="month",
            aggfunc=lambda x: x.sum(min_count=1),
        ) * 100

    fig, ax = plt.subplots(figsize=(10, len(pivot) * 0.8 + 1))
    if pivot.empty or pivot.dropna(how="all").empty:
        ax.text(
            0.5, 0.5, "No monthly returns available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#6b7a8d",
        )
        ax.set_title("Monthly Returns (%)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Label the columns the pivot actually has. A March-August backtest used
    # to show its six cells under Jan-Jun.
    xticklabels = [month_labels[int(month) - 1] for month in pivot.columns]
    # A NaN cell must stay blank, not print the literal string "nan".
    annotations = pivot.apply(
        lambda column: column.map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    )
    sns.heatmap(
        pivot, annot=annotations, fmt="", cmap="RdYlGn", center=0,
        cbar_kws={"label": "Return (%)"}, ax=ax,
        xticklabels=xticklabels,
    )
    ax.set_title("Monthly Returns (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Year")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ic_decay(
    decay: pd.Series,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot IC decay curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(decay.index, decay.values, marker="o", color="#1f77b4")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Forward Period (days)")
    ax.set_ylabel("Mean Rank IC")
    ax.set_title("IC Decay", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
