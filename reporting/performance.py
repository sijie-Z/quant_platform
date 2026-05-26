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

    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / running_max - 1) * 100

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
    """Plot rolling Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 4))

    rolling_ann_ret = strategy_returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_ann_vol = strategy_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rolling_sharpe = rolling_ann_ret / rolling_ann_vol

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#2ca02c", linewidth=1.0)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=rolling_sharpe.mean(), color="gray", linestyle="--", alpha=0.5,
               label=f"Mean: {rolling_sharpe.mean():.2f}")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_monthly_returns_heatmap(
    strategy_returns: pd.Series,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot monthly returns heatmap."""
    monthly = strategy_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Build pivot: year x month
    df = monthly.to_frame(name="return")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(values="return", index="year", columns="month", aggfunc="sum") * 100

    fig, ax = plt.subplots(figsize=(10, len(pivot) * 0.8 + 1))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        cbar_kws={"label": "Return (%)"}, ax=ax,
        xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
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
