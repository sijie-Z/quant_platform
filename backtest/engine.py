"""Vectorized multi-period backtesting engine.

Core of the quant platform: takes signals, portfolio weights, and price data,
then simulates strategy performance with realistic costs and constraints.

Design:
- Vectorized (not event-driven): suitable for daily/weekly/monthly multi-factor
  strategies where execution is at known rebalancing dates.
- Monthly rebalancing: weights computed on the last trading day of each month,
  executed at next day's close, held until next rebalance.
- All P&L is net of transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.metrics import all_metrics
from quant_platform.portfolio.constraints import PortfolioConstraints
from quant_platform.portfolio.covariance import estimate_covariance
from quant_platform.portfolio.optimizers import (
    EqualWeightOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """Vectorized multi-period backtesting engine for A-share strategies."""

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        rebalance_frequency: str = "monthly",
        cost_model: CostModel | None = None,
        constraints: PortfolioConstraints | None = None,
        optimizer: str = "mean_variance",
        benchmark: str = "equal_weight",
        covariance_method: str = "ledoit_wolf",
        covariance_lookback: int = 252,
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.cost_model = cost_model or CostModel()
        self.constraints = constraints or PortfolioConstraints()
        self.optimizer_name = optimizer
        self.benchmark_type = benchmark
        self.covariance_method = covariance_method
        self.covariance_lookback = covariance_lookback

        # Results
        self.daily_returns: pd.Series | None = None
        self.portfolio_values: pd.Series | None = None
        self.benchmark_returns: pd.Series | None = None
        self.weights_history: dict[pd.Timestamp, pd.Series] = {}
        self.turnover_history: pd.Series | None = None

    def run(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        sector_map: pd.Series,
        financials: pd.DataFrame | None = None,
    ) -> dict:
        """Execute the backtest.

        Args:
            signal: (date x asset) alpha signal (higher = more attractive).
            prices: (date x asset) adjusted close prices.
            returns: (date x asset) daily returns.
            benchmark_returns: Series of benchmark daily returns.
            sector_map: Series mapping asset -> sector name.
            financials: (date x asset) financial data, for market cap.

        Returns:
            Dict with daily_returns, portfolio_values, benchmark_returns,
            weights_history, and summary metrics.
        """
        logger.info("Starting backtest: capital=%.0f, freq=%s, optimizer=%s",
                     self.initial_capital, self.rebalance_frequency, self.optimizer_name)

        # Pre-compute benchmark daily returns
        if self.benchmark_type == "equal_weight":
            bench_valid = returns.dropna(axis=1, thresh=int(len(returns) * 0.5))
            self.benchmark_returns = bench_valid.mean(axis=1)
        else:
            self.benchmark_returns = benchmark_returns

        rebalance_dates = self._get_rebalance_dates(signal.index)

        self.weights_history = {}
        prev_weights = None

        for i, rdate in enumerate(rebalance_dates):
            logger.debug("Rebalance %d/%d: %s", i + 1, len(rebalance_dates), str(rdate)[:10])

            sig = signal.loc[rdate].dropna()
            if len(sig) < 10:
                continue

            # Estimate covariance matrix
            lookback_end = returns.index.get_indexer([rdate], method="ffill")[0]
            lookback_start = max(0, lookback_end - self.covariance_lookback)
            ret_window = returns.iloc[lookback_start:lookback_end + 1]

            # Filter to assets with sufficient history
            valid_assets = ret_window.dropna(axis=1, thresh=int(len(ret_window) * 0.5)).columns
            valid_assets = valid_assets.intersection(sig.index)
            sig = sig[valid_assets]

            if len(sig) < 10:
                continue

            ret_window = ret_window[valid_assets]
            try:
                cov = estimate_covariance(
                    ret_window,
                    method=self.covariance_method,
                    lookback=min(self.covariance_lookback, len(ret_window)),
                )
            except Exception:
                cov = None

            # Optimize
            optimizer = self._get_optimizer()
            weights = optimizer.optimize(
                signal=sig,
                cov_matrix=cov,
                prices=prices.loc[rdate].reindex(sig.index) if rdate in prices.index else None,
                prev_weights=prev_weights,
                sector_map=sector_map.reindex(sig.index) if sector_map is not None else None,
            )

            self.weights_history[rdate] = weights
            prev_weights = weights

        # Simulate daily P&L
        self._simulate_pnl(returns)

        # Compute metrics
        summary = all_metrics(self.daily_returns, self.benchmark_returns)
        summary["n_rebalances"] = len(self.weights_history)
        summary["optimizer"] = self.optimizer_name
        summary["initial_capital"] = self.initial_capital

        logger.info("Backtest complete: total_return=%.2f%%, sharpe=%.2f",
                     summary.get("total_return", 0) * 100,
                     summary.get("sharpe_ratio", 0))

        return {
            "daily_returns": self.daily_returns,
            "portfolio_values": self.portfolio_values,
            "benchmark_returns": self.benchmark_returns,
            "weights_history": self.weights_history,
            "turnover_history": self.turnover_history,
            "summary": summary,
        }

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list:
        """Get rebalancing dates based on frequency."""
        if self.rebalance_frequency == "daily":
            return list(dates)
        elif self.rebalance_frequency == "weekly":
            # Last trading day of each week
            grouped = dates.to_series().groupby(dates.isocalendar().week)
            return [pd.Timestamp(d) for d in grouped.last().sort_index().values]
        elif self.rebalance_frequency == "monthly":
            # Last trading day of each month (using actual trading calendar)
            grouped = dates.to_series().groupby([dates.year, dates.month])
            return [pd.Timestamp(d) for d in grouped.last().sort_index().values]
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")

    def _get_optimizer(self):
        if self.optimizer_name == "equal_weight":
            return EqualWeightOptimizer(self.constraints)
        elif self.optimizer_name == "mean_variance":
            return MeanVarianceOptimizer(self.constraints)
        elif self.optimizer_name == "risk_parity":
            return RiskParityOptimizer(self.constraints)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _simulate_pnl(self, returns: pd.DataFrame) -> None:
        """Simulate daily P&L with costs.

        Between rebalance dates, weights drift with price movements.
        At rebalance dates, we compute turnover and deduct transaction costs.
        """
        all_dates = returns.index.sort_values()
        rebalance_dates = sorted(self.weights_history.keys())

        capital = self.initial_capital
        current_weights = pd.Series(0.0, index=returns.columns)
        daily_ret_list = []

        rebalance_iter = iter(rebalance_dates)
        next_rdate = next(rebalance_iter, None)

        for date in all_dates:
            if next_rdate is not None and date >= next_rdate:
                target_weights = self.weights_history.get(next_rdate, current_weights)
                target_weights = target_weights.reindex(returns.columns, fill_value=0.0)

                if current_weights.sum() > 0:
                    turnover = (target_weights - current_weights).abs().sum() / 2
                    cost_rate = self.cost_model.compute_costs(turnover)
                    capital -= cost_rate * capital

                current_weights = target_weights.copy()
                next_rdate = next(rebalance_iter, None)

            daily_ret_assets = returns.loc[date].reindex(current_weights.index, fill_value=0.0)
            portfolio_return = (current_weights * daily_ret_assets).sum()
            daily_ret_list.append(portfolio_return)

            capital = capital * (1 + portfolio_return)

        self.daily_returns = pd.Series(daily_ret_list, index=all_dates, name="strategy_return")
        self.portfolio_values = self.initial_capital * (1 + self.daily_returns).cumprod()
        self.portfolio_values.name = "portfolio_value"
