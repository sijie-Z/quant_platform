"""Strategy capacity estimation.

Estimates the maximum AUM (Assets Under Management) a strategy can
handle before market impact erodes returns to unprofitable levels.

Core logic:
1. For each AUM level, compute the daily trade notional from rebalancing
2. Check if trade notional exceeds max_participation * daily_volume
3. If exceeded, cap the trade and add extra market impact cost
4. Run the full backtest with capped trades and impact costs
5. Output: AUM vs Sharpe/return curve

This answers the question: "At what fund size does this strategy stop working?"

Usage:
    estimator = CapacityEstimator(impact_model=SquareRootModel())
    curve = estimator.estimate(
        weights_history=engine.weights_history,
        prices=prices,
        volumes=volumes,
        aum_range=[1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9],
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from quant_core.market_impact import (
    SquareRootModel,
)

from quant_platform.backtest.cost_model import CostModel
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def default_volatility(returns: pd.DataFrame) -> pd.DataFrame:
    """Causal daily volatility from the platform's forward-return panel.

    ``returns[t]`` is the close(t)->close(t+1) return, so at execution day t
    only rows up to t-1 are realized. The rolling window is therefore shifted
    by one session before use; without the shift the impact model at day t
    read a volatility that included day t's own (future) return. Missing
    warm-up values fall back to the 2%/day default the estimator always used.
    """
    return returns.rolling(20, min_periods=5).std().shift(1).fillna(0.02)


@dataclass
class CapacityResult:
    """Capacity estimation result for a single AUM level."""
    aum: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_participation_rate: float
    max_participation_rate: float
    n_capped_days: int          # Days where trade was capped
    total_impact_cost: float    # Extra impact cost (bps)
    turnover: float


@dataclass
class CapacityCurve:
    """Full capacity curve across AUM levels."""
    results: list[CapacityResult]
    aum_range: list[float]
    sharpe_at_capacity: float       # Sharpe at estimated capacity
    capacity_aum: float             # Estimated max AUM (where Sharpe drops below threshold)
    capacity_threshold: float       # Sharpe threshold used

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy plotting."""
        rows = []
        for r in self.results:
            rows.append({
                "aum": r.aum,
                "aum_millions": r.aum / 1e6,
                "annualized_return": r.annualized_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "total_return": r.total_return,
                "avg_participation": r.avg_participation_rate,
                "max_participation": r.max_participation_rate,
                "n_capped_days": r.n_capped_days,
                "impact_cost_bps": r.total_impact_cost,
                "turnover": r.turnover,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Return summary dict."""
        return {
            "capacity_aum": self.capacity_aum,
            "capacity_aum_millions": round(self.capacity_aum / 1e6, 1),
            "sharpe_at_capacity": round(self.sharpe_at_capacity, 3),
            "capacity_threshold": self.capacity_threshold,
            "n_aum_levels": len(self.results),
            "sharpe_range": (
                round(self.results[-1].sharpe_ratio, 3) if self.results else 0,
                round(self.results[0].sharpe_ratio, 3) if self.results else 0,
            ),
        }


class CapacityEstimator:
    """Estimate strategy capacity from rebalancing history.

    Simulates the strategy at different AUM levels, applying:
    - Participation rate caps (max % of daily volume)
    - Market impact costs for large trades
    - Realistic A-share cost model (commission + stamp tax + slippage)

    Args:
        impact_model: Market impact model (default: SquareRootModel).
        max_participation: Maximum participation rate (default 10%).
        cost_model: Transaction cost model.
        sharpe_threshold: Minimum Sharpe to consider "capacity" (default 0.5).
    """

    def __init__(
        self,
        impact_model: Any | None = None,
        max_participation: float = 0.10,
        cost_model: CostModel | None = None,
        sharpe_threshold: float = 0.5,
    ):
        self.impact_model = impact_model or SquareRootModel()
        self.max_participation = max_participation
        self.cost_model = cost_model or CostModel()
        self.sharpe_threshold = sharpe_threshold

    def estimate(
        self,
        weights_history: dict[pd.Timestamp, pd.Series],
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
        aum_range: list[float] | None = None,
        volatility: pd.DataFrame | None = None,
    ) -> CapacityCurve:
        """Estimate strategy capacity across different AUM levels.

        Args:
            weights_history: Dict of rebalance_date -> target weights (from BacktestEngine).
            prices: (date x asset) close prices.
            volumes: (date x asset) daily trading volume (in shares or CNY).
            returns: (date x asset) daily returns.
            benchmark_returns: Benchmark daily returns for Sharpe calculation.
            aum_range: List of AUM levels to test. Default: [1M, 5M, 10M, 50M, 100M, 500M, 1B].
            volatility: (date x asset) daily volatility. If None, estimated
                causally from ``returns`` (``rolling(20).std().shift(1)``:
                row t uses only returns up to t-1). A caller-supplied panel
                must already be causal -- the estimator does not shift it.

        Returns:
            CapacityCurve with results for each AUM level.
        """
        if aum_range is None:
            aum_range = [1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9]

        if volatility is None:
            volatility = default_volatility(returns)

        rebalance_dates = sorted(weights_history.keys())
        all_dates = returns.index.sort_values()

        results = []
        for aum in aum_range:
            result = self._run_at_aum(
                aum=aum,
                weights_history=weights_history,
                prices=prices,
                volumes=volumes,
                returns=returns,
                volatility=volatility,
                benchmark_returns=benchmark_returns,
                rebalance_dates=rebalance_dates,
                all_dates=all_dates,
            )
            results.append(result)
            logger.info(
                "AUM %.0fM: Sharpe=%.2f, Return=%.1f%%, MaxDD=%.1f%%, capped=%d days, impact=%.0f bps",
                aum / 1e6, result.sharpe_ratio,
                result.annualized_return * 100,
                result.max_drawdown * 100,
                result.n_capped_days,
                result.total_impact_cost,
            )

        # Find capacity: largest AUM where Sharpe > threshold
        capacity_aum = 0.0
        sharpe_at_cap = 0.0
        for r in results:
            if r.sharpe_ratio >= self.sharpe_threshold:
                capacity_aum = r.aum
                sharpe_at_cap = r.sharpe_ratio

        curve = CapacityCurve(
            results=results,
            aum_range=aum_range,
            sharpe_at_capacity=sharpe_at_cap,
            capacity_aum=capacity_aum,
            capacity_threshold=self.sharpe_threshold,
        )

        logger.info(
            "Capacity estimate: AUM=%.0fM (Sharpe=%.2f at threshold=%.2f)",
            capacity_aum / 1e6, sharpe_at_cap, self.sharpe_threshold,
        )

        return curve

    def _run_at_aum(
        self,
        aum: float,
        weights_history: dict[pd.Timestamp, pd.Series],
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        returns: pd.DataFrame,
        volatility: pd.DataFrame,
        benchmark_returns: pd.Series | None,
        rebalance_dates: list,
        all_dates: pd.DatetimeIndex,
    ) -> CapacityResult:
        """Run backtest at a specific AUM level with capacity constraints."""
        capital = aum
        current_weights = pd.Series(0.0, index=returns.columns)
        daily_returns_list = []
        participation_rates = []
        capped_days = 0
        total_impact_bps = 0.0
        total_turnover = 0.0

        # Execute at the NEXT trading day's close, matching
        # BacktestEngine._execution_schedule. Applying the target on the signal
        # date itself captured the move the engine deliberately excludes.
        schedule = []
        for rdate in rebalance_dates:
            later = all_dates[all_dates > rdate]
            if len(later):
                schedule.append((later[0], rdate))
        schedule_idx = 0

        for date in all_dates:
            day_cost_rate = 0.0
            day_impact_rate = 0.0
            if schedule_idx < len(schedule) and date >= schedule[schedule_idx][0]:
                _exec_date, next_rdate = schedule[schedule_idx]
                target_weights = weights_history.get(next_rdate, current_weights)
                target_weights = target_weights.reindex(returns.columns, fill_value=0.0)

                # Compute trade notional
                trade_weights = target_weights - current_weights
                trade_notional = trade_weights * capital
                executed_weights = target_weights.copy()

                # Apply participation rate cap
                capped_impact = 0.0
                if date in volumes.index and date in prices.index:
                    vol = volumes.loc[date].reindex(returns.columns, fill_value=0)
                    prc = prices.loc[date].reindex(returns.columns, fill_value=0)
                    daily_volume_cny = vol * prc  # volume in CNY

                    for asset in trade_weights.index:
                        trade_cny = abs(trade_notional.get(asset, 0))
                        vol_cny = daily_volume_cny.get(asset, 0)

                        if vol_cny > 0 and trade_cny > 0:
                            participation = trade_cny / vol_cny
                            participation_rates.append(participation)

                            if participation > self.max_participation:
                                # Actually cap the trade: the old code only
                                # recorded an impact number and still put the
                                # full target into `current_weights`, so the
                                # reported returns were identical at every AUM.
                                capped_days += 1
                                max_trade = vol_cny * self.max_participation
                                scale = max_trade / trade_cny
                                executed_weights[asset] = (
                                    current_weights.get(asset, 0.0)
                                    + trade_weights.get(asset, 0.0) * scale
                                )

                                # Extra impact cost for the deferred portion.
                                # `impact.total` is a per-share cost, so it
                                # must be multiplied by the share count; the
                                # old code added it as if it were currency.
                                excess = trade_cny - max_trade
                                vol_val = volatility.loc[date].get(asset, 0.02) if date in volatility.index else 0.02
                                price_val = prc.get(asset, 100)
                                shares = int(excess / max(price_val, 1))
                                impact = self.impact_model.estimate(
                                    order_quantity=shares,
                                    market_volume=int(vol_cny / max(price_val, 1)),
                                    volatility=vol_val,
                                    price=price_val,
                                )
                                capped_impact += impact.total * shares

                # Standard turnover cost on what was actually executed
                executed_trade = executed_weights - current_weights
                turnover = abs(executed_trade).sum() / 2
                total_turnover += turnover
                # Two-way traded value: compute_costs halves stamp tax on the
                # assumption that half of its input is sells, so it needs
                # buys + sells, not the one-sided turnover kept above.
                cost_rate = self.cost_model.compute_costs(turnover * 2)
                day_cost_rate = cost_rate
                capital -= cost_rate * capital

                # Extra impact cost
                if capped_impact > 0 and capital > 0:
                    impact_bps = (capped_impact / capital) * 10000
                    total_impact_bps += impact_bps
                    day_impact_rate = capped_impact / capital
                    capital -= capped_impact

                current_weights = executed_weights
                schedule_idx += 1

            # Daily P&L, with weights drifting between rebalances. Costs are
            # liabilities of the day they are paid, so the reported return is
            # net of them -- the old series was gross and the AUM curve was
            # identical at every capital level.
            daily_ret_assets = returns.loc[date].reindex(current_weights.index, fill_value=0.0)
            portfolio_return = (current_weights * daily_ret_assets).sum()
            net_return = portfolio_return - day_cost_rate - day_impact_rate
            daily_returns_list.append(net_return)
            capital = capital * (1 + net_return)

            if current_weights.sum() > 1e-12:
                growth = current_weights * (1.0 + daily_ret_assets.fillna(0.0))
                cash_weight = max(0.0, 1.0 - float(current_weights.sum()))
                total_value = float(growth.sum()) + cash_weight
                if total_value > 1e-12:
                    current_weights = growth / total_value

        # Compute metrics with the same definitions as the backtest summary
        daily_returns = pd.Series(daily_returns_list, index=all_dates)

        from quant_platform.backtest.metrics import all_metrics

        metrics = all_metrics(daily_returns)
        ann_return = float(metrics.get("annual_return", 0.0))
        sharpe = float(metrics.get("sharpe_ratio", 0.0))
        total_ret = float(metrics.get("total_return", 0.0))
        max_dd = float(metrics.get("max_drawdown", 0.0))

        avg_participation = float(np.mean(participation_rates)) if participation_rates else 0.0
        max_participation = float(np.max(participation_rates)) if participation_rates else 0.0

        return CapacityResult(
            aum=aum,
            annualized_return=ann_return,
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 6),
            total_return=round(total_ret, 6),
            avg_participation_rate=round(avg_participation, 6),
            max_participation_rate=round(max_participation, 6),
            n_capped_days=capped_days,
            total_impact_cost=round(total_impact_bps, 2),
            turnover=round(total_turnover, 4),
        )
