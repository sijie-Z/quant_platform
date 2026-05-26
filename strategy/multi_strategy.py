"""Multi-strategy portfolio management.

Manages multiple strategies running simultaneously:
- Strategy registry with metadata
- Capital allocation across strategies
- Aggregate P&L and risk monitoring
- Strategy correlation analysis
- Rebalancing triggers

Inspired by multi-pod hedge fund structure (Millennium, Citadel).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    strategy_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    description: str = ""
    optimizer: str = "mean_variance"
    alpha_method: str = "icir_weighted"
    rebalance_frequency: str = "monthly"
    n_stocks: int = 300
    allocation_pct: float = 0.0          # % of total capital
    max_drawdown_limit: float = 0.15     # Per-strategy DD limit
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StrategyState:
    """Runtime state of a strategy."""
    strategy_id: str = ""
    capital_allocated: float = 0.0
    current_value: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    n_positions: int = 0
    last_rebalance: str = ""
    risk_level: str = "green"


class MultiStrategyManager:
    """Manages multiple strategies with capital allocation.

    Responsibilities:
    - Register and configure strategies
    - Allocate capital based on target weights
    - Track aggregate P&L and risk
    - Detect strategy correlations
    - Trigger rebalancing when allocations drift
    """

    def __init__(self, total_capital: float = 100_000_000):
        self.total_capital = total_capital
        self.strategies: dict[str, StrategyConfig] = {}
        self.states: dict[str, StrategyState] = {}
        self.daily_returns: dict[str, list[float]] = {}

    def add_strategy(self, config: StrategyConfig) -> str:
        """Register a new strategy."""
        self.strategies[config.strategy_id] = config
        self.states[config.strategy_id] = StrategyState(
            strategy_id=config.strategy_id,
            capital_allocated=self.total_capital * config.allocation_pct,
        )
        self.daily_returns[config.strategy_id] = []
        logger.info("Added strategy: %s (%s), allocation=%.1f%%",
                     config.name, config.strategy_id, config.allocation_pct * 100)
        return config.strategy_id

    def remove_strategy(self, strategy_id: str):
        """Remove a strategy."""
        self.strategies.pop(strategy_id, None)
        self.states.pop(strategy_id, None)
        self.daily_returns.pop(strategy_id, None)

    def update_strategy_pnl(self, strategy_id: str, daily_return: float):
        """Update a strategy's daily P&L."""
        if strategy_id not in self.states:
            return

        state = self.states[strategy_id]
        state.daily_pnl = state.capital_allocated * daily_return
        state.current_value += state.daily_pnl
        state.total_pnl = state.current_value - state.capital_allocated
        state.total_return = state.total_pnl / state.capital_allocated if state.capital_allocated > 0 else 0

        self.daily_returns[strategy_id].append(daily_return)

        # Update Sharpe
        rets = self.daily_returns[strategy_id]
        if len(rets) > 60:
            arr = np.array(rets[-252:])
            if arr.std() > 0:
                state.sharpe_ratio = float(arr.mean() / arr.std() * np.sqrt(252))

        # Update max DD
        if len(rets) > 0:
            cum = np.cumprod(1 + np.array(rets))
            peak = np.maximum.accumulate(cum)
            dd = ((cum - peak) / peak).min()
            state.max_drawdown = float(dd)

    def get_aggregate_metrics(self) -> dict:
        """Compute aggregate portfolio metrics across all strategies."""
        total_value = sum(s.current_value for s in self.states.values())
        total_pnl = sum(s.total_pnl for s in self.states.values())
        total_allocated = sum(s.capital_allocated for s in self.states.values())

        # Aggregate daily returns (weighted by allocation)
        all_returns = []
        weights = []
        for sid, config in self.strategies.items():
            if not config.is_active:
                continue
            rets = self.daily_returns.get(sid, [])
            if rets:
                all_returns.append(rets)
                weights.append(config.allocation_pct)

        # Portfolio return = weighted sum of strategy returns
        if all_returns and sum(weights) > 0:
            min_len = min(len(r) for r in all_returns)
            portfolio_rets = np.zeros(min_len)
            for rets, w in zip(all_returns, weights, strict=False):
                portfolio_rets += np.array(rets[-min_len:]) * w

            agg_sharpe = float(portfolio_rets.mean() / portfolio_rets.std() * np.sqrt(252)) if portfolio_rets.std() > 0 else 0
            cum = np.cumprod(1 + portfolio_rets)
            peak = np.maximum.accumulate(cum)
            agg_dd = float(((cum - peak) / peak).min())
        else:
            agg_sharpe = 0
            agg_dd = 0

        # Strategy correlation matrix
        corr_matrix = {}
        active_ids = [sid for sid, c in self.strategies.items() if c.is_active]
        if len(active_ids) > 1:
            min_len = min(len(self.daily_returns.get(sid, [])) for sid in active_ids)
            if min_len > 30:
                df = pd.DataFrame({
                    sid: self.daily_returns[sid][-min_len:]
                    for sid in active_ids
                })
                corr = df.corr()
                corr_matrix = corr.to_dict()

        return {
            "total_capital": self.total_capital,
            "total_allocated": total_allocated,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_return": total_pnl / total_allocated if total_allocated > 0 else 0,
            "aggregate_sharpe": agg_sharpe,
            "aggregate_max_dd": agg_dd,
            "n_strategies": len(self.strategies),
            "n_active": sum(1 for c in self.strategies.values() if c.is_active),
            "strategies": [
                {
                    "id": sid,
                    "name": config.name,
                    "allocation_pct": config.allocation_pct,
                    "value": self.states[sid].current_value,
                    "pnl": self.states[sid].total_pnl,
                    "return": self.states[sid].total_return,
                    "sharpe": self.states[sid].sharpe_ratio,
                    "max_dd": self.states[sid].max_drawdown,
                    "active": config.is_active,
                }
                for sid, config in self.strategies.items()
            ],
            "correlation_matrix": corr_matrix,
        }

    def allocate_capital(self, weights: dict[str, float]):
        """Reallocate capital across strategies.

        Args:
            weights: dict of strategy_id -> target allocation (should sum to 1.0)
        """
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning("Weights sum to %.3f, normalizing", total_weight)

        for sid, w in weights.items():
            if sid in self.strategies:
                normalized = w / total_weight if total_weight > 0 else 0
                self.strategies[sid].allocation_pct = normalized
                self.states[sid].capital_allocated = self.total_capital * normalized
                logger.info("Strategy %s: allocation=%.1f%%, capital=%s",
                           sid, normalized * 100, f"{self.states[sid].capital_allocated:,.0f}")

    def get_risk_alerts(self) -> list[dict]:
        """Check all strategies for risk breaches."""
        alerts = []
        for sid, state in self.states.items():
            config = self.strategies.get(sid)
            if not config:
                continue

            if abs(state.max_drawdown) > config.max_drawdown_limit:
                alerts.append({
                    "strategy_id": sid,
                    "strategy_name": config.name,
                    "type": "drawdown_breach",
                    "severity": "red",
                    "message": f"{config.name}: DD {state.max_drawdown:.2%} exceeds limit {config.max_drawdown_limit:.2%}",
                    "action": "Consider reducing allocation or stopping strategy",
                })

            if state.total_return < -0.10:
                alerts.append({
                    "strategy_id": sid,
                    "strategy_name": config.name,
                    "type": "loss_alert",
                    "severity": "orange",
                    "message": f"{config.name}: Total return {state.total_return:.2%}",
                })

        return alerts
