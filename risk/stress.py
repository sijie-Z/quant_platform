"""Stress testing scenarios for portfolio risk assessment.

Applies historical crisis scenarios to current portfolio to estimate
potential losses under extreme market conditions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Predefined stress scenarios: daily return shocks for broad market
STRESS_SCENARIOS = {
    "2008_financial_crisis": {
        "name": "2008 Global Financial Crisis",
        "market_shock": -0.09,       # ~9% single-day market drop
        "vol_shock": 3.0,            # 3x normal volatility
        "correlation_shock": 1.5,    # Correlations spike
        "liquidity_shock": 0.5,      # Liquidity cut in half
        "duration_days": 60,
    },
    "2015_ashare_crash": {
        "name": "2015 A-Share Crash",
        "market_shock": -0.08,
        "vol_shock": 4.0,
        "correlation_shock": 2.0,    # A-shares: very high correlation in crash
        "liquidity_shock": 0.3,      # Circuit breakers, trading halts
        "duration_days": 30,
    },
    "2020_covid_crash": {
        "name": "2020 COVID-19 Crash",
        "market_shock": -0.08,
        "vol_shock": 4.0,
        "correlation_shock": 1.5,
        "liquidity_shock": 0.4,
        "duration_days": 25,
    },
}


def run_stress_test(
    portfolio_returns: pd.Series,
    scenario_name: str = "2015_ashare_crash",
) -> dict:
    """Simulate portfolio impact under a stress scenario.

    Extrapolates from historical volatility to estimate P&L impact
    under the scenario's shock assumptions.

    Args:
        portfolio_returns: Historical daily portfolio returns.
        scenario_name: Key in STRESS_SCENARIOS.

    Returns:
        Dict with scenario details and estimated impact.
    """
    scenario = STRESS_SCENARIOS.get(scenario_name)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                         f"Available: {list(STRESS_SCENARIOS.keys())}")

    daily_vol = portfolio_returns.std()
    daily_mean = portfolio_returns.mean()

    # Stressed parameters
    shocked_vol = daily_vol * scenario["vol_shock"]
    shocked_mean = daily_mean + scenario["market_shock"] / scenario["duration_days"]

    # Simulate the stressed period
    np.random.seed(12345)
    shocked_returns = np.random.normal(
        shocked_mean, shocked_vol, size=scenario["duration_days"]
    )

    cumulative_return = (1 + shocked_returns).prod() - 1
    max_cumulative_loss = (1 + shocked_returns).cumprod().min() - 1

    return {
        "scenario": scenario["name"],
        "duration_days": scenario["duration_days"],
        "market_shock": scenario["market_shock"],
        "vol_multiplier": scenario["vol_shock"],
        "estimated_cumulative_return": cumulative_return,
        "estimated_max_drawdown": max_cumulative_loss,
    }


def run_all_stress_tests(portfolio_returns: pd.Series) -> pd.DataFrame:
    """Run all predefined stress scenarios and return results table."""
    results = []
    for name in STRESS_SCENARIOS:
        result = run_stress_test(portfolio_returns, name)
        results.append(result)
    return pd.DataFrame(results).set_index("scenario")
