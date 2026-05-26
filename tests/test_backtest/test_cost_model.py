"""Tests for transaction cost model."""

import pandas as pd
from quant_platform.backtest.cost_model import CostModel


def test_cost_model_basic():
    model = CostModel()
    turnover = pd.Series([100000, 50000], index=["A", "B"])
    costs = model.compute_costs(turnover)
    assert len(costs) == 2
    assert (costs > 0).all()


def test_cost_model_sell():
    model = CostModel(stamp_tax=0.001, commission=0.0003)
    turnover = pd.Series([100000], index=["A"])
    is_sell = pd.Series([True], index=["A"])

    costs_sell = model.compute_costs(turnover, is_sell=is_sell)
    is_buy = pd.Series([False], index=["A"])
    costs_buy = model.compute_costs(turnover, is_sell=is_buy)

    # Sell should cost more due to stamp tax
    assert costs_sell.iloc[0] > costs_buy.iloc[0]


def test_cost_proportional():
    turnover = pd.Series([10000], index=["A"])
    daily_vol = pd.Series([1000000], index=["A"])

    model_fixed = CostModel(slippage_model="fixed", slippage=0.001)
    model_prop = CostModel(slippage_model="proportional", slippage=0.001)

    cost_fixed = model_fixed.compute_costs(turnover, daily_volume=daily_vol)
    cost_prop = model_prop.compute_costs(turnover, daily_volume=daily_vol)

    assert len(cost_fixed) == 1
    assert len(cost_prop) == 1


def test_cost_zero_turnover():
    model = CostModel()
    turnover = pd.Series([0.0], index=["A"])
    costs = model.compute_costs(turnover)
    assert costs.iloc[0] == 0.0
