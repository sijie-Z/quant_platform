"""Tests for transaction cost model."""

import logging

import pandas as pd
import pytest
from quant_platform.backtest.cost_model import CostModel


def test_compute_costs_scalar_input_is_the_two_way_traded_value():
    """The contract the backtest engine depends on (regression for BUG-03).

    `compute_costs` charges commission and slippage on the value it is given,
    and halves stamp tax on the assumption that half of that value is sells.
    That assumption only holds for the *two-way* traded value (buys + sells),
    not for one-sided turnover.

    The engine used to pass one-sided turnover -- the convention documented in
    portfolio/constraints.py -- which halved commission, stamp tax and slippage
    together. On a 30%-one-sided-turnover monthly strategy that understated the
    cost by ~47 bp/year.
    """
    model = CostModel(commission=0.0003, stamp_tax=0.001, slippage=0.0005)
    one_sided = 0.30
    two_way = one_sided * 2

    expected = (
        two_way * 0.0003       # commission, both sides
        + one_sided * 0.001    # stamp tax, sells only
        + two_way * 0.0005     # slippage, both sides
    )

    assert model.compute_costs(two_way) == pytest.approx(expected)
    # And feeding it one-sided is exactly the halving that was the defect.
    assert model.compute_costs(two_way) == pytest.approx(
        2 * model.compute_costs(one_sided)
    )


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


class TestSlippageModelFallbackIsNotSilent:
    """Regression for BUG-19.

    `config/default.yaml` set `slippage_model: "impact"`, but the impact model
    needs per-asset `daily_volume` and `volatility`, and `BacktestEngine` passes
    neither. Every rebalance was therefore charged fixed slippage while the
    config claimed a market-impact model was running -- silently, because the
    fall-through had no branch of its own.
    """

    def _model(self, model_name):
        return CostModel(
            commission=0.0003, stamp_tax=0.001, slippage=0.0005,
            slippage_model=model_name,
        )

    def test_impact_without_inputs_warns_that_it_is_not_running(self, caplog):
        model = self._model("impact")
        with caplog.at_level(logging.WARNING):
            model.compute_costs(0.30)

        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert messages, "the fallback was silent"
        assert any("NOT what is running" in m for m in messages), messages
        assert any("daily_volume" in m for m in messages), messages

    def test_the_warning_is_emitted_once_not_per_rebalance(self, caplog):
        """It describes a configuration problem, so it must not repeat on every
        rebalance of a long backtest."""
        model = self._model("impact")
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                model.compute_costs(0.30)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"

    def test_fixed_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._model("fixed").compute_costs(0.30)
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_impact_with_inputs_engages_the_model_without_warning(self, caplog):
        """The contrast that makes the warning meaningful: given the inputs,
        the impact model produces a very different cost."""
        model = self._model("impact")
        with caplog.at_level(logging.WARNING):
            impact_cost = model.compute_costs(0.30, daily_volume=1_000_000, volatility=0.02)
        fixed_cost = self._model("fixed").compute_costs(0.30)

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]
        assert impact_cost != pytest.approx(fixed_cost, rel=0.01)
