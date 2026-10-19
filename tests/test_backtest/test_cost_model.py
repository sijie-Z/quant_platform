"""Tests for transaction cost model."""

import logging

import numpy as np
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


class TestImpactCostUnitsAndNoDoubleCharging:
    """Regression: the impact path mixed units and charged commission twice.

    `_compute_impact_cost` returned `ExecutionCostCalculator.calculate().total_cost`,
    which already contains commission (including the CNY 5.00 minimum) and
    stamp tax, and it adds a *per-share* impact figure to those currency
    amounts. `compute_costs` then added commission and stamp tax again. A 30%
    turnover rebalance therefore "cost" 5.0 -- five times the traded value --
    instead of the few basis points the fixed-slippage path charges.

    Units are pinned in `CostModel.compute_costs`: `daily_volume` is the
    asset's average daily traded value in CNY (人民币成交额), not share count.
    """

    _RATES = dict(commission=0.0003, stamp_tax=0.001, slippage=0.0005)

    def test_impact_cost_is_on_the_same_scale_as_fixed_slippage(self):
        impact = CostModel(slippage_model="impact", **self._RATES).compute_costs(
            0.30, daily_volume=1_000_000, volatility=0.02,
        )
        fixed = CostModel(slippage_model="fixed", **self._RATES).compute_costs(0.30)

        assert impact < 1e-2, f"impact cost {impact} is not return-sized"
        assert 0.1 * fixed < impact < 10 * fixed, (
            f"impact {impact} and fixed {fixed} are not the same order of magnitude"
        )

    def test_one_million_cny_trade_impact_is_bounded(self):
        model = CostModel(
            commission=0.0, stamp_tax=0.0, slippage=0.0, slippage_model="impact",
        )
        notional = 1_000_000.0
        cost = model.compute_costs(
            notional, daily_volume=10_000_000_000, volatility=0.02,
        )
        fraction = cost / notional

        # A CNY 1m order in a name trading CNY 10bn a day must stay a
        # bounded fraction of notional. The old code returned a flat CNY 5
        # minimum commission (5e-6 of notional, below the lower bound) plus a
        # per-share impact figure; multiplying by price/quantity wrongly gives
        # tens of percent.
        assert 1e-5 < fraction < 2e-2, (
            f"impact on a CNY 1m trade is {fraction:.2%} of notional; "
            "either the unit conversion or the minimum commission leaked in"
        )

    def test_impact_path_charges_commission_once(self):
        common = dict(stamp_tax=0.0, slippage=0.0, slippage_model="impact")
        charged = CostModel(commission=0.001, **common).compute_costs(
            1_000_000, daily_volume=10_000_000_000, volatility=0.02,
        )
        uncharged = CostModel(commission=0.0, **common).compute_costs(
            1_000_000, daily_volume=10_000_000_000, volatility=0.02,
        )

        # The only difference between the two models is the commission rate;
        # the delta must be exactly turnover * rate (charged once, not inside
        # the impact calculator as well).
        assert charged - uncharged == pytest.approx(1_000_000 * 0.001)


class TestVectorPathAlignsSeriesByLabel:
    """The vectorised paths used to rely on pandas label alignment; converting
    straight to numpy pairs values by position, so a volume Series in a
    different order silently mispriced every asset."""

    def test_proportional_path_aligns_by_label(self):
        model = CostModel(
            commission=0.0, stamp_tax=0.0, slippage=0.001,
            slippage_model="proportional",
        )
        turnover = pd.Series([1_000_000.0, 1_000_000.0], index=["A", "B"])
        # Deliberately reversed order: B liquid, A illiquid.
        volume = pd.Series([100_000_000.0, 1_000_000.0], index=["B", "A"])

        cost = model.compute_costs(turnover, daily_volume=volume)

        # A: 1e6 / 1e6 = 1 -> full 0.001; B: 1e6 / 1e8 -> sqrt(0.01) = 0.1x.
        assert cost.loc["A"] == pytest.approx(1_000_000 * 0.001)
        assert cost.loc["B"] == pytest.approx(1_000_000 * 0.001 * 0.1)

    def test_impact_path_aligns_by_label(self):
        model = CostModel(
            commission=0.0, stamp_tax=0.0, slippage=0.0, slippage_model="impact",
        )
        turnover = pd.Series([1_000_000.0, 1_000_000.0], index=["A", "B"])
        volume = pd.Series([10_000_000_000.0, 100_000_000.0], index=["B", "A"])
        volatility = pd.Series([0.02, 0.02], index=["A", "B"])

        cost = model.compute_costs(
            turnover, daily_volume=volume, volatility=volatility,
        )

        assert cost.loc["A"] / 1_000_000 == pytest.approx(0.02 * np.sqrt(0.01))
        assert cost.loc["B"] / 1_000_000 == pytest.approx(0.02 * np.sqrt(0.0001))


class TestVectorPathInputShapes:
    """`compute_costs` documents scalar/Series/array turnover; the vector
    paths must accept all three, including a scalar trade value against a
    per-asset volume vector (broadcast, not the single-order calculator)."""

    def test_ndarray_turnover_is_supported(self):
        model = CostModel(commission=0.001, stamp_tax=0.0, slippage=0.0)

        cost = model.compute_costs(np.array([1_000_000.0, 2_000_000.0]))

        assert cost.tolist() == pytest.approx([1_000.0, 2_000.0])

    def test_scalar_trade_against_per_asset_volume_uses_the_vector_rule(self):
        model = CostModel(
            commission=0.0, stamp_tax=0.0, slippage=0.0, slippage_model="impact",
        )
        volume = pd.Series([10_000_000_000.0, 100_000_000.0], index=["A", "B"])
        volatility = pd.Series([0.02, 0.02], index=["A", "B"])

        cost = model.compute_costs(
            1_000_000, daily_volume=volume, volatility=volatility,
        )

        assert isinstance(cost, pd.Series)
        # A: participation 1e-4; B: participation 1e-2.
        assert cost.loc["A"] == pytest.approx(1_000_000 * 0.02 * np.sqrt(1e-4))
        assert cost.loc["B"] == pytest.approx(1_000_000 * 0.02 * np.sqrt(1e-2))

    def test_ndarray_buy_sell_split_is_supported(self):
        model = CostModel(commission=0.001, stamp_tax=0.001, slippage=0.0)

        cost = model.compute_costs(
            np.array([0.5, 0.5]),
            buy_turnover=np.array([0.5, 0.0]),
            sell_turnover=np.array([0.0, 0.5]),
        )

        # Buy: commission only. Sell: commission + full 0.1% stamp tax.
        assert cost.tolist() == pytest.approx([0.0005, 0.001])
