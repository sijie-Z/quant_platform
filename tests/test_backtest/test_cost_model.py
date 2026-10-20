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
        + two_way * 0.00001    # transfer fee, both sides (B5)
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

    # commission + slippage + the both-sides transfer fee on the full
    # notional, stamp tax on the assumed sell half of the legacy
    # single-figure path.
    rate = (
        model.commission + model.slippage + model.transfer_fee
        + model.stamp_tax / 2
    )
    assert costs["A"] == pytest.approx(100000 * rate)
    assert costs["B"] == pytest.approx(50000 * rate)


def test_cost_model_sell():
    model = CostModel(stamp_tax=0.001, commission=0.0003)
    turnover = pd.Series([100000], index=["A"])
    is_sell = pd.Series([True], index=["A"])

    costs_sell = model.compute_costs(turnover, is_sell=is_sell)
    is_buy = pd.Series([False], index=["A"])
    costs_buy = model.compute_costs(turnover, is_sell=is_buy)

    assert costs_sell.iloc[0] == pytest.approx(
        100000 * (model.commission + 0.001 + model.slippage + model.transfer_fee),
    )
    assert costs_buy.iloc[0] == pytest.approx(
        100000 * (model.commission + model.slippage + model.transfer_fee),
    )
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
        # Both sides also pay the B5 transfer fee (0.001%).
        transfer = 1_000_000 * 0.00001
        assert cost.loc["A"] == pytest.approx(1_000_000 * 0.001 + transfer)
        assert cost.loc["B"] == pytest.approx(1_000_000 * 0.001 * 0.1 + transfer)

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

        # The B5 transfer fee is charged on both sides of every trade.
        assert cost.loc["A"] / 1_000_000 == pytest.approx(
            0.02 * np.sqrt(0.01) + 0.00001
        )
        assert cost.loc["B"] / 1_000_000 == pytest.approx(
            0.02 * np.sqrt(0.0001) + 0.00001
        )


class TestVectorPathInputShapes:
    """`compute_costs` documents scalar/Series/array turnover; the vector
    paths must accept all three, including a scalar trade value against a
    per-asset volume vector (broadcast, not the single-order calculator)."""

    def test_ndarray_turnover_is_supported(self):
        model = CostModel(commission=0.001, stamp_tax=0.0, slippage=0.0)

        cost = model.compute_costs(np.array([1_000_000.0, 2_000_000.0]))

        # commission plus the default both-sides transfer fee (B5).
        assert cost.tolist() == pytest.approx([1_010.0, 2_020.0])

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
        # A: participation 1e-4; B: participation 1e-2; plus the transfer fee.
        assert cost.loc["A"] == pytest.approx(
            1_000_000 * 0.02 * np.sqrt(1e-4) + 1_000_000 * 0.00001
        )
        assert cost.loc["B"] == pytest.approx(
            1_000_000 * 0.02 * np.sqrt(1e-2) + 1_000_000 * 0.00001
        )

    def test_ndarray_buy_sell_split_is_supported(self):
        model = CostModel(commission=0.001, stamp_tax=0.001, slippage=0.0)

        cost = model.compute_costs(
            np.array([0.5, 0.5]),
            buy_turnover=np.array([0.5, 0.0]),
            sell_turnover=np.array([0.0, 0.5]),
        )

        # Buy: commission + transfer. Sell: commission + full 0.1% stamp tax
        # + transfer (B5: the transfer fee is charged on both sides).
        assert cost.tolist() == pytest.approx([0.000505, 0.001005])


class TestMinimumCommission:
    """B5: the CNY 5 commission floor is per order, not per portfolio.

    The audit probe: a 20,000 CNY account sending ten 2,000 CNY orders paid
    6.00 CNY proportionally, while a broker charging ``max(notional*rate, 5)``
    per order would charge 50.00 CNY (25 bp of the account).
    """

    def test_ten_2000_cny_orders_pay_five_yuan_each(self):
        model = CostModel(
            commission=0.0003, stamp_tax=0.0, slippage=0.0,
            min_commission=5.0, transfer_fee=0.0,
        )
        orders = np.full(10, 2_000.0)
        cost_fraction = model.compute_costs(
            float(orders.sum()), buy_turnover=orders,
            portfolio_value=20_000.0,
        )

        assert float(np.sum(cost_fraction)) == pytest.approx(50.0 / 20_000.0)
        # Proportional commissioning alone would have charged 6.00 CNY.
        assert float(orders.sum()) * 0.0003 == pytest.approx(6.0)

    def test_without_a_portfolio_value_the_floor_cannot_be_applied(self):
        """Documented large-account approximation: the legacy contract does
        not know the CNY notional, so it must not invent a floor."""
        model = CostModel(
            commission=0.0003, stamp_tax=0.0, slippage=0.0,
            min_commission=5.0, transfer_fee=0.0,
        )
        assert model.compute_costs(10_000.0) == pytest.approx(3.0)


class TestTransferFee:
    """B5: the A-share transfer fee is charged on both buys and sells."""

    def test_buy_and_sell_pay_the_same_rate(self):
        model = CostModel(
            commission=0.0, stamp_tax=0.0, slippage=0.0, min_commission=0.0,
        )
        notional = 1_000_000.0

        buy = model.compute_costs(notional, buy_turnover=notional)
        sell = model.compute_costs(notional, sell_turnover=notional)

        assert model.transfer_fee == pytest.approx(0.00001)
        assert buy == pytest.approx(notional * 0.00001)
        assert sell == pytest.approx(notional * 0.00001)


class TestStampTaxSchedule:
    """B11: 0.1% before 2023-08-28 and 0.05% from that date, sell side only."""

    def test_statutory_rate_cut_takes_effect_on_2023_08_28(self):
        model = CostModel(
            commission=0.0, slippage=0.0, transfer_fee=0.0,  # stamp_tax=None
        )
        notional = 1_000_000.0

        before = model.compute_costs(
            notional, sell_turnover=notional, date="2023-08-25",
        )
        on_cut = model.compute_costs(
            notional, sell_turnover=notional, date="2023-08-28",
        )
        after = model.compute_costs(
            notional, sell_turnover=notional, date="2024-06-03",
        )

        assert before == pytest.approx(1_000.0)   # 0.1%
        assert on_cut == pytest.approx(500.0)     # 0.05%
        assert after == pytest.approx(500.0)      # 0.05%

    def test_buy_side_pays_no_stamp_tax(self):
        model = CostModel(
            commission=0.0, slippage=0.0, transfer_fee=0.0,
        )
        notional = 1_000_000.0
        assert model.compute_costs(
            notional, buy_turnover=notional, date="2024-06-03",
        ) == pytest.approx(0.0)

    def test_explicit_rate_overrides_the_schedule(self):
        model = CostModel(
            commission=0.0, stamp_tax=0.001, slippage=0.0, transfer_fee=0.0,
        )
        assert model.compute_costs(
            1_000_000.0, sell_turnover=1_000_000.0, date="2024-06-03",
        ) == pytest.approx(1_000.0)


class TestCostsAreChargedExactlyOnce:
    """Decomposition check: commission/slippage/transfer on both sides, stamp
    tax on the sell notional only, with no component charged twice."""

    def _model(self):
        return CostModel(
            commission=0.0003, slippage=0.001, min_commission=0.0,
            transfer_fee=0.00001,
        )

    def test_buy_swap_and_sell_costs_decompose_exactly(self):
        model = self._model()
        notional = 1_000_000.0
        common = (
            model.commission + model.slippage + model.transfer_fee
        )

        buy = model.compute_costs(
            notional, buy_turnover=notional, date="2024-06-03",
        )
        sell = model.compute_costs(
            notional, sell_turnover=notional, date="2024-06-03",
        )
        swap = model.compute_costs(
            2 * notional, buy_turnover=notional, sell_turnover=notional,
            date="2024-06-03",
        )

        assert buy == pytest.approx(notional * common)
        assert sell == pytest.approx(notional * (common + 0.0005))
        assert swap == pytest.approx(buy + sell)
