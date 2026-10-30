"""Greeks must be computed on add and hedged at share scale."""

import pytest
from quant_platform.risk.greeks import GreeksCalculator, OptionGreeks


def _atm_call(contracts: int = 1) -> OptionGreeks:
    return OptionGreeks(
        symbol="600519C1800",
        underlying="600519",
        option_type="call",
        strike=1800.0,
        expiry_days=30,
        position=contracts,
        spot=1800.0,
        volatility=0.25,
    )


def test_add_position_computes_greeks_immediately():
    calc = GreeksCalculator()
    calc.add_position(_atm_call())

    portfolio = calc.compute_portfolio_greeks()

    assert portfolio.total_delta != 0
    assert portfolio.total_gamma != 0


def test_hedge_quantity_is_in_shares_not_multiplied_twice():
    calc = GreeksCalculator()
    calc.add_position(_atm_call(contracts=10))

    portfolio = calc.compute_portfolio_greeks()
    expected = abs(portfolio.total_delta)

    orders = calc.get_hedge_orders()

    assert orders, "a 10-contract ATM position should require a hedge"
    assert orders[0]["quantity"] == pytest.approx(expected, abs=1)
    # The old implementation multiplied the share-equivalent delta by 100.
    assert orders[0]["quantity"] < expected * 2
