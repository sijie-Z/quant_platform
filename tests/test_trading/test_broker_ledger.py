"""Exact SimulatedBroker cash ledger for buys and sells.

The only exact sell-ledger assertion in the suite lived in
``test_broker_concurrency.py`` (a concurrency test). ``test_broker.py`` did not
pin cash at all, so flipping the sell fee signs (``cash += value - commission
- tax`` -> ``+= value + commission + tax``) survived the main broker module.
"""

import pytest

from quant_platform.trading.broker import (
    Order,
    OrderSide,
    OrderStatus,
    Position,
    SimulatedBroker,
)


def _broker() -> SimulatedBroker:
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    return broker


def test_same_price_round_trip_loses_exactly_commission_plus_tax():
    broker = _broker()

    buy = broker.place_order(
        Order(code="600519", side=OrderSide.BUY, quantity=100, price=10.0)
    )
    assert buy.status == OrderStatus.FILLED
    assert buy.commission == pytest.approx(5.0, rel=0, abs=1e-12)
    assert buy.tax == 0.0
    cash_after_buy = broker.get_account()["cash"]
    assert cash_after_buy == pytest.approx(998_995.0, rel=0, abs=1e-9)

    broker.new_trading_day()
    sell = broker.place_order(
        Order(code="600519", side=OrderSide.SELL, quantity=100, price=10.0)
    )
    assert sell.status == OrderStatus.FILLED
    assert sell.commission == pytest.approx(5.0, rel=0, abs=1e-12)
    assert sell.tax == pytest.approx(1.0, rel=0, abs=1e-12)

    cash_after_sell = broker.get_account()["cash"]
    assert cash_after_sell == pytest.approx(
        cash_after_buy + 1000.0 - sell.commission - sell.tax,
        rel=0,
        abs=1e-9,
    )
    # 5 + 5 commission + 1 stamp tax: exactly an 11 RMB loss. A mutant that
    # credits the fees ends at 1,000,001.
    assert cash_after_sell == pytest.approx(999_989.0, rel=0, abs=1e-9)
    assert broker.get_positions() == []


def test_buy_then_partial_sell_cash_reconciles():
    broker = _broker()

    buy = broker.place_order(
        Order(code="600519", side=OrderSide.BUY, quantity=200, price=10.0)
    )
    assert buy.status == OrderStatus.FILLED
    cash_after_buy = broker.get_account()["cash"]
    assert cash_after_buy == pytest.approx(997_995.0, rel=0, abs=1e-9)

    broker.new_trading_day()
    sell = broker.place_order(
        Order(code="600519", side=OrderSide.SELL, quantity=100, price=10.0)
    )
    assert sell.status == OrderStatus.FILLED

    cash_after_sell = broker.get_account()["cash"]
    assert cash_after_sell == pytest.approx(
        cash_after_buy + 1000.0 - sell.commission - sell.tax,
        rel=0,
        abs=1e-9,
    )
    assert cash_after_sell == pytest.approx(998_989.0, rel=0, abs=1e-9)

    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].code == "600519"
    assert positions[0].quantity == 100


def test_sell_fee_magnitudes_are_the_documented_rates():
    """Pin the per-trade cost model the ledger assertions rely on."""
    broker = _broker()
    broker._positions.clear()

    broker._positions["600519"] = Position(
        code="600519", quantity=1000, available=1000, avg_cost=9.0,
    )

    sell = broker.place_order(
        Order(code="600519", side=OrderSide.SELL, quantity=1000, price=10.0)
    )

    assert sell.status == OrderStatus.FILLED
    assert sell.commission == pytest.approx(max(10_000.0 * 0.0003, 5.0))
    assert sell.tax == pytest.approx(10_000.0 * 0.001)
    assert broker.get_account()["cash"] == pytest.approx(
        1_000_000.0 + 10_000.0 - 5.0 - 10.0, rel=0, abs=1e-9,
    )
