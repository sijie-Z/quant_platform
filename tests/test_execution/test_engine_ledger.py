"""ExecutionEngine cash ledger, fee accounting and T+1 regression tests.

Covers audit findings F1 (naked sells), F3 (T+1) and F8 (fees):
- buys increase quantity but not `available`; sells check `available`
- `new_trading_day()` releases prior-day purchases
- commission/stamp tax flow through cash, cost basis and realised P&L so a
  flat round trip reports exactly its fees (-1,600 on 100k @10)
- a buy that would overdraw a tracked cash balance is rejected untouched
"""

import pytest

from quant_platform.execution.engine import ExecutionEngine, OrderSide
from quant_platform.execution.models import OrderStatus


class TestCashLedgerAndFees:
    def test_round_trip_fees_reconcile(self):
        engine = ExecutionEngine(initial_cash=2_000_000)

        buy = engine.create_order("600519", OrderSide.BUY, 100_000)
        engine.submit_order(buy)
        engine.process_fill(buy, price=10.0, quantity=100_000, commission=300.0)

        # Buy commission is capitalised into the cost basis.
        position = engine.get_position("600519")
        assert position.quantity == 100_000
        assert position.avg_cost == pytest.approx(10.003)
        assert engine.cash == pytest.approx(2_000_000 - 1_000_300)

        # Same-day sell is T+1 locked even though shares are held.
        sell = engine.create_order("600519", OrderSide.SELL, 100_000)
        engine.submit_order(sell)
        with pytest.raises(ValueError, match=r"T\+1 locked"):
            engine.process_fill(
                sell, price=10.0, quantity=100_000,
                commission=300.0, tax=1_000.0,
            )
        assert sell.status == OrderStatus.SUBMITTED
        assert engine.get_position("600519").quantity == 100_000

        engine.new_trading_day()
        engine.process_fill(
            sell, price=10.0, quantity=100_000,
            commission=300.0, tax=1_000.0,
        )

        snapshot = engine.portfolio_snapshot({"600519": 10.0})
        # (10 - 10.003) * 100,000 - 300 commission - 1,000 stamp tax
        assert snapshot["total_realized_pnl"] == pytest.approx(-1_600)
        assert snapshot["total_unrealized_pnl"] == pytest.approx(0)
        assert snapshot["total_pnl"] == pytest.approx(-1_600)
        assert snapshot["cash"] == pytest.approx(2_000_000 - 1_600)
        assert snapshot["equity"] == pytest.approx(2_000_000 - 1_600)
        assert engine.cash == pytest.approx(2_000_000 - 1_600)

    def test_buy_that_would_overdraw_is_rejected_without_mutation(self):
        engine = ExecutionEngine(initial_cash=1_000)
        buy = engine.create_order("600519", OrderSide.BUY, 200)
        engine.submit_order(buy)

        with pytest.raises(ValueError, match="Insufficient cash"):
            engine.process_fill(buy, price=10.0, quantity=200, commission=5.0)

        assert buy.status == OrderStatus.SUBMITTED
        assert buy.fills == []
        assert engine.get_position("600519") is None
        assert engine.cash == pytest.approx(1_000)

    def test_naked_sell_raises_instead_of_realising_against_zero_cost(self):
        engine = ExecutionEngine(initial_cash=1_000)
        sell = engine.create_order("600519", OrderSide.SELL, 100)
        engine.submit_order(sell)

        with pytest.raises(ValueError, match="Naked sell"):
            engine.process_fill(sell, price=10.0, quantity=100)

        assert sell.status == OrderStatus.SUBMITTED
        assert sell.fills == []
        assert engine.get_position("600519") is None
        assert engine.cash == pytest.approx(1_000)
        assert engine.portfolio_snapshot({"600519": 10.0})["total_pnl"] == 0

    def test_sell_larger_than_holding_raises(self):
        engine = ExecutionEngine(initial_cash=10_000)
        buy = engine.create_order("600519", OrderSide.BUY, 100)
        engine.submit_order(buy)
        engine.process_fill(buy, price=10.0, quantity=100)
        engine.new_trading_day()

        sell = engine.create_order("600519", OrderSide.SELL, 200)
        engine.submit_order(sell)
        with pytest.raises(ValueError, match="Naked sell"):
            engine.process_fill(sell, price=10.0, quantity=200)
        assert engine.get_position("600519").quantity == 100
        assert engine.cash == pytest.approx(10_000 - 1_000)


class TestTPlusOne:
    def test_same_day_buy_is_not_sellable(self):
        engine = ExecutionEngine(initial_cash=100_000)
        buy = engine.create_order("X", OrderSide.BUY, 100)
        engine.submit_order(buy)
        engine.process_fill(buy, price=10.0, quantity=100)

        position = engine.get_position("X")
        assert position.quantity == 100
        assert position.available == 0

    def test_new_trading_day_releases_prior_day_buys(self):
        engine = ExecutionEngine(initial_cash=100_000)
        buy = engine.create_order("X", OrderSide.BUY, 100)
        engine.submit_order(buy)
        engine.process_fill(buy, price=10.0, quantity=100)

        engine.new_trading_day()
        assert engine.get_position("X").available == 100

        sell = engine.create_order("X", OrderSide.SELL, 100)
        engine.submit_order(sell)
        engine.process_fill(sell, price=11.0, quantity=100)
        assert engine.get_position("X").quantity == 0

    def test_t_plus_zero_disables_the_lock(self):
        engine = ExecutionEngine(initial_cash=100_000)
        engine.t_plus = 0
        buy = engine.create_order("X", OrderSide.BUY, 100)
        engine.submit_order(buy)
        engine.process_fill(buy, price=10.0, quantity=100)

        assert engine.get_position("X").available == 100
