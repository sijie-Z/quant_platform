"""Regression tests for the strategy/execution integration defects.

Covers F1 (duplicate rebalance), F2 (cash sufficiency), F3 (T+1 on the
orchestrator path), F4 (risk marks / real equity), F5 (strategy removal),
F9 (lot size), F10 (unknown strategy ids) and F11 (global sells-first).
"""

import pandas as pd
import pytest

from quant_platform.execution.engine import ExecutionEngine, OrderSide
from quant_platform.execution.models import OrderStatus
from quant_platform.strategy.multi_strategy import MultiStrategyManager, StrategyConfig
from quant_platform.strategy.portfolio_orchestrator import PortfolioOrchestrator


def build(total_capital=1_000_000, allocations=None, **kwargs):
    """Create a manager + orchestrator. `allocations` maps name -> weight."""
    allocations = allocations if allocations is not None else {"A": 1.0}
    ms = MultiStrategyManager(total_capital=total_capital)
    ids = {
        name: ms.add_strategy(StrategyConfig(name=name, allocation_pct=pct))
        for name, pct in allocations.items()
    }
    orch = PortfolioOrchestrator(ms, exec_engine=ExecutionEngine(), **kwargs)
    return orch, ms, ids


# ---------------------------------------------------------------------------
# F1 — rebalance idempotency / pending-order netting
# ---------------------------------------------------------------------------


class TestRebalanceIdempotency:
    def test_two_rebalances_before_fills_do_not_double_orders_or_positions(self):
        orch, _, ids = build(allocations={"A": 0.5, "B": 0.5})
        orch._last_prices = {"X": 10.0}
        for sid in ids.values():
            orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)

        first = orch.rebalance()
        assert len(first) == 2
        second = orch.rebalance()
        assert second == []

        open_orders = [
            order for order in orch.exec_engine.orders
            if order.status == OrderStatus.SUBMITTED
        ]
        assert len(open_orders) == 2

        orch.process_fills({"X": 10.0})
        # Each strategy targeted 1,500 shares (500k / 50 positions x 1.5 / 10).
        assert orch.exec_engine.get_position("X").quantity == 3_000
        assert orch._books[ids["A"]]["X"]["qty"] == 1_500
        assert orch._books[ids["B"]]["X"]["qty"] == 1_500
        assert orch.portfolio_summary()["positions"][0]["quantity"] == 3_000

    def test_smaller_target_supersedes_a_pending_buy(self):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        first = orch.rebalance()
        assert len(first) == 1
        assert first[0]["quantity"] == 3_000
        old_order = orch.exec_engine.get_order(first[0]["order_id"])

        # Target drops before the fill: the stale 3,000-share buy is cancelled
        # and replaced by a 1,000-share buy, not stacked on top of it.
        orch._targets[sid] = {"X": 1_000}
        second = orch.rebalance()

        assert old_order.status == OrderStatus.CANCELLED
        assert len(second) == 1
        assert second[0]["side"] == "buy"
        assert second[0]["quantity"] == 1_000
        open_buys = [
            order for order in orch.exec_engine.orders
            if order.status == OrderStatus.SUBMITTED
            and order.side == OrderSide.BUY
        ]
        assert len(open_buys) == 1

        orch.process_fills({"X": 10.0})
        assert orch.exec_engine.get_position("X").quantity == 1_000
        assert orch._books[sid]["X"]["qty"] == 1_000

    def test_target_reversal_cancels_pending_sells(self):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        assert orch._books[sid]["X"]["qty"] == 3_000

        # Day 2: exit creates a pending 3,000-share sell ...
        orch.on_signal("2025-01-09", pd.Series(dtype=float), strategy_id=sid)
        sells = [o for o in orch.rebalance() if o["side"] == "sell"]
        assert len(sells) == 1

        # ... but the target is restored before the fill: the sell is
        # cancelled and no buy-back order is created.
        orch._targets[sid] = {"X": 3_000}
        assert orch.rebalance() == []
        pending_sells = [
            order for order in orch.exec_engine.orders
            if order.side == OrderSide.SELL
            and order.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
        ]
        assert pending_sells == []

        orch.process_fills({"X": 10.0})
        assert orch._books[sid]["X"]["qty"] == 3_000

    def test_duplicate_sells_do_not_consume_another_strategys_shares(self):
        orch, _, ids = build(allocations={"A": 0.5, "B": 0.5})
        a, b = ids["A"], ids["B"]
        orch._last_prices = {"X": 10.0}
        for sid in (a, b):
            orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        assert orch.exec_engine.get_position("X").quantity == 3_000

        # Day 2: A exits while B keeps its position.
        orch.on_signal("2025-01-09", pd.Series(dtype=float), strategy_id=a)
        first = [o for o in orch.rebalance() if o["side"] == "sell"]
        assert len(first) == 1
        assert first[0]["quantity"] == 1_500
        # Idempotent while the sell is pending: no second 1,500-share sell.
        assert orch.rebalance() == []

        orch.process_fills({"X": 10.0})
        # A's sale consumed A's lots only; B still owns its 1,500 shares.
        assert orch.exec_engine.get_position("X").quantity == 1_500
        assert orch._books[a]["X"]["qty"] == 0
        assert orch._books[b]["X"]["qty"] == 1_500

        # Day 3: B exits. Its sale must realise against B's own cost basis,
        # never against a zeroed position (the old bug booked +750,000).
        orch.on_signal("2025-01-10", pd.Series(dtype=float), strategy_id=b)
        sells = [o for o in orch.rebalance() if o["side"] == "sell"]
        assert len(sells) == 1
        orch.process_fills({"X": 10.0})
        assert orch.exec_engine.get_position("X").quantity == 0

        summary = orch.portfolio_summary()
        expected_realized = (
            orch._books[a]["X"]["realized_pnl"]
            + orch._books[b]["X"]["realized_pnl"]
        )
        assert summary["realized_pnl"] == pytest.approx(expected_realized, abs=0.01)
        # A flat round trip with fees is a small loss, not a phantom gain.
        assert summary["realized_pnl"] < 0
        assert summary["total_pnl"] < 0


# ---------------------------------------------------------------------------
# F2 — cash sufficiency
# ---------------------------------------------------------------------------


class TestCashSufficiency:
    def test_signal_sizing_is_capped_at_strategy_capital(self):
        orch, _, ids = build(allocations={"A": 0.5, "B": 0.5})
        prices = {"X1": 10.0, "X2": 10.0, "X3": 10.0}
        orch._last_prices = dict(prices)
        signal = pd.Series({"X1": 0.5, "X2": 0.5, "X3": 0.5})
        for sid in ids.values():
            orch.on_signal(
                "2025-01-08", signal, strategy_id=sid, max_positions=3,
            )

        # Naive (1 + signal) sizing would target 750k per 500k strategy.
        for sid in ids.values():
            notional = sum(
                qty * prices[ticker]
                for ticker, qty in orch._targets[sid].items()
            )
            assert notional <= 500_000 + 1e-6

        orders = orch.rebalance()
        total_notional = sum(
            order["quantity"] * prices[order["ticker"]] for order in orders
        )
        assert total_notional <= 1_000_000 + 1e-6

        orch.process_fills(prices)
        assert orch.exec_engine.cash >= 0
        assert orch.cash_available >= 0
        assert orch.portfolio_summary()["cash_available"] >= 0
        assert sum(
            p.market_value for p in orch.exec_engine.positions
        ) <= 1_000_000

    def test_buy_that_would_overdraw_cash_is_rejected_at_fill(self):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal(
            "2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid,
            cash_per_position=1_000_000,
        )
        orders = orch.rebalance()
        assert len(orders) == 1

        # The price gaps 5% before the fill: the buy no longer fits the cash.
        orch.process_fills({"X": 10.5})

        order = orch.exec_engine.get_order(orders[0]["order_id"])
        assert order.status == OrderStatus.REJECTED
        assert orch.exec_engine.get_position("X") is None
        assert orch.exec_engine.cash == pytest.approx(1_000_000)
        assert orch.cash_available == pytest.approx(1_000_000)
        assert orch.portfolio_summary()["total_pnl"] == 0

    def test_cancel_releases_the_cash_lock(self):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal(
            "2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid,
            cash_per_position=1_000_000,
        )
        orders = orch.rebalance()
        locked = orch._locked_cash()
        assert locked > 0
        assert orch.cash_available == pytest.approx(1_000_000 - locked)

        assert orch.cancel_order(orders[0]["order_id"]) is True
        assert orch._locked_cash() == 0
        assert orch.cash_available == pytest.approx(1_000_000)

    def test_expiry_releases_the_cash_lock(self):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal(
            "2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid,
            cash_per_position=1_000_000,
        )
        orders = orch.rebalance()
        assert orch._locked_cash() > 0

        assert orch.expire_order(orders[0]["order_id"]) is True
        order = orch.exec_engine.get_order(orders[0]["order_id"])
        assert order.status == OrderStatus.EXPIRED
        assert orch._locked_cash() == 0
        assert orch.cash_available == pytest.approx(1_000_000)

    def test_risk_rejected_buy_does_not_lock_cash(self):
        class BlockingRisk:
            def check_pre_trade(self, payload):
                class Breach:
                    message = "blocked"

                return False, [Breach()]

        orch, _, ids = build(risk_monitor=BlockingRisk())
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        assert orch.rebalance() == []
        assert orch._locked_cash() == 0
        assert orch.cash_available == pytest.approx(1_000_000)


# ---------------------------------------------------------------------------
# F3 — T+1 on the orchestrator path
# ---------------------------------------------------------------------------


class TestOrchestratorTPlusOne:
    def test_same_day_sell_is_blocked_and_next_day_released(self):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"X": 10.0})

        position = orch.exec_engine.get_position("X")
        quantity = position.quantity
        assert quantity > 0
        assert position.available == 0

        # Same-day exit signal: T+1 makes the position unsellable.
        orch.on_signal("2025-01-08", pd.Series(dtype=float), strategy_id=sid)
        assert orch.rebalance() == []
        assert orch.exec_engine.get_position("X").quantity == quantity

        # Next session releases the prior-day buys and the exit fills.
        orch.on_signal("2025-01-09", pd.Series(dtype=float), strategy_id=sid)
        assert orch.exec_engine.get_position("X").available == quantity
        sells = [o for o in orch.rebalance() if o["side"] == "sell"]
        assert len(sells) == 1
        assert sells[0]["quantity"] == quantity
        orch.process_fills({"X": 10.0})
        assert orch.exec_engine.get_position("X").quantity == 0


# ---------------------------------------------------------------------------
# F4 — real cash ledger, marks, equity and day-start rebasing
# ---------------------------------------------------------------------------


class RecordingRiskMonitor:
    def __init__(self):
        self.updates = []

    def update_portfolio_state(self, **kwargs):
        self.updates.append(kwargs)

    def project_order(self, payload):
        pass

    def check_pre_trade(self, payload):
        return True, []


class TestRiskMarksAndEquity:
    def _buy_with_full_book(self, risk):
        orch, _, ids = build(risk_monitor=risk)
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal(
            "2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid,
            cash_per_position=1_000_000,
        )
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        return orch, sid

    def test_price_drop_shows_in_equity_daily_pnl_and_drawdown(self):
        risk = RecordingRiskMonitor()
        orch, _ = self._buy_with_full_book(risk)
        position = orch.exec_engine.get_position("X")
        assert position.quantity > 0
        assert position.market_value > 0  # marked at the fill, not stale zero
        assert risk.updates, "risk monitor never received the book"

        # 50% price drop.
        orch.process_fills({"X": 5.0})
        latest = risk.updates[-1]
        marked_value = position.quantity * 5.0
        expected_equity = orch.exec_engine.cash + marked_value
        assert latest["portfolio_value"] == pytest.approx(expected_equity, rel=1e-6)
        assert latest["portfolio_value"] < 600_000
        assert latest["daily_pnl"] < -400_000
        assert latest["positions"]["X"]["value"] == pytest.approx(marked_value)
        assert latest["positions"]["X"]["weight"] == pytest.approx(
            marked_value / expected_equity,
        )

        summary = orch.portfolio_summary()
        assert summary["equity"] == pytest.approx(expected_equity, abs=0.01)
        assert summary["total_pnl"] == pytest.approx(
            summary["equity"] - 1_000_000, abs=0.01,
        )

    def test_day_start_equity_rebases_when_a_new_trading_day_starts(self):
        risk = RecordingRiskMonitor()
        orch, sid = self._buy_with_full_book(risk)
        orch.process_fills({"X": 5.0})
        assert risk.updates[-1]["daily_pnl"] < -400_000

        # Keep the target at the actual holding so the rebase pass creates no
        # new order (the sizing budget left a small cash cushion).
        quantity = orch.exec_engine.get_position("X").quantity
        orch._targets[sid] = {"X": quantity}

        orch.new_trading_day()
        orch.rebalance()  # marks the book and rebases day-start equity

        rebased = risk.updates[-1]
        assert rebased["daily_pnl"] == pytest.approx(0)
        baseline = rebased["portfolio_value"]

        orch.process_fills({"X": 4.5})
        after = risk.updates[-1]
        assert after["portfolio_value"] == pytest.approx(baseline - quantity * 0.5)
        assert after["daily_pnl"] == pytest.approx(-quantity * 0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# F5 — strategy removal
# ---------------------------------------------------------------------------


class TestStrategyRemoval:
    def test_removal_cancels_orders_liquidates_and_purges(self):
        orch, ms, ids = build(allocations={"A": 0.5, "B": 0.5})
        a, b = ids["A"], ids["B"]
        orch._last_prices = {"X": 10.0}
        for sid in (a, b):
            orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        assert orch.exec_engine.get_position("X").quantity == 3_000
        cash_before = orch.exec_engine.cash

        # Day 2: A exits, leaving a pending sell, then is removed.
        orch.on_signal("2025-01-09", pd.Series(dtype=float), strategy_id=a)
        pending_sells = [o for o in orch.rebalance() if o["side"] == "sell"]
        assert len(pending_sells) == 1

        result = orch.remove_strategy(a)

        assert len(result["canceled_orders"]) == 1
        assert sum(item["quantity"] for item in result["liquidated"]) == 1_500
        assert a not in ms.strategies and a not in ms.states
        assert a not in orch._targets and a not in orch._books
        # A's shares are gone; B's are untouched.
        assert orch.exec_engine.get_position("X").quantity == 1_500
        assert orch._books[b]["X"]["qty"] == 1_500
        # Liquidation proceeds returned to the shared cash pool.
        assert orch.exec_engine.cash > cash_before
        assert orch.cash_available == pytest.approx(orch.exec_engine.cash)

        with pytest.raises(ValueError, match="Unknown strategy_id"):
            orch.on_signal("2025-01-09", pd.Series({"X": 0.5}), strategy_id=a)
        assert orch.rebalance() == []

    def test_remove_unknown_strategy_raises(self):
        orch, _, _ = build()
        with pytest.raises(ValueError, match="Unknown strategy_id"):
            orch.remove_strategy("ghost")

    def test_removal_via_manager_is_routed_through_the_orchestrator(self):
        orch, ms, ids = build()
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        orch.new_trading_day()

        ms.remove_strategy(sid)

        assert sid not in orch._books
        assert orch.exec_engine.get_position("X").quantity == 0


# ---------------------------------------------------------------------------
# F6/F7 — a strategy moved to 0% allocation stops trading and unwinds
# ---------------------------------------------------------------------------


class TestZeroAllocation:
    def test_zeroed_strategy_unwinds_and_records_no_pnl(self):
        orch, ms, ids = build(allocations={"A": 0.5, "B": 0.5})
        a, b = ids["A"], ids["B"]
        orch._last_prices = {"X": 10.0}
        for sid in (a, b):
            orch.on_signal("2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        assert orch._books[a]["X"]["qty"] == 1_500

        ms.allocate_capital({b: 1.0})
        assert ms.states[a].capital_allocated == 0
        orch.new_trading_day()

        sells = [o for o in orch.rebalance() if o["side"] == "sell"]
        assert len(sells) == 1
        assert sells[0]["strategy"] == a
        assert sells[0]["quantity"] == 1_500

        orch.process_fills({"X": 10.0})
        assert orch._books[a]["X"]["qty"] == 0
        assert orch._books[b]["X"]["qty"] == 1_500
        # Zero-capital segment records no P&L (the manager resets it at
        # allocation time and the orchestrator feeds it no returns).
        assert ms.states[a].total_pnl == 0
        assert ms.daily_returns[a] == []


# ---------------------------------------------------------------------------
# F9 — configured lot size, zero-quantity order skipping
# ---------------------------------------------------------------------------


class TestLotSize:
    def test_configured_lot_size_is_used_and_zero_orders_are_skipped(self):
        orch, _, ids = build(total_capital=2_000_000, lot_size=10)
        sid = ids["A"]
        orch._last_prices = {"X": 10.0}
        orch.on_signal(
            "2025-01-08", pd.Series({"X": 0.5}), strategy_id=sid,
            cash_per_position=1_000_000,
        )
        assert orch._targets[sid]["X"] == 150_000
        orch.rebalance()
        orch.process_fills({"X": 10.0})
        assert orch._books[sid]["X"]["qty"] == 150_000

        # Price ticks: the resized target is 149,980, i.e. a 20-share sell.
        # The old code rounded with the default 100-lot and created a
        # zero-quantity order that crashed the fill pass.
        orch._last_prices["X"] = 10.0007
        orch.on_signal(
            "2025-01-09", pd.Series({"X": 0.5}), strategy_id=sid,
            cash_per_position=1_000_000,
        )
        assert orch._targets[sid]["X"] == 149_980

        orders = orch.rebalance()
        assert len(orders) == 1
        assert orders[0]["side"] == "sell"
        assert orders[0]["quantity"] == 20
        assert orders[0]["quantity"] % orch.lot_size == 0
        orch.process_fills({"X": 10.0007})
        assert orch._books[sid]["X"]["qty"] == 149_980

        # A residual below one lot produces no order and no crash.
        orch._targets[sid] = {"X": 149_975}
        orch.new_trading_day()
        assert orch.rebalance() == []
        orch.process_fills({"X": 10.0007})


# ---------------------------------------------------------------------------
# F10 — unknown strategy ids
# ---------------------------------------------------------------------------


class TestUnknownStrategy:
    def test_unknown_strategy_signal_raises_and_trades_nothing(self):
        orch, _, _ = build()
        orch._last_prices = {"X": 10.0}

        with pytest.raises(ValueError, match="Unknown strategy_id"):
            orch.on_signal(
                "2025-01-08", pd.Series({"X": 0.5}), strategy_id="ghost",
            )

        assert "ghost" not in orch._targets
        assert "ghost" not in orch._books
        assert orch.exec_engine.orders == []
        assert orch.exec_engine.cash == pytest.approx(1_000_000)


# ---------------------------------------------------------------------------
# F11 — global sells-first
# ---------------------------------------------------------------------------


class TestSellsFirst:
    def test_rebalance_creates_all_sells_before_any_buy(self):
        orch, _, ids = build(allocations={"A": 0.5, "B": 0.5})
        a, b = ids["A"], ids["B"]
        orch._last_prices = {"Y": 10.0, "Z": 10.0}

        # A's `_targets` entry is created first, so a per-strategy loop would
        # create A's buy before B's sell.
        orch.on_signal("2025-01-08", pd.Series(dtype=float), strategy_id=a)
        orch.on_signal("2025-01-08", pd.Series({"Y": 0.5}), strategy_id=b)
        orch.rebalance()
        orch.process_fills({"Y": 10.0})

        orch.on_signal("2025-01-09", pd.Series({"Z": 0.5}), strategy_id=a)
        orch.on_signal("2025-01-09", pd.Series(dtype=float), strategy_id=b)
        orders = orch.rebalance()

        assert [order["side"] for order in orders] == ["sell", "buy"]
        assert orders[0]["strategy"] == b
        assert orders[0]["ticker"] == "Y"
        assert orders[1]["strategy"] == a
        assert orders[1]["ticker"] == "Z"

    def test_process_fills_executes_sells_before_buys(self, monkeypatch):
        orch, _, ids = build()
        sid = ids["A"]
        orch._last_prices = {"Y": 10.0}
        orch.on_signal("2025-01-08", pd.Series({"Y": 0.5}), strategy_id=sid)
        orch.rebalance()
        orch.process_fills({"Y": 10.0})
        orch.new_trading_day()

        # Create the buy first, then the sell: execution must still be
        # sells-first regardless of creation order.
        buy = orch.exec_engine.create_order("W", OrderSide.BUY, 100, strategy=sid)
        orch.exec_engine.submit_order(buy)
        sell = orch.exec_engine.create_order("Y", OrderSide.SELL, 100, strategy=sid)
        orch.exec_engine.submit_order(sell)

        recorded = []
        original = orch.exec_engine.process_fill

        def recording_fill(order, *args, **kwargs):
            recorded.append(order.side)
            return original(order, *args, **kwargs)

        monkeypatch.setattr(orch.exec_engine, "process_fill", recording_fill)
        orch.process_fills({"Y": 10.0, "W": 10.0})

        assert recorded == [OrderSide.SELL, OrderSide.BUY]
