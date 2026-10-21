"""ExecutionEngine corporate-action application (paper/backtest book).

The canonical execution engine must apply splits/dividends exactly once,
keep equity continuous, publish a ``corporate_action`` event and record a
cash-ledger entry; markers can be exported/restored across a restart.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.execution.engine import ExecutionEngine
from quant_platform.execution.models import OrderSide


def _engine_with_position() -> ExecutionEngine:
    engine = ExecutionEngine(initial_cash=100_000)
    order = engine.create_order("600519.SH", OrderSide.BUY, 100)
    engine.submit_order(order)
    engine.process_fill(order, price=10.0, quantity=100)
    return engine


def _split(ex_date="2024-01-05", ratio=2.0) -> dict:
    return {
        "ex_date": ex_date, "asset": "600519.SH", "kind": "split",
        "cash_per_share": 0.0, "split_ratio": ratio,
    }


def _cash(ex_date="2024-01-05", amount=0.5) -> dict:
    return {
        "ex_date": ex_date, "asset": "600519.SH", "kind": "cash",
        "cash_per_share": amount, "split_ratio": 1.0,
    }


def test_split_scales_position_and_keeps_equity():
    engine = _engine_with_position()
    cash_before = engine.cash
    equity_before = engine.cash + engine.get_position("600519.SH").market_value

    applied = engine.apply_corporate_actions(pd.DataFrame([_split()]))

    assert len(applied) == 1
    position = engine.get_position("600519.SH")
    assert position.quantity == 200
    assert position.avg_cost == pytest.approx(5.0)
    assert position.current_price == pytest.approx(5.0)
    assert position.market_value == pytest.approx(1000.0)
    assert engine.cash == pytest.approx(cash_before)
    assert engine.cash + position.market_value == pytest.approx(equity_before)


def test_cash_dividend_credits_cash_and_keeps_equity():
    engine = _engine_with_position()
    equity_before = engine.cash + engine.get_position("600519.SH").market_value

    engine.apply_corporate_actions(pd.DataFrame([_cash(amount=0.5)]))

    position = engine.get_position("600519.SH")
    assert engine.cash == pytest.approx(99_050.0)
    assert position.current_price == pytest.approx(9.5)
    assert engine.cash + position.market_value == pytest.approx(equity_before)
    assert position.realized_pnl == pytest.approx(50.0)
    ledger = engine.get_cash_ledger()
    assert len(ledger) == 1
    assert ledger[0]["cash_delta"] == pytest.approx(50.0)


def test_combined_cash_and_split_applies_cash_on_pre_split_quantity():
    engine = _engine_with_position()

    applied = engine.apply_corporate_actions(pd.DataFrame([{
        "ex_date": "2024-01-05", "asset": "600519.SH", "kind": "cash",
        "cash_per_share": 1.0, "split_ratio": 2.0,
    }]))

    assert len(applied) == 1
    assert applied[0]["kind"] == "cash+split"
    position = engine.get_position("600519.SH")
    assert position.quantity == 200
    assert position.avg_cost == pytest.approx(5.0)
    assert engine.cash == pytest.approx(99_100.0)  # 99_000 + 100 x 1.0


def test_replay_is_idempotent_and_markers_are_exportable():
    engine = _engine_with_position()
    actions = pd.DataFrame([_split(), _cash(ex_date="2024-01-06")])

    assert len(engine.apply_corporate_actions(actions)) == 2
    assert engine.apply_corporate_actions(actions) == []
    position = engine.get_position("600519.SH")
    assert position.quantity == 200
    assert engine.cash == pytest.approx(99_100.0)

    markers = engine.corporate_action_markers
    assert markers == [("600519.SH", "2024-01-05"), ("600519.SH", "2024-01-06")]

    restarted = _engine_with_position()
    restarted.restore_corporate_action_markers({"applied": markers})
    restarted.apply_corporate_actions(actions)
    assert restarted.get_position("600519.SH").quantity == 100
    assert restarted.cash == pytest.approx(99_000.0)


def test_ex_date_cutoff_leaves_future_actions_pending():
    engine = _engine_with_position()
    actions = pd.DataFrame([_split(ex_date="2024-01-05"), _cash(ex_date="2024-01-08")])

    assert len(engine.apply_corporate_actions(actions, ex_date="2024-01-05")) == 1
    position = engine.get_position("600519.SH")
    assert position.quantity == 200
    assert engine.cash == pytest.approx(99_000.0)

    assert len(engine.apply_corporate_actions(actions, ex_date="2024-01-08")) == 1
    assert engine.cash == pytest.approx(99_100.0)


def test_dated_new_trading_day_applies_the_loaded_feed_once():
    engine = _engine_with_position()
    engine.set_corporate_actions(pd.DataFrame([_split(ex_date="2024-01-05")]))

    assert engine.new_trading_day(date="2024-01-04") == []
    assert engine.get_position("600519.SH").quantity == 100

    applied = engine.new_trading_day(date="2024-01-05")
    assert len(applied) == 1
    position = engine.get_position("600519.SH")
    assert position.quantity == 200
    assert position.available == 200  # overnight holding is unlocked
    assert engine.new_trading_day(date="2024-01-08") == []


def test_corporate_action_event_is_published():
    from quant_platform.core.events import get_event_bus

    engine = _engine_with_position()
    engine.apply_corporate_actions(pd.DataFrame([_cash(amount=0.5)]))

    history = get_event_bus().get_history(topic="corporate_action", limit=50)
    payloads = [
        item["data"] for item in history
        if isinstance(item.get("data"), dict)
        and item["data"].get("ticker") == "600519.SH"
        and item["data"].get("date") == "2024-01-05"
        and item.get("source") == "execution_engine"
    ]
    assert payloads, "no corporate_action event for the dividend was published"
    payload = payloads[-1]
    assert payload["kind"] == "cash"
    assert payload["cash_delta"] == pytest.approx(50.0)
