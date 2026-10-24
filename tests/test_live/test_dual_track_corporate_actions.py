"""Dual-track TCA shadow: corporate actions on the PaperBroker book.

The parallel ``PaperBroker`` used by ``LiveRunner`` for TCA comparison now
keeps a fill-based comparison book.  The same split/dividend feed the
primary paper book receives is applied to it on each exchange-day advance,
so the shadow does not silently diverge from the book it is compared with.
"""

from __future__ import annotations

import pandas as pd
import pytest
from quant_platform.core.store import Store
from quant_platform.execution.paper_broker import LatencyModel, PaperBroker
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import Order, OrderSide, OrderType
from quant_platform.trading.live_runner import LiveRunner


def _action(ex_date, kind, cash=0.0, ratio=1.0, asset="600519.SH"):
    return {
        "ex_date": ex_date, "asset": asset, "kind": kind,
        "cash_per_share": cash, "split_ratio": ratio,
    }


def _buy(pb: PaperBroker, quantity: int = 100, price: float = 10.0):
    pb.seed_book_from_snapshot("600519.SH", {
        "bids": [[price, 10_000]], "asks": [[price, 10_000]],
    })
    record = pb.place_order(Order(
        code="600519.SH", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=quantity, price=price,
    ))
    assert record.filled_qty == quantity


def test_paper_broker_applies_split_and_dividend_to_the_comparison_book():
    pb = PaperBroker(
        initial_cash=100_000, latency=LatencyModel.ZERO,
        partial_fill_rate=0.0,
    )
    _buy(pb, quantity=100, price=10.0)
    assert pb.get_positions()["600519.SH"]["quantity"] == 100
    assert pb.get_cash() == pytest.approx(99_000.0)

    assert pb.set_corporate_actions(pd.DataFrame([
        _action("2024-01-05", "cash", cash=0.5),
        _action("2024-01-05", "split", ratio=2.0),
    ])) == 2

    applied = pb.new_trading_day("2024-01-05")
    assert len(applied) == 1, "one combined event shares one marker"
    assert applied[0]["kind"] == "cash+split"
    position = pb.get_positions()["600519.SH"]
    assert position["quantity"] == 200
    assert position["avg_cost"] == pytest.approx(5.0)
    # Cash first, on the pre-split quantity: 100 x 0.5 = 50.
    assert pb.get_cash() == pytest.approx(99_000.0 + 50.0)
    assert len(pb.get_cash_ledger()) == 1
    assert pb.get_applied_corporate_actions() == [("600519.SH", "2024-01-05")]
    # The stale pre-action ladder is dropped; the next cycle re-seeds it.
    assert "600519.SH" not in pb._books

    # Replaying the same ex-date is a no-op.
    assert pb.new_trading_day("2024-01-05") == []
    assert pb.get_positions()["600519.SH"]["quantity"] == 200
    assert pb.get_cash() == pytest.approx(99_050.0)


def test_paper_broker_does_not_mint_cash_on_oversized_sells():
    """The shadow mirrors primary sell sizes; it must credit only held shares."""
    pb = PaperBroker(
        initial_cash=100_000, latency=LatencyModel.ZERO,
        partial_fill_rate=0.0,
    )
    pb.seed_book_from_snapshot("600519.SH", {
        "bids": [[10.0, 10_000]], "asks": [[10.0, 10_000]],
    })
    record = pb.place_order(Order(
        code="600519.SH", side=OrderSide.SELL,
        order_type=OrderType.LIMIT, quantity=100, price=10.0,
    ))
    assert record.filled_qty == 100
    assert pb.get_cash() == pytest.approx(100_000.0), (
        "a sell with no shadow holdings credited fictitious cash"
    )

    _buy(pb, quantity=100, price=10.0)
    cash_before_sell = pb.get_cash()
    record = pb.place_order(Order(
        code="600519.SH", side=OrderSide.SELL,
        order_type=OrderType.LIMIT, quantity=300, price=10.0,
    ))
    assert record.filled_qty == 300  # the LOB fills; the shadow only owns 100
    assert pb.get_cash() == pytest.approx(cash_before_sell + 100 * 10.0)
    assert pb.get_positions()["600519.SH"]["quantity"] == 0


def _runner(tmp_path, dual_track: bool = True) -> LiveRunner:
    return LiveRunner(
        broker_type="simulated",
        initial_cash=1_000_000,
        dual_track=dual_track,
        risk_monitor=RiskMonitor(),
        store=Store(str(tmp_path / "runner.db")),
    )


def _fixed_signals(*signals):
    return lambda: list(signals)


def test_live_runner_forwards_the_feed_to_the_dual_track_book(tmp_path):
    runner = _runner(tmp_path)
    runner.set_universe(["600519"])
    feed = pd.DataFrame([
        _action("2024-01-03", "split", ratio=2.0),
        _action("2024-01-04", "cash", cash=0.5),
    ])
    assert runner.set_corporate_actions(feed, source="test") is True

    runner.set_prices({"600519": 10.0})
    runner._generate_signals = _fixed_signals({
        "code": "600519", "side": "buy", "target_value": 100_000.0,
        "strength": 0.9, "reason": "test",
    })
    runner.run_once(date="2024-01-02", prices={"600519": 10.0})
    shadow = runner._paper_broker
    first_quantity = shadow.get_positions()["600519.SH"]["quantity"]
    assert first_quantity > 0, "the dual-track book never bought"

    # Split ex-date: the shadow doubles its own quantity and halves cost.
    runner._generate_signals = _fixed_signals()
    runner.run_once(date="2024-01-03", prices={"600519": 5.0})
    assert shadow.get_positions()["600519.SH"]["quantity"] == (
        first_quantity * 2
    )
    assert shadow.get_positions()["600519.SH"]["avg_cost"] == pytest.approx(
        5.0,
    )
    cash_before_dividend = shadow.get_cash()

    # Cash ex-date: the shadow credits the dividend on its held quantity.
    runner.run_once(date="2024-01-04", prices={"600519": 4.5})
    expected = first_quantity * 2 * 0.5
    assert shadow.get_cash() == pytest.approx(
        cash_before_dividend + expected,
    )
    events = shadow.get_corporate_action_events()
    assert [event["date"] for event in events] == [
        "2024-01-03", "2024-01-04",
    ]
    # Run-report provenance reflects the primary paper book.
    provenance = runner.get_corporate_actions_applied()
    assert provenance["count"] == 2
    assert provenance["last_ex_date"] == "2024-01-04"
    assert provenance["source"] == "test"
