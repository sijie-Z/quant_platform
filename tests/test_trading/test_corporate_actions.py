"""Paper-engine corporate actions: splits and cash dividends.

Pinned behaviour:
- a 2:1 split doubles quantity, halves average cost and leaves market value
  and cash unchanged;
- a cash dividend credits ``quantity * cash_per_share`` (optionally taxed)
  and keeps NAV continuous when the raw mark carries the ex-date drop;
- re-running the same day is a no-op, and the markers survive a restart
  through the Store;
- actions apply on the ex-date as the exchange day advances (including a
  suspended day, where the cached mark is scaled rather than trusted);
- real brokers (QMT/XTP) never apply actions locally.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderType,
    QMTBroker,
    SimulatedBroker,
    XTPBroker,
)
from quant_platform.trading.live_engine import LiveEngine
from quant_platform.trading.live_engine import Position as LivePosition
from quant_platform.trading.live_runner import LiveRunner


def _split(ex_date="2024-01-05", asset="600519.SH", ratio=2.0):
    return {
        "ex_date": ex_date, "asset": asset, "kind": "split",
        "cash_per_share": 0.0, "split_ratio": ratio,
    }


def _cash(ex_date="2024-01-05", asset="600519.SH", amount=0.5):
    return {
        "ex_date": ex_date, "asset": asset, "kind": "cash",
        "cash_per_share": amount, "split_ratio": 1.0,
    }


def _broker_with_position(cash_dividend_tax_rate: float = 0.0) -> SimulatedBroker:
    broker = SimulatedBroker(
        initial_cash=100_000, cash_dividend_tax_rate=cash_dividend_tax_rate,
    )
    assert broker.connect()
    order = Order(
        code="600519.SH", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=100, price=10.0,
    )
    result = broker.place_order(order)
    assert result.status.value == "filled"
    return broker


def test_two_for_one_split_keeps_market_value_and_cash():
    broker = _broker_with_position()
    cash_before = broker.get_account()["cash"]
    broker.new_trading_day()  # the holding is overnight, so all shares unlock

    applied = broker.apply_corporate_actions(pd.DataFrame([_split()]))

    assert len(applied) == 1
    position = broker.get_positions()[0]
    assert position.quantity == 200
    assert position.available == 200  # split-added shares inherit the unlocked holding
    assert position.avg_cost == pytest.approx(5.0)
    assert position.market_value == pytest.approx(1000.0)
    assert broker.get_account()["market_value"] == pytest.approx(1000.0)
    assert broker.get_account()["cash"] == pytest.approx(cash_before)
    assert broker.get_account()["total_equity"] == pytest.approx(99_995.0)


def test_split_reseeds_the_order_book_at_the_post_split_price():
    """A stale ladder seeded at 10 would fill a sell at 5 against a 10 bid."""
    broker = _broker_with_position()
    broker.new_trading_day()
    broker.apply_corporate_actions(pd.DataFrame([_split()]))

    sell = Order(
        code="600519.SH", side=OrderSide.SELL,
        order_type=OrderType.LIMIT, quantity=200, price=5.0,
    )
    result = broker.place_order(sell)

    assert result.status.value == "filled"
    assert result.filled_price == pytest.approx(5.0)
    # 98_995 (post-buy) + 1_000 - 5 commission - 1 stamp tax
    assert broker.get_account()["cash"] == pytest.approx(99_989.0)


def test_cash_dividend_credits_cash_and_nav_is_continuous():
    broker = _broker_with_position()
    before = broker.get_account()

    applied = broker.apply_corporate_actions(pd.DataFrame([_cash(amount=0.5)]))

    assert len(applied) == 1
    position = broker.get_positions()[0]
    after = broker.get_account()
    assert after["cash"] == pytest.approx(before["cash"] + 50.0)
    # The cached mark absorbs the ex-dividend drop until a quote arrives.
    assert position.current_price == pytest.approx(9.5)
    assert after["market_value"] == pytest.approx(950.0)
    assert after["total_equity"] == pytest.approx(before["total_equity"])
    ledger = broker.get_cash_ledger()
    assert len(ledger) == 1
    assert ledger[0]["cash_delta"] == pytest.approx(50.0)


def test_cash_dividend_withholding_tax_is_configurable():
    broker = _broker_with_position(cash_dividend_tax_rate=0.10)
    before = broker.get_account()["cash"]

    broker.apply_corporate_actions(pd.DataFrame([_cash(amount=0.5)]))

    assert broker.get_account()["cash"] == pytest.approx(before + 45.0)
    assert broker.get_cash_ledger()[0]["tax"] == pytest.approx(5.0)


def test_combined_cash_and_split_event_applies_both_components():
    """A 送转+派息 event arrives as one cash row plus one split row sharing
    the (asset, ex_date) marker; the second component must not be skipped."""
    broker = _broker_with_position()
    broker.new_trading_day()
    actions = pd.DataFrame([{
        "ex_date": "2024-01-05", "asset": "600519.SH", "kind": "cash",
        "cash_per_share": 1.0, "split_ratio": 2.0,
    }])

    applied = broker.apply_corporate_actions(actions)

    assert len(applied) == 1
    assert applied[0]["kind"] == "cash+split"
    position = broker.get_positions()[0]
    assert position.quantity == 200
    assert position.avg_cost == pytest.approx(5.0)
    # The dividend is paid on the pre-split 100 shares.
    assert broker.get_account()["cash"] == pytest.approx(99_095.0)
    assert broker.get_account()["total_equity"] == pytest.approx(99_995.0)


def test_same_day_replay_does_not_double_apply():
    broker = _broker_with_position()
    actions = pd.DataFrame([_split(), _cash(ex_date="2024-01-06", amount=0.5)])

    first = broker.apply_corporate_actions(actions)
    second = broker.apply_corporate_actions(actions)

    assert len(first) == 2
    assert second == []
    position = broker.get_positions()[0]
    assert position.quantity == 200
    assert broker.get_account()["cash"] == pytest.approx(99_095.0)
    assert len(broker.get_corporate_action_events()) == 2
    assert len(broker.get_applied_corporate_actions()) == 2


def test_restart_replay_is_idempotent_through_the_store(tmp_path):
    store = Store(str(tmp_path / "trading.db"))
    broker = _broker_with_position()
    broker.configure_corporate_action_persistence(store)
    actions = pd.DataFrame([_split()])
    broker.set_corporate_actions(actions)
    assert broker.apply_corporate_actions()  # applies every loaded action

    restored = SimulatedBroker(initial_cash=100_000)
    restored.configure_corporate_action_persistence(store)
    restored.restore_positions(
        [{
            "code": p.code, "quantity": p.quantity, "available": p.available,
            "avg_cost": p.avg_cost, "current_price": p.current_price,
        } for p in broker.get_positions()],
        cash=broker.get_account()["cash"],
    )
    restored.set_corporate_actions(actions)

    assert restored.apply_corporate_actions() == []
    position = restored.get_positions()[0]
    assert position.quantity == 200
    assert restored.get_account()["cash"] == pytest.approx(
        broker.get_account()["cash"]
    )
    assert store.get_engine_state("paper_corporate_actions")["applied"] == [
        ["600519.SH", "2024-01-05"],
    ]
    store.close()


def test_actions_apply_only_when_the_exchange_day_reaches_the_ex_date():
    broker = _broker_with_position()
    broker.set_corporate_actions(pd.DataFrame([_split(ex_date="2024-01-05")]))

    assert broker.new_trading_day(date="2024-01-04") == []
    assert broker.get_positions()[0].quantity == 100

    applied = broker.new_trading_day(date="2024-01-05")
    assert len(applied) == 1
    assert broker.get_positions()[0].quantity == 200
    # Re-advancing to the same/next day does not re-apply.
    assert broker.new_trading_day(date="2024-01-08") == []


def test_split_on_a_suspended_day_scales_the_stale_mark():
    """No quote on the ex-date: the cached mark is scaled, so NAV cannot jump."""
    broker = _broker_with_position()
    before = broker.get_account()
    assert broker.get_positions()[0].current_price == 10.0

    broker.apply_corporate_actions(pd.DataFrame([_split(ex_date="2024-01-05")]))

    position = broker.get_positions()[0]
    assert position.quantity == 200
    assert position.current_price == pytest.approx(5.0)
    assert broker.get_account()["total_equity"] == pytest.approx(
        before["total_equity"]
    )


def test_stale_pre_split_quote_is_not_applied_after_a_split():
    """A suspended stock keeps feeding its pre-split close; doubling the
    quantity at that price would double the position's market value."""
    broker = _broker_with_position()
    broker.new_trading_day()
    broker.apply_corporate_actions(pd.DataFrame([_split()]))

    broker.update_market_prices({"600519.SH": 10.0})  # stale pre-split close

    position = broker.get_positions()[0]
    assert position.quantity == 200
    assert position.current_price == pytest.approx(5.0)
    assert broker.get_account()["market_value"] == pytest.approx(1000.0)

    broker.update_market_prices({"600519.SH": 5.2})  # plausible post-split quote
    assert broker.get_positions()[0].current_price == pytest.approx(5.2)


def test_real_brokers_do_not_apply_actions_locally():
    assert QMTBroker.supports_corporate_actions is False
    assert XTPBroker.supports_corporate_actions is False
    assert not hasattr(QMTBroker, "apply_corporate_actions")
    assert not hasattr(XTPBroker, "apply_corporate_actions")


# ---------------------------------------------------------------------------
# LiveEngine (alpha-v1.0 paper runtime)
# ---------------------------------------------------------------------------


def _live_engine():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=["600519.SH"])
    prices = pd.DataFrame(10.0, index=dates, columns=["600519.SH"])
    prices.loc[dates[2], "600519.SH"] = 5.0
    prices.loc[dates[3], "600519.SH"] = 4.5
    prices.attrs["price_basis"] = "raw"
    engine = LiveEngine()
    engine.load_history(returns, prices)
    engine.state.cash = 0.0
    engine.state.equity_peak = 1000.0
    engine.state.positions = [
        LivePosition(
            "600519.SH", str(dates[0])[:10], str(dates[-1])[:10], 100, 10.0,
        ),
    ]
    return engine, dates


def test_live_engine_applies_split_and_dividend_without_nav_jump():
    engine, dates = _live_engine()
    engine.set_corporate_actions(pd.DataFrame([
        _split(ex_date=dates[2]),
        _cash(ex_date=dates[3], amount=0.5),
    ]))

    first = engine.run_once(dates[1])
    assert "corporate_actions" not in first
    assert engine.state.positions[0].shares == 100

    split_day = engine.run_once(dates[2])
    assert split_day["corporate_actions"] == 1
    assert engine.state.positions[0].shares == 200
    assert engine.state.current_equity == pytest.approx(1000.0)

    dividend_day = engine.run_once(dates[3])
    assert dividend_day["corporate_actions"] == 1
    assert engine.state.cash == pytest.approx(100.0)  # 200 x 0.5
    assert engine.state.current_equity == pytest.approx(1000.0)
    assert len(engine.cash_ledger) == 1


def test_live_engine_combined_event_applies_both_components():
    engine, dates = _live_engine()
    engine.set_corporate_actions(pd.DataFrame([{
        "ex_date": dates[2], "asset": "600519.SH", "kind": "cash",
        "cash_per_share": 1.0, "split_ratio": 2.0,
    }]))

    result = engine.run_once(dates[2])

    assert result["corporate_actions"] == 1
    position = engine.state.positions[0]
    assert position.shares == 200
    assert engine.state.cash == pytest.approx(100.0)  # 100 pre-split x 1.0


def test_live_engine_catch_up_skips_positions_entered_after_the_ex_date():
    """An old action replayed from a full-history feed must not pay a
    position that was opened after its ex-date."""
    engine, dates = _live_engine()
    engine.state.positions = [
        LivePosition(
            "600519.SH", str(dates[2])[:10], str(dates[-1])[:10], 100, 5.0,
        ),
    ]
    engine.state.equity_peak = 500.0  # 100 x 5.0; avoids the kill switch
    engine.set_corporate_actions(pd.DataFrame([_split(ex_date=dates[1])]))

    engine.run_once(dates[2])  # catches up dates[1]'s split

    assert engine.state.positions[0].shares == 100
    assert engine.state.cash == pytest.approx(0.0)


def test_live_engine_ignores_a_stale_pre_split_mark_on_a_suspended_day():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=["600519.SH"])
    prices = pd.DataFrame(10.0, index=dates, columns=["600519.SH"])
    prices.loc[dates[3], "600519.SH"] = 5.2  # resume after the split
    prices.attrs["price_basis"] = "raw"
    engine = LiveEngine()
    engine.load_history(returns, prices)
    engine.state.cash = 0.0
    engine.state.equity_peak = 1000.0
    engine.state.positions = [
        LivePosition(
            "600519.SH", str(dates[0])[:10], str(dates[-1])[:10], 100, 10.0,
        ),
    ]
    engine.set_corporate_actions(pd.DataFrame([_split(ex_date=dates[2])]))

    engine.run_once(dates[1])
    ex_day = engine.run_once(dates[2])  # suspended: the panel repeats 10.0

    assert ex_day["corporate_actions"] == 1
    assert engine.state.positions[0].shares == 200
    assert engine.state.current_equity == pytest.approx(1000.0)

    engine.run_once(dates[3])  # real post-split quote
    assert engine.state.current_equity == pytest.approx(1040.0)


def test_live_engine_markers_survive_snapshot_restore():
    engine, dates = _live_engine()
    actions = pd.DataFrame([_split(ex_date=dates[2])])
    engine.set_corporate_actions(actions)
    engine.run_once(dates[2])
    snapshot = engine.to_snapshot()

    restored, _ = _live_engine()
    restored.restore_snapshot(snapshot)
    restored.set_corporate_actions(actions)

    assert restored.apply_corporate_actions(date=dates[2]) == []
    assert restored.state.positions[0].shares == 200
    assert restored.to_snapshot() == snapshot


def test_live_engine_does_not_apply_actions_of_a_real_broker():
    class _RealBrokerStub:
        supports_corporate_actions = False

    engine = LiveEngine(broker=_RealBrokerStub())
    engine.set_corporate_actions(pd.DataFrame([_split()]))
    assert engine.apply_corporate_actions(date="2024-01-05") == []


# ---------------------------------------------------------------------------
# LiveRunner paper path
# ---------------------------------------------------------------------------


class _RealBrokerStub(BrokerInterface):
    """Minimal real-broker shape: never applies corporate actions locally."""

    supports_corporate_actions = False

    def __init__(self):
        self.corporate_calls = 0

    def connect(self):
        return True

    def place_order(self, order):
        return order

    def cancel_order(self, order_id):
        return True

    def get_positions(self):
        return []

    def get_orders(self, status=None):
        return []

    def get_account(self):
        return {"cash": 0.0, "total_equity": 0.0, "n_positions": 0}

    def set_corporate_actions(self, actions):  # pragma: no cover - must not run
        self.corporate_calls += 1


@pytest.fixture
def runner_with_position(tmp_path):
    store = Store(str(tmp_path / "runner.db"))
    broker = SimulatedBroker(initial_cash=100_000)
    runner = LiveRunner(
        broker=broker, store=store, dual_track=False,
        risk_monitor=RiskMonitor(),
    )
    runner.set_universe(["600519"])
    broker.place_order(Order(
        code="600519.SH", side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=100, price=10.0,
    ))
    yield runner, broker, store
    store.close()


def test_live_runner_paper_path_applies_actions_and_persists_events(
    runner_with_position,
):
    runner, broker, store = runner_with_position
    actions = pd.DataFrame([
        _split(ex_date="2024-01-03"),
        _cash(ex_date="2024-01-04", amount=0.5),
    ])
    assert runner.set_corporate_actions(actions) is True

    runner.run_once(date="2024-01-02", prices={"600519.SH": 10.0})
    first = runner.run_once(date="2024-01-03", prices={"600519.SH": 5.0})
    assert first.price_basis == "raw"
    position = broker.get_positions()[0]
    assert position.quantity == 200

    second = runner.run_once(date="2024-01-04", prices={"600519.SH": 4.5})
    assert second.cash == pytest.approx(99_095.0)  # 98_995 + 200 x 0.5
    events = store.get_events(topic="corporate_action", limit=10)
    assert len(events) == 2
    markers = store.get_engine_state("paper_corporate_actions")["applied"]
    assert ["600519.SH", "2024-01-03"] in markers
    assert ["600519.SH", "2024-01-04"] in markers


def test_live_runner_never_forwards_actions_to_a_real_broker(tmp_path):
    store = Store(str(tmp_path / "runner_real.db"))
    broker = _RealBrokerStub()
    runner = LiveRunner(
        broker=broker, store=store, dual_track=False,
        risk_monitor=RiskMonitor(),
    )

    assert runner.set_corporate_actions(pd.DataFrame([_split()])) is False
    assert broker.corporate_calls == 0
    store.close()
