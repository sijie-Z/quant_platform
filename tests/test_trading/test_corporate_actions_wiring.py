"""Web ``LiveTradingEngine`` corporate-action wiring (paper book only).

The engine used to call ``broker.new_trading_day()`` with no date and never
loaded a feed, so a paper run through the web engine never applied a split
or a cash dividend.  These tests pin the wiring:

- a DataFrame feed (and a callable/provider feed) is applied exactly once on
  the ex-date, before sizing/marking;
- quantity, average cost, cash and equity reconcile; the NAV has no jump at
  the action;
- a same-day restart does not re-apply an ex-date (markers persist in the
  Store);
- a real broker never receives a local feed;
- a provider with no feed (empty schema) runs unchanged.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_platform.core.store import Store
from quant_platform.data.providers.base import DataProvider, empty_corporate_actions
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderType,
    SimulatedBroker,
)
from quant_platform.trading.engine import LiveTradingEngine

TZ = ZoneInfo("Asia/Shanghai")
ASSET = "600519.SH"


def _action(ex_date, kind, cash=0.0, ratio=1.0, asset=ASSET):
    return {
        "ex_date": ex_date, "asset": asset, "kind": kind,
        "cash_per_share": cash, "split_ratio": ratio,
    }


def _feed():
    return pd.DataFrame([
        _action("2024-01-04", "cash", cash=0.5),
        _action("2024-01-05", "split", ratio=2.0),
    ])


def _fixed_engine(broker, store, monkeypatch, **kwargs):
    """Engine whose clock, quote fetch and signals are deterministic."""
    engine = LiveTradingEngine(
        broker=broker,
        store=store,
        risk_monitor=RiskMonitor(),
        **kwargs,
    )
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    monkeypatch.setattr(engine, "_generate_signals", lambda positions=None: [])
    return engine


def _at(engine, day: str) -> None:
    fixed = datetime.fromisoformat(day).replace(hour=10, tzinfo=TZ)
    engine._exchange_now = lambda: fixed  # type: ignore[method-assign]


def _buy_100(broker: SimulatedBroker) -> None:
    assert broker.connect()
    result = broker.place_order(Order(
        code=ASSET, side=OrderSide.BUY,
        order_type=OrderType.LIMIT, quantity=100, price=10.0,
    ))
    assert result.status.value == "filled"


def _position(broker: SimulatedBroker):
    positions = [p for p in broker.get_positions() if p.code == ASSET]
    assert len(positions) == 1
    return positions[0]


def test_web_engine_applies_split_and_dividend_once_with_continuous_nav(
    tmp_path, monkeypatch,
):
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    engine = _fixed_engine(
        broker, store, monkeypatch, corporate_actions=_feed(),
    )

    # Day before the cash ex-date: nothing applies yet.
    _at(engine, "2024-01-03")
    engine.run_once()
    assert _position(broker).quantity == 100
    assert engine.get_corporate_actions_applied()["count"] == 0
    assert engine._corporate_actions_loaded is True
    assert engine._corporate_action_feed_rows == 2

    # Cash ex-date: 100 x 0.5 credited, mark drops by 0.5 -> equity flat.
    _at(engine, "2024-01-04")
    before = broker.get_account()["total_equity"]
    engine.run_once()
    position = _position(broker)
    assert position.quantity == 100
    assert position.avg_cost == pytest.approx(10.0)
    assert broker.get_account()["cash"] == pytest.approx(98_995.0 + 50.0)
    after = broker.get_account()["total_equity"]
    assert after == pytest.approx(before, abs=1e-6), (
        "the cash dividend caused a NAV jump"
    )

    # Split ex-date: quantity doubles, average cost halves, equity flat.
    _at(engine, "2024-01-05")
    before = broker.get_account()["total_equity"]
    engine.run_once()
    position = _position(broker)
    assert position.quantity == 200
    assert position.avg_cost == pytest.approx(5.0)
    assert broker.get_account()["total_equity"] == pytest.approx(
        before, abs=1e-6,
    ), "the split caused a NAV jump"

    provenance = engine.get_corporate_actions_applied()
    assert provenance == {
        "count": 2,
        "last_ex_date": "2024-01-05",
        "source": "constructor",
    }
    state = engine.get_state()
    assert state["corporate_actions_applied"]["count"] == 2
    assert state["corporate_actions_feed"]["loaded"] is True
    store.close()


def test_same_day_restart_does_not_reapply(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    engine = _fixed_engine(
        broker, store, monkeypatch, corporate_actions=_feed(),
    )
    _at(engine, "2024-01-04")
    engine.run_once()
    _at(engine, "2024-01-05")
    engine.run_once()
    assert _position(broker).quantity == 200
    cash_after_actions = broker.get_account()["cash"]

    # Same-day restart: a fresh broker (markers reloaded from the Store via
    # the engine's persistence wiring) and a fresh engine on the same date.
    restarted_broker = SimulatedBroker(initial_cash=100_000)
    assert restarted_broker.connect()
    restarted_broker.restore_positions(
        [p.to_dict() for p in broker.get_positions()], cash_after_actions,
    )
    engine2 = _fixed_engine(
        restarted_broker, store, monkeypatch, corporate_actions=_feed(),
    )
    _at(engine2, "2024-01-05")
    engine2.run_once()

    assert _position(restarted_broker).quantity == 200, (
        "a same-day restart re-applied the split"
    )
    assert restarted_broker.get_account()["cash"] == pytest.approx(
        cash_after_actions,
    )
    assert engine2.get_corporate_actions_applied()["count"] == 0
    store.close()


def test_callable_feed_is_resolved_lazily_on_the_day_advance(
    tmp_path, monkeypatch,
):
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    calls: list[tuple[str, str]] = []

    def loader(start_date, end_date):
        calls.append((start_date, end_date))
        return _feed()

    engine = _fixed_engine(
        broker, store, monkeypatch, corporate_actions=loader,
    )
    assert engine._corporate_actions_loaded is False
    _at(engine, "2024-01-05")
    engine.run_once()

    assert calls and calls[0][1] == "2024-01-05", (
        "the callable must receive the exchange day being advanced to"
    )
    assert _position(broker).quantity == 200
    assert engine.get_corporate_actions_applied()["source"].startswith(
        "callable:"
    )
    store.close()


class _FeedProvider(DataProvider):
    """Provider that serves a fixed action frame (or the empty schema)."""

    def __init__(self, frame=None):
        self._frame = frame
        self.windows: list[tuple[str, str]] = []

    def get_corporate_actions(self, start_date, end_date):
        self.windows.append((start_date, end_date))
        return (
            empty_corporate_actions() if self._frame is None
            else self._frame.copy()
        )

    def get_prices(self, start_date, end_date, fields=None):
        return pd.DataFrame()

    def get_financials(self, start_date, end_date):
        return pd.DataFrame()

    def get_benchmark(self, start_date, end_date):
        return pd.Series(dtype=float)

    def get_metadata(self):
        return pd.DataFrame()


def test_injected_provider_feed_is_loaded_on_first_advance(
    tmp_path, monkeypatch,
):
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    provider = _FeedProvider(_feed())
    engine = _fixed_engine(
        broker, store, monkeypatch, data_provider=provider,
    )
    _at(engine, "2024-01-05")
    engine.run_once()

    assert provider.windows, "the injected provider was never queried"
    start, end = provider.windows[0]
    assert end == "2024-01-05"
    assert start < end, "the provider window must include pending ex-dates"
    assert _position(broker).quantity == 200
    assert engine.get_corporate_actions_applied()["source"] == (
        "provider:_FeedProvider"
    )
    store.close()


def test_dynamic_provider_refreshes_on_every_day_advance(
    tmp_path, monkeypatch,
):
    """An action announced after engine start must still be applied."""
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))

    class _SequencedProvider(_FeedProvider):
        def __init__(self):
            super().__init__(None)
            self.calls = 0

        def get_corporate_actions(self, start_date, end_date):
            self.windows.append((start_date, end_date))
            self.calls += 1
            if self.calls == 1:
                return empty_corporate_actions()
            return pd.DataFrame([_action("2024-01-05", "split", ratio=2.0)])

    provider = _SequencedProvider()
    engine = _fixed_engine(
        broker, store, monkeypatch, data_provider=provider,
    )

    _at(engine, "2024-01-04")
    engine.run_once()
    assert _position(broker).quantity == 100

    _at(engine, "2024-01-05")
    engine.run_once()
    assert _position(broker).quantity == 200, (
        "the provider feed was loaded only once; the ex-date announced "
        "after engine start was missed"
    )
    assert provider.windows == [
        ("2022-11-30", "2024-01-04"), ("2024-01-04", "2024-01-05"),
    ]
    store.close()


def test_provider_window_starts_at_the_last_processed_day(
    tmp_path, monkeypatch,
):
    """A restart must not ask for (and apply) pre-book historic actions."""
    broker = SimulatedBroker(initial_cash=100_000)
    assert broker.connect()
    store = Store(str(tmp_path / "engine.db"))
    provider = _FeedProvider(_feed())
    engine = _fixed_engine(
        broker, store, monkeypatch, data_provider=provider,
    )
    engine._trading_day = "2024-01-04"
    _at(engine, "2024-01-05")
    engine.run_once()

    assert provider.windows[0] == ("2024-01-04", "2024-01-05")
    store.close()


def test_provider_with_no_feed_runs_unchanged(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    engine = _fixed_engine(
        broker, store, monkeypatch, data_provider=_FeedProvider(None),
    )
    _at(engine, "2024-01-05")
    before = broker.get_account()["total_equity"]
    engine.run_once()

    assert _position(broker).quantity == 100
    assert broker.get_account()["total_equity"] == pytest.approx(
        before, abs=1e-6,
    )
    assert engine.get_corporate_actions_applied() == {
        "count": 0, "last_ex_date": None, "source": "provider:_FeedProvider",
    }
    assert engine.get_state()["corporate_actions_feed"]["rows"] == 0
    store.close()


class _RealBrokerStub(BrokerInterface):
    """Minimal real-broker shape: settles corporate actions remotely."""

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


def test_timezone_aware_ex_date_matches_the_exchange_day(
    tmp_path, monkeypatch,
):
    """A UTC ex-date is pinned to the Shanghai exchange day it represents."""
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    feed = pd.DataFrame([{
        # 2024-01-04 16:00 UTC == 2024-01-05 00:00 Asia/Shanghai.
        "ex_date": pd.Timestamp("2024-01-04 16:00", tz="UTC"),
        "asset": ASSET, "kind": "split",
        "cash_per_share": 0.0, "split_ratio": 2.0,
    }])
    engine = _fixed_engine(
        broker, store, monkeypatch, corporate_actions=feed,
    )

    _at(engine, "2024-01-04")
    engine.run_once()
    assert _position(broker).quantity == 100, (
        "the UTC timestamp was mis-dated to the previous exchange day"
    )

    _at(engine, "2024-01-05")
    engine.run_once()
    assert _position(broker).quantity == 200
    assert engine.get_corporate_actions_applied()["last_ex_date"] == (
        "2024-01-05"
    )
    store.close()


def test_invalid_provider_feed_is_non_fatal(tmp_path, monkeypatch):
    """A broken feed must not abort the trading cycle (fail closed)."""
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))

    class _BrokenProvider(_FeedProvider):
        def get_corporate_actions(self, start_date, end_date):
            return pd.DataFrame({"not": ["the contract"]})

    engine = _fixed_engine(
        broker, store, monkeypatch, data_provider=_BrokenProvider(),
    )
    _at(engine, "2024-01-05")
    engine.run_once()  # must not raise

    assert _position(broker).quantity == 100
    assert engine.get_corporate_actions_applied()["count"] == 0
    store.close()


def test_duplicate_timezone_rows_collapse_to_one_event(tmp_path, monkeypatch):
    """Two tz stamps for the same Shanghai day must not double a split."""
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    feed = pd.DataFrame([
        {
            "ex_date": pd.Timestamp("2024-01-05 00:30", tz="Asia/Shanghai"),
            "asset": ASSET, "kind": "split",
            "cash_per_share": 0.0, "split_ratio": 2.0,
        },
        {
            "ex_date": pd.Timestamp("2024-01-04 16:30", tz="UTC"),
            "asset": ASSET, "kind": "split",
            "cash_per_share": 0.0, "split_ratio": 2.0,
        },
    ])
    engine = _fixed_engine(
        broker, store, monkeypatch, corporate_actions=feed,
    )
    _at(engine, "2024-01-05")
    engine.run_once()

    assert _position(broker).quantity == 200, (
        "duplicate rows for one exchange date multiplied the split twice"
    )
    assert engine._corporate_action_feed_rows == 1
    store.close()


def test_mixed_timezone_feed_keeps_every_row(tmp_path, monkeypatch):
    """Naive and tz-aware ex-dates in one feed must not drop rows."""
    broker = SimulatedBroker(initial_cash=100_000)
    _buy_100(broker)
    store = Store(str(tmp_path / "engine.db"))
    feed = [
        {
            # 2024-01-05 00:00 Asia/Shanghai == 2024-01-04 16:00 UTC.
            "ex_date": "2024-01-04 16:00+00:00", "asset": ASSET,
            "kind": "cash", "cash_per_share": 0.5, "split_ratio": 1.0,
        },
        {
            "ex_date": "2024-01-05", "asset": ASSET,
            "kind": "split", "cash_per_share": 0.0, "split_ratio": 2.0,
        },
        {
            "ex_date": "2024-01-05 09:00+08:00", "asset": ASSET,
            "kind": "cash", "cash_per_share": 0.7, "split_ratio": 1.0,
        },
    ]
    engine = _fixed_engine(
        broker, store, monkeypatch, corporate_actions=feed,
    )
    _at(engine, "2024-01-05")
    engine.run_once()

    position = _position(broker)
    assert position.quantity == 200, (
        "the naive split row was dropped by the mixed-timezone parse"
    )
    # Both cash rows collapse to one contract row per (date, asset, kind);
    # the dividend is paid once (last declared amount wins, as documented).
    assert broker.get_account()["cash"] == pytest.approx(98_995.0 + 70.0)
    store.close()


def test_real_broker_never_applies_a_local_feed(tmp_path, monkeypatch):
    broker = _RealBrokerStub()
    store = Store(str(tmp_path / "engine.db"))
    engine = LiveTradingEngine(
        broker=broker, store=store, risk_monitor=RiskMonitor(),
        corporate_actions=_feed(),
    )
    assert broker.corporate_calls == 0
    assert engine.set_corporate_actions(_feed()) == 0
    assert engine.get_corporate_actions_applied()["count"] == 0
    assert engine.get_state()["corporate_actions_feed"][
        "broker_supports_local"
    ] is False
    store.close()
