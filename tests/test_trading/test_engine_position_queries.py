"""D6 regression: a transient empty position query is not "flat".

The engine read the broker's empty list as an authoritative flat book: one
cycle with a transient empty answer bought a duplicate stake (signals saw no
holdings) and pruned every persisted position row. QMT also converted a
failed `query_stock_positions()` (None) into `[]`. The engine now
distinguishes unknown (None/raise) from flat, re-confirms a single empty
answer while the Store records holdings, and never prunes from an unknown
book.
"""

from __future__ import annotations

import time

import pytest

from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import Order, OrderSide, SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine, TradeSignal

CODES_WITH_HISTORY = ["600519"] + [f"00{i:04d}" for i in range(1, 30)]


class EmptyOnceBroker(SimulatedBroker):
    """Returns [] on exactly one `get_positions()` call, real positions after."""

    def __init__(self, empty_on_call: int, **kwargs):
        super().__init__(**kwargs)
        self._empty_on_call = empty_on_call
        self._position_calls = 0

    def get_positions(self):
        self._position_calls += 1
        if self._position_calls == self._empty_on_call:
            return []
        return super().get_positions()


class NoneBroker(SimulatedBroker):
    """Models QMT's failed query: the book state is unknown."""

    def get_positions(self):
        return None


class AlwaysEmptyBroker(SimulatedBroker):
    """A persistent empty answer that contradicts the ledger (probe J)."""

    def get_positions(self):
        return []


class RaisingBroker(SimulatedBroker):
    def get_positions(self):
        raise ConnectionError("position service unavailable")


class _FixedSignalGenerator:
    """Deterministic scores so the real signal path can run without alpha."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def update_prices(self, price_df):
        self._seen = price_df

    def generate(self):
        return dict(self._scores)


def _make_engine(tmp_path, broker, monkeypatch, db_name="engine.db"):
    engine = LiveTradingEngine(
        broker=broker,
        store=Store(str(tmp_path / db_name)),
        risk_monitor=RiskMonitor(),
    )
    monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    engine._trading_day = engine._exchange_trading_date()
    return engine


def _seed_holding(broker: SimulatedBroker, store: Store, quantity: int = 2000):
    broker.connect()
    broker.place_order(
        Order(code="600519", side=OrderSide.BUY, quantity=quantity, price=10.0)
    )
    # Read the book directly: `get_positions()` call counts are load-bearing
    # for EmptyOnceBroker and must not be consumed during setup.
    position = next(iter(broker._positions.values()))
    store.save_position({**position.to_dict(), "tenant_id": "default"})


def _install_signal_inputs(engine: LiveTradingEngine, scores: dict[str, float]):
    engine._price_history = {code: [10.0] * 20 for code in CODES_WITH_HISTORY}
    engine._signal_generator = _FixedSignalGenerator(scores)
    engine._n_stocks = 50
    engine._min_signal_score = 0.3
    engine._current_prices = {"600519": 10.0}
    engine._price_last_seen = {"600519": time.monotonic()}


def test_single_empty_answer_is_not_trusted_as_flat(tmp_path, monkeypatch):
    broker = EmptyOnceBroker(empty_on_call=1, initial_cash=1_000_000)
    store = Store(str(tmp_path / "engine.db"))
    _seed_holding(broker, store)
    engine = _make_engine(tmp_path, broker, monkeypatch)

    positions, known = engine._query_positions()

    assert known is False, (
        "an empty answer against a non-empty store is unknown, not flat"
    )
    assert positions is None


def test_transient_empty_signals_query_does_not_create_duplicate_exposure(
    tmp_path, monkeypatch,
):
    # Call order in the new engine: cycle-start query (1), then the empty
    # answer is consumed by the "signals" phase of the old code (2).
    broker = EmptyOnceBroker(empty_on_call=2, initial_cash=1_000_000)
    store = Store(str(tmp_path / "engine.db"))
    _seed_holding(broker, store, quantity=500)
    engine = _make_engine(tmp_path, broker, monkeypatch)
    _install_signal_inputs(
        engine, {code: 0.5 for code in CODES_WITH_HISTORY},
    )

    engine._execute_cycle()

    positions = broker.get_positions()
    assert len(positions) == 1
    # Target weight is 1/30 = 3.33%, current weight 0.5%: one rebalance buy
    # of 28 lots. The old code saw the empty answer as "flat" and sized a
    # fresh 3.33% stake (33 lots) on top of the existing 500 shares.
    assert positions[0].quantity == 3300, (
        f"duplicate exposure from a transient empty query: {positions[0].quantity}"
    )
    rows = store.get_positions("default")
    assert rows and rows[0]["quantity"] == 3300, "the store book was pruned"


def test_transient_empty_prune_query_does_not_erase_the_store_book(
    tmp_path, monkeypatch,
):
    # With no generated signals the old cycle made three position queries:
    # risk update (1), save loop (2), prune (3). The third returning [] wiped
    # every persisted row.
    broker = EmptyOnceBroker(empty_on_call=3, initial_cash=1_000_000)
    store = Store(str(tmp_path / "engine.db"))
    _seed_holding(broker, store)
    engine = _make_engine(tmp_path, broker, monkeypatch)

    engine._execute_cycle()

    rows = store.get_positions("default")
    assert rows and rows[0]["quantity"] == 2000, (
        "a transient empty prune query erased the persisted book"
    )
    assert broker.get_positions()[0].quantity == 2000


@pytest.mark.parametrize("broker_cls", [NoneBroker, RaisingBroker, AlwaysEmptyBroker])
def test_unknown_book_skips_sizing_and_pruning(tmp_path, monkeypatch, broker_cls):
    broker = broker_cls(initial_cash=1_000_000)
    store = Store(str(tmp_path / "engine.db"))
    store.save_position({
        "code": "600519", "quantity": 2000, "available": 2000,
        "avg_cost": 10.0, "current_price": 10.0, "market_value": 20000.0,
    })
    engine = _make_engine(tmp_path, broker, monkeypatch)

    def must_not_size(positions=None):
        raise AssertionError("signals must not be generated from an unknown book")

    monkeypatch.setattr(engine, "_generate_signals", must_not_size)

    cycle = engine._execute_cycle()

    assert cycle.orders == []
    rows = store.get_positions("default")
    assert rows and rows[0]["quantity"] == 2000, (
        "never delete store positions on an unknown position query"
    )


def test_normal_full_liquidation_still_prunes_the_store_book(tmp_path, monkeypatch):
    """The conservative 'never prune on empty' rule must not strand rows
    after the engine itself sells the last share."""
    broker = SimulatedBroker(initial_cash=1_000_000)
    store = Store(str(tmp_path / "engine.db"))
    _seed_holding(broker, store, quantity=1000)
    broker.new_trading_day()  # release T+1 so the position is sellable
    engine = _make_engine(tmp_path, broker, monkeypatch)
    engine._fetch_prices = lambda: None
    engine._current_prices = {"600519": 10.0}
    engine._price_last_seen = {"600519": time.monotonic()}
    exit_signal = TradeSignal(
        code="600519", side=OrderSide.SELL, target_weight=0.0,
        current_weight=0.05, signal_strength=0.0, reason="exit",
    )
    monkeypatch.setattr(
        engine, "_generate_signals", lambda positions=None: [exit_signal],
    )

    engine._execute_cycle()

    assert broker.get_positions() == []
    assert store.get_positions("default") == [], (
        "a filled sell that closes the book must prune the persisted row"
    )
