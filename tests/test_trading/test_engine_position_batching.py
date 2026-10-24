"""A17 regression: the engine persists the position book in one transaction.

`_sync_store_book` used to call `save_position` once per position, each call
taking the store lock and committing separately, and the book is persisted at
least twice per cycle. The bulk write must preserve D6 semantics: it is only
called with a known book, and stale rows of the same tenant are pruned.
"""

from __future__ import annotations

import threading

from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine


class _CountingStore(Store):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.save_position_calls = 0
        self.save_positions_calls = 0
        self.lock = threading.Lock()

    def save_position(self, pos):
        with self.lock:
            self.save_position_calls += 1
        return super().save_position(pos)

    def save_positions(self, positions, tenant_id="", prune_missing=False):
        with self.lock:
            self.save_positions_calls += 1
        return super().save_positions(
            positions, tenant_id=tenant_id, prune_missing=prune_missing,
        )


def _engine(store, tenant_id="default"):
    broker = SimulatedBroker(initial_cash=1_000_000)
    engine = LiveTradingEngine(
        broker=broker, store=store, risk_monitor=RiskMonitor(),
        tenant_id=tenant_id,
    )
    return engine


def _row(code, quantity=100, tenant_id="default"):
    return {
        "code": code, "tenant_id": tenant_id, "name": code,
        "quantity": quantity, "available": quantity, "avg_cost": 10.0,
        "current_price": 10.0, "market_value": quantity * 10.0,
        "unrealized_pnl": 0.0, "realized_pnl": 0.0,
    }


def test_sync_store_book_writes_once_per_persist(tmp_path):
    store = _CountingStore(str(tmp_path / "engine.db"))
    engine = _engine(store)
    rows = [_row("600519"), _row("000001"), _row("300750")]

    engine._sync_store_book(rows)

    assert store.save_positions_calls == 1
    assert store.save_position_calls == 0, (
        "the engine still commits once per position"
    )
    stored = {r["code"] for r in store.get_positions("default")}
    assert stored == {"600519", "000001", "300750"}


def test_sync_store_book_prunes_only_the_engine_tenant(tmp_path):
    store = _CountingStore(str(tmp_path / "engine_prune.db"))
    engine = _engine(store, tenant_id="fund_a")
    store.save_position(_row("999999", tenant_id="fund_b"))

    engine._sync_store_book([_row("600519", tenant_id="fund_a")])
    assert {r["code"] for r in store.get_positions("fund_a")} == {"600519"}
    assert {r["code"] for r in store.get_positions("fund_b")} == {"999999"}

    engine._sync_store_book([
        _row("600519", tenant_id="fund_a"), _row("000001", tenant_id="fund_a"),
    ])
    engine._sync_store_book([_row("000001", tenant_id="fund_a")])

    assert {r["code"] for r in store.get_positions("fund_a")} == {"000001"}
    assert {r["code"] for r in store.get_positions("fund_b")} == {"999999"}
    assert store.save_positions_calls == 3
    assert store.save_position_calls == 1  # only the explicit seed above


def test_bulk_book_write_is_one_transaction(tmp_path, monkeypatch):
    """A failure mid-book must leave the previous book intact."""
    store = Store(str(tmp_path / "atomic.db"))
    store.save_positions([
        _row("600519", tenant_id="default"),
        _row("000001", tenant_id="default"),
    ], tenant_id="default", prune_missing=True)

    original = store._position_values

    calls = {"n": 0}

    def failing(pos, tenant, now):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second row failed")
        return original(pos, tenant, now)

    monkeypatch.setattr(store, "_position_values", failing)
    try:
        store.save_positions(
            [_row("600519", tenant_id="default"), _row("300750", tenant_id="default")],
            tenant_id="default", prune_missing=True,
        )
    except RuntimeError:
        pass
    else:  # pragma: no cover - the injected failure must surface
        raise AssertionError("expected the bulk write to raise")

    # The generator aborted before the transaction was entered, so both old
    # rows survive; no partial book was committed.
    stored = {r["code"] for r in store.get_positions("default")}
    assert stored == {"600519", "000001"}
