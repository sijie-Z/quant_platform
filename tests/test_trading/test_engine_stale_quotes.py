"""D11 regression: quote staleness is tracked per symbol.

The guard kept one aggregate `_last_prices_update`. With 000001's feed dead
and only 600519 refreshing, the aggregate looked fresh, so the engine sold
000001 at its hours-old cached price. Freshness is now recorded per symbol,
sizing refuses stale/missing quotes, and only the current cycle's quotes are
passed to the broker for marks/liquidity.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import Order, OrderSide, SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine, TradeSignal

STALE_AGE = 3600.0


class _StaticMarket:
    def __init__(self, quotes):
        self._quotes = quotes

    def get_quotes(self, universe):
        return list(self._quotes)


def _quote(code: str, price: float) -> SimpleNamespace:
    return SimpleNamespace(code=code, price=price, change_pct=0.0, volume=1000.0)


def _make_engine(tmp_path, monkeypatch, broker=None,
                 patch_fetch: bool = True) -> LiveTradingEngine:
    engine = LiveTradingEngine(
        broker=broker or SimulatedBroker(initial_cash=1_000_000),
        store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
    )
    if patch_fetch:
        monkeypatch.setattr(engine, "_fetch_prices", lambda: None)
    engine._trading_day = engine._exchange_trading_date()
    return engine


def test_stale_symbol_is_not_sized(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch)
    engine._current_prices = {"000001": 8.0}
    engine._price_last_seen = {"000001": time.monotonic() - STALE_AGE}

    buy = TradeSignal(
        code="000001", side=OrderSide.BUY, target_weight=0.02,
        current_weight=0.0, signal_strength=0.4,
    )
    sell = TradeSignal(
        code="000001", side=OrderSide.SELL, target_weight=0.0,
        current_weight=0.1, signal_strength=0.0,
    )
    assert engine._signal_to_order(buy, 1_000_000) is None
    assert engine._signal_to_order(sell, 1_000_000) is None


def test_only_fresh_quotes_are_passed_to_the_broker(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=1_000_000)
    engine = _make_engine(tmp_path, monkeypatch, broker, patch_fetch=False)
    # 000001 is in the cached price map from an earlier cycle but its feed is
    # dead (an old per-symbol stamp); only 600519 refreshed this cycle.
    engine._current_prices = {"000001": 8.0}
    engine._price_last_seen = {"000001": time.monotonic() - STALE_AGE}
    engine._rt_market = _StaticMarket([_quote("600519", 1700.0)])

    seen: dict[str, float] = {}

    def record(prices):
        seen.update(prices)

    monkeypatch.setattr(broker, "update_market_prices", record)

    engine._fetch_prices()

    assert seen == {"600519": pytest.approx(1700.0)}, (
        "stale cached prices must not reach update_market_prices"
    )
    assert engine._quote_is_fresh("600519")
    assert not engine._quote_is_fresh("000001")


def test_cycle_does_not_trade_a_symbol_with_a_dead_feed(tmp_path, monkeypatch):
    broker = SimulatedBroker(initial_cash=1_000_000)
    broker.connect()
    broker.place_order(
        Order(code="000001", side=OrderSide.BUY, quantity=2500, price=8.0)
    )
    broker.new_trading_day()
    engine = _make_engine(tmp_path, monkeypatch, broker)
    # The last cached price is 8.0 but the quote is an hour old; the true
    # market has moved to 4.0. The aggregate freshness flag would be fresh
    # because another symbol keeps updating.
    engine._current_prices = {"000001": 8.0, "600519": 1700.0}
    engine._price_last_seen = {
        "000001": time.monotonic() - STALE_AGE,
        "600519": time.monotonic(),
    }
    engine._last_prices_update = "now"
    engine._last_prices_update_monotonic = time.monotonic()

    sell = TradeSignal(
        code="000001", side=OrderSide.SELL, target_weight=0.0,
        current_weight=0.2, signal_strength=0.0, reason="exit",
    )
    monkeypatch.setattr(
        engine, "_generate_signals", lambda positions=None: [sell],
    )
    placed: list = []

    def record_place(order):
        placed.append(order)
        raise AssertionError("no order may be placed from a stale quote")

    monkeypatch.setattr(broker, "place_order", record_place)

    engine._execute_cycle()

    assert placed == []
    positions = broker.get_positions()
    assert positions and positions[0].quantity == 2500, (
        "the stale-price sell must not execute"
    )
