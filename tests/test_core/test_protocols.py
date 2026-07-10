"""Tests for protocol layer objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest

from quant_platform.core.protocols import (
    AccountState, Position, Order, Fill, Signal,
    PortfolioSnapshot, Recommendation,
    OrderSide, OrderStatus, OrderType, RecommendationAction,
    validate_protocol, validate_weights,
)


class TestAccountState:
    def test_defaults(self):
        a = AccountState()
        assert a.cash == 0.0
        assert a.n_positions == 0

    def test_to_dict_rounds(self):
        a = AccountState(cash=100.12345, equity=200.67890)
        d = a.to_dict()
        assert d["cash"] == 100.12
        assert d["equity"] == 200.68

    def test_from_dict(self):
        d = {"cash": 5000, "equity": 10000, "market_value": 5000}
        a = AccountState.from_dict(d)
        assert a.cash == 5000
        assert a.equity == 10000

    def test_validate_happy(self):
        a = AccountState(cash=1000, market_value=9000, equity=10000)
        assert a.validate() == []

    def test_validate_negative_cash(self):
        a = AccountState(cash=-100)
        errors = a.validate()
        assert any("negative" in e for e in errors)

    def test_validate_reconciliation(self):
        a = AccountState(cash=0, market_value=0, equity=100)
        errors = a.validate()
        assert any("reconciliation" in e for e in errors)

    def test_frozen(self):
        a = AccountState()
        with pytest.raises(FrozenInstanceError):
            a.cash = 999  # type: ignore


class TestPosition:
    def test_basic(self):
        p = Position(code="600519", quantity=100, avg_cost=150.0)
        assert p.code == "600519"
        assert p.quantity == 100

    def test_validate_happy(self):
        p = Position(code="600519", quantity=100, available=100)
        assert p.validate() == []

    def test_negative_quantity(self):
        p = Position(code="600519", quantity=-5)
        errors = p.validate()
        assert any("negative" in e for e in errors)

    def test_available_exceeds_quantity(self):
        p = Position(code="600519", quantity=100, available=200)
        errors = p.validate()
        assert any("available" in e for e in errors)


class TestOrder:
    def test_default_side(self):
        o = Order()
        assert o.side == OrderSide.BUY

    def test_to_dict(self):
        o = Order(code="600519", side=OrderSide.BUY, quantity=100, price=150.0)
        d = o.to_dict()
        assert d["code"] == "600519"
        assert d["side"] == "buy"

    def test_from_dict_with_strings(self):
        d = {"code": "600519", "side": "sell", "quantity": 200, "price": 180.0}
        o = Order.from_dict(d)
        assert o.code == "600519"
        assert o.side == OrderSide.SELL

    def test_validate_happy(self):
        o = Order(code="600519", quantity=100, price=150.0)
        assert o.validate() == []

    def test_validate_zero_quantity(self):
        o = Order(code="600519", quantity=0, price=150.0)
        errors = o.validate()
        assert any("positive" in e for e in errors)

    def test_validate_overfilled(self):
        o = Order(code="600519", quantity=100, price=150.0, filled_quantity=200)
        errors = o.validate()
        assert any("filled_quantity" in e for e in errors)


class TestFill:
    def test_basic(self):
        f = Fill(order_id="ord1", code="600519", quantity=100, price=150.0)
        assert f.quantity == 100

    def test_validate_happy(self):
        f = Fill(order_id="ord1", code="600519", quantity=100, price=150.0)
        assert f.validate() == []

    def test_negative_price(self):
        f = Fill(order_id="ord1", code="600519", quantity=100, price=-1)
        errors = f.validate()
        assert any("positive" in e for e in errors)


class TestSignal:
    def test_basic(self):
        s = Signal(code="600519", direction="long", strength=0.5)
        assert s.code == "600519"
        assert s.strength == 0.5

    def test_validate_missing_code(self):
        s = Signal(code="", direction="long", strength=0.5)
        errors = s.validate()
        assert any("missing" in e for e in errors)

    def test_validate_strength_range(self):
        s = Signal(code="600519", direction="long", strength=5.0)
        errors = s.validate()
        assert any("range" in e for e in errors)


class TestPortfolioSnapshot:
    def test_basic(self):
        snap = PortfolioSnapshot(total_equity=100000, cash=20000, market_value=80000)
        assert snap.total_equity == 100000

    def test_validate_happy(self):
        snap = PortfolioSnapshot(total_equity=100000, cash=20000, market_value=80000)
        assert snap.validate() == []

    def test_validate_reconciliation(self):
        snap = PortfolioSnapshot(cash=0, market_value=0, total_equity=100)
        errors = snap.validate()
        assert any("reconciliation" in e for e in errors)


class TestRecommendation:
    def test_default_hold(self):
        r = Recommendation()
        assert r.action == RecommendationAction.HOLD

    def test_from_dict(self):
        d = {"code": "600519", "action": "BUY", "quantity": 100}
        r = Recommendation.from_dict(d)
        assert r.action == RecommendationAction.BUY
        assert r.quantity == 100

    def test_validate_missing_code(self):
        r = Recommendation(code="", action=RecommendationAction.BUY, quantity=100)
        errors = r.validate()
        assert any("missing" in e for e in errors)

    def test_validate_negative_quantity(self):
        r = Recommendation(code="600519", quantity=-5)
        errors = r.validate()
        assert any("negative" in e for e in errors)


class TestValidateHelpers:
    def test_validate_protocol(self):
        a = AccountState(cash=1000, equity=1000)
        assert validate_protocol("test", a) == []

    def test_validate_weights_pass(self):
        errors = validate_weights({"a": 0.5, "b": 0.5})
        assert errors == []

    def test_validate_weights_unbalanced(self):
        errors = validate_weights({"a": 0.8, "b": 0.1})
        assert any("sum" in e for e in errors)

    def test_validate_weights_negative(self):
        errors = validate_weights({"a": -0.1, "b": 1.1})
        assert any("negative" in e for e in errors)
