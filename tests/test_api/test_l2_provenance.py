"""Level 2 responses must say that the data is simulated and must not
present an empty book/tick set as a valid price."""

import pytest
from fastapi import HTTPException

import quant_platform.api.routes as routes


class _EmptyProvider:
    is_running = True

    def get_ticks(self, code, limit=100):
        return []

    def compute_trade_flow(self, code, n_ticks=100):
        return {"buy_volume": 0, "sell_volume": 0, "net_flow": 0, "buy_pct": 0.0}


@pytest.fixture
def empty_provider(monkeypatch):
    monkeypatch.setattr(routes, "_get_l2_provider", lambda: _EmptyProvider())


def test_vwap_without_ticks_is_not_a_zero_price(empty_provider):
    with pytest.raises(HTTPException) as exc:
        routes.l2_vwap("600519")

    assert exc.value.status_code == 404


def test_flow_without_ticks_is_not_fifty_fifty(empty_provider):
    with pytest.raises(HTTPException) as exc:
        routes.l2_trade_flow("600519")

    assert exc.value.status_code == 404
