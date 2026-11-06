"""API price-basis exposure (D15/D41).

Consumers must be able to tell raw tradeable prices from adjusted research
levels.  The live portfolio, OMS marks and smart-route execution all report
``price_basis == "raw"``; smart-route refuses to derive an execution price
from an adjusted-only provider.
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import quant_platform.api.routes as routes
from quant_platform.execution.oms import OrderManager


class _ResultSet:
    def __init__(self, rows):
        self._rows = list(rows)
        self._idx = -1
        self.error_code = "0"

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


def _install_fake_baostock(monkeypatch, by_code: dict[str, list[str]]):
    fake = types.ModuleType("baostock")
    fake.login = lambda: SimpleNamespace(error_code="0", error_msg="")
    fake.logout = lambda: None

    def query(bs_code, *args, **kwargs):
        digits = str(bs_code).split(".")[-1]
        return _ResultSet([by_code[digits]])

    fake.query_history_k_data_plus = query
    monkeypatch.setitem(sys.modules, "baostock", fake)


def test_live_portfolio_marks_are_raw(monkeypatch):
    monkeypatch.setattr(routes, "_holdings_data", {
        "holdings": [
            {"code": "600519", "hold_vol": 100},
            {"code": "000001", "hold_vol": 200},
        ],
        "n_stocks": 2,
        "total_shares": 300,
    })
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    _install_fake_baostock(monkeypatch, {
        "600519": [today, "10.0", "9.5"],
        "000001": [today, "12.0", "11.8"],
    })

    payload = routes._fetch_live_portfolio()

    assert payload["price_basis"] == "raw"
    assert payload["price_source"] == "baostock:adjustflag=3"
    # 100 x 10 + 200 x 12 = 3400 CNY at the raw traded prices.
    assert payload["total_value"] == pytest.approx(3400.0)
    assert all(row["close"] > 0 for row in payload["holdings"])
    assert all(row["price_basis"] == "raw" for row in payload["holdings"])


def test_oms_positions_report_price_basis(monkeypatch):
    oms = OrderManager(initial_cash=100_000.0)
    order = oms.create_order("600519.SH", "buy", 100)
    oms.submit_order(order.order_id)
    oms.fill_order(order.order_id, price=10.0)
    monkeypatch.setattr(routes, "_oms_instance", oms)

    payload = asyncio.run(routes.get_positions())

    assert payload["price_basis"] == "raw"
    assert payload["positions"][0]["current_price"] == pytest.approx(10.0)
    assert payload["positions"][0]["price_basis"] == "raw"


def _raw_metadata(close_panel: pd.DataFrame | None) -> pd.DataFrame:
    meta = pd.DataFrame(
        {"sector": ["食品饮料"], "listing_date": [pd.Timestamp("2020-01-01")],
         "delisting_date": [pd.NaT]},
        index=pd.Index(["600519.SH"], name="asset"),
    )
    if close_panel is not None:
        meta.attrs["raw_prices"] = {"close": close_panel, "price_basis": "raw"}
    return meta


def test_smart_route_executes_at_raw_close(monkeypatch):
    adjusted = pd.DataFrame(
        {"600519.SH": [20.0, 20.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    raw = pd.DataFrame(
        {"600519.SH": [10.0, 5.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    metadata = _raw_metadata(raw)
    monkeypatch.setattr(
        routes, "_load_data", lambda *a, **k: (
            adjusted, None, None, metadata, None, None,
        ),
    )
    oms = OrderManager(initial_cash=1_000_000.0)
    monkeypatch.setattr(routes, "_get_oms", lambda: oms)

    response = routes.smart_route_order({
        "ticker": "600519", "side": "buy", "quantity": 1000,
        "avg_daily_volume": 1_000_000, "execute": True,
    })

    assert response["price_basis"] == "raw"
    assert response["ticker"] == "600519.SH"
    # The fill used the last raw close (5.0), not the adjusted 20.0.
    position = oms.positions["600519.SH"]
    assert position.avg_cost == pytest.approx(5.0)


def test_smart_route_refuses_adjusted_only_provider(monkeypatch):
    adjusted = pd.DataFrame(
        {"600519.SH": [20.0, 20.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    monkeypatch.setattr(
        routes, "_load_data", lambda *a, **k: (
            adjusted, None, None, _raw_metadata(None), None, None,
        ),
    )
    monkeypatch.setattr(routes, "_get_oms", lambda: OrderManager(1_000_000.0))

    with pytest.raises(HTTPException) as exc:
        routes.smart_route_order({
            "ticker": "600519", "side": "buy", "quantity": 1000,
            "avg_daily_volume": 1_000_000, "execute": True,
        })
    assert exc.value.status_code == 400
    assert "raw" in exc.value.detail
