"""Live portfolio errors must be HTTP errors, not 200 + {"error": ...}.

The frontend treated the latter as a valid empty portfolio and reported a
successful import/refresh.
"""

import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import quant_platform.api.routes as routes


def test_no_holdings_imported_is_409(monkeypatch):
    monkeypatch.setattr(routes, "_holdings_data", None)

    with pytest.raises(HTTPException) as exc:
        routes.get_live_portfolio()

    assert exc.value.status_code == 409


def test_no_live_prices_is_503(monkeypatch):
    monkeypatch.setattr(routes, "_holdings_data", {
        "holdings": [{"code": "600519", "hold_vol": 100}],
        "n_stocks": 1,
        "total_shares": 100,
    })

    fake_bs = types.ModuleType("baostock")
    fake_bs.login = lambda: SimpleNamespace(error_code="0")
    fake_bs.logout = lambda: None
    fake_bs.query_history_k_data_plus = lambda *a, **k: SimpleNamespace(
        error_code="0", next=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "baostock", fake_bs)

    with pytest.raises(HTTPException) as exc:
        routes.get_live_portfolio()

    assert exc.value.status_code == 503
    assert "No live prices" in exc.value.detail
