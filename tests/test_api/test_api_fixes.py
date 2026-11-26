"""Regression tests for API contract fixes found in the runtime audit."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_data_quality_endpoint_returns_a_report(client):
    """It used to raise ZeroDivisionError on zero prices and always 500."""
    response = client.post("/api/data/quality", json={})

    assert response.status_code == 200
    body = response.json()
    assert body.get("checks"), body
    assert body.get("overall_status")


def test_oms_accepts_the_frontend_order_type_spelling(client):
    """The UI sends `order_type`; the route only read `type`."""
    created = client.post("/api/oms/order", json={
        "ticker": "600519",
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 10.0,
        "auto_submit": False,
    })
    assert created.status_code == 200
    order_id = created.json()["order_id"]

    client.post("/api/oms/fill", json={"order_id": order_id, "price": 10.0})
    blotter = client.get("/api/oms/blotter").json()["orders"]
    row = next(order for order in blotter if order["order_id"] == order_id)

    assert row["type"] == "limit"
