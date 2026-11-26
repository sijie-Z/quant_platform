"""Numeric bounds on ``/api/strategy/add`` allocation_pct.

The endpoint only tested for finiteness. NaN was rejected, but the range
check could be deleted without any test failing, so a 500% capital
allocation was accepted and registered.
"""

import pytest
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.app import create_app


@pytest.fixture
def local_client(monkeypatch):
    """Same local-client pattern as tests/test_api/test_security_contracts.py.

    The strategy manager is reset per test so registrations do not leak
    between cases or into other modules' assertions.
    """
    monkeypatch.delenv("QUANT_API_TOKEN", raising=False)
    monkeypatch.setattr(routes, "_multi_strategy_mgr", None)
    app = create_app(serve_frontend=False)
    return TestClient(app)


@pytest.mark.parametrize("bad", [1.5, 5.0, -0.1])
def test_allocation_outside_the_unit_interval_is_rejected(local_client, bad):
    resp = local_client.post(
        "/api/strategy/add",
        json={"name": f"bad_{bad}", "allocation_pct": bad},
    )

    assert resp.status_code == 422, resp.text
    assert "allocation_pct" in resp.json()["detail"]


@pytest.mark.parametrize("ok", [0.0, 1.0])
def test_allocation_bounds_are_inclusive(local_client, ok):
    resp = local_client.post(
        "/api/strategy/add",
        json={"name": f"edge_{ok}", "allocation_pct": ok},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == f"edge_{ok}"
    assert body["strategy_id"]

    listed = local_client.get("/api/strategy/list")
    assert listed.status_code == 200
    allocations = {
        row["name"]: row["allocation_pct"]
        for row in listed.json()["strategies"]
    }
    assert allocations[f"edge_{ok}"] == pytest.approx(ok)
