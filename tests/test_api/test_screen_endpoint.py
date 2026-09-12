"""Regression tests for the screener endpoint.

`POST /api/screen` used to call `_load_config`, a name that is neither defined
nor imported anywhere in the module. The call sits *outside* the endpoint's
try/except, so every request died with NameError instead of reaching the
data-loading guard, and nothing caught it because the endpoint had no test.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes


class _StubConfig:
    """Minimal stand-in for the Config object the endpoint expects."""


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("data source unavailable")

    monkeypatch.setattr(routes, "_load_data", _unavailable)
    monkeypatch.setattr(routes, "load_config", lambda config_path=None: _StubConfig())

    return TestClient(app)


VALID_BODY = {"rules": [{"factor": "pe_ratio", "operator": "lt", "value": 30}]}


def test_screen_reaches_the_data_load_guard(client):
    """A resolvable config loader is what gets us as far as the 503 guard."""
    resp = client.post("/api/screen", json=VALID_BODY)
    assert resp.status_code == 503
    assert "Data load failed" in resp.json()["detail"]


def test_screen_passes_request_config_path_through(monkeypatch):
    """The request's `config` field is forwarded to the loader."""
    seen = {}

    def _record(config_path=None):
        seen["config_path"] = config_path
        return _StubConfig()

    monkeypatch.setattr(routes, "load_config", _record)
    monkeypatch.setattr(routes, "_load_data", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError()))

    app = FastAPI()
    app.include_router(routes.router)
    TestClient(app).post("/api/screen", json={**VALID_BODY, "config": "custom.yaml"})

    assert seen["config_path"] == "custom.yaml"


def test_screen_rejects_malformed_rules(client):
    resp = client.post("/api/screen", json={"rules": [{"factor": "pe_ratio"}]})
    assert resp.status_code == 422
