"""The sweep endpoint may not select/tune parameters on holdout dates.

The guard must fire *before* any analytics work is scheduled: a 422 response
and a spy proving ``_run_analytics`` was never called. With the holdout
disabled (the default) behaviour must be byte-for-byte the old path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.api.schemas import SweepResult
from quant_platform.utils.config import load_config


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _raw_config(start="", end="", data_window=("2021-01-01", "2025-12-31")):
    return {
        "data": {"start_date": data_window[0], "end_date": data_window[1]},
        "research": {"holdout": {"start_date": start, "end_date": end}},
    }


def _request():
    return {
        "optimizers": ["equal_weight"],
        "frequencies": ["monthly"],
        "n_stocks_list": [50],
    }


@pytest.fixture
def analytics_spy(monkeypatch):
    """Replace the analytics scheduler with a recorder; returns the calls."""
    calls: list[object] = []

    async def fake_run_analytics(fn, *args):
        calls.append(args[0] if args else None)
        return SweepResult(table=[])

    monkeypatch.setattr(routes, "_run_analytics", fake_run_analytics)
    return calls


def test_overlapping_holdout_is_422_and_never_schedules_analytics(
    client, monkeypatch, analytics_spy,
):
    # One shared day (data window ends on the holdout's first day).
    config = load_config(raw=_raw_config(
        "2024-07-01", "2024-12-31",
        data_window=("2021-01-01", "2024-07-01"),
    ))
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "holdout" in detail
    assert "disable the holdout" in detail
    assert analytics_spy == []


def test_fully_contained_holdout_is_422(client, monkeypatch, analytics_spy):
    config = load_config(raw=_raw_config(
        "2024-01-01", "2024-12-31",
        data_window=("2021-01-01", "2025-12-31"),
    ))
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 422
    assert "holdout" in resp.json()["detail"]
    assert analytics_spy == []


def test_non_overlapping_window_still_runs(client, monkeypatch, analytics_spy):
    # Contiguous but not overlapping: data ends the day before the holdout.
    config = load_config(raw=_raw_config(
        "2024-07-01", "2024-12-31",
        data_window=("2021-01-01", "2024-06-30"),
    ))
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 200
    assert resp.json()["table"] == []
    assert len(analytics_spy) == 1


def test_disabled_holdout_behaves_as_before(client, monkeypatch, analytics_spy):
    config = load_config(raw=_raw_config("", ""))
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 200
    assert len(analytics_spy) == 1


def test_configs_without_a_research_section_still_work(
    client, monkeypatch, analytics_spy,
):
    # Other endpoint tests stub load_config with a minimal namespace; the guard
    # must treat a missing research/data section as "holdout disabled".
    monkeypatch.setattr(
        routes, "load_config",
        lambda _path=None: SimpleNamespace(
            data=SimpleNamespace(start_date="2021-01-01", end_date="2025-12-31"),
        ),
    )

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 200
    assert len(analytics_spy) == 1


def test_invalid_holdout_config_is_422(client, monkeypatch, analytics_spy):
    config = SimpleNamespace(
        data=SimpleNamespace(start_date="2021-01-01", end_date="2025-12-31"),
        research=SimpleNamespace(
            holdout=SimpleNamespace(
                start_date="2024-12-31", end_date="2024-01-01",
            ),
        ),
    )
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 422
    assert "strictly before" in resp.json()["detail"]
    assert analytics_spy == []


def test_enabled_holdout_without_data_window_is_422(
    client, monkeypatch, analytics_spy,
):
    config = SimpleNamespace(
        data=None,
        research=SimpleNamespace(
            holdout=SimpleNamespace(
                start_date="2024-01-01", end_date="2024-12-31",
            ),
        ),
    )
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)

    resp = client.post("/api/sweep", json=_request())

    assert resp.status_code == 422
    assert "data window" in resp.json()["detail"]
    assert analytics_spy == []
