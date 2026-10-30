"""The Prometheus middleware must actually record requests.

`/api/metrics` merely exported an always-empty collector: nothing in
production called the counter/gauge/histogram APIs.
"""

from fastapi.testclient import TestClient

from quant_platform.app import create_app


def test_api_requests_are_exported():
    app = create_app(serve_frontend=False)
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200

    text = client.get("/api/metrics").text
    assert "api_requests_total" in text
    assert "api_duration_seconds" in text
    assert "/api/health" in text


def test_pipeline_stages_are_instrumented():
    from quant_platform.utils.metrics import get_metrics, pipeline_stage

    with pipeline_stage("unit_test_stage"):
        pass

    text = get_metrics().export_text()
    assert "pipeline_stage_total" in text
    assert 'stage="unit_test_stage"' in text
    assert "pipeline_stage_duration_seconds" in text


def test_unmatched_paths_do_not_create_unbounded_series():
    """Unauthenticated 404 probes used to mint one label set per URL.

    The metrics middleware runs outside auth and fell back to the raw
    ``scope["path"]`` whenever the router matched nothing, so
    ``/api/<random>`` grew the counter and histogram without bound.
    """
    from quant_platform.utils.metrics import get_metrics

    app = create_app(serve_frontend=False)
    client = TestClient(app)
    metrics = get_metrics()
    metrics.reset()

    for index in range(25):
        assert client.get(f"/api/probe-{index}").status_code == 404

    snapshot = metrics.get_snapshot()
    request_series = {
        key: entry for key, entry in snapshot.items()
        if key.startswith("api_requests_total")
    }
    duration_series = {
        key: entry for key, entry in snapshot.items()
        if key.startswith("api_duration_seconds")
    }

    assert len(request_series) == 1, request_series
    assert len(duration_series) == 1, duration_series
    (entry,) = request_series.values()
    assert entry["labels"]["path"] == "unmatched"


def test_matched_route_template_keeps_dynamic_ids_bounded():
    from quant_platform.utils.metrics import get_metrics

    app = create_app(serve_frontend=False)
    client = TestClient(app)
    metrics = get_metrics()
    metrics.reset()

    for index in range(10):
        client.get(f"/api/run/run-{index}/status")

    text = metrics.export_text()
    assert 'path="/api/run/{run_id}/status"' in text
    assert 'path="/api/run/run-0/status"' not in text
