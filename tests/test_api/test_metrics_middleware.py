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
