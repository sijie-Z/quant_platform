"""P4 smoke tests: web server health and paper trading loop."""

from fastapi.testclient import TestClient
from quant_platform.app import create_app
from quant_platform.trading.live_runner import LiveRunner


def test_health_endpoint():
    app = create_app(serve_frontend=False)
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_paper_runner_smoke():
    runner = LiveRunner(
        broker_type="simulated",
        initial_cash=100_000,
        dual_track=False,
    )
    runner.set_universe(["600519", "000858"])
    report = runner.run(days=1, seed=42)

    assert report.days_traded == 1
    assert report.session_id
    assert report.total_orders >= 0
    assert report.final_value > 0
