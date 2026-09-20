"""Regression tests for the run store and the endpoints that read it.

`POST /api/run` is the product's main endpoint. It never once completed: the
pipeline raised on the way to storing its result, so every run was marked
`failed` and `/api/run/{id}/result` returned 404.
"""

import inspect
import json

import pytest

from quant_platform.api import routes
from quant_platform.api.schemas import RunRequest, RunResult


class TestThePipelineCompletes:
    """End-to-end, because every defect here was cross-module: the pipeline
    built one shape and its consumers expected another."""

    @pytest.fixture(scope="module")
    def completed_run(self):
        routes._run_store.clear()
        routes._run_status.clear()
        run_id = "test-run"
        routes._run_status[run_id] = {
            "status": "running", "progress": 0, "stage": "data",
            "started_at": "2026-01-01T00:00:00", "completed_at": None, "error": None,
        }
        try:
            routes._execute_pipeline(
                run_id, RunRequest(n_stocks=50, start_date="2023-01-01", end_date="2024-12-31")
            )
            yield run_id
        finally:
            routes._run_store.clear()
            routes._run_status.clear()

    def test_status_is_completed_and_a_result_is_stored(self, completed_run):
        status = routes._run_status[completed_run]
        assert status["status"] == "completed", status.get("error")
        assert completed_run in routes._run_store

    def test_the_stored_record_is_plain_python(self, completed_run):
        """Not Pydantic models. Every consumer reads it as a dict of dicts --
        `result.get("chart_data", {}).get("equity", [])` in the Monte Carlo,
        risk-decomposition and regime endpoints, `f.get("name")` over the
        factor list in the HTML report."""
        record = routes._run_store[completed_run]

        def offenders(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    yield from offenders(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, value in enumerate(obj[:3]):
                    yield from offenders(value, f"{path}[{i}]")
            elif not isinstance(obj, (str, int, float, bool, type(None))) or (
                type(obj).__module__ == "numpy"
            ):
                yield path, type(obj).__name__

        assert list(offenders(record)) == [], "the run store holds model objects"
        json.dumps(record)  # must be serialisable without a custom encoder

    def test_every_consumer_of_the_store_runs(self, completed_run):
        record = routes._run_store[completed_run]

        routes._decompose_risk(record)
        routes._detect_regime(record)
        routes._run_monte_carlo(record, {})

        RunResult(
            run_id=completed_run,
            performance=record.get("performance"),
            risk=record.get("risk"),
            stress_tests=record.get("stress_tests"),
            factors=record.get("factors"),
            exposure=record.get("exposure"),
            chart_data=record.get("chart_data"),
        )


class TestRiskDecompositionIsNotInvented:
    """`_decompose_risk` used to manufacture a decomposition.

    The numbers were internally consistent, so a test written against its
    output would have passed -- `total_risk` was the literal 15.2, a factor's
    risk share was `icir * 15`, its beta `0.5 + icir * 0.3`, and its
    t-statistic `icir * sqrt(252)`.
    """

    def test_it_reports_unavailable(self):
        record = {"factors": [{"name": "momentum_3m", "mean_ic": 0.02, "icir": 0.4}],
                  "chart_data": {"equity": [1.2], "dates": ["2024-01-01"]}}
        out = routes._decompose_risk(record)

        assert out["available"] is False
        assert out["factors"] == []
        assert out["total_risk_pct"] is None
        assert "factor panel" in out["reason"]

    def test_the_invented_constants_are_gone(self):
        body = inspect.getsource(routes._decompose_risk).split('"""')[2]
        assert "15.2" not in body, "the hardcoded total risk is back"
        assert "icir * 15" not in body, "risk shares derived from ICIR are back"
        assert "0.5 + icir" not in body, "betas derived from ICIR are back"

    def test_it_no_longer_raises_on_the_stored_shape(self):
        """The original crash: the stored factor entries are read with `.get`."""
        record = {"factors": [{"name": "roe", "mean_ic": 0.01, "std_ic": 0.03,
                               "icir": 0.33, "ic_positive_ratio": 0.5}],
                  "chart_data": {}}
        assert routes._decompose_risk(record)["available"] is False


class TestNoPandasTruthinessInThePipeline:
    """`if metadata` and `if sector_map_meta` both raised "The truth value of
    a DataFrame/Series is ambiguous", and both sat on the path that stores a
    completed run."""

    def test_the_guards_use_is_not_none(self):
        source = inspect.getsource(routes._execute_pipeline)
        assert "if metadata else" not in source
        assert "if sector_map_meta else" not in source
        assert "metadata is not None" in source
