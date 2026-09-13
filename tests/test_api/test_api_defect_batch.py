"""Regression tests for the API defect batch (BUG-36/38/39/43/44).

Each was a 500, a wrong result, or an unreachable endpoint. They share a shape:
the code was written against an assumption that the rest of the file did not
hold -- a constructor signature, a timestamp location, a route order, a numpy
scalar, a loop variable.
"""

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


# ── BUG-38: numpy bool is not JSON-serializable ──────────────────────────

class TestQualityReportIsJsonSafe:
    def test_report_serialises(self):
        """`passed` is derived from a numpy comparison, so it arrived as
        `numpy.bool_` and POST /api/data/quality returned 500 every time."""
        from quant_platform.data.providers.synthetic import SyntheticDataProvider
        from quant_platform.data.quality import DataQualityMonitor

        provider = SyntheticDataProvider(n_stocks=10, start_date="2023-01-01", end_date="2023-12-31")
        prices = provider.get_prices("2023-01-01", "2023-12-31")

        monitor = DataQualityMonitor()
        monitor.check_prices(prices)
        monitor.check_returns(prices.pct_change(fill_method=None).dropna())
        report = monitor.get_report()

        json.dumps(report)  # raises TypeError on numpy.bool_
        assert isinstance(report["checks"][0]["passed"], bool)
        assert not isinstance(report["checks"][0]["passed"], np.bool_)

    def test_coercion_happens_at_construction(self):
        """One place, so a new check cannot reintroduce the problem."""
        from quant_platform.data.quality import DataQualityCheck

        check = DataQualityCheck(
            name="x", passed=np.bool_(True), severity="info", message="m"
        )
        assert isinstance(check.passed, bool)
        assert not isinstance(check.passed, np.bool_)


# ── BUG-39: /fundamentals/stats shadowed by /fundamentals/{code} ─────────

class TestRoutesAreNotShadowed:
    def test_stats_is_registered_before_the_wildcard(self):
        """Starlette matches in registration order, so a `{code}` route declared
        first swallows every later literal path -- /api/fundamentals/stats was
        returning a fabricated stock named "stats"."""
        paths = [r.path for r in routes.router.routes if hasattr(r, "path")]
        code_idx = paths.index("/api/fundamentals/{code}")
        stats_idx = paths.index("/api/fundamentals/stats")
        assert stats_idx < code_idx, (
            f"/api/fundamentals/stats is at {stats_idx}, after "
            f"/api/fundamentals/{{code}} at {code_idx} -- it can never match"
        )


# ── BUG-44: attribution used a leftover loop variable ────────────────────

class TestAttribution:
    def test_attribution_returns_nothing_and_the_reason_is_recorded(self):
        """Two defects live in `_compute_attribution`. Only one is fixed here.

        **Fixed**: `common_dates` came from a `factor_df` left over from the
        previous loop -- whichever factor that loop ended on -- while the body
        read `processed_factors[name]`. The IC reported for each factor was
        computed over another factor's dates.

        **Not fixed, and the reason the function returns nothing at all**: the
        first loop does `returns.reindex(common)` where `common` comes from the
        factor's *columns* (asset codes), but the caller passes a date-indexed
        returns series. Every value becomes NaN, `fillna(0)` makes it constant,
        the IC is NaN, `factor_contribs` stays empty, and the second loop --
        the one that computes the reported `avg_ic` -- never runs.

        So the P&L attribution panel is empty because the function cannot
        compute anything, not because there is nothing to attribute. Asserted
        here as the current state so that whatever fixes it has to change this
        test deliberately, rather than the emptiness going unnoticed.
        """
        np.random.seed(0)
        cols = [f"S{i}" for i in range(10)]
        idx = pd.bdate_range("2023-01-01", periods=150)
        factors = {"f1": pd.DataFrame(np.random.randn(150, 10), index=idx, columns=cols)}
        returns = pd.Series(np.random.randn(150) * 0.01, index=idx)
        weights_history = {d: pd.Series(1 / 10, index=cols) for d in idx[::30]}

        assert routes._compute_attribution(factors, weights_history, returns) == []

    def test_the_stale_variable_is_gone_from_the_source(self):
        """The second loop should read the factor it names, not a leftover."""
        import inspect

        source = inspect.getsource(routes._compute_attribution)
        loop = source.split("for name, contrib in sorted(")[1]
        head = loop.split("for d in")[0]
        assert "factor_df = processed_factors[name]" in head, (
            "the second loop uses `factor_df` without assigning it"
        )


# ── BUG-36: QMT broker constructed with keys that do not exist ───────────

class TestQmtBrokerKeywordContract:
    def test_the_api_passes_keys_the_constructor_accepts(self):
        from quant_platform.trading.broker import QMTBroker

        accepted = set(inspect.signature(QMTBroker.__init__).parameters) - {"self"}
        # What api/routes.py passes when broker == "qmt".
        assert {"account", "server", "password", "mode"} <= accepted

    def test_the_old_keys_would_have_raised(self):
        from quant_platform.trading.broker import QMTBroker

        accepted = set(inspect.signature(QMTBroker.__init__).parameters)
        assert "qmt_path" not in accepted
        assert "account_id" not in accepted


# ── BUG-43: IC decay served the oldest run ──────────────────────────────

class TestLatestRunSelection:
    def test_latest_run_uses_the_status_index(self):
        """`_run_store` has no timestamps, so the old
        `max(..., key=lambda k: _run_store[k].get("started_at", ""))` compared
        empty strings and returned the first-inserted run."""
        routes._run_status.clear()
        routes._run_status["old"] = {"started_at": "2026-01-01T00:00:00", "status": "completed"}
        routes._run_status["new"] = {"started_at": "2026-06-01T00:00:00", "status": "completed"}
        try:
            assert routes._latest_run_id() == "new"
        finally:
            routes._run_status.clear()

    def test_no_runs_is_a_400_not_a_keyerror(self):
        routes._run_status.clear()
        with pytest.raises(Exception) as exc:
            routes._latest_run_id()
        assert getattr(exc.value, "status_code", None) == 400
