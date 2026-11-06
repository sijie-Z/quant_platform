"""Regression tests for the two analysis endpoints (BUG-45).

`/api/analysis/ic-decay` once manufactured an exponential decay curve from
each factor's mean IC plus noise, and `/api/analysis/correlation` built a
random matrix (including a rule that made factors whose names contain
"momentum" correlate by construction). Both now compute from the research
context's processed factor panel and forward returns.

`test_endpoint_wiring.py` covers the full request path; this module keeps the
source-level and panel-level guards that the original defect motivated.
"""

import inspect
import re
from pathlib import Path

from quant_platform.api import routes
from tests.test_api.research_helpers import make_context


class TestICDecayIsMeasured:
    def test_ic_decay_reports_real_values(self, prices, returns, sample_factor):
        context = make_context(
            prices, returns, {"momentum": sample_factor},
        )
        out = routes._compute_ic_decay(context)

        assert out["available"] is True
        assert out["horizons"] == [1, 5, 10, 20]
        assert len(out["factors"]) == 1
        points = out["factors"][0]["points"]
        assert [p["horizon"] for p in points] == [1, 5, 10, 20]
        assert all(p["n"] > 0 for p in points)
        assert any(p["ic"] != 0.0 for p in points)


class TestCorrelationIsMeasured:
    def test_correlation_reports_a_real_matrix(self, prices, returns, sample_factor):
        factors = {
            "momentum": sample_factor,
            "volatility": returns.rolling(20).std(),
        }
        context = make_context(prices, returns, factors)
        out = routes._compute_factor_correlation(context)

        assert out["available"] is True
        assert out["names"] == ["momentum", "volatility"]
        matrix = out["matrix"]
        assert len(matrix) == 2 and len(matrix[0]) == 2
        assert matrix[0][0] == 1.0 and matrix[1][1] == 1.0
        assert matrix[0][1] == matrix[1][0]
        assert -1.0 <= matrix[0][1] <= 1.0


class TestTheFabricationIsGone:
    """Source-level assertions, because fabricated values were internally
    consistent and would pass any test written against their output."""

    def test_ic_decay_no_longer_computes_a_decay_curve(self):
        body = inspect.getsource(routes._compute_ic_decay)
        assert "np.exp" not in body, "the decay curve is back"
        assert "random" not in body, "noise is back"

    def test_correlation_no_longer_randomises(self):
        body = inspect.getsource(routes._compute_factor_correlation)
        assert "random" not in body, "the random matrix is back"
        assert "momentum" not in body, "the hand-tuned momentum rule is back"


class TestThePanelsDoNotFabricate:
    """The endpoints are not the only place the fiction lived: both panels
    rebuilt the same numbers in the browser and never called the API."""

    COMPONENTS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "components"

    def _code(self, name: str) -> str:
        src = (self.COMPONENTS / name).read_text(encoding="utf-8")
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        src = re.sub(r"<!--.*?-->", "", src, flags=re.DOTALL)
        return src

    def test_ic_decay_panel_has_no_generator_and_calls_the_api(self):
        src = self._code("ICDecay.vue")
        assert "Math.exp(-lag" not in src, "the decay curve is generated client-side again"
        assert "1103515245" not in src, "the noise LCG is back"
        assert "getICDecay" in src, "the panel does not fetch its own data"

    def test_correlation_panel_has_no_generator_and_calls_the_api(self):
        src = self._code("FactorCorrelation.vue")
        assert "buildCorrelationMatrix" not in src, "the random matrix is back"
        assert "1103515245" not in src, "the noise LCG is back"
        assert "getFactorCorrelation" in src, "the panel does not fetch its own data"

    def test_both_panels_render_an_unavailable_state(self):
        for name in ("ICDecay.vue", "FactorCorrelation.vue"):
            src = (self.COMPONENTS / name).read_text(encoding="utf-8")
            assert "available === false" in src, (
                f"{name} would render an empty chart instead of saying why"
            )
