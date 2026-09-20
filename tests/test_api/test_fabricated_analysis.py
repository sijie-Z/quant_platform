"""Regression tests for the two fabricated analysis endpoints (BUG-45).

`/api/analysis/ic-decay` manufactured an exponential decay curve from each
factor's mean IC plus noise, with the half-life derived from the same constant.
`/api/analysis/correlation` built a random matrix, including a rule that made
factors whose names contain "momentum" correlate with each other by
construction. Neither read any data. They are the same defect as BUG-24 --
in-sample or invented numbers presented as measurements -- on two more panels.
"""

import inspect
import re
from pathlib import Path

import pytest

from quant_platform.api import routes
from quant_platform.api.routes import get_factor_correlation, get_ic_decay


@pytest.mark.asyncio
async def test_ic_decay_reports_unavailable():
    out = await get_ic_decay()
    assert out["available"] is False
    assert out["factors"] == [], "an IC decay curve was invented"
    assert "factor panel" in out["reason"]


@pytest.mark.asyncio
async def test_correlation_reports_unavailable():
    out = await get_factor_correlation()
    assert out["available"] is False
    assert out["names"] == []
    assert out["matrix"] == [], "a correlation matrix was invented"
    assert "panel" in out["reason"]


class TestTheFabricationIsGone:
    """Source-level assertions, because the fabricated values were internally
    consistent and would pass any test written against their output."""

    def test_ic_decay_no_longer_computes_a_decay_curve(self):
        body = inspect.getsource(get_ic_decay)
        assert "np.exp" not in body.split('"""')[2], "the decay curve is back"

    def test_correlation_no_longer_randomises(self):
        body = inspect.getsource(get_factor_correlation)
        # Skip the docstring, which quotes the removed code.
        code = body.split('"""')[2]
        assert "random" not in code, "the random matrix is back"
        assert "momentum" not in code, "the hand-tuned momentum rule is back"

    def test_response_shapes_are_preserved_for_the_panels(self):
        """Both front-end panels guard on length; a missing key breaks them,
        an empty list renders nothing."""
        import asyncio

        decay = asyncio.run(get_ic_decay())
        corr = asyncio.run(get_factor_correlation())
        assert set(decay) >= {"available", "reason", "factors"}
        assert set(corr) >= {"available", "reason", "names", "matrix"}


class TestThePanelsDoNotFabricate:
    """Fixing the endpoints was not enough to stop these two numbers.

    Both were *also* generated in the browser, and the panels never called the
    API at all. `ICDecay.vue` rebuilt `Math.exp(-lag * 0.15) * baseIC` with the
    base IC chosen by substring-matching the factor name, and
    `FactorCorrelation.vue` rebuilt the seeded random matrix including the
    "momentum factors correlate" rule. Removing the generators server-side
    would have left the panels drawing the same fiction from the client.
    """

    COMPONENTS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "components"

    def _code(self, name: str) -> str:
        """Component source with comments removed.

        The explanatory comments quote the expressions that were deleted, so a
        naive `in` check matches the prose explaining their removal.
        """
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
