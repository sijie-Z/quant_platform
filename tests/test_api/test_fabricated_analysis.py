"""Regression tests for the two fabricated analysis endpoints (BUG-45).

`/api/analysis/ic-decay` manufactured an exponential decay curve from each
factor's mean IC plus noise, with the half-life derived from the same constant.
`/api/analysis/correlation` built a random matrix, including a rule that made
factors whose names contain "momentum" correlate with each other by
construction. Neither read any data. They are the same defect as BUG-24 --
in-sample or invented numbers presented as measurements -- on two more panels.
"""

import inspect

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
