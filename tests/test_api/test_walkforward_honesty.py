"""Regression tests for POST /api/walkforward (BUG-24).

`_run_walkforward` once sliced the in-sample equity curve, labelled the
slices "OOS" and manufactured a Sharpe from them. It now runs the real
``WalkForwardValidator`` on the research context, with the signal recomputed
inside each fold from train-only data. The full request path is covered in
``test_endpoint_wiring.py``; these guards assert the honest-failure shape and
that the slicing code is gone.
"""

import inspect

import pandas as pd

from quant_platform.api import routes
from tests.test_api.research_helpers import make_context


def _short_context(n_days: int = 60):
    index = pd.bdate_range("2024-01-01", periods=n_days)
    assets = [f"{i:06d}.SZ" for i in range(40)]
    prices = pd.DataFrame(100.0, index=index, columns=assets)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    factors = {"momentum": prices.pct_change(fill_method=None)}
    return make_context(prices, returns, factors)


class TestWalkforwardDoesNotFabricate:
    def test_reports_unavailable_when_the_sample_cannot_hold_one_fold(self):
        out = routes._run_walkforward(
            _short_context(60),
            {
                "train_period": 252, "test_period": 63,
                "mode": "rolling", "folds": None,
            },
        )
        assert out["available"] is False
        assert out["fold_details"] == [], "folds were fabricated from the equity curve"
        assert out["oos_equity"] == []
        assert out["n_folds"] == 0
        assert "at least 315 sessions" in out["reason"]
        assert "has 60" in out["reason"]

    def test_stability_metrics_are_zero_not_invented_when_unavailable(self):
        stability = routes._run_walkforward(
            _short_context(),
            {"train_period": 252, "test_period": 63},
        )["stability"]
        assert set(stability.values()) == {0}

    def test_response_shape_is_preserved_for_the_panel(self):
        """WalkForward.vue guards on `fold_details?.length`, so it needs the key
        to exist and be a list -- otherwise the panel breaks rather than
        rendering nothing."""
        out = routes._run_walkforward(
            _short_context(),
            {"train_period": 252, "test_period": 63},
        )
        for key in ("n_folds", "mode", "train_period", "test_period",
                    "fold_details", "oos_equity", "oos_dates", "stability"):
            assert key in out, f"{key} missing; the panel would break"

    def test_echoes_the_requested_parameters_back(self):
        out = routes._run_walkforward(
            _short_context(),
            {"train_period": 252, "test_period": 63, "mode": "expanding"},
        )
        assert out["train_period"] == 252
        assert out["test_period"] == 63
        assert out["mode"] == "expanding"

    def test_the_validator_is_the_real_one(self):
        body = inspect.getsource(routes._run_walkforward)
        assert "WalkForwardValidator" in body
        assert "factors=context.processed_factors" in body, (
            "the signal is not recomputed per fold from train-only data"
        )
        assert "chart_data" not in body, "the stored equity curve is back"
        assert "synthetic walk-forward" not in body.lower()
