"""Regression tests for POST /api/walkforward (BUG-24).

`_run_walkforward` sliced the in-sample equity curve, labelled the slices "OOS"
and manufactured a Sharpe from them -- its own comment said so: "Generate
synthetic walk-forward results based on the run data". No signal was recomputed
and no model refit, so the numbers were in-sample performance wearing an
out-of-sample label, on the dashboard panel titled "Walk-Forward 验证".
"""

from quant_platform.api.routes import _run_walkforward


def _stored_run(n_days=800):
    """The shape _run_store holds: chart_data with dates and equity only."""
    dates = [f"2023-01-{i % 28 + 1:02d}" for i in range(n_days)]
    equity = [1.0 + i * 0.0005 for i in range(n_days)]
    return {"chart_data": {"dates": dates, "equity": equity}}


class TestWalkforwardDoesNotFabricate:
    def test_reports_unavailable_rather_than_inventing_folds(self):
        out = _run_walkforward(_stored_run(), {})
        assert out["available"] is False
        assert out["fold_details"] == [], "folds were fabricated from the equity curve"
        assert out["oos_equity"] == []
        assert out["n_folds"] == 0

    def test_says_why_and_points_at_the_real_path(self):
        out = _run_walkforward(_stored_run(), {})
        reason = out["reason"]
        assert "stored run" in reason
        # The two things that would actually be needed.
        assert "returns" in reason
        assert "factor" in reason
        assert "main.py walkforward" in reason

    def test_stability_metrics_are_zero_not_invented(self):
        stability = _run_walkforward(_stored_run(), {})["stability"]
        assert set(stability.values()) == {0}

    def test_response_shape_is_preserved_for_the_panel(self):
        """WalkForward.vue guards on `fold_details?.length`, so it needs the key
        to exist and be a list -- otherwise the panel breaks rather than
        rendering nothing."""
        out = _run_walkforward(_stored_run(), {})
        for key in ("n_folds", "mode", "train_period", "test_period",
                    "fold_details", "oos_equity", "oos_dates", "stability"):
            assert key in out, f"{key} missing; the panel would break"

    def test_echoes_the_requested_parameters_back(self):
        out = _run_walkforward(_stored_run(), {"train_period": 252, "test_period": 63, "mode": "expanding"})
        assert out["train_period"] == 252
        assert out["test_period"] == 63
        assert out["mode"] == "expanding"
