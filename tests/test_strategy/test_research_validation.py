"""Tests for Research Validation Sprint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.strategy.research_validation import (
    ResearchValidator,
    ValidationReport,
    ValidationStep,
    select_walk_forward_factors,
)


def _make_case(seed=0, n_dates=160, n_assets=25, strong=2, noise=2):
    """Shared returns panel plus factor panels correlated (or not) with it."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    base = rng.normal(size=(n_dates, n_assets))
    returns = pd.DataFrame(base, index=dates, columns=assets)

    factors = {}
    for i in range(strong):
        # Rank IC around 0.75 per date, so ICIR * sqrt(n) is huge.
        panel = base + rng.normal(scale=0.8, size=returns.shape)
        factors[f"strong_{i}"] = pd.DataFrame(panel, index=dates, columns=assets)
    for i in range(noise):
        panel = rng.normal(size=returns.shape)
        factors[f"noise_{i}"] = pd.DataFrame(panel, index=dates, columns=assets)
    return factors, returns


def _ic_summaries(factors, returns):
    """Build ic_results the same way the factor step does."""
    out = {}
    for name, factor in factors.items():
        summary = ic_summary(rank_ic(factor, returns))
        out[name] = {
            "mean_ic": summary["mean_ic"],
            "std_ic": summary["std_ic"],
            "icir": summary["icir"],
        }
    return out


class TestSelectWalkForwardFactors:
    def test_significant_factors_survive_and_noise_does_not(self):
        factors, returns = _make_case(seed=0, strong=2, noise=2)
        ic_results = _ic_summaries(factors, returns)

        result = select_walk_forward_factors(ic_results, factors, returns)

        assert set(result["survivors"]) == {"strong_0", "strong_1"}
        assert set(result["selected"]) == {"strong_0", "strong_1"}
        assert len(result["selected"]) == 2
        assert result["fallback"] is False
        assert result["selection_warning"] is None
        for name in ("noise_0", "noise_1"):
            assert result["p_values"][name] > 0.05
            assert result["p_values"][name] == result["p_values"][name]  # finite
        assert result["n_obs"]["strong_0"] > 0

    def test_fewer_than_two_survivors_takes_the_actual_count(self):
        factors, returns = _make_case(seed=1, strong=1, noise=2)
        ic_results = _ic_summaries(factors, returns)

        result = select_walk_forward_factors(ic_results, factors, returns)

        assert result["survivors"] == ["strong_0"]
        assert result["selected"] == ["strong_0"]
        assert result["fallback"] is False

    def test_all_insignificant_factors_fall_back_with_warning(self):
        factors, returns = _make_case(seed=2, strong=0, noise=3)
        ic_results = _ic_summaries(factors, returns)

        result = select_walk_forward_factors(ic_results, factors, returns)

        assert result["survivors"] == []
        assert result["fallback"] is True
        assert result["selection_warning"] == (
            "no factor survives BH-FDR; falling back to top-|ICIR|"
        )
        expected = [
            name for name, _ in sorted(
                ic_results.items(),
                key=lambda item: abs(item[1]["icir"]),
                reverse=True,
            )[:2]
        ]
        assert result["selected"] == expected

    def test_step_details_carry_factor_selection(self, monkeypatch):
        from quant_platform.backtest.walkforward import WalkForwardValidator

        factors, returns = _make_case(seed=3, strong=1, noise=2)
        ic_results = _ic_summaries(factors, returns)
        monkeypatch.setattr(
            WalkForwardValidator, "run",
            lambda self, **kwargs: {
                "fold_metrics": [], "stability": {}, "aggregate_metrics": {},
            },
        )

        validator = ResearchValidator(n_stocks=25)
        step = validator._step_walk_forward(
            factors, ic_results, None, returns, None, {}, None,
            list(factors), max_folds=2,
        )

        assert step.status == "completed"
        detail = step.details["factor_selection"]
        assert detail["selected"] == ["strong_0"]
        assert detail["fallback"] is False
        assert any("selection-biased" in w for w in step.warnings)

    def test_step_marks_the_fallback_in_details_and_warnings(self, monkeypatch):
        from quant_platform.backtest.walkforward import WalkForwardValidator

        factors, returns = _make_case(seed=4, strong=0, noise=3)
        ic_results = _ic_summaries(factors, returns)
        monkeypatch.setattr(
            WalkForwardValidator, "run",
            lambda self, **kwargs: {
                "fold_metrics": [], "stability": {}, "aggregate_metrics": {},
            },
        )

        validator = ResearchValidator(n_stocks=25)
        step = validator._step_walk_forward(
            factors, ic_results, None, returns, None, {}, None,
            list(factors), max_folds=2,
        )

        assert step.status == "completed"
        detail = step.details["factor_selection"]
        assert detail["fallback"] is True
        assert len(detail["selected"]) == 2
        assert "no factor survives BH-FDR; falling back to top-|ICIR|" in step.warnings


class TestValidationStep:
    def test_defaults(self):
        step = ValidationStep(name="test", category="eval", target="all", status="completed")
        assert step.name == "test"
        assert step.status == "completed"
        assert step.runtime_seconds == 0.0

    def test_to_dict(self):
        step = ValidationStep(
            name="test", category="eval", target="all",
            status="completed", runtime_seconds=1.5,
        )
        d = step.to_dict()
        assert d["name"] == "test"
        assert d["runtime_seconds"] == 1.5
        assert isinstance(d["runtime_seconds"], float)


class TestValidationReport:
    def test_defaults(self):
        report = ValidationReport(mode="quick", start_time="2026-01-01")
        assert report.mode == "quick"
        assert report.total_runtime == 0.0
        assert len(report.steps) == 0

    def test_to_dict(self):
        report = ValidationReport(mode="quick", start_time="2026-01-01")
        report.total_runtime = 42.5
        d = report.to_dict()
        assert d["mode"] == "quick"
        assert d["total_runtime"] == 42.5

    def test_print_report(self):
        report = ValidationReport(mode="quick", start_time="2026-01-01")
        report.total_runtime = 10.0
        lines = report.print_report()
        assert "quick" in lines
        assert "10.0s" in lines


class TestResearchValidator:
    def test_init(self):
        v = ResearchValidator(n_stocks=100)
        assert v.n_stocks == 100
        assert v.start_date == "2021-01-01"

    def test_quick_mode_structure(self, tmp_path):
        """Verify quick mode produces a report with correct structure."""
        v = ResearchValidator(
            n_stocks=50,
            start_date="2021-01-01",
            end_date="2022-01-01",
            timeout_seconds=120,
            # A quick synthetic sprint must not append rows to the real
            # research database: this test used to write 2022-01-01-dated
            # rows into data/factor_research.db on every suite run.
            factor_store_path=str(tmp_path / "factor_research.db"),
        )
        report = v.run_quick(max_factors=2, max_folds=1, n_symbols=50)
        assert isinstance(report, ValidationReport)
        assert report.mode == "quick"
        assert len(report.steps) > 0

        # Check that at least data loading completed
        data_step = next((s for s in report.steps if s.name == "load_data"), None)
        assert data_step is not None

        # Steps should have reasonable statuses
        for step in report.steps:
            assert step.status in ("completed", "skipped", "failed", "running")
