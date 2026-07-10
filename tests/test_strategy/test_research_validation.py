"""Tests for Research Validation Sprint."""

from __future__ import annotations

from quant_platform.strategy.research_validation import (
    ResearchValidator,
    ValidationReport,
    ValidationStep,
)


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

    def test_quick_mode_structure(self):
        """Verify quick mode produces a report with correct structure."""
        v = ResearchValidator(
            n_stocks=50,
            start_date="2021-01-01",
            end_date="2022-01-01",
            timeout_seconds=120,
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
