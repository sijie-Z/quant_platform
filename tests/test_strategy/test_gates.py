"""Tests for Strategy Evaluation Gates."""

import pytest
from quant_platform.strategy.gates import (
    GateConfig,
    GateRunner,
    GateResult,
    GateReport,
    PASS, WARNING, FAIL, REJECTED, SKIPPED,
    check_ic_quality,
    check_drawdown_risk,
    check_complexity,
    check_data_coverage,
)


class TestGateStatus:
    def test_pass_is_highest(self):
        assert PASS == "PASS"
        assert WARNING == "WARNING"

    def test_ic_quality_pass(self):
        ic = {"momentum": {"mean_ic": 0.05, "icir": 0.8},
              "volatility": {"mean_ic": 0.03, "icir": 0.5},
              "value": {"mean_ic": 0.04, "icir": 0.6}}
        config = GateConfig(minimum_ic=0.02, minimum_icir=0.0, minimum_factor_history_count=3)
        result = check_ic_quality(ic, config)
        assert result.status == PASS

    def test_ic_quality_empty(self):
        result = check_ic_quality({}, GateConfig())
        assert result.status == SKIPPED

    def test_ic_quality_fail_low_icir(self):
        ic = {"momentum": {"mean_ic": 0.01, "icir": 0.0}}
        config = GateConfig(minimum_ic=0.02, minimum_icir=0.1)
        result = check_ic_quality(ic, config)
        assert result.status == FAIL

    def test_drawdown_pass(self):
        summary = {"max_drawdown": 0.10, "sharpe_ratio": 0.5}
        config = GateConfig(maximum_drawdown=0.25)
        result = check_drawdown_risk(summary, config)
        assert result.status == PASS

    def test_drawdown_fail(self):
        summary = {"max_drawdown": 0.30, "sharpe_ratio": 0.5}
        config = GateConfig(maximum_drawdown=0.25)
        result = check_drawdown_risk(summary, config)
        assert result.status == FAIL

    def test_drawdown_kill(self):
        summary = {"max_drawdown": 0.40, "sharpe_ratio": 0.5}
        config = GateConfig(maximum_drawdown=0.25, kill_drawdown=0.35)
        result = check_drawdown_risk(summary, config)
        assert result.status == REJECTED

    def test_drawdown_negative_sharpe_warning(self):
        summary = {"max_drawdown": 0.10, "sharpe_ratio": -0.5}
        config = GateConfig(maximum_drawdown=0.25)
        result = check_drawdown_risk(summary, config)
        assert result.status == WARNING

    def test_complexity_pass(self):
        config = GateConfig(maximum_factor_count=10)
        result = check_complexity(5, 10, config)
        assert result.status == PASS

    def test_complexity_warning(self):
        config = GateConfig(maximum_factor_count=10)
        result = check_complexity(15, 30, config)
        assert result.status == WARNING

    def test_data_coverage_pass(self):
        config = GateConfig(minimum_price_coverage=0.80)
        result = check_data_coverage(100, 95, config)
        assert result.status == PASS

    def test_data_coverage_fail(self):
        config = GateConfig(minimum_price_coverage=0.80)
        result = check_data_coverage(100, 50, config)
        assert result.status == FAIL

    def test_gate_runner_empty(self):
        runner = GateRunner()
        report = runner.run(strategy_name="test")
        assert report.overall_status == PASS  # all skipped or pass
        assert len(report.gate_results) == 5

    def test_gate_runner_with_ic(self):
        runner = GateRunner()
        report = runner.run(
            strategy_name="test",
            ic_results={"mom": {"mean_ic": 0.05, "icir": 0.8}},
            n_factors=5,
            n_params=10,
        )
        assert len(report.gate_results) == 5
        assert report.to_dict()["strategy_name"] == "test"

    def test_gate_report_str(self):
        report = GateReport(strategy_name="test", overall_status=PASS, gate_results=[])
        lines = report.print_report()
        assert "Strategy Gates: test" in lines
        assert "PASS" in lines
