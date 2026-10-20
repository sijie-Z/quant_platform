"""Tests for Strategy Evaluation Gates."""

import pytest

from quant_platform.strategy.gates import (
    FAIL,
    PASS,
    REJECTED,
    SKIPPED,
    WARNING,
    GateConfig,
    GateReport,
    GateResult,
    GateRunner,
    check_complexity,
    check_data_coverage,
    check_drawdown_risk,
    check_ic_quality,
    check_trading_costs,
    count_assets_with_data,
)


class TestGateStatus:
    def test_pass_is_highest(self):
        assert PASS == "PASS"
        assert WARNING == "WARNING"

    def test_ic_quality_pass(self):
        ic = {"momentum": {"mean_ic": 0.05, "icir": 0.8},
              "volatility": {"mean_ic": 0.03, "icir": 0.5},
              "value": {"mean_ic": 0.04, "icir": 0.6}}
        config = GateConfig(minimum_ic=0.02, minimum_icir=0.0, minimum_factors_passing_ic=3)
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

    def test_default_icir_threshold_is_not_vacuous(self):
        """With the old default of 0.0, `abs(icir) >= 0` passed every factor
        and the gate could never fail, while still reporting an ICIR pass."""
        ic = {"weak": {"mean_ic": 0.05, "icir": 0.05}}
        result = check_ic_quality(ic, GateConfig())
        assert result.status == FAIL
        assert result.details["icir_gate_enabled"] is True

    def test_zero_threshold_disables_the_criterion_honestly(self):
        ic = {"flat": {"mean_ic": 0.05, "icir": 0.0}}
        config = GateConfig(minimum_icir=0.0, minimum_factors_passing_ic=1)
        result = check_ic_quality(ic, config)
        assert result.status == PASS
        assert "disabled" in result.message

    def test_none_metrics_do_not_raise(self):
        ic = {"broken": {"mean_ic": None, "icir": None}}
        result = check_ic_quality(ic, GateConfig())
        assert result.status == FAIL

    def test_all_factors_without_evaluable_observations_are_skipped(self):
        """No evidence is not evidence of a bad factor.

        A 30-stock universe used to yield NaN ICs for every factor; the gate
        compared the NaN metrics against ICIR=0.2 and reported a misleading
        weak-factor FAIL instead of saying the data was never evaluable.
        """
        ic = {
            "momentum": {"mean_ic": float("nan"), "icir": float("nan"), "n_obs": 0},
            "value": {"mean_ic": float("nan"), "icir": float("nan"), "n_obs": 3},
        }
        result = check_ic_quality(ic, GateConfig())

        assert result.status == SKIPPED
        assert "insufficient observations" in result.message.lower()
        assert "best n_obs=3" in result.message
        assert result.details["max_n_obs"] == 3

    def test_weak_factor_with_enough_observations_still_fails(self):
        """The observation-shortage escape hatch must not excuse a factor
        that has plenty of observations and a genuinely weak ICIR."""
        ic = {"weak": {"mean_ic": 0.001, "icir": 0.01, "n_obs": 200}}
        result = check_ic_quality(ic, GateConfig())

        assert result.status == FAIL
        assert "ICIR" in result.message

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
        assert len(report.gate_results) == 6

    def test_gate_runner_with_ic(self):
        runner = GateRunner()
        report = runner.run(
            strategy_name="test",
            ic_results={
                "mom": {"mean_ic": 0.05, "icir": 0.8},
                "value": {"mean_ic": 0.03, "icir": 0.5},
                "quality": {"mean_ic": 0.025, "icir": 0.3},
            },
            n_factors=5,
            n_params=10,
        )
        assert len(report.gate_results) == 6
        assert report.to_dict()["strategy_name"] == "test"

        by_name = {g.gate_name: g for g in report.gate_results}
        assert by_name["ic_quality"].status == PASS
        assert by_name["complexity"].status == PASS
        assert by_name["walk_forward"].status == SKIPPED
        assert by_name["data_coverage"].status == SKIPPED

    def test_gate_runner_rejects_a_weak_ic(self):
        """A `check_ic_quality` that always returns PASS satisfied every
        previous assertion, because none of them inspected a gate status."""
        report = GateRunner().run(
            strategy_name="weak",
            ic_results={"mom": {"mean_ic": 0.001, "icir": 0.01}},
            n_factors=5,
            n_params=10,
        )

        gate = next(g for g in report.gate_results if g.gate_name == "ic_quality")
        assert gate.status == FAIL

    def test_unknown_universe_is_skipped_not_faked(self):
        """`run_from_backtest_results` used to pass `len(first_weights)` as
        both the universe and the assets with data, so coverage was always
        100% and the gate always passed."""
        report = GateRunner().run_from_backtest_results(
            "test", backtest_result={"summary": {"max_drawdown": 0.1}},
        )
        gate = next(
            g for g in report.gate_results if g.gate_name == "data_coverage"
        )
        assert gate.status == SKIPPED

    def test_count_assets_with_data_measures_the_panel(self):
        import pandas as pd

        prices = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [1.0, None, None, None],
        })
        assert count_assets_with_data(prices) == (2, 1)

    def test_missing_panel_reports_zero(self):
        assert count_assets_with_data(None) == (0, 0)


class TestTradingCostGate:
    def test_cost_drag_above_limit_fails(self):
        result = check_trading_costs(
            {"annual_turnover": 2.0, "annual_cost_drag": 0.08}, GateConfig(),
        )
        assert result.status == FAIL

    def test_turnover_above_guideline_warns(self):
        result = check_trading_costs(
            {"annual_turnover": 8.0, "annual_cost_drag": 0.01}, GateConfig(),
        )
        assert result.status == WARNING

    def test_within_limits_passes(self):
        result = check_trading_costs(
            {"annual_turnover": 3.0, "annual_cost_drag": 0.02}, GateConfig(),
        )
        assert result.status == PASS

    def test_summary_without_cost_fields_is_skipped(self):
        assert check_trading_costs({}, GateConfig()).status == SKIPPED

    def test_gate_report_str(self):
        report = GateReport(strategy_name="test", overall_status=PASS, gate_results=[])
        lines = report.print_report()
        assert "Strategy Gates: test" in lines
        assert "PASS" in lines
