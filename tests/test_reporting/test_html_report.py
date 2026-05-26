"""Tests for reporting.html_report — HTML report generation."""

import os

import pytest

from quant_platform.reporting.html_report import generate_html_report


@pytest.fixture
def sample_results():
    return {
        "performance": {
            "total_return": 0.45,
            "annual_return": 0.08,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.12,
            "volatility": 0.15,
            "optimizer": "equal_weight",
        },
        "factors": [
            {"name": "momentum_1m", "mean_ic": 0.04, "icir": 2.1, "ic_positive_ratio": 0.65, "std_ic": 0.02},
            {"name": "rsi_14d", "mean_ic": 0.03, "icir": 1.8, "ic_positive_ratio": 0.60, "std_ic": 0.015},
        ],
        "risk": {"var_95": -0.02, "cvar_95": -0.03},
        "stress_tests": [
            {"scenario": "2008 Crisis", "cumulative_return": -0.08, "max_drawdown": -0.15},
            {"scenario": "2015 Crash", "cumulative_return": -0.12, "max_drawdown": -0.18},
        ],
        "exposure": {"sectors": {"金融": 0.25, "消费": 0.20, "科技": 0.15}},
        "chart_data": {
            "dates": ["2021-01-01", "2021-01-02"],
            "equity": [1.0, 1.01],
            "benchmark": [1.0, 1.005],
            "drawdown": [0, -0.005],
        },
    }


class TestHtmlReport:
    def test_generates_html_file(self, sample_results, tmp_path):
        output = str(tmp_path / "report.html")
        result_path = generate_html_report(sample_results, output_path=output)
        assert os.path.exists(result_path)
        with open(result_path, encoding="utf-8") as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_contains_kpi(self, sample_results, tmp_path):
        output = str(tmp_path / "report.html")
        result_path = generate_html_report(sample_results, output_path=output)
        with open(result_path, encoding="utf-8") as f:
            html = f.read()
        assert "45" in html or "0.45" in html

    def test_contains_factor_table(self, sample_results, tmp_path):
        output = str(tmp_path / "report.html")
        result_path = generate_html_report(sample_results, output_path=output)
        with open(result_path, encoding="utf-8") as f:
            html = f.read()
        assert "momentum_1m" in html

    def test_contains_stress_tests(self, sample_results, tmp_path):
        output = str(tmp_path / "report.html")
        result_path = generate_html_report(sample_results, output_path=output)
        with open(result_path, encoding="utf-8") as f:
            html = f.read()
        assert "2008 Crisis" in html

    def test_self_contained(self, sample_results, tmp_path):
        output = str(tmp_path / "report.html")
        result_path = generate_html_report(sample_results, output_path=output)
        with open(result_path, encoding="utf-8") as f:
            html = f.read()
        assert "<style" in html
