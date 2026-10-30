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
            # A real run carries the sample size; below 10 observations the
            # report renders n/a instead of the 0.0 defaults (R93).
            "total_days": 500,
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


class TestHtmlReportInjection:
    """Data-derived strings were interpolated raw; ``json.dumps`` left
    ``</script>`` intact, so provider/run metadata could break out."""

    PAYLOAD = '</script><img src=x onerror="alert(1)">'

    def _generate(self, results, tmp_path, title=None):
        output = str(tmp_path / "report.html")
        generate_html_report(
            results, output_path=output,
            **({"title": title} if title is not None else {}),
        )
        with open(output, encoding="utf-8") as fh:
            return fh.read()

    def test_titles_and_tables_are_escaped(self, sample_results, tmp_path):
        payload = self.PAYLOAD
        results = {
            **sample_results,
            "performance": {
                **sample_results["performance"], "optimizer": payload,
            },
            "factors": [{
                "name": payload, "mean_ic": 0.01, "icir": 1.0,
                "ic_positive_ratio": 0.5, "std_ic": 0.01,
            }],
            "exposure": {"sectors": {payload: 0.2}},
            "stress_tests": [{
                "scenario": payload, "cumulative_return": -0.1,
                "max_drawdown": -0.2,
            }],
            "provenance": {
                "effective_provider": payload,
                "requested_provider": "</title><script>alert(2)</script>",
            },
        }

        html = self._generate(results, tmp_path, title=payload)

        assert "</script><img" not in html
        assert "</title><script>" not in html
        assert "&lt;img src=x onerror=" in html

    def test_embedded_json_cannot_close_the_script_block(
        self, sample_results, tmp_path,
    ):
        results = {
            **sample_results,
            "chart_data": {
                **sample_results["chart_data"],
                "dates": [self.PAYLOAD, "2024-01-02"],
            },
        }

        html = self._generate(results, tmp_path)

        assert "</script><img" not in html
        assert "\\u003c/script" in html

    def test_report_carries_a_csp(self, sample_results, tmp_path):
        html = self._generate(sample_results, tmp_path)

        assert "Content-Security-Policy" in html
        assert "script-src 'unsafe-inline' https://cdn.jsdelivr.net" in html
