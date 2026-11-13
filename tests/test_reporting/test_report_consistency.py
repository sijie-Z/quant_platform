"""HTML report numbers must match the summary/metrics object.

Every KPI rendered by ``generate_html_report`` is parsed back out of the
generated HTML and compared with the ``performance`` / ``risk`` mapping that
the same call was given. The tests also pin the reporting-consistency wave
fixes: return-window labels (R90), date-aligned benchmark rendering (R91),
rolling-Sharpe axes and empty states (R92), n/a for small samples (R93),
VaR confidence labels (R97), non-finite factor stats (R99), empty inputs
(R100), signed vs unsigned formatters (R101) and optimizer fallbacks (R96).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pandas as pd
import pytest

from quant_platform.reporting.html_report import (
    MIN_OBSERVATIONS,
    generate_html_report,
)


def _render(results: dict, tmp_path, name: str = "report.html") -> str:
    path = generate_html_report(results, output_path=str(tmp_path / name))
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _kpi(html: str, metric: str) -> str:
    match = re.search(
        rf'data-kpi="{metric}".*?<div class="kpi-value[^"]*">([^<]*)</div>',
        html,
        re.S,
    )
    assert match is not None, f"KPI {metric!r} not found"
    return match.group(1).strip()


def _const(name: str, html: str):
    """JSON value of a ``const <name> = ...;`` declaration in the report."""
    match = re.search(rf"const {name} = (.+?);\n", html, re.S)
    assert match is not None, name
    return json.loads(match.group(1))


@pytest.fixture
def full_payload():
    """A 300-observation run with a shorter (200-day) benchmark."""
    dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=300)]
    return {
        "performance": {
            "total_return": 0.009754,
            "strategy_total_return_full_window": -0.01786,
            "benchmark_total_return": -0.017506,
            "excess_return": 0.02726,
            "benchmark_coverage": 200 / 300,
            "benchmark_days": 200,
            "total_days": 300,
            "annual_return": -0.015624,
            "annual_volatility": 0.20,
            "sharpe_ratio": -0.1116,
            "sortino_ratio": -0.2,
            "max_drawdown": -0.09,
            "calmar_ratio": -0.17,
            "win_rate": 0.48,
            "profit_loss_ratio": 0.9,
            "information_ratio": 0.5,
            "tracking_error": 0.03,
            "n_rebalances": 12,
            "optimizer": "mean_variance",
            "effective_optimizer": "mean_variance+equal_weight_fallback",
            "n_optimizer_fallbacks": 7,
            "optimizer_fallback_rate": 7 / 12,
        },
        "risk": {
            "historical_var": 0.0353,
            "parametric_var": 0.0246,
            "historical_cvar": 0.045,
            "confidence": 0.99,
            "horizon_days": 5,
        },
        "factors": [
            {"name": "momentum_1m", "mean_ic": 0.04, "icir": 2.1,
             "ic_positive_ratio": 0.65, "std_ic": 0.02},
        ],
        "chart_data": {
            "dates": dates,
            "equity": [1.0] * 300,
            "benchmark": [1.0] * 200,
            "drawdown": [0.0] * 300,
            "rolling_sharpe": [0.1] * 1750,
            "rolling_sharpe_dates": [
                str(d.date()) for d in pd.bdate_range("2021-01-04", periods=1750)
            ],
            "monthly_returns": {"years": ["2024"], "months": list(range(1, 13)),
                                "data": [[None] * 12]},
        },
        "stress_tests": [],
    }


class TestKpiConsistency:
    def test_api_payload_labels_common_window_when_fields_are_dropped(self, tmp_path):
        # api/schemas.py currently drops strategy_total_return_full_window and
        # benchmark_coverage. The equity curve is still full-history, so a
        # total_return that disagrees with it must be labelled as the
        # benchmark-common window, never presented as the full-history KPI.
        dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=300)]
        equity = [1.0] * 299 + [0.98214]
        results = {
            "performance": {
                "total_days": 300,
                "total_return": 0.00975,
                "annual_return": -0.015624,
                "annual_volatility": 0.20,
                "sharpe_ratio": -0.1116,
                "max_drawdown": -0.09,
            },
            "chart_data": {"dates": dates, "equity": equity,
                           "drawdown": [0.0] * 300},
        }
        html = _render(results, tmp_path, "api-shaped.html")

        assert _kpi(html, "total_return") == "+0.97%"
        assert "Total Return (benchmark window)" in html
        assert _kpi(html, "annual_return") == "-1.56%"
        assert "coverage unavailable in this payload" in html

    def test_api_payload_with_matching_equity_keeps_plain_label(self, tmp_path):
        dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=30)]
        equity = [1.0] * 29 + [1.05]
        results = {
            "performance": {"total_days": 30, "total_return": 0.05,
                            "annual_return": 0.05},
            "chart_data": {"dates": dates, "equity": equity,
                           "drawdown": [0.0] * 30},
        }
        html = _render(results, tmp_path, "api-full.html")
        assert _kpi(html, "total_return") == "+5.00%"
        assert "Total Return (benchmark window)" not in html
        assert "coverage unavailable" not in html

    def test_total_return_uses_full_window_and_states_it(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        perf = full_payload["performance"]

        assert _kpi(html, "total_return") == (
            f"{perf['strategy_total_return_full_window'] * 100:+.2f}%"
        )
        assert "Total Return (since inception)" in html
        # The benchmark-common window is shown separately, never as the headline.
        assert _kpi(html, "benchmark_window_return") == (
            f"{perf['total_return'] * 100:+.2f}%"
        )
        assert _kpi(html, "benchmark_coverage").startswith("66.7% (200/300")

    def test_every_kpi_number_matches_the_summary(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        perf = full_payload["performance"]
        risk = full_payload["risk"]

        assert _kpi(html, "annual_return") == f"{perf['annual_return'] * 100:+.2f}%"
        assert _kpi(html, "annual_volatility") == (
            f"{perf['annual_volatility'] * 100:.2f}%"
        )
        assert _kpi(html, "sharpe_ratio") == f"{perf['sharpe_ratio']:,.2f}"
        assert _kpi(html, "max_drawdown") == f"{perf['max_drawdown'] * 100:+.2f}%"
        assert _kpi(html, "win_rate") == f"{perf['win_rate'] * 100:.1f}%"
        assert _kpi(html, "information_ratio") == f"{perf['information_ratio']:,.2f}"
        assert _kpi(html, "n_rebalances") == str(perf["n_rebalances"])
        assert _kpi(html, "tracking_error") == f"{perf['tracking_error'] * 100:.2f}%"
        assert _kpi(html, "excess_return") == f"{perf['excess_return'] * 100:+.2f}%"
        assert _kpi(html, "historical_var") == (
            f"{abs(risk['historical_var']) * 100:.2f}%"
        )
        assert _kpi(html, "historical_cvar") == (
            f"{abs(risk['historical_cvar']) * 100:.2f}%"
        )

    def test_unsigned_formatters_for_vol_and_losses(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        # R101: a loss or a volatility is never displayed as a signed return.
        assert not _kpi(html, "annual_volatility").startswith("+")
        assert not _kpi(html, "tracking_error").startswith("+")
        assert not _kpi(html, "historical_var").startswith("+")
        assert not _kpi(html, "historical_cvar").startswith("+")
        # Signed returns keep the explicit sign.
        assert _kpi(html, "annual_return").startswith("-")
        assert _kpi(html, "excess_return").startswith("+")

    def test_var_labels_use_configured_confidence_and_horizon(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        # R97: 99% / 5-day, never a hardcoded 95%.
        assert "Historical VaR (99%, 5-day)" in html
        assert "Historical CVaR (99%, 5-day)" in html
        assert "VaR (95%)" not in html
        assert "CVaR (95%)" not in html

    def test_optimizer_fallbacks_are_displayed(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        assert "mean_variance+equal_weight_fallback" in html
        assert "7 fallbacks" in html
        assert "58.3%" in html  # 7/12 of rebalances

    def test_generated_timestamp_is_utc_aware(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        match = re.search(r"Generated: ([^<]+)", html)
        assert match is not None
        assert match.group(1).strip().endswith("+00:00")


class TestSmallSamples:
    def test_insufficient_observations_render_na(self, tmp_path):
        dates = [str(d.date()) for d in pd.bdate_range("2023-01-02", periods=5)]
        results = {
            "performance": {
                "total_days": 5,
                "total_return": -0.00043,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": -0.02,
                "win_rate": 0.0,
                "n_rebalances": 0,
            },
            "risk": {"confidence": 0.95, "horizon_days": 1, "historical_var": 0.02},
            "chart_data": {"dates": dates, "equity": [1.0] * 5, "drawdown": [0.0] * 5,
                           "rolling_sharpe": None},
        }
        html = _render(results, tmp_path, "short.html")

        for metric in (
            "total_return", "annual_return", "sharpe_ratio", "max_drawdown",
            "win_rate", "n_rebalances", "historical_var",
        ):
            assert _kpi(html, metric) == "n/a", metric
        assert f"Insufficient data: n={MIN_OBSERVATIONS - 5}" in html
        # R93: the old report printed Win Rate 0.0% and all-zero metrics.
        assert "0.00%" not in html

    def test_empty_inputs_do_not_crash(self, tmp_path):
        html = _render({"chart_data": {"dates": []}}, tmp_path, "empty.html")
        assert 'data-notice="no-data"' in html
        assert "Period: n/a" in html
        assert _kpi(html, "total_return") == "n/a"


class TestBenchmarkAlignment:
    def test_misaligned_benchmark_is_omitted_with_a_note(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        assert _const("benchmark", html) == []
        assert "Benchmark series omitted" in html
        assert "200 points cannot be date-aligned to 300 strategy dates" in html
        # R91: the old code plotted the 200 points at the first 200 dates.
        assert "name: 'Benchmark'" not in html

    def test_benchmark_with_dates_is_reindexed_by_date(self, full_payload, tmp_path):
        chart = full_payload["chart_data"]
        # Benchmark exists at dates 0, 1 and 5 only.
        chart["benchmark"] = [1.10, 1.20, 1.30]
        chart["benchmark_dates"] = [chart["dates"][0], chart["dates"][1],
                                    chart["dates"][5]]
        html = _render(full_payload, tmp_path, "aligned.html")

        benchmark = _const("benchmark", html)
        assert len(benchmark) == len(chart["dates"])
        assert benchmark[0] == 1.10
        assert benchmark[1] == 1.20
        assert benchmark[2] is None
        assert benchmark[5] == 1.30
        assert "name: 'Benchmark'" in html
        assert "Benchmark series omitted" not in html


    def test_equal_length_without_coverage_is_not_assumed_aligned(self, tmp_path):
        dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=10)]
        results = {
            "performance": {"total_days": 10, "total_return": 0.01},
            "chart_data": {"dates": dates, "equity": [1.0] * 9 + [1.01],
                           "drawdown": [0.0] * 10, "benchmark": [1.0] * 10},
        }
        html = _render(results, tmp_path, "equal-unknown.html")
        # R91: equal point counts after independent downsampling are not proof
        # of a shared clock.
        assert "date alignment cannot be verified" in html
        assert "name: 'Benchmark'" not in html

    def test_equal_length_with_partial_coverage_is_omitted(self, tmp_path):
        dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=10)]
        results = {
            "performance": {"total_days": 10, "total_return": 0.01,
                            "benchmark_coverage": 0.5},
            "chart_data": {"dates": dates, "equity": [1.0] * 9 + [1.01],
                           "drawdown": [0.0] * 10, "benchmark": [1.0] * 10},
        }
        html = _render(results, tmp_path, "equal-partial.html")
        assert "date alignment cannot be verified" in html
        assert "name: 'Benchmark'" not in html

    def test_equal_length_with_full_coverage_is_plotted(self, tmp_path):
        dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", periods=10)]
        results = {
            "performance": {"total_days": 10, "total_return": 0.01,
                            "benchmark_coverage": 1.0},
            "chart_data": {"dates": dates, "equity": [1.0] * 9 + [1.01],
                           "drawdown": [0.0] * 10, "benchmark": [1.0] * 10},
        }
        html = _render(results, tmp_path, "equal-full.html")
        assert "name: 'Benchmark'" in html
        assert "Benchmark series omitted" not in html


class TestRollingSharpeAxis:
    def test_supplied_rolling_dates_are_the_x_axis(self, full_payload, tmp_path):
        html = _render(full_payload, tmp_path)
        chart = full_payload["chart_data"]
        # R92: 1750 rolling points vs 300 strategy dates: dates.slice(-n)
        # would have shown all 300 dates for the first points.
        assert _const("rollingDates", html) == chart["rolling_sharpe_dates"]
        assert "data: rollingDates" in html

    def test_null_rolling_sharpe_renders_a_note_and_keeps_monthly_chart(self, tmp_path):
        dates = [str(d.date()) for d in pd.bdate_range("2023-01-02", periods=200)]
        results = {
            "performance": {
                "total_days": 200, "total_return": 0.05,
                "annual_return": 0.06, "annual_volatility": 0.1,
                "sharpe_ratio": 0.3, "max_drawdown": -0.05,
            },
            "chart_data": {
                "dates": dates,
                "equity": [1.0] * 200,
                "drawdown": [0.0] * 200,
                "rolling_sharpe": None,
                "monthly_returns": {
                    "years": ["2023"], "months": list(range(1, 13)),
                    "data": [[0.01, 0.02, None, None, None, None,
                              None, None, None, None, None, None]],
                },
            },
        }
        html = _render(results, tmp_path, "no-rolling.html")
        # R92: `dates.slice(-rollingSharpe.length)` on null threw before the
        # monthly chart was initialized.
        assert "Rolling Sharpe unavailable" in html
        assert "rollingSharpe.length" not in html.split("const monthlyData")[0]
        assert "const shChart = null;" in html
        assert "echarts.init(document.getElementById('monthlyChart'))" in html

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    def test_inline_script_executes_when_rolling_sharpe_is_null(self, tmp_path):
        """Execute the report's inline JS with DOM/ECharts stubs.

        The R92 defect was a runtime TypeError, so string assertions alone
        cannot catch a regression; this runs the same script a browser would.
        """
        dates = [str(d.date()) for d in pd.bdate_range("2023-01-02", periods=200)]
        results = {
            "performance": {
                "total_days": 200, "total_return": 0.05,
                "annual_return": 0.06, "annual_volatility": 0.1,
                "sharpe_ratio": 0.3, "max_drawdown": -0.05,
            },
            "chart_data": {
                "dates": dates,
                "equity": [1.0] * 200,
                "drawdown": [0.0] * 200,
                "rolling_sharpe": None,
                "monthly_returns": {
                    "years": ["2023"], "months": list(range(1, 13)),
                    "data": [[0.01] + [None] * 11],
                },
            },
        }
        html_path = tmp_path / "exec.html"
        generate_html_report(results, output_path=str(html_path))

        harness = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const match = html.match(/<script>([\s\S]*?)<\/script>/);
if (!match) { console.error('NO_INLINE_SCRIPT'); process.exit(2); }
const options = {};
const elements = {};
function el(id) {
  if (!elements[id]) elements[id] = { id: id, innerHTML: '' };
  return elements[id];
}
const echarts = {
  init: function (target) {
    return { setOption: function (o) { options[target.id] = o; }, resize: function () {} };
  },
  graphic: { LinearGradient: function () {} },
};
const document = { getElementById: el };
const window = { addEventListener: function () {} };
try {
  new Function('echarts', 'document', 'window', match[1])(echarts, document, window);
} catch (err) {
  console.error('SCRIPT_ERROR: ' + err.message);
  process.exit(1);
}
const monthly = options.monthlyChart;
console.log(JSON.stringify({
  charts: Object.keys(options),
  monthly_cells: monthly ? monthly.series[0].data.length : 0,
}));
"""
        completed = subprocess.run(
            [shutil.which("node"), "-e", harness, str(html_path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        assert "sharpeChart" not in summary["charts"]
        assert summary["monthly_cells"] == 1


class TestFactorAndStressTables:
    def test_non_finite_factor_statistics_render_na(self, tmp_path):
        results = {
            "performance": {"total_days": 100},
            "factors": [
                {"name": "bad_factor", "mean_ic": float("nan"),
                 "icir": float("nan"), "ic_positive_ratio": float("nan"),
                 "std_ic": float("nan")},
                {"name": "good_factor", "mean_ic": 0.02, "icir": 1.2,
                 "ic_positive_ratio": 0.55, "std_ic": 0.01},
            ],
            "chart_data": {"dates": ["2024-01-02"], "equity": [1.0]},
        }
        html = _render(results, tmp_path, "factors.html")
        factor_section = html.split("Factor IC Analysis", 1)[1].split("Sector Exposure", 1)[0]
        assert "nan" not in factor_section.lower()
        assert factor_section.count("n/a") >= 4
        assert "+0.0200" in factor_section  # the finite factor still renders

    def test_missing_stress_values_render_na(self, tmp_path):
        results = {
            "performance": {"total_days": 100},
            "stress_tests": [{"scenario": "2008 Crisis"}],
            "chart_data": {"dates": ["2024-01-02"], "equity": [1.0]},
        }
        html = _render(results, tmp_path, "stress.html")
        assert "2008 Crisis" in html
        stress_section = html.split("Stress Tests", 1)[1].split("Risk Metrics", 1)[0]
        assert "n/a" in stress_section
