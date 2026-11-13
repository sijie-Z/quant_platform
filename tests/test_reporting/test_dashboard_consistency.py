"""Dashboard text-report consistency tests (R90, R93, R96, R97, R100-R102)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.reporting.dashboard import (
    _analyze_drawdown_detail,
    generate_dashboard,
    resolve_run_dir,
    write_manifest,
)
from quant_platform.reporting.performance import (
    _monthly_returns_by_period_end,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
)


def _returns(n_days: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.Series(rng.normal(0.0004, 0.012, n_days), index=dates)


def _performance_section(report: str) -> str:
    return report.split("--- PERFORMANCE ---", 1)[1].split("--- DRAWDOWN", 1)[0]


class TestSmallSamples:
    def test_five_observations_print_na_not_zero(self, tmp_path):
        returns = pd.Series(
            [-0.001, 0.002, -0.02, 0.001, 0.003],
            index=pd.bdate_range("2023-01-02", periods=5),
        )
        report = generate_dashboard(
            {"daily_returns": returns}, output_dir=str(tmp_path), save_plots=False,
        )
        section = _performance_section(report)
        assert "INSUFFICIENT DATA: n=5" in section
        assert "n/a" in section
        # The old report printed Total Return 0.00% / Sharpe 0.00.
        assert "0.00%" not in section
        assert "Recovery Time" not in report  # drawdown detail suppressed
        assert "n/a (insufficient data)" in report

    def test_empty_series_produces_a_no_data_report(self, tmp_path):
        report = generate_dashboard(
            {"daily_returns": pd.Series(dtype=float)},
            output_dir=str(tmp_path),
            save_plots=False,
        )
        assert "--- NO DATA ---" in report
        assert "empty" in report.lower()


class TestWindows:
    def test_total_return_window_and_coverage_are_stated(self, tmp_path):
        returns = _returns()
        summary = {
            "total_days": 300,
            "total_return": 0.009754,                 # common window
            "strategy_total_return_full_window": -0.01786,
            "benchmark_coverage": 200 / 300,
            "benchmark_days": 200,
            "annual_return": -0.0156,
        }
        report = generate_dashboard(
            {"daily_returns": returns, "summary": summary},
            output_dir=str(tmp_path),
            save_plots=False,
        )
        assert "Total Return (since inception):" in report
        assert "-1.79%" in report
        assert "Benchmark-Window Return:" in report
        assert "+0.98%" in report
        assert "Benchmark Coverage:" in report
        assert "66.7% (200/300 days)" in report
        assert "Annual Return:" in report

    def test_benchmark_window_label_without_full_window_field(self, tmp_path):
        returns = _returns()
        summary = {
            "total_days": 300,
            "total_return": 0.009754,
            "benchmark_coverage": 200 / 300,
        }
        report = generate_dashboard(
            {"daily_returns": returns, "summary": summary},
            output_dir=str(tmp_path),
            save_plots=False,
        )
        # An older stored summary has no full-window field: label the window
        # rather than silently mixing it with the full-history chart.
        assert "Total Return (benchmark window):" in report
        # The window note must not simultaneously claim "since inception".
        assert "since inception" not in report
        assert (
            "total return and benchmark-based metrics (excess/IR/TE) use the "
            "common strategy/benchmark window" in report
        )


class TestLabels:
    def test_var_labels_follow_configured_confidence_and_horizon(self, tmp_path):
        report = generate_dashboard(
            {"daily_returns": _returns()},
            output_dir=str(tmp_path),
            save_plots=False,
            var_confidence=0.99,
            var_horizon=5,
        )
        assert "Historical VaR (99%, 5-day):" in report
        assert "Parametric VaR (99%, 5-day):" in report
        assert "Historical CVaR (99%, 5-day):" in report
        assert "VaR (95%)" not in report

    def test_effective_optimizer_and_fallbacks_are_reported(self, tmp_path):
        summary = {
            "total_days": 300,
            "total_return": 0.05,
            "optimizer": "mean_variance",
            "effective_optimizer": "mean_variance+equal_weight_fallback",
            "n_optimizer_fallbacks": 7,
            "optimizer_fallback_rate": 7 / 12,
            "n_rebalances": 12,
        }
        report = generate_dashboard(
            {"daily_returns": _returns(), "summary": summary},
            output_dir=str(tmp_path),
            save_plots=False,
        )
        assert "effective: mean_variance+equal_weight_fallback" in report
        assert "fallbacks=7 (58.3% of rebalances)" in report

    def test_provenance_is_read_from_the_run_manifest(self, tmp_path):
        # `cmd_analyze` passes no provenance; the run manifest supplies it.
        write_manifest(tmp_path, {
            "run_id": "abc",
            "requested_provider": "baostock",
            "provider": "baostock",
            "synthetic": False,
            "date_range": {"start": "2020-01-01", "end": "2024-12-31",
                           "data_date": "2024-12-31"},
        })
        report = generate_dashboard(
            {"daily_returns": _returns()},
            output_dir=str(tmp_path),
            save_plots=False,
        )
        assert "baostock (requested: baostock)" in report
        assert "2020-01-01 → 2024-12-31" in report

    def test_provenance_is_rendered(self, tmp_path):
        report = generate_dashboard(
            {"daily_returns": _returns()},
            output_dir=str(tmp_path),
            save_plots=False,
            provenance={
                "requested_provider": "tushare",
                "effective_provider": "tushare",
                "synthetic": True,
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "data_date": "2024-12-31",
            },
        )
        assert "Provider:" in report
        assert "tushare (SYNTHETIC) (requested: tushare)" in report
        assert "Data Range:" in report
        assert "2020-01-01 → 2024-12-31" in report


class TestDrawdownDetail:
    def test_unrecovered_drawdown_prints_na_not_none(self, tmp_path):
        # 12 observations keep the report out of the <10 n/a mode; the 10%
        # loss is never recovered by the following +0.1% days.
        returns = pd.Series(
            [-0.10] + [0.001] * 11,
            index=pd.bdate_range("2023-01-02", periods=12),
        )
        report = generate_dashboard(
            {"daily_returns": returns}, output_dir=str(tmp_path), save_plots=False,
        )
        assert "N/A (not yet recovered)" in report
        assert "None days" not in report

    def test_recovery_is_measured_from_the_trough(self):
        returns = pd.Series(
            [-0.10, 0.01, 0.01, 0.20] + [0.001] * 8,
            index=pd.bdate_range("2023-01-02", periods=12),
        )
        detail = _analyze_drawdown_detail(returns)
        assert detail is not None
        # Peak 2023-01-02, trough 2023-01-03, recovery 2023-01-06 (clamped to
        # the last label): trough -> recovery = 3 calendar days. The old code
        # printed peak -> recovery (4) beside the peak -> trough duration (1).
        assert detail["duration_days"] == 1
        assert detail["recovery_days_from_trough"] == 3

    def test_report_labels_recovery_interval(self, tmp_path):
        returns = pd.Series(
            [-0.10, 0.01, 0.01, 0.20] + [0.001] * 8,
            index=pd.bdate_range("2023-01-02", periods=12),
        )
        report = generate_dashboard(
            {"daily_returns": returns}, output_dir=str(tmp_path), save_plots=False,
        )
        assert "3 days (trough→recovery)" in report


class TestMonthlyBucketing:
    def test_returns_are_bucketed_by_period_end(self):
        index = pd.DatetimeIndex(["2023-12-29", "2024-01-31"])
        monthly = _monthly_returns_by_period_end(pd.Series([0.10, 0.20], index=index))

        # 2023-12-29 is the start of a period ending in January.
        assert np.isclose(monthly.loc[pd.Timestamp("2024-01-31")], 0.10)
        assert np.isclose(monthly.loc[pd.Timestamp("2024-02-29")], 0.20)
        assert (
            pd.Timestamp("2023-12-31") not in monthly.index
            or pd.isna(monthly.loc[pd.Timestamp("2023-12-31")])
        )

    def test_empty_series_returns_empty_monthly(self):
        assert _monthly_returns_by_period_end(pd.Series(dtype=float)).empty

    def test_heatmap_empty_months_are_blank_not_zero(self):
        january = pd.bdate_range("2024-01-02", "2024-01-31")
        april = pd.bdate_range("2024-04-01", "2024-04-30")
        returns = pd.Series(0.01, index=january.append(april))

        fig = plot_monthly_returns_heatmap(returns)
        try:
            texts = [text.get_text() for text in fig.axes[0].texts]
            # R98: an all-NaN month used to be plotted (and annotated) as a
            # real 0.0% month, and the literal "nan" could be printed.
            # Seaborn leaves masked NaN cells unannotated.
            assert "nan" not in [text.lower() for text in texts]
            assert "0.0" not in texts
            assert texts  # real months are still annotated
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


class TestRollingSharpePlot:
    def test_constant_returns_show_na_legend_not_nan_or_inf(self):
        returns = pd.Series(
            [0.001] * 300, index=pd.bdate_range("2023-01-02", periods=300),
        )
        fig = plot_rolling_sharpe(returns)
        try:
            legend = fig.axes[0].get_legend()
            assert legend is not None
            labels = [text.get_text() for text in legend.get_texts()]
            assert "Mean: n/a" in labels
            assert not any("nan" in label or "inf" in label for label in labels)
            assert any(
                "rf = 0" in text.get_text() for text in fig.axes[0].texts
            ) or "rf = 0" in fig.axes[0].get_title()
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


class TestManifestResolution:
    def test_corrupt_run_manifest_is_logged_not_silent(self, tmp_path, caplog):
        (tmp_path / "manifest.json").write_text("{broken", encoding="utf-8")
        with caplog.at_level("WARNING"):
            report = generate_dashboard(
                {"daily_returns": _returns()},
                output_dir=str(tmp_path),
                save_plots=False,
            )
        assert "Unreadable run manifest" in caplog.text
        assert "PERFORMANCE" in report

    def test_corrupt_pointer_warns_and_falls_back(self, tmp_path, caplog):
        (tmp_path / "latest_manifest.json").write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING"):
            resolved = resolve_run_dir(tmp_path)
        assert resolved == tmp_path
        assert "Ignoring unreadable latest_manifest.json" in caplog.text

    def test_pointer_to_missing_run_warns(self, tmp_path, caplog):
        (tmp_path / "latest_manifest.json").write_text(
            '{"run_dir": "runs/does-not-exist"}', encoding="utf-8",
        )
        with caplog.at_level("WARNING"):
            resolved = resolve_run_dir(tmp_path)
        assert resolved == tmp_path
        assert "missing/corrupt run" in caplog.text
