"""Smoke tests for reporting and dashboard generation."""

import os
import tempfile

import numpy as np
import pandas as pd

from quant_platform.reporting.dashboard import generate_dashboard
from quant_platform.reporting.performance import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns_heatmap,
    plot_rolling_sharpe,
)
from quant_platform.risk.exposure import exposure_report
from quant_platform.risk.stress import run_all_stress_tests
from quant_platform.risk.var import var_summary


class TestPerformancePlots:
    """Test that chart functions run without error."""

    @staticmethod
    def _make_returns(n_days=500):
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        rng = np.random.default_rng(42)
        strategy = pd.Series(rng.normal(0.0005, 0.015, n_days), index=dates)
        benchmark = pd.Series(rng.normal(0.0003, 0.012, n_days), index=dates)
        return strategy, benchmark

    def test_equity_curve(self):
        s, b = self._make_returns()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            fig = plot_equity_curve(s, b, save_path=path)
            assert fig is not None
            assert os.path.exists(path)

    def test_drawdown(self):
        s, _ = self._make_returns()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            fig = plot_drawdown(s, save_path=path)
            assert fig is not None
            assert os.path.exists(path)

    def test_rolling_sharpe(self):
        s, _ = self._make_returns(300)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            fig = plot_rolling_sharpe(s, window=60, save_path=path)
            assert fig is not None
            assert os.path.exists(path)

    def test_monthly_heatmap(self):
        s, _ = self._make_returns(400)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.png")
            fig = plot_monthly_returns_heatmap(s, save_path=path)
            assert fig is not None
            assert os.path.exists(path)


class TestRiskReporting:
    """Test risk analysis functions."""

    def test_var_summary(self):
        dates = pd.bdate_range("2023-01-01", periods=500)
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.015, 500), index=dates)
        summary = var_summary(returns)
        assert "historical_var" in summary
        assert "parametric_var" in summary
        assert "monte_carlo_var" in summary
        assert "historical_cvar" in summary
        assert summary["historical_var"] > 0

    def test_stress_tests(self):
        dates = pd.bdate_range("2023-01-01", periods=500)
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.015, 500), index=dates)
        stress = run_all_stress_tests(returns)
        assert len(stress) >= 3
        assert "2008_financial_crisis" in stress.index or "2008" in str(stress.index[0]).lower()

    def test_exposure_report(self):
        assets = [f"A{i:03d}" for i in range(100)]
        weights = pd.Series(np.ones(100) / 100, index=assets)
        sector_map = pd.Series(
            np.random.choice(["银行", "电子", "医药", "食品饮料"], 100),
            index=assets,
        )
        report = exposure_report(weights, sector_map)
        assert "n_assets" in report
        assert report["n_assets"] == 100
        assert "effective_n" in report
        assert report["effective_n"] > 0


class TestDashboard:
    """Test dashboard generation."""

    def test_generate_basic(self):
        dates = pd.bdate_range("2023-01-01", periods=300)
        rng = np.random.default_rng(42)
        strategy = pd.Series(rng.normal(0.0005, 0.015, 300), index=dates)
        benchmark = pd.Series(rng.normal(0.0003, 0.012, 300), index=dates)

        results = {
            "daily_returns": strategy,
            "benchmark_returns": benchmark,
            "weights_history": {},
        }

        with tempfile.TemporaryDirectory() as d:
            report = generate_dashboard(
                results, output_dir=d, save_plots=False,
            )
            assert isinstance(report, str)
            assert "PERFORMANCE" in report
            assert "RISK" in report
            assert "STRESS TESTS" in report

    def test_generate_with_weights(self):
        dates = pd.bdate_range("2023-01-01", periods=300)
        rng = np.random.default_rng(42)
        strategy = pd.Series(rng.normal(0.0005, 0.015, 300), index=dates)
        benchmark = pd.Series(rng.normal(0.0003, 0.012, 300), index=dates)

        assets = [f"A{i:03d}" for i in range(50)]
        weights = pd.Series(np.ones(50) / 50, index=assets)
        weights_history = {pd.Timestamp("2023-12-29"): weights}

        results = {
            "daily_returns": strategy,
            "benchmark_returns": benchmark,
            "weights_history": weights_history,
        }

        with tempfile.TemporaryDirectory() as d:
            report = generate_dashboard(
                results, output_dir=d, save_plots=True, plot_format="png",
            )
            assert isinstance(report, str)
            assert "EXPOSURE" in report
            # Check that files were saved
            assert os.path.exists(os.path.join(d, "daily_returns.csv"))
            assert os.path.exists(os.path.join(d, "benchmark_returns.csv"))
            assert os.path.exists(os.path.join(d, "equity_curve.png"))
