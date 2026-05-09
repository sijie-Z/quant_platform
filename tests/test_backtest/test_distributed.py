"""Tests for parallel backtest engine."""

import pytest

from quant_platform.backtest.distributed import (
    BacktestResult,
    ParallelBacktester,
    SweepResult,
)


class TestBacktestResult:
    def test_success(self):
        r = BacktestResult(params={"a": 1}, metrics={"sharpe_ratio": 1.5})
        assert r.success is True

    def test_failure(self):
        r = BacktestResult(params={"a": 1}, error="ValueError: bad")
        assert r.success is False

    def test_defaults(self):
        r = BacktestResult(params={})
        assert r.metrics == {}
        assert r.error is None
        assert r.duration_seconds == 0.0


class TestSweepResult:
    def test_summary(self):
        r1 = BacktestResult(params={"opt": "mvo"}, metrics={"sharpe_ratio": 1.5})
        r2 = BacktestResult(params={"opt": "ew"}, metrics={"sharpe_ratio": 0.8})
        sweep = SweepResult(results=[r1, r2], n_success=2)
        summary = sweep.summary()
        assert len(summary) == 2
        # Best first
        assert summary[0]["params"]["opt"] == "mvo"

    def test_summary_with_error(self):
        r1 = BacktestResult(params={"opt": "mvo"}, metrics={"sharpe_ratio": 1.5})
        r2 = BacktestResult(params={"opt": "bad"}, error="Failed")
        sweep = SweepResult(results=[r1, r2], n_success=1, n_failed=1)
        summary = sweep.summary()
        assert len(summary) == 2
        # Successful ones come first
        assert "error" in summary[1]


class TestParallelBacktester:
    def test_init(self):
        bt = ParallelBacktester(max_workers=2, timeout=60)
        assert bt.max_workers == 2
        assert bt.timeout == 60

    def test_find_best(self):
        r1 = BacktestResult(params={"a": 1}, metrics={"sharpe_ratio": 0.5})
        r2 = BacktestResult(params={"a": 2}, metrics={"sharpe_ratio": 1.5})
        r3 = BacktestResult(params={"a": 3}, error="fail")

        best = ParallelBacktester.find_best([r1, r2, r3])
        assert best.params["a"] == 2

    def test_find_best_empty(self):
        result = ParallelBacktester.find_best([])
        assert result is None

    def test_find_best_all_failed(self):
        r1 = BacktestResult(params={}, error="fail")
        result = ParallelBacktester.find_best([r1])
        assert result is None

    def test_find_best_custom_metric(self):
        r1 = BacktestResult(params={"a": 1}, metrics={"total_return": 0.1})
        r2 = BacktestResult(params={"a": 2}, metrics={"total_return": 0.3})

        best = ParallelBacktester.find_best([r1, r2], metric="total_return")
        assert best.params["a"] == 2

    def test_format_comparison(self):
        r1 = BacktestResult(
            params={"label": "MVO"},
            metrics={"sharpe_ratio": 1.5, "total_return": 0.3, "max_drawdown": -0.1, "annual_volatility": 0.2},
            duration_seconds=1.2,
        )
        r2 = BacktestResult(
            params={"label": "EW"},
            metrics={"sharpe_ratio": 0.8, "total_return": 0.15, "max_drawdown": -0.15, "annual_volatility": 0.18},
            duration_seconds=0.8,
        )

        text = ParallelBacktester.format_comparison([r1, r2])
        assert "PARALLEL BACKTEST COMPARISON" in text
        assert "MVO" in text
        assert "EW" in text
