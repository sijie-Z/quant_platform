"""Tests for parallel backtest engine."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.backtest.distributed import (
    BacktestResult,
    ParallelBacktester,
    SweepResult,
    _apply_config_overrides,
    _compute_metrics,
)
from quant_platform.config.schema import Config


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


class TestWorkerMetricsMatchMainBacktest:
    """Sweep results must be computed the same way as the main backtest.

    The worker divided annual return by annual volatility with no risk-free
    rate and no zero-volatility guard, and measured drawdown from the first
    *return* instead of from inception. A constant-return series had a
    floating-point standard deviation of ~3e-18, so "Sharpe" was ~1e16.
    """

    def test_constant_returns_have_zero_sharpe(self):
        daily = pd.Series(
            [0.001] * 252, index=pd.bdate_range("2023-01-02", periods=252),
        )

        metrics = _compute_metrics(daily)

        assert metrics["sharpe_ratio"] == 0.0

    def test_drawdown_is_measured_from_inception(self):
        daily = pd.Series(
            [-0.10, 0.0, 0.0], index=pd.bdate_range("2023-01-02", periods=3),
        )

        assert _compute_metrics(daily)["max_drawdown"] == pytest.approx(-0.10)

    def test_metrics_match_the_shared_metric_functions(self):
        from quant_platform.backtest.metrics import all_metrics

        rng = np.random.default_rng(11)
        daily = pd.Series(
            rng.normal(0.0005, 0.01, 300),
            index=pd.bdate_range("2022-01-03", periods=300),
        )

        metrics = _compute_metrics(daily)
        expected = all_metrics(daily)

        for key in (
            "total_return", "annual_return", "annual_volatility",
            "sharpe_ratio", "max_drawdown",
        ):
            assert metrics[key] == pytest.approx(expected[key], abs=1e-4), key


class TestParamGridAliases:
    """`param_grid={"optimizer": [...], "n_stocks": [...]}` is the documented
    usage, but the worker applied every key with `setattr` on the root Config.
    Flat keys silently created dead attributes and the sweep ran the same
    configuration N times. Keys must be mapped to their real config paths,
    and anything else must fail loudly.
    """

    def test_documented_flat_aliases_produce_different_configs(self):
        first, second = Config(), Config()

        _apply_config_overrides(first, {"optimizer": "equal_weight"})
        _apply_config_overrides(second, {"optimizer": "risk_parity"})

        assert first.portfolio.optimizer == "equal_weight"
        assert second.portfolio.optimizer == "risk_parity"
        # The old bug: `setattr(config, "optimizer", ...)` created this.
        assert not hasattr(first, "optimizer")

    def test_n_stocks_alias_reaches_the_universe(self):
        config = Config()

        _apply_config_overrides(config, {"n_stocks": 123})

        assert config.universe.n_stocks == 123
        assert not hasattr(config, "n_stocks")

    def test_dotted_paths_still_work(self):
        config = Config()

        _apply_config_overrides(config, {"backtest.rebalance_frequency": "weekly"})

        assert config.backtest.rebalance_frequency == "weekly"

    def test_unknown_flat_key_raises(self):
        with pytest.raises(ValueError, match="not_a_param"):
            _apply_config_overrides(Config(), {"not_a_param": 1})

    def test_unknown_dotted_path_raises(self):
        with pytest.raises(ValueError, match="portfolio.not_a_field"):
            _apply_config_overrides(Config(), {"portfolio.not_a_field": 1})

    def test_run_sweep_rejects_unknown_keys_before_spawning(self):
        """The validation must fire in the parent process, not turn into one
        failed cell per combination inside the worker pool."""
        bt = ParallelBacktester(max_workers=1)

        with pytest.raises(ValueError, match="not_a_param"):
            bt.run_sweep({"not_a_param": [1, 2]})
