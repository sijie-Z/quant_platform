"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest

from quant_platform.utils.config import load_config


class TestConfigLoading:
    """Test config file loading."""

    def test_load_default_config(self):
        """Default config should load without errors."""
        config = load_config()
        assert config.universe.n_stocks == 500
        assert config.backtest.initial_capital == 10_000_000
        assert config.portfolio.optimizer == "mean_variance"

    def test_load_custom_config(self):
        """Custom YAML config should be loadable."""
        import yaml

        custom = {
            "universe": {"n_stocks": 100, "exclude_st": False, "exclude_suspended": False},
            "data": {"start_date": "2023-01-01", "end_date": "2023-12-31", "frequency": "daily"},
            "alpha": {"method": "equal_weight", "lookback": 60, "min_icir": -0.5},
            "portfolio": {
                "optimizer": "equal_weight",
                "constraints": {
                    "long_only": True, "max_weight": 0.10,
                    "max_sector_exposure": 0.40, "max_turnover": 0.50, "lot_size": 100,
                },
                "covariance": {"method": "sample", "lookback": 60, "ewma_half_life": 30},
                "risk_aversion": 1.0,
            },
            "backtest": {
                "rebalance_frequency": "weekly", "initial_capital": 1_000_000,
                "benchmark": "equal_weight",
            },
            "costs": {
                "commission": 0.0005, "stamp_tax": 0.001,
                "slippage": 0.002, "slippage_model": "proportional",
            },
            "risk": {
                "var": {"confidence": 0.99, "horizon": 5, "method": "monte_carlo"},
                "stress_scenarios": ["2008_financial_crisis"],
            },
            "output": {
                "results_dir": "/tmp/results", "save_plots": False, "plot_format": "pdf",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom, f)
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config.universe.n_stocks == 100
            assert config.backtest.initial_capital == 1_000_000
            assert config.backtest.rebalance_frequency == "weekly"
            assert config.portfolio.optimizer == "equal_weight"
            assert config.portfolio.constraints.max_weight == 0.10
            assert config.costs.commission == 0.0005
            assert config.output.plot_format == "pdf"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_config_types(self):
        """Config values should have correct types."""
        config = load_config()
        assert isinstance(config.universe.n_stocks, int)
        assert isinstance(config.backtest.initial_capital, (int, float))
        assert isinstance(config.portfolio.constraints.long_only, bool)
        assert isinstance(config.portfolio.constraints.max_weight, float)
        assert isinstance(config.costs.commission, float)

    def test_nonexistent_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
