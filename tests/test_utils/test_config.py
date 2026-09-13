"""Tests for configuration loading and validation."""

import copy
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


class TestLoadConfigDoesNotMutateItsInput:
    """Regression for BUG-41.

    `_parse_config` removed `constraints`, `covariance` and `var` from the
    caller's dict with `pop()`. `main.py run` passes the same dict to
    `load_config()` and then to `VersionManager.save()` a few lines later, so
    every auto-saved snapshot was written *without* those sections -- and
    `config rollback` then silently reverted them to defaults.
    """

    @staticmethod
    def _raw():
        return {
            "portfolio": {
                "optimizer": "mean_variance",
                "constraints": {"max_weight": 0.05},
                "covariance": {"method": "ledoit_wolf"},
            },
            "risk": {"var": {"method": "historical"}},
        }

    def test_nested_sections_survive_the_call(self):
        raw = self._raw()
        load_config(raw=raw)
        assert "constraints" in raw["portfolio"]
        assert "covariance" in raw["portfolio"]
        assert "var" in raw["risk"]

    def test_the_input_dict_is_unchanged_entirely(self):
        raw = self._raw()
        before = copy.deepcopy(raw)
        load_config(raw=raw)
        assert raw == before

    def test_a_snapshot_taken_afterwards_still_has_them(self):
        """This is the actual failure mode: the snapshot is written after
        load_config() has already run."""
        raw = self._raw()
        load_config(raw=raw)
        snapshot = copy.deepcopy(raw)  # what VersionManager.save would persist
        assert snapshot["portfolio"]["constraints"]["max_weight"] == 0.05
        assert snapshot["portfolio"]["covariance"]["method"] == "ledoit_wolf"
        assert snapshot["risk"]["var"]["method"] == "historical"

    def test_values_are_still_parsed_into_the_config(self):
        """Not mutating the input must not mean losing the values."""
        cfg = load_config(raw=self._raw())
        assert cfg.portfolio.constraints.max_weight == 0.05
        assert cfg.portfolio.covariance.method == "ledoit_wolf"
        assert cfg.risk.var.method == "historical"
