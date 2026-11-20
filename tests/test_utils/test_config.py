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

    def test_default_cost_model_follows_the_statutory_schedule(self):
        """R9: stamp tax defaults to None (date schedule) and slippage to the
        impact model; an explicit value stays a flat override."""
        config = load_config()
        assert config.costs.stamp_tax is None
        assert config.costs.slippage_model == "impact"
        assert config.costs.impact_model == "composite"

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


class TestFactorSectionParsing:
    def test_params_and_processing_are_kept(self):
        """Both were flattened away, so editing them changed nothing."""
        cfg = load_config(raw={
            "factors": {
                "technical": {
                    "momentum_1m": {"enabled": True, "params": {"period": 10}},
                },
                "fundamental": {},
                "processing": {
                    "winsorize": {"enabled": True, "lower": 0.02, "upper": 0.98},
                    "standardize": {"enabled": True, "method": "rank"},
                    "neutralize": {"enabled": True, "by": ["sector"]},
                },
            },
        })

        assert cfg.factors.params["momentum_1m"] == {"period": 10}
        assert cfg.factors.processing["standardize"]["method"] == "rank"
        assert cfg.factors.enabled_technicals == ("momentum_1m",)
        assert cfg.factors.enabled_fundamentals == ()

    def test_an_explicitly_empty_section_enables_nothing(self):
        """`tuple(enabled) or default` silently re-enabled every default."""
        cfg = load_config(raw={"factors": {"technical": {}, "fundamental": {}}})

        assert cfg.factors.enabled_technicals == ()
        assert cfg.factors.enabled_fundamentals == ()

    def test_omitting_the_sections_keeps_the_defaults(self):
        cfg = load_config(raw={})

        assert "momentum_1m" in cfg.factors.enabled_technicals


class TestConfigValidation:
    """S12: nonsense values used to load and fail much later (or never).

    Each invalid value must raise ConfigValidationError naming the path.
    """

    @pytest.mark.parametrize("raw, needle", [
        ({"universe": {"n_stocks": -5}}, "universe.n_stocks"),
        ({"universe": {"n_stocks": 0}}, "universe.n_stocks"),
        ({"portfolio": {"constraints": {"max_weight": 2.5}}}, "max_weight"),
        ({"portfolio": {"constraints": {"max_weight": 0}}}, "max_weight"),
        ({"portfolio": {"constraints": {"max_sector_exposure": -1}}},
         "max_sector_exposure"),
        ({"portfolio": {"constraints": {"lot_size": 0}}}, "lot_size"),
        ({"costs": {"commission": 5.0}}, "commission"),
        ({"costs": {"stamp_tax": -0.5}}, "stamp_tax"),
        ({"costs": {"slippage": -0.1}}, "slippage"),
        ({"backtest": {"rebalance_frequency": "fortnightly"}},
         "rebalance_frequency"),
        ({"data": {"provider": "no_such_provider"}}, "provider"),
        ({"data": {"start_date": "not-a-date"}}, "start_date"),
        ({"data": {"start_date": "2025-01-01", "end_date": "2024-01-01"}},
         "start_date"),
        ({"risk": {"var": {"confidence": 1.5}}}, "confidence"),
        ({"execution": {"qmt": {"mode": "paper"}}}, "qmt.mode"),
    ])
    def test_invalid_values_are_rejected(self, raw, needle):
        from quant_platform.utils.config import ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(raw=raw)

        assert needle in str(exc_info.value)

    def test_unknown_per_factor_key_is_rejected(self):
        """`exclude_recent` beside `params` used to be silently dropped."""
        from quant_platform.utils.config import ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(raw={
                "factors": {
                    "technical": {
                        "momentum_12m": {
                            "enabled": True,
                            "params": {"period": 252},
                            "exclude_recent": 21,
                        },
                    },
                },
            })

        message = str(exc_info.value)
        assert "exclude_recent" in message
        assert "params" in message

    def test_valid_configs_still_load(self):
        """Backward compatibility: a normal custom config is unchanged."""
        cfg = load_config(raw={
            "universe": {"n_stocks": 100},
            "data": {"provider": "synthetic", "start_date": "2023-01-01"},
            "portfolio": {"constraints": {"max_weight": 0.10, "lot_size": 100}},
            "backtest": {"rebalance_frequency": "weekly"},
        })
        assert cfg.universe.n_stocks == 100

    def test_stamp_tax_null_and_explicit_override_are_both_valid(self):
        """R9: null selects the statutory schedule; an explicit rate in
        [0, 0.01] is a flat override."""
        assert load_config(raw={"costs": {"stamp_tax": None}}).costs.stamp_tax is None
        assert load_config(
            raw={"costs": {"stamp_tax": 0.001}},
        ).costs.stamp_tax == pytest.approx(0.001)



class TestEnvironmentOverrides:
    """S23: documented env overrides must actually apply (restart-only)."""

    def test_qmt_password_env_wins_over_config(self, monkeypatch):
        monkeypatch.setenv("QMT_PASSWORD", "env-secret")
        cfg = load_config(raw={
            "execution": {"qmt": {"password": "yaml-secret"}},
        })
        assert cfg.execution.qmt.password == "env-secret"

    def test_qmt_password_config_used_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("QMT_PASSWORD", raising=False)
        cfg = load_config(raw={"execution": {"qmt": {"password": "yaml-secret"}}})
        assert cfg.execution.qmt.password == "yaml-secret"

    def test_data_provider_env_override(self, monkeypatch):
        monkeypatch.setenv("QUANT_DATA_PROVIDER", "baostock")
        cfg = load_config(raw={"data": {"provider": "synthetic"}})
        assert cfg.data.provider == "baostock"
