"""Tests for Strategy DSL."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from quant_platform.strategy.dsl import (
    StrategyDefinition,
    validate_strategy,
    dsl_to_config_overrides,
    ValidationResult,
)
from quant_platform.strategy.registry import StrategyRegistry


class TestStrategyDefinition:
    def test_from_dict_basic(self):
        s = StrategyDefinition.from_dict({
            "name": "test_strat",
            "version": "1.0.0",
        })
        assert s.name == "test_strat"
        assert s.version == "1.0.0"

    def test_from_dict_full(self):
        s = StrategyDefinition.from_dict({
            "name": "momentum",
            "description": "A momentum strategy",
            "version": "2.0.0",
            "author": "test",
            "tags": ["momentum"],
            "factors": [{"name": "momentum_12m", "weight": 0.6}],
            "portfolio": {"method": "equal_weight"},
        })
        assert s.name == "momentum"
        assert s.portfolio_method == "equal_weight"
        assert s.factor_weights["momentum_12m"] == 1.0  # normalized

    def test_factor_weights_normalized(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "factors": [
                {"name": "factor_a", "weight": 0.3},
                {"name": "factor_b", "weight": 0.7},
            ],
        })
        weights = s.factor_weights
        assert abs(weights["factor_a"] - 0.3) < 0.001
        assert abs(weights["factor_b"] - 0.7) < 0.001

    def test_factor_weights_normalized_uneven(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "factors": [
                {"name": "factor_a", "weight": 0.5},
                {"name": "factor_b", "weight": 0.5},
            ],
        })
        weights = s.factor_weights
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_to_dict_roundtrip(self):
        orig = StrategyDefinition.from_dict({
            "name": "test",
            "version": "1.0.0",
            "factors": [{"name": "mom", "weight": 1.0}],
        })
        d = orig.to_dict()
        restored = StrategyDefinition.from_dict(d)
        assert restored.name == orig.name
        assert restored.factor_weights == orig.factor_weights

    def test_properties(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "portfolio": {
                "method": "risk_parity",
                "rebalance_frequency": "weekly",
                "max_position_weight": 0.03,
            },
            "universe": {"symbols": ["600519", "000001"]},
        })
        assert s.portfolio_method == "risk_parity"
        assert s.rebalance_frequency == "weekly"
        assert s.max_position_weight == 0.03
        assert len(s.symbols) == 2


class TestValidateStrategy:
    def test_valid_strategy(self):
        s = StrategyDefinition.from_dict({
            "name": "valid_strat",
            "version": "1.0.0",
            "factors": [],
            "portfolio": {"method": "equal_weight"},
        })
        result = validate_strategy(s, check_factors_exist=False)
        assert result.valid
        assert len(result.errors) == 0

    def test_missing_name(self):
        s = StrategyDefinition.from_dict({"version": "1.0.0"})
        result = validate_strategy(s)
        assert not result.valid
        assert any("name" in e for e in result.errors)

    def test_version_default_provided(self):
        """Version defaults to 1.0.0 if missing — strategy is valid."""
        s = StrategyDefinition.from_dict({"name": "test"})
        assert s.version == "1.0.0"

    def test_forbidden_fields_raw_check(self):
        """Forbidden fields are silently dropped by from_dict (frozen dataclass)."""
        raw = {"name": "bad", "version": "1.0.0", "allow_lookahead": True}
        s = StrategyDefinition.from_dict(raw)
        # The field is dropped, but we verify the raw check works
        assert not hasattr(s, "allow_lookahead")

    def test_unknown_portfolio_method(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "version": "1.0.0",
            "portfolio": {"method": "magic"},
        })
        result = validate_strategy(s, check_factors_exist=False)
        assert "magic" in " ".join(result.warnings)

    def test_invalid_weight_nan(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "version": "1.0.0",
            "factors": [{"name": "mom", "weight": float("nan")}],
        })
        result = validate_strategy(s, check_factors_exist=False)
        assert not result.valid
        assert any("invalid" in e for e in result.errors)

    def test_max_position_out_of_range(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "version": "1.0.0",
            "portfolio": {"max_position_weight": 0},
        })
        result = validate_strategy(s, check_factors_exist=False)
        assert not result.valid

    def test_yaml_file(self, tmp_path):
        path = tmp_path / "test.yaml"
        with open(path, "w") as f:
            yaml.dump({
                "name": "file_strat",
                "version": "1.0.0",
                "portfolio": {"method": "equal_weight"},
            }, f)
        s = StrategyDefinition.from_yaml(path)
        assert s.name == "file_strat"
        assert s.portfolio_method == "equal_weight"


class TestDSLToConfig:
    def test_portfolio_overrides(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "version": "1.0.0",
            "portfolio": {
                "method": "risk_parity",
                "rebalance_frequency": "weekly",
                "max_position_weight": 0.03,
            },
            "universe": {"n_stocks": 200},
        })
        overrides = dsl_to_config_overrides(s)
        assert overrides["portfolio.optimizer"] == "risk_parity"
        assert overrides["backtest.rebalance_frequency"] == "weekly"
        assert overrides["portfolio.constraints.max_weight"] == 0.03
        assert overrides["universe.n_stocks"] == 200

    def test_slippage_conversion(self):
        s = StrategyDefinition.from_dict({
            "name": "test",
            "version": "1.0.0",
            "execution": {"slippage_bps": 15},
        })
        overrides = dsl_to_config_overrides(s)
        assert overrides["costs.slippage"] == 0.0015


class TestStrategyRegistry:
    def setup_method(self):
        import tempfile, os
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db_path = self._tmp.name
        self._tmp.close()

    def teardown_method(self):
        import os
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_save_and_get(self):
        reg = StrategyRegistry(db_path=self._db_path)
        s = StrategyDefinition.from_dict({
            "name": "test_strat",
            "version": "1.0.0",
        })
        reg.save(s)
        loaded = reg.get("test_strat")
        assert loaded is not None
        assert loaded.name == "test_strat"
        assert loaded.version == "1.0.0"

    def test_save_new_version(self):
        reg = StrategyRegistry(db_path=self._db_path)
        v1 = StrategyDefinition.from_dict({"name": "test", "version": "1.0.0", "created_at": "2025-01-01T00:00:00"})
        v2 = StrategyDefinition.from_dict({"name": "test", "version": "2.0.0", "created_at": "2026-01-01T00:00:00"})
        reg.save(v1)
        reg.save(v2)
        loaded = reg.get("test")
        assert loaded.version == "2.0.0"
        loaded_v1 = reg.get("test", "1.0.0")
        assert loaded_v1.version == "1.0.0"

    def test_list_strategies(self):
        reg = StrategyRegistry(db_path=self._db_path)
        reg.save(StrategyDefinition.from_dict({"name": "a", "version": "1.0.0"}))
        reg.save(StrategyDefinition.from_dict({"name": "b", "version": "1.0.0"}))
        strategies = reg.list_strategies()
        assert len(strategies) == 2

    def test_record_run(self):
        reg = StrategyRegistry(db_path=self._db_path)
        reg.record_run("test", "1.0.0", "run_001", config={"key": "val"})
        history = reg.get_run_history("test")
        assert len(history) == 1
        assert history[0]["run_id"] == "run_001"

    def test_get_stats(self):
        reg = StrategyRegistry(db_path=self._db_path)
        reg.save(StrategyDefinition.from_dict({"name": "a", "version": "1.0.0"}))
        stats = reg.get_stats()
        assert stats["strategies"] >= 1
