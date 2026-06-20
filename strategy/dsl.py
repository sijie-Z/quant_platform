"""Strategy DSL — versioned, reproducible strategy definitions.

Strategy DSL separates *what the strategy is* from *how it runs*.
A strategy is a YAML file with name, version, factors, portfolio config, etc.
This enables versioning, audit trail, and comparison across strategy versions.

Design inspired by quawn's Strategy DSL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from quant_platform.factors.registry import get_registry
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Known keywords in DSL that would attempt to override no-lookahead
FORBIDDEN_FIELDS = {
    "allow_lookahead", "lookahead_override", "skip_no_lookahead",
    "use_future_data", "future_data", "no_lookahead_bypass",
    "leak_future", "signal_date_override",
}


@dataclass(frozen=True)
class StrategyDefinition:
    """Normalized strategy definition loaded from YAML/JSON."""

    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)

    # Strategy components
    universe: dict[str, Any] = field(default_factory=dict)
    factors: list[dict[str, Any]] = field(default_factory=list)
    regime: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at,
            "tags": list(self.tags),
            "universe": dict(self.universe),
            "factors": [dict(f) for f in self.factors],
            "regime": dict(self.regime),
            "portfolio": dict(self.portfolio),
            "risk": dict(self.risk),
            "execution": dict(self.execution),
            "validation": dict(self.validation),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategyDefinition:
        return cls(
            name=str(data.get("name") or "").strip(),
            description=str(data.get("description") or "").strip(),
            version=str(data.get("version") or "1.0.0").strip(),
            author=str(data.get("author") or "").strip(),
            created_at=str(data.get("created_at") or datetime.now().isoformat()),
            tags=list(data.get("tags") or []),
            universe=dict(data.get("universe") or {}),
            factors=list(data.get("factors") or []),
            regime=dict(data.get("regime") or {}),
            portfolio=dict(data.get("portfolio") or {}),
            risk=dict(data.get("risk") or {}),
            execution=dict(data.get("execution") or {}),
            validation=dict(data.get("validation") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    @property
    def factor_weights(self) -> dict[str, float]:
        """Get factor name -> weight mapping, normalized to sum=1."""
        weights: dict[str, float] = {}
        for item in self.factors:
            name = str(item.get("name") or "").strip().lower()
            if not name:
                continue
            weights[name] = float(item.get("weight", 1.0))

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    @property
    def symbols(self) -> list[str]:
        return [str(s).upper().strip() for s in
                list(self.universe.get("symbols") or []) if str(s).strip()]

    @property
    def portfolio_method(self) -> str:
        return str(self.portfolio.get("method") or "equal_weight").strip().lower()

    @property
    def rebalance_frequency(self) -> str:
        return str(self.portfolio.get("rebalance_frequency") or "monthly").strip().lower()

    @property
    def max_position_weight(self) -> float:
        val = self.portfolio.get("max_position_weight")
        if val is not None:
            return float(val)
        return 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategyDefinition:
        """Load strategy definition from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid strategy file: {path}")
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> StrategyDefinition:
        """Load strategy definition from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid strategy file: {path}")
        return cls.from_dict(data)


# ── Validator ──


@dataclass
class ValidationResult:
    """Strategy DSL validation result."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


_KNOWN_PORTFOLIO_METHODS = {"equal_weight", "mean_variance", "risk_parity"}
_KNOWN_REBALANCE_FREQUENCIES = {"daily", "weekly", "monthly"}
_KNOWN_UNIVERSE_TYPES = {"default", "custom", "sector", "etf", "large_cap"}


def validate_strategy(
    strategy: StrategyDefinition,
    check_factors_exist: bool = True,
) -> ValidationResult:
    """Validate a strategy definition.

    Checks:
    - Required fields (name, version)
    - No forbidden fields (no-lookahead overrides)
    - Factor names exist in registry
    - Portfolio method is known
    - Rebalance frequency is known
    - Weights are valid numbers
    """
    result = ValidationResult()

    # --- Required fields ---
    if not strategy.name:
        result.errors.append("strategy name is required")
        result.valid = False

    if not strategy.version:
        result.errors.append("strategy version is required")
        result.valid = False

    # --- Forbidden fields ---
    raw = strategy.to_dict()
    for key in FORBIDDEN_FIELDS:
        if key in raw or any(key in str(v) for v in raw.values() if isinstance(v, str)):
            result.errors.append(f"forbidden field '{key}' — would violate no-lookahead contract")
            result.valid = False

    # --- Universe ---
    universe_type = strategy.universe.get("type", "default")
    if universe_type not in _KNOWN_UNIVERSE_TYPES:
        result.warnings.append(f"unknown universe type '{universe_type}'")

    # --- Factors ---
    if not strategy.factors:
        result.warnings.append("no factors defined — strategy has no signal")

    # Registry check (optional, requires factors to be registered)
    registered_factors: set[str] = set()
    if check_factors_exist:
        try:
            registry = get_registry()
            registered_factors = set(f.lower() for f in registry.list_all())
        except Exception:
            result.warnings.append("could not load factor registry for validation")

    # Name and weight validation always runs
    for item in strategy.factors:
        name = str(item.get("name") or "").strip().lower()
        if not name:
            result.errors.append("factor entry missing 'name' field")
            result.valid = False
            continue
        if check_factors_exist and name not in registered_factors:
            result.warnings.append(f"factor '{name}' not found in registry")

        weight = item.get("weight", 1.0)
        try:
            w = float(weight)
            if not _isfinite(w):
                result.errors.append(f"invalid weight for factor '{name}': {weight}")
                result.valid = False
        except (TypeError, ValueError):
            result.errors.append(f"non-numeric weight for factor '{name}': {weight}")
            result.valid = False

    # --- Portfolio ---
    method = strategy.portfolio_method
    if method and method not in _KNOWN_PORTFOLIO_METHODS:
        result.warnings.append(f"unknown portfolio method '{method}'")

    freq = strategy.rebalance_frequency
    if freq and freq not in _KNOWN_REBALANCE_FREQUENCIES:
        result.warnings.append(f"unknown rebalance frequency '{freq}'")

    max_pos = strategy.max_position_weight
    if max_pos <= 0 or max_pos > 1:
        result.errors.append(f"max_position_weight out of range (0, 1]: {max_pos}")
        result.valid = False

    # --- Risk ---
    max_dd = strategy.risk.get("max_drawdown")
    if max_dd is not None:
        try:
            fdd = float(max_dd)
            if fdd <= 0 or fdd > 1:
                result.warnings.append(f"max_drawdown outside typical range (0, 1]: {fdd}")
        except (TypeError, ValueError):
            result.warnings.append(f"non-numeric max_drawdown: {max_dd}")

    # --- Validation ---
    min_ic = strategy.validation.get("minimum_ic")
    if min_ic is not None:
        try:
            if not _isfinite(float(min_ic)):
                result.errors.append("minimum_ic must be a finite number")
                result.valid = False
        except (TypeError, ValueError):
            result.errors.append("minimum_ic must be a number")

    return result


def _isfinite(v: float) -> bool:
    import math
    return math.isfinite(v)


# ── DSL → Config mapping ──


def dsl_to_config_overrides(strategy: StrategyDefinition) -> dict[str, Any]:
    """Convert a Strategy DSL definition into config overrides.

    These can be merged into the system config for pipeline execution.
    """
    overrides: dict[str, Any] = {}

    # Universe
    u = strategy.universe
    if "n_stocks" in u:
        overrides["universe.n_stocks"] = int(u["n_stocks"])
    if "exclude_st" in u:
        overrides["universe.exclude_st"] = bool(u["exclude_st"])

    # Factors — set enabled factors and their weights
    factor_names = []
    factor_overrides = {}
    for item in strategy.factors:
        name = str(item.get("name") or "").strip().lower()
        if name:
            factor_names.append(name)
    if factor_names:
        overrides["factors.enabled_technicals"] = factor_names
        overrides["factors.enabled_fundamentals"] = []

    # Portfolio
    if strategy.portfolio_method:
        overrides["portfolio.optimizer"] = strategy.portfolio_method
    if strategy.rebalance_frequency:
        overrides["backtest.rebalance_frequency"] = strategy.rebalance_frequency
    if strategy.max_position_weight:
        overrides["portfolio.constraints.max_weight"] = strategy.max_position_weight

    # Execution
    e = strategy.execution
    if "slippage_bps" in e:
        overrides["costs.slippage"] = float(e["slippage_bps"]) / 10000

    return overrides
