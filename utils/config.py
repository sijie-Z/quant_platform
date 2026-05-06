"""Configuration loading and access.

Loads YAML config and returns a typed Config object via schema.py dataclasses.
Supports environment variable overrides for sensitive or deployment-specific values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from quant_platform.config.schema import (
    AlphaConfig,
    BacktestConfig,
    Config,
    CostsConfig,
    CovarianceConfig,
    DataConfig,
    OutputConfig,
    PortfolioConfig,
    PortfolioConstraintsConfig,
    RiskConfig,
    UniverseConfig,
    VarConfig,
)


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw dict into Config dataclass with validation."""
    universe = UniverseConfig(**raw.get("universe", {}))
    data = DataConfig(**raw.get("data", {}))

    alpha_raw = raw.get("alpha", {})
    alpha = AlphaConfig(**alpha_raw)

    portfolio_raw = raw.get("portfolio", {})
    constraints_raw = portfolio_raw.pop("constraints", {})
    cov_raw = portfolio_raw.pop("covariance", {})
    constraints = PortfolioConstraintsConfig(**constraints_raw)
    covariance = CovarianceConfig(**cov_raw)
    portfolio = PortfolioConfig(
        constraints=constraints,
        covariance=covariance,
        **portfolio_raw,
    )

    backtest = BacktestConfig(**raw.get("backtest", {}))
    costs = CostsConfig(**raw.get("costs", {}))

    risk_raw = raw.get("risk", {})
    var_raw = risk_raw.pop("var", {})
    var_config = VarConfig(**var_raw)
    risk = RiskConfig(var=var_config, **risk_raw)

    output = OutputConfig(**raw.get("output", {}))

    return Config(
        universe=universe,
        data=data,
        alpha=alpha,
        portfolio=portfolio,
        backtest=backtest,
        costs=costs,
        risk=risk,
        output=output,
    )


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Priority: 1) provided config_path, 2) QUANT_CONFIG env var,
    3) default config at quant_platform/config/default.yaml
    """
    if config_path is None:
        config_path = os.environ.get("QUANT_CONFIG", None)

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _parse_config(raw)
