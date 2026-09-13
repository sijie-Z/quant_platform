"""Configuration loading and access.

Loads YAML config and returns a typed Config object via schema.py dataclasses.
Supports environment variable overrides for sensitive or deployment-specific values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    # Auto-load .env from project root
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv is optional

from quant_platform.config.schema import (
    AlphaConfig,
    BacktestConfig,
    Config,
    CostsConfig,
    CovarianceConfig,
    DataConfig,
    FactorsConfig,
    OutputConfig,
    PortfolioConfig,
    PortfolioConstraintsConfig,
    RiskConfig,
    UniverseConfig,
    VarConfig,
)


def _parse_factors(raw_factors: dict | None) -> FactorsConfig:
    """Parse factors config from YAML, extracting enabled factor names."""
    if raw_factors is None:
        return FactorsConfig()

    enabled_technicals = []
    tech_config = raw_factors.get("technical", {})
    for name, cfg in tech_config.items():
        if isinstance(cfg, dict) and cfg.get("enabled", True):
            enabled_technicals.append(name)

    enabled_fundamentals = []
    fund_config = raw_factors.get("fundamental", {})
    for name, cfg in fund_config.items():
        if isinstance(cfg, dict) and cfg.get("enabled", True):
            enabled_fundamentals.append(name)

    return FactorsConfig(
        enabled_technicals=tuple(enabled_technicals) or FactorsConfig.enabled_technicals,
        enabled_fundamentals=tuple(enabled_fundamentals) or FactorsConfig.enabled_fundamentals,
    )


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw dict into Config dataclass with validation."""
    universe = UniverseConfig(**raw.get("universe", {}))
    data = DataConfig(**raw.get("data", {}))

    alpha_raw = raw.get("alpha", {})
    alpha = AlphaConfig(**alpha_raw)

    portfolio_raw = raw.get("portfolio", {})
    # Non-mutating on purpose: these used to be `portfolio_raw.pop(...)`, which
    # removed the keys from the caller's dict. `main.py run` passes the same
    # dict to `load_config()` and then to `VersionManager.save()` a few lines
    # later, so every auto-saved config snapshot was written *without*
    # `portfolio.constraints`, `portfolio.covariance` and `risk.var` -- and
    # `config rollback` then silently reverted those sections to defaults.
    constraints = PortfolioConstraintsConfig(**portfolio_raw.get("constraints", {}))
    covariance = CovarianceConfig(**portfolio_raw.get("covariance", {}))
    portfolio = PortfolioConfig(
        constraints=constraints,
        covariance=covariance,
        **{k: v for k, v in portfolio_raw.items()
           if k not in ("constraints", "covariance")},
    )

    backtest = BacktestConfig(**raw.get("backtest", {}))
    costs = CostsConfig(**raw.get("costs", {}))

    risk_raw = raw.get("risk", {})
    var_config = VarConfig(**risk_raw.get("var", {}))
    risk = RiskConfig(
        var=var_config,
        **{k: v for k, v in risk_raw.items() if k != "var"},
    )

    output = OutputConfig(**raw.get("output", {}))
    factors = _parse_factors(raw.get("factors"))

    return Config(
        universe=universe,
        data=data,
        factors=factors,
        alpha=alpha,
        portfolio=portfolio,
        backtest=backtest,
        costs=costs,
        risk=risk,
        output=output,
    )


def load_config(
    config_path: str | Path | None = None,
    *,
    raw: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML file or pre-parsed dict.

    Priority: 1) raw dict, 2) provided config_path, 3) QUANT_CONFIG env var,
    4) default config at quant_platform/config/default.yaml

    Args:
        config_path: Path to YAML config file.
        raw: Pre-loaded raw dict (takes precedence over config_path).

    Returns:
        Config dataclass with validated fields.
    """
    if raw is not None:
        return _parse_config(raw)

    if config_path is None:
        config_path = os.environ.get("QUANT_CONFIG", None)

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _parse_config(raw)
