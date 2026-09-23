"""Configuration loading and access.

Loads YAML config and returns a typed Config object via schema.py dataclasses.
Supports environment variable overrides for sensitive or deployment-specific values.
"""

from __future__ import annotations

import dataclasses
import os
import typing
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

from quant_platform.config.schema import Config, FactorsConfig


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


def _coerce_value(field_type: Any, value: Any, path: str) -> Any:
    """Coerce one raw value to its declared type.

    Recurses into nested dataclasses and into ``list``/``tuple`` fields whose
    element type is a dataclass.
    """
    if dataclasses.is_dataclass(field_type):
        return _coerce(field_type, value, path)

    origin = typing.get_origin(field_type)
    if origin in (list, tuple):
        args = [a for a in typing.get_args(field_type) if a is not Ellipsis]
        element = args[0] if args else None
        if element is not None and dataclasses.is_dataclass(element):
            items = [
                _coerce(element, item, f"{path}[{i}]")
                for i, item in enumerate(value or [])
            ]
        else:
            items = list(value or [])
        return tuple(items) if origin is tuple else items

    return value


def _coerce(cls: type, raw: Any, path: str = "") -> Any:
    """Build dataclass ``cls`` from raw YAML, recursing into nested dataclasses.

    Driven by the schema's own field declarations (``dataclasses.fields`` +
    ``typing.get_type_hints``) rather than a hand-written list of sections, so a
    nested section added to ``config/schema.py`` is wired up automatically.
    A hand-written list is how ``data.synthetic`` came back as a raw dict while
    ``screener``/``execution``/``instruments`` were dropped entirely.

    Unknown keys are passed through to the constructor so they raise the same
    "unexpected keyword argument" TypeError as before -- a typo'd key must fail
    loudly rather than be silently ignored.
    """
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(
            f"config section '{path or cls.__name__}' must be a mapping, "
            f"got {type(raw).__name__}"
        )

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in hints:
            kwargs[key] = _coerce_value(hints[key], value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw dict into Config dataclass with validation.

    Every top-level section is built through :func:`_coerce`, which walks
    ``Config``'s declared field types. Adding a nested dataclass to
    ``config/schema.py`` therefore wires itself in; there is no list of sections
    here to forget to update.

    Non-mutating on purpose: this used to be ``portfolio_raw.pop(...)``, which
    removed the keys from the caller's dict. `main.py run` passes the same
    dict to `load_config()` and then to `VersionManager.save()` a few lines
    later, so every auto-saved config snapshot was written *without*
    `portfolio.constraints`, `portfolio.covariance` and `risk.var` -- and
    `config rollback` then silently reverted those sections to defaults.
    """
    hints = typing.get_type_hints(Config)
    sections: dict[str, Any] = {}

    for f in dataclasses.fields(Config):
        if f.name == "factors":
            # Not a plain mirror of the YAML: the `technical`/`fundamental`
            # sub-mappings are flattened to enabled-name tuples.
            sections["factors"] = _parse_factors(raw.get("factors"))
        elif f.name in raw:
            sections[f.name] = _coerce_value(hints[f.name], raw[f.name], f.name)
        # else: absent from YAML -> the dataclass default_factory applies.

    return Config(**sections)


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
