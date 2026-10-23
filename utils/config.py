"""Configuration loading and access.

Loads YAML config and returns a typed Config object via schema.py dataclasses.

Environment overrides (applied after the YAML is parsed, on every load):

- ``QMT_PASSWORD``            overrides ``execution.qmt.password`` (env wins;
                              the YAML value is a fallback for development).
- ``QUANT_DATA_PROVIDER``     overrides ``data.provider`` (deployment-specific).
- ``QUANT_LOG_LEVEL`` /
  ``LOG_LEVEL``               consumed by ``utils.logging.setup_logging``.

These are *restart-only* semantics: a running engine never re-reads the YAML
or the environment. Restart the process to pick up a change.

Validation is performed at load time (unless ``validate=False`` is passed by
callers that build synthetic/partial configs, e.g. schema-coercion tests).
Invalid ranges, enums, dates and provider names raise
:class:`ConfigValidationError` with an actionable message instead of failing
much later in the pipeline.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from datetime import date as _date
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
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigValidationError(ValueError):
    """Raised when a configuration value is invalid (actionable message)."""


# ── Allowed enums ──────────────────────────────────────────────────────────
_PROVIDERS = {"synthetic", "baostock", "tushare", "postgres"}
_DATA_FREQUENCIES = {"daily", "weekly", "monthly"}
_ALPHA_METHODS = {"equal_weight", "ic_weighted", "icir_weighted"}
_OPTIMIZERS = {"equal_weight", "mean_variance", "risk_parity"}
_COVARIANCE_METHODS = {"sample", "ledoit_wolf", "ewma"}
_REBALANCE_FREQUENCIES = {"daily", "weekly", "monthly"}
_BENCHMARKS = {"equal_weight", "market_cap_weight"}
_SLIPPAGE_MODELS = {"fixed", "proportional", "impact"}
_IMPACT_MODELS = {"composite", "almgren_chriss", "square_root"}
_VAR_METHODS = {"historical", "parametric", "monte_carlo"}
_PLOT_FORMATS = {"png", "pdf", "svg"}
_SCREENER_LOGIC = {"and", "or"}
_SCREENER_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "ne", "between"}
_ASSET_TYPES = {"stock", "etf", "future", "option", "index"}
_BROKER_TYPES = {"simulated", "paper", "qmt"}
_QMT_MODES = {"sim", "live"}
_STANDARDIZE_METHODS = {"zscore", "rank"}
_NEUTRALIZE_BY = {"sector", "log_market_cap", "market_cap"}
_FACTOR_SECTIONS = {"technical", "fundamental", "processing"}
_FACTOR_ENTRY_KEYS = {"enabled", "params"}


def _parse_factors(raw_factors: dict | None) -> FactorsConfig:
    """Parse factors config from YAML.

    A section that is absent falls back to the schema defaults; a section
    that is present but empty (or with every factor disabled) yields no
    factors. The old `tuple(enabled) or default` made an explicitly empty
    section silently enable the full default list.

    Unknown keys inside a factor entry are rejected instead of being silently
    dropped: ``momentum_12m: {enabled: true, exclude_recent: 21}`` used to
    look configured while the factor ran with its defaults. Non-constructor
    knobs belong under ``params``.
    """
    if raw_factors is None:
        return FactorsConfig()
    if not isinstance(raw_factors, dict):
        raise ConfigValidationError(
            "config section 'factors' must be a mapping, "
            f"got {type(raw_factors).__name__}",
        )

    unknown_sections = set(raw_factors) - _FACTOR_SECTIONS
    if unknown_sections:
        raise ConfigValidationError(
            "config.factors has unknown sections "
            f"{sorted(unknown_sections)}; allowed: {sorted(_FACTOR_SECTIONS)}",
        )

    def _enabled_names(section_key: str) -> tuple[str, ...] | None:
        if section_key not in raw_factors:
            return None
        section = raw_factors.get(section_key) or {}
        if not isinstance(section, dict):
            raise ConfigValidationError(
                f"config.factors.{section_key} must be a mapping",
            )
        return tuple(
            name for name, cfg in section.items()
            if isinstance(cfg, dict) and cfg.get("enabled", True)
        )

    params: dict[str, dict] = {}
    for section_key in ("technical", "fundamental"):
        section = raw_factors.get(section_key) or {}
        if not isinstance(section, dict):
            continue
        for name, cfg in section.items():
            if cfg is None:
                cfg = {}
            if not isinstance(cfg, dict):
                raise ConfigValidationError(
                    f"config.factors.{section_key}.{name} must be a mapping "
                    f"(got {type(cfg).__name__})",
                )
            unknown_keys = set(cfg) - _FACTOR_ENTRY_KEYS
            if unknown_keys:
                raise ConfigValidationError(
                    f"config.factors.{section_key}.{name} has unknown keys "
                    f"{sorted(unknown_keys)}; allowed: {sorted(_FACTOR_ENTRY_KEYS)}. "
                    "Move factor constructor knobs under 'params' "
                    f"(e.g. {sorted(unknown_keys)[0]}: ... -> "
                    "params: {" + f"{sorted(unknown_keys)[0]}: ..." + "}).",
                )
            if cfg.get("params"):
                if not isinstance(cfg["params"], dict):
                    raise ConfigValidationError(
                        f"config.factors.{section_key}.{name}.params must be "
                        "a mapping",
                    )
                params[name] = dict(cfg["params"])

    processing = raw_factors.get("processing") or {}
    if not isinstance(processing, dict):
        raise ConfigValidationError("config.factors.processing must be a mapping")

    enabled_technicals = _enabled_names("technical")
    enabled_fundamentals = _enabled_names("fundamental")

    return FactorsConfig(
        enabled_technicals=(
            enabled_technicals if enabled_technicals is not None
            else FactorsConfig.enabled_technicals
        ),
        enabled_fundamentals=(
            enabled_fundamentals if enabled_fundamentals is not None
            else FactorsConfig.enabled_fundamentals
        ),
        params=params,
        processing=dict(processing),
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


# ── Validation ────────────────────────────────────────────────────────────

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def _check_date(value: Any, path: str) -> _date:
    try:
        parsed = _date.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise ConfigValidationError(
            f"{path} must be an ISO date (YYYY-MM-DD), got {value!r}",
        ) from None
    _require(
        1990 <= parsed.year <= 2100,
        f"{path} must be between 1990-01-01 and 2100-12-31 (got {parsed})",
    )
    return parsed


def _validate_factors(factors: FactorsConfig) -> None:
    processing = factors.processing or {}
    unknown = set(processing) - {"winsorize", "standardize", "neutralize"}
    _require(
        not unknown,
        f"config.factors.processing has unknown keys {sorted(unknown)}; "
        "allowed: ['winsorize', 'standardize', 'neutralize']",
    )

    wins = processing.get("winsorize") or {}
    if wins:
        _require(isinstance(wins, dict), "config.factors.processing.winsorize must be a mapping")
        lower = wins.get("lower", 0.01)
        upper = wins.get("upper", 0.99)
        _require(
            0.0 <= float(lower) < float(upper) <= 1.0,
            "config.factors.processing.winsorize lower/upper must satisfy "
            f"0 <= lower < upper <= 1, got {lower}/{upper}",
        )

    std = processing.get("standardize") or {}
    if std:
        _require(isinstance(std, dict), "config.factors.processing.standardize must be a mapping")
        method = std.get("method", "zscore")
        _require(
            method in _STANDARDIZE_METHODS,
            f"config.factors.processing.standardize.method must be one of "
            f"{sorted(_STANDARDIZE_METHODS)}, got {method!r}",
        )

    neutral = processing.get("neutralize") or {}
    if neutral:
        _require(isinstance(neutral, dict), "config.factors.processing.neutralize must be a mapping")
        by = neutral.get("by", ["sector", "log_market_cap"])
        _require(
            isinstance(by, (list, tuple)),
            "config.factors.processing.neutralize.by must be a list",
        )
        bad = set(by) - _NEUTRALIZE_BY
        _require(
            not bad,
            f"config.factors.processing.neutralize.by has unknown entries "
            f"{sorted(bad)}; allowed: {sorted(_NEUTRALIZE_BY)}",
        )

    for name, params in (factors.params or {}).items():
        _require(
            isinstance(params, dict),
            f"config.factors.params.{name} must be a mapping",
        )


def validate_config(config: Config) -> Config:
    """Validate a parsed Config; raises ConfigValidationError on first error."""
    # Universe
    n_stocks = config.universe.n_stocks
    _require(
        isinstance(n_stocks, int) and not isinstance(n_stocks, bool) and n_stocks > 0,
        f"config.universe.n_stocks must be > 0, got {n_stocks!r}",
    )

    # Data
    data = config.data
    _require(
        data.provider in _PROVIDERS,
        f"config.data.provider must be one of {sorted(_PROVIDERS)}, "
        f"got {data.provider!r}",
    )
    _require(
        data.frequency in _DATA_FREQUENCIES,
        f"config.data.frequency must be one of {sorted(_DATA_FREQUENCIES)}, "
        f"got {data.frequency!r}",
    )
    start = _check_date(data.start_date, "config.data.start_date")
    end = _check_date(data.end_date, "config.data.end_date")
    _require(
        start <= end,
        f"config.data.start_date ({data.start_date}) must be <= "
        f"config.data.end_date ({data.end_date})",
    )
    # Warn when the sample extends past the published exchange-holiday table:
    # weekend handling is always right, but weekday closures after the last
    # published year would be treated as trading days.
    try:
        from quant_platform.utils.exchange_calendar import EXCHANGE_HOLIDAYS

        covered_year = max(int(day[:4]) for day in EXCHANGE_HOLIDAYS)
        if end.year > covered_year:
            logger.warning(
                "config.data.end_date %s is beyond the published exchange "
                "holiday table (through %d); weekday holidays in %d are "
                "treated as trading days. Extend EXCHANGE_HOLIDAYS.",
                data.end_date, covered_year, end.year,
            )
    except (ValueError, TypeError):  # table format guard
        pass
    strength = data.synthetic.alpha_strength
    _require(
        float(strength) >= 0,
        f"config.data.synthetic.alpha_strength must be >= 0, got {strength!r}",
    )

    # Factors
    _validate_factors(config.factors)

    # Alpha
    alpha = config.alpha
    _require(
        alpha.method in _ALPHA_METHODS,
        f"config.alpha.method must be one of {sorted(_ALPHA_METHODS)}, "
        f"got {alpha.method!r}",
    )
    _require(
        int(alpha.lookback) > 0,
        f"config.alpha.lookback must be > 0, got {alpha.lookback!r}",
    )
    _require(
        0.0 <= float(alpha.min_tradability) <= 1.0,
        "config.alpha.min_tradability must be in [0, 1], "
        f"got {alpha.min_tradability!r}",
    )

    # Portfolio
    portfolio = config.portfolio
    _require(
        portfolio.optimizer in _OPTIMIZERS,
        f"config.portfolio.optimizer must be one of {sorted(_OPTIMIZERS)}, "
        f"got {portfolio.optimizer!r}",
    )
    c = portfolio.constraints
    _require(
        0.0 < float(c.max_weight) <= 1.0,
        f"config.portfolio.constraints.max_weight must be in (0, 1], got {c.max_weight!r}",
    )
    _require(
        float(c.max_sector_exposure) >= 0.0,
        "config.portfolio.constraints.max_sector_exposure must be >= 0, "
        f"got {c.max_sector_exposure!r}",
    )
    _require(
        float(c.max_turnover) >= 0.0,
        "config.portfolio.constraints.max_turnover must be >= 0, "
        f"got {c.max_turnover!r}",
    )
    _require(
        isinstance(c.lot_size, int) and not isinstance(c.lot_size, bool)
        and c.lot_size >= 1,
        f"config.portfolio.constraints.lot_size must be >= 1, got {c.lot_size!r}",
    )
    cov = portfolio.covariance
    _require(
        cov.method in _COVARIANCE_METHODS,
        f"config.portfolio.covariance.method must be one of "
        f"{sorted(_COVARIANCE_METHODS)}, got {cov.method!r}",
    )
    _require(
        int(cov.lookback) > 0,
        f"config.portfolio.covariance.lookback must be > 0, got {cov.lookback!r}",
    )
    _require(
        int(cov.ewma_half_life) > 0,
        "config.portfolio.covariance.ewma_half_life must be > 0, "
        f"got {cov.ewma_half_life!r}",
    )
    _require(
        float(portfolio.risk_aversion) > 0,
        f"config.portfolio.risk_aversion must be > 0, got {portfolio.risk_aversion!r}",
    )

    # Backtest
    bt = config.backtest
    _require(
        bt.rebalance_frequency in _REBALANCE_FREQUENCIES,
        f"config.backtest.rebalance_frequency must be one of "
        f"{sorted(_REBALANCE_FREQUENCIES)}, got {bt.rebalance_frequency!r}",
    )
    _require(
        float(bt.initial_capital) > 0,
        f"config.backtest.initial_capital must be > 0, got {bt.initial_capital!r}",
    )
    _require(
        bt.benchmark in _BENCHMARKS,
        f"config.backtest.benchmark must be one of {sorted(_BENCHMARKS)}, "
        f"got {bt.benchmark!r}",
    )

    # Costs
    costs = config.costs
    _require(
        0.0 <= float(costs.commission) <= 0.01,
        "config.costs.commission must be in [0, 0.01] (0-1%), "
        f"got {costs.commission!r}",
    )
    _require(
        costs.stamp_tax is None or 0.0 <= float(costs.stamp_tax) <= 0.01,
        "config.costs.stamp_tax must be null (follow the statutory schedule: "
        "0.1% before 2023-08-28, 0.05% from that date) or in [0, 0.01] "
        f"(0-1%), got {costs.stamp_tax!r}",
    )
    _require(
        0.0 <= float(costs.slippage) <= 0.1,
        "config.costs.slippage must be in [0, 0.1], got "
        f"{costs.slippage!r}",
    )
    _require(
        costs.slippage_model in _SLIPPAGE_MODELS,
        f"config.costs.slippage_model must be one of {sorted(_SLIPPAGE_MODELS)}, "
        f"got {costs.slippage_model!r}",
    )
    _require(
        costs.impact_model in _IMPACT_MODELS,
        f"config.costs.impact_model must be one of {sorted(_IMPACT_MODELS)}, "
        f"got {costs.impact_model!r}",
    )

    # Risk
    var = config.risk.var
    _require(
        0.0 < float(var.confidence) < 1.0,
        f"config.risk.var.confidence must be in (0, 1), got {var.confidence!r}",
    )
    _require(
        int(var.horizon) >= 1,
        f"config.risk.var.horizon must be >= 1, got {var.horizon!r}",
    )
    _require(
        var.method in _VAR_METHODS,
        f"config.risk.var.method must be one of {sorted(_VAR_METHODS)}, "
        f"got {var.method!r}",
    )

    # Output
    out = config.output
    _require(
        bool(str(out.results_dir).strip()),
        "config.output.results_dir must not be empty",
    )
    _require(
        out.plot_format in _PLOT_FORMATS,
        f"config.output.plot_format must be one of {sorted(_PLOT_FORMATS)}, "
        f"got {out.plot_format!r}",
    )

    # Instruments
    for i, inst in enumerate(config.instruments or []):
        prefix = f"config.instruments[{i}]"
        _require(
            inst.asset_type in _ASSET_TYPES,
            f"{prefix}.asset_type must be one of {sorted(_ASSET_TYPES)}, "
            f"got {inst.asset_type!r}",
        )
        _require(
            float(inst.multiplier) > 0,
            f"{prefix}.multiplier must be > 0, got {inst.multiplier!r}",
        )
        _require(
            float(inst.tick_size) > 0,
            f"{prefix}.tick_size must be > 0, got {inst.tick_size!r}",
        )
        _require(
            int(inst.lot_size) >= 1,
            f"{prefix}.lot_size must be >= 1, got {inst.lot_size!r}",
        )
        _require(
            0.0 < float(inst.margin_rate) <= 1.0,
            f"{prefix}.margin_rate must be in (0, 1], got {inst.margin_rate!r}",
        )

    # Screener
    screener = config.screener
    _require(
        screener.logic in _SCREENER_LOGIC,
        f"config.screener.logic must be one of {sorted(_SCREENER_LOGIC)}, "
        f"got {screener.logic!r}",
    )
    _require(
        int(screener.min_stocks) >= 0,
        f"config.screener.min_stocks must be >= 0, got {screener.min_stocks!r}",
    )
    _require(
        int(screener.max_stocks) >= 0,
        f"config.screener.max_stocks must be >= 0, got {screener.max_stocks!r}",
    )
    if int(screener.max_stocks) > 0:
        _require(
            int(screener.max_stocks) >= int(screener.min_stocks),
            "config.screener.max_stocks must be >= min_stocks "
            f"({screener.max_stocks} < {screener.min_stocks})",
        )
    for i, rule in enumerate(screener.rules or []):
        _require(
            rule.operator in _SCREENER_OPERATORS,
            f"config.screener.rules[{i}].operator must be one of "
            f"{sorted(_SCREENER_OPERATORS)}, got {rule.operator!r}",
        )
        _require(
            bool(str(rule.factor).strip()),
            f"config.screener.rules[{i}].factor must not be empty",
        )

    # Execution
    execution = config.execution
    _require(
        execution.broker_type in _BROKER_TYPES,
        f"config.execution.broker_type must be one of {sorted(_BROKER_TYPES)}, "
        f"got {execution.broker_type!r}",
    )
    _require(
        execution.qmt.mode in _QMT_MODES,
        f"config.execution.qmt.mode must be one of {sorted(_QMT_MODES)}, "
        f"got {execution.qmt.mode!r}",
    )
    _require(
        float(execution.qmt.connect_timeout) > 0,
        "config.execution.qmt.connect_timeout must be > 0, "
        f"got {execution.qmt.connect_timeout!r}",
    )

    # Research holdout window (optional final-evaluation reservation).
    # R3: validated here rather than in HoldoutConfig.__post_init__ because
    # this module's contract is that ``validate=False`` skips validation for
    # synthetic/partial configs; a dataclass constructor cannot honour that
    # flag. HoldoutSpec.from_settings() re-validates at every consumption
    # point (sweep guards, research validation), so runtime stays fail-loud.
    from quant_platform.research.holdout import HoldoutSpec

    try:
        HoldoutSpec.from_settings(config)
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc

    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply documented environment overrides (env-first for secrets).

    Restart-only: a running process never re-reads the environment.
    """
    qmt_password = os.environ.get("QMT_PASSWORD", "")
    if qmt_password:
        config.execution.qmt.password = qmt_password
    provider = os.environ.get("QUANT_DATA_PROVIDER", "")
    if provider:
        config.data.provider = provider
    return config


def load_config(
    config_path: str | Path | None = None,
    *,
    raw: dict[str, Any] | None = None,
    validate: bool = True,
) -> Config:
    """Load configuration from YAML file or pre-parsed dict.

    Priority: 1) raw dict, 2) provided config_path, 3) QUANT_CONFIG env var,
    4) default config at quant_platform/config/default.yaml

    Args:
        config_path: Path to YAML config file.
        raw: Pre-loaded raw dict (takes precedence over config_path).
        validate: Run range/enum/date/provider validation (default True).
            Pass False only for tests that deliberately build synthetic
            mappings to exercise schema coercion.

    Returns:
        Config dataclass with validated fields.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigValidationError: If a value is outside its allowed range/enum
            or a factor entry has unknown keys.
    """
    if raw is not None:
        config = _parse_config(raw)
    else:
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

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Config file {config_path} must contain a YAML mapping at "
                f"the top level, got {type(raw).__name__}",
            )
        config = _parse_config(raw)

    config = _apply_env_overrides(config)
    if validate:
        validate_config(config)
    return config
