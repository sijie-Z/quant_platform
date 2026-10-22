"""Configuration schema validation using dataclasses."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UniverseConfig:
    n_stocks: int = 500
    exclude_st: bool = True
    exclude_suspended: bool = True


@dataclass
class SyntheticConfig:
    embedded_alpha: bool = False
    alpha_strength: float = 0.03


@dataclass
class DataConfig:
    provider: str = "baostock"
    validated: bool = False  # synthetic, baostock, tushare, postgres
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "daily"
    # A configured *real* provider that is unavailable must fail closed by
    # default: silently substituting synthetic research data made a broken
    # feed indistinguishable from production output. Set this (or pass
    # `--allow-synthetic-fallback`) only for development.
    allow_synthetic_fallback: bool = False
    # Freshness gate (D32): a panel whose completion marker is more than this
    # many *exchange sessions* behind the requested end is stale. Research
    # runs warn; live/decisive runs fail closed. 3 sessions tolerates a
    # weekend plus a T+1 publication lag.
    max_staleness_days: int = 3


@dataclass
class FactorProcessingConfig:
    winsorize_enabled: bool = True
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    standardize_enabled: bool = True
    standardize_method: str = "zscore"
    neutralize_enabled: bool = True
    neutralize_by: tuple[str, ...] = ("sector", "log_market_cap")


@dataclass
class AlphaConfig:
    method: str = "equal_weight"
    lookback: int = 252
    min_icir: float = 0.0
    tradability_gate: bool = False
    min_tradability: float = 0.3


@dataclass
class PortfolioConstraintsConfig:
    long_only: bool = True
    max_weight: float = 0.05
    max_sector_exposure: float = 0.30
    max_turnover: float = 0.30
    lot_size: int = 100


@dataclass
class CovarianceConfig:
    method: str = "ledoit_wolf"
    lookback: int = 252
    ewma_half_life: int = 63


@dataclass
class PortfolioConfig:
    optimizer: str = "mean_variance"
    constraints: PortfolioConstraintsConfig = field(default_factory=PortfolioConstraintsConfig)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    risk_aversion: float = 1.0


@dataclass
class BacktestConfig:
    rebalance_frequency: str = "monthly"
    initial_capital: float = 10_000_000
    benchmark: str = "equal_weight"


@dataclass
class CostsConfig:
    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.001
    slippage_model: str = "impact"  # fixed, proportional, impact
    impact_model: str = "composite"  # composite, almgren_chriss, square_root


@dataclass
class VarConfig:
    confidence: float = 0.95
    horizon: int = 1
    method: str = "historical"


@dataclass
class RiskConfig:
    var: VarConfig = field(default_factory=VarConfig)
    stress_scenarios: list[str] = field(default_factory=lambda: [
        "2008_financial_crisis", "2015_ashare_crash", "2020_covid_crash"
    ])


@dataclass
class OutputConfig:
    results_dir: str = "./results"
    save_plots: bool = True
    plot_format: str = "png"


@dataclass
class HoldoutConfig:
    """Final-evaluation holdout window (ISO dates; empty = disabled).

    The window is inclusive on both ends. Both dates must be set together and
    ``start_date < end_date``. Validation lives in ``validate_config()``
    (``utils/config.py``), not here: the config system's documented contract
    is that ``load_config(validate=False)`` skips validation for synthetic /
    partial configs (see ``utils/config.py``'s module docstring), and a
    dataclass constructor cannot honour that flag. The constructor only
    normalises ``None``/whitespace so empty means disabled.
    """
    start_date: str = ""
    end_date: str = ""

    def __post_init__(self) -> None:
        self.start_date = "" if self.start_date is None else str(self.start_date).strip()
        self.end_date = "" if self.end_date is None else str(self.end_date).strip()


@dataclass
class ResearchConfig:
    """Research governance settings (the holdout window)."""
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)


@dataclass
class InstrumentConfig:
    """Single instrument definition for cross-asset support."""
    symbol: str = ""
    asset_type: str = "stock"       # stock, etf, future, option, index
    exchange: str = "SSE"
    multiplier: float = 1.0
    tick_size: float = 0.01
    lot_size: int = 100
    margin_rate: float = 1.0
    commission_rate: float | None = None
    commission_per_lot: float = 0.0
    stamp_tax_rate: float | None = None
    t_plus: int = 1
    price_limit: float = 0.10
    underlying: str = ""
    expiry: str = ""


@dataclass
class QMTConfig:
    """QMT/miniQMT broker configuration."""
    account: str = ""               # sim/live account ID
    password: str = ""              # Override env QMT_PASSWORD (prefer env)
    server: str = "localhost:58610" # miniQMT TCP address
    mode: str = "sim"               # sim or live
    data_server: str = ""           # Market data server (defaults to server)
    connect_timeout: float = 5.0    # Connection timeout seconds


@dataclass
class ScreenRuleConfig:
    """Single screening condition."""
    factor: str = ""
    operator: str = "gt"    # gt, gte, lt, lte, eq, ne, between
    value: float | list[float] = 0.0


@dataclass
class ScreenerConfig:
    """Boolean factor screening configuration."""
    enabled: bool = False
    rules: list[ScreenRuleConfig] = field(default_factory=list)
    logic: str = "and"       # and | or
    min_stocks: int = 5
    max_stocks: int = 200


@dataclass
class ExecutionConfig:
    """Execution layer configuration (broker, QMT, etc.)."""
    broker_type: str = "simulated"  # simulated, paper, qmt
    qmt: QMTConfig = field(default_factory=QMTConfig)


@dataclass
class FactorsConfig:
    """Factor configuration: which factors are enabled."""
    enabled_technicals: tuple[str, ...] = (
        "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
        "volatility_20d", "volatility_60d",
        "turnover_20d", "rsi_14d", "amplitude_20d", "macd",
        "efficiency_ratio", "breakout_ignition", "trend_stage", "ma_convergence", "breakout_proximity", "pure_volatility", "mtf_resonance",
        "kmid", "klen", "kup", "klow", "ksft",
    )
    enabled_fundamentals: tuple[str, ...] = (
        "log_market_cap", "pb_ratio", "pe_ratio", "roe", "asset_growth",
    )
    # Per-factor constructor parameters (`momentum_1m: {params: {period: 10}}`).
    # They used to be dropped during parsing, so editing them changed nothing.
    params: dict[str, dict] = field(default_factory=dict)
    # `factors.processing`: winsorize / standardize / neutralize settings.
    processing: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorsConfig = field(default_factory=FactorsConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    instruments: list[InstrumentConfig] = field(default_factory=list)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
