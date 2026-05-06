"""Configuration schema validation using dataclasses."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UniverseConfig:
    n_stocks: int = 500
    exclude_st: bool = True
    exclude_suspended: bool = True


@dataclass
class DataConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "daily"


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
    method: str = "icir_weighted"
    lookback: int = 252
    min_icir: float = 0.0


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
    slippage_model: str = "fixed"


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
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
