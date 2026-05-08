"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(v: str) -> str:
    if not _DATE_RE.match(v):
        raise ValueError("Date must be YYYY-MM-DD format")
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date: {exc}") from exc
    return v


# --- Request schemas ---

class RunRequest(BaseModel):
    n_stocks: int = Field(default=500, ge=50, le=1000, description="Number of stocks")
    start_date: str = Field(default="2021-01-01", description="Start date YYYY-MM-DD")
    end_date: str = Field(default="2025-12-31", description="End date YYYY-MM-DD")
    optimizer: Literal["equal_weight", "mean_variance", "risk_parity"] = Field(
        default="mean_variance", description="Portfolio optimizer"
    )
    alpha_method: Literal["equal_weight", "ic_weighted", "icir_weighted"] = Field(
        default="icir_weighted", description="Alpha combination method"
    )
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = Field(
        default="monthly", description="Rebalance frequency"
    )
    covariance_method: Literal["sample", "ledoit_wolf", "ewma"] = Field(
        default="ledoit_wolf", description="Covariance estimation method"
    )
    use_tushare: bool = Field(default=False, description="Use Tushare real data (needs token)")
    use_baostock: bool = Field(default=False, description="Use Baostock real data (free, no key)")

    _validate_start = field_validator("start_date")(classmethod(lambda cls, v: _validate_date(v)))
    _validate_end = field_validator("end_date")(classmethod(lambda cls, v: _validate_date(v)))


class CompareRequest(BaseModel):
    optimizers: list[Literal["equal_weight", "mean_variance", "risk_parity"]] = Field(
        default=["equal_weight", "mean_variance", "risk_parity"]
    )
    n_stocks: int = Field(default=300, ge=50, le=500)


class SweepRequest(BaseModel):
    optimizers: list[Literal["equal_weight", "mean_variance", "risk_parity"]] = Field(
        default=["equal_weight", "mean_variance", "risk_parity"]
    )
    frequencies: list[Literal["daily", "weekly", "monthly"]] = Field(
        default=["monthly", "weekly"]
    )
    n_stocks_list: list[int] = Field(default=[200, 300])


# --- Response schemas ---

class PerformanceMetrics(BaseModel):
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_peak: str | None = None
    max_drawdown_trough: str | None = None
    win_rate: float
    profit_loss_ratio: float | None = None
    total_days: int
    information_ratio: float | None = None
    tracking_error: float | None = None
    excess_return: float | None = None
    n_rebalances: int
    optimizer: str
    initial_capital: float


class FactorICItem(BaseModel):
    name: str
    mean_ic: float
    std_ic: float
    icir: float
    ic_positive_ratio: float


class RiskMetrics(BaseModel):
    historical_var: float
    parametric_var: float
    historical_cvar: float


class StressTestItem(BaseModel):
    scenario: str
    cumulative_return: float
    max_drawdown: float


class HoldingItem(BaseModel):
    ticker: str
    weight: float
    sector: str | None = None
    pnl_pct: float | None = None


class ExposureInfo(BaseModel):
    n_assets: int
    effective_n: float
    top5_concentration: float
    top10_concentration: float
    sectors: dict[str, float]
    top_holdings: list[HoldingItem] | None = None


class DrawdownPeriod(BaseModel):
    start: str
    end: str
    trough: str
    depth: float
    duration_days: int
    recovered: bool
    recovery_date: str | None = None


class TurnoverItem(BaseModel):
    date: str
    turnover: float
    n_trades: int


class FactorScatterItem(BaseModel):
    factor_name: str
    points: list[dict[str, float]]
    ic: float
    icir: float
    t_stat: float


class AttributionItem(BaseModel):
    factor: str
    contribution_bps: float
    weight: float
    avg_ic: float


class ChartData(BaseModel):
    dates: list[str]
    equity: list[float]
    benchmark: list[float] | None = None
    drawdown: list[float] | None = None
    rolling_sharpe: list[float] | None = None
    rolling_sharpe_dates: list[str] | None = None
    monthly_returns: dict[str, list[float]] | None = None
    return_distribution: dict[str, Any] | None = None
    excess_cumulative: list[float] | None = None
    turnover: list[TurnoverItem] | None = None
    drawdown_periods: list[DrawdownPeriod] | None = None
    factor_scatter: list[FactorScatterItem] | None = None
    attribution: list[AttributionItem] | None = None


class RunStatus(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    stage: Literal["data", "factors", "alpha", "backtest", "report", "done"]
    started_at: str
    completed_at: str | None = None
    error: str | None = None


class RunResult(BaseModel):
    run_id: str
    performance: PerformanceMetrics | None = None
    risk: RiskMetrics | None = None
    stress_tests: list[StressTestItem] | None = None
    factors: list[FactorICItem] | None = None
    exposure: ExposureInfo | None = None
    chart_data: ChartData | None = None


class CompareResult(BaseModel):
    table: list[dict[str, Any]]
    chart_data: dict[str, ChartData] | None = None


class SweepResult(BaseModel):
    table: list[dict[str, Any]]
    best_params: dict[str, Any] | None = None


class ConfigInfo(BaseModel):
    default_config: dict[str, Any]
    available_options: dict[str, list[str]]
