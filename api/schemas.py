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


# --- ML Signal schemas ---

class MLTrainRequest(BaseModel):
    model_type: Literal["xgboost", "lightgbm", "ensemble"] = Field(
        default="lightgbm", description="ML model type"
    )
    train_window: int = Field(default=504, ge=100, le=2000, description="Training window (days)")
    n_splits: int = Field(default=5, ge=2, le=10, description="CV splits")
    retrain_frequency: int = Field(default=63, ge=21, le=252, description="Retrain frequency (days)")
    force_retrain: bool = Field(default=False, description="Force retrain even if model exists")


class MLPerformanceResponse(BaseModel):
    model_type: str
    test_ic: float
    test_icir: float
    train_ic: float
    feature_importance: dict[str, float]
    n_train_samples: int
    date: str


class MLSignalResponse(BaseModel):
    dates: list[str]
    assets: list[str]
    signal: list[list[float | None]]


# --- IC Monitor schemas ---

class ICMonitorRequest(BaseModel):
    rolling_window: int = Field(default=63, ge=21, le=252, description="Rolling IC window")
    decay_window: int = Field(default=126, ge=63, le=252, description="Decay detection window")
    significance_threshold: float = Field(default=0.03, ge=0.01, le=0.1)


class FactorICStatsResponse(BaseModel):
    name: str
    current_ic: float
    rolling_icir: float
    ic_trend: float
    ic_decay_rate: float
    half_life_days: int
    ic_positive_ratio: float
    alert_level: str


class ICMonitorSummary(BaseModel):
    factors: list[FactorICStatsResponse]
    alerts: list[dict[str, Any]]
    adaptive_weights: dict[str, float]


# --- Barra schemas ---

class BarraDecomposeRequest(BaseModel):
    date: str | None = Field(default=None, description="Date for decomposition (YYYY-MM-DD)")
    half_life: int = Field(default=252, ge=63, le=504, description="Covariance half-life")
    shrinkage_target: str = Field(default="identity", description="Shrinkage target")


class BarraRiskResponse(BaseModel):
    total_risk: float
    factor_risk: float
    specific_risk: float
    r_squared: float
    factor_contributions: dict[str, float]
    factor_exposures: dict[str, float]


class BarraCovarianceResponse(BaseModel):
    factors: list[str]
    covariance: list[list[float]]


# --- Parallel backtest schemas ---

class ParallelSweepRequest(BaseModel):
    param_grid: dict[str, list] = Field(description="Parameter grid: {param_name: [values]}")
    base_overrides: dict[str, Any] = Field(default_factory=dict, description="Base config overrides")
    metric: str = Field(default="sharpe_ratio", description="Optimization metric")
    max_workers: int | None = Field(default=None, description="Max parallel workers")


class ParallelSweepResponse(BaseModel):
    results: list[dict[str, Any]]
    best_params: dict[str, Any] | None
    best_metric: float | None
    total_duration: float
    n_success: int
    n_failed: int


# --- PostgreSQL Store schemas ---

class PostgresStoreStatsResponse(BaseModel):
    backend: str
    orders: int = 0
    positions: int = 0
    trades: int = 0
    pnl_snapshots: int = 0
    signals: int = 0


# --- WebSocket schemas ---

class WebSocketStatsResponse(BaseModel):
    source: str
    connected: bool
    subscribed: int
    cached_quotes: int
    message_count: int = 0
    last_message_age_s: float | None = None


class RealtimeQuoteResponse(BaseModel):
    code: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    prev_close: float = 0.0
    bid1_price: float = 0.0
    ask1_price: float = 0.0
    timestamp: str = ""


# --- Level 2 schemas ---

class OrderBookResponse(BaseModel):
    code: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    depth_imbalance: float = 0.0
    bids: list[dict[str, Any]]
    asks: list[dict[str, Any]]


class TickDataResponse(BaseModel):
    code: str
    timestamp: str
    price: float
    volume: float
    amount: float
    direction: str = ""


class VWAPResponse(BaseModel):
    code: str
    vwap: float
    n_ticks: int


class TradeFlowResponse(BaseModel):
    code: str
    buy_volume: float
    sell_volume: float
    net_flow: float
    buy_pct: float


class Level2StatsResponse(BaseModel):
    codes: int
    books_cached: int
    total_ticks: int
    running: bool


# --- Fundamental schemas ---

class FundamentalMetricsResponse(BaseModel):
    code: str
    pe_ttm: float = 0.0
    pb: float = 0.0
    ps_ttm: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    gross_margin: float = 0.0
    net_margin: float = 0.0
    revenue_growth: float = 0.0
    profit_growth: float = 0.0
    debt_ratio: float = 0.0
    dividend_yield: float = 0.0
    market_cap: float = 0.0


class FundamentalScreenRequest(BaseModel):
    codes: list[str] = Field(description="Stock codes to screen")
    pe_min: float | None = None
    pe_max: float | None = None
    pb_min: float | None = None
    pb_max: float | None = None
    roe_min: float | None = None
    roe_max: float | None = None
    market_cap_min: float | None = None
    dividend_yield_min: float | None = None
    debt_ratio_max: float | None = None


class FundamentalScreenResponse(BaseModel):
    total: int
    passed: int
    codes: list[str]


class FundamentalRankRequest(BaseModel):
    codes: list[str] = Field(description="Stock codes to rank")
    metric: str = Field(default="roe", description="Metric to rank by")
    ascending: bool = Field(default=False)
    top_n: int | None = Field(default=None, ge=1)


class FundamentalRankResponse(BaseModel):
    metric: str
    ranked: list[dict[str, Any]]


# --- Screener schemas ---

class ScreenRuleRequest(BaseModel):
    """A single screening rule from the API request."""
    factor: str = Field(..., description="Factor name, e.g. 'pe_ratio'")
    operator: Literal["gt", "gte", "lt", "lte", "eq", "ne", "between"] = Field(
        ..., description="Comparison operator"
    )
    value: float | list[float] = Field(..., description="Threshold value or [lo, hi]")


class ScreenRequest(BaseModel):
    rules: list[ScreenRuleRequest] = Field(..., description="List of screening rules")
    logic: Literal["and", "or"] = Field(default="and", description="Rule combination logic")
    date: str | None = Field(default=None, description="Target date YYYY-MM-DD (latest if None)")
    min_stocks: int = Field(default=5, ge=1, le=500, description="Minimum stocks (auto-relax)")
    max_stocks: int = Field(default=200, ge=1, le=1000, description="Maximum stocks (score cap)")
    config: str | None = Field(default=None, description="Config file path")


class ScreenStockInfo(BaseModel):
    code: str
    factors: dict[str, float]


class ScreenResponse(BaseModel):
    date: str
    logic: str
    num_rules: int
    total_stocks: int
    qualifying_stocks: int
    rules: list[str]
    results: list[ScreenStockInfo]
