"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

import itertools
import math
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Research/sweep windows: bounded like ``RunRequest`` so a caller cannot
#: turn one request into a multi-decade panel (or a 9999-12-31 OOM).
_MIN_REQUEST_DATE = date(2000, 1, 1)
_MAX_REQUEST_DATE = date(2100, 12, 31)
_MAX_REQUEST_SPAN_DAYS = 3653  # 10 years

#: Parallel-sweep overrides may reach these config roots only. ``data.*`` is
#: rejected (provider/path injection) except the two date keys, which are
#: validated and span-bounded.
_SWEEP_ALLOWED_ROOTS = frozenset({"universe", "portfolio", "backtest", "costs", "alpha"})
_SWEEP_DATA_DATE_KEYS = frozenset({"data.start_date", "data.end_date"})
_SWEEP_ENUMS = {
    "portfolio.optimizer": frozenset({"equal_weight", "mean_variance", "risk_parity"}),
    "alpha.method": frozenset({"equal_weight", "ic_weighted", "icir_weighted"}),
    "backtest.rebalance_frequency": frozenset({"daily", "weekly", "monthly"}),
    "portfolio.covariance.method": frozenset({"sample", "ledoit_wolf", "ewma"}),
    "costs.slippage_model": frozenset({"fixed", "proportional", "impact"}),
}
_SWEEP_RANGES: dict[str, tuple[float, float]] = {
    "universe.n_stocks": (50, 1000),
    "portfolio.constraints.max_weight": (0.0, 1.0),
    "portfolio.constraints.max_sector_exposure": (0.0, 1.0),
    "portfolio.constraints.max_turnover": (0.0, 1.0),
    "portfolio.constraints.lot_size": (1, 10_000),
    "portfolio.risk_aversion": (0.0, 100.0),
    "portfolio.covariance.lookback": (20, 2520),
    "portfolio.covariance.ewma_half_life": (5, 504),
    "backtest.initial_capital": (10_000.0, 1e12),
    "costs.commission": (0.0, 0.01),
    "costs.stamp_tax": (0.0, 0.01),
    "costs.slippage": (0.0, 0.05),
}
_SWEEP_KEY_PART_RE = re.compile(r"[a-z_][a-z0-9_]*\Z")
_SWEEP_FORBIDDEN_KEY_PARTS = (
    "path", "dir", "file", "url", "dsn", "token", "password", "passwd",
    "secret", "credential", "provider", "cache", "host", "port", "proxy", "env",
)


def _validate_date(v: str) -> str:
    if not _DATE_RE.match(v):
        raise ValueError("Date must be YYYY-MM-DD format")
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date: {exc}") from exc
    return v


def _parse_bounded_date(value: Any, label: str) -> date:
    """A YYYY-MM-DD date inside the supported research range."""
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise ValueError(f"{label} must be a YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid date: {exc}") from exc
    if not _MIN_REQUEST_DATE <= parsed <= _MAX_REQUEST_DATE:
        raise ValueError(
            f"{label} must be between {_MIN_REQUEST_DATE.isoformat()} and "
            f"{_MAX_REQUEST_DATE.isoformat()}"
        )
    return parsed


def _default_sweep_window() -> tuple[date, date]:
    """The configured data window, used when only one side is overridden."""
    from quant_platform.utils.config import load_config

    config = load_config()
    return (
        date.fromisoformat(str(config.data.start_date)),
        date.fromisoformat(str(config.data.end_date)),
    )


def _check_sweep_date_span(overrides: dict[str, Any]) -> None:
    from quant_platform.backtest.distributed import PARAM_ALIASES

    dates: dict[str, date] = {}
    for key, value in overrides.items():
        canonical = PARAM_ALIASES.get(key, key)
        if canonical in _SWEEP_DATA_DATE_KEYS:
            dates[canonical] = _parse_bounded_date(value, canonical)
    if not dates:
        return
    default_start, default_end = _default_sweep_window()
    start = dates.get("data.start_date", default_start)
    end = dates.get("data.end_date", default_end)
    if start >= end:
        raise ValueError("start_date must be before end_date")
    if (end - start).days > _MAX_REQUEST_SPAN_DAYS:
        raise ValueError("date span must be at most 10 years")


def _validate_sweep_override(key: Any, value: Any) -> None:
    """Validate one parallel-sweep override key/value pair.

    Unknown ``data.*``/path-like keys, non-scalar values and out-of-range
    numbers are rejected before any worker process starts (the audit
    reproduced ``n_stocks=10,000,000`` and 1900..2100 windows passing
    straight through to ``_load_data``).
    """
    from quant_platform.backtest.distributed import PARAM_ALIASES

    if not isinstance(key, str):
        raise ValueError("sweep override keys must be strings")
    canonical = PARAM_ALIASES.get(key, key)
    if canonical in _SWEEP_DATA_DATE_KEYS:
        _parse_bounded_date(value, canonical)
        return
    parts = canonical.split(".")
    if len(parts) < 2 or not all(_SWEEP_KEY_PART_RE.fullmatch(part) for part in parts):
        raise ValueError(f"invalid sweep override key {key!r}")
    root = parts[0]
    if root == "data":
        raise ValueError(
            "data.* overrides are not allowed; only data.start_date/data.end_date"
        )
    if root not in _SWEEP_ALLOWED_ROOTS:
        raise ValueError(
            f"sweep override {key!r} must be under one of "
            f"{sorted(_SWEEP_ALLOWED_ROOTS)}"
        )
    lowered_parts = canonical.lower().split(".")
    if any(part in _SWEEP_FORBIDDEN_KEY_PARTS for part in lowered_parts):
        raise ValueError(f"sweep override {key!r} is not allowed")
    if canonical in _SWEEP_ENUMS:
        if not isinstance(value, str) or value not in _SWEEP_ENUMS[canonical]:
            raise ValueError(
                f"{key!r} must be one of {sorted(_SWEEP_ENUMS[canonical])}"
            )
        return
    if canonical in _SWEEP_RANGES:
        low, high = _SWEEP_RANGES[canonical]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key!r} must be a number between {low} and {high}")
        number = float(value)
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key!r} must be between {low} and {high}")
        return
    if value is None or isinstance(value, (dict, list, tuple)):
        raise ValueError(f"{key!r} must be a scalar value")
    if isinstance(value, str) and len(value) > 64:
        raise ValueError(f"{key!r} value is too long (max 64 characters)")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{key!r} must be finite")



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

    @model_validator(mode="after")
    def _bound_the_window(self):
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("start_date must be before end_date")
        if (end - start).days > 3653:
            raise ValueError("date span must be at most 10 years")
        return self


class CompareRequest(BaseModel):
    optimizers: list[Literal["equal_weight", "mean_variance", "risk_parity"]] = Field(
        default=["equal_weight", "mean_variance", "risk_parity"], max_length=3
    )
    n_stocks: int = Field(default=300, ge=50, le=500)


class SweepRequest(BaseModel):
    optimizers: list[Literal["equal_weight", "mean_variance", "risk_parity"]] = Field(
        default=["equal_weight", "mean_variance", "risk_parity"], max_length=3,
    )
    frequencies: list[Literal["daily", "weekly", "monthly"]] = Field(
        default=["monthly", "weekly"], max_length=3
    )
    n_stocks_list: list[int] = Field(default=[200, 300], max_length=5)

    @field_validator("n_stocks_list")
    @classmethod
    def _bound_universes(cls, values: list[int]) -> list[int]:
        for value in values:
            if not 50 <= value <= 1000:
                raise ValueError("each universe size must be between 50 and 1000")
        return values

    @model_validator(mode="after")
    def _bound_the_product(self):
        """Dedupe and cap the grid: 100k optimizers meant 400k pipeline cells."""
        for name in ("optimizers", "frequencies", "n_stocks_list"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must not be empty")
            deduped = list(dict.fromkeys(values))
            if len(deduped) != len(values):
                setattr(self, name, deduped)
        combinations = (
            len(self.optimizers) * len(self.frequencies) * len(self.n_stocks_list)
        )
        if combinations > 200:
            raise ValueError(
                f"sweep expands to {combinations} combinations; the maximum is 200"
            )
        return self


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
    n_optimizer_fallbacks: int = 0
    optimizer_fallback_rate: float = 0.0
    effective_optimizer: str | None = None
    n_skipped_rebalances: int = 0


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


class MonthlyReturns(BaseModel):
    """Year x month grid for the dashboard's monthly-returns heatmap.

    The producer (`_build_chart_data`) and the consumer (`TerminalDashboard`'s
    heatmap) both already used this shape -- `years[i]` labels row i,
    `months[j]` labels column j, `data[i][j]` is that cell's return -- but the
    field was annotated `dict[str, list[float]]`, which no value of this shape
    can satisfy. `POST /api/run` therefore failed validation at its final stage
    on every request, so the run was marked failed and `/api/run/{id}/result`
    always returned 404.

    Cells with no observation are NaN; the chart skips them.
    """

    years: list[str] = Field(default_factory=list)
    months: list[int] = Field(default_factory=list)
    data: list[list[float | None]] = Field(default_factory=list)


class ChartData(BaseModel):
    dates: list[str]
    equity: list[float]
    benchmark: list[float] | None = None
    drawdown: list[float] | None = None
    rolling_sharpe: list[float] | None = None
    rolling_sharpe_dates: list[str] | None = None
    monthly_returns: MonthlyReturns | None = None
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
    # Multiple-testing correction for the trial grid (DSR + BH-FDR). All
    # optional so the response stays backward compatible for callers that
    # only read ``table``/``best_params``.
    num_trials: int | None = None
    best_dsr: dict[str, Any] | None = None
    selection_note: str | None = None


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
    weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Portfolio weights by ticker. Defaults to current OMS positions, "
            "then to equal weight across the universe. Note that an equal-weight "
            "whole-universe portfolio has zero net exposure to every "
            "cross-sectionally standardized factor by construction."
        ),
    )


class WalkForwardRequest(BaseModel):
    """POST /api/walkforward body.

    ``run_id`` is optional: the validator runs on the cached research
    context, and the field is accepted so the dashboard can keep sending the
    run it is displaying.  ``start_date``/``end_date`` override the
    configured data range for this request (and change the context cache
    key).
    """

    run_id: str | None = None
    mode: Literal["rolling", "expanding"] = Field(
        default="rolling", description="Rolling or expanding train window",
    )
    train_period: int = Field(default=504, ge=21, le=2520, description="Train sessions")
    test_period: int = Field(default=126, ge=5, le=1260, description="Test sessions")
    folds: int | None = Field(default=None, ge=1, le=50, description="Max folds")
    start_date: str | None = Field(default=None, description="ISO start date override")
    end_date: str | None = Field(default=None, description="ISO end date override")


class RiskDecomposeRequest(BaseModel):
    """POST /api/risk/decompose body (additive fields over the old run_id)."""

    run_id: str | None = None
    weights: dict[str, float] | None = Field(
        default=None, description="Portfolio weights; defaults to positions/equal weight",
    )
    date: str | None = Field(default=None, description="Exposure date (YYYY-MM-DD)")
    half_life: int = Field(default=252, ge=63, le=504)
    shrinkage_target: str = Field(default="identity")
    start_date: str | None = Field(default=None, description="ISO start date override")
    end_date: str | None = Field(default=None, description="ISO end date override")


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
    model_config = ConfigDict(allow_inf_nan=False)

    param_grid: dict[str, list] = Field(
        description=(
            "Parameter grid: {name: [values]}. Flat aliases optimizer/"
            "n_stocks/alpha_method/rebalance_frequency/covariance_method "
            "and dotted config paths (portfolio.optimizer) are supported; "
            "unknown keys are rejected before any worker starts."
        ),
    )
    base_overrides: dict[str, Any] = Field(default_factory=dict, description="Base config overrides")
    metric: str = Field(
        default="sharpe_ratio", max_length=64, pattern=r"^[a-z_][a-z0-9_]{0,63}$",
        description="Optimization metric",
    )
    max_workers: int = Field(
        default=4, ge=1, le=4, description="Max parallel workers (hard cap: 4)",
    )

    @model_validator(mode="after")
    def _bound_the_grid(self):
        if not self.param_grid:
            raise ValueError("param_grid must not be empty")
        if len(self.param_grid) > 10:
            raise ValueError("param_grid may have at most 10 parameters")
        combinations = 1
        for values in self.param_grid.values():
            if not isinstance(values, list) or not values:
                raise ValueError("each parameter must be a non-empty list of values")
            if len(values) > 20:
                raise ValueError("each parameter may have at most 20 values")
            combinations *= len(values)
        if combinations > 100:
            raise ValueError("param_grid expands to more than 100 combinations")

        # Every key/value (grid and base) must be a bounded scalar override,
        # and every effective combination must keep the date window sane.
        for key, value in self.base_overrides.items():
            _validate_sweep_override(key, value)
        for key, values in self.param_grid.items():
            for value in values:
                _validate_sweep_override(key, value)
        for combo in itertools.product(*self.param_grid.values()):
            merged = dict(self.base_overrides)
            merged.update(dict(zip(self.param_grid.keys(), combo, strict=False)))
            _check_sweep_date_span(merged)
        return self


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
    simulated: bool = False


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
    source: str = ""
    simulated: bool = False


# --- Level 2 schemas ---

class OrderBookResponse(BaseModel):
    code: str
    timestamp: str = ""
    simulated: bool = True
    source: str = "simulated"
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
    simulated: bool = True
    source: str = "simulated"


class VWAPResponse(BaseModel):
    code: str
    vwap: float
    n_ticks: int
    simulated: bool = True
    source: str = "simulated"


class TradeFlowResponse(BaseModel):
    code: str
    buy_volume: float
    sell_volume: float
    net_flow: float
    buy_pct: float
    simulated: bool = True
    source: str = "simulated"


class Level2StatsResponse(BaseModel):
    codes: int
    books_cached: int
    total_ticks: int
    running: bool
    simulated: bool = True
    source: str = "simulated"


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
    source: str = ""
    is_synthetic: bool = False


class FundamentalScreenRequest(BaseModel):
    codes: list[str] = Field(description="Stock codes to screen", max_length=100)
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
    codes: list[str] = Field(description="Stock codes to rank", max_length=100)
    metric: str = Field(default="roe", description="Metric to rank by")
    ascending: bool = Field(default=False)
    top_n: int | None = Field(default=None, ge=1)


class FundamentalRankResponse(BaseModel):
    metric: str
    ranked: list[dict[str, Any]]


# --- Screener schemas ---

class ScreenRuleRequest(BaseModel):
    """A single screening rule from the API request."""

    model_config = ConfigDict(allow_inf_nan=False)
    factor: str = Field(..., description="Factor name, e.g. 'pe_ratio'")
    operator: Literal["gt", "gte", "lt", "lte", "eq", "ne", "between"] = Field(
        ..., description="Comparison operator"
    )
    value: float | list[float] = Field(..., description="Threshold value or [lo, hi]")


class ScreenRequest(BaseModel):
    rules: list[ScreenRuleRequest] = Field(
        ..., description="List of screening rules", max_length=20,
    )
    logic: Literal["and", "or"] = Field(default="and", description="Rule combination logic")
    date: str | None = Field(default=None, description="Target date YYYY-MM-DD (latest if None)")
    min_stocks: int = Field(default=5, ge=1, le=500, description="Minimum stocks (auto-relax)")
    max_stocks: int = Field(default=200, ge=1, le=1000, description="Maximum stocks (score cap)")
    config: str | None = Field(
        default=None,
        description="Optional config path resolved inside the server config/ directory",
    )


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
