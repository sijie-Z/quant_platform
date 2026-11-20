"""Monitor API — Bloomberg Terminal-style aggregation endpoints.

Provides unified monitoring data for the frontend dashboard:
- Risk overview (factor exposures, concentration, drawdown, VaR)
- TCA summary (IS, delay/impact/timing decomposition)
- Factor status (rolling IC, attribution, decay alerts)
- Capacity gauge (AUM, participation rate, capacity curve)
- Config update (risk limits)
- Kill switch

All endpoints aggregate existing module data into dashboard-ready JSON.
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/monitor")


def _json_float(value: Any, default: Any = None) -> Any:
    """Finite float or ``default`` (Starlette JSON rejects NaN/inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _window_or_422(
    start_date: str | None, end_date: str | None,
) -> tuple[str | None, str | None]:
    from quant_platform.api.research_context import parse_window

    try:
        return parse_window(start_date, end_date)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def _research_context_or_503(
    start_date: str | None, end_date: str | None,
):
    from quant_platform.api.research_context import get_research_context

    try:
        return await get_research_context(
            start_date=start_date, end_date=end_date,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Monitor research context build failed")
        raise HTTPException(
            status_code=503,
            detail="Research data unavailable; see server logs",
            headers={"X-Error-Code": "research_context_unavailable"},
        ) from exc


async def _compute_cached(context, name: str, builder):
    """Run a derived monitor computation on the shared analytics pool."""
    from quant_platform.api import routes as routes_module

    cached = routes_module._derived_get(context, name, None)
    if cached is not None:
        if isinstance(cached, BaseModel):
            return cached.model_copy(update={"cached": True})
        return cached
    return await routes_module._run_analytics(
        _build_and_cache, context, name, builder,
    )


def _build_and_cache(context, name: str, builder):
    from quant_platform.api import routes as routes_module
    from quant_platform.api.research_context import (
        _key_lock_for,
        derived_cache_get,
        derived_cache_put,
    )

    key = routes_module.derived_key(context, name, None)
    with _key_lock_for(key):
        hit = derived_cache_get(key)
        if hit is not None:
            if isinstance(hit, BaseModel):
                return hit.model_copy(update={"cached": True})
            return hit
        result = builder(context)
        derived_cache_put(key, result)
        return result


# ── Pydantic models ──


class RiskOverviewResponse(BaseModel):
    available: bool = Field(
        False, description="Whether the risk state below was actually read"
    )
    reason: str = Field("", description="Why the risk overview is unavailable")
    factor_exposures: dict[str, float] = Field(default_factory=dict, description="Barra 10 factor exposures")
    factor_exposures_available: bool = Field(
        False,
        description="Whether factor_exposures contains a real decomposition",
    )
    factor_exposures_reason: str = Field(
        "",
        description="Why factor exposures are unavailable, when they are",
    )
    sector_concentration: dict[str, float] = Field(default_factory=dict, description="Top sector weights")
    current_drawdown: float = Field(0.0, description="Current drawdown %")
    volatility: float = Field(0.0, description="Annualized volatility")
    var_95: float = Field(0.0, description="95% VaR")
    cvar_95: float = Field(0.0, description="95% CVaR")
    risk_level: str = Field("UNKNOWN", description="Risk level (UNKNOWN when unread)")
    portfolio_value: float = Field(0.0)
    daily_pnl: float = Field(0.0)
    n_positions: int = Field(0)
    kill_switch: bool = Field(False, description="Whether the kill switch is active")
    persisted: bool = Field(
        False, description="Whether the risk state is durable across a restart",
    )
    limits: dict[str, float] = Field(
        default_factory=dict, description="Live RiskMonitor limits",
    )


class TCASummaryResponse(BaseModel):
    available: bool = Field(False, description="Whether the TCA metrics below are populated")
    reason: str = Field("", description="Why TCA is unavailable, when it is")
    n_orders: int = Field(0, description="Distinct stored orders represented by trades")
    n_trades: int = 0
    n_analyzed: int = Field(0, description="Trades joined to a reference price")
    mean_is_bps: float = 0.0
    mean_delay_bps: float | None = Field(
        None, description="Unmeasured without quote data (see decomposition)",
    )
    mean_impact_bps: float | None = Field(
        None, description="Unmeasured without quote data (see decomposition)",
    )
    mean_timing_bps: float | None = Field(
        None, description="Unmeasured without quote data (see decomposition)",
    )
    mean_arrival_bps: float | None = Field(
        None, description="Unmeasured without an arrival quote",
    )
    mean_fee_bps: float = Field(
        0.0, description="Explicit commission + tax as bps of executed notional",
    )
    median_is_bps: float = 0.0
    daily_trend: list[dict] = Field(default_factory=list, description="Daily IS trend [{date, is_bps}]")
    cost_breakdown: list[dict] = Field(default_factory=list, description="[{date, delay, impact, timing}]")
    by_ticker: dict[str, float] = Field(default_factory=dict, description="IS by ticker")
    decomposition: dict = Field(
        default_factory=dict,
        description="Which cost decompositions the stored fields support",
    )
    skipped: dict = Field(
        default_factory=dict, description="Trades excluded per reason",
    )
    limits: dict = Field(
        default_factory=dict, description="Row limits applied to the store reads",
    )
    as_of: str = Field("", description="Latest stored trade date")
    provider: str = "store"
    synthetic: bool = False
    config_hash: str = ""
    computed_at: str = ""
    cached: bool = False


class FactorStatusResponse(BaseModel):
    available: bool = Field(False, description="Whether rolling IC data is populated")
    reason: str = Field("", description="Why factor status is unavailable, when it is")
    factors: list[dict] = Field(default_factory=list, description="Factor IC stats [{name, ic, icir, trend, alert}]")
    rolling_ic: dict[str, list[float | None]] = Field(default_factory=dict, description="Rolling IC series per factor")
    ic_dates: list[str] = Field(default_factory=list, description="Dates for rolling IC")
    attribution: list[dict] = Field(default_factory=list, description="[{factor, contribution_bps}]")
    attribution_reason: str = Field(
        "",
        description="Why contribution attribution is not in this response",
    )
    decay_alerts: list[dict] = Field(default_factory=list, description="Factors with IC decay")
    disabled_factors: list[str] = Field(default_factory=list, description="Auto-decay disabled factors")
    required_sessions: int = Field(
        0, description="Sessions needed for a rolling IC observation",
    )
    window_sessions: int = Field(
        0, description="Trailing sessions the rolling statistics cover",
    )
    panel_start: str = Field("", description="First date of the analysis window")
    decay_replay: str = Field(
        "",
        description="Which auto-decay thresholds the disabled-factor replay used",
    )
    as_of: str = ""
    provider: str = ""
    synthetic: bool = False
    config_hash: str = ""
    computed_at: str = ""
    cached: bool = False


class CapacityGaugeResponse(BaseModel):
    available: bool = Field(False, description="Whether the capacity curve is populated")
    reason: str = Field("", description="Why capacity is unavailable, when it is")
    current_aum: float = 0.0
    capacity_aum: float = 0.0
    usage_pct: float = 0.0
    participation_rate: float = 0.0
    sharpe_at_capacity: float = 0.0
    aum_curve: list[dict] = Field(default_factory=list, description="[{aum, sharpe, return}]")
    weights_source: str = Field(
        "", description="request / oms_positions / equal_weight",
    )
    aum_source: str = Field("", description="pnl_history / positions / none")
    usage_available: bool = Field(
        True, description="False when no live book supplies the current AUM",
    )
    participation_assumption: dict = Field(
        default_factory=dict,
        description="ADV window and max participation used by the gauge",
    )
    curve_reason: str = Field(
        "", description="Why the AUM-Sharpe curve is empty, when it is",
    )
    as_of: str = ""
    provider: str = ""
    synthetic: bool = False
    config_hash: str = ""
    computed_at: str = ""
    cached: bool = False


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    max_position_pct: float | None = Field(None, ge=0.01, le=0.20, description="Max single position weight (0.01-0.20)")
    max_sector_pct: float | None = Field(None, ge=0.05, le=0.50, description="Max sector weight (0.05-0.50)")
    max_drawdown_pct: float | None = Field(None, ge=0.01, le=0.30, description="Drawdown halt threshold (0.01-0.30)")
    max_daily_loss_pct: float | None = Field(None, ge=0.005, le=0.10, description="Daily loss limit (0.005-0.10)")


class ConfigUpdateResponse(BaseModel):
    updated: list[str] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)


class KillSwitchRequest(BaseModel):
    activate: bool = Field(True, description="True to activate, False to deactivate")
    reason: str = Field("Manual activation from monitor", description="Reason for activation")


class KillSwitchResponse(BaseModel):
    active: bool
    message: str
    persisted: bool = Field(
        False, description="Whether the new state was written to durable storage",
    )
    persist_error: bool = Field(
        False, description="True when the durable write failed; state is memory-only",
    )


# ── Singleton references (lazy, shared with routes.py) ──

_core_store = None
_core_bus = None
_core_risk = None
_run_results_cache: dict[str, Any] = {}

# The getters run from the event loop *and* from `run_in_threadpool` workers.
# `if x is None: x = ...` without a lock could construct two Stores (each
# running DDL) or two risk monitors, leaking the first. Construction errors
# used to be swallowed with `except Exception: pass`, which turned every
# monitor panel into an unexplained empty payload.
_singleton_lock = threading.Lock()


def _get_store():
    global _core_store
    instance = _core_store
    if instance is None:
        with _singleton_lock:
            if _core_store is None:
                # Share the API's store instead of opening a second handle on
                # the same SQLite file: two Store objects for one DB made the
                # risk monitor see a "different target" on every dashboard
                # poll (S21 regression).
                try:
                    from quant_platform.api import routes as routes_module
                    _core_store = getattr(routes_module, "_core_store", None)
                except Exception:
                    _core_store = None
                if _core_store is None:
                    try:
                        from quant_platform.core.store import Store
                        _core_store = Store()
                    except Exception:
                        logger.exception("Monitor store construction failed")
                        return None
            instance = _core_store
    return instance


def _get_risk():
    global _core_risk
    instance = _core_risk
    if instance is None:
        with _singleton_lock:
            if _core_risk is None:
                try:
                    from quant_platform.risk.circuit_breaker import get_risk_monitor
                    _core_risk = get_risk_monitor()
                except Exception:
                    logger.exception("Monitor risk-monitor lookup failed")
                    return None
            instance = _core_risk
    # S21: attach the dashboard's monitor to the Store once. Re-attaching on
    # every poll with a different Store object used to reset the in-memory
    # risk state (including an armed kill switch). The tenant reset rule in
    # RiskMonitor now only fires on a genuine tenant change; this guard keeps
    # the monitor bound to its first store.
    store = _get_store()
    attach = getattr(instance, "attach_persistence", None)
    if (
        store is not None
        and callable(attach)
        and getattr(instance, "_store", None) is None
    ):
        try:
            attach(store, tenant_id="default", restore=False)
        except Exception:
            logger.exception("Monitor risk-state persistence setup failed")
    return instance


def _get_bus():
    global _core_bus
    instance = _core_bus
    if instance is None:
        with _singleton_lock:
            if _core_bus is None:
                try:
                    from quant_platform.core.events import get_event_bus
                    _core_bus = get_event_bus()
                except Exception:
                    logger.exception("Monitor event-bus lookup failed")
                    return None
            instance = _core_bus
    return instance


# ── Risk Overview ──


@router.get("/risk-overview", response_model=RiskOverviewResponse)
async def get_risk_overview():
    """Aggregate risk data: factor exposures, concentration, drawdown, VaR."""
    # SQLite reads, pandas and the VaR computation are all blocking; run the
    # whole aggregate in a worker thread so one dashboard poll cannot freeze
    # every HTTP/WebSocket handler on the event loop.
    return await run_in_threadpool(_compute_risk_overview)


def _compute_risk_overview() -> RiskOverviewResponse:
    result = RiskOverviewResponse(available=False, risk_level="UNKNOWN")
    result.factor_exposures_reason = (
        "no Barra decomposition is cached for the live book; "
        "POST /api/barra/decompose with portfolio weights to compute one"
    )

    # Get risk status from RiskMonitor. A failure here must not look like a
    # healthy zero-risk portfolio: the old code returned 200 with
    # risk_level="GREEN", portfolio_value=0 and vol=0 on any error.
    risk = _get_risk()
    if risk is None:
        result.reason = (
            "risk monitor is not available; the trading engine has not "
            "started or its state could not be read"
        )
    else:
        try:
            status = risk.get_status()
            result.risk_level = status.get("risk_level", "UNKNOWN")
            result.portfolio_value = status.get("portfolio_value", 0)
            result.daily_pnl = status.get("daily_pnl", 0)
            result.current_drawdown = abs(status.get("current_drawdown", 0))
            result.n_positions = status.get("n_positions", 0)
            result.kill_switch = bool(status.get("kill_switch", False))
            result.persisted = bool(status.get("persisted", False))
            result.limits = {
                str(k): float(v)
                for k, v in (status.get("limits") or {}).items()
            }
            result.available = True
        except Exception:
            # Short reason only; the traceback stays in the server log.
            logger.exception("Risk overview: risk status read failed")
            result.reason = "risk status unavailable"

    # Get positions from store for concentration analysis
    store = _get_store()
    if store:
        try:
            positions = store.get_positions()
            if positions:
                total_mv = sum(p.get("market_value", 0) for p in positions)
                if total_mv > 0:
                    # Sector concentration (simplified: group by first digit of code)
                    sectors: dict[str, float] = {}
                    for p in positions:
                        code = p.get("code", "")
                        sector = _code_to_sector(code)
                        weight = p.get("market_value", 0) / total_mv
                        sectors[sector] = sectors.get(sector, 0) + weight
                    # Top 5 sectors
                    sorted_sectors = sorted(sectors.items(), key=lambda x: -x[1])[:5]
                    result.sector_concentration = {k: round(v, 4) for k, v in sorted_sectors}

            # Compute VaR from P&L history
            pnl = store.get_pnl_history(days=60)
            if len(pnl) >= 10:
                equities = [p.get("total_equity", 0) for p in pnl]
                returns = pd.Series(equities).pct_change().dropna()
                if len(returns) >= 5:
                    from quant_platform.risk.var import var_summary
                    vs = var_summary(returns, confidence=0.95, horizon=1)
                    result.var_95 = round(vs.get("historical_var", 0), 6)
                    result.cvar_95 = round(vs.get("historical_cvar", 0), 6)
                    result.volatility = round(float(returns.std() * np.sqrt(252)), 6)
        except Exception:
            logger.exception("Risk overview: positions/VaR computation failed")
            if result.available:
                result.reason = "positions/VaR unavailable"
            else:
                result.reason = result.reason or "risk data unavailable"

    return result


# ── TCA Summary ──


@router.get("/tca-summary", response_model=TCASummaryResponse)
async def get_tca_summary():
    """Aggregate TCA data from stored trades."""
    return await run_in_threadpool(_compute_tca_summary)


def _compute_tca_summary() -> TCASummaryResponse:
    """Implementation shortfall from stored trades and their orders.

    ``trades`` carries execution price/quantity/side and ``orders`` carries
    the order's reference (limit/decision) price.  IS is measured against
    that reference: ``side_sign * (exec - reference) / reference * 1e4``.
    Delay/impact/timing separation needs arrival, VWAP and close quotes,
    which the store does not keep; the response says so in
    ``decomposition`` instead of splitting a number that was never measured.
    """
    result = TCASummaryResponse(
        computed_at=datetime.now(UTC).isoformat(),
        provider="store",
    )
    result.reason = "no stored trades"

    store = _get_store()
    if not store:
        result.reason = "no store is available to read trades from"
        return result

    try:
        trades = store.get_trades(limit=500)
    except Exception:
        logger.exception("TCA summary: trade read failed")
        result.reason = "stored trades could not be read"
        return result
    if not trades:
        return result

    result.n_trades = len(trades)
    result.n_orders = len({
        str(t.get("order_id", "")) for t in trades if t.get("order_id")
    })
    result.limits = {"trades": 500, "orders": 1000}
    try:
        orders = store.get_orders(limit=1000)
    except Exception:
        logger.exception("TCA summary: order read failed")
        orders = []
    orders_by_id = {o.get("order_id"): o for o in orders}

    skipped = {"missing_order": 0, "no_reference_price": 0, "invalid_trade": 0}
    rows: list[dict] = []
    for trade in trades:
        side = str(trade.get("side", "")).lower()
        if side not in ("buy", "sell"):
            skipped["invalid_trade"] += 1
            continue
        quantity = _json_float(trade.get("quantity"), 0.0) or 0.0
        price = _json_float(trade.get("price"), 0.0) or 0.0
        if quantity <= 0 or price <= 0:
            skipped["invalid_trade"] += 1
            continue

        order = orders_by_id.get(trade.get("order_id"))
        if order is None:
            skipped["missing_order"] += 1
            continue
        reference = _json_float(order.get("price"), 0.0) or 0.0
        if reference <= 0:
            skipped["no_reference_price"] += 1
            continue

        sign = 1.0 if side == "buy" else -1.0
        is_bps = sign * (price - reference) / reference * 1e4
        fee_bps = (
            (_json_float(trade.get("commission"), 0.0) or 0.0)
            + (_json_float(trade.get("tax"), 0.0) or 0.0)
        ) / (quantity * price) * 1e4
        executed_at = str(trade.get("executed_at", ""))
        rows.append({
            "trade_id": trade.get("trade_id", ""),
            "order_id": trade.get("order_id", ""),
            "code": str(trade.get("code", "")),
            "side": side,
            "quantity": quantity,
            "price": price,
            "reference": reference,
            "is_bps": _json_float(is_bps),
            "fee_bps": _json_float(fee_bps, 0.0),
            "date": executed_at[:10],
        })

    rows = [row for row in rows if row["is_bps"] is not None]
    result.skipped = skipped
    result.decomposition = {
        "available": False,
        "benchmark": "order_price",
        "fields": {
            "implementation_shortfall": True,
            "arrival": False,
            "vwap": False,
            "delay_impact_timing": False,
        },
        "reason": (
            "stored trades and orders carry execution price, quantity, side "
            "and the order's reference price; they carry no arrival/VWAP/close "
            "quote, so arrival, delay/impact/timing and VWAP costs cannot be "
            "computed. Implementation shortfall is measured against the "
            "order's reference price."
        ),
    }

    if not rows:
        result.reason = (
            f"{result.n_trades} stored trades, but none joins to an order "
            "with a positive reference price; implementation shortfall needs "
            "a reference price"
        )
        return result

    is_values = np.asarray([row["is_bps"] for row in rows], dtype=float)
    fees = np.asarray([row["fee_bps"] for row in rows], dtype=float)
    result.available = True
    result.reason = ""
    result.n_analyzed = len(rows)
    result.mean_is_bps = round(float(np.mean(is_values)), 6)
    result.median_is_bps = round(float(np.median(is_values)), 6)
    result.mean_fee_bps = round(float(np.mean(fees)), 6)
    # arrival/delay/impact/timing stay null: no quote data was stored.

    by_date: dict[str, list[float]] = {}
    for row in rows:
        if row["date"]:
            by_date.setdefault(row["date"], []).append(row["is_bps"])
    result.daily_trend = [
        {"date": day, "is_bps": round(float(np.mean(values)), 6)}
        for day, values in sorted(by_date.items())
    ]

    by_ticker: dict[str, list[float]] = {}
    for row in rows:
        by_ticker.setdefault(row["code"], []).append(row["is_bps"])
    result.by_ticker = {
        code: round(float(np.mean(values)), 6)
        for code, values in sorted(by_ticker.items())
    }

    dates = [row["date"] for row in rows if row["date"]]
    result.as_of = max(dates) if dates else ""
    return result


# ── Factor Status ──


@router.get("/factor-status", response_model=FactorStatusResponse)
async def get_factor_status(
    start_date: str | None = Query(None, description="ISO start date override"),
    end_date: str | None = Query(None, description="ISO end date override"),
):
    """Rolling factor IC/ICIR and decay status from the research context."""
    start, end = _window_or_422(start_date, end_date)
    context = await _research_context_or_503(start, end)
    return await _compute_cached(context, "factor_status", _compute_factor_status)


def _compute_factor_status(context) -> FactorStatusResponse:
    """Rolling IC/ICIR, trend and decay alerts from processed factor panels."""
    from quant_platform.api import routes as routes_module
    from quant_platform.factors.ic_monitor import (
        FactorICAutoDecay,
        FactorICMonitor,
        ICMonitorConfig,
    )

    result = FactorStatusResponse(**context.provenance())
    factors_in, window_sessions, panel_start = routes_module._analysis_window(
        context,
    )
    result.window_sessions = int(window_sessions)
    result.panel_start = panel_start
    config = ICMonitorConfig()
    # compute_rolling_ic needs `rolling_window` raw IC observations before it
    # emits its first smoothed value; compute_factor_stats then needs
    # `min_observations` of those.
    required = config.rolling_window + config.min_observations + 1
    result.required_sessions = int(required)

    monitor = FactorICMonitor(config=config)
    with warnings.catch_warnings():
        # Sparse cross-sections legitimately produce constant-input warnings
        # from the Spearman correlation; suppress them instead of emitting one
        # per date per factor on every poll.
        warnings.simplefilter("ignore")
        monitor.compute_all(
            factors_in, routes_module._analysis_returns(context, factors_in),
        )

    if not monitor.factor_stats:
        result.reason = (
            f"the research context has {context.n_sessions} sessions; rolling "
            f"IC needs at least {required} "
            f"({config.rolling_window}-day window + "
            f"{config.min_observations} observations). Extend the date range."
        )
        return result

    factors: list[dict] = []
    for name in sorted(monitor.factor_stats):
        stats = monitor.factor_stats[name]
        factors.append({
            "name": str(name),
            "current_ic": _json_float(stats.current_ic, 0.0),
            "icir": _json_float(stats.rolling_icir, 0.0),
            "trend": "up" if stats.ic_trend > 0 else "down",
            "trend_slope": _json_float(stats.ic_trend, 0.0),
            "alert": str(stats.alert_level),
            "half_life_days": int(stats.half_life_days),
            "positive_ratio": _json_float(stats.ic_positive_ratio, 0.0),
            "last_significant_date": str(stats.last_significant_date),
        })

    series_map = {
        name: monitor.ic_history[name]
        for name in monitor.factor_stats
        if name in monitor.ic_history
    }
    union_dates: list[Any] = []
    if series_map:
        union = set()
        for series in series_map.values():
            union.update(series.index)
        union_dates = sorted(union)
    tail_dates = union_dates[-60:]

    rolling_ic: dict[str, list[Any]] = {}
    for name, series in series_map.items():
        values: list[Any] = []
        for day in tail_dates:
            try:
                values.append(_json_float(series.get(day)))
            except Exception:  # pragma: no cover - defensive
                values.append(None)
        rolling_ic[name] = values
    result.rolling_ic = rolling_ic
    result.ic_dates = [
        str(pd.Timestamp(day).date()) for day in tail_dates
    ]

    result.decay_alerts = [
        {
            "factor": str(alert.get("factor", "")),
            "alert_level": str(alert.get("alert_level", "")),
            "current_ic": _json_float(alert.get("current_ic"), 0.0),
            "ic_decay_rate": _json_float(alert.get("ic_decay_rate"), 0.0),
            "half_life_days": int(alert.get("half_life_days", 0) or 0),
            "message": str(alert.get("message", "")),
        }
        for alert in monitor.get_alerts()
    ]

    # Replay each factor's real rolling-IC series through the same auto-decay
    # state machine the alpha combination uses.  The replay is bounded to the
    # trailing decay window (126 observations) so a dashboard poll does not
    # walk the full five-year history -- the state machine only ever judges
    # the recent window anyway.
    decay = FactorICAutoDecay()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, series in series_map.items():
            for value in series.tail(config.decay_window).to_numpy(dtype=float):
                decay.update(name, float(value))
                decay.check_and_update(name)
    result.disabled_factors = sorted(decay.disabled_factors)
    result.decay_replay = (
        "FactorICAutoDecay defaults (20-observation decay window, 0.01 "
        "threshold, 5-observation recovery window), replayed over the "
        f"trailing {config.decay_window} rolling-IC values"
    )

    result.attribution_reason = (
        "factor P&L contribution attribution needs realized strategy "
        "positions; this panel reports measured IC and decay from the "
        "research factor panels"
    )
    result.factors = factors
    result.available = True
    result.reason = ""
    return result


# ── Capacity Gauge ──


@router.get("/capacity-gauge", response_model=CapacityGaugeResponse)
async def get_capacity_gauge(
    start_date: str | None = Query(None, description="ISO start date override"),
    end_date: str | None = Query(None, description="ISO end date override"),
):
    """ADV-based strategy capacity from the research context's volume panel."""
    start, end = _window_or_422(start_date, end_date)
    context = await _research_context_or_503(start, end)
    # Not derived-cached: current_aum reads the live book (PnL/positions), so a
    # cached gauge would report a stale AUM for the TTL. The context itself
    # (the expensive part) is still cached.
    from quant_platform.api import routes as routes_module

    return await routes_module._run_analytics(_compute_capacity_gauge, context)


def _compute_capacity_gauge(context) -> CapacityGaugeResponse:
    """Capacity from the book's weights and a derived daily-volume panel.

    The pipeline exposes turnover, not volume.  ``turnover x shares`` recovers
    a share-volume panel, with ``shares = market_cap / close``; that identity
    is exact for providers whose market cap is price x shares and a
    documented approximation otherwise.  The response states the assumption.
    """
    from quant_platform.api import routes as routes_module
    from quant_platform.backtest.capacity import CapacityEstimator

    result = CapacityGaugeResponse(**context.provenance())

    prices = context.prices
    if prices is None or prices.empty:
        result.reason = "the research context has no price panel"
        return result

    # 1. Current book: stored positions win, else OMS, else equal weight.
    store = _get_store()
    positions: list[dict] = []
    if store is not None:
        try:
            positions = store.get_positions() or []
        except Exception:
            logger.exception("Capacity gauge: position read failed")
            positions = []

    weights = None
    weights_source = ""
    if positions:
        marks = {
            str(p.get("code", "")): _json_float(p.get("market_value"), 0.0) or 0.0
            for p in positions
        }
        candidate = pd.Series(marks, dtype=float).reindex(prices.columns).fillna(0.0)
        total = float(candidate.sum())
        if total > 0:
            weights = candidate / total
            weights_source = "oms_positions"
    if weights is None:
        raw_prices = None
        attrs = getattr(context.metadata, "attrs", None)
        if isinstance(attrs, dict):
            block = attrs.get("raw_prices")
            if isinstance(block, dict):
                raw_prices = block.get("close")
        weights, weights_source = routes_module._resolve_weights_with_source(
            None, prices, raw_prices,
        )

    # 2. Current AUM (a live-book measurement; 0 when there is no live book).
    current_aum = 0.0
    aum_source = "none"
    if store is not None:
        try:
            pnl = store.get_pnl_history(days=30) or []
            if pnl:
                current_aum = _json_float(pnl[-1].get("total_equity"), 0.0) or 0.0
                aum_source = "pnl_history"
            elif positions:
                current_aum = float(sum(
                    _json_float(p.get("market_value"), 0.0) or 0.0
                    for p in positions
                ))
                aum_source = "positions"
        except Exception:
            logger.exception("Capacity gauge: PnL read failed")

    result.current_aum = round(current_aum, 2)
    result.aum_source = aum_source
    result.usage_available = aum_source != "none"

    # 3. Volume panel: turnover x shares, shares = market_cap / close.
    turnover = context.turnover
    market_cap = None
    if context.fin_unstacked is not None and hasattr(context.fin_unstacked, "get"):
        market_cap = context.fin_unstacked.get("market_cap")
    if turnover is None or market_cap is None:
        result.reason = (
            "no volume panel: the research context has no turnover panel or "
            "no market-cap panel to derive share counts from"
        )
        result.weights_source = weights_source
        result.aum_source = aum_source
        return result

    turnover = turnover.reindex(index=prices.index, columns=prices.columns)
    market_cap = market_cap.reindex(index=prices.index, columns=prices.columns).ffill()
    close = prices.replace(0.0, np.nan)
    shares = market_cap / close
    volumes = (turnover * shares).replace([np.inf, -np.inf], np.nan)
    if volumes.dropna(how="all").empty:
        result.reason = (
            "the derived volume panel is empty (turnover or market cap is "
            "missing for every asset)"
        )
        result.weights_source = weights_source
        result.aum_source = aum_source
        return result

    # 4. ADV-based capacity.  Assumption: the whole book can be built (or
    # unwound) in one session at no more than `max_participation` of each
    # asset's average daily volume.
    max_participation = 0.10
    adv_window = 60
    adv_cny = (volumes.tail(adv_window) * prices.tail(adv_window)).mean()
    weighted = weights[weights.abs() > 1e-9]
    adv = adv_cny.reindex(weighted.index).astype(float)
    if weighted.empty:
        result.reason = "the resolved book has no non-zero weight"
        result.weights_source = weights_source
        result.aum_source = aum_source
        return result
    advances = adv.to_numpy(dtype=float)
    bad_mask = ~(np.isfinite(advances) & (advances > 0))
    if bad_mask.any():
        # Dropping an untradeable name from the min would overstate capacity.
        bad_names = [str(name) for name in weighted.index[bad_mask][:5]]
        more = int(bad_mask.sum()) - len(bad_names)
        result.reason = (
            "no positive average daily volume for weighted assets "
            + ", ".join(bad_names)
            + (f" (+{more} more)" if more > 0 else "")
            + "; an ADV capacity estimate would overstate capacity"
        )
        result.weights_source = weights_source
        result.aum_source = aum_source
        return result
    usable = adv

    capacity_aum = float(
        (usable * max_participation / weighted.reindex(usable.index).abs()).min()
    )
    policy = {
        "max_participation": max_participation,
        "adv_window_sessions": adv_window,
        "volume_source": (
            "turnover x (market_cap / adjusted close); exact when market cap "
            "is price x shares, an approximation otherwise"
        ),
        "capacity_definition": (
            "min over weighted assets of ADV_CNY x max_participation / "
            "|weight| -- the AUM at which building the book in one session "
            "stays under the participation cap"
        ),
        "curve_assumption": (
            "the current book is entered at a single rebalance about 12 "
            "months before the data end and held; the AUM-Sharpe curve "
            "applies participation caps and square-root impact over the "
            "remaining sessions"
        ),
    }

    result.capacity_aum = round(capacity_aum, 2)
    result.weights_source = weights_source
    result.participation_assumption = policy
    result.usage_pct = (
        round(current_aum / capacity_aum * 100.0, 4) if capacity_aum > 0 else 0.0
    )
    result.participation_rate = (
        round(
            float(
                (weighted.reindex(usable.index).abs() * current_aum)
                .div(usable)
                .max()
            ),
            6,
        )
        if current_aum > 0 else 0.0
    )

    # 5. AUM-Sharpe curve from the real capacity simulator, at AUM levels
    # around the ADV-based capacity.
    try:
        if len(prices.index) > 2:
            point = prices.index[-min(262, len(prices.index) - 1)]
            aum_range = sorted({
                max(1e5, float(round(capacity_aum * factor, -3)))
                for factor in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
            })
            estimator = CapacityEstimator(max_participation=max_participation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curve = estimator.estimate(
                    weights_history={pd.Timestamp(point): weights},
                    prices=prices,
                    volumes=volumes,
                    returns=context.returns,
                    benchmark_returns=context.benchmark,
                    aum_range=aum_range,
                )
            result.aum_curve = [
                {
                    "aum": float(row.aum),
                    "sharpe": _json_float(row.sharpe_ratio),
                    "return": _json_float(row.annualized_return),
                }
                for row in curve.results
            ]
            at_capacity = [
                row for row in curve.results if row.aum <= capacity_aum
            ]
            chosen = (
                max(at_capacity, key=lambda row: row.aum)
                if at_capacity else min(curve.results, key=lambda row: row.aum)
            )
            result.sharpe_at_capacity = _json_float(
                chosen.sharpe_ratio, 0.0,
            ) or 0.0
        else:
            result.curve_reason = (
                "the research context is too short to simulate an AUM curve"
            )
    except Exception:
        logger.exception("Capacity gauge: AUM curve simulation failed")
        result.curve_reason = (
            "the AUM-Sharpe curve simulation failed; the ADV-based capacity "
            "above is unaffected (see the server log)"
        )

    result.available = True
    result.reason = ""
    return result


# ── Config Update ──


@router.post("/config", response_model=ConfigUpdateResponse)
async def update_monitor_config(req: ConfigUpdateRequest):
    """Update risk limits. Validates bounds before applying."""
    risk = _get_risk()
    if not risk:
        raise HTTPException(status_code=503, detail="Risk monitor not available")

    updated = []
    limits = {}

    try:
        # Build the whole new limits object and swap it in with one attribute
        # assignment. The previous per-field writes left a window in which an
        # engine thread could observe a half-applied limit set (e.g. a new
        # position cap with the old drawdown cap), and a failure midway left
        # the monitor in that torn state.
        changes: dict[str, Any] = {}
        if req.max_position_pct is not None:
            # The dataclass field is `max_single_position_pct`. Writing
            # `max_position_pct` used to succeed silently on the instance and
            # be read by nothing, so the endpoint reported success while the
            # enforced limit never moved.
            changes["max_single_position_pct"] = req.max_position_pct
            updated.append("max_position_pct")

        if req.max_sector_pct is not None:
            changes["max_sector_pct"] = req.max_sector_pct
            updated.append("max_sector_pct")

        if req.max_drawdown_pct is not None:
            changes["max_drawdown_pct"] = req.max_drawdown_pct
            updated.append("max_drawdown_pct")

        if req.max_daily_loss_pct is not None:
            changes["max_daily_loss_pct"] = req.max_daily_loss_pct
            updated.append("max_daily_loss_pct")

        if changes:
            risk.limits = replace(risk.limits, **changes)

        # Read back current limits
        status = risk.get_status()
        limits = status.get("limits", {})

        # Log to EventBus
        bus = _get_bus()
        if bus:
            bus.publish("monitor.config_updated", {
                "updated": updated,
                "limits": limits,
            }, source="monitor")

    except Exception as e:
        logger.exception("Monitor config update failed")
        raise HTTPException(
            status_code=500,
            detail="Risk config update failed; see server logs",
            headers={"X-Error-Code": "monitor_config_update_failed"},
        ) from e

    return ConfigUpdateResponse(updated=updated, limits=limits)


# ── Kill Switch ──


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def monitor_kill_switch(req: KillSwitchRequest):
    """Activate or deactivate the Kill Switch."""
    risk = _get_risk()
    if not risk:
        raise HTTPException(status_code=503, detail="Risk monitor not available")

    try:
        if req.activate:
            risk.activate_kill_switch(reason=req.reason, actor="api/monitor")
            msg = f"Kill Switch ACTIVATED: {req.reason}"
        else:
            risk.deactivate_kill_switch(
                reason=req.reason, actor="api/monitor",
            )
            msg = "Kill Switch DEACTIVATED"

        # S21: tell the operator whether the emergency action is durable.
        try:
            status = risk.get_status()
            persisted = bool(status.get("persisted", False))
            persist_error = bool(status.get("persist_error", False))
        except Exception:
            persisted, persist_error = False, False

        # Broadcast via EventBus
        bus = _get_bus()
        if bus:
            bus.publish("risk.kill_switch", {
                "active": req.activate,
                "reason": req.reason,
                "persisted": persisted,
            }, source="monitor")

        return KillSwitchResponse(
            active=req.activate, message=msg, persisted=persisted,
            persist_error=persist_error,
        )

    except Exception as e:
        logger.exception("Monitor kill-switch update failed")
        raise HTTPException(
            status_code=500,
            detail="Kill switch update failed; see server logs",
            headers={"X-Error-Code": "monitor_kill_switch_failed"},
        ) from e


# ── Helpers ──


def _code_to_sector(code: str) -> str:
    """Map stock code to a simplified sector name."""
    if not code:
        return "Unknown"
    prefix = code[:3]
    sector_map = {
        "600": "沪市主板", "601": "沪市大盘", "603": "沪市中小",
        "688": "科创板", "000": "深市主板", "001": "深市中小",
        "002": "中小板", "003": "中小板", "300": "创业板",
        "301": "创业板",
    }
    return sector_map.get(prefix, "其他")
