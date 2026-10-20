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
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/monitor")


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
    n_orders: int = 0
    mean_is_bps: float = 0.0
    mean_delay_bps: float = 0.0
    mean_impact_bps: float = 0.0
    mean_timing_bps: float = 0.0
    mean_arrival_bps: float = 0.0
    median_is_bps: float = 0.0
    daily_trend: list[dict] = Field(default_factory=list, description="Daily IS trend [{date, is_bps}]")
    cost_breakdown: list[dict] = Field(default_factory=list, description="[{date, delay, impact, timing}]")
    by_ticker: dict[str, float] = Field(default_factory=dict, description="IS by ticker")


class FactorStatusResponse(BaseModel):
    available: bool = Field(False, description="Whether rolling IC data is populated")
    reason: str = Field("", description="Why factor status is unavailable, when it is")
    factors: list[dict] = Field(default_factory=list, description="Factor IC stats [{name, ic, icir, trend, alert}]")
    rolling_ic: dict[str, list[float]] = Field(default_factory=dict, description="Rolling IC series per factor")
    ic_dates: list[str] = Field(default_factory=list, description="Dates for rolling IC")
    attribution: list[dict] = Field(default_factory=list, description="[{factor, contribution_bps}]")
    decay_alerts: list[dict] = Field(default_factory=list, description="Factors with IC decay")
    disabled_factors: list[str] = Field(default_factory=list, description="Auto-decay disabled factors")


class CapacityGaugeResponse(BaseModel):
    available: bool = Field(False, description="Whether the capacity curve is populated")
    reason: str = Field("", description="Why capacity is unavailable, when it is")
    current_aum: float = 0.0
    capacity_aum: float = 0.0
    usage_pct: float = 0.0
    participation_rate: float = 0.0
    sharpe_at_capacity: float = 0.0
    aum_curve: list[dict] = Field(default_factory=list, description="[{aum, sharpe, return}]")


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
    """Aggregate TCA data from recent trades."""
    return await run_in_threadpool(_compute_tca_summary)


def _compute_tca_summary() -> TCASummaryResponse:
    result = TCASummaryResponse()
    result.reason = "no trades recorded"

    store = _get_store()
    if not store:
        return result

    try:
        trades = store.get_trades(limit=500)
        if not trades:
            return result

        # Group trades by date
        daily: dict[str, list[dict]] = {}
        for t in trades:
            ts = t.get("executed_at", "")
            date = ts[:10] if ts else ""
            if date:
                daily.setdefault(date, []).append(t)

        result.n_orders = len(trades)
        result.available = False
        result.reason = (
            "stored trades carry no arrival price, so implementation shortfall "
            "cannot be computed; execution/tca.py needs quote data"
        )

    except Exception:
        logger.exception("TCA summary computation failed")

    return result


# ── Factor Status ──


@router.get("/factor-status", response_model=FactorStatusResponse)
async def get_factor_status():
    """Factor IC monitoring, attribution, and decay alerts."""
    return await run_in_threadpool(_compute_factor_status)


def _compute_factor_status() -> FactorStatusResponse:
    result = FactorStatusResponse()
    result.reason = (
        "no rolling IC series is cached; POST /api/ic-monitor/compute to "
        "compute factor statistics"
    )

    store = _get_store()
    if not store:
        return result

    try:
        # Get signals for factor analysis
        signals = store.get_signals(limit=200)
        if signals:
            # Analyze factor values from signals
            factor_ics: dict[str, list[float]] = {}
            for sig in signals:
                fv = sig.get("factor_values", {})
                if isinstance(fv, str):
                    import json
                    try:
                        fv = json.loads(fv)
                    except Exception:
                        fv = {}
                for factor_name, value in fv.items():
                    if isinstance(value, (int, float)):
                        factor_ics.setdefault(factor_name, []).append(float(value))

    except Exception:
        logger.exception("Factor status computation failed")

    return result


# ── Capacity Gauge ──


@router.get("/capacity-gauge", response_model=CapacityGaugeResponse)
async def get_capacity_gauge():
    """Strategy capacity estimation and usage."""
    return await run_in_threadpool(_compute_capacity_gauge)


def _compute_capacity_gauge() -> CapacityGaugeResponse:
    result = CapacityGaugeResponse()
    result.reason = (
        "capacity curve is not precomputed; run the backtest capacity "
        "analysis to populate aum_curve/capacity_aum"
    )

    store = _get_store()
    if not store:
        return result

    try:
        positions = store.get_positions()
        pnl = store.get_pnl_history(days=30)

        # Current AUM from latest P&L
        if pnl:
            latest = pnl[-1]
            result.current_aum = latest.get("total_equity", 0)
        elif positions:
            result.current_aum = sum(p.get("market_value", 0) for p in positions)

    except Exception:
        logger.exception("Capacity gauge computation failed")

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
