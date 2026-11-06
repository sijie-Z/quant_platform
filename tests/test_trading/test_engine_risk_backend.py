"""D13: the engine's default risk backend must be the shared RiskMonitor.

The engine defaulted to `RealTimeRiskEngine`, whose compatibility
`update_portfolio_state` ignored `daily_pnl`/`positions`/`sector_weights` and
whose `pre_trade_check` only inspected the new order. A direct/embedded
engine (the API injects RiskMonitor explicitly) could therefore hold 4% of a
name and still be approved for another 4% against a 5% limit, and a
mark-to-market drawdown never armed the kill switch. The default is now the
same process-wide singleton the API kill switch mutates; injection remains
supported.
"""

from __future__ import annotations

import pytest

from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor, get_risk_monitor
from quant_platform.trading.broker import SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine


def test_engine_defaults_to_the_process_wide_risk_monitor(tmp_path):
    engine = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=1_000_000),
        store=Store(str(tmp_path / "engine.db")),
    )

    assert isinstance(engine._risk, RiskMonitor), (
        "the default must be RiskMonitor, not RealTimeRiskEngine"
    )
    assert engine._risk is get_risk_monitor(), (
        "the default must be the same singleton the API kill switch mutates"
    )


def test_default_backend_tracks_daily_pnl_and_sector_exposure(tmp_path):
    engine = LiveTradingEngine(
        broker=SimulatedBroker(initial_cash=1_000_000),
        store=Store(str(tmp_path / "engine.db")),
    )
    monitor = engine._risk
    # Snapshot the singleton so this test leaves no state behind.
    saved = {
        "portfolio_value": monitor.portfolio_value,
        "daily_pnl": monitor.daily_pnl,
        "positions": dict(monitor.positions),
        "sector_weights": dict(monitor.sector_weights),
        "peak_value": monitor.peak_value,
        "risk_level": monitor.risk_level,
        "kill_switch_active": monitor.kill_switch_active,
        "breaches": list(monitor.breaches),
        "order_timestamps": list(monitor.order_timestamps),
        "_last_state_breach": monitor._last_state_breach,
        "_active_breach_signatures": set(monitor._active_breach_signatures),
    }
    try:
        monitor.update_portfolio_state(
            portfolio_value=1_000_000,
            daily_pnl=-50_000,
            positions={},
            sector_weights={},
        )
        # RealTimeRiskEngine's compatibility shim dropped daily_pnl entirely.
        assert monitor.daily_pnl == pytest.approx(-50_000)

        monitor.update_portfolio_state(
            portfolio_value=1_000_000,
            daily_pnl=0,
            positions={"600519": {"value": 40_000, "weight": 0.04,
                                  "sector": "Liquor"}},
            sector_weights={"Liquor": 0.04},
        )
        approved, breaches = monitor.check_pre_trade({
            "ticker": "600519", "side": "buy", "quantity": 400,
            "price": 100.0,
        })
        assert not approved, "a second 4% in the same name must breach 5%"
        assert any(b.message for b in breaches)

        # Eight 4% buys in one sector must not pass a 30% sector cap.
        monitor.update_portfolio_state(
            portfolio_value=1_000_000,
            daily_pnl=0,
            positions={}, sector_weights={},
        )
        approved_count = 0
        for index in range(8):
            order = {
                "ticker": f"S{index}", "side": "buy",
                "quantity": 400, "price": 100.0, "sector": "Tech",
            }
            ok, _ = monitor.check_pre_trade(order)
            if ok:
                approved_count += 1
                monitor.project_order(order)
        assert approved_count < 8, (
            "the sector cap must stop the batch before 32% of equity"
        )
    finally:
        monitor.portfolio_value = saved["portfolio_value"]
        monitor.daily_pnl = saved["daily_pnl"]
        monitor.positions = saved["positions"]
        monitor.sector_weights = saved["sector_weights"]
        monitor.peak_value = saved["peak_value"]
        monitor.risk_level = saved["risk_level"]
        monitor.kill_switch_active = saved["kill_switch_active"]
        monitor.breaches = saved["breaches"]
        monitor.order_timestamps = saved["order_timestamps"]
        monitor._last_state_breach = saved["_last_state_breach"]
        monitor._active_breach_signatures = saved["_active_breach_signatures"]
