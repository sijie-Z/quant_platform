"""Executable QMT/live safety drills using production code and fault injection.

These are simulated broker drills, not a real miniQMT certification. They are
intended to be rerun before any real-money enablement and after execution
changes. No real broker connection or order is made.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "quant_core" / "src"))

import pandas as pd  # noqa: E402
from quant_platform.backtest.cost_model import CostModel  # noqa: E402
from quant_platform.backtest.engine import BacktestEngine  # noqa: E402
from quant_platform.core.store import Store  # noqa: E402
from quant_platform.execution.models import OrderStatus  # noqa: E402
from quant_platform.portfolio.constraints import PortfolioConstraints  # noqa: E402
from quant_platform.risk.circuit_breaker import RiskMonitor  # noqa: E402
from quant_platform.trading.broker import (  # noqa: E402
    Order,
    OrderSide,
    OrderType,
    Position,
    QMTBroker,
)
from quant_platform.trading.engine import CycleResult, LiveTradingEngine  # noqa: E402
from quant_platform.trading.live_engine import working_order_codes  # noqa: E402

WORK = ROOT / ".scratch" / "qmt_drill"


def connected_broker() -> QMTBroker:
    broker = QMTBroker(account="safety-drill")
    broker._HAS_XTQUANT = True
    broker._connected = True
    broker._trader = MagicMock()
    broker._account = MagicMock()
    return broker


def xt_order(order_id: str, *, status: int = 50, strategy_name: str = "",
             traded_volume: int = 0, traded_price: float = 0.0):
    return SimpleNamespace(
        order_id=order_id, stock_code="600519.SH", order_type=23,
        price_type=11, order_volume=100, price=10.0,
        traded_volume=traded_volume, traded_price=traded_price,
        order_status=status, status_msg="", order_time=0,
        strategy_name=strategy_name,
    )


def xt_trade(order_id: str, trade_id: str, volume: int, price: float):
    return SimpleNamespace(
        order_id=order_id, traded_id=trade_id,
        traded_volume=volume, traded_price=price,
    )


def drill_stale_quotes() -> dict:
    broker = connected_broker()
    now = dt.datetime.now().timestamp()
    old_age = broker._qmt_quote_age_seconds((now - 3600) * 1000)
    fresh_age = broker._qmt_quote_age_seconds(now * 1000)
    utc_instant = dt.datetime(2026, 10, 26, 2, 0,
                              tzinfo=dt.UTC)
    shanghai_age = broker._qmt_quote_age_seconds(
        "20261026100000", now=utc_instant,
    )
    assert old_age > 120 and abs(fresh_age) < 120
    assert abs(shanghai_age) < 1e-6
    assert broker._qmt_quote_age_seconds(None) is None
    return {"old_age_seconds": old_age, "fresh_age_seconds": fresh_age,
            "shanghai_vs_utc_age": shanghai_age}


def drill_fill_dedup_vwap() -> dict:
    broker = connected_broker()
    order = Order(code="600519", side=OrderSide.BUY, quantity=200,
                  price=10.0, status=OrderStatus.SUBMITTED,
                  broker_order_id="b-vwap")
    broker._orders = [order]
    broker._on_order_update(xt_order(
        "b-vwap", status=55, traded_volume=100, traded_price=10.0,
    ))
    broker._on_trade(xt_trade("b-vwap", "t1", 100, 10.0))
    broker._on_trade(xt_trade("b-vwap", "t2", 100, 11.0))
    broker._on_trade(xt_trade("b-vwap", "t2", 100, 11.0))  # duplicate
    assert order.filled_quantity == 200
    assert order.filled_price == 10.5
    assert order.status == OrderStatus.FILLED
    return {"filled_quantity": order.filled_quantity,
            "filled_price_vwap": order.filled_price,
            "duplicate_trade_ignored": True}


def drill_partial_cancellation() -> dict:
    broker = connected_broker()
    order = Order(code="600519", side=OrderSide.BUY, quantity=200,
                  price=10.0, status=OrderStatus.PARTIAL,
                  filled_quantity=100, filled_price=10.0,
                  broker_order_id="b-cancel")
    broker._orders = [order]
    broker._on_order_update(xt_order(
        "b-cancel", status=54, traded_volume=100, traded_price=10.0,
    ))
    broker._on_trade(xt_trade("b-cancel", "late-fill", 50, 10.0))
    # The ambiguous incremental print must not be guessed into the book.
    assert order.filled_quantity == 100
    assert order.status == OrderStatus.CANCELLED
    # A larger cumulative broker snapshot is authoritative and monotonic.
    broker._on_order_update(xt_order(
        "b-cancel", status=54, traded_volume=150, traded_price=10.0,
    ))
    assert order.filled_quantity == 150
    assert order.status == OrderStatus.CANCELLED
    return {"filled_quantity": order.filled_quantity,
            "terminal_status": order.status.value,
            "ambiguous_incremental_print_not_guessed": True}


def drill_submission_timeout_race() -> dict:
    broker = connected_broker()
    order = Order(code="600519", side=OrderSide.BUY,
                  order_type=OrderType.LIMIT, quantity=100, price=10.0)
    snapshot = xt_order(
        "b-race", status=50,
        strategy_name=f"quant_platform:{order.order_id}",
    )
    broker._trader.query_stock_orders.return_value = [snapshot]
    broker._trader.query_stock_trades.return_value = []

    def accept_then_lose_response(**_kwargs):
        broker._resync_from_broker()
        raise TimeoutError("response lost after acceptance")

    broker._trader.order_stock.side_effect = accept_then_lose_response
    result = broker.place_order(order)
    assert len(broker._orders) == 1
    assert result is broker._orders[0]
    assert result.broker_order_id == "b-race"
    assert result.status == OrderStatus.SUBMITTED
    assert broker._resync_from_broker() is True
    assert len(broker._orders) == 1
    assert all(item.status != OrderStatus.PENDING for item in broker._orders)
    working = working_order_codes(broker.get_orders_authoritative())
    assert working == {"600519.SH"}
    return {"tracked_orders": len(broker._orders),
            "recovered_broker_id": result.broker_order_id,
            "working_codes": sorted(working)}


class DelayedSellBroker:
    def __init__(self):
        self._position = Position(
            code="600519", quantity=100, available=100, market_value=1000.0,
        )
        self._orders = []
        self.placed = []

    def get_positions(self):
        return [self._position]

    def get_orders_authoritative(self):
        return list(self._orders)

    def place_order(self, order):
        order.status = OrderStatus.SUBMITTED
        self._orders.append(order)
        self.placed.append(order)
        return order


def drill_kill_switch_delayed_sell() -> dict:
    broker = DelayedSellBroker()
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(WORK / "kill_switch.db")),
        risk_monitor=RiskMonitor(),
    )
    cycle = CycleResult(cycle_id=1, timestamp="2026-10-27T10:00:00")
    engine._deleverage(cycle)
    engine._deleverage(cycle)
    assert len(broker.placed) == 1
    assert broker.placed[0].quantity == 100
    assert len(cycle.orders) == 1
    return {"cycles": 2, "orders_submitted": len(broker.placed),
            "quantity": broker.placed[0].quantity}


def drill_account_fail_closed() -> dict:
    broker = connected_broker()
    broker._positions["600519"] = MagicMock()
    broker._trader.query_stock_positions.side_effect = RuntimeError("positions down")
    assert broker.get_positions() is None
    broker._trader.query_stock_asset.side_effect = RuntimeError("asset down")
    try:
        broker.get_account()
    except QMTBroker.BrokerConnectionError:
        pass
    else:
        raise AssertionError("account query failure did not fail closed")
    engine = LiveTradingEngine(
        broker=broker, store=Store(str(WORK / "account_failure.db")),
        risk_monitor=RiskMonitor(),
    )
    positions, known = engine._query_positions()
    assert positions is None and known is False
    broker._trader.query_stock_orders.side_effect = RuntimeError("orders down")
    assert broker.get_orders_authoritative() is None
    return {"positions_known": known, "account_error": "BrokerConnectionError",
            "authoritative_orders": None}


def drill_cost_and_cash_funding() -> dict:
    fallback = CostModel(
        commission=0.0, stamp_tax=0.0, transfer_fee=0.0,
        min_commission=0.0, slippage=0.0007, slippage_model="impact",
    )
    invalid_cost = fallback.compute_costs(
        1_000_000.0, daily_volume=0.0, volatility=0.02,
    )
    assert invalid_cost == 700.0

    class RecordingCostModel(CostModel):
        def __init__(self):
            super().__init__(commission=0.01, stamp_tax=0.0, slippage=0.0,
                             min_commission=0.0, transfer_fee=0.0)
            self.trades = []

        def compute_costs(self, turnover, **kwargs):
            self.trades.append((float(kwargs["buy_turnover"].sum()),
                                float(kwargs["sell_turnover"].sum())))
            return super().compute_costs(turnover, **kwargs)

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(0.0, index=dates, columns=["A"])
    returns.loc[dates[1], "A"] = 0.10
    cost_model = RecordingCostModel()
    engine = BacktestEngine(
        initial_capital=1000.0, rebalance_frequency="monthly",
        cost_model=cost_model,
        constraints=PortfolioConstraints(max_turnover=1.0),
    )
    engine.weights_history = {
        dates[0]: pd.Series({"A": 1.0}),
        dates[2]: pd.Series({"A": 0.0}),
    }
    engine._simulate_pnl(returns)
    assert math.isclose(engine.weights_history[dates[0]]["A"], 0.99,
                        abs_tol=1e-12)
    assert math.isclose(engine.portfolio_values.loc[dates[1]], 1089.10,
                        abs_tol=1e-9)
    assert math.isclose(engine.portfolio_values.iloc[-1], 1078.21,
                        abs_tol=1e-9)
    assert cost_model.trades == [(1000.0, 0.0), (990.0, 0.0), (0.0, 1089.0)]
    assert engine.n_funding_scaled_days == 0
    return {"invalid_adv_cost": invalid_cost,
            "initial_weight": engine.weights_history[dates[0]]["A"],
            "final_nav": engine.portfolio_values.iloc[-1],
            "cost_calls": cost_model.trades}


DRILLS = {
    "stale_quotes": drill_stale_quotes,
    "fill_dedup_vwap": drill_fill_dedup_vwap,
    "partial_cancellation": drill_partial_cancellation,
    "submission_timeout_race": drill_submission_timeout_race,
    "kill_switch_delayed_sell": drill_kill_switch_delayed_sell,
    "account_fail_closed": drill_account_fail_closed,
    "cost_and_cash_funding": drill_cost_and_cash_funding,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs" / "research" /
                        "QMT_SAFETY_DRILL_LATEST.json")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for name, function in DRILLS.items():
        try:
            details = function()
            results.append({"name": name, "status": "PASS", "details": details})
            print(f"PASS {name}: {details}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    payload = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "simulated fault injection; no real broker connection",
        "xtquant_installed": importlib.util.find_spec("xtquant") is not None,
        "real_miniqmt_certified": False,
        "results": results,
        "all_passed": all(item["status"] == "PASS" for item in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"report: {args.out}", flush=True)
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
