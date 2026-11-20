from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "holdings_monitor.py"
spec = importlib.util.spec_from_file_location("holdings_monitor_test", SCRIPT)
monitor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = monitor
spec.loader.exec_module(monitor)


def test_missing_quote_is_excluded_from_totals_not_priced_at_cost():
    data = {
        "as_of": "2026-11-12",
        "cash_cny": 10.0,
        "funds": [{"value": 100.0}],
        "positions": [
            {"name": "quoted", "code": "sh510300", "shares": 100,
             "cost": 4.0, "confirmed": True},
            {"name": "missing", "code": "sh562970", "shares": 100,
             "cost": 2.0, "confirmed": True},
        ],
    }
    snapshot = monitor.build_snapshot(
        data,
        {"sh510300": {"price": 3.0, "pct": -1.0, "prev": 4.0}},
    )
    assert snapshot["securities_value"] == 300.0
    assert snapshot["securities_pnl"] == -100.0
    assert snapshot["grand_total"] == 410.0
    assert snapshot["securities_value_complete"] is False
    assert snapshot["grand_total_complete"] is False
    assert snapshot["missing_cost_cny"] == 200.0
    assert snapshot["missing_quotes"] == [
        {"name": "missing", "code": "sh562970"},
    ]
    missing = next(row for row in snapshot["positions"] if row["code"] == "sh562970")
    assert missing["price"] is None
    assert missing["value"] is None
    assert missing["pnl"] is None
    assert any("缺行情" in alert for alert in snapshot["alerts"])


def test_stale_fund_archive_is_flagged():
    snapshot = monitor.build_snapshot(
        {"as_of": "2026-10-29", "funds": [{"value": 100.0}],
         "positions": [], "cash_cny": 0.0},
        {},
    )
    assert any("基金金额仍是 2026-10-29" in alert for alert in snapshot["alerts"])
