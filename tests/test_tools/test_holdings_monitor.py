from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

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


def test_tencent_quote_parses_and_fails_soft(monkeypatch):
    import io

    class FakeResponse:
        def __init__(self, payload: bytes):
            self.payload = payload

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(_request, timeout=10):
        return FakeResponse(
            "v_sh562970=\"1~光伏ETF易方达~562970~0.756~0.747~0.750~225133~\";".encode("gbk")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    quote = monitor.tencent_quote("sh562970")
    assert quote["price"] == 0.756
    assert quote["prev"] == 0.747
    assert quote["pct"] == pytest.approx((0.756 / 0.747 - 1) * 100, rel=1e-6)

    def broken_urlopen(_request, timeout=10):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", broken_urlopen)
    assert monitor.tencent_quote("sh562970") is None
