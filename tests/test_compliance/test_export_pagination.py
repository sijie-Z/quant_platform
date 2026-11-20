"""A18 regression: compliance exports are not truncated at 1000 rows.

`export_trade_log`/`export_order_log` requested `limit=100_000`, but
`Store.get_trades`/`get_orders` clamp to 1000 for API safety (and the Python
date filter ran afterwards anyway). Any export whose range held more than
1000 rows silently omitted older records. The exporter now uses dedicated
keyset-paginated queries with no API clamp.
"""

from __future__ import annotations

import csv

import pytest

from quant_platform.compliance.exporter import ComplianceExporter
from quant_platform.core.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "export.db"))


def _seed_trades(store, n=1205):
    for i in range(n):
        day = (i % 28) + 1
        store.save_trade({
            "trade_id": f"t{i:05d}",
            "order_id": f"o{i:05d}",
            "code": "600519",
            "side": "buy",
            "quantity": 100,
            "price": 10 + (i % 7),
            "executed_at": f"2024-01-{day:02d}T10:{i % 60:02d}:00",
        })


def _seed_orders(store, n=1005):
    for i in range(n):
        day = (i % 28) + 1
        store.save_order({
            "order_id": f"o{i:05d}",
            "code": "600519",
            "side": "buy",
            "quantity": 100,
            "price": 10 + (i % 7),
            "status": "filled",
            "created_at": f"2024-02-{day:02d}T10:{i % 60:02d}:00",
        })


def test_trade_export_spans_more_than_1000_rows(store, tmp_path):
    _seed_trades(store, 1205)
    exporter = ComplianceExporter(store, output_dir=str(tmp_path / "out"))

    result = exporter.export_trade_log(
        start="2024-01-01", end="2024-01-31", format="csv",
    )

    assert result.n_records == 1205, "older trades were truncated"
    with open(result.file_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1206  # header + every trade
    trade_ids = {row[0] for row in rows[1:]}
    assert "t00000" in trade_ids and "t01204" in trade_ids


def test_order_export_spans_more_than_1000_rows(store, tmp_path):
    _seed_orders(store, 1005)
    exporter = ComplianceExporter(store, output_dir=str(tmp_path / "out"))

    result = exporter.export_order_log(
        start="2024-02-01", end="2024-02-28", format="csv",
    )

    assert result.n_records == 1005, "older orders were truncated"
    with open(result.file_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1006


def test_export_pagination_stays_tenant_scoped(store, tmp_path):
    for i in range(1100):
        store.save_trade({
            "trade_id": f"a{i:05d}", "order_id": f"oa{i:05d}",
            "tenant_id": "fund_a", "code": "600519", "side": "buy",
            "quantity": 100, "price": 10,
            "executed_at": "2024-03-01T10:00:00",
        })
    for i in range(20):
        store.save_trade({
            "trade_id": f"b{i:05d}", "order_id": f"ob{i:05d}",
            "tenant_id": "fund_b", "code": "000001", "side": "buy",
            "quantity": 100, "price": 10,
            "executed_at": "2024-03-01T10:00:00",
        })

    rows = store.get_trades_for_export(
        tenant_id="fund_a", start_date="2024-03-01", end_date="2024-03-31",
    )

    assert len(rows) == 1100
    assert {r["tenant_id"] for r in rows} == {"fund_a"}


def test_export_window_is_widened_for_timezone_offsets(store, tmp_path):
    """A +08:00 print on the end date must not be dropped by the SQL bound."""
    store.save_trade({
        "trade_id": "aware", "order_id": "o1", "code": "600519",
        "side": "buy", "quantity": 100, "price": 10,
        "executed_at": "2024-03-31T23:30:00+08:00",
    })
    exporter = ComplianceExporter(store, output_dir=str(tmp_path / "out"))

    result = exporter.export_trade_log(
        start="2024-03-01", end="2024-03-31", format="csv",
    )

    assert result.n_records == 1
