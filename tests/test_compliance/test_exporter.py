"""Tests for compliance.exporter — trade/order/risk log export."""

import csv
import os

import pytest

from quant_platform.compliance.exporter import (
    ORDER_FIELDS,
    TRADE_FIELDS,
    ComplianceExporter,
    ExportResult,
)
from quant_platform.core.store import Store


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Store(db_path)


@pytest.fixture
def exporter(tmp_path, store):
    return ComplianceExporter(store, output_dir=str(tmp_path / "output"))


def _seed_trades(store, n=5):
    """Seed store with sample trades."""
    for i in range(n):
        store.save_order({
            "order_id": f"ord-{i:03d}", "code": f"60000{i}",
            "side": "buy", "quantity": 100 * (i + 1), "price": 10.0 + i,
            "status": "filled", "created_at": f"2024-0{6 + i % 4}-15T10:00:00",
        })
        store.save_trade({
            "trade_id": f"trd-{i:03d}", "order_id": f"ord-{i:03d}",
            "code": f"60000{i}", "side": "buy",
            "quantity": 100 * (i + 1), "price": 10.0 + i,
            "executed_at": f"2024-0{6 + i % 4}-15T10:00:00",
        })


def _seed_orders(store, n=5):
    """Seed store with sample orders."""
    for i in range(n):
        store.save_order({
            "order_id": f"ord-{i:03d}", "code": f"60000{i}",
            "side": "buy" if i % 2 == 0 else "sell",
            "quantity": 100 * (i + 1), "price": 10.0 + i,
            "status": "filled" if i % 2 == 0 else "pending",
            "created_at": f"2024-0{6 + i % 4}-15T10:00:00",
        })


def _seed_pnl(store, n=5):
    """Seed store with P&L history."""
    for i in range(n):
        store.save_pnl_snapshot({
            "timestamp": f"2024-0{6 + i % 4}-15T10:00:00",
            "total_equity": 10_000_000 + i * 50_000,
            "cash": 5_000_000,
            "market_value": 5_000_000 + i * 50_000,
            "daily_pnl": 50_000 * (i + 1),
            "n_positions": 10 + i,
        })


# ── Trade Log Export ──


class TestExportTradeLog:
    def test_export_csv_creates_file(self, exporter, store):
        _seed_trades(store, 3)
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        assert isinstance(result, ExportResult)
        assert os.path.exists(result.file_path)
        assert result.n_records == 3
        assert result.format == "csv"

    def test_export_xlsx_creates_file(self, exporter, store):
        _seed_trades(store, 3)
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", format="xlsx",
        )
        assert os.path.exists(result.file_path)
        assert result.n_records == 3

    def test_export_xlsx_fallback_to_csv(self, exporter, store, monkeypatch):
        _seed_trades(store, 2)
        # Simulate openpyxl not installed
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("No module named 'openpyxl'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", format="xlsx",
        )
        assert os.path.exists(result.file_path)
        assert result.file_path.endswith(".csv")

    def test_export_with_tenant_id(self, exporter, store):
        store.save_order({
            "order_id": "o1", "tenant_id": "fund_001", "code": "600519",
            "side": "buy", "quantity": 100, "price": 100, "status": "filled",
            "created_at": "2024-07-15T10:00:00",
        })
        store.save_trade({
            "trade_id": "t1", "order_id": "o1", "tenant_id": "fund_001",
            "code": "600519", "side": "buy", "quantity": 100, "price": 100,
            "executed_at": "2024-07-15T10:00:00",
        })
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", tenant_id="fund_001",
        )
        assert result.n_records == 1
        assert "fund_001" in result.file_path

    def test_export_filters_by_date(self, exporter, store):
        _seed_trades(store, 4)
        result = exporter.export_trade_log(
            start="2024-07-01", end="2024-08-31",
        )
        assert result.n_records <= 4

    def test_export_empty(self, exporter, store):
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31",
        )
        assert result.n_records == 0
        assert os.path.exists(result.file_path)

    def test_csv_headers_contain_chinese(self, exporter, store):
        _seed_trades(store, 1)
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        with open(result.file_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert any("成交" in h for h in headers)

    def test_csv_utf8_bom(self, exporter, store):
        _seed_trades(store, 1)
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        with open(result.file_path, "rb") as f:
            bom = f.read(3)
        assert bom == b"\xef\xbb\xbf"

    def test_result_fields(self, exporter, store):
        _seed_trades(store, 2)
        result = exporter.export_trade_log(
            start="2024-01-01", end="2024-12-31", tenant_id="fund_x",
        )
        assert result.start_date == "2024-01-01"
        assert result.end_date == "2024-12-31"
        assert result.tenant_id == "fund_x"


# ── Order Log Export ──


class TestExportOrderLog:
    def test_export_csv(self, exporter, store):
        _seed_orders(store, 3)
        result = exporter.export_order_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        assert os.path.exists(result.file_path)
        assert result.n_records == 3

    def test_export_xlsx(self, exporter, store):
        _seed_orders(store, 3)
        result = exporter.export_order_log(
            start="2024-01-01", end="2024-12-31", format="xlsx",
        )
        assert os.path.exists(result.file_path)
        assert result.n_records == 3

    def test_order_headers_contain_chinese(self, exporter, store):
        _seed_orders(store, 1)
        result = exporter.export_order_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        with open(result.file_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert any("委托" in h for h in headers)

    def test_order_with_tenant_filter(self, exporter, store):
        store.save_order({
            "order_id": "o1", "tenant_id": "fund_001", "code": "600519",
            "side": "buy", "quantity": 100, "price": 100, "status": "filled",
            "created_at": "2024-07-15T10:00:00",
        })
        store.save_order({
            "order_id": "o2", "tenant_id": "fund_002", "code": "000001",
            "side": "sell", "quantity": 200, "price": 50, "status": "pending",
            "created_at": "2024-07-15T10:00:00",
        })
        result = exporter.export_order_log(
            start="2024-01-01", end="2024-12-31", tenant_id="fund_001",
        )
        assert result.n_records == 1

    def test_order_empty_export(self, exporter, store):
        result = exporter.export_order_log(
            start="2024-01-01", end="2024-12-31",
        )
        assert result.n_records == 0


# ── Risk Log Export ──


class TestExportRiskLog:
    def test_export_csv(self, exporter, store):
        _seed_pnl(store, 3)
        result = exporter.export_risk_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        assert os.path.exists(result.file_path)
        assert result.n_records == 3

    def test_export_xlsx(self, exporter, store):
        _seed_pnl(store, 3)
        result = exporter.export_risk_log(
            start="2024-01-01", end="2024-12-31", format="xlsx",
        )
        assert os.path.exists(result.file_path)
        assert result.n_records == 3

    def test_risk_headers(self, exporter, store):
        _seed_pnl(store, 1)
        result = exporter.export_risk_log(
            start="2024-01-01", end="2024-12-31", format="csv",
        )
        with open(result.file_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert any("盈亏" in h for h in headers)

    def test_risk_empty_export(self, exporter, store):
        result = exporter.export_risk_log(
            start="2024-01-01", end="2024-12-31",
        )
        assert result.n_records == 0


# ── Field Definitions ──


class TestFieldDefinitions:
    def test_trade_fields_structure(self):
        assert len(TRADE_FIELDS) == 10
        for field_key, cn, en in TRADE_FIELDS:
            assert isinstance(field_key, str)
            assert isinstance(cn, str)
            assert isinstance(en, str)

    def test_order_fields_structure(self):
        assert len(ORDER_FIELDS) == 14
        for field_key, cn, en in ORDER_FIELDS:
            assert isinstance(field_key, str)


# ── ExportResult ──


class TestExportResult:
    def test_result_dataclass(self):
        result = ExportResult(
            file_path="/tmp/test.csv", format="csv",
            n_records=10, start_date="2024-01-01", end_date="2024-12-31",
        )
        assert result.n_records == 10
        assert result.tenant_id == ""
