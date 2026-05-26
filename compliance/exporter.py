"""Compliance exporter — trade/order/risk log export for regulatory reporting.

Exports trading records in formats required by Chinese private fund
regulations (私募基金交易记录保存格式). Supports Excel (.xlsx) and CSV.

Fields follow CSRC naming conventions with English aliases for
cross-border reporting.

Usage:
    from quant_platform.compliance.exporter import ComplianceExporter

    exporter = ComplianceExporter(store)
    exporter.export_trade_log(
        start="2024-01-01", end="2024-12-31",
        tenant_id="fund_001", format="xlsx",
    )
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# ── Field mappings: Chinese regulatory name ↔ English alias ──

TRADE_FIELDS = [
    ("trade_id", "成交编号", "Trade ID"),
    ("order_id", "委托编号", "Order ID"),
    ("tenant_id", "基金编号", "Fund ID"),
    ("code", "证券代码", "Security Code"),
    ("side", "买卖方向", "Side"),
    ("quantity", "成交数量", "Quantity"),
    ("price", "成交价格", "Price"),
    ("commission", "佣金", "Commission"),
    ("tax", "印花税", "Stamp Tax"),
    ("executed_at", "成交时间", "Execution Time"),
]

ORDER_FIELDS = [
    ("order_id", "委托编号", "Order ID"),
    ("tenant_id", "基金编号", "Fund ID"),
    ("code", "证券代码", "Security Code"),
    ("side", "买卖方向", "Side"),
    ("order_type", "委托类型", "Order Type"),
    ("quantity", "委托数量", "Quantity"),
    ("price", "委托价格", "Price"),
    ("filled_quantity", "已成交数量", "Filled Qty"),
    ("filled_price", "成交均价", "Avg Fill Price"),
    ("status", "委托状态", "Status"),
    ("commission", "佣金", "Commission"),
    ("tax", "印花税", "Stamp Tax"),
    ("created_at", "创建时间", "Created At"),
    ("updated_at", "更新时间", "Updated At"),
]


@dataclass
class ExportResult:
    """Result of an export operation."""
    file_path: str
    format: str
    n_records: int
    start_date: str
    end_date: str
    tenant_id: str = ""


class ComplianceExporter:
    """Export trading records for regulatory compliance.

    Reads from Store tables and writes Excel/CSV files with
    Chinese regulatory field names.

    Args:
        store: Store instance for data access.
        output_dir: Directory for exported files (default: ./compliance_output).
    """

    def __init__(self, store: Any, output_dir: str = "compliance_output"):
        self._store = store
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_trade_log(
        self,
        start: str,
        end: str,
        tenant_id: str | None = None,
        format: str = "xlsx",
    ) -> ExportResult:
        """Export trade log for a date range.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            tenant_id: Optional tenant filter.
            format: "xlsx" or "csv".

        Returns:
            ExportResult with file path and record count.
        """
        trades = self._store.get_trades(
            tenant_id=tenant_id or "", limit=100_000
        )
        filtered = self._filter_by_date(trades, "executed_at", start, end)

        filename = f"trade_log_{start}_{end}"
        if tenant_id:
            filename += f"_{tenant_id}"

        if format == "xlsx":
            path = self._write_xlsx(filtered, TRADE_FIELDS, filename)
        else:
            path = self._write_csv(filtered, TRADE_FIELDS, filename)

        logger.info("Exported %d trades to %s", len(filtered), path)
        return ExportResult(
            file_path=str(path),
            format=format,
            n_records=len(filtered),
            start_date=start,
            end_date=end,
            tenant_id=tenant_id or "",
        )

    def export_order_log(
        self,
        start: str,
        end: str,
        tenant_id: str | None = None,
        format: str = "csv",
    ) -> ExportResult:
        """Export order log for a date range.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            tenant_id: Optional tenant filter.
            format: "xlsx" or "csv".

        Returns:
            ExportResult with file path and record count.
        """
        orders = self._store.get_orders(
            tenant_id=tenant_id or "", limit=100_000
        )
        filtered = self._filter_by_date(orders, "created_at", start, end)

        filename = f"order_log_{start}_{end}"
        if tenant_id:
            filename += f"_{tenant_id}"

        if format == "xlsx":
            path = self._write_xlsx(filtered, ORDER_FIELDS, filename)
        else:
            path = self._write_csv(filtered, ORDER_FIELDS, filename)

        logger.info("Exported %d orders to %s", len(filtered), path)
        return ExportResult(
            file_path=str(path),
            format=format,
            n_records=len(filtered),
            start_date=start,
            end_date=end,
            tenant_id=tenant_id or "",
        )

    def export_risk_log(
        self,
        start: str,
        end: str,
        tenant_id: str | None = None,
        format: str = "csv",
    ) -> ExportResult:
        """Export P&L history as risk log.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            tenant_id: Optional tenant filter (unused for P&L).
            format: "xlsx" or "csv".

        Returns:
            ExportResult with file path and record count.
        """
        pnl = self._store.get_pnl_history(days=365 * 5)
        filtered = self._filter_by_date(pnl, "timestamp", start, end)

        risk_fields = [
            ("timestamp", "快照时间", "Timestamp"),
            ("total_equity", "总权益", "Total Equity"),
            ("cash", "现金", "Cash"),
            ("market_value", "持仓市值", "Market Value"),
            ("daily_pnl", "当日盈亏", "Daily P&L"),
            ("daily_pnl_pct", "当日收益率", "Daily Return %"),
            ("cumulative_pnl", "累计盈亏", "Cumulative P&L"),
            ("n_positions", "持仓数量", "Positions"),
            ("max_drawdown", "最大回撤", "Max Drawdown"),
            ("sharpe_ratio", "夏普比率", "Sharpe Ratio"),
        ]

        filename = f"risk_log_{start}_{end}"
        if tenant_id:
            filename += f"_{tenant_id}"

        if format == "xlsx":
            path = self._write_xlsx(filtered, risk_fields, filename)
        else:
            path = self._write_csv(filtered, risk_fields, filename)

        logger.info("Exported %d risk records to %s", len(filtered), path)
        return ExportResult(
            file_path=str(path),
            format=format,
            n_records=len(filtered),
            start_date=start,
            end_date=end,
            tenant_id=tenant_id or "",
        )

    def _filter_by_date(
        self, records: list[dict], date_field: str, start: str, end: str
    ) -> list[dict]:
        """Filter records by date range on a given field."""
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end + "T23:59:59")

        filtered = []
        for r in records:
            raw = r.get(date_field, "")
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(str(raw))
                if start_dt <= dt <= end_dt:
                    filtered.append(r)
            except (ValueError, TypeError):
                continue
        return filtered

    def _write_xlsx(
        self, records: list[dict], fields: list[tuple], filename: str
    ) -> Path:
        """Write records to Excel file."""
        try:
            from openpyxl import Workbook
        except ImportError:
            logger.warning("openpyxl not installed, falling back to CSV")
            return self._write_csv(records, fields, filename)

        wb = Workbook()
        ws = wb.active
        ws.title = "交易记录"

        # Header row: Chinese name + English alias
        headers = [f"{cn}({en})" for _, cn, en in fields]
        ws.append(headers)

        # Data rows
        for record in records:
            row = [record.get(field, "") for field, _, _ in fields]
            ws.append(row)

        path = self._output_dir / f"{filename}.xlsx"
        wb.save(str(path))
        return path

    def _write_csv(
        self, records: list[dict], fields: list[tuple], filename: str
    ) -> Path:
        """Write records to CSV file with UTF-8 BOM for Excel compatibility."""
        path = self._output_dir / f"{filename}.csv"

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            # Header: Chinese name with English alias
            headers = [f"{cn}({en})" for _, cn, en in fields]
            writer.writerow(headers)

            for record in records:
                row = [record.get(field, "") for field, _, _ in fields]
                writer.writerow(row)

        return path
