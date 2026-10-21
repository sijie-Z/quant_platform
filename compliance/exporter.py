"""Compliance exporter — trade/order/risk log export for regulatory reporting.

Exports trading records in formats required by Chinese private fund
regulations (私募基金交易记录保存格式). Supports Excel (.xlsx) and CSV.

Fields follow CSRC naming conventions with English aliases for
cross-border reporting.

Tenant isolation
----------------
Trade and order logs are always scoped to exactly one tenant: an explicit
``tenant_id`` argument, or — when it is omitted — the current
:class:`quant_platform.core.context.TenantContext` (``"default"`` when no
context is set). A cross-tenant file can only be produced with the explicit
``include_all_tenants=True`` admin flag; the resolved scope is stamped on the
:class:`ExportResult` and on the audit record.

``pnl_history`` has no ``tenant_id`` column (schema limitation), so the risk
log is inherently a whole-book export. It requires the explicit
``include_all_tenants=True`` acknowledgement, a specific tenant is refused,
and the result/filename/audit are labeled ``all`` — the exporter never
silently merges books or presents a whole-book file under a per-fund label.

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
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from quant_platform.core.audit import AuditAction, AuditLog
from quant_platform.core.context import TenantContext
from quant_platform.core.store import EXCHANGE_TZ
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: Formats the exporter can actually produce. Anything else is a caller bug:
#: the old code silently wrote CSV for every unknown string and echoed the
#: requested name back in ExportResult.
EXPORT_FORMATS = frozenset({"csv", "xlsx"})

#: Characters that make spreadsheet software treat a cell as a formula
#: (CWE-1236). Tab and carriage return also start a dangerous cell.
_DANGEROUS_PREFIXES = frozenset("=+-@\t\r")

#: Control characters openpyxl refuses to write anywhere in a cell
#: (IllegalCharacterError). They are removed, not escaped, since they carry
#: no meaning in a compliance record.
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

#: Strict date shape: the window is compared as strings, so zero padding is
#: part of the contract (strptime alone accepts "2024-6-1").
_DATE_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Dates outside this range overflow when the window is widened by a day.
_MIN_EXPORT_DATE = datetime(1900, 1, 1)
_MAX_EXPORT_DATE = datetime(9998, 12, 31)

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


def _sanitize_cell(value: Any) -> Any:
    """Neutralize spreadsheet formula injection (CWE-1236).

    A leading ``=``, ``+``, ``-``, ``@``, tab or carriage return can turn an
    attacker-controlled cell into a formula that Excel/LibreOffice evaluates
    on open. Prefixing the value with an apostrophe keeps it text; numbers
    are left untouched so they stay numeric. XML-illegal control characters
    are removed first (openpyxl rejects them anywhere in a cell), and a
    dangerous character exposed by that removal is still neutralized. The
    same cell transformation is applied to CSV and XLSX output, so both
    formats carry identical payloads.
    """
    if not isinstance(value, str) or not value:
        return value
    text = _ILLEGAL_XML_CHARS.sub("", value)
    if not text:
        return ""
    if text[0] in _DANGEROUS_PREFIXES or ord(text[0]) < 0x20:
        return "'" + text
    return text


def _safe_filename_token(value: str) -> str:
    """Reduce a caller-supplied token to characters safe in a filename."""
    token = _UNSAFE_FILENAME_CHARS.sub("_", str(value)).strip("._-")
    return token[:64] or "all"


@dataclass
class ExportResult:
    """Result of an export operation."""
    file_path: str
    format: str
    n_records: int
    start_date: str
    end_date: str
    tenant_id: str = ""
    n_skipped: int = 0


class ComplianceExporter:
    """Export trading records for regulatory compliance.

    Reads from Store tables and writes Excel/CSV files with
    Chinese regulatory field names.

    Args:
        store: Store instance for data access.
        output_dir: Directory for exported files (default: ./compliance_output).
        audit_log: AuditLog to record export success/failure. When omitted a
            log backed by the same store is created.
    """

    def __init__(
        self,
        store: Any,
        output_dir: str = "compliance_output",
        audit_log: AuditLog | None = None,
    ):
        self._store = store
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._audit = audit_log if audit_log is not None else AuditLog(store)

    # ── Public exports ──────────────────────────────────────────────────

    def export_trade_log(
        self,
        start: str,
        end: str,
        tenant_id: str | None = None,
        format: str = "xlsx",
        include_all_tenants: bool = False,
    ) -> ExportResult:
        """Export trade log for a date range.

        Args:
            start: Start date (YYYY-MM-DD), inclusive.
            end: End date (YYYY-MM-DD), inclusive.
            tenant_id: Tenant filter. Defaults to the current TenantContext.
            format: "xlsx" or "csv".
            include_all_tenants: Explicit admin flag for a cross-tenant file.

        Returns:
            ExportResult with file path and record count.
        """
        tenant = ""
        scope_resolved = False
        export_format = str(format)
        # Validation and scope resolution run inside the try so a refused or
        # malformed export attempt still leaves an audit record.
        try:
            start, end = self._validate_window(start, end)
            export_format = self._normalize_format(format)
            tenant = self._resolve_tenant(tenant_id, include_all_tenants)
            scope_resolved = True

            # Dedicated export query: no API row clamp (get_trades stops at
            # 1000, which silently dropped older records of a larger date
            # range) and internal keyset pagination for the whole range.
            trades = self._store.get_trades_for_export(
                tenant_id=tenant, start_date=start, end_date=end,
            )
            filtered, skipped = self._filter_by_date(
                trades, "executed_at", start, end,
            )
            path, actual_format = self._write(
                filtered, TRADE_FIELDS, "trade_log", start, end, tenant,
                export_format,
            )
        except Exception as exc:
            self._record_audit(
                "trade_log", str(start), str(end),
                tenant if scope_resolved else (
                    str(tenant_id) if tenant_id else None
                ),
                export_format, 0, 0, "", "failure", exc,
            )
            raise

        self._record_audit(
            "trade_log", start, end, tenant, actual_format, len(filtered),
            skipped, str(path), "success",
        )
        logger.info("Exported %d trades to %s", len(filtered), path)
        return ExportResult(
            file_path=str(path),
            format=actual_format,
            n_records=len(filtered),
            start_date=start,
            end_date=end,
            tenant_id=tenant,
            n_skipped=skipped,
        )

    def export_order_log(
        self,
        start: str,
        end: str,
        tenant_id: str | None = None,
        format: str = "csv",
        include_all_tenants: bool = False,
    ) -> ExportResult:
        """Export order log for a date range.

        Args:
            start: Start date (YYYY-MM-DD), inclusive.
            end: End date (YYYY-MM-DD), inclusive.
            tenant_id: Tenant filter. Defaults to the current TenantContext.
            format: "xlsx" or "csv".
            include_all_tenants: Explicit admin flag for a cross-tenant file.

        Returns:
            ExportResult with file path and record count.
        """
        tenant = ""
        scope_resolved = False
        export_format = str(format)
        try:
            start, end = self._validate_window(start, end)
            export_format = self._normalize_format(format)
            tenant = self._resolve_tenant(tenant_id, include_all_tenants)
            scope_resolved = True

            # Dedicated export query: no API row clamp, keyset pagination.
            orders = self._store.get_orders_for_export(
                tenant_id=tenant, start_date=start, end_date=end,
            )
            filtered, skipped = self._filter_by_date(
                orders, "created_at", start, end,
            )
            path, actual_format = self._write(
                filtered, ORDER_FIELDS, "order_log", start, end, tenant,
                export_format,
            )
        except Exception as exc:
            self._record_audit(
                "order_log", str(start), str(end),
                tenant if scope_resolved else (
                    str(tenant_id) if tenant_id else None
                ),
                export_format, 0, 0, "", "failure", exc,
            )
            raise

        self._record_audit(
            "order_log", start, end, tenant, actual_format, len(filtered),
            skipped, str(path), "success",
        )
        logger.info("Exported %d orders to %s", len(filtered), path)
        return ExportResult(
            file_path=str(path),
            format=actual_format,
            n_records=len(filtered),
            start_date=start,
            end_date=end,
            tenant_id=tenant,
            n_skipped=skipped,
        )

    def export_risk_log(
        self,
        start: str,
        end: str,
        tenant_id: str | None = None,
        format: str = "csv",
        include_all_tenants: bool = False,
    ) -> ExportResult:
        """Export P&L history as risk log.

        ``pnl_history`` has no ``tenant_id`` column, so this export is
        inherently whole-book. It requires the explicit
        ``include_all_tenants=True`` acknowledgement; a specific
        ``tenant_id`` is refused. The result is stamped with tenant ``""``
        and the filename/audit record say ``all`` — a whole-book file is
        never presented as one fund's data.

        Args:
            start: Start date (YYYY-MM-DD), inclusive.
            end: End date (YYYY-MM-DD), inclusive.
            tenant_id: Must be omitted — P&L rows carry no tenant scope.
            format: "xlsx" or "csv".
            include_all_tenants: Required explicit acknowledgement of a
                whole-book (cross-tenant) export.

        Returns:
            ExportResult with file path and record count.
        """
        tenant = ""
        scope_resolved = False
        export_format = str(format)
        try:
            start, end = self._validate_window(start, end)
            export_format = self._normalize_format(format)
            if tenant_id:
                raise ValueError(
                    "pnl_history has no tenant_id column: a per-tenant risk "
                    "log cannot be produced."
                )
            if not include_all_tenants:
                raise ValueError(
                    "pnl_history has no tenant_id column, so the risk log is a "
                    "whole-book export. Pass include_all_tenants=True to "
                    "acknowledge that the file contains every book in this "
                    "store."
                )
            tenant = ""
            scope_resolved = True

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

            # Widened SQL window (the store's own query appends T23:59:59 to
            # the end date); the Python half-open filter below is
            # authoritative, so a record at end-of-day + fractional second is
            # not lost and a timezone-offset record cannot slip the window.
            sql_start = (
                datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
            sql_end = (
                datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            pnl = self._store.get_pnl_history(
                start_date=sql_start, end_date=sql_end,
            )
            filtered, skipped = self._filter_by_date(
                pnl, "timestamp", start, end,
            )
            path, actual_format = self._write(
                filtered, risk_fields, "risk_log", start, end, tenant,
                export_format,
            )
        except Exception as exc:
            self._record_audit(
                "risk_log", str(start), str(end),
                tenant if scope_resolved else (
                    str(tenant_id) if tenant_id else None
                ),
                export_format, 0, 0, "", "failure", exc,
            )
            raise

        self._record_audit(
            "risk_log", start, end, tenant, actual_format, len(filtered),
            skipped, str(path), "success",
        )
        logger.info("Exported %d risk records to %s", len(filtered), path)
        return ExportResult(
            file_path=str(path),
            format=actual_format,
            n_records=len(filtered),
            start_date=start,
            end_date=end,
            tenant_id=tenant,
            n_skipped=skipped,
        )

    # ── Scope / window helpers ──────────────────────────────────────────

    @staticmethod
    def _resolve_tenant(
        tenant_id: str | None, include_all_tenants: bool,
    ) -> str:
        """Resolve the tenant scope for a trade/order export.

        Empty string means "explicit cross-tenant export" and is only
        returned when the caller passed the admin flag. Asking for one
        tenant and all tenants at once is a contradictory request and is
        rejected rather than silently letting the flag win.
        """
        if include_all_tenants:
            if tenant_id:
                raise ValueError(
                    "tenant_id and include_all_tenants are mutually exclusive"
                )
            return ""
        if tenant_id:
            return str(tenant_id)
        return TenantContext.get_current().tenant_id or "default"

    @staticmethod
    def _validate_window(start: str, end: str) -> tuple[str, str]:
        """Validate the date window before any store query runs."""
        for label, value in (("start", start), ("end", end)):
            if (
                not isinstance(value, str)
                or _DATE_SHAPE.fullmatch(value) is None
            ):
                raise ValueError(
                    f"{label} must be a YYYY-MM-DD date, got {value!r}"
                )
            try:
                parsed = datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"{label} must be a YYYY-MM-DD date, got {value!r}"
                ) from None
            if not _MIN_EXPORT_DATE <= parsed <= _MAX_EXPORT_DATE:
                raise ValueError(
                    f"{label} is outside the supported export range "
                    f"1900-01-01..9998-12-31, got {value!r}"
                )
        if start > end:
            raise ValueError(f"start {start!r} is after end {end!r}")
        return start, end

    @staticmethod
    def _normalize_format(format: str) -> str:
        value = str(format).lower() if format is not None else ""
        if value not in EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export format {format!r}; "
                f"expected one of {sorted(EXPORT_FORMATS)}"
            )
        return value

    def _filter_by_date(
        self, records: list[dict], date_field: str, start: str, end: str,
    ) -> tuple[list[dict], int]:
        """Filter records by a half-open [start, end + 1 day) window.

        Returns the matching records and the number of records skipped for a
        missing or unparseable date field (counted and logged, never silently
        dropped). Naive timestamps are already on the exchange basis;
        timezone-aware ones are converted to Asia/Shanghai before comparison,
        independent of the host timezone.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)

        filtered: list[dict] = []
        skipped = 0
        for record in records:
            raw = record.get(date_field, "")
            if not raw:
                skipped += 1
                continue
            try:
                dt = datetime.fromisoformat(str(raw))
            except (ValueError, TypeError):
                skipped += 1
                continue
            if dt.tzinfo is not None:
                dt = dt.astimezone(EXCHANGE_TZ).replace(tzinfo=None)
            if start_dt <= dt < end_dt:
                filtered.append(record)

        if skipped:
            logger.warning(
                "Skipped %d record(s) with a missing or unparseable %s",
                skipped, date_field,
            )
        return filtered, skipped

    # ── Writers ─────────────────────────────────────────────────────────

    def _write(
        self,
        records: list[dict],
        fields: list[tuple],
        kind: str,
        start: str,
        end: str,
        tenant: str,
        export_format: str,
    ) -> tuple[Path, str]:
        """Write records and return the actual file path and format.

        The filename carries an exchange-time stamp and a random token, so a
        re-export never overwrites an earlier file (provenance) and two
        concurrent exports cannot race on the same path. Files are written to
        a temporary name and atomically renamed into place.
        """
        stamp = datetime.now(EXCHANGE_TZ).strftime("%Y%m%dT%H%M%S%f")
        token = uuid.uuid4().hex[:8]
        tenant_token = _safe_filename_token(tenant) if tenant else "all"
        filename = f"{kind}_{start}_{end}_{tenant_token}_{stamp}_{token}"

        if export_format == "xlsx":
            return self._write_xlsx(records, fields, filename)
        return self._write_csv(records, fields, filename), "csv"

    def _write_xlsx(
        self, records: list[dict], fields: list[tuple], filename: str,
    ) -> tuple[Path, str]:
        """Write records to Excel file. Returns (path, actual_format)."""
        try:
            from openpyxl import Workbook
        except ImportError:
            logger.warning("openpyxl not installed, falling back to CSV")
            return self._write_csv(records, fields, filename), "csv"

        wb = Workbook()
        ws = wb.active
        ws.title = "交易记录"

        # Header row: Chinese name + English alias
        ws.append([f"{cn}({en})" for _, cn, en in fields])

        # Data rows. Cell values are neutralized so a string starting with
        # "=" stays text (data_type 's') instead of becoming a formula ('f').
        for record in records:
            ws.append([
                _sanitize_cell(record.get(field, "")) for field, _, _ in fields
            ])

        path = self._output_dir / f"{filename}.xlsx"
        tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        try:
            wb.save(str(tmp))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path, "xlsx"

    def _write_csv(
        self, records: list[dict], fields: list[tuple], filename: str,
    ) -> Path:
        """Write records to CSV file with UTF-8 BOM for Excel compatibility."""
        path = self._output_dir / f"{filename}.csv"
        tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                # Header: Chinese name with English alias
                writer.writerow([f"{cn}({en})" for _, cn, en in fields])

                for record in records:
                    writer.writerow([
                        _sanitize_cell(record.get(field, ""))
                        for field, _, _ in fields
                    ])
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    # ── Audit ───────────────────────────────────────────────────────────

    def _record_audit(
        self,
        kind: str,
        start: str,
        end: str,
        tenant: str | None,
        export_format: str,
        n_records: int,
        n_skipped: int,
        file_path: str,
        result: str,
        error: Exception | None = None,
    ) -> None:
        """Record an export success/failure in the compliance audit trail.

        ``tenant`` is the resolved scope: ``""`` means an explicit whole-book
        export (recorded as ``"all"``), while ``None`` means the attempt was
        refused before the scope could be resolved (recorded as
        ``"unknown"``).

        An audit-write failure is logged but never replaces the export's own
        outcome (and never masks an export exception).
        """
        if tenant is None:
            scope = "unknown"
        else:
            scope = tenant or "all"
        details = {
            "export": kind,
            "tenant_id": scope,
            "start": start,
            "end": end,
            "format": export_format,
            "n_records": n_records,
            "n_skipped": n_skipped,
            "file_path": file_path,
        }
        if error is not None:
            details["error"] = f"{type(error).__name__}: {error}"
        try:
            self._audit.log(
                AuditAction.EXPORT,
                component="compliance.exporter",
                details=details,
                reason=f"{kind} export {start}..{end}",
                result=result,
                # Scope the audit row to the export's resolved tenant (when
                # one exists) so tenant-scoped audit queries find exports of
                # that tenant's data even from a default context.
                tenant_id=tenant or "",
            )
        except Exception:
            logger.exception("Failed to write export audit record")
