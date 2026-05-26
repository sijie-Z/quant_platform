"""QMT/miniQMT utility functions — symbol mapping, error codes, conversions.

Provides the translation layer between the platform's internal order/position
models and xtquant's API conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from quant_platform.trading.broker import Order, OrderSide, OrderStatus, OrderType, Position


# ── Exchange detection ──


def classify_exchange(code: str) -> str:
    """Return 'SH' or 'SZ' based on A-share code prefix."""
    if "." in code:
        code = code.split(".")[0]
    return "SH" if code.startswith(("6", "9")) else "SZ"


# ── Symbol mapping ──


def to_qmt_code(code: str) -> str:
    """Convert internal code to QMT format.

    '600519'  -> '600519.SH'
    '000001'  -> '000001.SZ'
    '600519.SH' -> '600519.SH'  (no-op)
    """
    if "." in code:
        return code
    exchange = classify_exchange(code)
    return f"{code}.{exchange}"


def from_qmt_code(code: str) -> str:
    """Convert QMT format back to internal code.

    '600519.SH' -> '600519'
    '000001.SZ' -> '000001'
    """
    return code.split(".")[0] if "." in code else code


def to_qmt_exchange(code: str) -> int:
    """Return QMT exchange constant for a code.

    Import safety: returns int 1 for SH, 2 for SZ.
    """
    exchange = classify_exchange(code)
    return 1 if exchange == "SH" else 2  # 1=SH, 2=SZ


# ── Order type mapping ──


def to_qmt_price_type(order_type: OrderType) -> int:
    """Map internal OrderType to xtconstant price type.

    Constants (xtconstant):
        LATEST_PRICE = 5   (market / best price)
        FIX_PRICE    = 11  (limit)
    """
    if order_type == OrderType.MARKET:
        return 5  # LATEST_PRICE
    return 11   # FIX_PRICE


def to_qmt_order_type(side: OrderSide) -> int:
    """Map OrderSide to xtconstant order type.

    Constants:
        STOCK_BUY  = 23
        STOCK_SELL = 24
    """
    return 23 if side == OrderSide.BUY else 24


# ── Status mapping ──


class QMTOrderStatus(IntEnum):
    """QMT order status codes (xtconstant)."""
    UNREPORTED = 0
    WAIT_REPORTING = 1
    REPORTED = 2
    WAIT_CANCEL = 10
    PART_CANCELED = 11
    CANCELED = 12
    PART_FILLED = 22
    FILLED = 23
    PART_CANCEL_FILLED = 24
    WAIT_SENDING = 50


QMT_STATUS_MAP: dict[int, OrderStatus] = {
    0:  OrderStatus.PENDING,
    1:  OrderStatus.PENDING,
    2:  OrderStatus.SUBMITTED,
    10: OrderStatus.SUBMITTED,
    11: OrderStatus.PARTIAL,
    12: OrderStatus.CANCELLED,
    22: OrderStatus.PARTIAL,
    23: OrderStatus.FILLED,
    24: OrderStatus.PARTIAL,
    50: OrderStatus.PENDING,
}


def qmt_status_to_internal(qmt_status: int) -> OrderStatus:
    """Convert QMT order status code to internal OrderStatus."""
    return QMT_STATUS_MAP.get(qmt_status, OrderStatus.PENDING)


# ── QMT error code descriptions ──


QMT_ERROR_MESSAGES: dict[int, str] = {
    -1: "未知错误",
    -2: "网络连接失败",
    -3: "登录失败",
    -4: "账户未登录",
    -5: "参数错误",
    -6: "资金不足",
    -7: "持仓不足",
    -8: "不在交易时间",
    -9: "股票停牌",
    -10: "涨跌停限制",
    -11: "撤单失败-订单不存在",
    -12: "撤单失败-订单已成交",
    -13: "重复下单",
    -14: "下单数量不符合要求",
    -15: "价格超出涨跌停范围",
}


def describe_qmt_error(code: int) -> str:
    """Return human-readable error message for a QMT error code."""
    return QMT_ERROR_MESSAGES.get(code, f"QMT error {code}")


# ── Conversion helpers ──


@dataclass
class QMTExecution:
    """Parsed QMT trade/execution report."""
    order_id: str = ""
    trade_id: str = ""
    code: str = ""
    price: float = 0.0
    quantity: int = 0
    amount: float = 0.0
    side: str = ""
    time: str = ""


def qmt_trade_to_dict(qmt_trade: Any) -> dict:
    """Convert a QMT trade object to a plain dict.

    Handles both xtquant objects and mock dicts during testing.
    """
    if isinstance(qmt_trade, dict):
        return qmt_trade

    result: dict[str, Any] = {}
    for attr in ("order_id", "trade_id", "stock_code", "price",
                 "volume", "trade_amount", "order_type", "trade_time"):
        val = getattr(qmt_trade, attr, None)
        if val is not None:
            key = attr.replace("stock_code", "code").replace("volume", "quantity")
            result[key] = val
    return result


def qmt_position_to_dict(qmt_pos: Any) -> dict:
    """Convert QMT position object to plain dict."""
    if isinstance(qmt_pos, dict):
        return qmt_pos

    result: dict[str, Any] = {}
    for attr in ("stock_code", "volume", "can_use_volume", "avg_price",
                 "market_value", "open_price"):
        val = getattr(qmt_pos, attr, None)
        if val is not None:
            key = attr.replace("stock_code", "code")
            result[key] = val
    return result
