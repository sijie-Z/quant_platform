"""QMT/miniQMT utility functions — symbol mapping, error codes, conversions.

Provides the translation layer between the platform's internal order/position
models and xtquant's API conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from quant_platform.core.instrument import (
    normalize_symbol,
    symbol_code,
    symbol_exchange,
)
from quant_platform.trading.broker import OrderSide, OrderStatus, OrderType

# ── Exchange detection ──


def classify_exchange(code: str) -> str:
    """Return the exchange (``SH``/``SZ``/``BJ``) for an A-share symbol.

    Accepts bare codes, suffix forms (``600519.SH``), prefix forms
    (``sz300001``, ``sh.688001``) and Beijing codes (``4``/``8``/``920`` via
    ``bj.``).  The old implementation split on ``"."`` and treated everything
    that did not start with 6/9 as Shenzhen, so ``sh.688001`` mapped to SZ and
    every Beijing name was routed to Shenzhen.
    """
    exchange = symbol_exchange(code)
    if exchange:
        return exchange
    # Defensive fallback for malformed input: keep the historical default.
    return "SH" if str(code).startswith(("6", "9")) else "SZ"


# ── Symbol mapping ──


def to_qmt_code(code: str) -> str:
    """Convert an internal symbol to QMT's canonical ``CODE.EXCHANGE`` form.

    Accepts bare codes, prefix styles and already-canonical symbols:
    ``600519`` -> ``600519.SH``, ``sz300001`` -> ``300001.SZ``,
    ``920001`` -> ``920001.BJ``.
    """
    canonical = normalize_symbol(code)
    if symbol_exchange(canonical):
        return canonical
    exchange = classify_exchange(code)
    digits = symbol_code(code) or str(code)
    return f"{digits}.{exchange}"


def from_qmt_code(code: str) -> str:
    """Convert a QMT symbol back to its bare six-digit internal code.

    ``600519.SH`` -> ``600519``; unknown/malformed values are returned
    unchanged (without the suffix) so callers can still log them.
    """
    digits = symbol_code(code)
    if digits is not None:
        return digits
    return code.split(".")[0] if "." in code else code


def to_qmt_exchange(code: str) -> int:
    """Return the QMT market constant for a symbol.

    QMT/xtquant market ids: 1 = Shanghai, 2 = Shenzhen, 0 = Beijing.
    """
    exchange = classify_exchange(code)
    if exchange == "SH":
        return 1
    if exchange == "SZ":
        return 2
    return 0  # Beijing (bj.xxxxxx)


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


def from_qmt_order_type(qmt_order_type: Any) -> OrderSide:
    """Map an xtconstant order type (``XtOrder.order_type``) back to OrderSide.

    23 = STOCK_BUY, 24 = STOCK_SELL. Unknown/missing values default to BUY,
    matching the historical direction assumption used when such an order
    could only be read back from the broker (there was no direction before).
    """
    try:
        value = int(qmt_order_type)
    except (TypeError, ValueError):
        return OrderSide.BUY
    return OrderSide.SELL if value == 24 else OrderSide.BUY


def from_qmt_price_type(qmt_price_type: Any) -> OrderType:
    """Map an xtconstant price type (``XtOrder.price_type``) back to OrderType.

    5 = LATEST_PRICE (market), 11 = FIX_PRICE (limit). Anything else is
    treated as a limit order because that is the only safe interpretation of
    an explicit ``price`` field.
    """
    try:
        value = int(qmt_price_type)
    except (TypeError, ValueError):
        return OrderType.LIMIT
    return OrderType.MARKET if value == 5 else OrderType.LIMIT


# ── Status mapping ──


class QMTOrderStatus(IntEnum):
    """QMT order status codes (xtconstant.xt_order_status).

    xtquant reports ``XtOrder.order_status`` with these values. The table
    previously used 0/1/2/10/11/12/22/23/24/50 -- codes from a different
    API -- so every real XtOrder status fell through to PENDING.
    """
    UNREPORTED = 48
    WAIT_REPORTING = 49
    REPORTED = 50
    REPORTED_CANCEL = 51        # reported, cancellation pending
    PARTSUCC_CANCEL = 52        # partially filled, cancellation pending
    PART_CANCEL = 53            # partially filled, rest cancelled
    CANCELED = 54
    PART_SUCC = 55
    SUCCEEDED = 56
    JUNK = 57                   # rejected / invalid order
    UNKNOWN = 255


QMT_STATUS_MAP: dict[int, OrderStatus] = {
    QMTOrderStatus.UNREPORTED: OrderStatus.PENDING,
    QMTOrderStatus.WAIT_REPORTING: OrderStatus.PENDING,
    QMTOrderStatus.REPORTED: OrderStatus.SUBMITTED,
    QMTOrderStatus.REPORTED_CANCEL: OrderStatus.SUBMITTED,
    QMTOrderStatus.PARTSUCC_CANCEL: OrderStatus.PARTIAL,
    QMTOrderStatus.PART_CANCEL: OrderStatus.CANCELLED,
    QMTOrderStatus.CANCELED: OrderStatus.CANCELLED,
    QMTOrderStatus.PART_SUCC: OrderStatus.PARTIAL,
    QMTOrderStatus.SUCCEEDED: OrderStatus.FILLED,
    QMTOrderStatus.JUNK: OrderStatus.REJECTED,
    QMTOrderStatus.UNKNOWN: OrderStatus.PENDING,
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

    Handles both xtquant objects and mock dicts during testing. Real XtTrade
    reports carry the ``traded_*`` names (``traded_id`` / ``traded_price`` /
    ``traded_volume`` / ``traded_amount``); the plain aliases are kept for
    mocks and for the legacy async-response shape.
    """
    if isinstance(qmt_trade, dict):
        return qmt_trade

    result: dict[str, Any] = {}
    for attr, key in (
        # xtquant XtTrade fields.
        ("traded_id", "trade_id"),
        ("traded_price", "price"),
        ("traded_volume", "quantity"),
        ("traded_amount", "amount"),
        ("order_sysid", "order_sysid"),
        # Legacy/mock field names (applied after, so they win when present).
        ("order_id", "order_id"),
        ("trade_id", "trade_id"),
        ("stock_code", "code"),
        ("price", "price"),
        ("volume", "quantity"),
        ("trade_amount", "amount"),
        ("order_type", "order_type"),
        ("trade_time", "trade_time"),
    ):
        val = getattr(qmt_trade, attr, None)
        if val is not None:
            result[key] = val
    return result


def qmt_order_to_dict(qmt_order: Any) -> dict:
    """Convert a QMT order object (``XtOrder``) to a plain dict.

    Handles both xtquant objects and mock dicts. Only the fields the
    reconciliation path needs are extracted; ``order_id`` is the broker order
    id returned by ``order_stock``/``query_stock_orders`` and ``order_status``
    is the raw xtconstant status code (map with ``qmt_status_to_internal``).
    """
    if isinstance(qmt_order, dict):
        return qmt_order

    result: dict[str, Any] = {}
    for attr, key in (
        ("order_id", "order_id"),
        ("order_sysid", "order_sysid"),
        ("stock_code", "code"),
        ("order_type", "order_type"),
        ("order_volume", "order_volume"),
        ("price_type", "price_type"),
        ("price", "price"),
        ("traded_volume", "traded_volume"),
        ("traded_price", "traded_price"),
        ("order_status", "order_status"),
        ("status_msg", "status_msg"),
        ("order_time", "order_time"),
        ("strategy_name", "strategy_name"),
    ):
        val = getattr(qmt_order, attr, None)
        if val is not None:
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
