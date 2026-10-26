"""Live Engine — alpha-v1.0 运行时系统.

把 batch backtest 改成 event-driven 持续运行程序.

架构:
  while market_is_open:
    data = get_data()
    state.update(data)
    signal = alpha(state)
    position = risk(signal)
    broker.send_orders(position)
    sleep(interval)

当前: paper mode (mock data + simulated execution)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import normalize_corporate_actions
from quant_platform.execution.models import OrderSide, OrderStatus
from quant_platform.shared.symbols import normalize_symbol, price_limit_threshold

logger = logging.getLogger(__name__)

# Bases that describe a return-generating (adjusted) series.  A paper/live
# fill or mark must never use them: shares and cash are denominated in the
# traded CNY price, and a split/dividend would silently re-scale the book.
ADJUSTED_PRICE_BASES = frozenset({"qfq", "hfq", "adjusted"})


def working_order_quantities(
    orders, side: OrderSide | None = None, include_partial: bool = True,
) -> dict[str, int]:
    """Remaining quantities by canonical symbol for genuinely working orders.

    ``include_partial=False`` is for IOC-style adapters (SimulatedBroker)
    whose partial fills have already had the unfilled remainder cancelled.
    QMT partial orders remain working until a terminal status arrives.
    """
    active = {OrderStatus.PENDING, OrderStatus.SUBMITTED}
    if include_partial:
        active.add(OrderStatus.PARTIAL)
    remaining: dict[str, int] = {}
    for order in orders or []:
        if getattr(order, "status", None) not in active:
            continue
        if side is not None and getattr(order, "side", None) != side:
            continue
        quantity = int(getattr(order, "quantity", 0) or 0)
        filled = int(getattr(order, "filled_quantity", 0) or 0)
        open_quantity = max(0, quantity - filled)
        if open_quantity > 0:
            code = normalize_symbol(getattr(order, "code", ""))
            remaining[code] = remaining.get(code, 0) + open_quantity
    return remaining


def working_order_codes(orders, include_partial: bool = True) -> set[str]:
    """Canonical symbols with unfilled quantity still working at the broker.

    A submitted order is exposure already committed, even though it is not in
    the position book yet. Callers use this to suppress duplicate intents
    until the broker reports a fill or cancellation.
    """
    return set(working_order_quantities(orders, include_partial=include_partial))


def normalize_corporate_action_feed(actions) -> pd.DataFrame:
    """Normalize a provider feed and pin ex-dates to naive exchange days.

    ``normalize_corporate_actions`` parses the contract but leaves a
    timezone-aware ``ex_date`` tz-aware (and its vectorised ``to_datetime``
    coerces *naive* rows to ``NaT`` when a column mixes aware and naive
    values).  A paper engine advances with naive ``YYYY-MM-DD`` exchange
    dates, so a tz-aware value can raise or silently mis-date an action.

    Ex-dates are therefore pinned **element-wise before** the contract
    parser: a tz-aware timestamp is converted to Asia/Shanghai and its
    timezone dropped, then normalized to midnight; a naive value is only
    normalized; an unparseable value becomes ``NaT`` (dropped by the
    contract parser, like before).  Mixed-timezone feeds therefore keep
    every row.
    """
    if actions is None:
        return normalize_corporate_actions(None)
    frame = (
        actions.copy() if isinstance(actions, pd.DataFrame)
        else pd.DataFrame(actions)
    )
    if "ex_date" in frame.columns and len(frame):
        frame["ex_date"] = frame["ex_date"].map(_pin_corporate_action_date)
    return normalize_corporate_actions(frame)


def _pin_corporate_action_date(value) -> pd.Timestamp:
    """One ex-date -> naive midnight in the exchange timezone (``NaT`` on junk)."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if timestamp is pd.NaT:
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.normalize()


def summarize_corporate_action_events(
    events, source: str = "",
) -> dict[str, Any]:
    """Provenance summary for a run/session report.

    ``events`` is the list of audit dicts returned by an action application
    (one entry per ``(asset, ex_date)`` event).  The summary answers "were
    corporate actions applied in this run?", with the count, the latest
    ex-date seen and where the feed came from (``"pipeline:baostock"``,
    ``"synthetic"``, ``"manual"``, ...).  An empty/absent feed yields
    ``{"count": 0, "last_ex_date": None, "source": source}`` rather than
    disappearing from the report.
    """
    applied = [event for event in (events or []) if isinstance(event, dict)]
    dates = [
        str(event.get("date") or "")[:10]
        for event in applied
        if str(event.get("date") or "")
    ]
    return {
        "count": len(applied),
        "last_ex_date": max(dates) if dates else None,
        "source": str(source or ""),
    }


def ensure_raw_price_basis(
    price_basis: Any, *, context: str = "live/paper marks",
) -> str:
    """Validate that a price panel is in raw CNY terms.

    Returns the normalized label.  An explicitly adjusted basis is a hard
    error (D15); an unlabeled legacy panel returns ``"unknown"`` so the
    caller can warn rather than silently assume.  Callers that need a hard
    guarantee must check for ``"raw"``.
    """
    text = str(price_basis or "").strip().lower()
    if text in ADJUSTED_PRICE_BASES:
        raise ValueError(
            f"{context} require raw (traded) CNY prices, got "
            f"price_basis={text!r}. Adjusted levels are not tradeable: use "
            "DataPipeline.get_raw_close(), a broker quote, or a real-time "
            "feed."
        )
    return text or "unknown"

# ── 策略参数 (alpha-v1.0, 已锁定) ──
SIGNAL_H = 40
HOLD_H = 80
SELECT_PCT = 0.20
VOL_PERCENTILE = 0.70
INITIAL_CAPITAL = 10_000_000
COST_BPS = 15
MAX_DRAWDOWN = 0.30


@dataclass
class Position:
    asset: str
    entry_date: str
    exit_date: str
    shares: int
    entry_price: float


@dataclass
class State:
    """系统运行时状态."""
    cash: float = INITIAL_CAPITAL
    positions: list[Position] = field(default_factory=list)
    equity_peak: float = INITIAL_CAPITAL
    current_equity: float = INITIAL_CAPITAL
    total_trades: int = 0
    last_rebalance: str = ""
    next_rebalance: str = ""
    vol_series: list[float] = field(default_factory=list)
    last_processed_date: str = ""
    status: str = "init"


class LiveEngine:
    """alpha-v1.0 Live Runtime.

    Usage:
        engine = LiveEngine(data_provider, broker)
        engine.run()
    """

    def __init__(self, data_provider=None, broker=None, cash_dividend_tax_rate: float = 0.0):
        self.data_provider = data_provider
        self.broker = broker
        self.state = State()
        self._history: pd.DataFrame | None = None
        self._day_count = 0
        self._rebalance_dates: list[pd.Timestamp] = []
        # Basis of `self._prices`; money paths require "raw" (or an explicitly
        # acknowledged legacy panel).
        self._price_basis: str = "unknown"
        # Corporate actions (paper runtime only).  Markers persist inside the
        # engine snapshot, so a daemon restart never re-applies an ex-date.
        self._cash_dividend_tax_rate = max(0.0, float(cash_dividend_tax_rate))
        self._corporate_actions: pd.DataFrame | None = None
        self._applied_corporate_actions: set[tuple[str, str]] = set()
        self.cash_ledger: list[dict[str, Any]] = []
        self.corporate_action_events: list[dict[str, Any]] = []
        #: asset -> (ex_date, ratio) for a split applied on that session; used
        #: to value a suspended ex-date at the post-split reference instead of
        #: the stale pre-split close the panel repeats that day.
        self._split_valuation_override: dict[str, tuple[str, float]] = {}

    def set_corporate_actions(self, actions) -> int:
        """Load the cash/split feed applied as exchange days advance.

        Real brokers (QMT/XTP) settle corporate actions on their side; when
        one is wired to this engine the feed is stored but never applied
        locally, matching the "no double accounting" rule.  Returns the
        number of actions loaded.
        """
        frame = normalize_corporate_actions(actions)
        self._corporate_actions = frame
        if not self._corporate_actions_supported():
            logger.warning(
                "Broker %s settles corporate actions remotely; the feed is "
                "stored but will NOT be applied to the local book",
                type(self.broker).__name__,
            )
        return len(frame)

    def _corporate_actions_supported(self) -> bool:
        """Whether this engine owns the book (paper) or a broker does."""
        if self.broker is None:
            return True
        return bool(getattr(self.broker, "supports_corporate_actions", False))

    def apply_corporate_actions(
        self, actions=None, date: pd.Timestamp | str | None = None,
    ) -> list[dict[str, Any]]:
        """Apply actions with ``ex_date <= date`` once (paper book only).

        A split multiplies ``shares`` and divides ``entry_price`` (position
        value unchanged); a cash dividend credits
        ``shares * cash_per_share * (1 - tax)`` to ``state.cash``.  The
        adjusted research price panel already carries the ex-date drop, so
        equity is continuous across the action.  Applied ``(asset, ex_date)``
        markers make replays no-ops; they travel in
        :meth:`to_snapshot`.
        """
        if not self._corporate_actions_supported():
            return []
        if actions is not None:
            frame = normalize_corporate_actions(actions)
        else:
            frame = self._corporate_actions
        if frame is None or frame.empty:
            return []
        cutoff = pd.Timestamp(date).normalize() if date is not None else None
        rows = frame.copy()
        rows["_order"] = rows["kind"].map(
            lambda kind: 0 if kind == "cash" else 1
        )
        rows = rows.sort_values(["ex_date", "asset", "_order"], kind="stable")
        applied: list[dict[str, Any]] = []
        # One marker per (asset, ex_date) event; a combined cash+split pair
        # shares the key and both components are applied before recording it.
        for (group_date, group_asset), group in rows.groupby(
            ["ex_date", "asset"], sort=False,
        ):
            action_date = pd.Timestamp(group_date).normalize()
            if cutoff is not None and action_date > cutoff:
                continue
            asset = normalize_symbol(group_asset)
            marker = (asset, action_date.date().isoformat())
            if marker in self._applied_corporate_actions:
                continue
            self._applied_corporate_actions.add(marker)
            event = self._apply_action_group(asset, marker, group)
            if event is not None:
                self.corporate_action_events.append(event)
                applied.append(event)
        return applied

    def _apply_action_group(
        self, asset: str, marker: tuple[str, str], group: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Apply all components sharing one ``(asset, ex_date)`` marker.

        Only positions held *before* the ex-date are entitled: a position
        entered on or after the ex-date must not receive a caught-up past
        action (the daemon replays the latest date, so the window can be
        wide).  An empty ``entry_date`` (legacy/imported book) is treated as
        pre-existing.
        """
        ex_date = marker[1]
        matches = [
            position for position in self.state.positions
            if normalize_symbol(position.asset) == asset
            and str(position.entry_date or "") < ex_date
        ]
        cash_total = 0.0
        tax_total = 0.0
        cash_per_share_total = 0.0
        split_ratio_total = 1.0
        components: list[dict[str, Any]] = []
        timestamp = datetime.now().isoformat(timespec="seconds")

        # Cash first: a 送转+派息 dividend is declared per pre-split share.
        for _, row in group.iterrows():
            if str(row["kind"]) != "cash":
                continue
            cash_per_share = float(row["cash_per_share"])
            if not (cash_per_share > 0):
                continue
            quantity = sum(p.shares for p in matches)
            gross = quantity * cash_per_share
            tax = gross * self._cash_dividend_tax_rate
            net = gross - tax
            cash_total += net
            tax_total += tax
            cash_per_share_total += cash_per_share
            components.append({
                "kind": "cash",
                "cash_per_share": cash_per_share,
                "cash_delta": net,
                "tax": tax,
            })
            if quantity > 0:
                self.cash_ledger.append({
                    "date": ex_date,
                    "asset": asset,
                    "kind": "cash_dividend",
                    "quantity": quantity,
                    "cash_per_share": cash_per_share,
                    "tax": round(tax, 6),
                    "cash_delta": round(net, 6),
                    "description": f"{quantity} shares x {cash_per_share}/share",
                    "timestamp": timestamp,
                })
        if cash_total:
            self.state.cash += cash_total

        for _, row in group.iterrows():
            if str(row["kind"]) != "split":
                continue
            ratio = float(row["split_ratio"])
            if not (ratio > 0) or ratio == 1.0:
                continue
            split_ratio_total *= ratio
            components.append({"kind": "split", "split_ratio": ratio})
            for position in matches:
                position.shares = int(round(position.shares * ratio))
                position.entry_price = (
                    position.entry_price / ratio
                    if position.entry_price > 0 else 0.0
                )
            if matches:
                self._split_valuation_override[asset] = (ex_date, ratio)

        if not components:
            return None
        kinds = sorted({component["kind"] for component in components})
        event: dict[str, Any] = {
            "date": ex_date,
            "asset": asset,
            "kind": "+".join(kinds),
            "positions": len(matches),
            "quantity": sum(p.shares for p in matches),
            "cash_delta": cash_total,
            "tax": tax_total,
            "cash_per_share": cash_per_share_total,
            "split_ratio": split_ratio_total,
            "components": components,
            "timestamp": timestamp,
        }
        self._publish_corporate_action(event)
        return event

    @staticmethod
    def _publish_corporate_action(event: dict[str, Any]) -> None:
        try:
            from quant_platform.core.events import get_event_bus

            get_event_bus().publish(
                "corporate_action", event, source="live_engine",
            )
        except Exception:  # monitoring must never break the runtime
            pass

    # ── 初始化 ──

    def load_history(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        price_basis: str | None = None,
    ):
        """加载历史数据用于状态初始化.

        ``prices`` must be a **raw** (traded CNY) panel: every fill, equity
        mark and kill-switch liquidation in this engine multiplies shares by
        these levels.  An adjusted (``qfq``/``hfq``) panel raises instead of
        silently marking the book at a level that never traded (D15).
        ``price_basis`` overrides the panel's ``attrs["price_basis"]``.
        """
        if prices is None:
            raise ValueError("LiveEngine.load_history requires a price panel")
        panel_basis = getattr(prices, "attrs", {}).get("price_basis")
        declared = price_basis
        if declared is not None and panel_basis is not None:
            declared_norm = ensure_raw_price_basis(
                declared, context="LiveEngine fills/equity marks",
            )
            panel_norm = ensure_raw_price_basis(
                panel_basis, context="LiveEngine fills/equity marks",
            )
            if (
                declared_norm in {"raw", "qfq", "hfq"}
                and panel_norm in {"raw", "qfq", "hfq"}
                and declared_norm != panel_norm
            ):
                raise ValueError(
                    f"price_basis override {declared_norm!r} contradicts the "
                    f"panel's declared basis {panel_norm!r}; refusing to "
                    "guess which one is true"
                )
        if declared is None:
            declared = panel_basis
        if declared is None:
            logger.warning(
                "LiveEngine.load_history received an unlabeled price panel; "
                "assuming legacy raw prices. New callers must pass "
                "DataPipeline.get_raw_close() (attrs['price_basis']='raw')."
            )
            basis = "unknown"
        else:
            basis = ensure_raw_price_basis(
                declared, context="LiveEngine fills/equity marks",
            )
        self._price_basis = basis
        self._returns = returns
        self._prices = prices
        # `returns[t]` is the forward return t -> t+1. The signal at t must
        # only use returns realized by the close of t, i.e. `returns[t-1]`.
        self._realized_returns = returns.shift(1)
        # Candidate rebalance dates for batch replay. Live runs decide from
        # `last_rebalance` because the daemon calls `run_once` with the
        # latest data date, which need not land on a multiple of HOLD_H.
        dates = returns.index
        self._rebalance_dates = [dates[i] for i in range(HOLD_H, len(dates), HOLD_H)]
        self.state.next_rebalance = str(self._first_rebalance_after(HOLD_H))[:10] if len(dates) > HOLD_H else ""
        logger.info("History loaded: %d days, %d rebalance dates", len(returns), len(self._rebalance_dates))

    def _first_rebalance_after(self, day_idx: int) -> pd.Timestamp:
        dates = self._returns.index
        if day_idx < len(dates):
            return dates[day_idx]
        return dates[-1]

    def _is_due_for_rebalance(self, date: pd.Timestamp) -> bool:
        """Whether this date is a scheduled rebalance.

        Batch replay yields every date and expects one entry every HOLD_H
        sessions. The daemon yields only the latest date, so membership in a
        precomputed list meant it never rebalanced at all.
        """
        idx = self._returns.index.get_loc(date)
        if idx < HOLD_H:
            return False
        if not self.state.last_rebalance:
            return True
        try:
            last_idx = self._returns.index.get_loc(
                pd.Timestamp(self.state.last_rebalance)
            )
        except KeyError:
            return True
        return idx - last_idx >= HOLD_H

    # ── 状态持久化 ──

    # v2: prices/equity are recorded on the raw (traded CNY) basis. A v1
    # snapshot was produced by the hfq-marked engine and is not comparable
    # (same cash, different price level); restoring it would show a fake
    # drawdown and could trip the kill switch on the first raw mark.
    SNAPSHOT_VERSION = 2

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize all runtime state for persistence across processes."""
        return {
            "version": self.SNAPSHOT_VERSION,
            "cash": self.state.cash,
            "positions": [asdict(p) for p in self.state.positions],
            "equity_peak": self.state.equity_peak,
            "current_equity": self.state.current_equity,
            "total_trades": self.state.total_trades,
            "last_rebalance": self.state.last_rebalance,
            "next_rebalance": self.state.next_rebalance,
            "vol_series": list(self.state.vol_series),
            "status": self.state.status,
            "last_processed_date": self.state.last_processed_date,
            "day_count": self._day_count,
            # Additive (v2 stays readable): applied corporate-action markers
            # must survive a restart so a replayed ex-date is a no-op.
            "applied_corporate_actions": sorted(
                [asset, date_text]
                for asset, date_text in self._applied_corporate_actions
            ),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Rehydrate state from :meth:`to_snapshot`.

        Call this AFTER ``load_history``: loading history recomputes
        ``next_rebalance`` and would otherwise overwrite the persisted value.

        Raises:
            ValueError: if the snapshot version is unsupported.
        """
        version = snapshot.get("version")
        if version != self.SNAPSHOT_VERSION:
            raise ValueError(
                f"Unsupported LiveEngine snapshot version: {version!r} "
                f"(current {self.SNAPSHOT_VERSION}). v1 snapshots were marked "
                "at adjusted (hfq) levels; this engine marks at raw CNY "
                "prices. Do not restore across the price-basis migration: "
                "reconcile the book against the broker and start a fresh "
                "snapshot."
            )

        self.state.cash = float(snapshot.get("cash", INITIAL_CAPITAL))
        self.state.positions = [
            Position(**position) for position in snapshot.get("positions", [])
        ]
        self.state.equity_peak = float(snapshot.get("equity_peak", INITIAL_CAPITAL))
        self.state.current_equity = float(snapshot.get("current_equity", INITIAL_CAPITAL))
        self.state.total_trades = int(snapshot.get("total_trades", 0))
        self.state.last_rebalance = snapshot.get("last_rebalance", "")
        self.state.next_rebalance = snapshot.get("next_rebalance", "")
        self.state.vol_series = list(snapshot.get("vol_series", []))
        self.state.status = snapshot.get("status", "init")
        self.state.last_processed_date = snapshot.get("last_processed_date", "")
        self._day_count = int(snapshot.get("day_count", 0))
        self._applied_corporate_actions = {
            (normalize_symbol(asset), str(date_text))
            for asset, date_text in snapshot.get("applied_corporate_actions", [])
        }

    # ── 核心循环 ──

    def run_once(self, date: pd.Timestamp) -> dict[str, Any]:
        """单日运行 (每次调用 = 一个交易日).

        这是 live engine 的核心——每次市场收盘后调用一次.
        """
        if self._returns is None or self._prices is None:
            return {"status": "no_data"}

        date_str = str(date)[:10]

        # A killed engine stays killed until an operator explicitly restores
        # or clears it; and a date already processed must not be replayed
        # (a daemon restart used to double-apply a day).
        if self.state.status == "killed":
            return {
                "date": date_str, "action": "killed", "status": "killed",
                "equity": self.state.current_equity, "cash": self.state.cash,
            }
        if self.state.last_processed_date and date_str <= self.state.last_processed_date:
            return {
                "date": date_str, "action": "none", "status": "already_processed",
                "equity": self.state.current_equity, "cash": self.state.cash,
            }

        self._day_count += 1
        result: dict[str, Any] = {"date": date_str, "action": "none"}

        # 0. Corporate actions: every pending ex-date up to this session is
        #    applied once (markers persisted in snapshots).  Done before the
        #    equity mark so cash and shares move together on the ex-date.
        applied_actions = self.apply_corporate_actions(date=date)
        if applied_actions:
            result["corporate_actions"] = len(applied_actions)

        # 1. 更新权益
        self._update_equity(date)

        # 2. 检查是否需要调仓
        is_rebalance = self._is_due_for_rebalance(date)

        if is_rebalance:
            result["action"] = self._execute_rebalance(date)
            self.state.last_rebalance = date_str

            # 更新下次调仓日
            idx = self._returns.index.get_loc(date)
            next_idx = idx + HOLD_H
            if next_idx < len(self._returns.index):
                self.state.next_rebalance = str(self._returns.index[next_idx])[:10]

        # 3. 检查 kill switch
        drawdown = (self.state.equity_peak - self.state.current_equity) / self.state.equity_peak
        if drawdown > MAX_DRAWDOWN:
            self._emergency_flat(date)
            result["action"] = "kill_switch"
            result["reason"] = f"drawdown={drawdown:.2%}"

        result["equity"] = self.state.current_equity
        result["cash"] = self.state.cash
        result["drawdown"] = drawdown
        result["n_positions"] = len(self.state.positions)
        result["price_basis"] = self._price_basis

        self.state.last_processed_date = date_str
        return result

    def run(self, start: str | None = None, end: str | None = None):
        """跑完整回放 (batch 模式, 模拟持续运行)."""
        if self._returns is None:
            raise ValueError("No data")

        dates = self._returns.index
        if start:
            dates = dates[dates >= start]
        if end:
            dates = dates[dates <= end]

        logger.info("LiveEngine starting: %s -> %s (%d days)",
                     str(dates[0])[:10], str(dates[-1])[:10], len(dates))

        results = []
        for date in dates:
            r = self.run_once(date)
            if r["action"] != "none":
                logger.info("[%s] %s%s", r["date"], r["action"],
                           f" ({r.get('reason','')})" if r.get("reason") else "")
            results.append(r)

        final = results[-1]
        logger.info("LiveEngine done. Equity: %.2f, Trades: %d, Status: %s",
                     final["equity"], self.state.total_trades, self.state.status)
        return results

    # ── 内部方法 ──

    def _update_equity(self, date: pd.Timestamp):
        """计算当日总权益 (逐日盯市).

        On a split's ex-date a suspended stock keeps feeding its stale
        pre-split close; valuing the doubled share count at that level would
        double the position.  The quote is therefore validated against the
        post-split reference (previous close / ratio): a move larger than the
        board's daily limit cannot be a real one-session return and the
        reference price is used for that mark instead.
        """
        date_str = str(date)[:10]
        for asset, (override_date, _ratio) in list(
            self._split_valuation_override.items()
        ):
            if override_date != date_str:
                del self._split_valuation_override[asset]

        pos_value = 0.0
        if self._prices is not None and date in self._prices.index:
            px = self._prices.loc[date]
            for pos in self.state.positions:
                if pos.asset in px.index and not pd.isna(px[pos.asset]):
                    price = float(px[pos.asset])
                    asset = normalize_symbol(pos.asset)
                    override = self._split_valuation_override.get(asset)
                    if override is not None and override[0] == date_str:
                        reference = self._post_split_reference(
                            pos.asset, date, override[1],
                        )
                        limit = price_limit_threshold(asset)
                        if (
                            reference is not None
                            and price > reference * (1 + limit + 1e-9)
                        ):
                            logger.warning(
                                "[%s] %s split ex-date: stale pre-split quote "
                                "%.4f ignored for valuation (post-split "
                                "reference %.4f)",
                                date_str, pos.asset, price, reference,
                            )
                            price = reference
                        else:
                            del self._split_valuation_override[asset]
                    pos_value += pos.shares * price

        self.state.current_equity = self.state.cash + pos_value
        if self.state.current_equity > self.state.equity_peak:
            self.state.equity_peak = self.state.current_equity

    def _post_split_reference(
        self, asset: str, date: pd.Timestamp, ratio: float,
    ) -> float | None:
        """Previous observable close divided by the split ratio."""
        if self._prices is None or asset not in self._prices.columns:
            return None
        history = self._prices[asset].loc[:date]
        history = history.iloc[:-1].dropna()
        if history.empty or not (ratio > 0):
            return None
        return float(history.iloc[-1]) / ratio

    def _execute_rebalance(self, date: pd.Timestamp) -> str:
        """执行调仓."""
        if self._returns is None or self._prices is None:
            return "error"

        day_idx = self._returns.index.get_loc(date)
        n_stocks = len(self._returns.columns)

        # ── 平到期仓位 ──
        # Expired positions must be SOLD, not filtered out of the list: the
        # old one-line filter made their market value vanish from equity
        # without crediting cash or charging the exit cost (#3).
        date_str = str(date)[:10]
        exit_price_row = self._prices.loc[date] if date in self._prices.index else None
        exit_cost = COST_BPS / 10000
        surviving = []
        for pos in self.state.positions:
            if pos.exit_date > date_str:
                surviving.append(pos)
                continue
            px = exit_price_row.get(pos.asset) if exit_price_row is not None else None
            if px is None or pd.isna(px) or px <= 0:
                # No observable exit price: keep the position rather than
                # silently vaporising its value.
                surviving.append(pos)
                continue
            self.state.cash += pos.shares * float(px) * (1 - exit_cost)
            self.state.total_trades += 1
        self.state.positions = surviving

        # ── Vol filter ──
        market_vol = self._realized_returns.mean(axis=1).rolling(20).std()
        # Threshold over information available on `date`; the full-sample
        # quantile let the filter see the future.
        history_vol = market_vol.loc[:date].dropna()
        vol_thresh = (
            history_vol.quantile(VOL_PERCENTILE) if len(history_vol) else np.nan
        )
        if date in market_vol.index and not pd.isna(vol_thresh) and market_vol[date] > vol_thresh:
            return "skip_high_vol"

        # ── 信号 ──
        past_ret = self._realized_returns.rolling(
            SIGNAL_H, min_periods=SIGNAL_H,
        ).apply(
            lambda x: np.prod(1 + x) - 1 if len(x) > 0 else 0, raw=True
        )
        pr = past_ret.iloc[day_idx]
        valid = pr.dropna().sort_values()
        n_select = max(1, int(n_stocks * SELECT_PCT))

        if len(valid) < n_select:
            return "skip_insufficient_data"

        selected = valid.head(n_select)
        n = len(selected)
        cap_per = self.state.cash / n

        # ── 买入 ──
        price_row = self._prices.loc[date] if date in self._prices.index else None
        if price_row is None:
            return "skip_no_price"

        total_spent = 0.0
        for asset in selected.index:
            ep = price_row[asset] if asset in price_row.index else 0
            if pd.isna(ep) or ep <= 0:
                continue
            shares = max(100, int(cap_per / ep / 100) * 100)
            spent = shares * ep
            if total_spent + spent > self.state.cash:
                break  # 现金不够

            exit_date = self._returns.index[min(day_idx + HOLD_H, len(self._returns.index) - 1)]
            self.state.positions.append(Position(
                asset=asset,
                entry_date=str(date)[:10],
                exit_date=str(exit_date)[:10],
                shares=shares,
                entry_price=ep,
            ))
            total_spent += spent
            self.state.total_trades += 1

        self.state.cash -= total_spent  # 买入不扣成本, 卖出时扣
        return f"enter_{n}"

    def _emergency_flat(self, date: pd.Timestamp):
        """紧急平仓 (kill switch)."""
        cost = COST_BPS / 10000
        price_row = self._prices.loc[date] if date in self._prices.index else None

        for pos in self.state.positions:
            if price_row is not None and pos.asset in price_row.index:
                px = price_row[pos.asset]
                if not pd.isna(px) and px > 0:
                    self.state.cash += pos.shares * px * (1 - cost)

        self.state.positions = []
        self.state.status = "killed"
        logger.warning("[%s] KILL SWITCH triggered — all positions closed", str(date)[:10])

    def status_report(self) -> str:
        """生成状态报告."""
        dd = (self.state.equity_peak - self.state.current_equity) / self.state.equity_peak if self.state.equity_peak > 0 else 0
        return (
            f"\n{'=' * 60}\n"
            f"  LiveEngine — alpha-v1.0\n"
            f"{'=' * 60}\n"
            f"  Status:      {self.state.status}\n"
            f"  Equity:      {self.state.current_equity:>12,.2f}\n"
            f"  Cash:        {self.state.cash:>12,.2f}\n"
            f"  Positions:   {len(self.state.positions):>6d}\n"
            f"  Drawdown:    {dd*100:>10.2f}%\n"
            f"  Trades:      {self.state.total_trades:>6d}\n"
            f"  Price basis: {self._price_basis}\n"
            f"  Last rebal:  {self.state.last_rebalance}\n"
            f"  Next rebal:  {self.state.next_rebalance}\n"
            f"{'=' * 60}"
        )
