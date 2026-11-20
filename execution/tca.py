"""Transaction Cost Analysis (TCA) — institutional-grade execution quality measurement.

Implements three industry-standard TCA benchmarks:
1. Implementation Shortfall (IS): Decision price vs final portfolio value
2. Arrival Price: Execution vs market price at order arrival time
3. VWAP Benchmark: Execution vs volume-weighted average price

Plus decomposition into:
- Delay cost: Price move between decision and arrival
- Market impact: Price move during execution
- Timing cost: Opportunity cost of execution timing
- Realized spread: Bid-ask bounce capture

Reference:
- Perold (1988): "The Implementation Shortfall"
- Kissell & Glantz (2003): "Optimal Trading Strategies"
- Almgren & Chriss (2000): "Optimal Execution"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────


class TCABenchmark(StrEnum):
    """TCA benchmark types."""
    ARRIVAL_PRICE = "arrival_price"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    CLOSE_PRICE = "close_price"


@dataclass
class TCAResult:
    """TCA result for a single order."""
    order_id: str
    ticker: str
    side: str
    quantity: int
    benchmark: TCABenchmark

    # Prices
    decision_price: float = 0.0     # Price when decision was made
    arrival_price: float = 0.0      # Price when order arrived at market
    avg_exec_price: float = 0.0     # Average execution price
    vwap_price: float = 0.0         # VWAP during execution window
    close_price: float = 0.0        # Closing price on execution day

    # Cost decomposition (all in bps)
    implementation_shortfall_bps: float = 0.0
    arrival_cost_bps: float = 0.0
    vwap_cost_bps: float = 0.0
    delay_cost_bps: float = 0.0     # Decision → Arrival
    market_impact_bps: float = 0.0  # Arrival → Execution
    timing_cost_bps: float = 0.0    # Execution vs VWAP
    realized_spread_bps: float = 0.0

    # Fill details
    num_fills: int = 0
    fill_duration_seconds: float = 0.0
    participation_rate: float = 0.0

    @property
    def total_cost_bps(self) -> float:
        """Total cost relative to decision price."""
        return self.implementation_shortfall_bps

    @property
    def signed_cost_bps(self) -> float:
        """Cost adjusted for trade direction (positive = unfavorable)."""
        sign = 1.0 if self.side == "buy" else -1.0
        return sign * self.implementation_shortfall_bps


@dataclass
class TCASummary:
    """Aggregated TCA statistics across multiple orders."""
    n_orders: int = 0
    total_notional: float = 0.0

    # Mean costs in bps
    mean_is_bps: float = 0.0
    mean_arrival_bps: float = 0.0
    mean_vwap_bps: float = 0.0
    mean_delay_bps: float = 0.0
    mean_impact_bps: float = 0.0
    mean_timing_bps: float = 0.0

    # Distribution
    median_is_bps: float = 0.0
    p25_is_bps: float = 0.0
    p75_is_bps: float = 0.0
    std_is_bps: float = 0.0

    # By side
    buy_mean_bps: float = 0.0
    sell_mean_bps: float = 0.0

    # Per-ticker breakdown
    by_ticker: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# VWAP Calculator
# ──────────────────────────────────────────────────────────────────────


class VWAPCalculator:
    """Compute VWAP from tick/bar data.

    VWAP = Σ(Price_i × Volume_i) / Σ(Volume_i)

    Supports both intraday tick data and daily bar approximation.
    """

    @staticmethod
    def from_bars(
        prices: pd.Series,
        volumes: pd.Series,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> float:
        """Compute VWAP from OHLCV bars.

        Uses typical price (H+L+C)/3 weighted by volume for bar data,
        which is more accurate than close-price VWAP.

        Args:
            prices: Series of typical prices or close prices.
            volumes: Series of volumes.
            start_time: Optional start filter.
            end_time: Optional end filter.

        Returns:
            VWAP value.
        """
        mask = pd.Series(True, index=prices.index)
        if start_time is not None:
            mask &= prices.index >= start_time
        if end_time is not None:
            mask &= prices.index <= end_time

        p = prices[mask]
        v = volumes[mask]

        total_volume = v.sum()
        if total_volume <= 0:
            return float(p.mean()) if len(p) > 0 else 0.0

        return float((p * v).sum() / total_volume)

    @staticmethod
    def from_ticks(
        prices: pd.Series,
        volumes: pd.Series,
    ) -> float:
        """Compute VWAP from tick-level data."""
        total_volume = volumes.sum()
        if total_volume <= 0:
            return 0.0
        return float((prices * volumes).sum() / total_volume)


# ──────────────────────────────────────────────────────────────────────
# TCA Engine
# ──────────────────────────────────────────────────────────────────────


class TCAEngine:
    """Transaction Cost Analysis engine.

    Computes execution quality metrics against multiple benchmarks.

    Usage:
        engine = TCAEngine()
        result = engine.analyze_order(
            order_id="ORD001",
            ticker="600519.SH",
            side="buy",
            quantity=10000,
            fills=[{"price": 100.5, "quantity": 10000, "timestamp": "..."}],
            decision_price=100.0,
            arrival_price=100.2,
            vwap_price=100.3,
            close_price=101.0,
            market_volume=5000000,
        )
        summary = engine.summarize([result1, result2, ...])
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate

    def analyze_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        quantity: int,
        fills: list[dict[str, Any]],
        decision_price: float,
        arrival_price: float,
        vwap_price: float,
        close_price: float,
        market_volume: int = 0,
        arrival_time: datetime | None = None,
        first_fill_time: datetime | None = None,
        last_fill_time: datetime | None = None,
    ) -> TCAResult:
        """Analyze execution quality for a single order.

        Args:
            order_id: Unique order identifier.
            ticker: Stock ticker.
            side: "buy" or "sell".
            quantity: Order quantity.
            fills: List of fill dicts with "price", "quantity", "timestamp".
            decision_price: Price when investment decision was made.
            arrival_price: Price when order reached the market.
            vwap_price: VWAP during execution window.
            close_price: Closing price on execution day.
            market_volume: Total market volume during execution window.
            arrival_time: Time order arrived at market.
            first_fill_time: Time of first fill.
            last_fill_time: Time of last fill.

        Returns:
            TCAResult with all cost metrics.
        """
        if not fills:
            return TCAResult(
                order_id=order_id,
                ticker=ticker,
                side=side,
                quantity=quantity,
                benchmark=TCABenchmark.IMPLEMENTATION_SHORTFALL,
            )

        # Compute average execution price
        total_qty = sum(f["quantity"] for f in fills)
        if total_qty <= 0:
            avg_exec_price = 0.0
        else:
            avg_exec_price = sum(
                f["price"] * f["quantity"] for f in fills
            ) / total_qty

        # Direction sign: buy = +1, sell = -1
        # For buys, paying more is worse (positive cost = bad)
        # For sells, receiving less is worse (positive cost = bad)
        sign = 1.0 if side == "buy" else -1.0

        # Implementation Shortfall: (exec_price - decision_price) / decision_price
        is_bps = 0.0
        if decision_price > 0:
            is_bps = sign * (avg_exec_price - decision_price) / decision_price * 10000

        # Arrival Price cost: (exec_price - arrival_price) / arrival_price
        arrival_bps = 0.0
        if arrival_price > 0:
            arrival_bps = sign * (avg_exec_price - arrival_price) / arrival_price * 10000

        # VWAP cost: (exec_price - vwap) / vwap
        vwap_bps = 0.0
        if vwap_price > 0:
            vwap_bps = sign * (avg_exec_price - vwap_price) / vwap_price * 10000

        # Delay cost: (arrival_price - decision_price) / decision_price
        delay_bps = 0.0
        if decision_price > 0:
            delay_bps = sign * (arrival_price - decision_price) / decision_price * 10000

        # Market impact: (exec_price - arrival_price) / arrival_price
        # Same as arrival_bps but decomposed separately
        impact_bps = arrival_bps

        # Timing cost: exec_price vs VWAP
        timing_bps = vwap_bps

        # Realized spread: measures bid-ask bounce capture
        # For buys: if exec < close → captured spread (negative cost = good)
        # For sells: if exec > close → captured spread
        realized_spread_bps = 0.0
        if close_price > 0:
            realized_spread_bps = sign * (avg_exec_price - close_price) / close_price * 10000

        # Fill duration
        fill_duration = 0.0
        if first_fill_time and last_fill_time:
            fill_duration = (last_fill_time - first_fill_time).total_seconds()

        # Participation rate
        participation = 0.0
        if market_volume > 0:
            participation = total_qty / market_volume

        return TCAResult(
            order_id=order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            benchmark=TCABenchmark.IMPLEMENTATION_SHORTFALL,
            decision_price=decision_price,
            arrival_price=arrival_price,
            avg_exec_price=avg_exec_price,
            vwap_price=vwap_price,
            close_price=close_price,
            implementation_shortfall_bps=round(is_bps, 2),
            arrival_cost_bps=round(arrival_bps, 2),
            vwap_cost_bps=round(vwap_bps, 2),
            delay_cost_bps=round(delay_bps, 2),
            market_impact_bps=round(impact_bps, 2),
            timing_cost_bps=round(timing_bps, 2),
            realized_spread_bps=round(realized_spread_bps, 2),
            num_fills=len(fills),
            fill_duration_seconds=fill_duration,
            participation_rate=round(participation, 6),
        )

    #: Accepted order-execution date columns, in priority order.
    _DATE_COLUMNS = ("execution_date", "date", "trade_date")

    @staticmethod
    def _execution_date(order: pd.Series) -> pd.Timestamp:
        """Return the order's execution date.

        The date must be explicit: the previous batch path implicitly used the
        final row of the price frame as "the close", which is a future
        function.
        """
        for column in TCAEngine._DATE_COLUMNS:
            if column in order.index:
                value = order[column]
                if not pd.isna(value):
                    return pd.Timestamp(value)
        raise ValueError(
            "orders_df must provide an execution date column "
            f"(one of {list(TCAEngine._DATE_COLUMNS)}) for every order"
        )

    @staticmethod
    def _benchmark_from_prices(
        prices_df: pd.DataFrame,
        ticker: str,
        execution_date: pd.Timestamp,
    ) -> tuple[float, float]:
        """Return ``(vwap, close)`` for ``ticker`` on ``execution_date``.

        Supported layouts:

        * Long: columns include ``vwap`` and ``close``, optionally a
          ``ticker``/``code`` column selecting the instrument, and the date
          comes from an ``execution_date``/``date``/``trade_date`` column or
          the index. A frame without a ticker column is treated as a
          single-instrument price series.
        * Wide with 2-level ``MultiIndex`` columns: one level holds the
          ``vwap``/``close`` fields, the other the tickers; the index is the
          date.

        Raises:
            ValueError: the layout is unsupported or the benchmark value is
                missing/non-finite/non-positive.
            KeyError: no row exists for ``ticker`` on ``execution_date``.
        """
        frame = prices_df
        columns = frame.columns

        if isinstance(columns, pd.MultiIndex):
            if columns.nlevels != 2:
                raise ValueError(
                    "prices_df MultiIndex columns must have exactly 2 levels "
                    "(field, ticker)"
                )
            field_level = None
            for level in range(columns.nlevels):
                values = columns.get_level_values(level)
                if "vwap" in values and "close" in values:
                    field_level = level
                    break
            if field_level is None:
                raise ValueError(
                    "prices_df MultiIndex columns must contain 'vwap' and "
                    "'close' on one level"
                )
            ticker_level = 1 - field_level
            if ticker not in columns.get_level_values(ticker_level):
                raise KeyError(
                    f"No price data for ticker {ticker!r} in prices_df"
                )
            sub = frame.xs(ticker, axis=1, level=ticker_level)
            vwap_values = sub["vwap"]
            close_values = sub["close"]
            dates = pd.DatetimeIndex(frame.index)
        else:
            if "vwap" not in columns or "close" not in columns:
                raise ValueError(
                    "prices_df must contain 'vwap' and 'close' columns "
                    "(optionally with a 'ticker'/'code' column); got "
                    f"{list(columns)}"
                )
            sub = frame
            ticker_column = next(
                (c for c in ("ticker", "code") if c in frame.columns), None,
            )
            if ticker_column is not None:
                sub = frame[frame[ticker_column] == ticker]
                if sub.empty:
                    raise KeyError(
                        f"No price data for ticker {ticker!r} in prices_df"
                    )

            date_column = next(
                (c for c in TCAEngine._DATE_COLUMNS if c in sub.columns), None,
            )
            if date_column is not None:
                dates = pd.DatetimeIndex(pd.to_datetime(sub[date_column]))
            else:
                dates = pd.DatetimeIndex(pd.to_datetime(sub.index))
            if len(dates) != len(sub):
                raise ValueError(
                    f"Could not align price dates for ticker {ticker!r}"
                )
            vwap_values = sub["vwap"]
            close_values = sub["close"]

        target = pd.Timestamp(execution_date).normalize()
        matches = dates.normalize() == target
        if not matches.any():
            raise KeyError(
                f"No price data for ticker {ticker!r} on {target.date()} "
                "in prices_df"
            )
        row = int(np.flatnonzero(matches)[0])
        vwap = float(vwap_values.iloc[row])
        close = float(close_values.iloc[row])

        for name, value in (("vwap", vwap), ("close", close)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Invalid {name} for ticker {ticker!r} on "
                    f"{target.date()}: {value!r}"
                )
        return vwap, close

    def analyze_from_dataframe(
        self,
        orders_df: pd.DataFrame,
        fills_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> list[TCAResult]:
        """Batch TCA analysis from DataFrames.

        Every order must carry its execution date -- an ``execution_date``
        column (preferred) or a ``date``/``trade_date`` column. Benchmarks
        come from the price row for that date only. The previous version used
        the full-sample mean as VWAP and the last row as the close (a future
        function), and silently returned zeros for the documented
        ``[vwap, close, volume]`` layout.

        Args:
            orders_df: columns [order_id, ticker, side, quantity,
                decision_price, arrival_price, execution_date]. ``date`` and
                ``trade_date`` are accepted aliases for ``execution_date``.
            fills_df: columns [order_id, price, quantity, timestamp].
            prices_df: per-date benchmark prices; see
                :meth:`_benchmark_from_prices` for the supported layouts. A
                missing ticker/date/value raises instead of yielding 0.

        Returns:
            List of TCAResult.

        Raises:
            ValueError: an order has no execution date, or a benchmark value
                is missing/non-finite/non-positive.
            KeyError: no price row exists for the ticker/date pair.
        """
        results = []

        for _, order in orders_df.iterrows():
            oid = order["order_id"]
            order_fills = fills_df[fills_df["order_id"] == oid]

            if len(order_fills) == 0:
                continue

            fills_list = order_fills[["price", "quantity", "timestamp"]].to_dict("records")

            ticker = order["ticker"]
            execution_date = self._execution_date(order)
            vwap, close = self._benchmark_from_prices(
                prices_df, ticker, execution_date,
            )

            result = self.analyze_order(
                order_id=oid,
                ticker=ticker,
                side=order["side"],
                quantity=int(order["quantity"]),
                fills=fills_list,
                decision_price=float(order.get("decision_price", 0)),
                arrival_price=float(order.get("arrival_price", 0)),
                vwap_price=vwap,
                close_price=close,
            )
            results.append(result)

        return results

    @staticmethod
    def summarize(results: list[TCAResult]) -> TCASummary:
        """Aggregate TCA results across multiple orders.

        Args:
            results: List of TCAResult from analyze_order().

        Returns:
            TCASummary with aggregated statistics.
        """
        if not results:
            return TCASummary()

        is_vals = [r.implementation_shortfall_bps for r in results]
        arrival_vals = [r.arrival_cost_bps for r in results]
        vwap_vals = [r.vwap_cost_bps for r in results]
        delay_vals = [r.delay_cost_bps for r in results]
        impact_vals = [r.market_impact_bps for r in results]
        timing_vals = [r.timing_cost_bps for r in results]

        buy_results = [r for r in results if r.side == "buy"]
        sell_results = [r for r in results if r.side == "sell"]

        buy_mean = np.mean([r.implementation_shortfall_bps for r in buy_results]) if buy_results else 0.0
        sell_mean = np.mean([r.implementation_shortfall_bps for r in sell_results]) if sell_results else 0.0

        # Per-ticker breakdown
        ticker_costs: dict[str, list[float]] = {}
        for r in results:
            ticker_costs.setdefault(r.ticker, []).append(r.implementation_shortfall_bps)
        by_ticker = {t: float(np.mean(v)) for t, v in ticker_costs.items()}

        notional = sum(
            r.avg_exec_price * r.quantity for r in results if r.avg_exec_price > 0
        )

        return TCASummary(
            n_orders=len(results),
            total_notional=notional,
            mean_is_bps=round(float(np.mean(is_vals)), 2),
            mean_arrival_bps=round(float(np.mean(arrival_vals)), 2),
            mean_vwap_bps=round(float(np.mean(vwap_vals)), 2),
            mean_delay_bps=round(float(np.mean(delay_vals)), 2),
            mean_impact_bps=round(float(np.mean(impact_vals)), 2),
            mean_timing_bps=round(float(np.mean(timing_vals)), 2),
            median_is_bps=round(float(np.median(is_vals)), 2),
            p25_is_bps=round(float(np.percentile(is_vals, 25)), 2),
            p75_is_bps=round(float(np.percentile(is_vals, 75)), 2),
            std_is_bps=round(float(np.std(is_vals)), 2),
            buy_mean_bps=round(float(buy_mean), 2),
            sell_mean_bps=round(float(sell_mean), 2),
            by_ticker=by_ticker,
        )

    def to_dataframe(self, results: list[TCAResult]) -> pd.DataFrame:
        """Convert TCA results to a DataFrame for reporting.

        Args:
            results: List of TCAResult.

        Returns:
            DataFrame with one row per order.
        """
        rows = []
        for r in results:
            rows.append({
                "order_id": r.order_id,
                "ticker": r.ticker,
                "side": r.side,
                "quantity": r.quantity,
                "decision_price": r.decision_price,
                "arrival_price": r.arrival_price,
                "avg_exec_price": r.avg_exec_price,
                "vwap_price": r.vwap_price,
                "close_price": r.close_price,
                "is_bps": r.implementation_shortfall_bps,
                "arrival_bps": r.arrival_cost_bps,
                "vwap_bps": r.vwap_cost_bps,
                "delay_bps": r.delay_cost_bps,
                "impact_bps": r.market_impact_bps,
                "timing_bps": r.timing_cost_bps,
                "realized_spread_bps": r.realized_spread_bps,
                "num_fills": r.num_fills,
                "participation_rate": r.participation_rate,
            })

        return pd.DataFrame(rows)
