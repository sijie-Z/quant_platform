"""NAV Calculator — daily net asset value computation with fee accrual.

Implements fund NAV calculation following Chinese private fund standards:
- Total NAV = cash + market_value - accrued_fees
- NAV per unit = total NAV / total units
- Management fee: annualized, accrued daily
- Performance fee: high-water mark method (20% above HWM)
- High-water mark: tracks the peak NAV per unit

Usage:
    calc = NAVCalculator(store, annual_mgmt_fee=0.02, perf_fee_rate=0.20)
    nav = calc.calculate_daily_nav(date="2024-03-15")
    calc.save_nav(nav)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NAV:
    """Daily NAV record."""
    date: str
    nav_total: float          # Total fund NAV (净资产总额)
    nav_per_unit: float       # NAV per unit (单位净值)
    total_units: float        # Total fund units (总份额)
    cash: float               # Cash balance
    market_value: float       # Portfolio market value
    mgmt_fee: float           # Management fee accrued today
    perf_fee: float           # Performance fee accrued today
    high_water_mark: float    # High-water mark (历史最高单位净值)
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    accrued_mgmt_fee: float = 0.0   # Cumulative management fee accrued to date
    accrued_perf_fee: float = 0.0   # Cumulative performance fee accrued to date

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "nav_total": round(self.nav_total, 4),
            "nav_per_unit": round(self.nav_per_unit, 4),
            "total_units": round(self.total_units, 2),
            "cash": round(self.cash, 2),
            "market_value": round(self.market_value, 2),
            "mgmt_fee": round(self.mgmt_fee, 2),
            "perf_fee": round(self.perf_fee, 2),
            "high_water_mark": round(self.high_water_mark, 4),
            "daily_return": round(self.daily_return, 6),
            "cumulative_return": round(self.cumulative_return, 6),
            "accrued_mgmt_fee": round(self.accrued_mgmt_fee, 2),
            "accrued_perf_fee": round(self.accrued_perf_fee, 2),
        }


class NAVCalculator:
    """Daily NAV computation with management and performance fee accrual.

    Args:
        store: Store instance for data persistence.
        annual_mgmt_fee: Annual management fee rate (e.g., 0.02 = 2%).
        perf_fee_rate: Performance fee rate (e.g., 0.20 = 20%).
        initial_nav_per_unit: Initial NAV per unit (default: 1.0).
        initial_units: Initial total units (default: 10,000,000).
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        store: Any,
        annual_mgmt_fee: float = 0.02,
        perf_fee_rate: float = 0.20,
        initial_nav_per_unit: float = 1.0,
        initial_units: float = 10_000_000,
    ):
        self._store = store
        self._annual_mgmt_fee = annual_mgmt_fee
        self._perf_fee_rate = perf_fee_rate
        self._initial_nav_per_unit = initial_nav_per_unit
        self._total_units = initial_units
        self._high_water_mark = initial_nav_per_unit
        # Fees are liabilities: once accrued they stay deducted from NAV until
        # actually paid. The old code subtracted only the current day's fee
        # from an undiminished asset base, so NAV was permanently overstated.
        self._accrued_mgmt_fee = 0.0
        self._accrued_perf_fee = 0.0

        # Try to restore state from last NAV record
        self._restore_state()

    def _restore_state(self) -> None:
        """Restore HWM and units from the most recent NAV record.

        This asked for `days=1` and then relied on `date > cutoff`, where
        cutoff is *yesterday's date string*. A record written yesterday does
        not satisfy `>`, so the lookup only ever found a record written today.
        Any restart on a new calendar day silently reset the high-water mark to
        1.0 and the unit count to the initial value -- which books a
        performance fee against investors who are still below high water.

        Restoring state means "read the latest record we have", not "read
        yesterday's" -- so this asks the store for the latest row directly
        rather than for a window ending today.
        """
        try:
            latest = self._store.get_latest_nav()
            if latest:
                self._high_water_mark = latest.get(
                    "high_water_mark", self._initial_nav_per_unit
                )
                self._total_units = latest.get("total_units", self._total_units)
                self._accrued_mgmt_fee = float(latest.get("accrued_mgmt_fee", 0) or 0)
                self._accrued_perf_fee = float(latest.get("accrued_perf_fee", 0) or 0)
        except (AttributeError, Exception):
            pass

    def calculate_daily_nav(
        self,
        date: str | None = None,
        cash: float | None = None,
        market_value: float | None = None,
    ) -> NAV:
        """Calculate NAV for a given date.

        If cash/market_value not provided, reads from store positions.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.
            cash: Cash balance. If None, uses store.
            market_value: Portfolio market value. If None, uses store.

        Returns:
            NAV record with all fields computed.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        if cash is None or market_value is None:
            cash, market_value = self._get_portfolio_values()

        gross_assets = cash + market_value

        # Previous record strictly before this date, and the number of trading
        # days elapsed. `get_nav_history(days=2)` missed every Friday when run
        # on Monday, which zeroed the daily return and collapsed cash.
        previous = []
        try:
            history = self._store.get_nav_history(days=365 * 10)
            previous = [row for row in history if str(row.get("date", "")) < date]
        except (AttributeError, Exception):
            previous = []

        elapsed_days = 1
        if previous:
            try:
                elapsed_days = max(
                    1, int(np.busday_count(str(previous[-1]["date"]), date)),
                )
            except (TypeError, ValueError):
                elapsed_days = 1

        # Management fee is a liability: accrue it cumulatively and deduct the
        # whole accrual from gross assets every day. Charging it on the
        # post-fee base makes the accrual compound, as a real fee does.
        fee_base = max(0.0, gross_assets - self._accrued_mgmt_fee)
        mgmt_fee = self._accrue_mgmt_fee(fee_base, days=elapsed_days)
        self._accrued_mgmt_fee += mgmt_fee

        gross_nav = gross_assets - self._accrued_mgmt_fee
        nav_per_unit_gross = gross_nav / self._total_units if self._total_units > 0 else 0

        # Performance fee: 20% of the new high above the high-water mark.
        # The HWM is the gross-of-performance-fee peak per unit, so the fee is
        # charged once per new high. The old code set the HWM to the *net*
        # NAV, which let the same gain be charged again after the fee.
        if nav_per_unit_gross > self._high_water_mark:
            perf_fee = (
                (nav_per_unit_gross - self._high_water_mark)
                * self._total_units * self._perf_fee_rate
            )
            self._high_water_mark = nav_per_unit_gross
        else:
            perf_fee = 0.0
        self._accrued_perf_fee += perf_fee

        # Net NAV
        nav_total = gross_nav - perf_fee
        nav_per_unit = nav_total / self._total_units if self._total_units > 0 else 0

        # Returns
        daily_return = 0.0
        cumulative_return = 0.0
        if previous:
            prev_nav = previous[-1].get("nav_per_unit", self._initial_nav_per_unit)
            if prev_nav > 0:
                daily_return = nav_per_unit / prev_nav - 1
        cumulative_return = nav_per_unit / self._initial_nav_per_unit - 1

        return NAV(
            date=date,
            nav_total=nav_total,
            nav_per_unit=nav_per_unit,
            total_units=self._total_units,
            cash=cash,
            market_value=market_value,
            mgmt_fee=mgmt_fee,
            perf_fee=perf_fee,
            high_water_mark=self._high_water_mark,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            accrued_mgmt_fee=self._accrued_mgmt_fee,
            accrued_perf_fee=self._accrued_perf_fee,
        )

    def save_nav(self, nav: NAV) -> None:
        """Persist NAV record to store."""
        self._store.save_nav(nav.to_dict())

    def update_daily_nav(
        self,
        date: str | None = None,
        cash: float | None = None,
        market_value: float | None = None,
    ) -> NAV:
        """Calculate and save daily NAV (convenience method).

        Intended to be called at end-of-day by the trading engine.
        """
        nav = self.calculate_daily_nav(date=date, cash=cash, market_value=market_value)
        self.save_nav(nav)
        logger.info(
            "NAV updated: date=%s nav/unit=%.4f total=%.2f mgmt_fee=%.2f perf_fee=%.2f",
            nav.date, nav.nav_per_unit, nav.nav_total, nav.mgmt_fee, nav.perf_fee,
        )
        return nav

    def _get_portfolio_values(self) -> tuple[float, float]:
        """Get cash and market value from store positions."""
        try:
            positions = self._store.get_positions()
            market_value = sum(p.get("market_value", 0) for p in positions)
            # Latest snapshot regardless of age: a 24-hour window returns
            # nothing on a Monday and silently set cash to zero.
            pnl = self._store.get_pnl_history(days=365 * 10)
            cash = pnl[-1].get("cash", 0) if pnl else 0
            return cash, market_value
        except (AttributeError, Exception):
            return 0, 0

    def _accrue_mgmt_fee(self, nav: float, days: int = 1) -> float:
        """Management fee accrual.

        mgmt_fee = NAV * annual_rate / 252 * elapsed_trading_days
        """
        return nav * self._annual_mgmt_fee / self.TRADING_DAYS_PER_YEAR * days

    def _accrue_perf_fee(self, nav_per_unit: float) -> float:
        """Performance fee with high-water mark.

        Only charged when nav_per_unit exceeds the high-water mark.
        Fee = (nav_per_unit - HWM) * total_units * perf_fee_rate
        """
        if nav_per_unit <= self._high_water_mark:
            return 0.0
        gain = nav_per_unit - self._high_water_mark
        return gain * self._total_units * self._perf_fee_rate
