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

        # Try to restore state from last NAV record
        self._restore_state()

    def _restore_state(self) -> None:
        """Restore HWM and units from the latest NAV record."""
        try:
            history = self._store.get_nav_history(days=1)
            if history:
                latest = history[-1]
                self._high_water_mark = latest.get(
                    "high_water_mark", self._initial_nav_per_unit
                )
                self._total_units = latest.get("total_units", self._total_units)
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

        # Daily management fee accrual
        mgmt_fee = self._accrue_mgmt_fee(cash + market_value)

        # Gross NAV before performance fee
        gross_nav = cash + market_value - mgmt_fee
        nav_per_unit_gross = gross_nav / self._total_units if self._total_units > 0 else 0

        # Performance fee (high-water mark method)
        perf_fee = self._accrue_perf_fee(nav_per_unit_gross)

        # Net NAV
        nav_total = gross_nav - perf_fee
        nav_per_unit = nav_total / self._total_units if self._total_units > 0 else 0

        # Update high-water mark
        if nav_per_unit > self._high_water_mark:
            self._high_water_mark = nav_per_unit

        # Returns
        daily_return = 0.0
        cumulative_return = 0.0
        try:
            history = self._store.get_nav_history(days=2)
            if history:
                prev = history[-1]
                prev_nav = prev.get("nav_per_unit", self._initial_nav_per_unit)
                if prev_nav > 0:
                    daily_return = nav_per_unit / prev_nav - 1
            cumulative_return = nav_per_unit / self._initial_nav_per_unit - 1
        except (AttributeError, Exception):
            pass

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
            # Cash is inferred from P&L history if available
            pnl = self._store.get_pnl_history(days=1)
            if pnl:
                cash = pnl[-1].get("cash", 0)
            else:
                cash = 0
            return cash, market_value
        except (AttributeError, Exception):
            return 0, 0

    def _accrue_mgmt_fee(self, nav: float) -> float:
        """Daily management fee accrual.

        mgmt_fee = NAV * annual_rate / 252
        """
        return nav * self._annual_mgmt_fee / self.TRADING_DAYS_PER_YEAR

    def _accrue_perf_fee(self, nav_per_unit: float) -> float:
        """Performance fee with high-water mark.

        Only charged when nav_per_unit exceeds the high-water mark.
        Fee = (nav_per_unit - HWM) * total_units * perf_fee_rate
        """
        if nav_per_unit <= self._high_water_mark:
            return 0.0
        gain = nav_per_unit - self._high_water_mark
        return gain * self._total_units * self._perf_fee_rate
