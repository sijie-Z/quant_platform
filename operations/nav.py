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

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: Strict date shape. ``strptime`` alone accepts "2024-6-3", but state
#: selection compares date strings lexicographically, so a mixed shape would
#: corrupt ordering (and create a second row for the same day).
_DATE_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}")


class NAVDataUnavailableError(RuntimeError):
    """Raised when a NAV cannot be built from trustworthy inputs.

    The calculator fails closed: a storage read error or a missing cash
    snapshot must not be turned into ``(0, 0)`` and persisted as a
    valid-looking zero NAV (which would report a -100% cumulative return).
    """


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
        """Full-precision representation for persistence.

        Values are rounded only at presentation. Rounding before the row is
        re-read as the next day's fee/HWM input produced spurious daily
        returns on unchanged NAVs and fee/HWM drift (4dp NAV, 2dp fees).
        """
        return {
            "date": self.date,
            "nav_total": self.nav_total,
            "nav_per_unit": self.nav_per_unit,
            "total_units": self.total_units,
            "cash": self.cash,
            "market_value": self.market_value,
            "mgmt_fee": self.mgmt_fee,
            "perf_fee": self.perf_fee,
            "high_water_mark": self.high_water_mark,
            "daily_return": self.daily_return,
            "cumulative_return": self.cumulative_return,
            "accrued_mgmt_fee": self.accrued_mgmt_fee,
            "accrued_perf_fee": self.accrued_perf_fee,
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

    # Management fees accrue on calendar days: the accrual window is calendar
    # time, so the annual rate is divided by 365. This used to count business
    # days but divide by 252, which charged 262/252 of the stated annual fee
    # in a normal year and up to six days of fees over a Spring Festival week.
    DAYS_PER_YEAR = 365.0

    #: Lookback used to find the state row before a calculation date. Matches
    #: the store's get_nav_history clamp (3650 days); older rows cannot be
    #: returned by that API, so a NAV older than this starts from initial
    #: state.
    MAX_HISTORY_DAYS = 3650

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
        self._initial_units = initial_units
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

        Fail closed: a store error while reading persisted state aborts
        construction instead of leaving HWM at 1.0 and re-creating the
        below-high-water fee bug. AttributeError means the store simply has no
        NAV API (pure in-memory use), which keeps the initial state.
        """
        try:
            latest = self._store.get_latest_nav()
        except AttributeError:
            return
        if not latest:
            return
        self._apply_state(latest)

    def _apply_state(self, row: dict) -> None:
        """Validate and install one persisted NAV row as the accrual state.

        Fail closed on values that would corrupt the next calculation: a
        zero/negative high-water mark would charge a performance fee on the
        entire NAV, and non-positive units make NAV per unit meaningless.
        """
        try:
            high_water_mark = float(
                row.get("high_water_mark", self._initial_nav_per_unit)
            )
            total_units = float(row.get("total_units", self._initial_units))
            accrued_mgmt = float(row.get("accrued_mgmt_fee", 0) or 0)
            accrued_perf = float(row.get("accrued_perf_fee", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise NAVDataUnavailableError(
                f"unreadable persisted NAV state: {exc}"
            ) from exc
        if not math.isfinite(high_water_mark) or high_water_mark <= 0:
            raise NAVDataUnavailableError(
                f"invalid persisted high-water mark {high_water_mark!r}"
            )
        if not math.isfinite(total_units) or total_units <= 0:
            raise NAVDataUnavailableError(
                f"invalid persisted total units {total_units!r}"
            )
        if not math.isfinite(accrued_mgmt) or accrued_mgmt < 0:
            raise NAVDataUnavailableError(
                f"invalid persisted accrued management fee {accrued_mgmt!r}"
            )
        if not math.isfinite(accrued_perf) or accrued_perf < 0:
            raise NAVDataUnavailableError(
                f"invalid persisted accrued performance fee {accrued_perf!r}"
            )
        self._high_water_mark = high_water_mark
        self._total_units = total_units
        self._accrued_mgmt_fee = accrued_mgmt
        self._accrued_perf_fee = accrued_perf

    def calculate_daily_nav(
        self,
        date: str | None = None,
        cash: float | None = None,
        market_value: float | None = None,
    ) -> NAV:
        """Calculate NAV for a given date.

        If cash/market_value not provided, reads from store positions. A
        store read failure propagates (NAVDataUnavailableError or the store's own
        error); it is never turned into a zero NAV.

        State (HWM, units, accrued fees) is re-read from the last persisted
        row before ``date`` on every call. In-memory state from a calculation
        that was not saved is discarded: persist the row, then recompute.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.
            cash: Cash balance. If None, uses store.
            market_value: Portfolio market value. If None, uses store.

        Returns:
            NAV record with all fields computed.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        date = self._validate_date(date)

        if cash is None or market_value is None:
            cash, market_value = self._get_portfolio_values()
        cash = float(cash)
        market_value = float(market_value)
        if not math.isfinite(cash) or not math.isfinite(market_value):
            raise NAVDataUnavailableError(
                f"non-finite portfolio values cash={cash!r} "
                f"market_value={market_value!r}"
            )

        gross_assets = cash + market_value

        # Previous record strictly before this date, and the calendar days
        # elapsed.
        try:
            history = self._store.get_nav_history(days=self.MAX_HISTORY_DAYS)
        except AttributeError:
            history = []
        previous = [row for row in history if str(row.get("date", "")) < date]

        # State is always re-derived from the last persisted row *before* the
        # date being calculated. Two bugs shared this root:
        #   - a backdated date (no exact same-day row) kept whatever state the
        #     constructor had restored from a LATER row, so June inherited
        #     December's fees and high-water mark;
        #   - a second calculator instance that was created before the first
        #     day was saved accrued on its stale in-memory zero for day two.
        # In-memory state from an unsaved calculation is deliberately
        # discarded: persist the row, then recompute.
        if previous:
            self._apply_state(previous[-1])
        else:
            self._high_water_mark = self._initial_nav_per_unit
            self._accrued_mgmt_fee = 0.0
            self._accrued_perf_fee = 0.0
            self._total_units = self._initial_units

        elapsed_days = 1
        if previous:
            try:
                prev_dt = datetime.strptime(str(previous[-1]["date"])[:10], "%Y-%m-%d")
                cur_dt = datetime.strptime(date[:10], "%Y-%m-%d")
                elapsed_days = max(1, (cur_dt - prev_dt).days)
            except (TypeError, ValueError):
                elapsed_days = 1

        # Management fee is a liability: accrue it cumulatively and deduct the
        # whole accrual from gross assets every day. The base is net of all
        # fees already owed, so neither accrual compounds on money that no
        # longer belongs to investors.
        fee_base = max(
            0.0,
            gross_assets - self._accrued_mgmt_fee - self._accrued_perf_fee,
        )
        mgmt_fee = self._accrue_mgmt_fee(fee_base, days=elapsed_days)
        self._accrued_mgmt_fee += mgmt_fee

        gross_nav = gross_assets - self._accrued_mgmt_fee
        nav_per_unit_gross = gross_nav / self._total_units if self._total_units > 0 else 0

        # Performance fee: 20% of the new high above the high-water mark.
        # The HWM is the gross-of-performance-fee peak per unit, so the fee is
        # charged once per new high. The old code set the HWM to the *net*
        # NAV, which let the same gain be charged again after the fee.
        if nav_per_unit_gross > self._high_water_mark:
            perf_fee = self._accrue_perf_fee(nav_per_unit_gross)
            self._high_water_mark = nav_per_unit_gross
        else:
            perf_fee = 0.0
        self._accrued_perf_fee += perf_fee

        # Net NAV. The whole accrued performance-fee liability stays deducted;
        # subtracting only today's fee added every earlier fee back to NAV.
        nav_total = gross_nav - self._accrued_perf_fee
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
        """Persist NAV record to store.

        Refuses to persist a non-positive or non-finite NAV: a zero NAV is
        indistinguishable from "the store read failed" and would be recorded
        as a -100% cumulative return. The date must be strict ``YYYY-MM-DD``
        because state selection compares date strings lexicographically.
        """
        self._validate_date(nav.date)
        if not math.isfinite(nav.nav_total) or nav.nav_total <= 0:
            raise NAVDataUnavailableError(
                f"refusing to persist NAV total {nav.nav_total!r} "
                f"(date={nav.date})"
            )
        if not math.isfinite(nav.nav_per_unit) or nav.nav_per_unit <= 0:
            raise NAVDataUnavailableError(
                f"refusing to persist NAV per unit {nav.nav_per_unit!r} "
                f"(date={nav.date})"
            )
        self._store.save_nav(nav.to_dict())

    def update_daily_nav(
        self,
        date: str | None = None,
        cash: float | None = None,
        market_value: float | None = None,
    ) -> NAV:
        """Calculate and save daily NAV (convenience method).

        Intended to be called at end-of-day by the trading engine. A NAV that
        cannot be calculated from trustworthy inputs raises; nothing is
        persisted.

        Only the latest date can be recalculated safely: replacing an earlier
        row does NOT cascade a recompute to the later rows, whose accrued
        fees/HWM were derived from the old values. A backdated correction
        needs a full recompute from that date forward.
        """
        nav = self.calculate_daily_nav(date=date, cash=cash, market_value=market_value)
        self.save_nav(nav)
        logger.info(
            "NAV updated: date=%s nav/unit=%.4f total=%.2f mgmt_fee=%.2f perf_fee=%.2f",
            nav.date, nav.nav_per_unit, nav.nav_total, nav.mgmt_fee, nav.perf_fee,
        )
        return nav

    @staticmethod
    def _validate_date(date: str) -> str:
        """Require a strict, zero-padded YYYY-MM-DD date string."""
        if not isinstance(date, str) or _DATE_SHAPE.fullmatch(date) is None:
            raise ValueError(f"date must be YYYY-MM-DD, got {date!r}")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"date must be YYYY-MM-DD, got {date!r}") from None
        return date

    def _get_portfolio_values(self) -> tuple[float, float]:
        """Get cash and market value from store positions.

        Fail closed: store read failures propagate instead of degrading to
        ``(0, 0)``. An empty P&L history means the cash balance is unknown
        (not zero), so that is refused too.
        """
        positions = self._store.get_positions()
        market_value = sum(
            float(p.get("market_value", 0) or 0) for p in positions
        )
        # Latest snapshot regardless of age: a 24-hour window returns
        # nothing on a Monday and silently set cash to zero.
        pnl = self._store.get_pnl_history(days=self.MAX_HISTORY_DAYS)
        if not pnl:
            raise NAVDataUnavailableError(
                "no P&L snapshot available to source the cash balance"
            )
        cash = float(pnl[-1].get("cash", 0) or 0)
        return cash, market_value

    def _accrue_mgmt_fee(self, nav: float, days: int = 1) -> float:
        """Management fee accrual.

        mgmt_fee = NAV * annual_rate / 365 * elapsed calendar days
        """
        return nav * self._annual_mgmt_fee / self.DAYS_PER_YEAR * days

    def _accrue_perf_fee(self, nav_per_unit: float) -> float:
        """Performance fee with high-water mark.

        Only charged when nav_per_unit exceeds the high-water mark.
        Fee = (nav_per_unit - HWM) * total_units * perf_fee_rate
        """
        if nav_per_unit <= self._high_water_mark:
            return 0.0
        gain = nav_per_unit - self._high_water_mark
        return gain * self._total_units * self._perf_fee_rate
