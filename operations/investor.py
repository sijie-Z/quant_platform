"""Investor portal — fund performance view without exposing holdings.

Provides a sanitized view of fund performance for investors:
- NAV curve (净值曲线)
- Cumulative return (累计收益)
- Drawdown (回撤)
- Sharpe ratio
- Monthly returns

Intentionally hides:
- Individual stock positions
- Factor weights
- Signal details
- Trade-level data

Usage:
    portal = InvestorPortal(store, nav_calculator)
    view = portal.get_investor_view()
    print(view["nav_curve"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InvestorView:
    """Sanitized fund performance view for investors."""
    nav_curve: pd.DataFrame      # date, nav_per_unit, daily_return
    cumulative_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    monthly_returns: pd.DataFrame  # year x month
    fund_name: str = ""
    inception_date: str = ""
    latest_nav: float = 0.0
    aum: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fund_name": self.fund_name,
            "inception_date": self.inception_date,
            "latest_nav": round(self.latest_nav, 4),
            "aum": round(self.aum, 2),
            "cumulative_return": round(self.cumulative_return, 4),
            "annualized_return": round(self.annualized_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "volatility": round(self.volatility, 4),
            "nav_curve": self.nav_curve.to_dict(orient="records"),
            "monthly_returns": self.monthly_returns.to_dict(orient="index"),
        }


class InvestorPortal:
    """Investor-facing fund performance portal.

    Reads NAV history from store and computes performance metrics
    without exposing position-level details.

    Args:
        store: Store instance for NAV data.
        fund_name: Display name for the fund.
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, store: Any, fund_name: str = "量化多因子基金"):
        self._store = store
        self._fund_name = fund_name

    def get_investor_view(self, days: int = 365 * 3) -> InvestorView:
        """Generate investor performance view.

        Args:
            days: Lookback period in days.

        Returns:
            InvestorView with performance metrics.
        """
        nav_records = self._store.get_nav_history(days=days)

        if not nav_records:
            return self._empty_view()

        # Build DataFrame
        df = pd.DataFrame(nav_records)
        if "nav_per_unit" not in df.columns:
            return self._empty_view()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["nav_per_unit"] = pd.to_numeric(df["nav_per_unit"], errors="coerce")
        df = df.dropna(subset=["nav_per_unit"])

        if len(df) < 2:
            return self._empty_view()

        # NAV curve
        nav_curve = df[["nav_per_unit"]].copy()
        nav_curve["daily_return"] = nav_curve["nav_per_unit"].pct_change()

        # Performance metrics
        latest_nav = float(df["nav_per_unit"].iloc[-1])
        inception_nav = float(df["nav_per_unit"].iloc[0])
        inception_date = str(df.index[0].date())
        # The lookback window is not inception. Read the earliest record we
        # have so cumulative return, annualization and drawdown describe the
        # fund, not the chart range.
        all_records: list[dict] = []
        try:
            all_records = self._store.get_nav_history(days=365 * 30)
        except Exception:
            all_records = []
        if all_records:
            inception_date = str(all_records[0]["date"])
            inception_nav = float(all_records[0]["nav_per_unit"])
        cumulative_return = latest_nav / inception_nav - 1

        # Annualize over the fund's actual elapsed life. Using the first and
        # last record of the 3-year lookback divided the *inception* return
        # by a window as short as one day: a 6-year fund at +50% reported
        # 17.65%/yr instead of ~7%.
        try:
            inception_dt = pd.Timestamp(inception_date)
            elapsed_years = (df.index[-1] - inception_dt).days / 365.25
        except (TypeError, ValueError):
            elapsed_years = (df.index[-1] - df.index[0]).days / 365.25
        annualized_return = (
            (1 + cumulative_return) ** (1 / elapsed_years) - 1
            if elapsed_years > 0 else 0.0
        )

        # Drawdown over the full available history, for the same reason: a
        # drawdown before the lookback window used to be invisible.
        if all_records and "nav_per_unit" in all_records[0]:
            full_nav = pd.Series(
                [float(r["nav_per_unit"]) for r in all_records],
                index=pd.to_datetime([r["date"] for r in all_records]),
            ).sort_index().dropna()
        else:
            full_nav = df["nav_per_unit"]
        peak = full_nav.expanding().max()
        drawdown = (full_nav - peak) / peak
        max_drawdown = float(drawdown.min())

        # Volatility and Sharpe
        daily_returns = nav_curve["daily_return"].dropna()
        volatility = float(daily_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR))
        mean_return = float(daily_returns.mean() * self.TRADING_DAYS_PER_YEAR)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0

        # Monthly returns
        monthly_returns = self._compute_monthly_returns(df)

        # AUM
        aum = 0.0
        if "nav_total" in df.columns:
            aum = float(df["nav_total"].iloc[-1])

        return InvestorView(
            nav_curve=nav_curve.reset_index(),
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            monthly_returns=monthly_returns,
            fund_name=self._fund_name,
            inception_date=inception_date,
            latest_nav=latest_nav,
            aum=aum,
        )

    def _compute_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute monthly return matrix (year x month)."""
        if "nav_per_unit" not in df.columns:
            return pd.DataFrame()

        # `"M"` was removed in pandas 3; MonthEnd works on 2.x and 3.x.
        monthly = df["nav_per_unit"].resample(pd.offsets.MonthEnd()).last()
        # A month with no observation is missing, not a loss of 100%. Dropping
        # it lets the next available month own the multi-month return instead
        # of becoming NaN because its predecessor was NaN.
        monthly = monthly.dropna()
        monthly_ret = monthly.pct_change().dropna()

        if len(monthly_ret) == 0:
            return pd.DataFrame()

        result = {}
        for date, ret in monthly_ret.items():
            year = date.year
            month = date.month
            result.setdefault(year, {})[month] = round(float(ret), 4)

        # Build DataFrame with months as columns
        years = sorted(result.keys())
        months = list(range(1, 13))
        data = []
        for year in years:
            row = [result.get(year, {}).get(m, None) for m in months]
            data.append(row)

        return pd.DataFrame(
            data,
            index=years,
            columns=[f"M{m}" for m in months],
        )

    def _empty_view(self) -> InvestorView:
        """Return empty view when no data is available."""
        return InvestorView(
            nav_curve=pd.DataFrame(columns=["date", "nav_per_unit", "daily_return"]),
            cumulative_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            volatility=0.0,
            monthly_returns=pd.DataFrame(),
            fund_name=self._fund_name,
        )
