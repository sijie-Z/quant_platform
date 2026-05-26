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
        cumulative_return = latest_nav / inception_nav - 1

        n_days = len(df)
        n_years = n_days / self.TRADING_DAYS_PER_YEAR
        annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Drawdown
        peak = df["nav_per_unit"].expanding().max()
        drawdown = (df["nav_per_unit"] - peak) / peak
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

        inception_date = str(df.index[0].date()) if len(df) > 0 else ""

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

        monthly = df["nav_per_unit"].resample("M").last()
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
