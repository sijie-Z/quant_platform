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

All headline metrics in one :class:`InvestorView` describe the *same* window
(the full stored history by default, or the explicit ``days`` lookback), so a
return, its annualization, the drawdown and the Sharpe ratio never mix
periods. ``InvestorView.window_days`` records the window used.

The store's ``get_nav_history`` clamps every window to 3650 days; the portal
uses the same documented cap (:attr:`InvestorPortal.MAX_HISTORY_DAYS`)
instead of silently asking for 30 years and reporting a 10-year window as
inception.

Note: this is an operations API with no runtime caller yet (tracked as R52);
it is kept rather than deleted so the investor surface can be wired without
reimplementing the metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_platform.backtest.metrics import (
    annualized_volatility as _annualized_volatility,
)
from quant_platform.backtest.metrics import (
    sharpe_ratio as _sharpe_ratio,
)
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert NaN/inf (and pandas NA) to None so json.dumps() is valid."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _rounded(value: Any, digits: int) -> Any:
    value = _json_safe(value)
    return None if value is None else round(value, digits)


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
    window_days: int = 0

    def to_dict(self) -> dict:
        """JSON-serializable representation.

        Dates become ISO strings and NaN becomes None; pandas Timestamps and
        NaN previously made ``json.dumps(view.to_dict())`` raise TypeError.
        """
        nav_curve = self.nav_curve.copy()
        if "date" in nav_curve.columns and not nav_curve.empty:
            nav_curve["date"] = pd.to_datetime(
                nav_curve["date"]
            ).dt.strftime("%Y-%m-%d")
        nav_records = [
            {key: _json_safe(value) for key, value in record.items()}
            for record in nav_curve.to_dict(orient="records")
        ]

        monthly: dict[int, dict[str, Any]] = {}
        for year, row in self.monthly_returns.to_dict(orient="index").items():
            monthly[int(year)] = {
                str(month): _json_safe(value) for month, value in row.items()
            }

        return {
            "fund_name": self.fund_name,
            "inception_date": self.inception_date,
            "latest_nav": _rounded(self.latest_nav, 4),
            "aum": _rounded(self.aum, 2),
            "cumulative_return": _rounded(self.cumulative_return, 4),
            "annualized_return": _rounded(self.annualized_return, 4),
            "max_drawdown": _rounded(self.max_drawdown, 4),
            "sharpe_ratio": _rounded(self.sharpe_ratio, 4),
            "volatility": _rounded(self.volatility, 4),
            "window_days": self.window_days,
            "nav_curve": nav_records,
            "monthly_returns": monthly,
        }


class InvestorPortal:
    """Investor-facing fund performance portal.

    Reads NAV history from store and computes performance metrics
    without exposing position-level details.

    Args:
        store: Store instance for NAV data.
        fund_name: Display name for the fund.
    """

    #: Same cap as Store.get_nav_history's _clamp_days maximum.
    MAX_HISTORY_DAYS = 3650
    #: Annualizing a handful of days produces absurd numbers (2 records and a
    #: 1% move once reported 3677%/yr); below this span the view reports
    #: 0.0 instead.
    MIN_ANNUALIZATION_DAYS = 30

    def __init__(self, store: Any, fund_name: str = "量化多因子基金"):
        self._store = store
        self._fund_name = fund_name

    def get_investor_view(self, days: int | None = None) -> InvestorView:
        """Generate investor performance view.

        Args:
            days: Lookback period in days. ``None`` (default) uses the full
                history the store can return (capped at MAX_HISTORY_DAYS).

        Returns:
            InvestorView with performance metrics, all computed over the
            same window.
        """
        window = (
            self.MAX_HISTORY_DAYS
            if days is None
            else min(max(1, int(days)), self.MAX_HISTORY_DAYS)
        )
        nav_records = self._store.get_nav_history(days=window)

        if not nav_records and window < self.MAX_HISTORY_DAYS:
            # A fund dormant for longer than the requested lookback still has
            # history; the old code returned an all-zero view before ever
            # asking for it.
            logger.info(
                "Lookback of %d days is empty; falling back to full history",
                window,
            )
            nav_records = self._store.get_nav_history(
                days=self.MAX_HISTORY_DAYS
            )
            window = self.MAX_HISTORY_DAYS

        if not nav_records:
            return self._empty_view(window)

        # Build DataFrame
        df = pd.DataFrame(nav_records)
        if "nav_per_unit" not in df.columns:
            return self._empty_view(window)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["nav_per_unit"] = pd.to_numeric(df["nav_per_unit"], errors="coerce")
        df = df.dropna(subset=["nav_per_unit"])

        if len(df) < 2:
            return self._empty_view(window)

        # NAV curve
        nav_curve = df[["nav_per_unit"]].copy()
        nav_curve["daily_return"] = nav_curve["nav_per_unit"].pct_change()

        # Performance metrics — one window for every headline number.
        latest_nav = float(df["nav_per_unit"].iloc[-1])
        inception_nav = float(df["nav_per_unit"].iloc[0])
        inception_date = str(df.index[0].date())
        cumulative_return = latest_nav / inception_nav - 1

        elapsed_days = (df.index[-1] - df.index[0]).days
        # Annualize over the fund's actual elapsed life. Using the first and
        # last record of a short lookback divided the *inception* return by a
        # window as short as one day: a 6-year fund at +50% reported
        # 17.65%/yr instead of ~7%. The same guard keeps a 2-record series
        # from reporting thousands of percent per year.
        if (
            elapsed_days >= self.MIN_ANNUALIZATION_DAYS
            and cumulative_return > -1
        ):
            elapsed_years = elapsed_days / 365.25
            annualized_return = (
                (1 + cumulative_return) ** (1 / elapsed_years) - 1
            )
        else:
            annualized_return = 0.0

        # Drawdown over the same window.
        full_nav = df["nav_per_unit"]
        peak = full_nav.expanding().max()
        drawdown = (full_nav - peak) / peak
        max_drawdown = float(drawdown.min())

        # Volatility and Sharpe. Both use the platform-wide definition from
        # backtest.metrics (calendar-time annualization inferred from the
        # series, risk_free=3%), so the displayed volatility is the exact
        # denominator of the reported Sharpe. The old code annualized the
        # displayed vol with a hardcoded sqrt(252) while Sharpe inferred
        # ~365 periods/year from daily NAV data, leaving the two metrics
        # inconsistent by ~17% on the same series.
        daily_returns = nav_curve["daily_return"].dropna()
        if (
            len(daily_returns) < 2
            or elapsed_days < self.MIN_ANNUALIZATION_DAYS
        ):
            # One return has no sample variance, and annualizing a handful of
            # days produces absurd numbers (3 daily NAVs once implied a
            # 27,000 Sharpe). Undefined, not infinite.
            volatility = 0.0
            sharpe_ratio = 0.0
        else:
            volatility = float(_annualized_volatility(daily_returns))
            sharpe_ratio = float(_sharpe_ratio(daily_returns))

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
            window_days=window,
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

    def _empty_view(self, window_days: int = 0) -> InvestorView:
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
            window_days=window_days,
        )
