"""Fundamental factors for A-share multi-factor strategies.

Implements commonly used fundamental factors:
- log_market_cap: Natural log of market capitalization (size factor)
- pb_ratio: Price-to-book ratio (value factor)
- pe_ratio: Price-to-earnings ratio (value factor)
- roe: Return on equity (quality/profitability factor)
- asset_growth: Year-over-year asset growth rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.factors.base import BaseFactor, FactorCategory
from quant_platform.factors.registry import get_registry


class LogMarketCap(BaseFactor):
    """Natural log of market capitalization.

    Smaller stocks tend to outperform (size premium). Log transform
    reduces skew from very large caps.
    """

    category = FactorCategory.FUNDAMENTAL

    @property
    def name(self) -> str:
        return "log_market_cap"

    def compute(
        self, prices: pd.DataFrame, financials: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        if financials is None:
            raise ValueError("Financial data required for log_market_cap")
        mcap = financials["market_cap"].copy()
        return np.log(mcap.clip(lower=1))  # Clip to avoid log(0) or log(negative)


class PbRatio(BaseFactor):
    """Price-to-book ratio (inverted: book/price for value factor direction).

    Lower PB = cheaper (value). We negate so higher factor = cheaper.
    This ensures the factor direction matches alpha (higher = better).
    """

    category = FactorCategory.FUNDAMENTAL

    @property
    def name(self) -> str:
        return "pb_ratio"

    def compute(
        self, prices: pd.DataFrame, financials: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        if financials is None:
            raise ValueError("Financial data required for pb_ratio")
        pb = financials["pb_ratio"].copy()
        # Invert: higher factor = cheaper stock (book/price)
        return -pb


class PeRatio(BaseFactor):
    """Price-to-earnings ratio (inverted for alpha direction).

    Lower PE = cheaper. Negate so higher = better.
    """

    category = FactorCategory.FUNDAMENTAL

    @property
    def name(self) -> str:
        return "pe_ratio"

    def compute(
        self, prices: pd.DataFrame, financials: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        if financials is None:
            raise ValueError("Financial data required for pe_ratio")
        pe = financials["pe_ratio"].copy()
        # Handle negative earnings (PE is undefined)
        pe = pe.clip(lower=1)  # Cap at 1 to avoid extreme negative values
        return -pe


class ROE(BaseFactor):
    """Return on Equity: net income / shareholders' equity.

    Higher ROE indicates more profitable companies (quality factor).
    """

    category = FactorCategory.FUNDAMENTAL

    @property
    def name(self) -> str:
        return "roe"

    def compute(
        self, prices: pd.DataFrame, financials: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        if financials is None:
            raise ValueError("Financial data required for roe")
        return financials["roe"].copy()


class AssetGrowth(BaseFactor):
    """Year-over-year total asset growth rate.

    High asset growth may indicate overinvestment and lower future returns
    (asset growth anomaly). We negate so higher factor = lower growth = better.
    """

    category = FactorCategory.FUNDAMENTAL

    @property
    def name(self) -> str:
        return "asset_growth"

    def compute(
        self, prices: pd.DataFrame, financials: pd.DataFrame | None = None, **kwargs
    ) -> pd.DataFrame:
        if financials is None:
            raise ValueError("Financial data required for asset_growth")
        growth = financials["asset_growth"].copy()
        # Negate: asset growth anomaly — high growth predicts lower returns
        return -growth


def register_all():
    registry = get_registry()
    for cls in [LogMarketCap, PbRatio, PeRatio, ROE, AssetGrowth]:
        registry.register(cls)
