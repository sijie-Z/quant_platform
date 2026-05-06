"""Abstract base class for factors.

Every factor must implement:
- compute(): produce raw factor values per asset per date
- name(): unique factor identifier
- category(): technical, fundamental, or custom
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class FactorCategory(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    CUSTOM = "custom"


@dataclass
class FactorResult:
    """Container for computed factor values and metadata."""
    name: str
    category: FactorCategory
    values: pd.DataFrame  # date x asset, raw factor values
    params: dict[str, Any] | None = None


class BaseFactor(ABC):
    """Abstract factor.

    Subclasses implement compute() which returns raw factor values
    as a (date x asset) DataFrame. The framework handles processing
    (winsorization, standardization, neutralization) separately.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique factor name, e.g. 'momentum_1m'."""
        ...

    @property
    @abstractmethod
    def category(self) -> FactorCategory:
        """Factor category for organization and processing."""
        ...

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        financials: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute raw factor values.

        Args:
            prices: DataFrame (date x asset) of price data.
            financials: Optional DataFrame (date x asset) of financial data.

        Returns:
            DataFrame (date x asset) of raw factor values.
        """
        ...

    def run(
        self,
        prices: pd.DataFrame,
        financials: pd.DataFrame | None = None,
        **kwargs,
    ) -> FactorResult:
        """Compute and return a FactorResult."""
        if financials is not None:
            values = self.compute(prices, financials=financials, **kwargs)
        else:
            values = self.compute(prices, **kwargs)
        return FactorResult(
            name=self.name,
            category=self.category,
            values=values,
            params=self.params,
        )
