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

import pandas as pd

#: Financial fields that are metadata, not factor inputs.  Masking them would
#: turn publish timestamps into NaT for no benefit.
_FINANCIAL_METADATA_FIELDS = frozenset({"publish_date", "fiscal_period_end"})


def mask_financials(
    financials: pd.DataFrame | dict[str, pd.DataFrame] | None,
    excluded: pd.DataFrame,
) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
    """Apply a (date x asset) exclusion mask to a financials panel.

    Accepts the shapes factors are called with: a dict of (date x asset)
    panels, a single-level (date x asset) frame, or the pipeline's
    multi-level (field, asset) frame produced by
    ``financials.unstack("asset")``.
    """
    if financials is None:
        return None

    def _mask_one(panel: pd.DataFrame) -> pd.DataFrame:
        mask = excluded.reindex(
            index=panel.index, columns=panel.columns, fill_value=False,
        )
        return panel.mask(mask)

    if isinstance(financials, dict):
        return {name: _mask_one(panel) for name, panel in financials.items()}

    if not isinstance(financials, pd.DataFrame):
        return financials

    if isinstance(financials.columns, pd.MultiIndex):
        fields = list(financials.columns.get_level_values(0).unique())
        blocks = []
        for field in fields:
            block = financials[field]
            blocks.append(
                block
                if field in _FINANCIAL_METADATA_FIELDS
                else _mask_one(block)
            )
        masked = pd.concat(blocks, axis=1, keys=fields)
        masked.columns.names = financials.columns.names
        return masked

    return _mask_one(financials)


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
        """Compute and return a FactorResult.

        The research price panel blanks suspension days so that *returns*
        never difference across a forward-fill. Level-based factors still
        need a continuous price path: with blanks, one suspension NaN'd every
        momentum value for the next 252 days (momentum_12m coverage fell to
        0/85 on real-shaped data). Factor levels therefore use the stale
        forward-filled close; the IC target remains the masked return panel.

        The panel also carries the full asset-day eligibility mask (ST,
        pre-listing, delisted and suspended rows) in
        ``attrs["excluded_mask"]``.  Those rows must stay blank: a blanket
        `ffill()` used to fill them with a stale close, so excluded names
        re-entered every cross-section.  The same mask is applied to the
        ``financials`` panel, because fundamental factors derive their values
        from financial statements rather than prices and previously bypassed
        the mask entirely (an ST name kept a live ROE/market-cap value while
        its price factors were correctly NaN).

        Set ``fill_suspensions = False`` on a factor that genuinely needs the
        blanks.
        """
        excluded = prices.attrs.get("excluded_mask")
        if getattr(self, "fill_suspensions", True):
            price_panel = prices.ffill()
            if excluded is not None:
                price_panel = price_panel.mask(excluded)
        else:
            price_panel = prices
        if financials is not None:
            if excluded is not None:
                financials = mask_financials(financials, excluded)
            values = self.compute(price_panel, financials=financials, **kwargs)
        else:
            values = self.compute(price_panel, **kwargs)
        return FactorResult(
            name=self.name,
            category=self.category,
            values=values,
            params=self.params,
        )
