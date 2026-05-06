"""Portfolio constraints for optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PortfolioConstraints:
    """Investment constraints for portfolio optimization.

    Attributes:
        long_only: Only positive weights (no shorting).
        max_weight: Maximum weight per asset (e.g., 0.05 = 5%).
        max_sector_exposure: Maximum total weight per sector.
        max_turnover: Maximum one-sided turnover per rebalance.
        lot_size: Minimum trading unit (100 shares for A-shares).
        risk_aversion: Trade-off between return and risk.
    """
    long_only: bool = True
    max_weight: float = 0.05
    max_sector_exposure: float = 0.30
    max_turnover: float = 0.30
    lot_size: int = 100
    risk_aversion: float = 1.0

    @classmethod
    def from_config(cls, config) -> PortfolioConstraints:
        """Create from a PortfolioConstraintsConfig dataclass."""
        return cls(
            long_only=config.long_only,
            max_weight=config.max_weight,
            max_sector_exposure=config.max_sector_exposure,
            max_turnover=config.max_turnover,
            lot_size=config.lot_size,
        )
