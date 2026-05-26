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
        lot_size: Default minimum trading unit (100 shares for A-shares).
                  Overridden per-instrument when asset_universe is provided.
        risk_aversion: Trade-off between return and risk.
        asset_universe: Cross-asset instrument registry for per-instrument
                        lot_size/multiplier lookups. If None, uses default lot_size.
    """
    long_only: bool = True
    max_weight: float = 0.05
    max_sector_exposure: float = 0.30
    max_turnover: float = 0.30
    lot_size: int = 100
    risk_aversion: float = 1.0
    target_volatility: float = 0.0  # Annualized target vol, 0 = disabled
    asset_universe: object | None = None  # AssetUniverse, avoid circular import

    def get_lot_size(self, symbol: str) -> int:
        """Get lot size for a specific symbol.

        Falls back to default lot_size if no instrument found
        or no asset_universe configured (backward compatible).
        """
        if self.asset_universe is not None:
            inst = self.asset_universe.get(symbol)
            if inst is not None:
                return inst.lot_size
        return self.lot_size

    def get_multiplier(self, symbol: str) -> float:
        """Get contract multiplier for a symbol (1.0 for equities)."""
        if self.asset_universe is not None:
            inst = self.asset_universe.get(symbol)
            if inst is not None:
                return inst.multiplier
        return 1.0

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
