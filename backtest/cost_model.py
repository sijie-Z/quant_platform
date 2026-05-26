"""Transaction cost model for A-share market and cross-asset.

Models realistic trading costs:
- Commission: ~0.03% (万三) per trade, both buy and sell
- Stamp tax: 0.1% (千一) on sell only (A-share specific, 0 for futures/ETFs)
- Slippage: Market impact, modeled as fixed or proportional to trade size

Cross-asset support:
- Per-instrument commission/stamp_tax rates via AssetUniverse
- Futures use per-lot fixed commission instead of notional rate
- ETFs exempt from stamp tax
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CostModel:
    """A-share transaction cost model with cross-asset support."""

    def __init__(
        self,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
        slippage: float = 0.001,
        slippage_model: str = "fixed",
        asset_universe=None,
    ):
        """
        Args:
            commission: Commission rate per trade (default 0.03%).
            stamp_tax: Stamp tax on sell only (default 0.1%).
            slippage: Slippage assumption (default 0.1%).
            slippage_model: 'fixed' for constant bps, 'proportional' for
                           size-dependent impact.
            asset_universe: AssetUniverse for per-instrument cost overrides.
        """
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.slippage_model = slippage_model
        self.asset_universe = asset_universe

    def compute_costs(
        self,
        turnover,
        is_sell=None,
        daily_volume=None,
        symbol=None,
    ):
        """Compute total transaction costs as a proportion of trade value.

        Args:
            turnover: Trade value (scalar, Series, or array).
            is_sell: Boolean Series, True for sell trades.
            daily_volume: Daily volume per asset for proportional slippage.
            symbol: Instrument symbol for per-instrument cost lookup.

        Returns:
            Total cost as proportion of trade value (same type as turnover).
        """
        # Look up per-instrument rates if available
        commission_rate = self.commission
        stamp_tax_rate = self.stamp_tax
        if symbol and self.asset_universe is not None:
            inst = self.asset_universe.get(symbol)
            if inst is not None:
                if inst.commission_rate is not None:
                    commission_rate = inst.commission_rate
                if inst.stamp_tax_rate is not None:
                    stamp_tax_rate = inst.stamp_tax_rate

        abs_turnover = abs(turnover) if isinstance(turnover, (int, float)) else turnover.abs()

        commission_cost = abs_turnover * commission_rate

        if is_sell is not None:
            stamp_cost = abs_turnover * stamp_tax_rate * is_sell.astype(float)
        else:
            stamp_cost = abs_turnover * stamp_tax_rate * 0.5

        if self.slippage_model == "proportional" and daily_volume is not None:
            participation = abs_turnover / daily_volume.clip(lower=1)
            slippage_cost = abs_turnover * self.slippage * np.sqrt(participation)
        else:
            slippage_cost = abs_turnover * self.slippage

        return commission_cost + stamp_cost + slippage_cost

    def compute_costs_instrument(self, price: float, quantity: int,
                                 side: str = "buy", symbol: str = "") -> float:
        """Compute absolute cost for a single trade using instrument metadata.

        Used by broker/engine when instrument info is available.
        Falls back to global rates if no instrument found.
        """
        if symbol and self.asset_universe is not None:
            inst = self.asset_universe.get(symbol)
            if inst is not None:
                notional = inst.notional(price, quantity)
                comm = inst.commission(price, quantity, side)
                tax = inst.stamp_tax(price, quantity, side)
                slip = notional * self.slippage
                return comm + tax + slip

        notional = price * quantity
        comm = notional * self.commission
        tax = notional * self.stamp_tax if side == "sell" else 0.0
        slip = notional * self.slippage
        return comm + tax + slip
