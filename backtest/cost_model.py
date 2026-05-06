"""Transaction cost model for A-share market.

Models realistic trading costs:
- Commission: ~0.03% (万三) per trade, both buy and sell
- Stamp tax: 0.1% (千一) on sell only (A-share specific)
- Slippage: Market impact, modeled as fixed or proportional to trade size
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CostModel:
    """A-share transaction cost model."""

    def __init__(
        self,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
        slippage: float = 0.001,
        slippage_model: str = "fixed",
    ):
        """
        Args:
            commission: Commission rate per trade (default 0.03%).
            stamp_tax: Stamp tax on sell only (default 0.1%).
            slippage: Slippage assumption (default 0.1%).
            slippage_model: 'fixed' for constant bps, 'proportional' for
                           size-dependent impact.
        """
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.slippage_model = slippage_model

    def compute_costs(
        self,
        turnover,
        is_sell=None,
        daily_volume=None,
    ):
        """Compute total transaction costs as a proportion of trade value.

        Args:
            turnover: Trade value (scalar, Series, or array).
            is_sell: Boolean Series, True for sell trades.
            daily_volume: Daily volume per asset for proportional slippage.

        Returns:
            Total cost as proportion of trade value (same type as turnover).
        """
        abs_turnover = abs(turnover) if isinstance(turnover, (int, float)) else turnover.abs()

        commission_cost = abs_turnover * self.commission

        if is_sell is not None:
            stamp_cost = abs_turnover * self.stamp_tax * is_sell.astype(float)
        else:
            stamp_cost = abs_turnover * self.stamp_tax * 0.5

        if self.slippage_model == "proportional" and daily_volume is not None:
            participation = abs_turnover / daily_volume.clip(lower=1)
            slippage_cost = abs_turnover * self.slippage * np.sqrt(participation)
        else:
            slippage_cost = abs_turnover * self.slippage

        return commission_cost + stamp_cost + slippage_cost
