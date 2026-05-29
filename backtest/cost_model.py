"""Transaction cost model for A-share market and cross-asset.

Models realistic trading costs:
- Commission: ~0.03% (万三) per trade, both buy and sell
- Stamp tax: 0.1% (千一) on sell only (A-share specific, 0 for futures/ETFs)
- Market impact: fixed slippage OR sophisticated model (Almgren-Chriss, Square-Root)

The CostModel uses the market impact model from execution/market_impact.py when
volume and volatility data are available. Otherwise falls back to fixed slippage.

Cross-asset support:
- Per-instrument commission/stamp_tax rates via AssetUniverse
- Futures use per-lot fixed commission instead of notional rate
- ETFs exempt from stamp tax
"""

from __future__ import annotations

import numpy as np

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class CostModel:
    """A-share transaction cost model with market impact integration.

    Supports two slippage modes:
    - "fixed": Constant bps slippage (fast, no data needed).
    - "impact": Uses ExecutionCostCalculator (Almgren-Chriss + Square-Root ensemble).
      Requires daily_volume and volatility per asset. More accurate for large orders.
    """

    def __init__(
        self,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
        slippage: float = 0.001,
        slippage_model: str = "fixed",
        asset_universe=None,
        impact_model: str = "composite",  # composite, almgren_chriss, square_root
    ):
        """
        Args:
            commission: Commission rate per trade (default 0.03%).
            stamp_tax: Stamp tax on sell only (default 0.1%).
            slippage: Slippage assumption (default 0.1%). Used as fixed bps or
                      as baseline for impact model.
            slippage_model: 'fixed' for constant bps, 'proportional' for
                           size-dependent impact, 'impact' for full market impact model.
            asset_universe: AssetUniverse for per-instrument cost overrides.
            impact_model: Which impact model to use when slippage_model='impact'.
        """
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage
        self.slippage_model = slippage_model
        self.asset_universe = asset_universe

        # Lazy-load impact model (only when needed)
        self._impact_calculator = None
        self._impact_model_name = impact_model

    def compute_costs(
        self,
        turnover,
        is_sell=None,
        daily_volume=None,
        symbol=None,
        volatility=None,
    ):
        """Compute total transaction costs as a proportion of trade value.

        Args:
            turnover: Trade value (scalar, Series, or array).
            is_sell: Boolean Series, True for sell trades.
            daily_volume: Daily volume per asset for proportional/impact slippage.
            symbol: Instrument symbol for per-instrument cost lookup.
            volatility: Daily volatility for impact model (annualized / sqrt(252)).

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

        # --- Market impact / slippage ---
        if self.slippage_model == "impact" and daily_volume is not None and volatility is not None:
            # Use full market impact model
            slippage_cost = self._compute_impact_cost(
                abs_turnover, daily_volume, volatility
            )
            logger.debug("Impact cost: turnover=%.2f, cost=%.2f", abs_turnover, slippage_cost)
        elif self.slippage_model == "proportional" and daily_volume is not None:
            # Size-dependent slippage: impact ∝ sqrt(participation)
            participation = abs_turnover / daily_volume.clip(lower=1)
            slippage_cost = abs_turnover * self.slippage * np.sqrt(participation)
        else:
            # Fixed slippage (default, fast)
            slippage_cost = abs_turnover * self.slippage

        return commission_cost + stamp_cost + slippage_cost

    def _compute_impact_cost(self, turnover, daily_volume, volatility):
        """Compute market impact cost using the sophisticated impact model."""
        if self._impact_calculator is None:
            self._init_impact_calculator()

        # Handle both scalar and array inputs
        if isinstance(turnover, (int, float)):
            # Estimate quantity from turnover and an average price
            # We use a reference price of ~10 for A-shares when price is unknown
            # This gives a rough quantity estimate for impact calculation
            price = 10.0
            quantity = max(1, int(turnover / price))
            vol = volatility if isinstance(volatility, (int, float)) else 0.02
            mkt_vol = max(1, int(daily_volume)) if isinstance(daily_volume, (int, float)) else 1

            cost = self._impact_calculator.calculate(
                order_quantity=quantity,
                price=price,
                side="buy",
                market_volume=mkt_vol,
                volatility=vol,
            )
            return cost.total_cost

        # Array/Series path
        # Vectorized: compute bps cost per unit of turnover
        # Simplified — use square-root scaling per asset
        participation = turnover / daily_volume.clip(lower=1)
        impact_bps = np.sqrt(participation) * volatility.clip(lower=0.001)
        return turnover * impact_bps

    def _init_impact_calculator(self):
        """Lazy-init the market impact calculator."""
        try:
            from quant_platform.execution.market_impact import (
                CompositeImpactModel,
                ExecutionCostCalculator,
            )
            self._impact_calculator = ExecutionCostCalculator(
                impact_model=CompositeImpactModel(),
                commission_rate=self.commission,
                stamp_tax_rate=self.stamp_tax,
            )
            logger.info("Market impact model initialized: %s", self._impact_model_name)
        except ImportError as e:
            logger.warning("Market impact model not available, falling back to fixed: %s", e)
            self.slippage_model = "fixed"

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
