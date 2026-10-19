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
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def _common_index(*values):
    """Label index shared by the pandas Series among `values`, or None."""
    indexes = [value.index for value in values if isinstance(value, pd.Series)]
    if not indexes:
        return None
    index = indexes[0]
    for other in indexes[1:]:
        index = index.union(other)
    return index


def _aligned_arrays(*values, index=None):
    """Convert values to float arrays, reindexing Series onto `index`.

    The vectorised cost paths used to rely on pandas label alignment. A plain
    ``np.asarray`` conversion pairs values by position, which silently
    mismatches assets when callers pass Series in different orders.
    """
    arrays = []
    for value in values:
        if isinstance(value, pd.Series):
            if index is not None:
                value = value.reindex(index)
            arrays.append(value.to_numpy(dtype=float))
        else:
            arrays.append(np.asarray(value, dtype=float))
    return arrays


class CostModel:
    """A-share transaction cost model with market impact integration.

    Supports two slippage modes:
    - "fixed": Constant bps slippage (fast, no data needed).
    - "impact": Uses ExecutionCostCalculator (Almgren-Chriss + Square-Root ensemble).
      Requires daily_volume (average daily traded value in CNY, 人民币成交额)
      and volatility per asset. More accurate for large orders.
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
        # Set once, the first time a configured model cannot be used, so the
        # warning names the problem without repeating on every rebalance.
        self._slippage_fallback_warned = False

    def compute_costs(
        self,
        turnover,
        is_sell=None,
        daily_volume=None,
        symbol=None,
        volatility=None,
        buy_turnover=None,
        sell_turnover=None,
    ):
        """Compute total transaction costs as a proportion of trade value.

        Args:
            turnover: Trade value (scalar, Series, or array). Only used for
                the legacy single-figure path; when ``buy_turnover`` /
                ``sell_turnover`` are supplied, costs are computed from the
                actual split instead.
            is_sell: Boolean Series, True for sell trades.
            daily_volume: Average daily traded value per asset in CNY
                (人民币成交额), **not** a share count. Used by the
                proportional and impact slippage models; participation is
                traded CNY / daily traded CNY.
            symbol: Instrument symbol for per-instrument cost lookup.
            volatility: Daily volatility for impact model (annualized / sqrt(252)).
            buy_turnover: Per-asset buy notional (same units as ``turnover``).
                When given, commission and slippage are charged on the total
                two-way value and stamp tax on ``sell_turnover`` only.
            sell_turnover: Per-asset sell notional (same units as ``turnover``).

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

        if buy_turnover is not None or sell_turnover is not None:
            # Explicit buy/sell split: stamp tax is a sell-side tax, so it is
            # charged on the sell notional only. Charging it on a total with
            # `is_sell=None` (the legacy path below) assumes half of every
            # trade is a sell, which overcharges from-cash builds and
            # undercharges full liquidations.
            buy = np.abs(0.0 if buy_turnover is None else buy_turnover)
            sell = np.abs(0.0 if sell_turnover is None else sell_turnover)
            traded = buy + sell
            commission_cost = traded * commission_rate
            stamp_cost = sell * stamp_tax_rate
        else:
            abs_turnover = np.abs(turnover)
            traded = abs_turnover
            commission_cost = abs_turnover * commission_rate

            if is_sell is not None:
                stamp_cost = abs_turnover * stamp_tax_rate * is_sell.astype(float)
            else:
                stamp_cost = abs_turnover * stamp_tax_rate * 0.5

        # --- Market impact / slippage ---
        if self.slippage_model == "impact" and daily_volume is not None and volatility is not None:
            # `_compute_impact_cost` returns a dimensionless impact rate
            # (impact cost / traded notional).
            impact_rate = self._compute_impact_cost(traded, daily_volume, volatility)
            slippage_cost = traded * impact_rate
            logger.debug("Impact cost: turnover=%s, rate=%s", traded, impact_rate)
        elif self.slippage_model == "proportional" and daily_volume is not None:
            # Size-dependent slippage: impact ∝ sqrt(participation) of daily
            # traded value (both in CNY). Series inputs are aligned by label.
            index = _common_index(traded, daily_volume)
            traded_values, volume = _aligned_arrays(traded, daily_volume, index=index)
            participation = np.where(
                np.asarray(traded_values) != 0,
                np.asarray(traded_values) / np.clip(volume, 1.0, None),
                0.0,
            )
            slippage_cost = traded_values * self.slippage * np.sqrt(participation)
            if index is not None:
                slippage_cost = pd.Series(slippage_cost, index=index)
        else:
            # Fixed slippage (default, fast)
            #
            # Reached whenever a non-fixed model was configured but the inputs
            # it needs were not supplied. That used to be completely silent:
            # `config/default.yaml` asked for "impact", `BacktestEngine` passed
            # neither `daily_volume` nor `volatility`, and every rebalance was
            # charged fixed slippage while the config claimed otherwise. The
            # numbers were not wrong, but the description of them was.
            if self.slippage_model != "fixed":
                missing = [
                    name for name, value in (
                        ("daily_volume", daily_volume),
                        ("volatility", volatility),
                    )
                    if value is None
                ]
                if not self._slippage_fallback_warned:
                    self._slippage_fallback_warned = True
                    logger.warning(
                        "slippage_model=%r is not in use: compute_costs() was "
                        "given no %s. Falling back to fixed slippage of %.1f "
                        "bps -- the configured model is NOT what is running.",
                        self.slippage_model,
                        " or ".join(missing) or "supporting inputs",
                        self.slippage * 10000,
                    )
            slippage_cost = traded * self.slippage

        return commission_cost + stamp_cost + slippage_cost

    # Reference price used to translate a CNY traded value into a share
    # quantity for the impact model, which takes shares. The square-root and
    # Almgren-Chriss components are price-invariant.
    _IMPACT_REFERENCE_PRICE = 10.0

    def _compute_impact_cost(self, turnover, daily_volume, volatility):
        """Market impact as a fraction of traded notional (CNY / CNY).

        Args:
            turnover: Traded value in CNY.
            daily_volume: Average daily traded value in CNY (人民币成交额),
                the same unit as ``turnover``.
            volatility: Daily volatility.

        Returns:
            Impact cost divided by the traded notional, i.e. the impact
            return the caller multiplies by the traded value.

        Only ``ExecutionCostCalculator.calculate().market_impact.total`` is
        used. `total_cost` also contains commission (with a CNY 5 minimum)
        and stamp tax, and it adds a *per-share* impact figure to those
        currency amounts; returning it here double-charged commission and
        stamp tax and put the result in the wrong units. `market_impact.total`
        is a per-share price impact, so it is multiplied by the share
        quantity (derived from the CNY turnover at the reference price) and
        divided by the CNY turnover to obtain a dimensionless rate.

        The vector path uses the square-root impact rule directly:
        ``sqrt(turnover / daily_volume) * volatility``. Since both volume
        figures are traded value in CNY, no price conversion is needed and
        the participation rate is price-free. Note that the two branches use
        different models: the single-order (all-scalar) branch applies the
        configured impact ensemble, while the vector branch applies the pure
        square-root rule, so callers should keep shapes consistent within a
        call.
        """
        if self._impact_calculator is None:
            self._init_impact_calculator()
        if self._impact_calculator is None:
            # The import failed and `_init_impact_calculator` switched to
            # fixed slippage; return its rate rather than dereferencing None.
            return self.slippage

        if np.isscalar(turnover) and np.isscalar(daily_volume) and np.isscalar(volatility):
            traded = float(turnover)
            if traded <= 0:
                return 0.0
            price = self._IMPACT_REFERENCE_PRICE
            quantity = max(1, int(traded / price))
            # daily_volume is traded value in CNY; the impact model takes a
            # share count, so use the same reference price as for quantity.
            market_volume = max(1, int(float(daily_volume) / price))
            cost = self._impact_calculator.calculate(
                order_quantity=quantity,
                price=price,
                side="buy",
                market_volume=market_volume,
                volatility=float(volatility),
            )
            return float(cost.market_impact.total) * quantity / traded

        # Array/Series path: impact rate per unit of traded value. Series
        # inputs are aligned by label (see _aligned_arrays). A scalar traded
        # value against a per-asset volume vector is broadcast here rather
        # than fed to the single-order calculator.
        index = _common_index(turnover, daily_volume, volatility)
        traded, volume, vol = _aligned_arrays(
            turnover, daily_volume, volatility, index=index,
        )
        participation = np.where(
            np.asarray(traded) != 0,
            np.abs(np.asarray(traded)) / np.clip(volume, 1.0, None),
            0.0,
        )
        rate = np.sqrt(participation) * np.clip(vol, 0.001, None)
        return pd.Series(rate, index=index) if index is not None else rate

    def _init_impact_calculator(self):
        """Lazy-init the market impact calculator."""
        try:
            from quant_core.market_impact import (
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
