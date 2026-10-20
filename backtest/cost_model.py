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

# Statutory A-share stamp tax schedule. The rate was cut from 0.1% to 0.05%
# on 2023-08-28; a CostModel built without an explicit `stamp_tax` follows
# these dates when the trade date is supplied. An explicit `stamp_tax` value
# is a flat override (e.g. `config.costs.stamp_tax`).
STAMP_TAX_SCHEDULE: tuple[tuple[pd.Timestamp, float], ...] = (
    (pd.Timestamp("2023-08-28"), 0.0005),
)
STAMP_TAX_BEFORE_SCHEDULE = 0.001


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

    The vectorised path charges commission (with a per-order CNY minimum when
    the per-order notional is known), the both-sides transfer fee, the
    date-dependent sell-side stamp tax, and slippage. Per-instrument
    commission/stamp overrides from an ``AssetUniverse`` are applied when the
    traded Series' index carries the symbols (the backtest engine path).
    """

    def __init__(
        self,
        commission: float = 0.0003,
        stamp_tax: float | None = None,
        slippage: float = 0.001,
        slippage_model: str = "fixed",
        asset_universe=None,
        impact_model: str = "composite",  # composite, almgren_chriss, square_root
        min_commission: float = 5.0,
        transfer_fee: float = 0.00001,
    ):
        """
        Args:
            commission: Commission rate per trade (default 0.03%).
            stamp_tax: Flat stamp tax on sells. ``None`` (default) follows the
                statutory date schedule (0.1% before 2023-08-28, 0.05% from
                that date) when ``compute_costs`` is given a ``date``; without
                a date the pre-change 0.1% rate is used for backward
                compatibility. An explicit value overrides the schedule
                (config override).
            slippage: Slippage assumption (default 0.1%). Used as fixed bps or
                      as baseline for impact model.
            slippage_model: 'fixed' for constant bps, 'proportional' for
                           size-dependent impact, 'impact' for full market impact model.
            asset_universe: AssetUniverse for per-instrument cost overrides.
            impact_model: Which impact model to use when slippage_model='impact'.
            min_commission: Commission floor in CNY per order. Applied on the
                per-asset notional only when ``compute_costs`` is told the
                portfolio value (i.e. the values passed are CNY notionals).
                The weight-based engine is a per-asset-order approximation:
                one order per rebalanced name.
            transfer_fee: Transfer fee rate, both sides (default 0.001% =
                0.00001).
        """
        self.commission = commission
        # Numeric attribute kept for backward compatibility; the schedule is
        # consulted instead when this was left at the default and a date is
        # supplied.
        self._stamp_tax_override = stamp_tax is not None
        self.stamp_tax = STAMP_TAX_BEFORE_SCHEDULE if stamp_tax is None else stamp_tax
        self.slippage = slippage
        self.slippage_model = slippage_model
        self.asset_universe = asset_universe
        self.min_commission = min_commission
        self.transfer_fee = transfer_fee

        # Lazy-load impact model (only when needed)
        self._impact_calculator = None
        self._impact_model_name = impact_model
        # Set once, the first time a configured model cannot be used, so the
        # warning names the problem without repeating on every rebalance.
        self._slippage_fallback_warned = False

    def _stamp_rate(self, date=None):
        """Stamp-tax rate for a trade date (flat override wins)."""
        if self._stamp_tax_override or date is None:
            return self.stamp_tax
        rate = STAMP_TAX_BEFORE_SCHEDULE
        for change_date, new_rate in STAMP_TAX_SCHEDULE:
            if pd.Timestamp(date) >= change_date:
                rate = new_rate
        return rate

    def _instrument_rates(self, traded, symbol, date=None):
        """Per-instrument commission/stamp overrides for a vectorised call.

        Returns ``(commission_rate, stamp_rate)`` where either can be a
        Series aligned to ``traded.index``. The engine passes a Series whose
        index is the asset universe, so the overrides are reachable from the
        vector path as well as from the scalar `symbol=` one (B7).
        """
        base_commission = self.commission
        base_stamp = self._stamp_rate(date)
        if self.asset_universe is None:
            return base_commission, base_stamp

        index = traded.index if isinstance(traded, pd.Series) else None
        if isinstance(symbol, str) and symbol:
            inst = self.asset_universe.get(symbol)
            if inst is None:
                return base_commission, base_stamp
            commission_rate = (
                inst.commission_rate
                if inst.commission_rate is not None else base_commission
            )
            stamp_rate = (
                inst.stamp_tax_rate
                if inst.stamp_tax_rate is not None else base_stamp
            )
            return commission_rate, stamp_rate

        if index is None:
            return base_commission, base_stamp

        commission_series = pd.Series(base_commission, index=index, dtype=float)
        stamp_series = pd.Series(base_stamp, index=index, dtype=float)
        for asset in index:
            inst = self.asset_universe.get(asset) if isinstance(asset, str) else None
            if inst is None:
                continue
            if inst.commission_rate is not None:
                commission_series.loc[asset] = inst.commission_rate
            if inst.stamp_tax_rate is not None:
                stamp_series.loc[asset] = inst.stamp_tax_rate
        return commission_series, stamp_series

    def compute_costs(
        self,
        turnover,
        is_sell=None,
        daily_volume=None,
        symbol=None,
        volatility=None,
        buy_turnover=None,
        sell_turnover=None,
        date=None,
        portfolio_value=None,
    ):
        """Compute total transaction costs.

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
            symbol: Instrument symbol for per-instrument cost lookup. For a
                Series input, either a scalar symbol (single-asset Series) or
                a sequence of symbols aligned to the Series. When omitted the
                Series index itself is treated as the symbol list, which is
                what the backtest engine relies on (B7).
            volatility: Daily volatility for impact model (annualized / sqrt(252)).
            buy_turnover: Per-asset buy notional (same units as ``turnover``).
                When given, commission and slippage are charged on the total
                two-way value and stamp tax on ``sell_turnover`` only.
            sell_turnover: Per-asset sell notional (same units as ``turnover``).
            date: Trade date. Selects the statutory stamp-tax rate when no
                flat ``stamp_tax`` override was configured (B11).
            portfolio_value: Current portfolio value in CNY. When supplied,
                ``turnover`` / ``buy_turnover`` / ``sell_turnover`` MUST be
                CNY notional amounts, the per-order minimum commission is
                applied, and the returned costs are a fraction of
                ``portfolio_value``. When omitted the legacy contract holds:
                values are the caller's trade-value units and the returned
                cost is in the same units; the CNY floor cannot be applied
                because the notional is unknown (documented large-account
                approximation for such callers).

        Returns:
            Total cost as a proportion of trade value (same type as turnover;
            a fraction of portfolio value when ``portfolio_value`` is given).
        """
        if buy_turnover is not None or sell_turnover is not None:
            # Explicit buy/sell split: stamp tax is a sell-side tax, so it is
            # charged on the sell notional only. Charging it on a total with
            # `is_sell=None` (the legacy path below) assumes half of every
            # trade is a sell, which overcharges from-cash builds and
            # undercharges full liquidations.
            buy = np.abs(0.0 if buy_turnover is None else buy_turnover)
            sell = np.abs(0.0 if sell_turnover is None else sell_turnover)
            traded = buy + sell
            rate_probe = buy if isinstance(buy, pd.Series) else sell
        else:
            abs_turnover = np.abs(turnover)
            traded = abs_turnover
            rate_probe = abs_turnover

        commission_rate, stamp_tax_rate = self._instrument_rates(
            rate_probe, symbol, date,
        )

        if buy_turnover is not None or sell_turnover is not None:
            commission_cost = traded * commission_rate
            stamp_cost = sell * stamp_tax_rate
        else:
            commission_cost = traded * commission_rate
            if is_sell is not None:
                stamp_cost = traded * stamp_tax_rate * is_sell.astype(float)
            else:
                stamp_cost = traded * stamp_tax_rate * 0.5

        commission_cost = self._apply_commission_floor(
            commission_cost, traded, portfolio_value is not None,
        )
        transfer_cost = traded * self.transfer_fee

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

        total = commission_cost + stamp_cost + transfer_cost + slippage_cost
        if portfolio_value is not None:
            # Inputs were CNY notionals of a CNY portfolio; return a fraction.
            return total / float(portfolio_value)
        return total

    def _apply_commission_floor(self, commission_cost, traded, notional_known):
        """Apply the CNY per-order minimum where the notional is known.

        The floor is meaningful only when ``traded`` is a CNY amount, which
        :meth:`compute_costs` signals with ``portfolio_value``. Outside that
        contract the amounts may be weights or arbitrary units and a CNY floor
        would be nonsense, so this is a no-op there.
        """
        if not notional_known or self.min_commission <= 0:
            return commission_cost
        if isinstance(commission_cost, pd.Series):
            floored = commission_cost.clip(lower=self.min_commission)
            if isinstance(traded, pd.Series):
                floored = floored.where(traded.reindex(floored.index) > 0, 0.0)
            return floored
        arr = np.asarray(commission_cost, dtype=float)
        traded_arr = np.asarray(traded, dtype=float)
        floored = np.maximum(arr, float(self.min_commission))
        if arr.ndim == 0:
            return float(floored) if float(traded_arr) > 0 else 0.0
        return np.where(traded_arr > 0, floored, 0.0)

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
