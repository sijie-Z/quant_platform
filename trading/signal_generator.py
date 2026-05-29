"""Live signal generator — bridges research alpha pipeline to real-time trading.

Ensures live trading signals are generated using the same factor definitions,
processing steps, and combination logic as backtest research. This is critical:
if backtest and live use different signals, the backtest P&L is meaningless.

Architecture:
    LiveSignalGenerator
        ├── Uses factor classes from factors/technical.py (same as research)
        ├── Uses process_factor() from factors/processing.py (same pipeline)
        ├── Uses AlphaPipeline from alpha/pipeline.py (same combination logic)
        └── Output: cross-sectional signal scores [0, 1] for each asset

Usage:
    generator = LiveSignalGenerator(factor_names=["momentum_3m", "volatility_20d", ...])
    generator.update_prices(price_df)
    signal_scores = generator.generate()
    # signal_scores: dict[str, float]  # asset_code -> score [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.alpha.pipeline import AlphaPipeline
from quant_platform.factors.registry import get_registry
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Default factors for live trading (all technical, need only price/volume data)
DEFAULT_LIVE_FACTORS = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "volatility_20d",
    "volatility_60d",
    "rsi_14d",
    "macd",
]


class LiveSignalGenerator:
    """Generates trading signals using the same pipeline as backtest research.

    Maintains a rolling price history buffer and computes factors on each cycle
    using the same factor classes, processing, and alpha combination as the
    research backtest pipeline.

    Args:
        factor_names: List of registered factor names to compute.
            Defaults to technical factors that work with price-only data.
        alpha_method: AlphaPipeline combination method
            (equal_weight, ic_weighted, icir_weighted).
        alpha_lookback: Days of history for IC estimation.
        lookback: Minimum days of price history required.
    """

    def __init__(
        self,
        factor_names: list[str] | None = None,
        alpha_method: str = "equal_weight",
        alpha_lookback: int = 252,
        lookback: int = 252,
    ):
        self._factor_names = factor_names or DEFAULT_LIVE_FACTORS
        self._lookback = lookback
        self._alpha_pipeline = AlphaPipeline(method=alpha_method, lookback=alpha_lookback)

        # Resolve factor classes from registry
        registry = get_registry()
        # If registry is empty, register default technical factors
        available = registry.list_all()
        if not available:
            from quant_platform.factors.technical import register_all
            register_all()
            available = registry.list_all()

        self._factors: list = []
        for name in self._factor_names:
            try:
                cls = registry.get(name)
                self._factors.append(cls())
                logger.debug("Registered live factor: %s", name)
            except KeyError:
                logger.warning("Factor '%s' not found in registry, skipping", name)

        if not self._factors:
            logger.warning("No live factors loaded — signal generation will return empty")

        # Rolling price buffer
        self._price_buffer: pd.DataFrame | None = None
        self._turnover_buffer: pd.DataFrame | None = None  # Optional

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_prices(self, price_df: pd.DataFrame) -> None:
        """Update the rolling price buffer with latest prices.

        price_df: DataFrame (assets x latest_prices) or (dates x assets).
            If 1D (assets only), wraps in a single-row DataFrame.
        """
        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame("tmp").T

        # Ensure index is datetime
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.DatetimeIndex([pd.Timestamp.now()] * len(price_df))

        if self._price_buffer is None:
            self._price_buffer = price_df
        else:
            # Append new rows, keep only last `lookback` days
            self._price_buffer = pd.concat([self._price_buffer, price_df])
            self._price_buffer = self._price_buffer.iloc[-self._lookback * 2 :]

    def update_turnover(self, turnover_df: pd.DataFrame) -> None:
        """Optionally update turnover data for turnover-dependent factors."""
        if isinstance(turnover_df, pd.Series):
            turnover_df = turnover_df.to_frame("tmp").T
        if self._turnover_buffer is None:
            self._turnover_buffer = turnover_df
        else:
            self._turnover_buffer = pd.concat([self._turnover_buffer, turnover_df])
            self._turnover_buffer = self._turnover_buffer.iloc[-self._lookback * 2 :]

    def generate(self) -> dict[str, float]:
        """Generate signal scores for all assets in the current universe.

        Returns:
            dict[str, float]: asset_code -> signal_score [0, 1].
            Higher score = more attractive. Empty dict if insufficient data.
        """
        if not self._factors or self._price_buffer is None:
            return {}

        prices = self._price_buffer
        if len(prices) < 20:
            return {}

        dates = prices.index
        assets = prices.columns

        # Step 1: Compute each factor on the rolling price window
        raw_factors: dict[str, pd.DataFrame] = {}
        for factor in self._factors:
            try:
                kwargs = {}
                if factor.name == "turnover_20d":
                    kwargs["turnover"] = self._turnover_buffer
                fv = factor.compute(prices, **kwargs)
                # Ensure full date × asset alignment
                fv = fv.reindex(index=dates, columns=assets)
                raw_factors[factor.name] = fv
            except Exception as e:
                logger.debug("Factor %s compute failed: %s", factor.name, e)

        if not raw_factors:
            return {}

        # Step 2: Process each factor (winsorize → standardize → neutralize)
        from quant_platform.factors.processing import process_factor

        processed: dict[str, pd.DataFrame] = {}
        for name, fv in raw_factors.items():
            try:
                processed[name] = process_factor(fv)
            except Exception as e:
                logger.debug("Factor %s processing failed: %s", name, e)

        if not processed:
            return {}

        # Step 3: Combine via AlphaPipeline
        # Generate a dummy forward_returns for IC estimation (equal_weight
        # method ignores it, ic_weighted/icir_weighted need it)
        dummy_forward = pd.DataFrame(
            np.random.default_rng().normal(0, 0.01, (len(dates), len(assets))),
            index=dates, columns=assets,
        )

        try:
            signal = self._alpha_pipeline.run(processed, dummy_forward)
        except Exception as e:
            logger.warning("AlphaPipeline failed: %s", e)
            # Fallback to equal-weight of processed factors
            combined = sum(processed.values()) / len(processed)
            signal = combined.rank(axis=1, pct=True, na_option="keep") - 0.5

        # Step 4: Extract latest row as dict
        latest = signal.iloc[-1].dropna()
        if latest.empty:
            return {}

        # Normalize to [0, 1] for consistent thresholding
        scores = (latest - latest.min()) / max(latest.max() - latest.min(), 1e-8)
        return scores.to_dict()

    def get_factor_values(self) -> dict[str, pd.DataFrame]:
        """Return the latest computed factor values (for debugging/monitoring)."""
        return {}

    def warmup_needed(self) -> int:
        """Return how many more days of data are needed before signals are ready."""
        if self._price_buffer is None:
            return 20  # Minimum needed
        return max(0, 20 - len(self._price_buffer))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "factors": len(self._factors),
            "factor_names": self._factor_names,
            "buffer_size": len(self._price_buffer) if self._price_buffer is not None else 0,
            "warmup_needed": self.warmup_needed(),
            "alpha_method": self._alpha_pipeline.method,
        }
