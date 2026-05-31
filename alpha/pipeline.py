"""Alpha pipeline: from processed factors to trading signals.

The signal pipeline:
1. Combine multiple factors into a raw alpha score
2. Cross-sectional rank normalization
3. Output: final signal (date x asset), higher = more attractive
"""

from __future__ import annotations

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class AlphaPipeline:
    """Transforms factors into a final cross-sectional alpha signal.

    Args:
        method: Factor combination method.
        lookback: Days for IC/ICIR estimation.
        min_icir: Minimum ICIR to include factor.
        tradability_gate: If True, applies per-stock profile-based
            tradability filter to suppress noisy stocks.
        min_tradability: Threshold [0,1] for tradability gate.
            Higher = only the smoothest, most trend-like stocks pass.
    """

    def __init__(
        self,
        method: str = "icir_weighted",
        lookback: int = 252,
        min_icir: float = 0.0,
        tradability_gate: bool = False,
        min_tradability: float = 0.3,
    ):
        self.method = method
        self.lookback = lookback
        self.min_icir = min_icir
        self.tradability_gate = tradability_gate
        self.min_tradability = min_tradability

    def run(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        prices: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Combine factors and produce final signal.

        Args:
            factors: Dict of (date x asset) processed factor values.
            forward_returns: (date x asset) forward returns for IC estimation.
            prices: Optional (date x asset) prices, needed for tradability gate.
            volume: Optional (date x asset) volume, needed for tradability gate.

        Returns:
            (date x asset) final alpha signal, cross-sectionally ranked.
        """
        from quant_platform.alpha.combination import (
            combine_equal_weight,
            combine_ic_weighted,
            combine_icir_weighted,
        )

        if self.method == "equal_weight":
            raw = combine_equal_weight(factors)
        elif self.method == "ic_weighted":
            raw = combine_ic_weighted(factors, forward_returns, self.lookback)
        elif self.method == "icir_weighted":
            raw = combine_icir_weighted(
                factors, forward_returns, self.lookback, self.min_icir
            )
        else:
            raise ValueError(f"Unknown combination method: {self.method}")

        # Cross-sectional rank normalization
        signal = raw.rank(axis=1, pct=True, na_option="keep")
        # Center to [-0.5, 0.5] range
        signal = signal - 0.5

        # Optional: tradability gate
        if self.tradability_gate:
            if prices is None:
                logger.warning(
                    "Tradability gate enabled but prices not provided — skipping"
                )
            else:
                from quant_platform.risk.profile_classifier import (
                    apply_tradability_gate,
                )
                signal = apply_tradability_gate(
                    signal, prices, volume=volume,
                    min_tradability=self.min_tradability,
                )
                logger.info(
                    "Tradability gate applied: min_tradability=%.2f",
                    self.min_tradability,
                )

        logger.info("Alpha signal generated: method=%s, shape=%s", self.method, signal.shape)
        return signal
