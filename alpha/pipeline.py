"""Alpha pipeline: from processed factors to trading signals.

The signal pipeline:
1. Combine multiple factors into a raw alpha score
2. Cross-sectional rank normalization
3. Output: final signal (date x asset), higher = more attractive
"""

from __future__ import annotations

import pandas as pd

from quant_platform.factors.evaluation import rank_ic
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class AlphaPipeline:
    """Transforms factors into a final cross-sectional alpha signal."""

    def __init__(
        self,
        method: str = "icir_weighted",
        lookback: int = 252,
        min_icir: float = 0.0,
    ):
        self.method = method
        self.lookback = lookback
        self.min_icir = min_icir

    def run(
        self,
        factors: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine factors and produce final signal.

        Args:
            factors: Dict of (date x asset) processed factor values.
            forward_returns: (date x asset) forward returns for IC estimation.

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

        logger.info("Alpha signal generated: method=%s, shape=%s", self.method, signal.shape)
        return signal
