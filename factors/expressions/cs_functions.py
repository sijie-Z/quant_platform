"""Cross-section expression functions for the expression factor engine.

Cross-section operations work across all assets on a single date.
They are computed per row (date) of the date×asset matrix.
"""

from __future__ import annotations

import numpy as np

from quant_platform.factors.expression_engine import DataProxy


def cs_rank(feature: DataProxy) -> DataProxy:
    """Cross-sectional rank, normalized to [0, 1]."""
    result = feature.df.rank(axis=1, pct=True, na_option="keep")
    return DataProxy(result)


def cs_mean(feature: DataProxy) -> DataProxy:
    """Cross-sectional mean (per date)."""
    result = feature.df.mean(axis=1)
    # Broadcast back to full shape
    result = result.to_frame().T if result.ndim == 0 else result
    return DataProxy(feature.df.subtract(result, axis=0).add(result, axis=0))


def cs_std(feature: DataProxy) -> DataProxy:
    """Cross-sectional standard deviation (per date)."""
    result = feature.df.std(axis=1, ddof=0)
    return DataProxy(result)


def cs_sum(feature: DataProxy) -> DataProxy:
    """Cross-sectional sum (per date)."""
    result = feature.df.sum(axis=1)
    return DataProxy(result)


def cs_scale(feature: DataProxy) -> DataProxy:
    """Scale by sum of absolute values in cross-section."""
    abs_sum = feature.df.abs().sum(axis=1)
    result = feature.df.div(abs_sum.replace(0, np.nan), axis=0)
    return DataProxy(result)


def cs_zscore(feature: DataProxy) -> DataProxy:
    """Cross-sectional z-score: (x - mean) / std (per date)."""
    mean = feature.df.mean(axis=1)
    std = feature.df.std(axis=1, ddof=0).replace(0, np.nan)
    z = feature.df.sub(mean, axis=0).div(std, axis=0)
    return DataProxy(z)


# List of all cross-section functions for registration
CS_FUNCTIONS = [
    cs_rank,
    cs_mean,
    cs_std,
    cs_sum,
    cs_scale,
    cs_zscore,
]
