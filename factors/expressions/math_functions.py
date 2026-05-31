"""Math and utility expression functions for the expression factor engine.

These functions provide basic mathematical operations that can be
combined with ts_* and cs_* functions to build complex expressions.
"""

from __future__ import annotations

import numpy as np

from quant_platform.factors.expression_engine import DataProxy


def log(feature: DataProxy) -> DataProxy:
    """Natural logarithm of feature values."""
    return DataProxy(np.log(feature.df.clip(lower=1e-8)))


def abs_val(feature: DataProxy) -> DataProxy:
    """Absolute value."""
    return DataProxy(feature.df.abs())


def sign(feature: DataProxy) -> DataProxy:
    """Sign function: -1, 0, or 1."""
    return DataProxy(np.sign(feature.df))


def pow_val(feature: DataProxy, exponent: float) -> DataProxy:
    """Raise feature values to the given power."""
    return DataProxy(feature.df ** exponent)


def sqrt(feature: DataProxy) -> DataProxy:
    """Square root of feature values."""
    return DataProxy(np.sqrt(feature.df.clip(lower=0)))


def less(a: DataProxy, b: DataProxy | float) -> DataProxy:
    """Element-wise less-than comparison (returns 0/1)."""
    if isinstance(b, DataProxy):
        return DataProxy((a.df < b.df).astype(float))
    return DataProxy((a.df < b).astype(float))


def greater(a: DataProxy, b: DataProxy | float) -> DataProxy:
    """Element-wise greater-than comparison (returns 0/1)."""
    if isinstance(b, DataProxy):
        return DataProxy((a.df > b.df).astype(float))
    return DataProxy((a.df > b).astype(float))


def if_else(condition: DataProxy, true_val: DataProxy | float, false_val: DataProxy | float) -> DataProxy:
    """Element-wise if-else.

    Args:
        condition: Values treated as boolean (0 = False, non-zero = True).
        true_val: Value if condition is True.
        false_val: Value if condition is False.
    """
    c = condition.df.astype(bool)
    t = true_val.df if isinstance(true_val, DataProxy) else true_val
    f = false_val.df if isinstance(false_val, DataProxy) else false_val
    import pandas as _pd
    return DataProxy(_pd.DataFrame(np.where(c, t, f).astype(float), index=condition.df.index, columns=condition.df.columns))


def scale(feature: DataProxy, a: float = 1.0) -> DataProxy:
    """Multiply feature values by scalar a."""
    return DataProxy(feature.df * a)


def neg(feature: DataProxy) -> DataProxy:
    """Negate feature values."""
    return DataProxy(-feature.df)


# List of all math functions for registration
MATH_FUNCTIONS = [
    log, abs_val, sign, pow_val, sqrt,
    less, greater, if_else,
    scale, neg,
]
