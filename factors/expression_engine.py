"""Expression-based Factor Engine.

Inspired by vnpy.alpha's factor expression system. This module allows
users to define factors as string formulas instead of writing Python classes.

Example:
    # Instead of creating a Python class:
    #   class MyFactor(MomentumFactor):
    #       def compute(self, prices): ...

    # You can write an expression:
    #   "ts_sum(close_pct, 21)"                          → 21-day cumulative return
    #   "ts_std(close_pct, 20)"                          → 20-day volatility
    #   "ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)"  → complex alpha

The system uses a DataProxy class with operator overloading so that
expressions can be evaluated using Python's built-in eval() in a
controlled namespace.

Architecture:
    DataProxy wraps a (date × asset) DataFrame/Series and delegates
    all arithmetic/comparison ops to Pandas + groupby logic.

    calculate_by_expression(df, expr) → pd.DataFrame
        Creates DataProxies for dataframe columns, adds registered
        functions to namespace, evaluates the expression.

    ExpressionFactor(BaseFactor) wraps expressions into the existing
    factor interface, so they work anywhere a regular factor works.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from quant_platform.factors.base import BaseFactor, FactorCategory
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

EXPRESSION_FUNCTIONS: dict[str, Callable] = {}


def register_expression_functions(functions: list[Callable]) -> None:
    """Register a list of functions by their __name__."""
    for func in functions:
        EXPRESSION_FUNCTIONS[func.__name__] = func


def get_expression_function(name: str) -> Callable | None:
    return EXPRESSION_FUNCTIONS.get(name)


# ---------------------------------------------------------------------------
# DataProxy: wraps a DataFrame with operator overloading
# ---------------------------------------------------------------------------

class DataProxy:
    """Vectorized data container with operator overloading.

    Wraps a (date × asset) DataFrame or Series and delegates all
    arithmetic and comparison operations to pandas, returning new
    DataProxies. This enables expression chaining like:
        ts_sum(close / ts_delay(close, 1) - 1, 21)

    Internally:
    - Arithmetic ops (+, -, *, /, **, etc.) → per-element ops on DataFrames
    - Comparison ops (>, <, ==, etc.) → bool DataFrames
    - All ops return DataProxy for further chaining
    """

    def __init__(self, df: pd.DataFrame | pd.Series):
        if isinstance(df, pd.Series):
            self.df: pd.DataFrame = df.to_frame()
        else:
            self.df = df

    def _as_proxy(self, other: Any) -> pd.DataFrame:
        """Normalize the other operand to a DataFrame."""
        if isinstance(other, DataProxy):
            return other.df
        if isinstance(other, pd.DataFrame):
            return other
        if isinstance(other, pd.Series):
            return other.to_frame()
        # Scalar
        return self.df.copy().fillna(other)  # placeholder, handled in ops

    def result(self, s: pd.Series | pd.DataFrame) -> DataProxy:
        """Wrap a computed result back into a DataProxy."""
        if isinstance(s, pd.Series):
            return DataProxy(s.to_frame(s.name or "result"))
        return DataProxy(s)

    # -- Arithmetic --
    def __add__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df + other.df)
        return self.result(self.df + other)

    def __radd__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(other.df + self.df)
        return self.result(other + self.df)

    def __sub__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df - other.df)
        return self.result(self.df - other)

    def __rsub__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(other.df - self.df)
        return self.result(other - self.df)

    def __mul__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df * other.df)
        return self.result(self.df * other)

    def __rmul__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(other.df * self.df)
        return self.result(other * self.df)

    def __truediv__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df / other.df.replace(0, np.nan))
        return self.result(self.df / other)

    def __rtruediv__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(other.df / self.df.replace(0, np.nan))
        return self.result(other / self.df.replace(0, np.nan))

    def __floordiv__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df // other.df)
        return self.result(self.df // other)

    def __mod__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df % other.df)
        return self.result(self.df % other)

    def __pow__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result(self.df ** other.df)
        return self.result(self.df ** other)

    def __abs__(self) -> DataProxy:
        return self.result(self.df.abs())

    def __neg__(self) -> DataProxy:
        return self.result(-self.df)

    # -- Comparison (return int for boolean compatibility) --
    def __gt__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result((self.df > other.df).astype(float))
        return self.result((self.df > other).astype(float))

    def __ge__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result((self.df >= other.df).astype(float))
        return self.result((self.df >= other).astype(float))

    def __lt__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result((self.df < other.df).astype(float))
        return self.result((self.df < other).astype(float))

    def __le__(self, other: Any) -> DataProxy:
        if isinstance(other, DataProxy):
            return self.result((self.df <= other.df).astype(float))
        return self.result((self.df <= other).astype(float))

    def __eq__(self, other: Any) -> DataProxy:  # type: ignore[override]
        if isinstance(other, DataProxy):
            return self.result((self.df == other.df).astype(float))
        return self.result((self.df == other).astype(float))

    def __ne__(self, other: Any) -> DataProxy:  # type: ignore[override]
        if isinstance(other, DataProxy):
            return self.result((self.df != other.df).astype(float))
        return self.result((self.df != other).astype(float))

    # -- Other --
    def __invert__(self) -> DataProxy:
        return self.result((self.df == 0).astype(float))

    def __getitem__(self, key: str) -> DataProxy:
        if isinstance(self.df, pd.DataFrame) and key in self.df.columns:
            return DataProxy(self.df[[key]])
        raise KeyError(f"Column '{key}' not found")


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

def calculate_by_expression(
    df: pd.DataFrame | dict[str, pd.DataFrame],
    expression: str,
    extra_functions: dict[str, Callable] | None = None,
) -> pd.DataFrame:
    """Evaluate a string expression using the DataProxy system.

    Args:
        df: DataFrame with columns as named variables for the expression.
            If a dict is provided, each key becomes a named variable.
        expression: String expression, e.g. "ts_sum(close_pct, 21)".
        extra_functions: Additional functions to register temporarily.

    Returns:
        DataFrame (date × asset) with the computed result.

    Raises:
        ValueError: If expression evaluation fails.
    """
    # Build namespace
    namespace: dict[str, Any] = {}

    if isinstance(df, dict):
        for name, data in df.items():
            if isinstance(data, DataProxy):
                namespace[name] = data
            else:
                namespace[name] = DataProxy(data)
    else:
        for col in df.columns:
            namespace[col] = DataProxy(df[col])

    # Add registered functions
    namespace.update(EXPRESSION_FUNCTIONS)

    # Add extra functions
    if extra_functions:
        namespace.update(extra_functions)

    # Add built-in imports needed by some functions
    namespace["np"] = np
    namespace["pd"] = pd

    try:
        result = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as e:
        raise ValueError(
            f"Expression evaluation failed: {e}\n"
            f"Expression: {expression}\n"
            f"Available variables: {[k for k in namespace if not k.startswith('_')]}"
        ) from e

    if isinstance(result, DataProxy):
        return result.df
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame("result")
    raise ValueError(f"Expression returned unexpected type: {type(result)}")


# ---------------------------------------------------------------------------
# ExpressionFactor: wraps a string expression into BaseFactor interface
# ---------------------------------------------------------------------------


class ExpressionFactor(BaseFactor):
    """A factor defined by a string expression.

    Allows users to define new factors without writing Python code,
    simply by providing an expression string.

    Example:
        factor = ExpressionFactor(
            name="my_alpha",
            expression="ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)",
            params={"period": 10},
        )
        result = factor.compute(prices)
    """

    category = FactorCategory.CUSTOM

    def __init__(
        self,
        name: str,
        expression: str,
        params: dict[str, Any] | None = None,
    ):
        super().__init__(params)
        self._name = name
        self._expression = expression

    @property
    def name(self) -> str:
        return self._name

    @property
    def expression(self) -> str:
        return self._expression

    def compute(
        self,
        prices: pd.DataFrame,
        financials: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute factor value by evaluating the expression.

        The following variables are available in expressions:
        - close:        Close prices (date × asset)
        - close_pct:    Daily returns (pct_change)
        - volume:       Volume/turnover (if provided in kwargs)
        - open, high, low:  OHLC prices
        - Any columns in financials DataFrame
        """
        # Build input DataFrames
        data: dict[str, pd.DataFrame] = {
            "close": prices,
            "close_pct": prices.pct_change(fill_method=None),
        }

        # Add OHLC if available
        if "open" in kwargs:
            data["open"] = kwargs["open"]
        if "high" in kwargs:
            data["high"] = kwargs["high"]
        if "low" in kwargs:
            data["low"] = kwargs["low"]

        # Add volume / turnover
        if "turnover" in kwargs and kwargs["turnover"] is not None:
            data["volume"] = kwargs["turnover"]
        elif "volume" in kwargs and kwargs["volume"] is not None:
            data["volume"] = kwargs["volume"]

        # Add financials columns
        if financials is not None:
            if isinstance(financials, pd.DataFrame):
                # financials is (date x asset) with a 'name' attribute
                fin_col_name = getattr(financials, 'name', None)
                if fin_col_name:
                    data[fin_col_name] = financials
                else:
                    # Multiple columns: add each
                    for col in financials.columns:
                        data[col] = financials[col]

        # Add any extra kwargs from params
        for key, val in self.params.items():
            if isinstance(val, pd.DataFrame):
                data[key] = val

        result = calculate_by_expression(data, self._expression)
        return result


# ---------------------------------------------------------------------------
# Convenience: create factors from config dict
# ---------------------------------------------------------------------------


def create_expression_factors(
    expressions: dict[str, str],
) -> list[ExpressionFactor]:
    """Create ExpressionFactor instances from a config dict.

    Args:
        expressions: {factor_name: expression_string}

    Returns:
        List of ExpressionFactor instances.
    """
    factors = []
    for name, expr in expressions.items():
        factors.append(ExpressionFactor(name=name, expression=expr))
    return factors
