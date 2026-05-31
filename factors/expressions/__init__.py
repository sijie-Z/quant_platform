"""Expression function library for the expression factor engine.

Auto-registers all built-in functions when imported.
"""

from quant_platform.factors.expression_engine import register_expression_functions
from quant_platform.factors.expressions.ts_functions import TS_FUNCTIONS
from quant_platform.factors.expressions.cs_functions import CS_FUNCTIONS
from quant_platform.factors.expressions.math_functions import MATH_FUNCTIONS

# Register all built-in functions on import
ALL_FUNCTIONS = TS_FUNCTIONS + CS_FUNCTIONS + MATH_FUNCTIONS
register_expression_functions(ALL_FUNCTIONS)
