"""Capability-layer Protocols (stable interfaces).

Per ADR-0004 (Protocol Before Plugin), interfaces are locked before
implementations. Each Protocol here has at least one Reference Implementation
(`ref/`); new providers MUST implement the existing Protocol rather than
spawning a parallel interface.

Protocols defined:
    MarketDataProvider   — daily OHLCV + fundamentals, with explicit adjust + bias flags
    UniverseProvider     — stock universe at a point in time; honesty about PIT vs approximate
    Broker               — submit / cancel / positions / balance / orders
    LLM                  — complete / embed
    Factor               — compute(panel) -> cross-sectional factor values
    Evaluator            — IC / ICIR / DSR / BH-FDR evaluation API
"""

from quant_platform.framework.contracts.market_data import MarketDataProvider
from quant_platform.framework.contracts.universe import UniverseProvider
from quant_platform.framework.contracts.broker import Broker
from quant_platform.framework.contracts.llm import LLM
from quant_platform.framework.contracts.factor import Factor
from quant_platform.framework.contracts.evaluator import Evaluator

__all__ = [
    "MarketDataProvider",
    "UniverseProvider",
    "Broker",
    "LLM",
    "Factor",
    "Evaluator",
]
