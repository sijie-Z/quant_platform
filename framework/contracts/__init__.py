"""Capability-layer Protocols (stable interfaces).

STATUS: PROPOSAL (2026-10) — six Protocols, zero conforming production
implementations. Nothing that runs research imports this package; it is an
interface proposal, not an enforced contract. See docs/DEAD_CODE_STATUS.md
and issue #70.

Per docs/ARCHITECTURE_V2.md ADR-0004 (Protocol Before Plugin), interfaces are
locked before implementations. There is no `ref/` directory today: the
"reference implementation" claim this docstring used to make was false.

Protocols defined:
    MarketDataProvider   — daily OHLCV + fundamentals, with explicit adjust + bias flags
    UniverseProvider     — stock universe at a point in time; honesty about PIT vs approximate
    Broker               — submit / cancel / positions / balance / orders
    LLM                  — complete / embed
    Factor               — compute(panel) -> cross-sectional factor values
    Evaluator            — IC / ICIR / DSR / BH-FDR evaluation API
"""

from quant_platform.framework.contracts.broker import Broker
from quant_platform.framework.contracts.evaluator import Evaluator
from quant_platform.framework.contracts.factor import Factor
from quant_platform.framework.contracts.llm import LLM
from quant_platform.framework.contracts.market_data import MarketDataProvider
from quant_platform.framework.contracts.universe import UniverseProvider

__all__ = [
    "MarketDataProvider",
    "UniverseProvider",
    "Broker",
    "LLM",
    "Factor",
    "Evaluator",
]
