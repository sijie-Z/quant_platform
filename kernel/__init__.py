"""TradingKernel — alpha-v1.0 系统控制平面.

STATUS: PROPOSAL (2026-10) — `kernel/kernel.py` has zero runtime importers;
`daemon/runner.py` still runs LiveEngine + SafetySystem directly. See
docs/DEAD_CODE_STATUS.md / #77.

把零件装成机器。从 "collection of components" 变成 "autonomous trading organism"。
"""

from .kernel import TradingKernel

__all__ = ["TradingKernel"]
