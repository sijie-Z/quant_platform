"""Quant Research OS — Capability Layer package.

This package hosts the capability-layer Protocols under `framework.contracts`.
Concrete factor content and strategies live in `lab/`; there is no `prod/`
package in this repo (as of 2026-10).

The dependency direction is one-way: `framework` MUST NOT import `lab`.
This is enforced by tests/test_architecture/test_core_boundaries.py and
tests/test_architecture/test_boundaries.py, which run in CI via `pytest tests/`
(there is no import-linter). See docs/ARCHITECTURE_V2.md for design intent --
that document is status-annotated as a proposal, not a guarantee (see #73).
"""

__all__ = ["contracts"]
