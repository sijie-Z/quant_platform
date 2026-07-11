"""Quant Research OS — Capability Layer package.

This package hosts the open, capability-layer contracts (Protocols) and their
Reference Implementations. Per CONSTITUTION.md Principle 4 (Protocol over
Implementation) and Principle 5 (Capability Layer Never Contains Alpha), this
package defines stable interfaces; concrete factor content / strategies stay
in the Knowledge (lab) and Production (prod) layers.

The dependency direction is one-way: `framework` MUST NOT import `lab` or
`prod` (enforced by import-linter). See ARCHITECTURE.md.
"""

__all__ = ["contracts", "registry", "trust"]
