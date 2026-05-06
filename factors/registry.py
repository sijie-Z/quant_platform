"""Factor registry for automatic factor discovery and management."""

from __future__ import annotations

from quant_platform.factors.base import BaseFactor, FactorCategory


class FactorRegistry:
    """Registry of all available factors.

    Factors can be registered explicitly or discovered via the
    factor classes' __init_subclass__ hook.
    """

    _instance: FactorRegistry | None = None
    _factors: dict[str, type[BaseFactor]]

    def __new__(cls) -> FactorRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}
        return cls._instance

    def register(self, factor_cls: type[BaseFactor]) -> None:
        """Register a factor class."""
        # Create a temp instance to get the name
        inst = factor_cls()
        self._factors[inst.name] = factor_cls

    def get(self, name: str) -> type[BaseFactor]:
        """Get a factor class by name."""
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found. Available: {list(self._factors.keys())}")
        return self._factors[name]

    def list_by_category(self, category: FactorCategory | None = None) -> list[str]:
        """List factor names, optionally filtered by category."""
        result = []
        for name, cls in self._factors.items():
            inst = cls()
            if category is None or inst.category == category:
                result.append(name)
        return result

    def list_all(self) -> list[str]:
        """List all registered factor names."""
        return list(self._factors.keys())

    def clear(self) -> None:
        """Clear all registered factors (useful for testing)."""
        self._factors.clear()


def get_registry() -> FactorRegistry:
    """Get the global factor registry singleton."""
    return FactorRegistry()
