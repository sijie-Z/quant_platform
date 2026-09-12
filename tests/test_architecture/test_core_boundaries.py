"""Core/live/lab import boundary tests.

These are the boundaries that must hold before quant_platform can be split
into quant-core / quant-live / quant-lab:

    core (research) -> live/lab: forbidden
    live             -> lab:     forbidden
    lab              -> live:    forbidden
    live, lab        -> core:    allowed

Shared modules that both layers need live in the top-level `shared/`
package. This test fails if a live-only module starts leaking back into the
research core.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

CORE_DIRS = {
    "factors", "alpha", "portfolio", "backtest", "data", "framework",
    "research", "utils", "shared",
}
LIVE_DIRS = {
    "core", "trading", "execution", "api", "risk", "strategy",
    "operations", "compliance", "monitoring", "daemon", "kernel",
    "services", "regime_router",
}
LAB_DIRS = {"lab", "tools"}
PLATFORM_FILES = {"main.py", "app.py", "run_factor.py"}

# All previously shared modules were moved to `shared/`; the whitelist is
# intentionally empty so any new live dependency from core fails the test.
SHARED_ALLOWED: set[str] = set()


def _layer(path: Path) -> str | None:
    parts = path.relative_to(REPO_ROOT).parts
    if len(parts) < 2:
        if path.name in PLATFORM_FILES:
            return "platform"
        return None
    top = parts[0]
    if top in CORE_DIRS:
        return "core"
    if top in LIVE_DIRS:
        return "live"
    if top in LAB_DIRS:
        return "lab"
    return None


def _imports(file_path: Path) -> set[str]:
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def _violations() -> list[str]:
    violations: list[str] = []
    for py_file in REPO_ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__", "node_modules", "dist"} for part in py_file.parts):
            continue
        src = _layer(py_file)
        if src not in ("core", "live", "lab"):
            continue
        for imp in _imports(py_file):
            target = None
            if imp == "quant_core" or imp.startswith("quant_core."):
                target = "core"
                continue
            for layer_name, dirs in (
                ("core", CORE_DIRS),
                ("live", LIVE_DIRS),
                ("lab", LAB_DIRS),
            ):
                for d in dirs:
                    prefix = f"quant_platform.{d}"
                    if imp == prefix or imp.startswith(prefix + "."):
                        target = layer_name
                        break
                if target:
                    break
            if target is None:
                continue
            if imp in SHARED_ALLOWED:
                continue
            if src == target:
                continue
            if src == "core" and target in ("live", "lab") or src == "live" and target == "lab" or src == "lab" and target == "live":
                violations.append(f"{py_file.relative_to(REPO_ROOT)} imports {imp}")
    return sorted(set(violations))


def test_core_never_imports_live_or_lab():
    bad = _violations()
    assert not bad, (
        "Core/live/lab boundary broken. Shared modules that core already "
        "depends on must move into core before the split:\n" + "\n".join(bad)
    )


def test_shared_modules_listed_for_migration():
    """Fail loudly if the shared-module whitelist drifts without review."""
    expected: set[str] = set()
    assert expected == SHARED_ALLOWED
