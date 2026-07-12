"""Architecture import boundary tests.
=====================================
Enforces the dependency direction declared in CLAUDE.md / governance:

    framework/  ← no dependency on lab/ or prod/
    lab/        ← may depend on framework/, NOT on prod/
    prod/       ← may depend on framework/ and lab/, but not the reverse

A violation fails CI immediately — no human review needed.
This is executable architecture, not documentation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# These are the protected boundary packages (relative to repo root)
PROTECTED = {
    "framework": {"may_not_import": {"quant_platform.lab", "quant_platform.prod"}},
    "lab": {"may_not_import": {"quant_platform.prod"}},
}
# Only keep entries for directories that actually exist
PROTECTED = {k: v for k, v in PROTECTED.items() if (REPO_ROOT / k).is_dir()}


def _collect_imports(file_path: Path) -> set[str]:
    """Extract all import targets from a Python file (stdlib-aware)."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.add(node.module)
    return imports


def _py_files_under(directory: Path) -> list[Path]:
    """All .py files under directory, excluding __pycache__ and .venv."""
    files: list[Path] = []
    for p in directory.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts or "node_modules" in p.parts:
            continue
        files.append(p)
    return files


# ── test cases ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layer",
    [d for d in PROTECTED if (REPO_ROOT / d).is_dir()],
)
def test_protected_layer_does_not_import_upstream(layer: str):
    """Check that `layer/` has no forbidden imports into layers above it."""
    rule = PROTECTED[layer]
    forbidden = rule["may_not_import"]
    layer_dir = REPO_ROOT / layer

    if not layer_dir.is_dir():
        pytest.skip(f"Layer directory '{layer}/' does not exist")

    violations: list[str] = []
    for py_file in _py_files_under(layer_dir):
        imports = _collect_imports(py_file)
        for imp in imports:
            for banned in forbidden:
                if imp == banned or imp.startswith(banned + "."):
                    violations.append(
                        f"  {py_file.relative_to(REPO_ROOT)} imports '{imp}'"
                        f" (violates: {layer}/ may not import {banned})"
                    )

    if violations:
        header = (
            f"\n{'='*72}\n"
            f"  ARCHITECTURE VIOLATION: {layer}/ boundary broken\n"
            f"{'='*72}\n"
            f"  {layer}/ may NOT import from: {', '.join(forbidden)}\n"
            f"  Fix: refactor into {layer}/ or move shared code to a lower layer.\n"
            f"{'='*72}\n"
        )
        pytest.fail(header + "\n".join(violations))


@pytest.mark.parametrize(
    "target",
    [
        "quant_platform.lab",
        "quant_platform.prod",
    ],
)
def test_framework_never_imports_lab_or_prod(target: str):
    """Framework must remain independent of lab/ and prod/.

    This test is intentionally duplicated from the parametrized version
    to produce a clear, grep-friendly failure message in CI.
    """
    framework_dir = REPO_ROOT / "framework"
    if not framework_dir.is_dir():
        pytest.skip("framework/ directory does not exist")

    violations = []
    for py_file in _py_files_under(framework_dir):
        imports = _collect_imports(py_file)
        for imp in imports:
            if imp == target or imp.startswith(target + "."):
                violations.append(
                    f"  {py_file.relative_to(REPO_ROOT)} imports '{imp}'"
                )

    assert not violations, (
        f"\n{'='*72}\n"
        f"  ARCHITECTURE VIOLATION: framework/ imports {target}\n"
        f"{'='*72}\n"
        f"  framework/ is the capability layer — it must NEVER depend on\n"
        f"  research (lab/) or production (prod/) code.\n"
        f"  Fix: extract shared logic into framework/ itself, or move the\n"
        f"  import target to a lower layer.\n"
        f"{'='*72}\n"
        + "\n".join(violations)
    )


# ── discovery ───────────────────────────────────────────────────────


def _discover_layer(path: str) -> str | None:
    """Map an import path to the layer it belongs to."""
    for layer in ["framework", "lab", "prod"]:
        if path == f"quant_platform.{layer}" or path.startswith(f"quant_platform.{layer}."):
            return layer
    return None


def test_all_layers_report_current_boundary_state():
    """Informational test: prints current import topology.

    Always passes. Use -v to see the output.
    """
    layers_found: dict[str, list[str]] = {}
    for layer_dir_name in PROTECTED:
        d = REPO_ROOT / layer_dir_name
        if not d.is_dir():
            continue
        for py_file in _py_files_under(d):
            imports = _collect_imports(py_file)
            for imp in imports:
                target_layer = _discover_layer(imp)
                if target_layer and target_layer != layer_dir_name:
                    layers_found.setdefault(layer_dir_name, [])
                    layers_found[layer_dir_name].append(
                        f"{py_file.relative_to(REPO_ROOT)} → {imp}"
                    )

    print("\n  Current import topology (cross-layer only):")
    if not layers_found:
        print("    (none — all layers are internally coherent)")
    else:
        for src, edges in sorted(layers_found.items()):
            print(f"  {src}/:")
            for e in sorted(set(edges)):
                print(f"    {e}")
