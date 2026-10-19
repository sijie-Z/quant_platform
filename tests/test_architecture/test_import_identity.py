"""Issue #69: every entry point must import ONE module identity.

The repo root *is* the `quant_platform` package. When the project was
installed with `packages.find include=["*"]`, its subdirectories were also
installed as top-level packages, so

    import data.providers.synthetic
    import quant_platform.data.providers.synthetic

loaded the same file as two different modules, with two class objects and two
copies of every module-level singleton (the factor registry showed 0 vs 22
entries). These tests make that failure mode visible.
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Names that are importable as top-level first-party modules.
FIRST_PARTY = {
    path.name
    for path in REPO.iterdir()
    if path.is_dir() and (path / "__init__.py").exists()
}
FIRST_PARTY |= {path.stem for path in REPO.glob("*.py")}
ALLOWED = {"quant_platform", "tests", "__init__"}


def _first_party_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_no_bare_first_party_imports():
    violations = []
    for path in REPO.rglob("*.py"):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name in _first_party_imports(source) & FIRST_PARTY:
            if name in ALLOWED:
                continue
            violations.append(f"{path.relative_to(REPO)}: import {name}")

    assert not violations, (
        "flat first-party imports recreate the dual module identity:\n"
        + "\n".join(sorted(violations))
    )


def test_canonical_imports_do_not_load_flat_aliases():
    import quant_platform.app  # noqa: F401
    import quant_platform.core.events  # noqa: F401
    import quant_platform.data.providers.synthetic as synthetic
    import quant_platform.factors.registry as registry_mod
    import quant_platform.factors.technical as technical
    import quant_platform.main  # noqa: F401

    flat = FIRST_PARTY - ALLOWED
    loaded = flat & set(sys.modules)
    assert not loaded, f"canonical imports loaded flat aliases: {sorted(loaded)}"

    from quant_platform.data.providers.synthetic import SyntheticDataProvider
    from quant_platform.factors.registry import get_registry

    assert synthetic.SyntheticDataProvider is SyntheticDataProvider
    assert registry_mod.get_registry is get_registry

    technical.register_all()
    assert len(get_registry().list_all()) >= 22


def test_event_bus_is_one_singleton():
    from quant_platform.core.events import get_event_bus

    assert get_event_bus() is get_event_bus()
    assert "core.events" not in sys.modules
