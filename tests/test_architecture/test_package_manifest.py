"""The package manifest must match the tree.

The wheel used to contain only the root package's top-level modules: every
subpackage was missing, so `pip install .` produced a CLI that failed on
import. Enumerating packages is exact but easy to forget; this test makes the
omission fail CI instead of a user's install.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"tests", "build", "packages", "frontend", "node_modules"}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _expected_packages() -> set[str]:
    packages = set()
    for init in ROOT.rglob("__init__.py"):
        rel = init.parent.relative_to(ROOT)
        parts = rel.parts
        if any(
            part in SKIP_DIRS or part.startswith(".")
            for part in parts
        ):
            continue
        name = "quant_platform" if not parts else (
            "quant_platform." + ".".join(parts)
        )
        packages.add(name)
    return packages


def test_every_subpackage_is_declared():
    declared = set(_pyproject()["tool"]["setuptools"]["packages"])

    missing = _expected_packages() - declared
    assert not missing, (
        f"subpackages missing from pyproject.toml packages: {sorted(missing)}"
    )


def test_default_config_ships_as_package_data():
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]

    assert "*.yaml" in package_data["quant_platform.config"]
    assert (ROOT / "config" / "default.yaml").exists()
