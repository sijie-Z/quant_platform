"""Issue #77: unwired packages must carry a machine-readable status.

Seven packages have zero runtime importers. They are annotated (rather than
deleted) so the decision is explicit; this test keeps the annotations from
silently disappearing.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UNWIRED = [
    "framework/contracts",
    "services",
    "kernel",
    "shared",
    "regime_router",
    "compliance",
    "research",
]


@pytest.mark.parametrize("package", UNWIRED)
def test_unwired_package_declares_its_status(package):
    init = REPO / package / "__init__.py"
    assert init.exists(), f"missing {init}"

    head = init.read_text(encoding="utf-8")[:800]
    assert "STATUS:" in head, f"{package} lost its STATUS banner"
