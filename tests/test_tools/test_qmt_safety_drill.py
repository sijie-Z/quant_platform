"""The executable QMT safety drill suite must stay green."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "qmt_safety_drill.py"


@pytest.fixture()
def drill():
    spec = importlib.util.spec_from_file_location("qmt_safety_drill_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_fault_injection_drills_pass(drill, tmp_path):
    drill.WORK = tmp_path
    for name, function in drill.DRILLS.items():
        details = function()
        assert details is not None, name
