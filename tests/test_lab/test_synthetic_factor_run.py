"""Tests for the offline synthetic factor runner (P0 validation path).

These tests must never touch `data/trading.db` or `data/reports/`. The
Registry is the project's research knowledge base -- docs/ARCHITECTURE_V2.md
makes "Knowledge compounds" a design principle -- so a test run writing into
it is indistinguishable from a real research run. `run_synthetic_factor`
therefore takes explicit `db_path` / `out_dir`, and every test here overrides
them.
"""

from pathlib import Path

import pytest
from quant_platform.lab.registry import RunStore
from quant_platform.lab.runs.synthetic_factor_run import (
    available_factors,
    run_synthetic_factor,
)


@pytest.fixture
def isolated_paths(tmp_path):
    """Registry and report directory kept well away from the real data/ tree."""
    return {"db_path": tmp_path / "research.db", "out_dir": tmp_path / "reports"}


def test_available_factors_contains_momentum():
    assert "momentum_12m" in available_factors()


def test_run_synthetic_factor_persists_report(isolated_paths):
    run_id = run_synthetic_factor(
        "momentum_12m", n_stocks=3, n_days=120, seed=1, **isolated_paths
    )
    assert run_id.startswith("run_")
    assert (isolated_paths["out_dir"] / f"{run_id}.md").exists()


def test_run_is_recorded_in_the_given_registry(isolated_paths):
    """Positive proof that `db_path` is honoured, not silently ignored."""
    run_id = run_synthetic_factor(
        "momentum_12m", n_stocks=3, n_days=120, seed=1, **isolated_paths
    )
    store = RunStore(str(isolated_paths["db_path"]))
    assert store.get_run(run_id) is not None


def test_run_does_not_write_into_the_real_data_tree(isolated_paths):
    """Regression guard: the runner used to hardcode data/reports + DEFAULT_DB."""
    real_reports = Path("data/reports")
    before = {p.name for p in real_reports.glob("run_*.md")} if real_reports.exists() else set()

    run_synthetic_factor("momentum_12m", n_stocks=3, n_days=120, seed=1, **isolated_paths)

    after = {p.name for p in real_reports.glob("run_*.md")} if real_reports.exists() else set()
    assert after == before
