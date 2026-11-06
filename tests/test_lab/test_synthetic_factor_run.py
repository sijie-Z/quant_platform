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


def test_zero_ic_observations_is_not_a_success(isolated_paths):
    """R5: momentum_12m needs 252+21 days; a 120-day panel produced
    n_ic_obs=0 but was recorded as status=success in the live registry."""
    import json

    run_id = run_synthetic_factor(
        "momentum_12m", n_stocks=3, n_days=120, seed=1, **isolated_paths
    )
    store = RunStore(str(isolated_paths["db_path"]))
    row = store.get_run(run_id)

    evaluation = json.loads(row["evaluation"])
    assert evaluation["n_ic_obs"] == 0
    assert row["status"] == "failed"
    assert "insufficient_ic_observations" in (row["reason"] or "")
    assert store.is_trusted(run_id) is False

    report = (isolated_paths["out_dir"] / f"{run_id}.md").read_text(encoding="utf-8")
    assert "RUN FAILED" in report
    assert "insufficient_ic_observations" in report


def test_synthetic_run_records_lineage(isolated_paths):
    """R26: synthetic runs used to carry NULL data_meta/universe_meta."""
    import json

    run_id = run_synthetic_factor(
        "momentum_12m", n_stocks=5, n_days=500, seed=3, **isolated_paths
    )
    store = RunStore(str(isolated_paths["db_path"]))
    row = store.get_run(run_id)

    data_meta = json.loads(row["data_meta"])
    universe_meta = json.loads(row["universe_meta"])
    assert row["data_source"] == "synthetic"
    assert row["factor"] == "momentum_12m"
    assert data_meta["provider"] == "synthetic"
    assert data_meta["seed"] == 3
    assert data_meta["panel_start"] and data_meta["panel_end"]
    assert universe_meta["pit"] is False
    assert "synthetic_data_not_market_data" in universe_meta["bias_warning"]

    evaluation = json.loads(row["evaluation"])
    assert evaluation["n_ic_obs"] >= 20
    assert row["status"] == "success"
