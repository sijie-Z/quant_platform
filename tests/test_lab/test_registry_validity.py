"""Tests for research-run result validity metadata.

A fix that changes numerical output invalidates earlier results without
deleting them. `NO_LOOKAHEAD_CONTRACT.md`'s remediation procedure asks for the
affected versions to be *marked*, so the Registry needs somewhere to record
that -- otherwise "which results does this fix invalidate?" can only be
answered by reading commit messages.
"""

import sqlite3

import pytest
from quant_platform.lab.registry import RunStore


@pytest.fixture
def store(tmp_path):
    return RunStore(str(tmp_path / "registry.db"))


@pytest.fixture
def two_runs(store):
    # Inputs differ per run on purpose: run_id is `run_<unix seconds>_<input
    # hash>`, so identical inputs within the same second collide on the primary
    # key. That collision is a separate defect; this fixture sidesteps it so
    # these tests stay about validity.
    for i in (1, 2):
        store.begin_run(f"slice_{i}", {"factor": "roe", "n": i})
    return store.list_runs()


class TestSchema:
    def test_new_database_has_validity_columns(self, tmp_path):
        db = str(tmp_path / "new.db")
        RunStore(db)
        cols = {
            r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(research_runs)")
        }
        assert "validity" in cols
        assert "affected_by" in cols

    def test_existing_database_is_migrated(self, tmp_path):
        """CREATE TABLE IF NOT EXISTS does not add columns to an existing
        table, so the store has to alter it."""
        db = str(tmp_path / "old.db")
        conn = sqlite3.connect(db)
        conn.execute(
            """CREATE TABLE research_runs (
                   run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                   status TEXT NOT NULL, slice TEXT NOT NULL,
                   factor TEXT, evaluation TEXT)"""
        )
        conn.execute(
            "INSERT INTO research_runs VALUES "
            "('legacy', '2026-01-01', 'success', 's', 'roe', '{}')"
        )
        conn.commit()
        conn.close()

        store = RunStore(db)  # opening it must migrate

        cols = {
            r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(research_runs)")
        }
        assert {"validity", "affected_by"} <= cols
        # And the pre-existing row survives with a sensible default.
        assert store.get_run("legacy") is not None
        assert store.get_run("legacy")["validity"] == "valid"


class TestMarkValidity:
    def test_marking_a_run(self, store, two_runs):
        run_id = two_runs[0]["run_id"]
        assert store.mark_validity(run_id, "affected", "BUG-03") is True

        run = store.get_run(run_id)
        assert run["validity"] == "affected"
        assert "BUG-03" in run["affected_by"]

    def test_unknown_run_returns_false(self, store):
        assert store.mark_validity("does-not-exist", "affected", "BUG-03") is False

    def test_reasons_accumulate_without_duplicating(self, store, two_runs):
        """A run can be affected by several findings, but the same cause
        should not be recorded twice."""
        run_id = two_runs[0]["run_id"]
        store.mark_validity(run_id, "affected", "BUG-03")
        store.mark_validity(run_id, "affected", "BUG-03")  # duplicate
        store.mark_validity(run_id, "affected", "BUG-22")

        import json

        causes = json.loads(store.get_run(run_id)["affected_by"])
        assert causes == ["BUG-03", "BUG-22"]

    def test_other_runs_are_left_alone(self, store, two_runs):
        store.mark_validity(two_runs[0]["run_id"], "invalidated", "BUG-03")
        assert store.get_run(two_runs[1]["run_id"])["validity"] == "valid"

    def test_rejects_unknown_state(self, store, two_runs):
        with pytest.raises(ValueError, match="validity must be one of"):
            store.mark_validity(two_runs[0]["run_id"], "probably-fine", "x")


class TestListAffected:
    def test_empty_when_nothing_is_marked(self, store, two_runs):
        assert store.list_affected() == []

    def test_returns_only_marked_runs(self, store, two_runs):
        store.mark_validity(two_runs[0]["run_id"], "affected", "BUG-03")
        affected = store.list_affected()
        assert [r["run_id"] for r in affected] == [two_runs[0]["run_id"]]

    def test_filters_by_reason(self, store, two_runs):
        store.mark_validity(two_runs[0]["run_id"], "affected", "BUG-03")
        store.mark_validity(two_runs[1]["run_id"], "affected", "BUG-22")

        assert len(store.list_affected("BUG-03")) == 1
        assert store.list_affected("BUG-03")[0]["run_id"] == two_runs[0]["run_id"]
        assert len(store.list_affected("BUG-22")) == 1
        assert store.list_affected("BUG-99") == []

    def test_superseded_and_invalidated_are_included(self, store, two_runs):
        """Anything that is no longer plain 'valid' needs to surface."""
        store.mark_validity(two_runs[0]["run_id"], "superseded", "BUG-03")
        store.mark_validity(two_runs[1]["run_id"], "invalidated", "BUG-22")
        assert len(store.list_affected()) == 2
