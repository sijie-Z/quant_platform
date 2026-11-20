"""Tests for research-run result validity metadata.

A fix that changes numerical output invalidates earlier results without
deleting them. `NO_LOOKAHEAD_CONTRACT.md`'s remediation procedure asks for the
affected versions to be *marked*, so the Registry needs somewhere to record
that -- otherwise "which results does this fix invalidate?" can only be
answered by reading commit messages.
"""

import json
import sqlite3

import pytest
from quant_platform.lab.registry import RunStore


@pytest.fixture
def store(tmp_path):
    return RunStore(str(tmp_path / "registry.db"))


@pytest.fixture
def two_runs(store):
    # Inputs differ per run on purpose; run_id also carries a uuid4 suffix, so
    # identical inputs within the same second no longer collide.
    run_ids = []
    for i in (1, 2):
        run_ids.append(store.begin_run(f"slice_{i}", {
            "factor": "roe", "n": i, "data_source": "test_provider",
            "universe_meta": {"pit": False},
        }))
    # Rows open as status='failed' until finish_run flips them; a trusted
    # filter now requires a successful, fully traceable run.
    for run_id in run_ids:
        store.finish_run(run_id, status="success")
    return store.list_runs()


class TestTrustRequiresSuccess:
    def test_a_failed_run_is_not_trusted(self, store):
        """`begin_run` inserts status='failed' and validity defaults to
        'valid'; is_trusted used to return True for a run that never
        produced a result."""
        run_id = store.begin_run("slice", {"n": 1})

        assert store.is_trusted(run_id) is False
        assert store.list_runs(validity="valid") == []

    def test_a_successful_run_is_trusted(self, store):
        run_id = store.begin_run("slice", {
            "n": 2, "factor": "roe", "data_source": "test",
            "universe_meta": {"pit": False},
        })
        store.finish_run(run_id, status="success")

        assert store.is_trusted(run_id) is True
        assert [r["run_id"] for r in store.list_runs(validity="valid")] == [run_id]

    def test_success_without_lineage_is_not_trusted(self, store):
        """R4: success + validity=valid is not enough.

        The registry docstring promises full traceability (criterion #5).
        A run with data_source=None, factor=None and empty metadata used to
        be reported as trusted.
        """
        run_id = store.begin_run("slice", {})
        store.finish_run(run_id, status="success")

        assert store.is_trusted(run_id) is False
        assert store.get_run(run_id)["data_source"] is None
        assert store.has_complete_lineage(store.get_run(run_id)) is False

    @pytest.mark.parametrize("missing", ["data_source", "factor", "universe_meta"])
    def test_each_lineage_component_is_required(self, store, missing):
        inputs = {
            "data_source": "akshare",
            "factor": "roe",
            "universe_meta": {"pit": False},
            "n": 1,
        }
        inputs[missing] = "" if missing != "universe_meta" else {}
        run_id = store.begin_run("slice", inputs)
        store.finish_run(run_id, status="success")
        assert store.is_trusted(run_id) is False

    def test_declared_pit_none_is_not_lineage(self, store):
        """A universe_meta that declares pit=None is "undeclared", not PIT."""
        run_id = store.begin_run("slice", {
            "data_source": "akshare", "factor": "roe",
            "universe_meta": {"pit": None},
        })
        store.finish_run(run_id, status="success")
        assert store.is_trusted(run_id) is False

        # An explicit False is a declaration (the run is honest about its
        # survivorship bias) and does satisfy the lineage criterion.
        run_id2 = store.begin_run("slice", {
            "data_source": "akshare", "factor": "roe",
            "universe_meta": {"pit": False},
        })
        store.finish_run(run_id2, status="success")
        assert store.is_trusted(run_id2) is True

    def test_an_invalidated_success_is_not_trusted(self, store):
        run_id = store.begin_run("slice", {"n": 3})
        store.finish_run(run_id, status="success")
        store.mark_validity(run_id, "invalidated", "BUG-03")

        assert store.is_trusted(run_id) is False


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
        table, so the store has to alter it.

        R28: the old migration only added validity/affected_by, leaving the
        registry readable but unwritable — begin_run failed on the missing
        data_source column, finish_run on report_path. Every expected column
        is now added, and the migrated store must accept a full run.
        """
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
        assert {
            "data_source", "data_meta", "universe_provider", "universe_meta",
            "factor_params", "input_hash", "report_path", "warnings", "reason",
        } <= cols
        # And the pre-existing row survives with a sensible default.
        assert store.get_run("legacy") is not None
        assert store.get_run("legacy")["validity"] == "valid"

        # The migrated registry must be writable end to end.
        run_id = store.begin_run("migrated_slice", {
            "data_source": "akshare", "factor": "roe",
            "universe_meta": {"pit": False},
        })
        store.finish_run(run_id, status="success", evaluation={"ic": 0.1},
                         report_path="reports/x.md", warnings=["w1"])
        row = store.get_run(run_id)
        assert row["data_source"] == "akshare"
        assert row["report_path"] == "reports/x.md"
        assert json.loads(row["warnings"]) == ["w1"]
        assert store.is_trusted(run_id) is True


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


class TestPublicReadApi:
    def test_invalidated_row_is_machine_readable(self, store, two_runs):
        run_id = two_runs[0]["run_id"]
        store.mark_validity(run_id, "invalidated", "BUG-03")

        assert store.is_trusted(run_id) is False
        rows = {r["run_id"]: r for r in store.list_runs()}
        assert rows[run_id]["validity"] == "invalidated"
        assert json.loads(rows[run_id]["affected_by"]) == ["BUG-03"]
        assert [r["run_id"] for r in store.list_runs(validity="invalidated")] == [run_id]
        assert any(r["run_id"] != run_id for r in store.list_runs(validity="valid"))

    def test_mark_survives_reopen(self, store, two_runs, tmp_path):
        run_id = two_runs[0]["run_id"]
        store.mark_validity(run_id, "invalidated", "BUG-03")

        reopened = RunStore(str(tmp_path / "registry.db"))
        assert reopened.get_run(run_id)["validity"] == "invalidated"


class TestSnapshotBridge:
    def test_export_import_round_trip(self, store, two_runs, tmp_path):
        run_id = two_runs[0]["run_id"]
        store.mark_validity(run_id, "affected", "BUG-03")
        snapshot = tmp_path / "registry.json"
        assert store.export_runs(str(snapshot)) == 2

        other = RunStore(str(tmp_path / "other.db"))
        assert other.import_runs(str(snapshot)) == 2
        row = other.get_run(run_id)
        assert row["validity"] == "affected"
        assert json.loads(row["affected_by"]) == ["BUG-03"]

    def test_import_does_not_overwrite_local_marks(self, store, two_runs, tmp_path):
        run_id = two_runs[0]["run_id"]
        store.mark_validity(run_id, "invalidated", "BUG-03")
        snapshot = tmp_path / "registry.json"
        store.export_runs(str(snapshot))

        # A stale snapshot says "valid"; importing it must not undo the mark.
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        for row in payload:
            row["validity"] = "valid"
            row["affected_by"] = []
        snapshot.write_text(json.dumps(payload), encoding="utf-8")

        assert store.import_runs(str(snapshot)) == 0
        assert store.get_run(run_id)["validity"] == "invalidated"


class TestRunIdCollision:
    def test_identical_inputs_in_same_second_do_not_collide(self, store):
        """R8: run_id used to be run_<unix seconds>_<input hash>, so a second
        identical-input run started in the same second raised IntegrityError.
        """
        inputs = {"factor": "roe", "data_source": "test"}
        run_ids = [store.begin_run("slice", inputs) for _ in range(5)]

        assert len(set(run_ids)) == 5
        # input_hash stays a separate, identical field for reproducibility.
        hashes = {store.get_run(r)["input_hash"] for r in run_ids}
        assert len(hashes) == 1
        assert all(r.startswith("run_") for r in run_ids)


class TestWarningsAreAdditive:
    def test_second_finish_run_does_not_wipe_warnings(self, store):
        """R9: the failure path writes a traceback excerpt, then calls
        finish_run again after generating the report; the second call used to
        overwrite warnings with []."""
        run_id = store.begin_run("slice", {"warnings": ["input_warning"]})
        store.finish_run(
            run_id, status="failed", reason="boom",
            warnings=["traceback_excerpt: RuntimeError: boom"],
        )
        store.finish_run(run_id, status="failed", reason="boom",
                         report_path="reports/failure.md")

        warnings = json.loads(store.get_run(run_id)["warnings"])
        assert "input_warning" in warnings
        assert any(w.startswith("traceback_excerpt") for w in warnings)

    def test_explicit_warnings_merge_without_duplicates(self, store):
        run_id = store.begin_run("slice", {})
        store.finish_run(run_id, status="failed", warnings=["w1", "w2"])
        store.finish_run(run_id, status="failed", warnings=["w2", "w3"])

        assert json.loads(store.get_run(run_id)["warnings"]) == ["w1", "w2", "w3"]

    def test_data_meta_can_be_finalised_with_the_actual_window(self, store):
        """begin_run records the requested window; finish_run can replace it
        with the window the provider actually returned."""
        run_id = store.begin_run("slice", {
            "data_source": "akshare", "factor": "roe",
            "universe_meta": {"pit": False},
            "data_meta": {"requested_end": "2026-10-21"},
        })
        store.finish_run(
            run_id, status="success",
            data_meta={"requested_end": "2026-10-21",
                       "data_start": "2021-01-04",
                       "data_end": "2026-10-20"},
        )
        meta = json.loads(store.get_run(run_id)["data_meta"])
        assert meta["data_end"] == "2026-10-20"

    def test_mark_validity_is_atomic_read_modify_write(self, store):
        """The merge reads affected_by and writes it back; BEGIN IMMEDIATE
        makes the two-step update atomic."""
        run_id = store.begin_run("slice", {"data_source": "x", "factor": "f",
                                           "universe_meta": {"pit": False}})
        store.mark_validity(run_id, "affected", "BUG-1")
        store.mark_validity(run_id, "affected", "BUG-2")
        assert json.loads(store.get_run(run_id)["affected_by"]) == ["BUG-1", "BUG-2"]

    def test_cli_import_and_mark(self, store, two_runs, tmp_path):
        from types import SimpleNamespace

        from quant_platform.main import cmd_runs

        snapshot = tmp_path / "registry.json"
        store.export_runs(str(snapshot))
        target_db = str(tmp_path / "cli.db")

        assert cmd_runs(SimpleNamespace(
            db=target_db, subcommand="import", path=str(snapshot),
        )) == 0

        run_id = two_runs[0]["run_id"]
        assert cmd_runs(SimpleNamespace(
            db=target_db, subcommand="mark", run_id=run_id,
            state="invalidated", reason="BUG-03",
        )) == 0

        assert RunStore(target_db).get_run(run_id)["validity"] == "invalidated"
