"""Research run registry — the trust-bearing record store.

This is NOT a second trading-state store. It shares data/trading.db with
core.store (Principle of no-second-system) but uses a SEPARATE table
`research_runs` that the trading path never touches.

Per v0.1 acceptance criteria #5 and #6:
    #5 — every run is fully traceable: data lineage + universe trust + factor
         + evaluation + warnings.
    #6 — FAILED runs MUST also be recorded (status=failed + reason + trust_meta).
         A research system that only remembers successes produces severe
         selection bias. Knowledge records the negative too.

The Trust First discipline is enforced HERE, not in READMEs: this store reads
the provider's trust_meta() verbatim and persists it. Whatever approximation a
run used is permanently and machine-queryable in the record.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = "data/trading.db"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _content_hash(payload: dict[str, Any]) -> str:
    """Deterministic hash of a run's inputs for reproducibility."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class RunStore:
    """Minimal, dependency-free SQLite store for research runs.

    Schema is created with IF NOT EXISTS, so it coexists safely with the
    trading-state tables in the same database file.
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id            TEXT PRIMARY KEY,
                    timestamp         TEXT NOT NULL,
                    status            TEXT NOT NULL,           -- success | failed
                    reason            TEXT,                    -- failure reason if any
                    slice             TEXT NOT NULL,           -- e.g. first_honest_research_run
                    data_source       TEXT,
                    data_meta         TEXT,                    -- JSON: provider/adjust/basis/warnings
                    universe_provider TEXT,
                    universe_meta     TEXT,                    -- JSON: kind/pit/bias_warning
                    factor            TEXT,
                    factor_params     TEXT,                    -- JSON
                    evaluation        TEXT,                    -- JSON: ic/icir/dsr/bh_fdr/...
                    input_hash        TEXT,                    -- reproducibility fingerprint
                    report_path       TEXT,
                    warnings          TEXT,                    -- JSON list
                    validity          TEXT DEFAULT 'valid',    -- valid|affected|invalidated|superseded
                    affected_by       TEXT DEFAULT '[]'        -- JSON list of reasons, e.g. ["BUG-03"]
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status ON research_runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_factor ON research_runs(factor);
                """
            )

            # Databases created before the newer columns exist need them
            # added: CREATE TABLE IF NOT EXISTS leaves an existing table alone.
            # Earlier "migrations" only added validity/affected_by, leaving a
            # legacy table readable but unwritable: begin_run failed on the
            # missing data_source column and finish_run on report_path. Diff
            # every expected column and add what is missing.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(research_runs)")}
            missing = [
                (name, ddl)
                for name, ddl in self._MIGRATABLE_COLUMNS.items()
                if name not in cols
            ]
            for name, ddl in missing:
                conn.execute(f"ALTER TABLE research_runs ADD COLUMN {name} {ddl}")
            conn.commit()

    #: Columns that may be absent in a legacy ``research_runs`` table, with
    #: the DDL used to add them. The four NOT NULL columns (run_id, timestamp,
    #: status, slice) cannot be added by ALTER TABLE and must already exist —
    #: every historical schema in the project has them.
    _MIGRATABLE_COLUMNS = {
        "reason": "TEXT",
        "data_source": "TEXT",
        "data_meta": "TEXT",
        "universe_provider": "TEXT",
        "universe_meta": "TEXT",
        "factor": "TEXT",
        "factor_params": "TEXT",
        "evaluation": "TEXT",
        "input_hash": "TEXT",
        "report_path": "TEXT",
        "warnings": "TEXT",
        "validity": "TEXT DEFAULT 'valid'",
        "affected_by": "TEXT DEFAULT '[]'",
    }

    def begin_run(self, slice_name: str, inputs: dict[str, Any]) -> str:
        """Open a run record BEFORE work begins. Returns run_id.

        Recording starts in status='failed' so a crash leaves an honest trace;
        successful completion flips it to 'success' via finish_run(). This
        guarantees criterion #6 even on uncaught exceptions.
        """
        # A per-run uuid4 suffix keeps two identical-input runs started in
        # the same second from colliding on the primary key. The input hash
        # remains its own column for reproducibility queries.
        run_id = (
            f"run_{int(time.time())}_{_content_hash(inputs)}_{uuid.uuid4().hex[:8]}"
        )
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO research_runs
                   (run_id, timestamp, status, slice, data_source, data_meta,
                    universe_provider, universe_meta, factor, factor_params,
                    evaluation, input_hash, warnings)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    _utcnow_iso(),
                    "failed",  # default; flipped on success
                    slice_name,
                    inputs.get("data_source"),
                    json.dumps(inputs.get("data_meta", {}), default=str),
                    inputs.get("universe_provider"),
                    json.dumps(inputs.get("universe_meta", {}), default=str),
                    inputs.get("factor"),
                    json.dumps(inputs.get("factor_params", {}), default=str),
                    json.dumps({}, default=str),
                    _content_hash(inputs),
                    json.dumps(inputs.get("warnings", []), default=str),
                ),
            )
            conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        evaluation: dict | None = None,
        report_path: str | None = None,
        reason: str | None = None,
        warnings: list[str] | None = None,
        data_meta: dict | None = None,
    ) -> None:
        """Finalise a run.

        Warnings are additive: ``None`` leaves the stored list untouched (the
        failure path writes a traceback excerpt, then calls finish_run again
        after generating the report — the old unconditional overwrite wiped
        the excerpt to ``[]``), and an explicit list is merged into whatever
        is already stored, without duplicates.

        ``data_meta``, when provided, replaces the lineage metadata recorded
        at ``begin_run`` — used to persist the *actual* data window once the
        provider has returned it.
        """
        with self._conn() as conn:
            # Warnings are merged read-modify-write; take the write lock first
            # so two concurrent finalisers cannot drop each other's warnings.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT warnings, data_meta FROM research_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            try:
                existing = json.loads(row["warnings"] or "[]") if row else []
            except (json.JSONDecodeError, TypeError):
                existing = []
            if not isinstance(existing, list):
                existing = []
            merged = list(existing)
            for warning in warnings or []:
                if warning not in merged:
                    merged.append(warning)

            if data_meta is not None:
                new_data_meta = json.dumps(data_meta, default=str)
            else:
                new_data_meta = row["data_meta"] if row else None

            conn.execute(
                """UPDATE research_runs
                   SET status=?, evaluation=?, report_path=?, reason=?,
                       warnings=?, data_meta=?
                   WHERE run_id=?""",
                (
                    status,
                    json.dumps(evaluation or {}, default=str),
                    report_path,
                    reason,
                    json.dumps(merged, default=str),
                    new_data_meta,
                    run_id,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM research_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 50, validity: str | None = None) -> list[dict]:
        """Recent runs, optionally filtered by validity state.

        A validity filter makes "show me everything that is not trustworthy"
        a query rather than a Python loop over all rows.
        """
        query = "SELECT * FROM research_runs"
        params: list[Any] = []
        if validity is not None:
            if validity not in self.VALIDITY_STATES:
                raise ValueError(
                    f"validity must be one of {self.VALIDITY_STATES}, got {validity!r}"
                )
            query += " WHERE COALESCE(validity, 'valid')=?"
            params.append(validity)
            if validity in self.TRUSTED_STATES:
                # Validity defaults to 'valid' and `begin_run` opens rows as
                # 'failed', so filtering on validity alone returned runs that
                # never produced a result. A trusted result requires success.
                query += " AND status='success'"
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def has_complete_lineage(run: dict) -> bool:
        """Whether a run declares the lineage criterion #5 requires.

        A run is traceable only when it names its data source, its factor and
        the universe's point-in-time status. The old ``is_trusted`` accepted a
        success run with ``data_source=NULL``, ``factor=NULL`` and empty
        metadata, which is the exact opposite of what the module promises.
        """
        if not str(run.get("data_source") or "").strip():
            return False
        if not str(run.get("factor") or "").strip():
            return False
        try:
            universe_meta = json.loads(run.get("universe_meta") or "{}")
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(universe_meta, dict):
            return False
        # A present-but-None pit is "undeclared" in the generated reports,
        # so it does not satisfy the lineage criterion either.
        return universe_meta.get("pit") is not None

    def is_trusted(self, run_id: str) -> bool:
        """Whether a run is successful, still valid, and fully traceable."""
        run = self.get_run(run_id)
        if run is None:
            return False
        return (
            run.get("status") == "success"
            and (run.get("validity") or "valid") in self.TRUSTED_STATES
            and self.has_complete_lineage(run)
        )

    # ── Result validity ─────────────────────────────────────────────────
    #
    # A bug fix that changes numerical output invalidates earlier results
    # without making them disappear. NO_LOOKAHEAD_CONTRACT's own remediation
    # procedure asks for the affected versions to be *marked*, not deleted:
    #
    #     1. 这是一个 bug，提交 issue 或 PR
    #     2. 修复后在该文件中更新"已在代码中实现的位置"
    #     3. 如果该 bug 影响了之前的回测结果，标注 affected 版本
    #
    # These methods are what makes step 3 answerable by machine instead of by
    # reading commit messages.

    VALIDITY_STATES = ("valid", "affected", "invalidated", "superseded")
    VALIDITY_MEANINGS = {
        "valid": "trusted as-is",
        "affected": "a fixed defect may change the numbers; treat as suspect",
        "invalidated": "known wrong; do not use",
        "superseded": "replaced by a newer run",
    }
    TRUSTED_STATES = ("valid",)

    def mark_validity(self, run_id: str, validity: str, reason: str = "") -> bool:
        """Record whether a run's result can still be trusted.

        Args:
            run_id: The run to mark.
            validity: One of VALIDITY_STATES.
            reason: What affected it, e.g. "BUG-03". Appended to the run's
                affected_by list if not already present, so a run can be
                affected by several things and the history is additive.

        Returns:
            True if the run existed and was updated.
        """
        if validity not in self.VALIDITY_STATES:
            raise ValueError(
                f"validity must be one of {self.VALIDITY_STATES}, got {validity!r}"
            )

        with self._conn() as conn:
            # Read-modify-write on affected_by: two concurrent marks with
            # different reasons could each read the same list and the second
            # UPDATE would drop the first reason. Take the write lock before
            # the SELECT so the merge is atomic.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT affected_by FROM research_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return False

            try:
                existing = json.loads(row["affected_by"] or "[]")
            except json.JSONDecodeError:
                existing = []
            if reason and reason not in existing:
                existing.append(reason)

            conn.execute(
                "UPDATE research_runs SET validity=?, affected_by=? WHERE run_id=?",
                (validity, json.dumps(existing), run_id),
            )
        return True

    def list_affected(self, reason: str | None = None) -> list[dict]:
        """Runs whose results are no longer plain 'valid'.

        Args:
            reason: Optional filter, e.g. "BUG-03" -- returns only runs marked
                as affected by that specific cause.

        Returns:
            Matching runs, newest first.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM research_runs "
                "WHERE COALESCE(validity, 'valid') != 'valid' "
                "ORDER BY timestamp DESC"
            ).fetchall()

        out = [dict(r) for r in rows]
        if reason:
            filtered = []
            for r in out:
                try:
                    causes = json.loads(r.get("affected_by") or "[]")
                except json.JSONDecodeError:
                    causes = []
                if reason in causes:
                    filtered.append(r)
            out = filtered
        return out

    # ── Snapshot bridge ─────────────────────────────────────────────────
    #
    # The runtime database is gitignored; the committed JSON export is the
    # only copy of the historical corpus. These two methods make that corpus
    # importable (so its validity can be marked) and re-exportable.

    _COLUMNS = (
        "run_id", "timestamp", "status", "reason", "slice", "data_source",
        "data_meta", "universe_provider", "universe_meta", "factor",
        "factor_params", "evaluation", "input_hash", "report_path",
        "warnings", "validity", "affected_by",
    )
    _JSON_FIELDS = (
        "data_meta", "universe_meta", "factor_params", "evaluation",
        "warnings", "affected_by",
    )

    def import_runs(self, path: str) -> int:
        """Load a JSON export into this registry. Returns rows inserted.

        ``INSERT OR IGNORE`` leaves an existing local row untouched, so
        importing the committed snapshot never overwrites a validity mark
        that was made locally.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("registry export must be a JSON list")

        imported = 0
        with self._conn() as conn:
            for record in payload:
                if not isinstance(record, dict):
                    raise ValueError("every registry row must be a JSON object")
                missing = {"run_id", "timestamp", "status", "slice"} - set(record)
                if missing:
                    raise ValueError(f"registry row missing {sorted(missing)}: {record!r}")

                values = dict(record)
                if not values.get("validity"):
                    values["validity"] = "valid"
                for field in self._JSON_FIELDS:
                    value = values.get(field)
                    if value is None:
                        values[field] = "[]" if field in ("warnings", "affected_by") else "{}"
                    elif not isinstance(value, str):
                        values[field] = json.dumps(value, default=str)

                placeholders = ",".join("?" for _ in self._COLUMNS)
                cursor = conn.execute(
                    f"INSERT OR IGNORE INTO research_runs ({','.join(self._COLUMNS)}) "
                    f"VALUES ({placeholders})",
                    tuple(values.get(column) for column in self._COLUMNS),
                )
                imported += cursor.rowcount
            conn.commit()
        return imported

    def export_runs(self, path: str, validity: str | None = None) -> int:
        """Write runs to a JSON snapshot, newest first. Returns the row count."""
        rows = self.list_runs(limit=1_000_000, validity=validity)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return len(rows)
