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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = "data/trading.db"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
                    warnings          TEXT                     -- JSON list
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status ON research_runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_factor ON research_runs(factor);
                """
            )
            conn.commit()

    def begin_run(self, slice_name: str, inputs: dict[str, Any]) -> str:
        """Open a run record BEFORE work begins. Returns run_id.

        Recording starts in status='failed' so a crash leaves an honest trace;
        successful completion flips it to 'success' via finish_run(). This
        guarantees criterion #6 even on uncaught exceptions.
        """
        run_id = f"run_{int(time.time())}_{_content_hash(inputs)}"
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
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE research_runs
                   SET status=?, evaluation=?, report_path=?, reason=?, warnings=?
                   WHERE run_id=?""",
                (
                    status,
                    json.dumps(evaluation or {}, default=str),
                    report_path,
                    reason,
                    json.dumps(warnings or [], default=str),
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

    def list_runs(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM research_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
