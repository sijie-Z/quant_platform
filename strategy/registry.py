"""Strategy registry — versioned persistence for strategy DSL definitions.

Persists strategy definitions and run history in SQLite.
Enables: strategy versioning, run audit trail, comparison.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.strategy.dsl import StrategyDefinition
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyRegistry:
    """Persistent registry for strategy definitions and run history."""

    def __init__(self, db_path: str = "data/strategy_registry.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS strategy_definitions (
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (name, version)
                );

                CREATE TABLE IF NOT EXISTS strategy_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    config TEXT DEFAULT '{}',
                    summary TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'completed',
                    started_at TEXT NOT NULL,
                    completed_at TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_runs_name ON strategy_runs(strategy_name);
                CREATE INDEX IF NOT EXISTS idx_runs_time ON strategy_runs(started_at);
            """)
        logger.info("StrategyRegistry initialized: %s", self._db_path)

    def save(self, strategy: StrategyDefinition) -> None:
        """Save or update a strategy definition."""
        now = strategy.created_at or datetime.now().isoformat()
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO strategy_definitions
                (name, version, definition, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                strategy.name, strategy.version,
                json.dumps(strategy.to_dict()),
                now,
            ))
        logger.info("Saved strategy: %s v%s", strategy.name, strategy.version)

    def get(self, name: str, version: str = "") -> StrategyDefinition | None:
        """Get a strategy definition by name and optional version."""
        with self._conn() as conn:
            if version:
                row = conn.execute(
                    "SELECT * FROM strategy_definitions WHERE name = ? AND version = ?",
                    (name, version),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM strategy_definitions WHERE name = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (name,),
                ).fetchone()
            if row is None:
                return None
            data = json.loads(row["definition"])
            return StrategyDefinition.from_dict(data)

    def list_strategies(self) -> list[dict]:
        """List all strategies with latest version info."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT name, version, created_at
                FROM strategy_definitions
                ORDER BY name, created_at DESC
            """).fetchall()
            seen = set()
            result = []
            for r in rows:
                if r["name"] not in seen:
                    seen.add(r["name"])
                    result.append(dict(r))
            return result

    def list_versions(self, name: str) -> list[dict]:
        """List all versions of a strategy."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, version, created_at FROM strategy_definitions "
                "WHERE name = ? ORDER BY created_at DESC",
                (name,),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_run(
        self,
        strategy_name: str,
        strategy_version: str,
        run_id: str,
        config: dict | None = None,
        summary: dict | None = None,
    ) -> None:
        """Record a strategy run."""
        now = datetime.now().isoformat()
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO strategy_runs
                (strategy_name, strategy_version, run_id, config, summary,
                 status, started_at)
                VALUES (?, ?, ?, ?, ?, 'completed', ?)
            """, (
                strategy_name, strategy_version, run_id,
                json.dumps(config or {}),
                json.dumps(summary or {}),
                now,
            ))

    def get_run_history(
        self, strategy_name: str = "", limit: int = 20
    ) -> list[dict]:
        """Get run history, optionally filtered by strategy name."""
        with self._conn() as conn:
            if strategy_name:
                rows = conn.execute(
                    "SELECT * FROM strategy_runs WHERE strategy_name = ? "
                    "ORDER BY started_at DESC LIMIT ?",
                    (strategy_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM strategy_runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d.get("config", "{}"))
                d["summary"] = json.loads(d.get("summary", "{}"))
                result.append(d)
            return result

    def get_stats(self) -> dict:
        """Get registry statistics."""
        with self._conn() as conn:
            return {
                "strategies": conn.execute(
                    "SELECT COUNT(DISTINCT name) FROM strategy_definitions"
                ).fetchone()[0],
                "versions": conn.execute(
                    "SELECT COUNT(*) FROM strategy_definitions"
                ).fetchone()[0],
                "runs": conn.execute(
                    "SELECT COUNT(*) FROM strategy_runs"
                ).fetchone()[0],
            }
