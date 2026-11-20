"""Factor Research Store — persistent factor research database.

Stores factor definitions, values, evaluation history, backtest history,
walk-forward fold history, stability diagnostics, and regime diagnostics.

Purpose: factor research reproducibility and lifecycle management.
Every factor study leaves a trace in SQLite — you can always go back
and ask "why did this factor stop working?"

Inspired by quawn's Factor Store design.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    """Timezone-aware UTC timestamp, matching the research registry.

    The old naive ``datetime.now().isoformat()`` mixed local-time records
    with the registry's UTC ones; cross-store ordering and joins could not be
    compared.
    """
    return datetime.now(UTC).isoformat()


@dataclass
class FactorDefinition:
    """Definition metadata for a factor."""
    name: str
    category: str  # technical / fundamental / custom
    description: str = ""
    higher_is_better: bool = True
    fundamental_required: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "higher_is_better": self.higher_is_better,
            "fundamental_required": self.fundamental_required,
            "params": self.params,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FactorDefinition:
        return cls(
            name=d["name"],
            category=d.get("category", "technical"),
            description=d.get("description", ""),
            higher_is_better=d.get("higher_is_better", True),
            fundamental_required=d.get("fundamental_required", False),
            params=d.get("params", {}),
            version=d.get("version", "1.0.0"),
        )


@dataclass
class FactorEvalRecord:
    """Single factor evaluation result."""
    factor_name: str
    signal_date: str
    rank_ic: float
    pearson_ic: float
    icir: float
    coverage: float
    n_assets: int
    run_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    # Lineage / lifecycle (audit R10): without these an evaluation row cannot
    # be traced to a provider or invalidated later.
    data_source: str = ""
    adjust: str = ""
    pit: bool | None = None
    validity: str = "valid"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "signal_date": self.signal_date,
            "rank_ic": round(self.rank_ic, 6),
            "pearson_ic": round(self.pearson_ic, 6),
            "icir": round(self.icir, 6),
            "coverage": round(self.coverage, 4),
            "n_assets": self.n_assets,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data_source": self.data_source,
            "adjust": self.adjust,
            "pit": self.pit,
            "validity": self.validity,
            "warnings": list(self.warnings),
        }


@dataclass
class FactorBacktestRecord:
    """Single factor long-short backtest result."""
    factor_name: str
    signal_date: str
    long_return: float
    short_return: float
    spread_return: float
    long_n: int
    short_n: int
    turnover: float
    run_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    data_source: str = ""
    adjust: str = ""
    pit: bool | None = None
    validity: str = "valid"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "signal_date": self.signal_date,
            "long_return": round(self.long_return, 6),
            "short_return": round(self.short_return, 6),
            "spread_return": round(self.spread_return, 6),
            "long_n": self.long_n,
            "short_n": self.short_n,
            "turnover": round(self.turnover, 6),
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data_source": self.data_source,
            "adjust": self.adjust,
            "pit": self.pit,
            "validity": self.validity,
            "warnings": list(self.warnings),
        }


@dataclass
class WalkForwardFoldRecord:
    """Single walk-forward fold result."""
    factor_name: str
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    train_ic: float
    test_ic: float
    run_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    data_source: str = ""
    adjust: str = ""
    pit: bool | None = None
    validity: str = "valid"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "fold_id": self.fold_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_sharpe": round(self.train_sharpe, 4),
            "test_sharpe": round(self.test_sharpe, 4),
            "train_ic": round(self.train_ic, 6),
            "test_ic": round(self.test_ic, 6),
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data_source": self.data_source,
            "adjust": self.adjust,
            "pit": self.pit,
            "validity": self.validity,
            "warnings": list(self.warnings),
        }


@dataclass
class FactorStabilityRecord:
    """Stability diagnostics for a factor."""
    factor_name: str
    ic_mean: float
    ic_std: float
    icir: float
    sharpe_mean: float
    sharpe_std: float
    sharpe_consistency: float  # fraction of folds with positive sharpe
    positive_folds: int
    total_folds: int
    coverage: float
    run_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    data_source: str = ""
    adjust: str = ""
    pit: bool | None = None
    validity: str = "valid"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "factor_name": self.factor_name,
            "ic_mean": round(self.ic_mean, 6),
            "ic_std": round(self.ic_std, 6),
            "icir": round(self.icir, 4),
            "sharpe_mean": round(self.sharpe_mean, 4),
            "sharpe_std": round(self.sharpe_std, 4),
            "sharpe_consistency": round(self.sharpe_consistency, 4),
            "positive_folds": self.positive_folds,
            "total_folds": self.total_folds,
            "coverage": round(self.coverage, 4),
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data_source": self.data_source,
            "adjust": self.adjust,
            "pit": self.pit,
            "validity": self.validity,
            "warnings": list(self.warnings),
        }


class FactorResearchStore:
    """Persistent factor research database.

    Stores factor definitions, values, and all evaluation/validation history
    in SQLite. Thread-safe, WAL mode, same pattern as core Store.
    """

    def __init__(self, db_path: str = "data/factor_research.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ── Schema ──

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
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
                CREATE TABLE IF NOT EXISTS factor_definitions (
                    factor_name TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    higher_is_better INTEGER DEFAULT 1,
                    fundamental_required INTEGER DEFAULT 0,
                    params TEXT DEFAULT '{}',
                    version TEXT DEFAULT '1.0.0',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    value REAL,
                    coverage REAL,
                    version TEXT DEFAULT '1.0.0',
                    data_source TEXT DEFAULT '',
                    adjust TEXT DEFAULT '',
                    pit INTEGER,
                    validity TEXT DEFAULT 'valid',
                    warnings TEXT DEFAULT '[]',
                    run_id TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(factor_name, symbol, signal_date, version)
                );

                CREATE TABLE IF NOT EXISTS factor_evaluation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    rank_ic REAL,
                    pearson_ic REAL,
                    icir REAL,
                    coverage REAL,
                    n_assets INTEGER DEFAULT 0,
                    run_id TEXT DEFAULT '',
                    data_source TEXT DEFAULT '',
                    adjust TEXT DEFAULT '',
                    pit INTEGER,
                    validity TEXT DEFAULT 'valid',
                    warnings TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_backtest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    long_return REAL,
                    short_return REAL,
                    spread_return REAL,
                    long_n INTEGER DEFAULT 0,
                    short_n INTEGER DEFAULT 0,
                    turnover REAL DEFAULT 0,
                    run_id TEXT DEFAULT '',
                    data_source TEXT DEFAULT '',
                    adjust TEXT DEFAULT '',
                    pit INTEGER,
                    validity TEXT DEFAULT 'valid',
                    warnings TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_walk_forward_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    fold_id INTEGER NOT NULL,
                    train_start TEXT,
                    train_end TEXT,
                    test_start TEXT,
                    test_end TEXT,
                    train_sharpe REAL,
                    test_sharpe REAL,
                    train_ic REAL,
                    test_ic REAL,
                    run_id TEXT DEFAULT '',
                    data_source TEXT DEFAULT '',
                    adjust TEXT DEFAULT '',
                    pit INTEGER,
                    validity TEXT DEFAULT 'valid',
                    warnings TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_stability_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    ic_mean REAL,
                    ic_std REAL,
                    icir REAL,
                    sharpe_mean REAL,
                    sharpe_std REAL,
                    sharpe_consistency REAL,
                    positive_folds INTEGER DEFAULT 0,
                    total_folds INTEGER DEFAULT 0,
                    coverage REAL DEFAULT 0,
                    run_id TEXT DEFAULT '',
                    data_source TEXT DEFAULT '',
                    adjust TEXT DEFAULT '',
                    pit INTEGER,
                    validity TEXT DEFAULT 'valid',
                    warnings TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    rank_ic REAL,
                    n_samples INTEGER DEFAULT 0,
                    run_id TEXT DEFAULT '',
                    data_source TEXT DEFAULT '',
                    adjust TEXT DEFAULT '',
                    pit INTEGER,
                    validity TEXT DEFAULT 'valid',
                    warnings TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factor_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factor_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    params TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_fv_lookup ON factor_values(factor_name, symbol, signal_date);
                CREATE INDEX IF NOT EXISTS idx_feh_factor ON factor_evaluation_history(factor_name, signal_date);
                CREATE INDEX IF NOT EXISTS idx_fbh_factor ON factor_backtest_history(factor_name, signal_date);
                CREATE INDEX IF NOT EXISTS idx_fwh_factor ON factor_walk_forward_history(factor_name, fold_id);
                CREATE INDEX IF NOT EXISTS idx_fsh_factor ON factor_stability_history(factor_name);
                CREATE INDEX IF NOT EXISTS idx_frh_factor ON factor_regime_history(factor_name, regime);
            """)
            self._migrate_schema(conn)
        logger.info("FactorResearchStore initialized: %s", self._db_path)

    #: Provenance/lifecycle columns added after the original schema. Existing
    #: databases get them through ALTER TABLE; without the migration, writes
    #: with lineage metadata fail on a store created by an older checkout.
    _PROVENANCE_COLUMNS = {
        "data_source": "TEXT DEFAULT ''",
        "adjust": "TEXT DEFAULT ''",
        "pit": "INTEGER",
        "validity": "TEXT DEFAULT 'valid'",
        "warnings": "TEXT DEFAULT '[]'",
    }
    _PROVENANCE_TABLES = (
        "factor_values",
        "factor_evaluation_history",
        "factor_backtest_history",
        "factor_walk_forward_history",
        "factor_stability_history",
        "factor_regime_history",
    )

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        for table in self._PROVENANCE_TABLES:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for name, ddl in self._PROVENANCE_COLUMNS.items():
                if name not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        # factor_values predates the run_id convention; history tables had it.
        value_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(factor_values)")
        }
        if "run_id" not in value_cols:
            conn.execute(
                "ALTER TABLE factor_values ADD COLUMN run_id TEXT DEFAULT ''"
            )
        conn.commit()

    # ── Factor Definitions ──

    def save_definition(self, definition: FactorDefinition) -> None:
        """Upsert a factor definition."""
        now = _utcnow_iso()
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO factor_definitions
                (factor_name, category, description, higher_is_better,
                 fundamental_required, params, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(factor_name) DO UPDATE SET
                    category = excluded.category,
                    description = excluded.description,
                    higher_is_better = excluded.higher_is_better,
                    fundamental_required = excluded.fundamental_required,
                    params = excluded.params,
                    version = excluded.version,
                    updated_at = excluded.updated_at
            """, (
                definition.name, definition.category, definition.description,
                int(definition.higher_is_better), int(definition.fundamental_required),
                json.dumps(definition.params), definition.version,
                now, now,
            ))

    def get_definition(self, factor_name: str) -> FactorDefinition | None:
        """Get a factor definition by name."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM factor_definitions WHERE factor_name = ?",
                (factor_name,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            return FactorDefinition(
                name=d["factor_name"],
                category=d["category"],
                description=d.get("description", ""),
                higher_is_better=bool(d["higher_is_better"]),
                fundamental_required=bool(d["fundamental_required"]),
                params=json.loads(d.get("params", "{}")),
                version=d.get("version", "1.0.0"),
            )

    def list_definitions(self) -> list[FactorDefinition]:
        """List all registered factor definitions."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM factor_definitions ORDER BY factor_name"
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                result.append(FactorDefinition(
                    name=d["factor_name"],
                    category=d["category"],
                    description=d.get("description", ""),
                    higher_is_better=bool(d["higher_is_better"]),
                    fundamental_required=bool(d["fundamental_required"]),
                    params=json.loads(d.get("params", "{}")),
                    version=d.get("version", "1.0.0"),
                ))
            return result

    # ── Factor Values ──

    def save_values(
        self,
        factor_name: str,
        values: list[dict],
        coverage: float | None = None,
        version: str = "1.0.0",
        data_source: str = "",
        adjust: str = "",
        pit: bool | None = None,
        validity: str = "valid",
        warnings: list[str] | None = None,
        run_id: str = "",
    ) -> int:
        """Save factor values for multiple (symbol, signal_date) pairs.

        Args:
            factor_name: Factor identifier.
            values: List of dicts with 'symbol', 'signal_date', 'value' keys.
            coverage: Optional coverage fraction.
            version: Factor version.
            data_source: Provider that produced the values (lineage).
            adjust: Price adjustment basis, e.g. "hfq".
            pit: Whether the universe/values are point-in-time; None = undeclared.
            validity: valid | affected | invalidated | superseded.
            warnings: Data-quality warnings stored with the rows.
            run_id: Research run that produced these values.

        Returns:
            Number of rows inserted/updated.
        """
        now = _utcnow_iso()
        warnings_json = json.dumps(list(warnings or []))
        pit_val = None if pit is None else int(bool(pit))
        rows = [
            (factor_name, v["symbol"], v["signal_date"], v.get("value"),
             coverage, version, data_source, adjust, pit_val, validity,
             warnings_json, run_id, now)
            for v in values
        ]
        if not rows:
            return 0
        with self._lock, self._conn() as conn:
            before = conn.total_changes
            conn.executemany("""
                INSERT INTO factor_values
                (factor_name, symbol, signal_date, value, coverage, version,
                 data_source, adjust, pit, validity, warnings, run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(factor_name, symbol, signal_date, version) DO UPDATE SET
                    value = excluded.value,
                    coverage = excluded.coverage,
                    data_source = excluded.data_source,
                    adjust = excluded.adjust,
                    pit = excluded.pit,
                    validity = excluded.validity,
                    warnings = excluded.warnings,
                    run_id = excluded.run_id
            """, rows)
            return conn.total_changes - before

    def get_values(
        self,
        factor_name: str,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Query factor values."""
        with self._conn() as conn:
            query = "SELECT * FROM factor_values WHERE factor_name = ?"
            params: list[Any] = [factor_name]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                query += " AND signal_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND signal_date <= ?"
                params.append(end_date)
            query += " ORDER BY signal_date DESC LIMIT ?"
            params.append(limit)
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    # ── Evaluation History ──

    def save_evaluation(self, record: FactorEvalRecord) -> None:
        """Save a factor evaluation record."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO factor_evaluation_history
                (factor_name, signal_date, rank_ic, pearson_ic, icir,
                 coverage, n_assets, run_id, data_source, adjust, pit,
                 validity, warnings, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.factor_name, record.signal_date,
                record.rank_ic, record.pearson_ic, record.icir,
                record.coverage, record.n_assets,
                record.run_id, record.data_source, record.adjust,
                None if record.pit is None else int(bool(record.pit)),
                record.validity, json.dumps(list(record.warnings)),
                record.timestamp,
            ))

    def get_evaluation_history(
        self,
        factor_name: str,
        limit: int = 252,
    ) -> list[FactorEvalRecord]:
        """Get recent evaluation records for a factor."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM factor_evaluation_history
                WHERE factor_name = ?
                ORDER BY signal_date DESC LIMIT ?
            """, (factor_name, limit)).fetchall()
            return [FactorEvalRecord(
                factor_name=r["factor_name"],
                signal_date=r["signal_date"],
                # NULL metrics (a run that could not measure them) stay
                # missing on read: coercing to 0.0 made `factor-store history`
                # print measured-looking zeros for unmeasured rows.
                rank_ic=float(r["rank_ic"]) if r["rank_ic"] is not None else float("nan"),
                pearson_ic=float(r["pearson_ic"]) if r["pearson_ic"] is not None else float("nan"),
                icir=float(r["icir"]) if r["icir"] is not None else float("nan"),
                coverage=float(r["coverage"]) if r["coverage"] is not None else float("nan"),
                n_assets=r["n_assets"] or 0,
                run_id=r["run_id"] if r["run_id"] else "",
                timestamp=r["timestamp"],
                data_source=r["data_source"] or "",
                adjust=r["adjust"] or "",
                pit=bool(r["pit"]) if r["pit"] is not None else None,
                validity=r["validity"] or "valid",
                warnings=json.loads(r["warnings"] or "[]"),
            ) for r in rows]

    # ── Backtest History ──

    def save_backtest(self, record: FactorBacktestRecord) -> None:
        """Save a factor backtest record."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO factor_backtest_history
                (factor_name, signal_date, long_return, short_return,
                 spread_return, long_n, short_n, turnover, run_id,
                 data_source, adjust, pit, validity, warnings, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.factor_name, record.signal_date,
                record.long_return, record.short_return, record.spread_return,
                record.long_n, record.short_n, record.turnover,
                record.run_id, record.data_source, record.adjust,
                None if record.pit is None else int(bool(record.pit)),
                record.validity, json.dumps(list(record.warnings)),
                record.timestamp,
            ))

    # ── Walk-Forward History ──

    def save_walk_forward_fold(self, record: WalkForwardFoldRecord) -> None:
        """Save a walk-forward fold record."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO factor_walk_forward_history
                (factor_name, fold_id, train_start, train_end,
                 test_start, test_end, train_sharpe, test_sharpe,
                 train_ic, test_ic, run_id, data_source, adjust, pit,
                 validity, warnings, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.factor_name, record.fold_id,
                record.train_start, record.train_end,
                record.test_start, record.test_end,
                record.train_sharpe, record.test_sharpe,
                record.train_ic, record.test_ic,
                record.run_id, record.data_source, record.adjust,
                None if record.pit is None else int(bool(record.pit)),
                record.validity, json.dumps(list(record.warnings)),
                record.timestamp,
            ))

    def get_walk_forward_summary(self, factor_name: str) -> dict | None:
        """Get aggregated walk-forward results for a factor."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT
                    COUNT(*) as n_folds,
                    AVG(test_sharpe) as mean_test_sharpe,
                    AVG(train_sharpe) as mean_train_sharpe,
                    AVG(test_ic) as mean_test_ic,
                    AVG(train_ic) as mean_train_ic,
                    SUM(CASE WHEN test_sharpe > 0 THEN 1 ELSE 0 END) as positive_folds
                FROM factor_walk_forward_history
                WHERE factor_name = ?
            """, (factor_name,)).fetchall()
            row = rows[0] if rows else None
            if row is None or row["n_folds"] == 0:
                return None
            d = dict(row)
            return {
                "factor_name": factor_name,
                "n_folds": d["n_folds"],
                "mean_test_sharpe": round(d["mean_test_sharpe"], 4),
                "mean_train_sharpe": round(d["mean_train_sharpe"], 4),
                "mean_test_ic": round(d["mean_test_ic"], 6),
                "mean_train_ic": round(d["mean_train_ic"], 6),
                "positive_folds": d["positive_folds"],
                "consistency": round(d["positive_folds"] / max(d["n_folds"], 1), 4),
            }

    # ── Stability History ──

    def save_stability(self, record: FactorStabilityRecord) -> None:
        """Save a stability diagnostics record."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO factor_stability_history
                (factor_name, ic_mean, ic_std, icir, sharpe_mean, sharpe_std,
                 sharpe_consistency, positive_folds, total_folds, coverage,
                 run_id, data_source, adjust, pit, validity, warnings, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.factor_name,
                record.ic_mean, record.ic_std, record.icir,
                record.sharpe_mean, record.sharpe_std,
                record.sharpe_consistency, record.positive_folds,
                record.total_folds, record.coverage,
                record.run_id, record.data_source, record.adjust,
                None if record.pit is None else int(bool(record.pit)),
                record.validity, json.dumps(list(record.warnings)),
                record.timestamp,
            ))

    # ── Regime History ──

    def save_regime_diagnostics(
        self,
        factor_name: str,
        regime_ics: dict[str, float],
        regime_samples: dict[str, int],
        run_id: str = "",
        data_source: str = "",
        adjust: str = "",
        pit: bool | None = None,
        validity: str = "valid",
        warnings: list[str] | None = None,
    ) -> None:
        """Save factor performance by market regime."""
        now = _utcnow_iso()
        warnings_json = json.dumps(list(warnings or []))
        pit_val = None if pit is None else int(bool(pit))
        with self._lock, self._conn() as conn:
            for regime, ic in regime_ics.items():
                n = regime_samples.get(regime, 0)
                conn.execute("""
                    INSERT INTO factor_regime_history
                    (factor_name, regime, rank_ic, n_samples, run_id,
                     data_source, adjust, pit, validity, warnings, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (factor_name, regime, ic, n, run_id,
                      data_source, adjust, pit_val, validity,
                      warnings_json, now))

    def get_regime_diagnostics(self, factor_name: str) -> list[dict]:
        """Get regime diagnostics for a factor."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM factor_regime_history
                WHERE factor_name = ?
                ORDER BY id DESC LIMIT 50
            """, (factor_name,)).fetchall()
            return [dict(r) for r in rows]

    # ── Factor Ranking ──

    def get_factor_ranking(self) -> list[dict]:
        """Rank factors by combined health score.

        Health score blends: ICIR, coverage, stability, consistency.
        """
        with self._conn() as conn:
            # Latest row per factor joined on the monotonic id, not on the
            # timestamp text: legacy rows carry naive local timestamps while
            # new rows carry UTC offsets, so text ordering across the two
            # formats is not chronological. Invalidated/superseded rows are
            # excluded so a marked-wrong result cannot drive the ranking.
            rows = conn.execute("""
                SELECT
                    s.factor_name,
                    s.ic_mean,
                    s.ic_std,
                    s.icir,
                    s.sharpe_mean,
                    s.sharpe_std,
                    s.sharpe_consistency,
                    s.coverage,
                    s.timestamp as last_updated
                FROM factor_stability_history s
                INNER JOIN (
                    SELECT factor_name, MAX(id) as max_id
                    FROM factor_stability_history
                    WHERE COALESCE(validity, 'valid') = 'valid'
                    GROUP BY factor_name
                ) latest ON s.factor_name = latest.factor_name
                    AND s.id = latest.max_id
                ORDER BY s.icir DESC
            """).fetchall()

            ranked = []
            for r in rows:
                d = dict(r)
                # Sign-aware ICIR score. abs(ICIR) promoted a factor with
                # ICIR=-0.5 to 0.91 health while a correct-sign ICIR=+0.1
                # scored 0.59: wrong-direction factors ranked first.
                icir_score = max(0.0, min(1.0, (d["icir"] or 0.0) / 0.5))
                coverage_score = d["coverage"] or 0
                consistency = d["sharpe_consistency"]
                if consistency is None:
                    # Stability was not computed for the latest run (no
                    # walk-forward folds). Scoring it as a measured zero made
                    # 30% of the health score structurally zero for factors
                    # that were simply never validated; renormalize the
                    # measured terms and surface the missing status instead.
                    health = (icir_score * 0.4 + coverage_score * 0.3) / 0.7
                    consistency_out = None
                    stability_status = "unavailable"
                else:
                    health = (icir_score * 0.4 + coverage_score * 0.3
                              + float(consistency) * 0.3)
                    consistency_out = round(float(consistency), 4)
                    stability_status = "measured"
                ranked.append({
                    "factor_name": d["factor_name"],
                    "ic_mean": round(d["ic_mean"], 6) if d["ic_mean"] else 0,
                    "icir": round(d["icir"], 4) if d["icir"] else 0,
                    "coverage": round(d["coverage"], 4) if d["coverage"] else 0,
                    "consistency": consistency_out,
                    "stability_status": stability_status,
                    "health_score": round(health, 4),
                })

            ranked.sort(key=lambda x: x["health_score"], reverse=True)
            return ranked

    # ── Version Management ──

    def save_version(
        self,
        factor_name: str,
        version: str,
        description: str = "",
        params: dict | None = None,
    ) -> None:
        """Record a factor version."""
        now = _utcnow_iso()
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO factor_versions
                (factor_name, version, description, params, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                factor_name, version, description,
                json.dumps(params or {}), now,
            ))

    # ── Stats ──

    def get_stats(self) -> dict:
        """Get store statistics."""
        with self._conn() as conn:
            tables = [
                "factor_definitions", "factor_values",
                "factor_evaluation_history", "factor_backtest_history",
                "factor_walk_forward_history", "factor_stability_history",
                "factor_regime_history", "factor_versions",
            ]
            stats = {}
            for table in tables:
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                    stats[table] = count
                except Exception:
                    stats[table] = 0
            # Extra: distinct factors
            stats["distinct_factors"] = conn.execute(
                "SELECT COUNT(DISTINCT factor_name) FROM factor_definitions"
            ).fetchone()[0]
            stats["db_path"] = self._db_path
            return stats

    def clear(self):
        """Clear all data (for testing)."""
        with self._lock, self._conn() as conn:
            tables = [
                "factor_definitions", "factor_values",
                "factor_evaluation_history", "factor_backtest_history",
                "factor_walk_forward_history", "factor_stability_history",
                "factor_regime_history", "factor_versions",
            ]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        logger.warning("FactorResearchStore cleared")
