"""SQLite persistence layer — every piece of state is stored.

Tables:
    orders      — full order lifecycle with timestamps
    positions   — current positions with cost basis
    trades      — executed fills (linked to orders)
    pnl_history — time-series P&L snapshots
    signals     — alpha signal history for analysis
    sessions    — trading session records
    events      — event log for audit trail
    config      — strategy configuration snapshots
    eod_reconciliation — once-per-day marker for the end-of-day run

Usage:
    store = Store('data/trading.db')
    store.save_order(order)
    store.save_trade(trade)
    store.get_positions()
    store.get_pnl_history(days=30)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

#: Time basis for new rows in ``pnl_history``/``sessions``/``events``
#: (S17): Asia/Shanghai wall clock, stored naive. The scheduler has always
#: used this basis; the engine and Store defaults used the host clock, so a
#: UTC container wrote rows eight hours off and a UTC-5 host sixteen. Legacy
#: rows are NOT migrated (see the ops2 report): only rows written from this
#: version on use the single documented basis.
EXCHANGE_TZ = ZoneInfo("Asia/Shanghai")


def exchange_now() -> datetime:
    """Current exchange-local (Asia/Shanghai) wall clock, naive.

    Derived from the epoch clock, never from the host time zone, so changing
    ``TZ`` or a frozen host clock cannot move the stored timestamp domain.
    """
    return datetime.fromtimestamp(time.time(), EXCHANGE_TZ).replace(tzinfo=None)


def exchange_now_iso() -> str:
    """ISO timestamp in the documented store basis (Shanghai naive)."""
    return exchange_now().isoformat()


def _clamp_limit(limit, default: int = 100, maximum: int = 1000) -> int:
    """Bound a caller-supplied row limit.

    SQLite treats LIMIT -1 as "no limit", so a negative query parameter could
    dump a whole table; huge values overflow the int64 bind.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0, min(value, maximum))


def _clamp_days(days, default: int = 30, maximum: int = 3650) -> int:
    try:
        value = int(days)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(value, maximum))


@dataclass(frozen=True)
class RetentionPolicy:
    """Per-table maximum age for the retention job (S10).

    A value of ``0`` (or negative) disables retention for that table. Defaults
    follow the task book: events 180d, signals/sessions 1y, orders/trades/P&L
    5y, NAV 10y.
    """

    orders_days: int = 5 * 365
    trades_days: int = 5 * 365
    pnl_days: int = 5 * 365
    signals_days: int = 365
    sessions_days: int = 365
    events_days: int = 180
    nav_days: int = 10 * 365

    @classmethod
    def from_env(cls) -> RetentionPolicy:
        """Build a policy from ``QUANT_RETENTION_*_DAYS`` overrides.

        Unknown/unparsable values fall back to the defaults rather than
        crashing a daemon at startup.
        """
        import os

        defaults = cls()
        fields = {
            "orders_days": "QUANT_RETENTION_ORDERS_DAYS",
            "trades_days": "QUANT_RETENTION_TRADES_DAYS",
            "pnl_days": "QUANT_RETENTION_PNL_DAYS",
            "signals_days": "QUANT_RETENTION_SIGNALS_DAYS",
            "sessions_days": "QUANT_RETENTION_SESSIONS_DAYS",
            "events_days": "QUANT_RETENTION_EVENTS_DAYS",
            "nav_days": "QUANT_RETENTION_NAV_DAYS",
        }
        values = {}
        for field, env in fields.items():
            raw = os.environ.get(env)
            if raw is None or raw == "":
                values[field] = getattr(defaults, field)
                continue
            try:
                values[field] = int(raw)
            except ValueError:
                logger.warning("Ignoring invalid %s=%r; using default", env, raw)
                values[field] = getattr(defaults, field)
        return cls(**values)

    def is_enabled(self) -> bool:
        return any(
            getattr(self, field) > 0
            for field in (
                "orders_days", "trades_days", "pnl_days", "signals_days",
                "sessions_days", "events_days", "nav_days",
            )
        )


class Store:
    """Thread-safe SQLite store for all trading state.

    Uses WAL mode for concurrent read/write.
    All writes are serialized through a lock.
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # One long-lived connection per thread. Opening a fresh SQLite
        # connection, re-issuing the WAL pragmas and closing it on every read
        # and write cost ~2.8 ms per operation -- ~95x the cost of the insert
        # itself -- and a live cycle performs hundreds of them.
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @contextmanager
    def _conn(self):
        """Get this thread's database connection and commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        """Close this thread's connection (call before deleting the DB file).

        The WAL is checkpointed first so the committed rows live in the main
        database file after shutdown (S16). Checkpoint failure must not stop
        the connection from closing.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                logger.warning("WAL checkpoint on close failed: %s", exc)
            conn.close()
            self._local.conn = None

    def checkpoint(self) -> None:
        """Flush the WAL into the main database file (TRUNCATE)."""
        with self._conn() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def vacuum(self) -> None:
        """Rebuild the database file to return freed pages to the OS.

        VACUUM cannot run inside a transaction; every write commits through
        ``_conn`` so the connection is idle here. Failures are logged and
        reported (a locked database must not abort a retention run).
        """
        conn = self._get_connection()
        conn.execute("VACUUM")

    def size_bytes(self) -> int:
        """Size of the main database file on disk, in bytes."""
        try:
            return Path(self._db_path).stat().st_size
        except OSError:
            return 0

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    tenant_id TEXT DEFAULT 'default',
                    code TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT DEFAULT 'limit',
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    filled_quantity INTEGER DEFAULT 0,
                    filled_price REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    commission REAL DEFAULT 0,
                    tax REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    broker_order_id TEXT DEFAULT '',
                    error_msg TEXT DEFAULT '',
                    strategy_id TEXT DEFAULT '',
                    signal_id TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    code TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    name TEXT DEFAULT '',
                    quantity INTEGER NOT NULL,
                    available INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    -- Two tenants may hold the same ticker. The key used to be
                    -- `code` alone (with the tenant_id column present and
                    -- indexed but not part of the key), so the second tenant's
                    -- write replaced the first one's row.
                    PRIMARY KEY (tenant_id, code)
                );

                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    tenant_id TEXT DEFAULT 'default',
                    order_id TEXT NOT NULL,
                    code TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    tax REAL DEFAULT 0,
                    executed_at TEXT NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                );

                CREATE TABLE IF NOT EXISTS pnl_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    daily_pnl_pct REAL DEFAULT 0,
                    cumulative_pnl REAL DEFAULT 0,
                    n_positions INTEGER DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    tenant_id TEXT DEFAULT 'default',
                    code TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strength REAL NOT NULL,
                    factor_values TEXT DEFAULT '{}',
                    strategy_id TEXT DEFAULT '',
                    generated_at TEXT NOT NULL,
                    consumed INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    strategy_id TEXT DEFAULT '',
                    broker TEXT DEFAULT 'simulated',
                    status TEXT DEFAULT 'active',
                    started_at TEXT NOT NULL,
                    stopped_at TEXT DEFAULT '',
                    total_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    config TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    tenant_id TEXT DEFAULT 'default',
                    topic TEXT NOT NULL,
                    data TEXT DEFAULT '{}',
                    source TEXT DEFAULT '',
                    timestamp REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS config_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nav_history (
                    date TEXT PRIMARY KEY,
                    nav_total REAL NOT NULL,
                    nav_per_unit REAL NOT NULL,
                    total_units REAL NOT NULL,
                    cash REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    mgmt_fee REAL DEFAULT 0,
                    perf_fee REAL DEFAULT 0,
                    high_water_mark REAL DEFAULT 1.0,
                    daily_return REAL DEFAULT 0,
                    cumulative_return REAL DEFAULT 0,
                    accrued_mgmt_fee REAL DEFAULT 0,
                    accrued_perf_fee REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS engine_state (
                    strategy_id TEXT PRIMARY KEY,
                    state_json  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                -- One marker per completed end-of-day reconciliation.
                -- CREATE TABLE IF NOT EXISTS doubles as the migration for
                -- databases written before this table existed.
                CREATE TABLE IF NOT EXISTS eod_reconciliation (
                    date TEXT PRIMARY KEY,
                    timestamp TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_orders_code ON orders(code);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);
                CREATE INDEX IF NOT EXISTS idx_trades_order ON trades(order_id);
                CREATE INDEX IF NOT EXISTS idx_pnl_ts ON pnl_history(timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_code ON signals(code);
                CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
            """)

            # Migrate existing tables: add tenant_id if missing
            self._migrate_tenant_id(conn)
            # ... and give positions a tenant-scoped key
            self._migrate_positions_primary_key(conn)
            # ... and carry cumulative fee accruals on existing NAV databases
            nav_cols = {r["name"] for r in conn.execute("PRAGMA table_info(nav_history)")}
            if "accrued_mgmt_fee" not in nav_cols:
                conn.execute(
                    "ALTER TABLE nav_history ADD COLUMN accrued_mgmt_fee REAL DEFAULT 0"
                )
            if "accrued_perf_fee" not in nav_cols:
                conn.execute(
                    "ALTER TABLE nav_history ADD COLUMN accrued_perf_fee REAL DEFAULT 0"
                )

            # Create tenant_id indexes after migration
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_orders_tenant ON orders(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_trades_tenant ON trades(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_positions_tenant ON positions(tenant_id);

                -- Query-plan indexes for the default history queries.
                --
                -- These covered growing tables (trades/events/signals/
                -- sessions) whose filter+sort columns had no index: every
                -- dashboard poll did a full scan and a temp B-tree sort.
                -- They are created *after* the tenant migration so the
                -- composite order index works on databases written before
                -- multi-tenancy, and `IF NOT EXISTS` keeps startup idempotent
                -- for existing databases.
                CREATE INDEX IF NOT EXISTS idx_trades_code_executed
                    ON trades(code, executed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trades_executed
                    ON trades(executed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_topic_timestamp
                    ON events(topic, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_signals_consumed_generated
                    ON signals(consumed, generated_at DESC);
                -- The unfiltered signal query (consumed=-1) still sorts by
                -- generated_at, so it needs a leading-column index of its own.
                CREATE INDEX IF NOT EXISTS idx_signals_generated
                    ON signals(generated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_started
                    ON sessions(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_orders_tenant_status_created
                    ON orders(tenant_id, status, created_at DESC);
            """)
        logger.info("Store initialized: %s", self._db_path)

    def _migrate_tenant_id(self, conn: sqlite3.Connection) -> None:
        """Add tenant_id column to existing tables if missing."""
        for table in ("orders", "trades", "positions", "signals", "events"):
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "tenant_id" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT DEFAULT 'default'")
                logger.info("Migrated %s: added tenant_id", table)

    def _migrate_positions_primary_key(self, conn: sqlite3.Connection) -> None:
        """Rebuild `positions` so its key is (tenant_id, code).

        SQLite cannot alter a primary key, so the table is recreated and the
        rows copied. Databases written before this have `code` alone as the
        key, which means a ticker could exist for only one tenant: the second
        tenant's `INSERT OR REPLACE` matched on `code` and replaced the first
        tenant's row outright.
        """
        info = conn.execute("PRAGMA table_info(positions)").fetchall()
        if not info:
            return
        if [row[1] for row in info if row[5]] == ["tenant_id", "code"]:
            return

        # Rebuild the column list from what is actually there, so this keeps
        # working if columns are added later.
        defs = []
        for row in info:
            name, ctype, notnull, dflt = row[1], row[2], row[3], row[4]
            if name in ("tenant_id", "code"):
                continue
            col = f'"{name}" {ctype}'
            if notnull:
                col += " NOT NULL"
            if dflt is not None:
                col += f" DEFAULT {dflt}"
            defs.append(col)

        cols = ", ".join(f'"{row[1]}"' for row in info)
        conn.executescript(f"""
            CREATE TABLE positions_new (
                "tenant_id" TEXT NOT NULL DEFAULT 'default',
                "code" TEXT NOT NULL,
                {", ".join(defs)},
                PRIMARY KEY (tenant_id, code)
            );
            INSERT OR REPLACE INTO positions_new ({cols}) SELECT {cols} FROM positions;
            DROP TABLE positions;
            ALTER TABLE positions_new RENAME TO positions;
        """)
        n = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        logger.info("Migrated positions: primary key is now (tenant_id, code), %d rows kept", n)

    # ── Orders ──

    #: Columns of `orders`, in schema order. `save_order` writes only the
    #: columns the caller actually supplies.
    _ORDER_FIELDS = (
        "order_id", "tenant_id", "code", "side", "order_type", "quantity",
        "price", "filled_quantity", "filled_price", "status", "commission",
        "tax", "slippage", "broker_order_id", "error_msg", "strategy_id",
        "signal_id", "created_at", "updated_at",
    )
    #: Columns the schema declares NOT NULL without a SQL default.
    _ORDER_REQUIRED = ("order_id", "code", "side", "quantity", "price")

    def save_order(self, order: dict):
        """Insert an order, or update only the fields the caller supplied.

        The previous `INSERT OR REPLACE` deleted the row and rebuilt it from
        the incoming dict, so a later partial save -- e.g. the fill DTO a
        broker returns, which carries neither `strategy_id` nor `signal_id`
        -- reset every omitted column to its default. This upsert touches
        only the caller's columns; on first insert the SQL defaults fill the
        rest (created_at/updated_at have none, so they get a timestamp).
        """
        with self._lock, self._conn() as conn:
            self._save_order(conn, order)

    def _save_order(self, conn: sqlite3.Connection, order: dict) -> None:
        values = {k: order[k] for k in self._ORDER_FIELDS if k in order}
        for key in self._ORDER_REQUIRED:
            if key not in values:
                raise KeyError(key)

        insert = dict(values)
        now = datetime.now().isoformat()
        insert.setdefault("tenant_id", "default")
        insert.setdefault("created_at", now)
        insert.setdefault("updated_at", now)

        insert_cols = [c for c in self._ORDER_FIELDS if c in insert]
        # Only caller-supplied columns are overwritten; tenant_id keeps the
        # row's existing value unless the caller names one.
        update_cols = [c for c in values if c != "order_id"]
        sql = (
            f"INSERT INTO orders ({', '.join(insert_cols)}) "
            f"VALUES ({', '.join('?' for _ in insert_cols)}) "
            "ON CONFLICT(order_id) DO UPDATE SET "
            + ", ".join(f"{c} = excluded.{c}" for c in update_cols)
        )
        conn.execute(sql, tuple(insert[c] for c in insert_cols))

    def get_orders(self, status: str = "", code: str = "", limit: int = 100,
                   tenant_id: str = "",
                   start_date: str = "", end_date: str = "") -> list[dict]:
        """Get orders with optional filters.

        `start_date`/`end_date` are ISO dates (YYYY-MM-DD) applied in SQL.
        Compliance exports used to fetch the newest `limit` rows and filter
        afterwards, so a requested historical range could come back empty
        once the book had more rows than the limit.
        """
        limit = _clamp_limit(limit)
        with self._conn() as conn:
            query = "SELECT * FROM orders WHERE 1=1"
            params: list = []
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if status:
                query += " AND status = ?"
                params.append(status)
            if code:
                query += " AND code = ?"
                params.append(code)
            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date)
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date + "T23:59:59")
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    #: Batch size for compliance export pagination.
    _EXPORT_BATCH = 1000

    @staticmethod
    def _widen_export_window(start_date: str, end_date: str) -> tuple[str, str]:
        """Widen the SQL date window by a day on each side.

        The exporter's Python filter remains authoritative (it normalizes
        timezone-aware timestamps to local wall-clock time). A raw SQL bound
        could otherwise drop a +08:00 print on the end date before Python
        ever saw it.
        """
        start, end = start_date, end_date
        if start_date:
            try:
                start = (
                    datetime.fromisoformat(start_date) - timedelta(days=1)
                ).strftime("%Y-%m-%d")
            except ValueError:
                start = start_date
        if end_date:
            try:
                end = (
                    datetime.fromisoformat(end_date) + timedelta(days=1)
                ).strftime("%Y-%m-%d")
            except ValueError:
                end = end_date
        return start, end

    def get_orders_for_export(
        self, tenant_id: str = "", start_date: str = "", end_date: str = "",
    ) -> list[dict]:
        """Every order in the (widened) range, pages internally.

        Compliance exports need the complete record set. `get_orders` clamps
        to 1000 rows for API safety, so an export over a larger range silently
        lost older rows. This dedicated query has no API clamp and pages with
        a keyset cursor instead of OFFSET.
        """
        start, end = self._widen_export_window(start_date, end_date)
        collected: list[dict] = []
        last: tuple[str, str] | None = None
        while True:
            query = "SELECT * FROM orders WHERE 1=1"
            params: list = []
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if start:
                query += " AND created_at >= ?"
                params.append(start)
            if end:
                query += " AND created_at <= ?"
                params.append(end + "T23:59:59")
            if last is not None:
                query += (
                    " AND (created_at < ? OR "
                    "(created_at = ? AND order_id < ?))"
                )
                params.extend([last[0], last[0], last[1]])
            query += " ORDER BY created_at DESC, order_id DESC LIMIT ?"
            params.append(self._EXPORT_BATCH)
            with self._conn() as conn:
                batch = [dict(r) for r in conn.execute(query, params).fetchall()]
            collected.extend(batch)
            if len(batch) < self._EXPORT_BATCH:
                return collected
            last = (batch[-1]["created_at"], batch[-1]["order_id"])

    # ── Positions ──

    _POSITION_INSERT = """
        INSERT OR REPLACE INTO positions
        (code, tenant_id, name, quantity, available, avg_cost, current_price,
         market_value, unrealized_pnl, realized_pnl, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    @staticmethod
    def _position_values(pos: dict, tenant_id: str, now: str) -> tuple:
        code = pos["code"]
        quantity = pos["quantity"]
        avg_cost = pos["avg_cost"]
        return (
            code, pos.get("tenant_id", tenant_id),
            pos.get("name", ""), quantity,
            pos.get("available", quantity), avg_cost,
            pos.get("current_price", 0), pos.get("market_value", 0),
            pos.get("unrealized_pnl", 0), pos.get("realized_pnl", 0),
            now,
        )

    def save_position(self, pos: dict):
        """Save or update a position."""
        with self._lock, self._conn() as conn:
            conn.execute(
                self._POSITION_INSERT,
                self._position_values(pos, "default", datetime.now().isoformat()),
            )

    def save_positions(
        self,
        positions: list[dict],
        tenant_id: str = "",
        prune_missing: bool = False,
    ) -> None:
        """Upsert a whole position book in one transaction.

        The live engine used to call `save_position` once per position inside
        a loop -- each call took the write lock and committed separately, so
        a 50-position book cost tens of commits *per persist* (and it is
        persisted at least twice per cycle). This writes the book with one
        `executemany` and, when `prune_missing` is set, deletes the tenant's
        rows that are not in the book in the same transaction. Only call it
        with a *known* book: pruning from an unknown/empty query is what
        erased persisted positions before (D6).

        `tenant_id` (when given) is the prune scope and the default tenant for
        rows that do not carry one; a row's own `tenant_id` still wins.
        """
        tenant = tenant_id or "default"
        now = datetime.now().isoformat()
        rows = [self._position_values(pos, tenant, now) for pos in positions or []]
        with self._lock, self._conn() as conn:
            if rows:
                conn.executemany(self._POSITION_INSERT, rows)
            if prune_missing and tenant_id:
                codes = [row[0] for row in rows]
                if codes:
                    placeholders = ", ".join("?" for _ in codes)
                    conn.execute(
                        "DELETE FROM positions WHERE tenant_id = ? "
                        f"AND code NOT IN ({placeholders})",
                        [tenant_id, *codes],
                    )
                else:
                    conn.execute(
                        "DELETE FROM positions WHERE tenant_id = ?",
                        (tenant_id,),
                    )

    def delete_position(self, code: str, tenant_id: str = ""):
        """Remove a position."""
        with self._lock, self._conn() as conn:
            if tenant_id:
                conn.execute(
                    "DELETE FROM positions WHERE code = ? AND tenant_id = ?",
                    (code, tenant_id),
                )
            else:
                conn.execute("DELETE FROM positions WHERE code = ?", (code,))

    def get_positions(self, tenant_id: str = "") -> list[dict]:
        """Get current positions.

        With `tenant_id`, only that tenant's rows. With the default empty
        string, **every tenant's rows**, which is the behaviour callers
        currently get: `core/scheduler.py:208`, `operations/nav.py:207` and
        `api/monitor.py` read positions without naming a tenant, and neither
        `TradingScheduler` nor `NAVCalculator` carries one. Those callers
        therefore see a merged book once more than one tenant is trading;
        giving them a tenant is a separate change from the key fix above.
        """
        with self._conn() as conn:
            if tenant_id:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE quantity > 0 AND tenant_id = ?",
                    (tenant_id,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE quantity > 0").fetchall()
            return [dict(r) for r in rows]

    # ── Trades ──

    def save_trade(self, trade: dict):
        """Record an executed trade.

        `trade_id` defaults to a fresh uuid4, not to `order_id`: an order can
        fill in several prints, and the old default made the second one hit
        the `trades` UNIQUE constraint.
        """
        with self._lock, self._conn() as conn:
            self._save_trade(conn, trade)

    def _save_trade(self, conn: sqlite3.Connection, trade: dict) -> None:
        conn.execute("""
                INSERT INTO trades
                (trade_id, tenant_id, order_id, code, side, quantity, price,
                 commission, tax, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            trade.get('trade_id') or uuid.uuid4().hex,
            trade.get('tenant_id', 'default'),
            trade['order_id'], trade['code'], trade['side'],
            trade['quantity'], trade['price'],
            trade.get('commission', 0), trade.get('tax', 0),
            trade.get('executed_at', datetime.now().isoformat()),
        ))

    def save_order_with_trade(self, order: dict, trade: dict):
        """Write an order and its fill atomically (one transaction).

        The two used to be saved in separate transactions, so a failing
        trade insert left the order row already updated and its fill -- and
        audit record -- missing. Sharing one `_conn()` block makes the pair
        all-or-nothing. `trade` may omit `trade_id`; a uuid4 is generated.
        """
        with self._lock, self._conn() as conn:
            self._save_order(conn, order)
            self._save_trade(conn, trade)

    def get_trades(self, code: str = "", limit: int = 100,
                   tenant_id: str = "",
                   start_date: str = "", end_date: str = "") -> list[dict]:
        """Get trade history.

        `start_date`/`end_date` are ISO dates applied in SQL so a historical
        export is not limited to the newest rows.
        """
        limit = _clamp_limit(limit)
        with self._conn() as conn:
            query = "SELECT * FROM trades WHERE 1=1"
            params: list = []
            if code:
                query += " AND code = ?"
                params.append(code)
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if start_date:
                query += " AND executed_at >= ?"
                params.append(start_date)
            if end_date:
                query += " AND executed_at <= ?"
                params.append(end_date + "T23:59:59")
            query += " ORDER BY executed_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_trades_for_export(
        self, tenant_id: str = "", start_date: str = "", end_date: str = "",
    ) -> list[dict]:
        """Every trade in the (widened) range, pages internally.

        See `get_orders_for_export`: compliance exports must not be truncated
        by the 1000-row API clamp.
        """
        start, end = self._widen_export_window(start_date, end_date)
        collected: list[dict] = []
        last: tuple[str, str] | None = None
        while True:
            query = "SELECT * FROM trades WHERE 1=1"
            params: list = []
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if start:
                query += " AND executed_at >= ?"
                params.append(start)
            if end:
                query += " AND executed_at <= ?"
                params.append(end + "T23:59:59")
            if last is not None:
                query += (
                    " AND (executed_at < ? OR "
                    "(executed_at = ? AND trade_id < ?))"
                )
                params.extend([last[0], last[0], last[1]])
            query += " ORDER BY executed_at DESC, trade_id DESC LIMIT ?"
            params.append(self._EXPORT_BATCH)
            with self._conn() as conn:
                batch = [dict(r) for r in conn.execute(query, params).fetchall()]
            collected.extend(batch)
            if len(batch) < self._EXPORT_BATCH:
                return collected
            last = (batch[-1]["executed_at"], batch[-1]["trade_id"])

    # ── P&L History ──

    def save_pnl_snapshot(self, snapshot: dict):
        """Save a P&L snapshot."""
        with self._lock, self._conn() as conn:
            self._save_pnl(conn, snapshot)

    def _save_pnl(self, conn: sqlite3.Connection, snapshot: dict) -> None:
        conn.execute("""
                INSERT INTO pnl_history
                (timestamp, total_equity, cash, market_value, daily_pnl,
                 daily_pnl_pct, cumulative_pnl, n_positions, max_drawdown, sharpe_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            snapshot.get('timestamp', exchange_now_iso()),
            snapshot['total_equity'], snapshot['cash'],
            snapshot['market_value'], snapshot.get('daily_pnl', 0),
            snapshot.get('daily_pnl_pct', 0), snapshot.get('cumulative_pnl', 0),
            snapshot.get('n_positions', 0), snapshot.get('max_drawdown', 0),
            snapshot.get('sharpe_ratio', 0),
        ))

    def save_pnl_and_mark_eod(
        self, snapshot: dict, date: str, timestamp: str | None = None,
    ) -> bool:
        """Write the EOD snapshot and its marker in one transaction.

        The two writes were separate transactions, so a crash in between left
        a snapshot with no marker and the next run appended a second snapshot
        for the same day. One transaction makes the pair all-or-nothing. The
        marker is inserted with ``OR IGNORE`` first and the snapshot is only
        written when the insert actually claimed the day, so two racing
        processes also produce exactly one snapshot.

        Returns True when this call reconciled the day, False when another
        run had already claimed it.
        """
        with self._lock, self._conn() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO eod_reconciliation (date, timestamp) "
                "VALUES (?, ?)",
                (date, timestamp or exchange_now_iso()),
            )
            if cursor.rowcount == 0:
                return False
            self._save_pnl(conn, snapshot)
            return True

    def get_pnl_history(self, days: int = 30,
                        start_date: str = "", end_date: str = "") -> list[dict]:
        """Get P&L history.

        `days` is the default window. Explicit `start_date`/`end_date`
        (YYYY-MM-DD) override it, so a compliance export can request a
        period older than the default window.
        """
        days = _clamp_days(days)
        with self._conn() as conn:
            query = "SELECT * FROM pnl_history WHERE 1=1"
            params: list = []
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            else:
                cutoff = (exchange_now() - timedelta(days=days)).isoformat()
                query += " AND timestamp > ?"
                params.append(cutoff)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date + "T23:59:59")
            query += " ORDER BY timestamp"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── EOD Reconciliation ──

    def has_eod_reconciliation(self, date: str) -> bool:
        """Whether the end-of-day reconciliation ran for `date` (YYYY-MM-DD)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM eod_reconciliation WHERE date = ?", (date,)
            ).fetchone()
            return row is not None

    def mark_eod_reconciliation(self, date: str, timestamp: str | None = None):
        """Record that the EOD reconciliation for `date` has completed.

        The marker lives in SQLite so it survives a process restart; without
        it a scheduler restarted after the close appended a second P&L
        snapshot for the same day.
        """
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eod_reconciliation (date, timestamp) "
                "VALUES (?, ?)",
                (date, timestamp or exchange_now_iso()),
            )

    def get_last_eod_reconciliation(self) -> str:
        """Latest exchange date (YYYY-MM-DD) with an EOD marker, or ''.

        Used by the scheduler's startup catch-up scan (S3): a stand-alone
        scheduler that was down for a few days compares this against the
        exchange calendar and reconciles the gap.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(date) AS last_date FROM eod_reconciliation"
            ).fetchone()
        return str(row["last_date"] or "") if row else ""

    def get_last_pnl_timestamp(self) -> str:
        """Timestamp of the newest P&L snapshot, or '' when there is none."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) AS last_ts FROM pnl_history"
            ).fetchone()
        return str(row["last_ts"] or "") if row else ""

    # ── Signals ──

    def save_signal(self, signal: dict):
        """Save an alpha signal."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO signals
                    (signal_id, tenant_id, code, direction, strength, factor_values,
                     strategy_id, generated_at, consumed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                signal['signal_id'], signal.get('tenant_id', 'default'),
                signal['code'], signal['direction'],
                signal['strength'], json.dumps(signal.get('factor_values', {})),
                signal.get('strategy_id', ''),
                signal.get('generated_at', datetime.now().isoformat()),
                signal.get('consumed', 0),
            ))

    def get_signals(self, consumed: int = -1, limit: int = 100,
                    tenant_id: str = "") -> list[dict]:
        """Get signals. consumed=-1 for all, 0 for unconsumed, 1 for consumed."""
        limit = _clamp_limit(limit)
        with self._conn() as conn:
            query = "SELECT * FROM signals WHERE 1=1"
            params: list = []
            if consumed >= 0:
                query += " AND consumed = ?"
                params.append(consumed)
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            query += " ORDER BY generated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if 'factor_values' in d and isinstance(d['factor_values'], str):
                    try:
                        d['factor_values'] = json.loads(d['factor_values'])
                    except json.JSONDecodeError:
                        d['factor_values'] = {}
                result.append(d)
            return result

    # ── Sessions ──

    #: Session columns written from a caller's dict, in INSERT order.
    _SESSION_FIELDS = (
        "strategy_id", "broker", "status", "started_at",
        "stopped_at", "total_trades", "total_pnl", "config",
    )

    def save_session(self, session: dict):
        """Insert or update a trading session record.

        A session is written twice under the same id -- once by `start()`, once
        by `stop()` -- and the second call carries only the fields it knows
        about. `INSERT OR REPLACE` deleted the row and reinserted it from that
        second dict, so the defaults for the absent keys silently discarded the
        start record: `started_at` became the stop time, `broker` fell back to
        'simulated', `total_trades` to 0 and `config` to '{}'. A stopped
        session then read as one that began when it ended.

        Fields the caller supplies still overwrite; fields it omits are kept.
        """
        values = {
            "strategy_id": session.get("strategy_id", ""),
            "broker": session.get("broker", "simulated"),
            "status": session.get("status", "active"),
            "started_at": session.get("started_at", exchange_now_iso()),
            "stopped_at": session.get("stopped_at", ""),
            "total_trades": session.get("total_trades", 0),
            "total_pnl": session.get("total_pnl", 0),
            "config": json.dumps(session.get("config", {})),
        }
        session_id = session["session_id"]

        with self._lock, self._conn() as conn:
            existing = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

            if existing is None:
                conn.execute(
                    "INSERT INTO sessions (session_id, strategy_id, broker, status, "
                    "started_at, stopped_at, total_trades, total_pnl, config) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, *(values[f] for f in self._SESSION_FIELDS)),
                )
                return

            merged = {
                f: (values[f] if f in session else existing[f])
                for f in self._SESSION_FIELDS
            }
            conn.execute(
                "UPDATE sessions SET strategy_id = ?, broker = ?, status = ?, "
                "started_at = ?, stopped_at = ?, total_trades = ?, total_pnl = ?, "
                "config = ? WHERE session_id = ?",
                (*(merged[f] for f in self._SESSION_FIELDS), session_id),
            )

    def get_sessions(self, limit: int = 20) -> list[dict]:
        """Get trading session history."""
        # SQLite treats LIMIT -1 as unlimited; the API used to forward -1 raw.
        limit = _clamp_limit(limit, default=20, maximum=1000)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── Events (Audit Trail) ──

    def log_event(self, event: dict):
        """Log an event for audit trail."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT INTO events (event_id, tenant_id, topic, data, source, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                event.get('event_id', ''), event.get('tenant_id', 'default'),
                event.get('topic', ''),
                json.dumps(event.get('data', {})),
                event.get('source', ''), event.get('timestamp', time.time()),
                exchange_now_iso(),
            ))

    def get_events(self, topic: str = "", limit: int = 100,
                   tenant_id: str = "") -> list[dict]:
        """Get event log, optionally filtered by tenant."""
        limit = _clamp_limit(limit)
        with self._conn() as conn:
            query = "SELECT * FROM events WHERE 1=1"
            params: list = []
            if topic:
                query += " AND topic = ?"
                params.append(topic)
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if 'data' in d and isinstance(d['data'], str):
                    try:
                        d['data'] = json.loads(d['data'])
                    except json.JSONDecodeError:
                        d['data'] = {}
                result.append(d)
            return result

    # ── Config Snapshots ──

    def save_config_snapshot(self, config: dict, description: str = ""):
        """Save a configuration snapshot."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT INTO config_snapshots (config, description, created_at)
                    VALUES (?, ?, ?)
                """, (json.dumps(config), description, datetime.now().isoformat()))

    def get_config_snapshots(self, limit: int = 10) -> list[dict]:
        """Get config snapshot history."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM config_snapshots ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── NAV History ──

    def save_nav(self, nav: dict):
        """Save a NAV record."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO nav_history
                    (date, nav_total, nav_per_unit, total_units, cash, market_value,
                     mgmt_fee, perf_fee, high_water_mark, daily_return, cumulative_return,
                     accrued_mgmt_fee, accrued_perf_fee)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                nav['date'], nav['nav_total'], nav['nav_per_unit'],
                nav['total_units'], nav.get('cash', 0),
                nav.get('market_value', 0), nav.get('mgmt_fee', 0),
                nav.get('perf_fee', 0), nav.get('high_water_mark', 1.0),
                nav.get('daily_return', 0), nav.get('cumulative_return', 0),
                nav.get('accrued_mgmt_fee', 0), nav.get('accrued_perf_fee', 0),
            ))

    def get_nav_history(self, days: int = 365 * 3) -> list[dict]:
        days = _clamp_days(days, default=365 * 3)
        """Get NAV history for the last N days."""
        with self._conn() as conn:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT * FROM nav_history WHERE date > ? ORDER BY date",
                (cutoff,)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_nav(self) -> dict | None:
        """Return the most recent NAV record, or None if there is none.

        Distinct from get_nav_history(): that one takes a *window ending now*,
        so it cannot answer "what was the last state we recorded" when the
        process has not run for a while. State restoration needs the latest
        row, which is what this returns.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM nav_history ORDER BY date DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    # ── Engine State ──

    def save_engine_state(self, state: dict, strategy_id: str = "default") -> None:
        """Persist an engine snapshot atomically (one row, one transaction).

        Used by the daemon so positions, cash and the processed-date watermark
        survive a process restart. The watermark lives in the same row, so a
        crash cannot leave "cash updated but date not advanced" behind.
        """
        payload = json.dumps(state, default=str)
        updated_at = exchange_now_iso()
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO engine_state
                   (strategy_id, state_json, updated_at) VALUES (?, ?, ?)""",
                (strategy_id, payload, updated_at),
            )

    def get_engine_state(self, strategy_id: str = "default") -> dict | None:
        """Load a persisted engine snapshot, or None if none exists."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM engine_state WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["state_json"])

    def get_latest_engine_state(self, tenant_id: str = "") -> tuple[str, dict] | None:
        """Newest *engine* snapshot as ``(key, state)``, or None.

        The scheduler is not constructed with the engine's tenant-scoped key
        in every embedding (and a standalone scheduler has no engine at all),
        so EOD equity lookup (S2) falls back to whichever engine most recently
        persisted a snapshot. Rows written by other features into the same
        key/value table (``risk_state:{tenant}``) and daemon replay snapshots
        (a different schema) are not engine books: accepting them mis-sourced
        equity. Only snapshots that carry an engine book (cash + positions)
        are eligible; with ``tenant_id`` only that tenant's engine key is
        considered. Unreadable JSON is ignored rather than crashing EOD.
        """
        query = "SELECT strategy_id, state_json FROM engine_state"
        params: tuple = ()
        if tenant_id:
            query += " WHERE strategy_id = ?"
            params = (f"live_engine:{tenant_id}",)
        else:
            query += " WHERE strategy_id NOT LIKE 'risk_state:%'"
        query += " ORDER BY updated_at DESC LIMIT 20"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            try:
                state = json.loads(row["state_json"])
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring unreadable engine_state row %s", row["strategy_id"],
                )
                continue
            if (
                isinstance(state, dict)
                and "cash" in state
                and "positions" in state
                and ("portfolio_value" in state or "total_equity" in state)
            ):
                return str(row["strategy_id"]), state
        return None

    # ── Retention (S10) ──

    #: ``(table, policy field, timestamp column, date-only, primary key)``.
    #: Deletion is always by the table's own timestamp; `orders` additionally
    #: requires that no *retained* trade still references the row, so a policy
    #: that keeps trades longer than orders cannot orphan fills. Deletes use
    #: the ids of the rows that were archived, so a concurrent insert with a
    #: historical timestamp is never deleted without an archive copy.
    _RETENTION_TABLES = (
        ("events", "events_days", "created_at", False, "id"),
        ("pnl_history", "pnl_days", "timestamp", False, "id"),
        ("signals", "signals_days", "generated_at", False, "signal_id"),
        ("sessions", "sessions_days", "started_at", False, "session_id"),
        ("trades", "trades_days", "executed_at", False, "trade_id"),
        ("orders", "orders_days", "created_at", False, "order_id"),
        ("nav_history", "nav_days", "date", True, "date"),
    )

    #: Rows deleted per SQL statement, below SQLite's bound-parameter limit.
    _RETENTION_DELETE_CHUNK = 500

    def run_retention(
        self,
        policy: RetentionPolicy | None = None,
        archive_dir: str | Path | None = None,
        vacuum: bool = True,
        now: datetime | None = None,
    ) -> dict:
        """Archive then delete rows older than each table's max age.

        Every eligible batch is written to a dated CSV under ``archive_dir``
        *before* the DELETE runs (temp file + fsync + os.replace); if the
        archive cannot be written the rows are kept. Afterwards the WAL is
        checkpointed and, when anything was deleted, the database is
        vacuumed. Returns per-table counts, archive paths and the resulting
        database size (also exported as the ``store_size_bytes`` gauge).

        The job is idempotent: a second run finds no eligible rows and writes
        no archive.
        """
        policy = policy or RetentionPolicy()
        now = now or exchange_now()
        archive_root = Path(archive_dir or "data/archive")
        result: dict = {
            "tables": {},
            "checkpointed": False,
            "vacuumed": False,
            "db_size_bytes": 0,
        }
        deleted_total = 0

        for table, field, column, date_only, id_column in self._RETENTION_TABLES:
            days = int(getattr(policy, field))
            entry: dict = {
                "enabled": days > 0, "archived": 0, "deleted": 0,
                "archive_path": "", "cutoff": "",
            }
            if days <= 0:
                result["tables"][table] = entry
                continue

            cutoff = (
                (now - timedelta(days=days)).strftime("%Y-%m-%d")
                if date_only
                else (now - timedelta(days=days)).isoformat()
            )
            where = f"{column} < ?"
            extra = ""
            if table == "orders":
                extra = (
                    " AND NOT EXISTS (SELECT 1 FROM trades "
                    "WHERE trades.order_id = orders.order_id)"
                )
            entry["cutoff"] = cutoff

            with self._conn() as conn:
                rows = [
                    dict(row) for row in conn.execute(
                        f"SELECT * FROM {table} WHERE {where}{extra}", (cutoff,)
                    ).fetchall()
                ]
            if not rows:
                result["tables"][table] = entry
                continue

            archive_path = None
            try:
                archive_root.mkdir(parents=True, exist_ok=True)
                stamp = now.strftime("%Y%m%dT%H%M%S%f")
                archive_path = archive_root / f"{table}_{stamp}.csv"
                tmp_path = archive_path.with_name(archive_path.name + ".tmp")
                with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(
                        fh, fieldnames=list(rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, archive_path)
            except OSError as exc:
                logger.critical(
                    "Retention: archiving %s failed (%s); rows kept", table, exc,
                )
                entry["error"] = str(exc)
                result["tables"][table] = entry
                continue

            entry["archive_path"] = str(archive_path)
            entry["archived"] = len(rows)
            ids = [row[id_column] for row in rows]
            # SQLite's bound-variable limit (32,766 on CPython 3.12) is easily
            # exceeded by a real backlog; one giant IN-list archived 40k rows
            # and then failed the DELETE, so retention could never drain.
            deleted = 0
            chunk_size = self._RETENTION_DELETE_CHUNK
            with self._lock, self._conn() as conn:
                for offset in range(0, len(ids), chunk_size):
                    chunk = ids[offset:offset + chunk_size]
                    placeholders = ", ".join("?" for _ in chunk)
                    delete_where = f"{id_column} IN ({placeholders})"
                    if extra:
                        delete_where += extra
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE {delete_where}",
                        tuple(chunk),
                    )
                    if cursor.rowcount > 0:
                        deleted += cursor.rowcount
            entry["deleted"] = deleted
            deleted_total += deleted
            result["tables"][table] = entry
            logger.info(
                "Retention: archived and deleted %d %s row(s) older than %s",
                deleted, table, cutoff,
            )

        try:
            self.checkpoint()
            result["checkpointed"] = True
        except sqlite3.Error as exc:
            logger.error("Retention: WAL checkpoint failed: %s", exc)
        if vacuum and deleted_total > 0:
            try:
                self.vacuum()
                result["vacuumed"] = True
            except sqlite3.Error as exc:
                logger.error("Retention: VACUUM failed: %s", exc)

        result["db_size_bytes"] = self.size_bytes()
        try:
            from quant_platform.utils.metrics import get_metrics

            get_metrics().gauge(
                "store_size_bytes",
                help="SQLite trading database file size in bytes",
            ).set(result["db_size_bytes"])
        except Exception:  # metrics must never break retention
            pass
        return result

    # ── Stats ──

    def get_stats(self) -> dict:
        """Get store statistics."""
        with self._conn() as conn:
            return {
                # The real backend, not a hardcoded label. `/api/postgres/stats`
                # always claimed "sqlite" because it never looked at the store
                # it had constructed.
                "backend": "sqlite",
                "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
                "positions": conn.execute("SELECT COUNT(*) FROM positions WHERE quantity > 0").fetchone()[0],
                "trades": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
                "pnl_snapshots": conn.execute("SELECT COUNT(*) FROM pnl_history").fetchone()[0],
                "signals": conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0],
                "sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "db_path": self._db_path,
                "db_size_bytes": self.size_bytes(),
            }
