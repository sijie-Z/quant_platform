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

Usage:
    store = Store('data/trading.db')
    store.save_order(order)
    store.save_trade(trade)
    store.get_positions()
    store.get_pnl_history(days=30)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class Store:
    """Thread-safe SQLite store for all trading state.

    Uses WAL mode for concurrent read/write.
    All writes are serialized through a lock.
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self):
        """Get a database connection."""
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
                    code TEXT PRIMARY KEY,
                    tenant_id TEXT DEFAULT 'default',
                    name TEXT DEFAULT '',
                    quantity INTEGER NOT NULL,
                    available INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    current_price REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    updated_at TEXT NOT NULL
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
                    cumulative_return REAL DEFAULT 0
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

            # Create tenant_id indexes after migration
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_orders_tenant ON orders(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_trades_tenant ON trades(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_positions_tenant ON positions(tenant_id);
            """)
        logger.info("Store initialized: %s", self._db_path)

    def _migrate_tenant_id(self, conn: sqlite3.Connection) -> None:
        """Add tenant_id column to existing tables if missing."""
        for table in ("orders", "trades", "positions"):
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "tenant_id" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT DEFAULT 'default'")
                logger.info("Migrated %s: added tenant_id", table)

    # ── Orders ──

    def save_order(self, order: dict):
        """Save or update an order."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO orders
                    (order_id, tenant_id, code, side, order_type, quantity, price,
                     filled_quantity, filled_price, status, commission, tax,
                     slippage, broker_order_id, error_msg, strategy_id, signal_id,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                order['order_id'], order.get('tenant_id', 'default'),
                order['code'], order['side'],
                order.get('order_type', 'limit'), order['quantity'], order['price'],
                order.get('filled_quantity', 0), order.get('filled_price', 0),
                order.get('status', 'pending'), order.get('commission', 0),
                order.get('tax', 0), order.get('slippage', 0),
                order.get('broker_order_id', ''), order.get('error_msg', ''),
                order.get('strategy_id', ''), order.get('signal_id', ''),
                order.get('created_at', datetime.now().isoformat()),
                order.get('updated_at', datetime.now().isoformat()),
            ))

    def get_orders(self, status: str = "", code: str = "", limit: int = 100,
                   tenant_id: str = "") -> list[dict]:
        """Get orders with optional filters."""
        with self._conn() as conn:
            query = "SELECT * FROM orders WHERE 1=1"
            params = []
            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)
            if status:
                query += " AND status = ?"
                params.append(status)
            if code:
                query += " AND code = ?"
                params.append(code)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── Positions ──

    def save_position(self, pos: dict):
        """Save or update a position."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO positions
                    (code, tenant_id, name, quantity, available, avg_cost, current_price,
                     market_value, unrealized_pnl, realized_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                pos['code'], pos.get('tenant_id', 'default'),
                pos.get('name', ''), pos['quantity'],
                pos.get('available', pos['quantity']), pos['avg_cost'],
                pos.get('current_price', 0), pos.get('market_value', 0),
                pos.get('unrealized_pnl', 0), pos.get('realized_pnl', 0),
                datetime.now().isoformat(),
            ))

    def delete_position(self, code: str, tenant_id: str = ""):
        """Remove a position."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM positions WHERE code = ?", (code,))

    def get_positions(self, tenant_id: str = "") -> list[dict]:
        """Get all current positions."""
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
        """Record an executed trade."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT INTO trades
                    (trade_id, tenant_id, order_id, code, side, quantity, price,
                     commission, tax, executed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                trade.get('trade_id', trade.get('order_id', '')),
                trade.get('tenant_id', 'default'),
                trade['order_id'], trade['code'], trade['side'],
                trade['quantity'], trade['price'],
                trade.get('commission', 0), trade.get('tax', 0),
                trade.get('executed_at', datetime.now().isoformat()),
            ))

    def get_trades(self, code: str = "", limit: int = 100,
                   tenant_id: str = "") -> list[dict]:
        """Get trade history."""
        with self._conn() as conn:
            if code:
                if tenant_id:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE code = ? AND tenant_id = ? "
                        "ORDER BY executed_at DESC LIMIT ?",
                        (code, tenant_id, limit)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE code = ? ORDER BY executed_at DESC LIMIT ?",
                        (code, limit)).fetchall()
            else:
                if tenant_id:
                    rows = conn.execute(
                        "SELECT * FROM trades WHERE tenant_id = ? "
                        "ORDER BY executed_at DESC LIMIT ?",
                        (tenant_id, limit)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM trades ORDER BY executed_at DESC LIMIT ?",
                        (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── P&L History ──

    def save_pnl_snapshot(self, snapshot: dict):
        """Save a P&L snapshot."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT INTO pnl_history
                    (timestamp, total_equity, cash, market_value, daily_pnl,
                     daily_pnl_pct, cumulative_pnl, n_positions, max_drawdown, sharpe_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                snapshot.get('timestamp', datetime.now().isoformat()),
                snapshot['total_equity'], snapshot['cash'],
                snapshot['market_value'], snapshot.get('daily_pnl', 0),
                snapshot.get('daily_pnl_pct', 0), snapshot.get('cumulative_pnl', 0),
                snapshot.get('n_positions', 0), snapshot.get('max_drawdown', 0),
                snapshot.get('sharpe_ratio', 0),
            ))

    def get_pnl_history(self, days: int = 30) -> list[dict]:
        """Get P&L history for the last N days."""
        with self._conn() as conn:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            rows = conn.execute(
                "SELECT * FROM pnl_history WHERE timestamp > ? ORDER BY timestamp",
                (cutoff,)).fetchall()
            return [dict(r) for r in rows]

    # ── Signals ──

    def save_signal(self, signal: dict):
        """Save an alpha signal."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO signals
                    (signal_id, code, direction, strength, factor_values,
                     strategy_id, generated_at, consumed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                signal['signal_id'], signal['code'], signal['direction'],
                signal['strength'], json.dumps(signal.get('factor_values', {})),
                signal.get('strategy_id', ''),
                signal.get('generated_at', datetime.now().isoformat()),
                signal.get('consumed', 0),
            ))

    def get_signals(self, consumed: int = -1, limit: int = 100) -> list[dict]:
        """Get signals. consumed=-1 for all, 0 for unconsumed, 1 for consumed."""
        with self._conn() as conn:
            if consumed >= 0:
                rows = conn.execute(
                    "SELECT * FROM signals WHERE consumed = ? ORDER BY generated_at DESC LIMIT ?",
                    (consumed, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM signals ORDER BY generated_at DESC LIMIT ?",
                    (limit,)).fetchall()
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

    def save_session(self, session: dict):
        """Save a trading session record."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, strategy_id, broker, status, started_at,
                     stopped_at, total_trades, total_pnl, config)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                session['session_id'], session.get('strategy_id', ''),
                session.get('broker', 'simulated'), session.get('status', 'active'),
                session.get('started_at', datetime.now().isoformat()),
                session.get('stopped_at', ''),
                session.get('total_trades', 0), session.get('total_pnl', 0),
                json.dumps(session.get('config', {})),
            ))

    def get_sessions(self, limit: int = 20) -> list[dict]:
        """Get trading session history."""
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
                    INSERT INTO events (event_id, topic, data, source, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                event.get('event_id', ''), event.get('topic', ''),
                json.dumps(event.get('data', {})),
                event.get('source', ''), event.get('timestamp', time.time()),
                datetime.now().isoformat(),
            ))

    def get_events(self, topic: str = "", limit: int = 100) -> list[dict]:
        """Get event log."""
        with self._conn() as conn:
            if topic:
                rows = conn.execute(
                    "SELECT * FROM events WHERE topic = ? ORDER BY timestamp DESC LIMIT ?",
                    (topic, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,)).fetchall()
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
                     mgmt_fee, perf_fee, high_water_mark, daily_return, cumulative_return)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                nav['date'], nav['nav_total'], nav['nav_per_unit'],
                nav['total_units'], nav.get('cash', 0),
                nav.get('market_value', 0), nav.get('mgmt_fee', 0),
                nav.get('perf_fee', 0), nav.get('high_water_mark', 1.0),
                nav.get('daily_return', 0), nav.get('cumulative_return', 0),
            ))

    def get_nav_history(self, days: int = 365 * 3) -> list[dict]:
        """Get NAV history for the last N days."""
        with self._conn() as conn:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT * FROM nav_history WHERE date > ? ORDER BY date",
                (cutoff,)).fetchall()
            return [dict(r) for r in rows]

    # ── Stats ──

    def get_stats(self) -> dict:
        """Get store statistics."""
        with self._conn() as conn:
            return {
                "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
                "positions": conn.execute("SELECT COUNT(*) FROM positions WHERE quantity > 0").fetchone()[0],
                "trades": conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0],
                "pnl_snapshots": conn.execute("SELECT COUNT(*) FROM pnl_history").fetchone()[0],
                "signals": conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0],
                "sessions": conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "db_path": self._db_path,
            }
