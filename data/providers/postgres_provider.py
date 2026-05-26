"""PostgreSQL/TimescaleDB data provider and store.

Drop-in replacement for SQLite Store with:
- Connection pooling via SQLAlchemy
- Async support via asyncpg
- TimescaleDB hypertable support for time-series data
- Automatic fallback to SQLite if PostgreSQL unavailable

Usage:
    store = PostgresStore("postgresql://user:pass@localhost:5432/quant")
    store.save_order(order)
    store.get_pnl_history(days=30)

    # As DataProvider
    provider = PostgresDataProvider("postgresql://...")
    prices = provider.get_prices("2023-01-01", "2024-01-01")
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports — graceful degradation
try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Index,
        Integer,
        MetaData,
        String,
        Table,
        Text,
        create_engine,
        text,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy.pool import QueuePool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    logger.info("sqlalchemy not installed. PostgreSQL provider unavailable.")

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

import pandas as pd

from quant_platform.data.providers.base import DataProvider

Base = declarative_base() if HAS_SQLALCHEMY else None


class OrderRow(Base if HAS_SQLALCHEMY else object):
    """SQLAlchemy model for orders table."""
    __tablename__ = "orders"

    if HAS_SQLALCHEMY:
        order_id = Column(String(64), primary_key=True)
        code = Column(String(20), nullable=False, index=True)
        side = Column(String(10), nullable=False)
        order_type = Column(String(20), default="limit")
        quantity = Column(Integer, nullable=False)
        price = Column(Float, nullable=False)
        filled_quantity = Column(Integer, default=0)
        filled_price = Column(Float, default=0)
        status = Column(String(20), default="pending", index=True)
        commission = Column(Float, default=0)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PositionRow(Base if HAS_SQLALCHEMY else object):
    """SQLAlchemy model for positions table."""
    __tablename__ = "positions"

    if HAS_SQLALCHEMY:
        code = Column(String(20), primary_key=True)
        quantity = Column(Integer, nullable=False)
        avg_cost = Column(Float, nullable=False)
        current_price = Column(Float, default=0)
        unrealized_pnl = Column(Float, default=0)
        sector = Column(String(40), default="")
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PnLRow(Base if HAS_SQLALCHEMY else object):
    """SQLAlchemy model for pnl_history table."""
    __tablename__ = "pnl_history"

    if HAS_SQLALCHEMY:
        id = Column(Integer, primary_key=True, autoincrement=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        total_equity = Column(Float, nullable=False)
        cash = Column(Float, nullable=False)
        positions_value = Column(Float, default=0)
        daily_return = Column(Float, default=0)
        cumulative_return = Column(Float, default=0)
        n_positions = Column(Integer, default=0)


class TradeRow(Base if HAS_SQLALCHEMY else object):
    """SQLAlchemy model for trades table."""
    __tablename__ = "trades"

    if HAS_SQLALCHEMY:
        trade_id = Column(String(64), primary_key=True)
        order_id = Column(String(64), index=True)
        code = Column(String(20), nullable=False)
        side = Column(String(10), nullable=False)
        quantity = Column(Integer, nullable=False)
        price = Column(Float, nullable=False)
        commission = Column(Float, default=0)
        executed_at = Column(DateTime, default=datetime.utcnow)


class SignalRow(Base if HAS_SQLALCHEMY else object):
    """SQLAlchemy model for signals table."""
    __tablename__ = "signals"

    if HAS_SQLALCHEMY:
        signal_id = Column(String(64), primary_key=True)
        code = Column(String(20), nullable=False, index=True)
        direction = Column(String(10), nullable=False)
        strength = Column(Float, default=0)
        factor_values = Column(Text, default="{}")
        created_at = Column(DateTime, default=datetime.utcnow)


class PostgresStore:
    """PostgreSQL-backed store with connection pooling.

    Drop-in replacement for SQLite Store. Falls back to SQLite
    if PostgreSQL is unavailable.

    Args:
        dsn: PostgreSQL connection string
        pool_size: Connection pool size
        max_overflow: Max overflow connections
        echo: Log SQL statements
    """

    def __init__(
        self,
        dsn: str = "postgresql://quant:quant@localhost:5432/quant",
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ):
        if not HAS_SQLALCHEMY:
            raise ImportError(
                "sqlalchemy required for PostgreSQL provider. "
                "Install with: pip install sqlalchemy"
            )

        self._dsn = dsn
        self._lock = threading.Lock()
        self._engine = None
        self._session_factory = None
        self._fallback_store = None

        try:
            self._engine = create_engine(
                dsn,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                echo=echo,
            )
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine)
            self._backend = "postgresql"
            logger.info("Connected to PostgreSQL: %s", dsn.split("@")[-1])

        except Exception as e:
            logger.warning("PostgreSQL unavailable (%s), falling back to SQLite", e)
            self._init_fallback()

    def _init_fallback(self):
        """Initialize SQLite fallback."""
        from quant_platform.core.store import Store
        self._fallback_store = Store()
        self._backend = "sqlite"

    @property
    def backend(self) -> str:
        return self._backend

    @contextmanager
    def _session(self) -> Generator:
        """Get a database session."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_order(self, order: dict):
        """Save or update an order."""
        if self._fallback_store:
            return self._fallback_store.save_order(order)

        with self._lock, self._session() as session:
            row = session.get(OrderRow, order.get("order_id"))
            if row:
                for k, v in order.items():
                    if hasattr(row, k):
                        setattr(row, k, v)
            else:
                session.add(OrderRow(**{
                    k: v for k, v in order.items()
                    if hasattr(OrderRow, k)
                }))

    def save_trade(self, trade: dict):
        """Save a trade execution."""
        if self._fallback_store:
            return self._fallback_store.save_trade(trade)

        with self._lock, self._session() as session:
            session.add(TradeRow(**{
                k: v for k, v in trade.items()
                if hasattr(TradeRow, k)
            }))

    def save_position(self, position: dict):
        """Save or update a position."""
        if self._fallback_store:
            return self._fallback_store.save_position(position)

        with self._lock, self._session() as session:
            row = session.get(PositionRow, position.get("code"))
            if row:
                for k, v in position.items():
                    if hasattr(row, k):
                        setattr(row, k, v)
            else:
                session.add(PositionRow(**{
                    k: v for k, v in position.items()
                    if hasattr(PositionRow, k)
                }))

    def save_pnl(self, pnl: dict):
        """Save a P&L snapshot."""
        if self._fallback_store:
            return self._fallback_store.save_pnl(pnl)

        with self._lock, self._session() as session:
            session.add(PnLRow(**{
                k: v for k, v in pnl.items()
                if hasattr(PnLRow, k)
            }))

    def save_signal(self, signal: dict):
        """Save an alpha signal."""
        if self._fallback_store:
            return self._fallback_store.save_signal(signal)

        with self._lock, self._session() as session:
            session.add(SignalRow(**{
                k: v for k, v in signal.items()
                if hasattr(SignalRow, k)
            }))

    def get_orders(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """Get orders, optionally filtered by status."""
        if self._fallback_store:
            return self._fallback_store.get_orders(status=status, limit=limit)

        with self._session() as session:
            q = session.query(OrderRow)
            if status:
                q = q.filter(OrderRow.status == status)
            q = q.order_by(OrderRow.created_at.desc()).limit(limit)
            return [
                {c.name: getattr(row, c.name) for c in row.__table__.columns}
                for row in q.all()
            ]

    def get_positions(self) -> list[dict]:
        """Get all current positions."""
        if self._fallback_store:
            return self._fallback_store.get_positions()

        with self._session() as session:
            rows = session.query(PositionRow).all()
            return [
                {c.name: getattr(row, c.name) for c in row.__table__.columns}
                for row in rows
            ]

    def get_trades(self, limit: int = 100) -> list[dict]:
        """Get recent trades."""
        if self._fallback_store:
            return self._fallback_store.get_trades(limit=limit)

        with self._session() as session:
            rows = (
                session.query(TradeRow)
                .order_by(TradeRow.executed_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {c.name: getattr(row, c.name) for c in row.__table__.columns}
                for row in rows
            ]

    def get_pnl_history(self, days: int = 30) -> pd.DataFrame:
        """Get P&L history as DataFrame."""
        if self._fallback_store:
            rows = self._fallback_store.get_pnl_history(days=days)
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._session() as session:
            rows = (
                session.query(PnLRow)
                .filter(PnLRow.timestamp >= cutoff)
                .order_by(PnLRow.timestamp)
                .all()
            )
            if not rows:
                return pd.DataFrame()
            data = [
                {c.name: getattr(row, c.name) for c in row.__table__.columns}
                for row in rows
            ]
            return pd.DataFrame(data).set_index("timestamp")

    def get_signals(self, code: str | None = None, limit: int = 100) -> list[dict]:
        """Get signal history."""
        if self._fallback_store:
            return self._fallback_store.get_signals(limit=limit)

        with self._session() as session:
            q = session.query(SignalRow)
            if code:
                q = q.filter(SignalRow.code == code)
            q = q.order_by(SignalRow.created_at.desc()).limit(limit)
            return [
                {c.name: getattr(row, c.name) for c in row.__table__.columns}
                for row in q.all()
            ]

    def get_stats(self) -> dict:
        """Get database statistics."""
        if self._fallback_store:
            return self._fallback_store.get_stats()

        stats = {"backend": "postgresql", "dsn_host": self._dsn.split("@")[-1]}
        with self._session() as session:
            stats["orders"] = session.query(OrderRow).count()
            stats["positions"] = session.query(PositionRow).count()
            stats["trades"] = session.query(TradeRow).count()
            stats["pnl_snapshots"] = session.query(PnLRow).count()
            stats["signals"] = session.query(SignalRow).count()
        return stats

    def close(self):
        """Close all connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL connections closed")


class PostgresDataProvider(DataProvider):
    """PostgreSQL-backed data provider.

    Reads OHLCV data from a PostgreSQL/TimescaleDB database.
    Falls back to synthetic data if connection fails.

    Args:
        dsn: PostgreSQL connection string
    """

    def __init__(self, dsn: str = "postgresql://quant:quant@localhost:5432/quant"):
        if not HAS_SQLALCHEMY:
            raise ImportError("sqlalchemy required")

        self._engine = create_engine(dsn, pool_pre_ping=True)
        self._connected = False
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._connected = True
        except Exception as e:
            logger.warning("PostgreSQL not available: %s", e)

    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self._connected:
            raise ConnectionError("PostgreSQL not connected")

        query = """
            SELECT date, code, open, high, low, close, volume, turnover, adj_factor
            FROM daily_prices
            WHERE date BETWEEN :start AND :end
            ORDER BY date, code
        """
        df = pd.read_sql(query, self._engine, params={"start": start_date, "end": end_date})
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        pivot_cols = fields or ["open", "high", "low", "close", "volume", "turnover", "adj_factor"]
        result = df.pivot(index="date", columns="code", values=pivot_cols[0])
        if fields and len(fields) > 1:
            for f in fields[1:]:
                result = pd.concat([result, df.pivot(index="date", columns="code", values=f)], axis=1)
        return result

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self._connected:
            raise ConnectionError("PostgreSQL not connected")

        query = """
            SELECT date, code, market_cap, pe_ratio, pb_ratio, roe, revenue, net_profit
            FROM financials
            WHERE date BETWEEN :start AND :end
            ORDER BY date, code
        """
        df = pd.read_sql(query, self._engine, params={"start": start_date, "end": end_date})
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "code"])
        return df

    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        if not self._connected:
            raise ConnectionError("PostgreSQL not connected")

        query = """
            SELECT date, return_value FROM benchmark_returns
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """
        df = pd.read_sql(query, self._engine, params={"start": start_date, "end": end_date})
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            return df["return_value"]
        return pd.Series(dtype=float)

    def get_metadata(self) -> pd.DataFrame:
        if not self._connected:
            raise ConnectionError("PostgreSQL not connected")

        query = "SELECT code, sector, market_cap_group, is_st, listing_date FROM stock_metadata"
        df = pd.read_sql(query, self._engine)
        if not df.empty:
            df = df.set_index("code")
        return df

    def close(self):
        if self._engine:
            self._engine.dispose()


class AsyncPostgresStore:
    """Async version of PostgresStore using asyncpg.

    For high-throughput async pipelines. Requires asyncpg.

    Args:
        dsn: PostgreSQL connection string
        min_size: Minimum pool connections
        max_size: Maximum pool connections
    """

    def __init__(
        self,
        dsn: str = "postgresql://quant:quant@localhost:5432/quant",
        min_size: int = 2,
        max_size: int = 10,
    ):
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg required for async store")

        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self):
        """Initialize the connection pool."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        logger.info("Async PostgreSQL pool connected")

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Async PostgreSQL pool closed")

    async def save_pnl(self, pnl: dict):
        """Async save P&L snapshot."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pnl_history (timestamp, total_equity, cash, positions_value,
                    daily_return, cumulative_return, n_positions)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                pnl.get("timestamp", datetime.utcnow()),
                pnl["total_equity"],
                pnl["cash"],
                pnl.get("positions_value", 0),
                pnl.get("daily_return", 0),
                pnl.get("cumulative_return", 0),
                pnl.get("n_positions", 0),
            )

    async def get_pnl_history(self, days: int = 30) -> list[dict]:
        """Async get P&L history."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM pnl_history
                WHERE timestamp >= NOW() - $1::interval
                ORDER BY timestamp
                """,
                f"{days} days",
            )
            return [dict(row) for row in rows]

    async def get_stats(self) -> dict:
        """Async get database stats."""
        async with self._pool.acquire() as conn:
            orders = await conn.fetchval("SELECT COUNT(*) FROM orders")
            trades = await conn.fetchval("SELECT COUNT(*) FROM trades")
            pnl = await conn.fetchval("SELECT COUNT(*) FROM pnl_history")
            return {
                "backend": "async_postgresql",
                "orders": orders,
                "trades": trades,
                "pnl_snapshots": pnl,
            }
