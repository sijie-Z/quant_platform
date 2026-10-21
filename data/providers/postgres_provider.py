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

import numpy as np
import pandas as pd

from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import FINANCIAL_COLUMNS, DataQualityError

Base = declarative_base() if HAS_SQLALCHEMY else None

#: Close bases ``daily_prices.close`` may carry.
_PRICE_BASES = ("raw", "qfq", "hfq")

#: Columns the ``daily_prices`` query exposes (the provider's field universe).
_PIVOT_COLUMNS = (
    "open", "high", "low", "close", "volume", "turnover", "adj_factor",
)


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
        from quant_core.store import Store
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

    def save_order_with_trade(self, order: dict, trade: dict):
        """Write an order and its trade in one transaction."""
        if self._fallback_store:
            return self._fallback_store.save_order_with_trade(order, trade)

        with self._lock, self._session() as session:
            row = session.get(OrderRow, order.get("order_id"))
            if row:
                for key, value in order.items():
                    if hasattr(row, key):
                        setattr(row, key, value)
            else:
                session.add(OrderRow(**{
                    k: v for k, v in order.items() if hasattr(OrderRow, k)
                }))
            session.add(TradeRow(**{
                k: v for k, v in trade.items() if hasattr(TradeRow, k)
            }))

    def _state_store(self):
        """Local metadata store for engine state / EOD markers.

        These are process-local runtime records; keeping them on the local
        SQLite file avoids adding lifecycle tables to the market-data schema
        and matches the default deployment.
        """
        if self._fallback_store is not None:
            return self._fallback_store
        if not hasattr(self, "_local_state_store"):
            from quant_core.store import Store

            self._local_state_store = Store()
        return self._local_state_store

    def save_engine_state(self, state: dict, strategy_id: str):
        return self._state_store().save_engine_state(state, strategy_id)

    def get_engine_state(self, strategy_id: str):
        return self._state_store().get_engine_state(strategy_id)

    def has_eod_reconciliation(self, date: str) -> bool:
        return self._state_store().has_eod_reconciliation(date)

    def mark_eod_reconciliation(self, date: str) -> None:
        return self._state_store().mark_eod_reconciliation(date)

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

    Adjustment contract: by default ``daily_prices.close`` must be the
    **raw** (unadjusted) close and ``daily_prices.adj_factor`` the cumulative
    adjustment divisor, so the pipeline computes
    ``close_adj = close / adj_factor`` exactly once.  This provider declares
    that contract explicitly (``price_adjustment="raw"``,
    ``adjusted_price_basis="hfq"``, ``adjustment_factor_available=True``) and
    stamps the matching ``attrs`` on every panel it returns, so the pipeline
    never has to infer the basis from factor variance.

    A database that only stores an already-adjusted close must say so with
    ``close_adjustment="qfq"``/``"hfq"``; its ``adj_factor`` column is then
    required to be a placeholder ``1.0``.  A non-constant (or non-unit)
    factor alongside an adjusted close is rejected with
    :class:`~quant_platform.data.schema.DataQualityError`: dividing that
    panel by the factor would double-adjust it.  A NULL, zero, negative or
    non-finite divisor is always rejected, never silently divided.

    ``st_available`` is False: this provider has no ST history (the static
    ``stock_metadata.is_st`` flag is all it carries), so callers must treat
    per-date ST screening as unavailable.  SQL NULL ``is_st`` values are
    coalesced to False for the static flag only.

    Args:
        dsn: PostgreSQL connection string.
        close_adjustment: Basis of ``daily_prices.close``: ``"raw"``
            (default; OHLCV plus a divisor) or ``"qfq"``/``"hfq"`` when the
            table stores an already-adjusted close with ``adj_factor=1``.
    """

    # Explicit price-basis declaration, resolved by DataPipeline.  ``raw``
    # close + a divisor column is the shared contract; the pipeline's own
    # ``close / adj_factor`` series is the back-adjusted (hfq) one.
    price_adjustment = "raw"
    adjusted_price_basis = "hfq"
    adjustment_factor_available = True

    def __init__(
        self,
        dsn: str = "postgresql://quant:quant@localhost:5432/quant",
        close_adjustment: str = "raw",
    ):
        if not HAS_SQLALCHEMY:
            raise ImportError("sqlalchemy required")

        basis = str(close_adjustment).strip().lower()
        if basis not in _PRICE_BASES:
            raise ValueError(
                "close_adjustment must be one of "
                f"{_PRICE_BASES}, got {close_adjustment!r}"
            )
        self.price_adjustment = basis
        # raw close + divisor is the shared back-adjustment (hfq) contract.
        self.adjusted_price_basis = "hfq" if basis == "raw" else basis
        self.adjustment_factor_available = basis == "raw"

        self._engine = create_engine(dsn, pool_pre_ping=True)
        self._connected = False
        # No point-in-time ST history is supplied by this source.
        self.st_available = False
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self._connected = True
        except Exception as e:
            # main.py relies on a construction failure to fall back to
            # synthetic data, which the class docstring promises. Swallowing
            # the error here produced a half-connected provider: main logged
            # "Using PostgreSQL data provider" and the first pipeline call
            # aborted with ConnectionError instead of falling back.
            logger.warning("PostgreSQL not available: %s", e)
            raise ConnectionError(f"PostgreSQL not available: {e}") from e

    def _basis_attrs(self) -> dict:
        """The explicit basis declaration this provider stamps on its panels."""
        return {
            "price_basis": self.price_adjustment,
            "adjusted_price_basis": self.adjusted_price_basis,
            "adjustment_factor_available": bool(
                self.adjustment_factor_available
            ),
        }

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
        pivot_cols = list(fields) if fields else list(_PIVOT_COLUMNS)
        # The fields projection contract applies to empty results too: an
        # unknown name must raise, not silently materialise a column that
        # no data backed.
        unknown_fields = [c for c in pivot_cols if c not in _PIVOT_COLUMNS]
        if unknown_fields:
            raise ValueError(f"Unknown price fields: {unknown_fields}")
        if df.empty:
            # Empty results must still satisfy the MultiIndex contract:
            # downstream code calls get_level_values("asset") unconditionally.
            # The basis declaration travels on the empty frame too so a run
            # with no rows still resolves the provider's contract explicitly.
            index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
            empty = pd.DataFrame(index=index, columns=list(pivot_cols), dtype=float)
            empty.attrs.update(self._basis_attrs())
            return empty

        df["date"] = pd.to_datetime(df["date"])
        missing = [c for c in pivot_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown price fields: {missing}")

        # Validate the documented adjustment convention before it reaches the
        # pipeline's blind division.  A NULL/zero/negative/infinite divisor
        # would produce inf/NaN adjusted prices or silently halve returns.
        factors = pd.to_numeric(df["adj_factor"], errors="coerce")
        factor_values = factors.to_numpy(dtype=float)
        valid = np.isfinite(factor_values) & (factor_values > 0)
        if not bool(valid.all()):
            raise DataQualityError(
                "daily_prices.adj_factor must be a positive finite value "
                f"(declared close basis {self.price_adjustment!r}); "
                f"{int((~valid).sum())} row(s) violate the contract"
            )
        if self.price_adjustment in {"qfq", "hfq"}:
            # The DB stores an already-adjusted close, so the factor is a
            # placeholder.  Anything else means the panel would be divided
            # again by the pipeline -- the classic double-adjustment error.
            non_unit = int((np.abs(factor_values - 1.0) > 1e-12).sum())
            if non_unit:
                raise DataQualityError(
                    "daily_prices.close is declared "
                    f"{self.price_adjustment!r} but adj_factor is not 1.0 on "
                    f"{non_unit} row(s). The pipeline divides close by "
                    "adj_factor, so this panel would be double-adjusted. "
                    "Store raw close plus a divisor, or an already-adjusted "
                    "close with adj_factor=1."
                )

        # Contract (data/providers/base.py): MultiIndex (date, asset) with one
        # column per field.
        result = df.set_index(["date", "code"])[pivot_cols].sort_index()
        result.index = result.index.set_names(["date", "asset"])
        result.attrs.update(self._basis_attrs())
        return result

    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self._connected:
            raise ConnectionError("PostgreSQL not connected")

        select_cols = ", ".join(
            ["date", "code", *FINANCIAL_COLUMNS, "publish_date"]
        )
        query = f"""
            SELECT {select_cols}
            FROM financials
            WHERE date BETWEEN :start AND :end
            ORDER BY date, code
        """
        df = pd.read_sql(query, self._engine, params={"start": start_date, "end": end_date})

        index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        if df.empty:
            return pd.DataFrame(
                index=index, columns=[*FINANCIAL_COLUMNS, "publish_date"]
            )

        df["date"] = pd.to_datetime(df["date"])
        missing = [c for c in FINANCIAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "financials table is missing required columns: "
                f"{missing} (expected FINANCIAL_COLUMNS from data.schema)"
            )
        # publish_date is required for point-in-time use; when the table does
        # not carry it, expose NaT explicitly instead of pretending the panel
        # is PIT-capable.
        if "publish_date" not in df.columns:
            logger.warning(
                "financials table has no publish_date; get_financials_as_of "
                "cannot filter this panel point-in-time"
            )
            df["publish_date"] = pd.NaT
        else:
            df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

        df = df.set_index(["date", "code"])
        df.index = df.index.set_names(["date", "asset"])
        return df.sort_index()

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
            series = df["return_value"]
            series.name = "benchmark"
            return series
        return pd.Series(dtype=float, name="benchmark")

    def get_metadata(self) -> pd.DataFrame:
        if not self._connected:
            raise ConnectionError("PostgreSQL not connected")

        query = (
            "SELECT code, sector, market_cap_group, is_st, listing_date, "
            "delisting_date FROM stock_metadata"
        )
        df = pd.read_sql(query, self._engine)
        if df.empty:
            return pd.DataFrame(
                columns=["sector", "market_cap_group", "is_st",
                         "listing_date", "delisting_date"],
                index=pd.Index([], name="asset"),
            )

        df = df.set_index("code")
        df.index.name = "asset"
        # SQL NULL means "unknown"; the static flag cannot express that and a
        # bare `~is_st` on object NULLs raises TypeError downstream.  Coalesce
        # for this static flag, and expose st_available=False on the provider
        # so the absence of a real ST history remains visible.
        if "is_st" in df.columns:
            df["is_st"] = df["is_st"].fillna(False).astype(bool)
        for col in ("listing_date", "delisting_date"):
            if col not in df.columns:
                df[col] = pd.NaT
            df[col] = pd.to_datetime(df[col], errors="coerce")
        if "sector" not in df.columns:
            df["sector"] = "Unknown"
        if "market_cap_group" not in df.columns:
            df["market_cap_group"] = "mid"
        df.attrs["st_available"] = False
        return df

    def get_st_timeseries(self) -> pd.DataFrame:
        """Explicitly expose that no ST history is available."""
        return pd.DataFrame(
            columns=["asset", "trigger_date", "announce_date", "end_date", "is_st"]
        )

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
