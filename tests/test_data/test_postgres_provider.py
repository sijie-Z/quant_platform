"""Tests for PostgreSQL data provider and store."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from quant_platform.data.providers.postgres_provider import (
    AsyncPostgresStore,
    PostgresDataProvider,
    PostgresStore,
)


class TestPostgresStore:
    def test_init_requires_sqlalchemy(self):
        """Should raise ImportError if sqlalchemy not available."""
        with patch("quant_platform.data.providers.postgres_provider.HAS_SQLALCHEMY", False):
            with pytest.raises(ImportError, match="sqlalchemy required"):
                PostgresStore()

    def test_fallback_to_sqlite(self):
        """Should fallback to SQLite when PostgreSQL unavailable."""
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("Connection refused")
            store = PostgresStore(dsn="postgresql://bad:bad@localhost/bad")
            assert store.backend == "sqlite"
            assert store._fallback_store is not None

    def test_save_order_fallback(self):
        """Save order should work via SQLite fallback."""
        import uuid
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            store.save_order({
                "order_id": f"ord-{uuid.uuid4().hex[:8]}",
                "code": "600519",
                "side": "buy",
                "quantity": 100,
                "price": 1800.0,
            })

    def test_save_trade_fallback(self):
        """Save trade should work via SQLite fallback."""
        import uuid
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            store.save_trade({
                "trade_id": f"trd-{uuid.uuid4().hex[:8]}",
                "order_id": f"ord-{uuid.uuid4().hex[:8]}",
                "code": "600519",
                "side": "buy",
                "quantity": 100,
                "price": 1800.0,
            })

    def test_get_orders_fallback(self):
        """Get orders should work via SQLite fallback."""
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            orders = store.get_orders()
            assert isinstance(orders, list)

    def test_get_positions_fallback(self):
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            positions = store.get_positions()
            assert isinstance(positions, (list, pd.DataFrame))

    def test_get_pnl_history_fallback(self):
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            pnl = store.get_pnl_history(days=30)
            assert isinstance(pnl, (pd.DataFrame, list))

    def test_get_stats_fallback(self):
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            stats = store.get_stats()
            assert "orders" in stats
            assert "trades" in stats

    def test_save_signal_fallback(self):
        import uuid
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            store.save_signal({
                "signal_id": f"sig-{uuid.uuid4().hex[:8]}",
                "code": "600519",
                "direction": "buy",
                "strength": 0.8,
            })

    def test_get_signals_fallback(self):
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            store = PostgresStore()
            signals = store.get_signals()
            assert isinstance(signals, list)


class TestPostgresDataProvider:
    def test_init_requires_sqlalchemy(self):
        with patch("quant_platform.data.providers.postgres_provider.HAS_SQLALCHEMY", False):
            with pytest.raises(ImportError):
                PostgresDataProvider()

    def test_not_connected_raises(self):
        """Should raise ConnectionError when not connected."""
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_instance = MagicMock()
            mock_instance.connect.side_effect = Exception("No PG")
            mock_engine.return_value = mock_instance

            provider = PostgresDataProvider()
            assert not provider._connected
            with pytest.raises(ConnectionError):
                provider.get_prices("2023-01-01", "2024-01-01")


class TestAsyncPostgresStore:
    def test_init_requires_asyncpg(self):
        with patch("quant_platform.data.providers.postgres_provider.HAS_ASYNCPG", False):
            with pytest.raises(ImportError, match="asyncpg required"):
                AsyncPostgresStore()

    def test_init_with_asyncpg(self):
        with patch("quant_platform.data.providers.postgres_provider.HAS_ASYNCPG", True):
            store = AsyncPostgresStore()
            assert store._pool is None
