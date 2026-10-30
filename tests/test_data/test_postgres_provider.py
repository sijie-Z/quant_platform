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
    @pytest.fixture
    def fallback_store(self, tmp_path, monkeypatch):
        """A PostgresStore whose connection failed over to a throwaway SQLite."""
        monkeypatch.chdir(tmp_path)
        with patch(
            "quant_platform.data.providers.postgres_provider.create_engine"
        ) as mock_engine:
            mock_engine.side_effect = Exception("No PG")
            return PostgresStore()

    def test_init_requires_sqlalchemy(self):
        """Should raise ImportError if sqlalchemy not available."""
        with patch("quant_platform.data.providers.postgres_provider.HAS_SQLALCHEMY", False):
            with pytest.raises(ImportError, match="sqlalchemy required"):
                PostgresStore()

    def test_fallback_to_sqlite(self, fallback_store):
        """Should fallback to SQLite when PostgreSQL unavailable."""
        assert fallback_store.backend == "sqlite"
        assert fallback_store._fallback_store is not None

    def test_save_order_fallback(self, fallback_store):
        """The fallback must persist the row, not just accept the call."""
        import uuid
        order_id = f"ord-{uuid.uuid4().hex[:8]}"
        fallback_store.save_order({
            "order_id": order_id,
            "code": "600519",
            "side": "buy",
            "quantity": 100,
            "price": 1800.0,
            "status": "submitted",
        })

        saved = next(
            o for o in fallback_store.get_orders()
            if o["order_id"] == order_id
        )
        assert saved["side"] == "buy"
        assert saved["quantity"] == 100
        assert saved["price"] == pytest.approx(1800.0)
        assert saved["status"] == "submitted"

    def test_save_trade_fallback(self, fallback_store):
        """The fallback must persist the trade with its own identifiers."""
        import uuid
        trade_id = f"trd-{uuid.uuid4().hex[:8]}"
        order_id = f"ord-{uuid.uuid4().hex[:8]}"
        fallback_store.save_trade({
            "trade_id": trade_id,
            "order_id": order_id,
            "code": "600519",
            "side": "buy",
            "quantity": 100,
            "price": 1800.0,
            "commission": 54.0,
        })

        saved = next(
            t for t in fallback_store.get_trades()
            if t["trade_id"] == trade_id
        )
        assert saved["order_id"] == order_id
        assert saved["quantity"] == 100
        assert saved["commission"] == pytest.approx(54.0)

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

    def test_save_signal_fallback(self, fallback_store):
        import uuid
        signal_id = f"sig-{uuid.uuid4().hex[:8]}"
        fallback_store.save_signal({
            "signal_id": signal_id,
            "code": "600519",
            "direction": "buy",
            "strength": 0.8,
            "factor_values": {"momentum_1m": 0.4},
        })

        saved = next(
            s for s in fallback_store.get_signals()
            if s["signal_id"] == signal_id
        )
        assert saved["direction"] == "buy"
        assert saved["strength"] == pytest.approx(0.8)
        assert saved["factor_values"] == {"momentum_1m": 0.4}

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

    def test_unavailable_database_raises_at_construction(self):
        """The documented fallback depends on construction failing.

        `main.py` catches the exception and switches to synthetic data. The
        old constructor swallowed it, so main logged success and the first
        pipeline call crashed instead of falling back.
        """
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_instance = MagicMock()
            mock_instance.connect.side_effect = Exception("No PG")
            mock_engine.return_value = mock_instance

            with pytest.raises(ConnectionError, match="not available"):
                PostgresDataProvider()

    def test_get_prices_matches_the_multiindex_contract(self):
        """Must return MultiIndex (date, asset) with one column per field.

        The old implementation pivoted only `fields[0]` (open by default) and
        produced duplicate column labels for multiple fields, so
        `DataPipeline` received no `close` column.
        """
        with patch("quant_platform.data.providers.postgres_provider.create_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            provider = PostgresDataProvider()

        frame = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "code": ["A", "B", "A"],
            "open": [10.0, 20.0, 11.0],
            "high": [11.0, 21.0, 12.0],
            "low": [9.0, 19.0, 10.0],
            "close": [10.5, 20.5, 11.5],
            "volume": [1000, 2000, 1100],
            "turnover": [0.01, 0.02, 0.01],
            "adj_factor": [1.0, 1.0, 1.0],
        })

        with patch(
            "quant_platform.data.providers.postgres_provider.pd.read_sql",
            return_value=frame,
        ):
            prices = provider.get_prices(
                "2024-01-01", "2024-01-04", fields=["open", "close"],
            )

        assert prices.index.names == ["date", "asset"]
        assert list(prices.columns) == ["open", "close"]
        assert prices.loc[(pd.Timestamp("2024-01-02"), "A"), "close"] == 10.5


class TestAsyncPostgresStore:
    def test_init_requires_asyncpg(self):
        with patch("quant_platform.data.providers.postgres_provider.HAS_ASYNCPG", False):
            with pytest.raises(ImportError, match="asyncpg required"):
                AsyncPostgresStore()

    def test_init_with_asyncpg(self):
        with patch("quant_platform.data.providers.postgres_provider.HAS_ASYNCPG", True):
            store = AsyncPostgresStore()
            assert store._pool is None
