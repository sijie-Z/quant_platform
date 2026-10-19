"""LiveEngine state snapshot and replay semantics."""

import pandas as pd
import pytest

from quant_platform.trading.live_engine import LiveEngine, Position


def test_snapshot_round_trip_preserves_full_state():
    engine = LiveEngine()
    engine.state.cash = 123.0
    engine.state.positions = [
        Position("A", "2024-01-01", "2024-03-01", 100, 10.0),
        Position("B", "2024-01-02", "2024-03-02", 200, 20.0),
    ]
    engine.state.equity_peak = 500.0
    engine.state.current_equity = 400.0
    engine.state.total_trades = 7
    engine.state.last_rebalance = "2024-01-01"
    engine.state.next_rebalance = "2024-03-01"
    engine.state.status = "running"
    engine.state.last_processed_date = "2024-01-05"
    engine._day_count = 42

    snapshot = engine.to_snapshot()

    restored = LiveEngine()
    restored.restore_snapshot(snapshot)

    assert restored.to_snapshot() == snapshot
    assert restored.state.positions[1].asset == "B"
    assert restored._day_count == 42


def test_unknown_snapshot_version_is_rejected():
    with pytest.raises(ValueError, match="snapshot version"):
        LiveEngine().restore_snapshot({"version": 99})


@pytest.fixture
def engine_with_data():
    dates = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    prices = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
    engine = LiveEngine()
    engine.load_history(returns, prices)
    return engine, dates


def test_processed_date_is_not_replayed(engine_with_data):
    engine, dates = engine_with_data
    engine.state.last_processed_date = str(dates[2])[:10]

    result = engine.run_once(dates[2])

    assert result["status"] == "already_processed"
    assert engine._day_count == 0


def test_killed_engine_stays_killed(engine_with_data):
    engine, dates = engine_with_data
    engine.state.status = "killed"

    result = engine.run_once(dates[0])

    assert result["action"] == "killed"
    assert engine._day_count == 0


def test_run_once_advances_the_watermark(engine_with_data):
    engine, dates = engine_with_data

    engine.run_once(dates[0])

    assert engine.state.last_processed_date == str(dates[0])[:10]
    # A second call for the same day is a no-op.
    assert engine.run_once(dates[0])["status"] == "already_processed"
