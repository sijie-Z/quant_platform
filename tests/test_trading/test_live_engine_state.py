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


def test_expired_positions_are_sold_instead_of_disappearing():
    """Issue #3: the old filter dropped expired positions without crediting cash."""
    dates = pd.bdate_range("2024-01-01", periods=200)
    returns = pd.DataFrame(0.0, index=dates, columns=["A"])
    prices = pd.DataFrame(20.0, index=dates, columns=["A"])

    engine = LiveEngine()
    engine.load_history(returns, prices)
    exit_day = engine._rebalance_dates[0]

    engine.state.cash = 0.0
    engine.state.equity_peak = 0.0
    engine.state.positions = [
        Position("A", "2024-01-01", str(exit_day)[:10], 100, 10.0),
    ]

    engine.run_once(exit_day)

    assert engine.state.positions == []
    assert engine.state.cash == pytest.approx(100 * 20.0 * (1 - 0.0015))


def _alternating_panel(n_dates=200, n_assets=30):
    import numpy as np

    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    rets = np.tile(
        np.array([0.01, -0.01] * (n_dates // 2 + 1))[:n_dates, None],
        (1, n_assets),
    )
    returns = pd.DataFrame(rets, index=dates, columns=assets)
    prices = pd.DataFrame(20.0, index=dates, columns=assets)
    return returns, prices


def test_the_latest_date_can_rebalance():
    """The daemon only ever calls `run_once(index[-1])`, which was never in
    the precomputed `[HOLD_H::HOLD_H]` schedule, so it never traded."""
    returns, prices = _alternating_panel()
    engine = LiveEngine()
    engine.load_history(returns, prices)

    result = engine.run_once(returns.index[-1])

    assert result["action"].startswith("enter_")
    assert engine.state.positions


def test_the_signal_ignores_the_unrealized_last_return():
    """`returns[t]` is the forward return t -> t+1; a signal computed from it
    at t is look-ahead. Only the last row is changed here."""
    returns1, prices = _alternating_panel()
    returns2 = returns1.copy()
    returns2.iloc[-1] = 0.5

    outcomes = []
    for returns in (returns1, returns2):
        engine = LiveEngine()
        engine.load_history(returns, prices)
        engine.run_once(returns.index[-1])
        outcomes.append([
            (p.asset, p.shares, p.entry_price) for p in engine.state.positions
        ])

    assert outcomes[0] == outcomes[1]


def test_batch_replay_rebalances_every_hold_period():
    returns, prices = _alternating_panel()
    engine = LiveEngine()
    engine.load_history(returns, prices)

    results = engine.run()

    entries = [r for r in results if r["action"].startswith("enter_")]
    assert len(entries) == 2  # indices 80 and 160 of 200
