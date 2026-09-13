"""Regression tests for the vectorized backtest engine."""

import pandas as pd
import pytest
from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.engine import BacktestEngine


class TestRebalanceDates:
    def test_weekly_grouping_across_years(self):
        dates = pd.DatetimeIndex([
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
            "2025-01-08", "2025-01-09", "2025-01-10",
        ])
        engine = BacktestEngine(rebalance_frequency="weekly")
        result = engine._get_rebalance_dates(dates)

        assert len(result) == 4
        assert result == sorted(result)
        assert pd.Timestamp("2024-01-05") in result
        assert pd.Timestamp("2024-01-12") in result
        assert pd.Timestamp("2025-01-03") in result
        assert pd.Timestamp("2025-01-10") in result


class TestSimulatePnl:
    def test_initial_rebalance_charges_cost_and_records_turnover(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        target = pd.Series([0.5, 0.5], index=["A", "B"])

        engine = BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency="monthly",
            cost_model=CostModel(commission=0.001, stamp_tax=0, slippage=0),
        )
        engine.weights_history = {dates[1]: target}
        engine._simulate_pnl(returns)

        # One-sided turnover is 0.5 (cash -> fully invested), so the two-way
        # traded value is 1.0 and the cost is 1.0 * commission = 0.001.
        # This used to assert 999_500 -- the half-charged figure produced by
        # passing one-sided turnover to a cost model that expects two-way.
        assert engine.portfolio_values.iloc[-1] == pytest.approx(999_000.0)
        assert engine.turnover_history is not None
        assert len(engine.turnover_history) == 1
        # The recorded turnover stays one-sided: it is a metric, not the base
        # the cost model is fed.
        assert engine.turnover_history.iloc[0] == pytest.approx(0.5)
