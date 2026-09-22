"""Regression tests for the vectorized backtest engine."""

import numpy as np
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


class TestCovarianceWindowSeesNoFuture:
    """`returns` carries the forward return close(t) -> close(t+1).

    data/pipeline.py:279 builds it as `close.pct_change().shift(-1)`, so the
    row dated t is only observable once t+1 has closed. At rdate's close --
    the moment the weights are chosen -- the row dated rdate has not happened.
    The covariance window used to end `lookback_end + 1`, which is inclusive
    of exactly that row.
    """

    N_ASSETS = 12
    N_DAYS = 400

    def _run(self, monkeypatch):
        import quant_platform.backtest.engine as engine_mod

        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2022-01-03", periods=self.N_DAYS)
        assets = [f"S{i:02d}" for i in range(self.N_ASSETS)]

        closes = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (self.N_DAYS, self.N_ASSETS)), axis=0)),
            index=dates, columns=assets,
        )
        # The pipeline's convention, including the unobservable last row.
        returns = closes.pct_change(fill_method=None).shift(-1)
        signal = pd.DataFrame(
            rng.normal(size=(self.N_DAYS, self.N_ASSETS)), index=dates, columns=assets,
        )
        sector_map = pd.Series("Tech", index=assets)

        windows = []

        def capture(ret_window, method, lookback):
            windows.append(ret_window)
            return None

        monkeypatch.setattr(engine_mod, "estimate_covariance", capture)

        engine = BacktestEngine(
            rebalance_frequency="monthly",
            optimizer="equal_weight",
            covariance_method="sample",
            covariance_lookback=252,
        )
        engine.run(
            signal=signal, prices=closes, returns=returns,
            benchmark_returns=None, sector_map=sector_map,
        )
        # The engine's own record of which dates produced weights, in order.
        return sorted(engine.weights_history), windows, returns

    def test_window_never_contains_the_rebalance_day(self, monkeypatch):
        rdates, windows, _ = self._run(monkeypatch)

        assert windows, "the engine never estimated a covariance"
        assert len(windows) == len(rdates), "one covariance per rebalance"

        for rdate, window in zip(rdates, windows, strict=True):
            assert rdate not in window.index, (
                f"covariance at {rdate.date()} used the bar dated {rdate.date()}, "
                "which is the return from that close to the next one"
            )
            assert window.index[-1] < rdate, (
                f"covariance at {rdate.date()} ends on {window.index[-1].date()}"
            )

    def test_window_is_the_configured_length(self, monkeypatch):
        rdates, windows, _ = self._run(monkeypatch)

        # The lookback is `covariance_lookback` rows, not one more. Windows
        # fill up as the sample grows and then stay pinned there; they used
        # to reach 253.
        lengths = [len(w) for w in windows]
        assert all(n <= 252 for n in lengths), f"a window exceeded the lookback: {max(lengths)}"
        assert lengths == sorted(lengths), "the window grows then stops growing"
        assert lengths[-1] == 252, "the last window is pinned at the lookback"


class TestExecutionTiming:
    """The signal is computed *from* rdate's close, so it cannot also be
    traded at that same close.

    The engine used to apply the new weights at `date >= rdate` and therefore
    earn `returns.loc[rdate]` -- the move from the previous close to the very
    close the signal was read off. The module docstring, ASHARE_PITFALLS.md
    and the T+1 notes all say execution is at the next trading day's close;
    the code never did that.
    """

    SIGNAL_DAY = 3

    @staticmethod
    def _returns(signal_day_return: float = 0.10):
        """Ten flat days, with one big move on the signal day itself."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        r = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        r.loc[dates[TestExecutionTiming.SIGNAL_DAY], "A"] = signal_day_return
        return r

    @staticmethod
    def _engine(timing: str, returns: pd.DataFrame, cost_model=None) -> BacktestEngine:
        # Zero cost by default: these tests are about *when* a weight takes
        # effect, and a fee would blur the timing signal they assert on. The
        # cost's own date is checked separately below.
        engine = BacktestEngine(
            rebalance_frequency="monthly",
            optimizer="equal_weight",
            execution_timing=timing,
            cost_model=cost_model or CostModel(commission=0, stamp_tax=0, slippage=0),
        )
        engine.weights_history = {
            returns.index[TestExecutionTiming.SIGNAL_DAY]: pd.Series({"A": 1.0, "B": 0.0})
        }
        return engine

    def test_the_signal_days_own_return_is_not_captured(self):
        returns = self._returns()
        engine = self._engine("next_close", returns)

        engine._simulate_pnl(returns)

        assert engine.daily_returns.loc[returns.index[self.SIGNAL_DAY]] == pytest.approx(0.0), (
            "the portfolio earned a move that happened before the decision"
        )
        # Nothing else moved, so the whole run is flat.
        assert engine.daily_returns.abs().sum() == pytest.approx(0.0)

    def test_the_old_timing_did_capture_it(self):
        """Pins `signal_close` to the behaviour being migrated away from, so
        the comparison harness measures the real old semantics."""
        returns = self._returns()
        engine = self._engine("signal_close", returns)

        engine._simulate_pnl(returns)

        assert engine.daily_returns.loc[returns.index[self.SIGNAL_DAY]] == pytest.approx(0.10)

    def test_turnover_is_recorded_on_the_execution_date(self):
        returns = self._returns()
        engine = self._engine("next_close", returns)

        engine._simulate_pnl(returns)

        exec_date = returns.index[self.SIGNAL_DAY + 1]
        assert list(engine.turnover_history.index) == [exec_date]
        assert engine.turnover_history.iloc[0] == pytest.approx(0.5)

    def test_the_old_timing_recorded_it_on_the_signal_date(self):
        returns = self._returns()
        engine = self._engine("signal_close", returns)

        engine._simulate_pnl(returns)

        assert list(engine.turnover_history.index) == [returns.index[self.SIGNAL_DAY]]

    def test_a_signal_with_no_following_day_is_dropped(self):
        """Nothing can be executed on the last bar, and trading it at that bar's
        own close is the assumption being removed."""
        returns = self._returns()
        last = returns.index[-1]
        engine = self._engine("next_close", returns)
        engine.weights_history = {last: pd.Series({"A": 1.0, "B": 0.0})}

        engine._simulate_pnl(returns)

        assert len(engine.turnover_history) == 0
        assert engine.daily_returns.abs().sum() == pytest.approx(0.0)

    def test_an_unknown_timing_is_rejected(self):
        with pytest.raises(ValueError, match="execution_timing"):
            BacktestEngine(execution_timing="same_bar")

    def test_the_transaction_cost_is_charged_on_the_execution_date(self):
        returns = self._returns()
        engine = self._engine(
            "next_close", returns,
            cost_model=CostModel(commission=0.001, stamp_tax=0, slippage=0),
        )

        engine._simulate_pnl(returns)

        signal_date = returns.index[self.SIGNAL_DAY]
        exec_date = returns.index[self.SIGNAL_DAY + 1]
        assert engine.daily_returns.loc[signal_date] == pytest.approx(0.0)
        # One-sided turnover 0.5 -> two-way 1.0 -> 0.001 of capital.
        assert engine.daily_returns.loc[exec_date] == pytest.approx(-0.001)
