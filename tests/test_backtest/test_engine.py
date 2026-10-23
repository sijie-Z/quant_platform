"""Regression tests for the vectorized backtest engine."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.backtest.cost_model import CostModel
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.portfolio.constraints import PortfolioConstraints


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
        # traded value is 1.0 and the cost is 1.0 * commission = 0.001 plus
        # the B5 both-sides transfer fee (1e-5 of the 1.0 notional) = 0.00101.
        # (The old figure passed one-sided turnover to a cost model that
        # expects two-way.)
        assert engine.portfolio_values.iloc[-1] == pytest.approx(998_990.0)
        assert engine.turnover_history is not None
        assert len(engine.turnover_history) == 1
        # The recorded turnover stays one-sided: it is a metric, not the base
        # the cost model is fed.
        assert engine.turnover_history.iloc[0] == pytest.approx(0.5)

    def test_execution_clamps_turnover_to_the_configured_cap(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        engine = BacktestEngine(
            rebalance_frequency="monthly",
            constraints=PortfolioConstraints(max_turnover=0.30),
        )
        # First rebalance deploys from cash (exempt from the cap); the second
        # swaps half the portfolio, which must be clamped to 30%.
        engine.weights_history = {
            dates[1]: pd.Series({"A": 0.5, "B": 0.5}),
            dates[3]: pd.Series({"A": 1.0, "B": 0.0}),
        }

        engine._simulate_pnl(returns)

        assert engine.turnover_history is not None
        assert engine.turnover_history.iloc[0] == pytest.approx(0.5)
        assert engine.turnover_history.iloc[-1] == pytest.approx(0.30)

    def test_weights_drift_between_rebalances(self):
        """A held portfolio must not be silently rebalanced every day."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        returns = pd.DataFrame(
            {"A": [0.0, 0.5, 0.5], "B": [0.0, 0.0, 0.0]}, index=dates,
        )
        engine = BacktestEngine(
            rebalance_frequency="monthly",
            # Zero-cost for the drift assertion: the B5 defaults (5 CNY floor,
            # both-sides transfer fee) would charge the initial deployment.
            cost_model=CostModel(
                commission=0, stamp_tax=0, slippage=0,
                min_commission=0, transfer_fee=0,
            ),
        )
        engine.weights_history = {dates[0]: pd.Series({"A": 0.5, "B": 0.5})}

        engine._simulate_pnl(returns)

        # day 1: 50/50 -> +25%. A's weight drifts to 60%, so day 2 earns 30%.
        # Daily rebalancing would have produced 25% again.
        assert engine.daily_returns.iloc[1] == pytest.approx(0.25)
        assert engine.daily_returns.iloc[2] == pytest.approx(0.30)

    def test_equal_weight_benchmark_uses_per_date_universe(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B", "C"])
        returns.loc[dates[:60], "C"] = np.nan
        returns.loc[dates[60]:, "C"] = 0.05

        engine = BacktestEngine(benchmark="equal_weight")
        benchmark = engine._compute_equal_weight_benchmark(returns)

        assert benchmark.loc[dates[59]] == pytest.approx(0.0)
        # The late-listing asset enters on its first observable date; the old
        # whole-sample filter dropped it from every date.
        assert benchmark.loc[dates[60]] == pytest.approx(0.05 / 3)


class TestSummaryReportsOptimizerFallbacks:
    def test_fallback_count_and_effective_optimizer_are_exposed(self):
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2023-01-02", periods=120)
        assets = [f"S{i:02d}" for i in range(30)]
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (120, 30)), axis=0)),
            index=dates, columns=assets,
        )
        returns = prices.pct_change(fill_method=None).shift(-1)
        signal = pd.DataFrame(
            rng.normal(size=(120, 30)), index=dates, columns=assets,
        )
        sector_map = pd.Series("Tech", index=assets)

        class FallbackOptimizer:
            fallback_count = 0
            last_fallback_reason = None

            def optimize(self, signal=None, **_kwargs):
                self.fallback_count += 1
                self.last_fallback_reason = "stub infeasible"
                return pd.Series(1.0 / len(signal), index=signal.index)

        engine = BacktestEngine(rebalance_frequency="monthly")
        stub = FallbackOptimizer()
        engine._get_optimizer = lambda: stub

        result = engine.run(
            signal=signal, prices=prices, returns=returns,
            benchmark_returns=None, sector_map=sector_map,
        )
        summary = result["summary"]

        assert summary["n_rebalances"] > 0
        # M13: `n_rebalances` counts executed rebalances only, while the last
        # month-end signal has no following trading day: it is optimized (and
        # falls back) but never executed. Fallbacks are counted at optimization
        # time, so the rate covers every optimized rebalance.
        assert summary["n_rebalances"] == len(engine.weights_history)
        assert summary["n_optimizer_fallbacks"] == len(engine.optimizer_outcomes)
        assert summary["n_optimizer_fallbacks"] == (
            summary["n_rebalances"] + len(engine.unexecuted_signals)
        )
        assert summary["optimizer_fallback_rate"] == pytest.approx(
            summary["n_optimizer_fallbacks"] / summary["n_rebalances"]
        )
        assert summary["effective_optimizer"].endswith("+equal_weight_fallback")
        assert len(engine.unexecuted_signals) == 1
        assert sum(
            outcome["executed"] for outcome in engine.optimizer_outcomes.values()
        ) == summary["n_rebalances"]


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

        def capture(ret_window, method, lookback, ewma_half_life=63):
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
        # M13 keeps only executed rebalances in `weights_history`; a final
        # rebalance with no following trading day is still optimized, so use
        # the optimizer-outcome record for "one covariance per rebalance".
        return sorted(engine.optimizer_outcomes), windows, returns

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
        # cost's own date is checked separately below. The B5 defaults (5 CNY
        # floor, both-sides transfer fee) are disabled explicitly.
        engine = BacktestEngine(
            rebalance_frequency="monthly",
            optimizer="equal_weight",
            execution_timing=timing,
            cost_model=cost_model or CostModel(
                commission=0, stamp_tax=0, slippage=0,
                min_commission=0, transfer_fee=0,
            ),
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
            cost_model=CostModel(
                commission=0.001, stamp_tax=0, slippage=0,
                min_commission=0, transfer_fee=0,
            ),
        )

        engine._simulate_pnl(returns)

        signal_date = returns.index[self.SIGNAL_DAY]
        exec_date = returns.index[self.SIGNAL_DAY + 1]
        assert engine.daily_returns.loc[signal_date] == pytest.approx(0.0)
        # One-sided turnover 0.5 -> two-way 1.0 -> 0.001 of capital.
        assert engine.daily_returns.loc[exec_date] == pytest.approx(-0.001)


class TestStampTaxDirection:
    """Stamp tax is a sell-side tax; the engine used to charge it on buys.

    `_simulate_pnl` passed the total two-way traded value with `is_sell=None`,
    so `CostModel` assumed exactly half of it was sells: a 100%-cash portfolio
    building its initial position paid 0.05% stamp tax on buys it never owed,
    and a full liquidation paid only half of the 0.1% it did owe.
    """

    @staticmethod
    def _engine() -> BacktestEngine:
        return BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency="monthly",
            # Isolate the stamp tax: no commission floor, no transfer fee.
            cost_model=CostModel(
                commission=0.0, stamp_tax=0.001, slippage=0.0,
                min_commission=0, transfer_fee=0,
            ),
            # Disable the turnover cap so the liquidation below is complete.
            constraints=PortfolioConstraints(max_turnover=1.0),
        )

    @staticmethod
    def _flat_returns():
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame(0.0, index=dates, columns=["A", "B"]), dates

    def test_building_from_cash_pays_no_stamp_tax(self):
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {dates[1]: pd.Series({"A": 0.5, "B": 0.5})}

        engine._simulate_pnl(returns)

        # The initial deployment is 100% buys: no stamp tax is due, so the
        # day's return is exactly flat.
        assert engine.daily_returns.loc[dates[2]] == pytest.approx(0.0)
        assert engine.portfolio_values.iloc[-1] == pytest.approx(1_000_000.0)

    def test_full_liquidation_pays_the_full_sell_side_stamp_tax(self):
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {
            dates[1]: pd.Series({"A": 0.5, "B": 0.5}),
            dates[3]: pd.Series({"A": 0.0, "B": 0.0}),
        }

        engine._simulate_pnl(returns)

        # Selling 100% of the portfolio is charged the full 0.1% stamp tax.
        # The old code passed the two-way value with is_sell=None and charged
        # only half of it (-0.0005).
        assert engine.daily_returns.loc[dates[4]] == pytest.approx(-0.001)
        assert engine.portfolio_values.iloc[-1] == pytest.approx(999_000.0)


class TestExecutionDayTradability:
    """B1: the pipeline's flags gate the *execution* day, not the signal day.

    A buy is blocked at limit-up or on a non-tradable (suspension/ST/listing)
    day; a sell is blocked at limit-down or on a non-tradable day. The blocked
    target is carried to the next tradable day, and no turnover or cost is
    booked on the blocked day.
    """

    @staticmethod
    def _engine(cost_model=None) -> BacktestEngine:
        return BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency="monthly",
            optimizer="equal_weight",
            # A liquidation is a one-sided 0.5 turnover; do not clamp it.
            constraints=PortfolioConstraints(max_turnover=1.0),
            cost_model=cost_model or CostModel(
                commission=0, stamp_tax=0, slippage=0,
                min_commission=0, transfer_fee=0,
            ),
        )

    @staticmethod
    def _flat_returns(n=5):
        dates = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame(0.0, index=dates, columns=["A", "B"]), dates

    def test_buy_at_limit_up_is_deferred_and_costs_nothing_that_day(self):
        returns, dates = self._flat_returns()
        engine = self._engine(CostModel(
            commission=0.001, stamp_tax=0, slippage=0,
            min_commission=0, transfer_fee=0,
        ))
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        limit_up = pd.DataFrame(False, index=dates, columns=["A", "B"])
        limit_up.loc[dates[1], "A"] = True

        engine._simulate_pnl(returns, limit_up=limit_up)

        assert engine.daily_returns.loc[dates[1]] == pytest.approx(0.0)
        assert list(engine.turnover_history.index) == [dates[2]]
        assert engine.daily_returns.loc[dates[2]] == pytest.approx(-0.001)

    def test_buy_on_a_suspended_day_is_deferred(self):
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
        tradable.loc[dates[1], "A"] = False

        engine._simulate_pnl(returns, tradable_mask=tradable)

        assert engine.daily_returns.loc[dates[1]] == pytest.approx(0.0)
        assert list(engine.turnover_history.index) == [dates[2]]
        assert engine.turnover_history.iloc[0] == pytest.approx(0.5)

    def test_sell_at_limit_down_is_deferred(self):
        returns, dates = self._flat_returns()
        engine = self._engine(CostModel(
            commission=0.001, stamp_tax=0, slippage=0,
            min_commission=0, transfer_fee=0,
        ))
        engine.weights_history = {
            dates[0]: pd.Series({"A": 1.0, "B": 0.0}),
            dates[2]: pd.Series({"A": 0.0, "B": 0.0}),
        }
        limit_down = pd.DataFrame(False, index=dates, columns=["A", "B"])
        limit_down.loc[dates[3], "A"] = True

        engine._simulate_pnl(returns, limit_down=limit_down)

        # Build executes on dates[1]; the liquidation is blocked on dates[3]
        # (no turnover, no cost) and fills on dates[4].
        assert list(engine.turnover_history.index) == [dates[1], dates[4]]
        assert engine.daily_returns.loc[dates[3]] == pytest.approx(0.0)
        assert engine.daily_returns.loc[dates[4]] == pytest.approx(-0.001)

    def test_a_single_gate_panel_does_not_require_the_others(self):
        """Supplying only limit_up must work; the omitted panels are neutral."""
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        limit_up = pd.DataFrame(False, index=dates, columns=["A", "B"])

        engine._simulate_pnl(returns, limit_up=limit_up)

        assert list(engine.turnover_history.index) == [dates[1]]

    def test_a_deferred_subset_does_not_wipe_the_rest_of_the_book(self):
        """A blocked buy retried on its own must not NaN out executed holdings.

        The retry delta is indexed by the blocked assets only; adding it to
        the full book with plain `+` aligns every other label to NaN and
        silently liquidates the portfolio.
        """
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {dates[0]: pd.Series({"A": 0.5, "B": 0.5})}
        tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
        tradable.loc[dates[1], "A"] = False

        engine._simulate_pnl(returns, tradable_mask=tradable)

        held = engine.weights_history[dates[0]]
        assert held.sum() == pytest.approx(1.0)
        assert held["A"] == pytest.approx(0.5)
        assert held["B"] == pytest.approx(0.5)
        assert float(held.isna().sum()) == 0.0


class TestUniverseBuyGate:
    """R6b: index membership gates buys only.

    A non-member buy is blocked (and retried); an index *exit* must never
    block the sale of a position already held. Suspension still blocks both
    sides through ``tradable_mask``. Omitting ``universe_mask`` is
    bit-identical to an all-True panel.
    """

    @staticmethod
    def _engine() -> BacktestEngine:
        return BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency="monthly",
            optimizer="equal_weight",
            constraints=PortfolioConstraints(max_turnover=1.0),
            cost_model=CostModel(
                commission=0, stamp_tax=0, slippage=0,
                min_commission=0, transfer_fee=0,
            ),
        )

    @staticmethod
    def _flat_returns(n=5):
        dates = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame(0.0, index=dates, columns=["A", "B"]), dates

    def test_non_member_buy_is_blocked_and_retried(self):
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        universe = pd.DataFrame(True, index=dates, columns=["A", "B"])
        universe.loc[dates[1], "A"] = False  # not a constituent on day 1

        engine._simulate_pnl(returns, universe_mask=universe)

        # The buy is rejected on dates[1] and fills on dates[2] once the
        # name is a member again.
        assert list(engine.turnover_history.index) == [dates[2]]
        assert engine.turnover_history.iloc[0] == pytest.approx(0.5)

    def test_member_buy_is_allowed(self):
        returns, dates = self._flat_returns()
        engine = self._engine()
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        universe = pd.DataFrame(True, index=dates, columns=["A", "B"])

        engine._simulate_pnl(returns, universe_mask=universe)

        assert list(engine.turnover_history.index) == [dates[1]]

    def test_index_exit_does_not_block_the_sell(self):
        """The core R6b regression: leaving the index must not trap a name."""
        returns, dates = self._flat_returns(n=6)
        engine = self._engine()
        engine.weights_history = {
            dates[0]: pd.Series({"A": 1.0, "B": 0.0}),  # bought at dates[1]
            dates[2]: pd.Series({"A": 0.0, "B": 0.0}),  # sold at dates[3]
        }
        universe = pd.DataFrame(True, index=dates, columns=["A", "B"])
        universe.loc[dates[2]:, "A"] = False  # A exits the index on day 2

        engine._simulate_pnl(returns, universe_mask=universe)

        # Buy filled while still a member; the exit did not stop the sell.
        assert dates[1] in engine.turnover_history.index
        assert dates[3] in engine.turnover_history.index
        assert engine.weights_history[dates[2]]["A"] == pytest.approx(0.0)

    def test_suspension_blocks_both_sides_even_for_members(self):
        returns, dates = self._flat_returns(n=6)
        universe = pd.DataFrame(True, index=dates, columns=["A", "B"])

        # Buy side: A is suspended on the execution day -> deferred.
        buy_engine = self._engine()
        buy_engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        tradable_buy = pd.DataFrame(True, index=dates, columns=["A", "B"])
        tradable_buy.loc[dates[1], "A"] = False
        buy_engine._simulate_pnl(
            returns, tradable_mask=tradable_buy, universe_mask=universe,
        )
        assert list(buy_engine.turnover_history.index) == [dates[2]]

        # Sell side: A is suspended when the liquidation would execute.
        sell_engine = self._engine()
        sell_engine.weights_history = {
            dates[0]: pd.Series({"A": 1.0, "B": 0.0}),
            dates[2]: pd.Series({"A": 0.0, "B": 0.0}),
        }
        tradable_sell = pd.DataFrame(True, index=dates, columns=["A", "B"])
        tradable_sell.loc[dates[3], "A"] = False
        sell_engine._simulate_pnl(
            returns, tradable_mask=tradable_sell, universe_mask=universe,
        )
        assert dates[3] not in sell_engine.turnover_history.index
        assert dates[4] in sell_engine.turnover_history.index

    def test_omitting_universe_mask_matches_all_true_panel(self):
        returns, dates = self._flat_returns(n=6)

        def run(**extra):
            engine = self._engine()
            engine.weights_history = {
                dates[0]: pd.Series({"A": 1.0, "B": 0.0}),
                dates[2]: pd.Series({"A": 0.5, "B": 0.5}),
            }
            tradable = pd.DataFrame(True, index=dates, columns=["A", "B"])
            tradable.loc[dates[3], "B"] = False
            engine._simulate_pnl(returns, tradable_mask=tradable, **extra)
            return engine

        without = run()
        with_all_true = run(
            universe_mask=pd.DataFrame(True, index=dates, columns=["A", "B"]),
        )

        pd.testing.assert_series_equal(
            without.daily_returns, with_all_true.daily_returns,
        )
        pd.testing.assert_series_equal(
            without.turnover_history, with_all_true.turnover_history,
        )
        pd.testing.assert_series_equal(
            without.portfolio_values, with_all_true.portfolio_values,
        )


class TestCapacityWiring:
    """R8: the engine hands the optimizer ADV in CNY = volume (shares) x
    raw (unadjusted) close, averaged over 20 sessions, plus the run's AUM
    convention. Without raw prices the cap is refused, never computed on the
    adjusted research panel."""

    N_ASSETS = 12
    N_DAYS = 120

    @staticmethod
    def _inputs():
        rng = np.random.default_rng(5)
        dates = pd.bdate_range("2023-01-02", periods=TestCapacityWiring.N_DAYS)
        assets = [f"S{i:02d}" for i in range(TestCapacityWiring.N_ASSETS)]
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(
                rng.normal(0, 0.01, (len(dates), len(assets))), axis=0,
            )),
            index=dates, columns=assets,
        )
        returns = prices.pct_change(fill_method=None).shift(-1)
        signal = pd.DataFrame(1.0, index=dates, columns=assets)
        volume = pd.DataFrame(
            rng.integers(100_000, 2_000_000, (len(dates), len(assets))),
            index=dates, columns=assets,
        ).astype(float)
        return prices, returns, signal, volume

    @staticmethod
    def _run(constraints, daily_volume, inputs=None, raw_prices=None):
        prices, returns, signal, _volume = inputs or TestCapacityWiring._inputs()

        class RecordingOptimizer:
            fallback_count = 0
            last_fallback_reason = None

            def __init__(self):
                self.calls = []

            def optimize(self, signal=None, **kwargs):
                self.calls.append({**kwargs, "signal": signal})
                return pd.Series(
                    1.0 / len(signal), index=signal.index, dtype=float,
                )

        engine = BacktestEngine(
            initial_capital=10_000_000,
            rebalance_frequency="monthly",
            constraints=constraints,
        )
        recorder = RecordingOptimizer()
        engine._get_optimizer = lambda: recorder
        engine.run(
            signal=signal, prices=prices, returns=returns,
            benchmark_returns=None, sector_map=None,
            daily_volume=daily_volume, raw_prices=raw_prices,
        )
        return engine, recorder, prices, _volume

    @staticmethod
    def _raw_prices(prices):
        """Unadjusted CNY prices distinct from the adjusted research panel."""
        factors = pd.Series(
            np.linspace(1.0, 2.0, prices.shape[1]), index=prices.columns,
        )
        return prices.div(factors, axis=1)

    def test_adv_is_volume_times_raw_close_rolling_20(self):
        """B1: the CNY ADV must use the unadjusted close, never the adjusted
        research panel (which would overstate traded value by the adjustment
        factor)."""
        constraints = PortfolioConstraints(
            max_participation=0.10, max_turnover=1.0,
        )
        inputs = self._inputs()
        adjusted, volume = inputs[0], inputs[3]
        raw = self._raw_prices(adjusted)
        engine, recorder, _, _ = self._run(
            constraints, daily_volume=volume, inputs=inputs, raw_prices=raw,
        )

        expected = (volume * raw).rolling(20, min_periods=5).mean()
        wrong_basis = (volume * adjusted).rolling(20, min_periods=5).mean()
        rdates = list(engine.optimizer_outcomes)
        assert rdates, "the engine never called the optimizer"
        assert len(recorder.calls) == len(rdates)

        for rdate, call in zip(rdates, recorder.calls, strict=True):
            adv = call["adv"]
            assert adv is not None, f"no ADV passed at {rdate.date()}"
            assert call["aum"] == pytest.approx(engine.initial_capital)
            expected_row = expected.loc[rdate].reindex(call["signal"].index)
            pd.testing.assert_series_equal(
                adv.sort_index(), expected_row.sort_index(),
                rtol=1e-12, atol=1e-9,
            )
            # Not the share count, and not the adjusted-price notional.
            assert not np.allclose(
                adv.to_numpy(), volume.loc[rdate].reindex(adv.index).to_numpy(),
            )
            wrong_row = wrong_basis.loc[rdate].reindex(adv.index)
            assert not np.allclose(
                adv.to_numpy(), wrong_row.to_numpy(),
            ), "ADV used the adjusted price panel instead of raw CNY prices"

    def test_capacity_without_raw_prices_warns_and_passes_none(self, caplog):
        """Fail closed: an adjusted price panel is not a CNY price, so the
        capacity cap is disabled (with a warning) rather than computed on a
        wrong basis."""
        constraints = PortfolioConstraints(
            max_participation=0.10, max_turnover=1.0,
        )
        inputs = self._inputs()
        with caplog.at_level("WARNING"):
            engine, recorder, _, _ = self._run(
                constraints, daily_volume=inputs[3], inputs=inputs,
                raw_prices=None,
            )
        assert any(
            "no raw_prices" in record.getMessage()
            for record in caplog.records
        )
        assert recorder.calls
        for call in recorder.calls:
            assert call["adv"] is None
            assert call["aum"] == pytest.approx(engine.initial_capital)

    def test_capacity_off_by_default_passes_none(self):
        inputs = self._inputs()
        constraints = PortfolioConstraints(max_turnover=1.0)
        engine, recorder, _, _ = self._run(
            constraints, daily_volume=inputs[3], inputs=inputs,
        )
        assert recorder.calls
        for call in recorder.calls:
            assert call["adv"] is None
            assert call["aum"] is None

    def test_configured_capacity_without_volume_warns_and_passes_none(self, caplog):
        constraints = PortfolioConstraints(
            max_participation=0.10, max_turnover=1.0,
        )
        with caplog.at_level("WARNING"):
            engine, recorder, _, _ = self._run(
                constraints, daily_volume=None,
            )
        assert any(
            "no daily_volume panel" in record.getMessage()
            for record in caplog.records
        )
        assert recorder.calls
        for call in recorder.calls:
            assert call["adv"] is None
            assert call["aum"] == pytest.approx(engine.initial_capital)


    def test_share_volume_is_converted_to_cny_for_the_cost_model(self):
        """R8 redefined run(daily_volume=...) as a share panel. The cost
        model still expects CNY, so the engine must multiply by the *raw*
        close before handing the row over (the old pass-through would
        overstate the participation rate by 1/price)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        raw_prices = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
        volume = pd.DataFrame(1_000_000.0, index=dates, columns=["A", "B"])

        engine = BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency="monthly",
            cost_model=CostModel(
                commission=0, stamp_tax=0, slippage=0.001,
                slippage_model="proportional", min_commission=0,
                transfer_fee=0,
            ),
            constraints=PortfolioConstraints(max_turnover=1.0),
        )
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        engine._simulate_pnl(returns, daily_volume=volume, raw_prices=raw_prices)

        # Buy 1,000,000 CNY of A against 10,000,000 CNY daily traded value:
        # 10% participation -> slippage 0.001 * sqrt(0.10).
        expected_cost = 0.001 * np.sqrt(1_000_000.0 / (1_000_000.0 * 10.0))
        assert engine.daily_returns.loc[dates[1]] == pytest.approx(
            -expected_cost
        )


    def test_share_volume_without_raw_prices_falls_back_to_fixed_slippage(self, caplog):
        """m5 (review 2): a share panel must never be fed to a CNY cost
        model. Without raw_prices the volume is dropped and the cost model
        falls back to its configured fixed slippage (warning)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        volume = pd.DataFrame(1_000_000.0, index=dates, columns=["A", "B"])

        engine = BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency="monthly",
            cost_model=CostModel(
                commission=0, stamp_tax=0, slippage=0.001,
                slippage_model="proportional", min_commission=0,
                transfer_fee=0,
            ),
            constraints=PortfolioConstraints(max_turnover=1.0),
        )
        engine.weights_history = {dates[0]: pd.Series({"A": 1.0, "B": 0.0})}
        with caplog.at_level("WARNING"):
            engine._simulate_pnl(returns, daily_volume=volume)

        assert any(
            "dropping the volume" in record.getMessage()
            for record in caplog.records
        )
        # Fixed slippage on the 1.0 two-way traded value, not the
        # participation-based proportional cost.
        assert engine.daily_returns.loc[dates[1]] == pytest.approx(-0.001)


class TestWeeklyIsoYearBoundary:
    """B2: weekly rebalances group by ISO year + ISO week, chronologically."""

    def test_late_december_does_not_collide_with_january_week_one(self):
        dates = pd.DatetimeIndex([
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-12",
            "2024-12-30", "2024-12-31",
            "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-10",
        ])
        result = BacktestEngine(
            rebalance_frequency="weekly",
        )._get_rebalance_dates(dates)

        assert result == sorted(result), "rebalance dates must be chronological"
        assert pd.Timestamp("2024-01-05") in result, (
            "the early-January-2024 rebalance collapsed into the next year"
        )
        assert pd.Timestamp("2024-12-31") not in result, (
            "a late-December session became a spurious week-1 rebalance"
        )
        assert pd.Timestamp("2025-01-03") in result

    def test_january_optimizer_never_receives_december_holdings(self):
        dates = pd.bdate_range("2024-01-02", "2025-01-15")
        assets = [f"S{i:02d}" for i in range(12)]
        rng = np.random.default_rng(11)
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(
                rng.normal(0, 0.005, (len(dates), len(assets))), axis=0,
            )),
            index=dates, columns=assets,
        )
        returns = prices.pct_change(fill_method=None).shift(-1)
        signal = pd.DataFrame(
            rng.normal(size=(len(dates), len(assets))),
            index=dates, columns=assets,
        )

        engine = BacktestEngine(
            rebalance_frequency="weekly",
            optimizer="equal_weight",
            covariance_method="sample",
            constraints=PortfolioConstraints(max_weight=1.0, max_turnover=1.0),
        )
        inner = engine._get_optimizer()

        class RecordingOptimizer:
            def __init__(self):
                self.calls = []
                self.fallback_count = 0
                self.last_fallback_reason = None

            def optimize(self, signal=None, **kwargs):
                weights = inner.optimize(signal=signal, **kwargs)
                self.calls.append((kwargs.get("prev_weights"), weights.copy()))
                return weights

        recorder = RecordingOptimizer()
        engine._get_optimizer = lambda: recorder
        engine.run(
            signal=signal, prices=prices, returns=returns,
            benchmark_returns=None,
            sector_map=pd.Series("Tech", index=assets),
        )

        order = list(engine.optimizer_outcomes)
        assert order == sorted(order), "optimizer calls are not chronological"
        assert pd.Timestamp("2024-12-31") not in order
        assert any(d.year == 2024 and d.month == 1 for d in order), (
            "no January-2024 rebalance was processed"
        )

        # Each call receives the previous chronological target as prev_weights,
        # so no January-2024 call can see December-2024 holdings.
        for i, (prev, _target) in enumerate(recorder.calls):
            if i == 0:
                assert prev is None
                continue
            assert order[i - 1] < order[i]
            pd.testing.assert_series_equal(prev, recorder.calls[i - 1][1])
