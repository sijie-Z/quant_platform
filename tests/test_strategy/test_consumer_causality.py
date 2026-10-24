"""Causal-prefix regressions for the live-sim consumers (D16).

``DataPipeline.returns[t]`` is the forward return t -> t+1.  The reversal
paper trader, the regime router and the execution adapter used to feed that
panel directly into ``rolling(H)`` as if it were trailing history (and the
reversal trader also used a full-sample volatility quantile).  These tests
pin the fix:

* truncating the input at date t must not change any signal, volatility
  threshold or realized P&L computed at or before t;
* the signal at t must reflect only returns completed by the close of t
  (checked with crafted future moves that would flip the selection);
* the adapter must apply ``returns[prev_date]`` to the holdings carried from
  ``prev_date`` to ``date``, not ``returns[date]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.data.pipeline import realized_returns
from quant_platform.regime_router.adapter import ExecutionAdapter
from quant_platform.regime_router.router import RegimeRouterStub
from quant_platform.trading import reversal_paper_trader as rpt
from quant_platform.trading.reversal_paper_trader import (
    ReversalPaperTrader,
    compound,
)


def make_returns(n_dates: int = 140, n_assets: int = 6, seed: int = 11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates, name="date")
    assets = [f"{i:06d}.SZ" for i in range(1, n_assets + 1)]
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.012, (n_dates, n_assets)),
        index=dates,
        columns=assets,
    )
    # Pipeline convention: the final row has no t+1.
    returns.iloc[-1] = np.nan
    return returns


def prices_from_returns(returns: pd.DataFrame, start: float = 100.0):
    return start * (1.0 + returns.fillna(0.0)).cumprod()


# ---------------------------------------------------------------------------
# Reversal paper trader
# ---------------------------------------------------------------------------


class TestReversalPaperTraderIsCausal:
    def test_signal_and_vol_inputs_are_prefix_stable(self):
        returns = make_returns()
        trader = ReversalPaperTrader()
        cut = 100

        past_full, vol_full, thr_full = trader.compute_signals(
            returns, signal_h=5
        )
        past_cut, vol_cut, thr_cut = trader.compute_signals(
            returns.iloc[: cut + 1], signal_h=5
        )

        pd.testing.assert_frame_equal(
            past_full.iloc[: cut + 1], past_cut, obj="past return"
        )
        pd.testing.assert_series_equal(
            vol_full.iloc[: cut + 1], vol_cut, obj="market vol"
        )
        pd.testing.assert_series_equal(
            thr_full.iloc[: cut + 1], thr_cut, obj="vol threshold"
        )

    def test_signal_uses_completed_returns_only(self):
        """A future move must not enter today's 'past H-day return'."""
        dates = pd.bdate_range("2024-01-02", periods=6, name="date")
        returns = pd.DataFrame(
            {
                "A": [0.02, -0.10, 0.50, 0.0, 0.0, 0.0],
                "B": [0.01, 0.02, -0.20, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        trader = ReversalPaperTrader()
        past_ret, _, _ = trader.compute_signals(returns, signal_h=2)

        # At date index 2 the completed window is returns[0:2].
        expected_a = compound(np.array([0.02, -0.10]))
        expected_b = compound(np.array([0.01, 0.02]))
        assert past_ret.iloc[2]["A"] == pytest.approx(expected_a)
        assert past_ret.iloc[2]["B"] == pytest.approx(expected_b)

        # The unshifted panel would have included the future returns[2]
        # (+50% for A, -20% for B) and flipped the winner/loser ordering.
        unshifted = returns.rolling(2, min_periods=2).apply(compound, raw=True)
        assert unshifted.iloc[2]["A"] > unshifted.iloc[2]["B"]
        assert past_ret.iloc[2]["A"] < past_ret.iloc[2]["B"]

    def test_vol_threshold_is_trailing_not_full_sample(self):
        """A late volatility regime must not move an early threshold."""
        dates = pd.bdate_range("2022-01-03", periods=200, name="date")
        assets = [f"{i:06d}.SZ" for i in range(1, 5)]
        rng = np.random.default_rng(7)

        returns = pd.DataFrame(
            rng.normal(0.0, 0.002, (200, 4)), index=dates, columns=assets
        )
        returns.iloc[120:] = rng.normal(0.0, 0.06, (80, 4))
        returns.iloc[-1] = np.nan

        trader = ReversalPaperTrader()
        _, market_vol, vol_thresh = trader.compute_signals(returns, signal_h=5)
        cut = 100

        # Expanding quantile at `cut` equals the quantile of the values seen
        # so far ...
        expected = market_vol.iloc[: cut + 1].dropna().quantile(rpt.VOL_PERCENTILE)
        assert vol_thresh.iloc[cut] == pytest.approx(expected)

        # ... and differs from the full-sample quantile, which would have
        # imported the late high-volatility regime into an early decision.
        full_sample = market_vol.dropna().quantile(rpt.VOL_PERCENTILE)
        assert vol_thresh.iloc[cut] < full_sample * 0.5

    def test_prefix_run_matches_on_the_prefix(self, monkeypatch):
        """End-to-end: trades and equity at rebalances <= cut are unchanged."""
        monkeypatch.setattr(rpt, "SIGNAL_H", 5)
        monkeypatch.setattr(rpt, "HOLD_H", 10)
        returns = make_returns(n_dates=130)
        prices = prices_from_returns(returns)
        cut_idx = 75

        full = ReversalPaperTrader()
        full.load_data(returns, prices)
        full.run()

        prefix = ReversalPaperTrader()
        prefix.load_data(returns.iloc[: cut_idx + 1], prices.iloc[: cut_idx + 1])
        prefix.run()

        cut_str = str(returns.index[cut_idx])[:10]
        # The final snapshot/trade of each run is the artificial liquidation at
        # that run's own last date; the prefix's last date is the cut itself,
        # so drop it to compare like with like.
        full_equity = [
            e for e in full.equity_curve[:-1] if e["date"] <= cut_str
        ]
        prefix_equity = [
            e for e in prefix.equity_curve[:-1] if e["date"] <= cut_str
        ]
        assert full_equity == prefix_equity
        assert len(full_equity) > 3, "fixture must exercise several rebalances"

        full_trades = [t for t in full.trades[:-1] if t.date <= cut_str]
        prefix_trades = [t for t in prefix.trades[:-1] if t.date <= cut_str]
        assert full_trades == prefix_trades
        assert any(t.action in ("enter", "exit") for t in full_trades)


# ---------------------------------------------------------------------------
# Regime router
# ---------------------------------------------------------------------------


class TestRegimeRouterIsCausal:
    def test_signal_is_prefix_stable(self):
        returns = make_returns()
        router = RegimeRouterStub()
        cut = 80

        full = router.generate_signal(returns, signal_h=10)
        prefix = router.generate_signal(returns.iloc[: cut + 1], signal_h=10)

        pd.testing.assert_frame_equal(full.iloc[: cut + 1], prefix)

    def test_signal_uses_completed_returns_only(self):
        dates = pd.bdate_range("2024-01-02", periods=6, name="date")
        returns = pd.DataFrame(
            {
                "A": [0.02, -0.10, 0.50, 0.0, 0.0, 0.0],
                "B": [0.01, 0.02, -0.20, 0.0, 0.0, 0.0],
            },
            index=dates,
        )
        router = RegimeRouterStub()

        signal = router.generate_signal(returns, signal_h=2)
        # Reversal: the completed trailing loser (A, -8.2%) ranks above the
        # completed winner (B, +3.0%).
        assert signal.iloc[2]["A"] > signal.iloc[2]["B"]

        # Without the shift the future move (A +50%, B -20%) flips it.
        past = returns.rolling(2, min_periods=2).apply(compound, raw=True)
        unshifted_signal = -(past.rank(axis=1, pct=True)) - 0.5
        assert unshifted_signal.iloc[2]["A"] < unshifted_signal.iloc[2]["B"]


# ---------------------------------------------------------------------------
# Execution adapter
# ---------------------------------------------------------------------------


class TestExecutionAdapterIsCausal:
    def test_returns_are_prefix_stable(self):
        returns = make_returns(n_dates=60, n_assets=8, seed=5)
        prices = prices_from_returns(returns)
        rebalances = [returns.index[i] for i in (10, 20, 30, 40)]
        cut_idx = 35

        adapter = ExecutionAdapter(signal_h=5, hold_h=10, rebalance_freq_days=10)
        full = adapter.run(returns, prices, rebalance_dates=rebalances)

        prefix_adapter = ExecutionAdapter(
            signal_h=5, hold_h=10, rebalance_freq_days=10
        )
        prefix = prefix_adapter.run(
            returns.iloc[: cut_idx + 1],
            prices.iloc[: cut_idx + 1],
            rebalance_dates=rebalances,
        )

        pd.testing.assert_series_equal(
            full.reindex(prefix.index), prefix, obj="adapter daily returns"
        )

    def test_holding_return_is_the_realized_return_of_the_holding_day(self):
        """Weights set at t earn ``returns[t]`` on the next day, not
        ``returns[next_day]`` (which is the following day's future return)."""
        dates = pd.bdate_range("2024-01-02", periods=12, name="date")
        values = [0.01, -0.02, 0.03, 0.04, -0.05, 0.06,
                  0.07, -0.08, 0.09, 0.10, -0.11, 0.12]
        returns = pd.DataFrame({"A": values}, index=dates)
        prices = prices_from_returns(returns)

        adapter = ExecutionAdapter(
            signal_h=3, hold_h=10, rebalance_freq_days=10
        )
        port_returns = adapter.run(
            returns, prices, rebalance_dates=[dates[3], dates[8]]
        )

        # weights from dates[3] are held into dates[4]; the realized return
        # over that holding day is returns[dates[3]].
        assert port_returns.loc[dates[4]] == pytest.approx(
            returns.loc[dates[3], "A"]
        )
        assert returns.loc[dates[3], "A"] != returns.loc[dates[4], "A"]

    def test_realized_return_helper_matches_pipeline_convention(self):
        returns = make_returns(n_dates=20, n_assets=2)
        shifted = realized_returns(returns)
        assert shifted.iloc[0].isna().all()
        pd.testing.assert_frame_equal(
            shifted.iloc[1:].reset_index(drop=True),
            returns.iloc[:-1].reset_index(drop=True),
        )
