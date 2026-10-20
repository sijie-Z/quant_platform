"""Live-signal parity: the CLI runner and the web engine must agree.

`LiveRunner` (``main.py trade``) and `LiveTradingEngine` (web API) used to
score symbols with two different composites: a 4-factor weighted raw-return
score on the CLI side and a 7-factor equal-weight cross-sectional percentile
on the engine side. With the same price history they now produce exactly the
same Series and the same ranking.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from quant_platform.core.store import Store
from quant_platform.risk.circuit_breaker import RiskMonitor
from quant_platform.trading.broker import OrderSide, SimulatedBroker
from quant_platform.trading.engine import LiveTradingEngine
from quant_platform.trading.live_runner import LiveRunner
from quant_platform.trading.signals import (
    MIN_HISTORY,
    SIGNAL_HISTORY_SESSIONS,
    rank_symbols,
)


def _panel(
    n_days: int = 180,
    n_assets: int = 12,
    seed: int = 11,
) -> pd.DataFrame:
    """Synthetic price history on canonical symbol labels."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    symbols = [f"{600000 + i}.SH" for i in range(n_assets)]
    drift = rng.normal(0.0004, 0.0002, n_assets)
    steps = rng.normal(0, 0.015, (n_days, n_assets)) + drift
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=symbols,
    )


@pytest.fixture
def runner(tmp_path) -> LiveRunner:
    return LiveRunner(
        broker_type="simulated",
        initial_cash=1_000_000,
        dual_track=False,
        risk_monitor=RiskMonitor(),
        store=Store(str(tmp_path / "runner.db")),
    )


@pytest.fixture
def engine(tmp_path) -> LiveTradingEngine:
    return LiveTradingEngine(
        broker=SimulatedBroker(1_000_000),
        store=Store(str(tmp_path / "engine.db")),
        risk_monitor=RiskMonitor(),
    )


class TestIdenticalScores:
    def test_same_history_produces_the_same_series(self, runner, engine):
        panel = _panel()

        runner_scores = runner._score_prices(panel)
        engine_scores = engine._score_prices(panel)

        pd.testing.assert_series_equal(
            runner_scores, engine_scores, rtol=1e-12, atol=1e-12,
        )

    def test_same_history_produces_the_same_ranking(self, runner, engine):
        panel = _panel()
        runner_scores = runner._score_prices(panel)
        engine_scores = engine._score_prices(panel)

        assert rank_symbols(runner_scores) == rank_symbols(engine_scores)
        assert [code for code, _ in rank_symbols(
            runner_scores, limit=5, min_score=0.3,
        )] == [code for code, _ in rank_symbols(
            engine_scores, limit=5, min_score=0.3,
        )]

    def test_runner_accumulated_history_matches_the_engine_panel(
        self, runner, engine,
    ):
        """The runner's own buffer path (mapping of lists) and the engine's
        DataFrame path must agree on the same history."""
        panel = _panel()
        runner.set_universe(list(panel.columns))
        runner.set_price_history({
            code: panel[code].tolist() for code in panel.columns
        })

        pd.testing.assert_series_equal(
            runner._score_prices(), engine._score_prices(panel),
            rtol=1e-12, atol=1e-12,
        )

    def test_tied_cross_section_ranks_identically(self, runner, engine):
        panel = _panel(n_assets=11)
        panel["600099.SH"] = panel["600000.SH"].to_numpy()

        runner_scores = runner._score_prices(panel)
        engine_scores = engine._score_prices(panel)
        assert runner_scores["600099.SH"] == runner_scores["600000.SH"]

        runner_ranked = rank_symbols(runner_scores)
        assert runner_ranked == rank_symbols(engine_scores)
        tied = [code for code, score in runner_ranked
                if score == runner_scores["600000.SH"]]
        assert tied == sorted(tied), "ties must be ordered by symbol"

    def test_ordering_is_stable_across_repeated_calls(self, runner, engine):
        panel = _panel()
        first = rank_symbols(runner._score_prices(panel))
        second = rank_symbols(engine._score_prices(panel))
        third = rank_symbols(runner._score_prices(panel))

        assert first == second == third

    def test_selected_universe_matches_end_to_end(self, runner, engine):
        """The full signal paths pick the same BUY set from one history."""
        panel = _panel()
        last_prices = panel.iloc[-1].to_dict()

        runner.set_universe(list(panel.columns))
        # History strictly before the session: set_prices appends the current
        # close, so including it here would double-count the decision row.
        runner.set_price_history({
            code: panel[code].tolist()[:-1] for code in panel.columns
        })
        runner.set_prices(last_prices)
        runner_codes = {
            signal["code"]
            for signal in runner._generate_signals()
            if signal["side"] == "buy"
        }

        engine._price_history = {
            code: panel[code].tolist() for code in panel.columns
        }
        engine._current_prices = dict(last_prices)
        engine._price_last_seen = {
            code: time.monotonic() for code in panel.columns
        }
        engine._last_fetch_codes = set(panel.columns)
        engine_codes = {
            signal.code
            for signal in engine._generate_signals(positions=[])
            if signal.side == OrderSide.BUY
        }

        assert runner_codes, "the historical panel must produce buys"
        assert runner_codes == engine_codes


class TestInsufficientHistory:
    def test_both_paths_exclude_symbols_without_enough_history(
        self, runner, engine,
    ):
        short = _panel(n_days=MIN_HISTORY - 1)

        runner_scores = runner._score_prices(short)
        engine_scores = engine._score_prices(short)

        assert runner_scores.isna().all()
        assert engine_scores.isna().all()
        assert rank_symbols(runner_scores) == []
        assert rank_symbols(engine_scores) == []

    def test_short_symbol_is_excluded_on_both_paths(self, runner, engine):
        panel = _panel()
        short_code = panel.columns[-1]
        # The symbol only has 10 observed closes; mask the rest of its history
        # so the engine's DataFrame path sees the same short symbol.
        masked = panel.copy()
        masked.loc[masked.index[:-10], short_code] = np.nan
        history = {
            code: masked[code].dropna().tolist() for code in masked.columns
        }

        runner_scores = runner._score_prices(history)
        engine_scores = engine._score_prices(masked)

        assert np.isnan(runner_scores[short_code])
        assert np.isnan(engine_scores[short_code])
        assert runner_scores.drop(short_code).notna().all()
        assert engine_scores.drop(short_code).notna().all()
        # The short symbol must not change the scores of complete symbols.
        pd.testing.assert_series_equal(
            runner_scores.drop(short_code),
            engine_scores.drop(short_code),
            rtol=1e-12, atol=1e-12,
        )
        pd.testing.assert_series_equal(
            runner_scores.drop(short_code),
            engine._score_prices(panel.drop(columns=[short_code])),
            rtol=1e-12, atol=1e-12,
        )


class TestRunnerSignalWindow:
    def test_simulated_run_preloads_at_least_126_sessions(self, tmp_path):
        """A cold replay would never score 3/6-month momentum; `run()` must
        warm the shared window before the first traded session."""
        runner = LiveRunner(
            broker_type="simulated",
            initial_cash=1_000_000,
            dual_track=False,
            risk_monitor=RiskMonitor(),
            store=Store(str(tmp_path / "runner.db")),
        )
        runner.set_universe(["600519", "000858", "000001"])

        runner.run(days=1, seed=3)

        lengths = [len(history) for history in runner._price_history.values()]
        assert lengths, "no price history was recorded"
        assert min(lengths) >= 126, (
            f"signal history window too short for six-month momentum: {lengths}"
        )
        assert max(lengths) <= SIGNAL_HISTORY_SESSIONS

    def test_history_provider_preloads_the_signal_window(self, tmp_path):
        """The runner takes its history from the configured daily-close feed
        (a real broker has only one quote per cycle, not 127 sessions)."""
        panel = _panel(n_days=SIGNAL_HISTORY_SESSIONS, n_assets=3)
        symbols = ["600519.SH", "000858.SZ", "000001.SZ"]
        panel.columns = symbols

        runner = LiveRunner(
            broker_type="simulated",  # provider mechanics only; no orders sent
            initial_cash=1_000_000,
            dual_track=False,
            risk_monitor=RiskMonitor(),
            store=Store(str(tmp_path / "runner.db")),
        )
        runner.set_universe(symbols)
        runner.set_history_provider(
            lambda codes, sessions: {
                code: panel[code].tolist()[-sessions:] for code in codes
            },
        )
        runner._load_signal_history()

        for code in symbols:
            assert len(runner._price_history[code]) == SIGNAL_HISTORY_SESSIONS

        scores = runner._score_prices()
        assert scores.notna().all()
        pd.testing.assert_series_equal(
            scores, runner._score_prices(panel), rtol=1e-12, atol=1e-12,
        )

    def test_real_broker_run_trades_from_provider_history(
        self, tmp_path, monkeypatch,
    ):
        """The one-day real-broker path must get its window from the feed;
        without it no score can exist (it must not invent one)."""
        import quant_platform.trading.live_runner as live_runner_module

        monkeypatch.setattr(
            live_runner_module, "is_trading_day", lambda day: True,
        )
        panel = _panel(n_days=SIGNAL_HISTORY_SESSIONS)
        symbols = list(panel.columns)

        runner = LiveRunner(
            broker_type="simulated",  # exercised through the real-broker branch
            initial_cash=1_000_000,
            dual_track=False,
            risk_monitor=RiskMonitor(),
            store=Store(str(tmp_path / "runner.db")),
        )
        runner._is_simulated = False
        runner.set_universe(symbols)
        runner.set_price_provider(
            lambda codes: {code: float(panel[code].iloc[-1]) for code in codes},
        )
        runner.set_history_provider(
            # History ends at the previous session; set_prices appends today's
            # quote, so including it would count the decision row twice.
            lambda codes, sessions: {
                code: panel[code].tolist()[:-1][-sessions:]
                for code in codes
            },
        )

        report = runner.run(days=1)

        assert report.days_traded == 1
        assert report.total_orders > 0, (
            "the provider history produced no signals on the live path"
        )
        assert report.status == "completed"

    def test_engine_quote_buffer_matches_runner_history(self, runner, engine):
        """Exercise the engine's real buffer path (per-fetch append + cap) and
        compare it with the runner's preloaded history."""
        from types import SimpleNamespace

        panel = _panel(n_days=140)

        class _RowAtATimeMarket:
            def __init__(self, panel):
                self._panel = panel
                self._row = 0

            def get_quotes(self, codes):
                row = self._panel.iloc[self._row]
                self._row += 1
                return [
                    SimpleNamespace(
                        code=code, price=float(row[code]),
                        change_pct=0.0, volume=0.0,
                    )
                    for code in codes
                ]

        engine._rt_market = _RowAtATimeMarket(panel)
        engine.set_universe(list(panel.columns))
        for _ in range(len(panel)):
            engine._fetch_prices()

        engine_history = {
            code: list(values) for code, values in engine._price_history.items()
        }
        assert engine_history, "the engine recorded no quotes"
        assert max(len(values) for values in engine_history.values()) == (
            SIGNAL_HISTORY_SESSIONS
        )

        runner.set_universe(list(panel.columns))
        runner.set_price_history({
            code: values[-SIGNAL_HISTORY_SESSIONS:]
            for code, values in engine_history.items()
        })

        pd.testing.assert_series_equal(
            engine._score_prices(
                pd.DataFrame(engine_history),
            ),
            runner._score_prices(),
            rtol=1e-12, atol=1e-12,
        )
