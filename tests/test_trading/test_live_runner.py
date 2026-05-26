"""Tests for LiveRunner — production trading runner."""

import numpy as np
import pytest

from quant_platform.trading.live_runner import (
    DailyReport,
    LiveRunner,
    SessionReport,
)
from quant_platform.execution.paper_broker import LatencyModel


# ── DailyReport ──


class TestDailyReport:
    def test_defaults(self):
        r = DailyReport()
        assert r.date == ""
        assert r.portfolio_value == 0.0
        assert r.risk_level == "GREEN"

    def test_to_dict(self):
        r = DailyReport(
            date="2026-01-15",
            portfolio_value=10_500_000,
            cash=2_000_000,
            daily_pnl=50_000,
            daily_return_pct=0.005,
            n_positions=25,
            n_orders=10,
            n_fills=8,
            total_commission=150.0,
            total_tax=80.0,
            total_slippage=50.0,
            drawdown_pct=0.01,
        )
        d = r.to_dict()
        assert d["date"] == "2026-01-15"
        assert d["portfolio_value"] == 10500000.0
        assert d["risk_level"] == "GREEN"
        assert d["n_positions"] == 25


# ── SessionReport ──


class TestSessionReport:
    def test_to_dict(self):
        r = SessionReport(
            session_id="abc123",
            start_date="2026-01-01",
            end_date="2026-01-31",
            days_traded=22,
            total_orders=220,
            total_fills=200,
            initial_capital=10_000_000,
            final_value=10_500_000,
            total_return_pct=5.0,
            annualized_return_pct=60.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=3.0,
            avg_daily_volume=10_200_000,
        )
        d = r.to_dict()
        assert d["session_id"] == "abc123"
        assert d["total_return_pct"] == 5.0
        assert d["sharpe_ratio"] == 1.5
        assert d["max_drawdown_pct"] == 3.0


# ── LiveRunner ──


class TestLiveRunnerInit:
    def test_default_init(self):
        runner = LiveRunner(broker_type="simulated")
        assert runner._initial_cash == 10_000_000
        assert runner._dual_track is True
        assert runner._paper_broker is not None

    def test_no_dual_track(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        assert runner._paper_broker is None

    def test_custom_cash(self):
        runner = LiveRunner(broker_type="simulated", initial_cash=5_000_000)
        assert runner._initial_cash == 5_000_000

    def test_get_state(self):
        runner = LiveRunner(broker_type="simulated")
        state = runner.get_state()
        assert state["running"] is False
        assert state["dual_track"] is True
        assert state["broker"] == "SimulatedBroker"


class TestLiveRunnerUniverse:
    def setup_method(self):
        self.runner = LiveRunner(broker_type="simulated", dual_track=False)

    def test_set_universe(self):
        codes = ["600519", "000858", "000001", "002001", "300750"]
        self.runner.set_universe(codes)
        assert len(self.runner._universe) == 5

    def test_empty_universe_no_signals(self):
        signals = self.runner._generate_signals()
        assert len(signals) == 0

    def test_set_universe_filters_empty(self):
        self.runner.set_universe(["", "  ", "600519"])
        assert len(self.runner._universe) == 1

    def test_set_prices(self):
        self.runner.set_universe(["600519", "000001"])
        self.runner.set_prices({"600519": 1800.0, "000001": 15.0})
        assert self.runner._current_prices["600519"] == 1800.0
        assert "000001" in self.runner._price_history


class TestLiveRunnerSignalGeneration:
    def setup_method(self):
        self.runner = LiveRunner(broker_type="simulated", dual_track=False)
        self.runner.set_universe(["600519", "000858", "000001", "002001", "300750"])

    def test_generates_signals_with_prices(self):
        self.runner.set_prices({
            "600519": 1800.0, "000858": 160.0, "000001": 15.0,
            "002001": 25.0, "300750": 200.0,
        })
        # Random walk price history with moderate upward drift
        rng = np.random.default_rng(123)
        for code in self.runner._universe:
            price = self.runner._current_prices[code]
            for _ in range(35):
                price *= (1 + rng.normal(0.001, 0.012))
                self.runner._price_history.setdefault(code, []).append(price)
        signals = self.runner._generate_signals()
        assert len(signals) > 0
        for sig in signals:
            assert "code" in sig
            assert sig["side"] == "buy"
            assert sig["target_value"] > 0


class TestLiveRunnerSingleCycle:
    def setup_method(self):
        self.runner = LiveRunner(broker_type="simulated", dual_track=False)
        self.runner.set_universe(["600519", "000001"])

    def test_run_once_returns_report(self):
        self.runner.set_prices({"600519": 1800.0, "000001": 15.0})
        for code in self.runner._universe:
            for i in range(30):
                self.runner._price_history.setdefault(code, []).append(
                    self.runner._current_prices.get(code, 10.0) * (1 + i * 0.002)
                )
        report = self.runner.run_once(date="2026-01-15")
        assert isinstance(report, DailyReport)
        assert report.date == "2026-01-15"
        assert report.n_orders >= 0

    def test_run_once_accumulates_reports(self):
        self.runner.set_prices({"600519": 1800.0, "000001": 15.0})
        for code in self.runner._universe:
            for i in range(30):
                self.runner._price_history.setdefault(code, []).append(
                    self.runner._current_prices.get(code, 10.0) * (1 + i * 0.002)
                )
        self.runner.run_once(date="2026-01-15")
        self.runner.run_once(date="2026-01-16")
        assert len(self.runner._daily_reports) == 2


class TestLiveRunnerMultiDay:
    def test_run_multi_day(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        runner.set_universe(["600519", "000001", "002001"])
        report = runner.run(days=5)
        assert isinstance(report, SessionReport)
        assert report.days_traded == 5
        assert report.total_orders > 0

    def test_run_requires_universe(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        with pytest.raises(ValueError, match="Universe not set"):
            runner.run(days=5)

    def test_multi_day_generates_report(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        runner.set_universe(["600519", "000001", "002001"])
        report = runner.run(days=10)
        assert report.initial_capital == 10_000_000
        assert report.days_traded == 10
        assert len(report.daily_reports) == 10


class TestLiveRunnerDualTrack:
    def test_dual_track_paper_broker_created(self):
        runner = LiveRunner(broker_type="simulated", dual_track=True)
        assert runner._paper_broker is not None
        assert isinstance(runner._paper_broker._latency.base_ms, float)

    def test_dual_track_run(self):
        runner = LiveRunner(broker_type="simulated", dual_track=True)
        runner.set_universe(["600519", "000001", "002001"])
        report = runner.run(days=5)
        assert report.days_traded == 5

    def test_no_dual_track_run(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        runner.set_universe(["600519"])
        assert runner._paper_broker is None
        report = runner.run(days=5)
        assert isinstance(report, SessionReport)


class TestLiveRunnerReport:
    def test_generate_empty_report(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        report = runner.generate_report()
        assert isinstance(report, SessionReport)
        assert report.total_orders == 0

    def test_report_after_run(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        runner.set_universe(["600519", "000001"])
        report = runner.run(days=10)
        d = report.to_dict()
        assert "session_id" in d
        assert d["days_traded"] == 10


class TestLiveRunnerState:
    def test_state_before_run(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        s = runner.get_state()
        assert s["running"] is False
        assert s["cycles"] == 0
        assert s["trades"] == 0

    def test_state_after_run(self):
        runner = LiveRunner(broker_type="simulated", dual_track=False)
        runner.set_universe(["600519"])
        runner.run(days=3)
        s = runner.get_state()
        assert s["cycles"] == 3
        assert s["universe_size"] == 1
