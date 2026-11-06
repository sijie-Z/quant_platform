"""Regression tests for the research scripts (audit R60/R62-R67).

The scripts are imported by path. Their module import only defines
constants/functions (no data fetch), so the pure helpers are testable without
network access.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.utils.research_window import (
    compact_date,
    resolve_end_date,
    stamp_provenance,
    year_windows,
)

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load(name: str, filename: str):
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


@pytest.fixture(scope="module")
def xv():
    return _load("_xv_research_under_test", "run_xv_synthetic_vs_real.py")


@pytest.fixture(scope="module")
def rq1():
    return _load("_rq1_under_test", "run_rq1_cluster_attribution.py")


@pytest.fixture(scope="module")
def rq2():
    return _load("_rq2_under_test", "run_rq2_cluster_sign_flip.py")


@pytest.fixture(scope="module")
def rq7():
    return _load("_rq7_under_test", "run_rq7_phase_diagram.py")


@pytest.fixture(scope="module")
def cost_mod():
    return _load("_cost_under_test", "run_cost_test.py")


# ---------------------------------------------------------------------------
# R60 — no hardcoded end dates
# ---------------------------------------------------------------------------

class TestResearchWindow:
    def test_explicit_value(self):
        assert resolve_end_date("2026-10-21") == "2026-10-21"
        assert resolve_end_date("20261021") == "2026-10-21"

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("QUANT_RESEARCH_END_DATE", "2024-03-05")
        assert resolve_end_date() == "2024-03-05"
        # Explicit argument wins over the environment.
        assert resolve_end_date("2024-01-01") == "2024-01-01"

    def test_defaults_to_today(self, monkeypatch):
        monkeypatch.delenv("QUANT_RESEARCH_END_DATE", raising=False)
        assert resolve_end_date(today=date(2025, 7, 9)) == "2025-07-09"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="invalid research end date"):
            resolve_end_date("2025-13-01")
        with pytest.raises(ValueError):
            resolve_end_date("not-a-date")

    def test_year_windows_clip_final_year(self):
        windows = year_windows(2023, "2025-06-30")
        assert windows[0] == ("2023", "2023-01-01", "2023-12-31")
        assert windows[-1] == ("2025", "2025-01-01", "2025-06-30")

    def test_stamp_provenance_adds_constant_columns(self):
        df = pd.DataFrame({"x": [1, 2]})
        out = stamp_provenance(df, "2021-01-01", "2026-10-21")
        assert list(out["data_start"]) == ["2021-01-01"] * 2
        assert list(out["data_end"]) == ["2026-10-21"] * 2
        # Input frame is not mutated.
        assert "data_start" not in df.columns

    def test_no_hardcoded_end_dates_in_scope(self):
        patterns = [
            re.compile(r"(?:REAL_END|END_DATE|SYNTHETIC_END)\s*=\s*['\"]\d{4}-\d{2}-\d{2}"),
            re.compile(r"\bend_date\s*=\s*['\"]\d{4}-\d{2}-\d{2}"),
            re.compile(r"['\"]end_date['\"]\s*:\s*['\"]\d{4}-\d{2}-\d{2}"),
            re.compile(r"\bend_date=['\"]\d{4}-\d{2}-\d{2}"),
        ]
        files = [ROOT / "run_factor.py", *(TOOLS / n for n in (
            "common_runner.py", "sanity_check.py", "construction_attribution.py",
            "run_rq1_cluster_attribution.py", "run_rq2_cluster_sign_flip.py",
            "run_rq3_stability.py", "run_rq5b_horizon_alignment.py",
            "run_rq6_horizon_scan.py", "run_rq7_phase_diagram.py",
            "run_rq8_latent_state_probe.py", "run_xv_synthetic_vs_real.py",
            "run_ceiling_probe.py", "run_market_structure_map_v1.py",
            "run_market_structure_v3.py", "run_oos_verification.py",
            "run_probe_phase_aware.py", "run_regime_heatmap_v4.py",
            "run_cost_test.py", "diagnose_ic_v2.py",
        )), *(ROOT / "lab" / "runs").glob("*.py")]
        offenders = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                if pattern.search(text):
                    offenders.append(f"{path.name}: {pattern.pattern}")
        assert offenders == []

    def test_compact_date(self):
        assert compact_date("2026-10-21") == "20261021"


# ---------------------------------------------------------------------------
# R62 — TOP8 must be the eight individual factors, not the clusters
# ---------------------------------------------------------------------------

def _factor_panels(n_assets=6, n_dates=8):
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    cols = [f"A{i}" for i in range(n_assets)]
    rng = np.random.default_rng(5)
    return {
        name: pd.DataFrame(rng.normal(size=(n_dates, n_assets)),
                           index=dates, columns=cols)
        for name in (
            "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
            "rsi_14d", "turnover_20d", "trend_stage", "breakout_proximity",
        )
    }


class TestTop8Experiment:
    def test_top8_is_built_from_individual_factors(self, xv):
        pf = _factor_panels()
        cs = xv.build_signals(pf)

        assert "TOP8" in cs
        stack = [pf[f] for f in xv.TOP8_FACTORS if f in pf]
        expected = (sum(stack) / len(stack)).rank(axis=1, pct=True) - 0.5
        pd.testing.assert_frame_equal(cs["TOP8"], expected)

        # The old code built TOP8 from the four cluster signals, making it
        # numerically identical to ALL.
        assert not np.allclose(cs["TOP8"].to_numpy(), cs["ALL"].to_numpy())

    def test_run_rq1_uses_the_top8_signal(self, xv, monkeypatch):
        pf = _factor_panels()
        cs = xv.build_signals(pf)
        dates = pf["momentum_1m"].index
        returns = pd.DataFrame(0.0, index=dates, columns=pf["momentum_1m"].columns)

        captured = {}

        def fake_bt(signal, prices, rets, benchmark, sector_map):
            captured.setdefault("signals", {})[len(captured["signals"])] = signal.copy()
            return {"summary": {}}

        monkeypatch.setattr(xv, "run_bt", fake_bt)
        result = xv.run_rq1(cs, pf["momentum_1m"], returns, None, None)

        assert list(result["Experiment"]) == list(xv.COMBOS.keys())
        top8_index = list(xv.COMBOS.keys()).index("TOP8")
        passed = captured["signals"][top8_index]
        assert np.allclose(passed.to_numpy(), cs["TOP8"].to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------
# R63 — contemporaneous month labels and trading-day %Time
# ---------------------------------------------------------------------------

class TestRegimeLabels:
    def test_month_labels_are_contemporaneous(self, xv):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        benchmark = pd.Series(0.0, index=dates)
        benchmark[benchmark.index.month == 1] = 0.02
        benchmark[benchmark.index.month == 2] = -0.02

        daily = xv.classify_regimes(benchmark, trading_days=dates)
        jan = daily[daily.index.month == 1]
        feb = daily[daily.index.month == 2]
        assert (jan == "Bull").all()
        # The old forward-fill carried January's month-end label into
        # February, labelling these days Bull instead of Bear.
        assert (feb == "Bear").all()

    def test_pct_time_sums_to_100_over_labelled_trading_days(self, xv):
        dates = pd.bdate_range("2024-01-01", "2024-03-29")
        benchmark = pd.Series(0.0, index=dates)
        benchmark[benchmark.index.month == 1] = 0.02   # Bull
        benchmark[benchmark.index.month == 2] = -0.02  # Bear
        # March: no benchmark rows -> unlabelled days
        benchmark = benchmark[benchmark.index.month != 3]

        daily = xv.classify_regimes(benchmark, trading_days=dates)
        labelled = daily.dropna()
        assert daily.isna().sum() > 0

        shares = [
            xv.regime_pct_time(daily.index[daily == r], daily)
            for r in ("Bull", "Bear", "Sideways")
        ]
        assert sum(shares) == pytest.approx(100.0, abs=1e-9)
        # Denominator is labelled TRADING days, not the 88 calendar days.
        assert shares[0] == pytest.approx(
            len(labelled[labelled == "Bull"]) / len(labelled) * 100, abs=1e-9
        )


# ---------------------------------------------------------------------------
# R64/R65 — failures are NaN + Status/Error; turnover uses the engine key
# ---------------------------------------------------------------------------

class TestFailureAndTurnoverFields:
    def test_rq1_failed_backtest_is_nan_not_zero(self, xv, monkeypatch):
        pf = _factor_panels()
        cs = xv.build_signals(pf)
        dates = pf["momentum_1m"].index
        returns = pd.DataFrame(0.0, index=dates, columns=pf["momentum_1m"].columns)

        def boom(*args, **kwargs):
            raise RuntimeError("engine failure")

        monkeypatch.setattr(xv, "run_bt", boom)
        result = xv.run_rq1(cs, pf["momentum_1m"], returns, None, None)

        assert result["Sharpe"].isna().all()
        assert result["MDD"].isna().all()
        assert set(result["Status"]) == {"failed"}
        assert result["Error"].str.contains("engine failure").all()

    def test_rq3_failed_years_are_kept_with_nan(self, xv, monkeypatch):
        dates = pd.bdate_range("2023-01-02", "2024-12-31")
        cols = ["A0", "A1", "A2"]
        prices = pd.DataFrame(100.0, index=dates, columns=cols)
        returns = pd.DataFrame(0.0, index=dates, columns=cols)
        cs = {
            name: pd.DataFrame(0.1, index=dates, columns=cols)
            for name in (*xv.CLUSTER_NAMES, xv.ALL_NAME)
        }

        def boom(*args, **kwargs):
            raise RuntimeError("engine failure")

        monkeypatch.setattr(xv, "run_bt", boom)
        yearly, _ = xv.run_rq3(cs, prices, returns, None, None)

        assert not yearly.empty
        assert yearly["Sharpe"].isna().all()
        assert yearly["MDD"].isna().all()
        assert set(yearly["Status"]) == {"failed"}
        # Failed years are rows, not silently dropped.
        assert set(yearly["Year"]) == {"2023", "2024"}

    def test_rq1_extract_metrics_reads_annual_turnover(self, rq1):
        metrics = rq1.extract_metrics(
            {"summary": {"sharpe_ratio": 1.2, "max_drawdown": -0.1,
                         "annual_turnover": 2.5}},
            {"mean_ic": 0.01, "icir": 0.5},
        )
        assert metrics["Turnover"] == pytest.approx(2.5)
        assert metrics["Status"] == "ok"

    def test_rq2_extract_metrics_reads_annual_turnover(self, rq2):
        metrics = rq2.extract_metrics(
            {"summary": {"sharpe_ratio": 1.2, "max_drawdown": -0.1,
                         "annual_turnover": 3.5}},
            {"mean_ic": 0.01, "icir": 0.5},
        )
        assert metrics["Turnover"] == pytest.approx(3.5)

    def test_rq1_failed_metrics_are_nan(self, rq1):
        metrics = rq1.failed_metrics({"mean_ic": 0.01, "icir": 0.2},
                                     RuntimeError("x"))
        assert np.isnan(metrics["Sharpe"])
        assert np.isnan(metrics["MDD"])
        assert np.isnan(metrics["Turnover"])
        assert metrics["Status"] == "failed"

    def test_rq2_failed_metrics_are_nan(self, rq2):
        metrics = rq2.failed_metrics({"mean_ic": 0.01, "icir": 0.2},
                                     RuntimeError("x"))
        assert np.isnan(metrics["Sharpe"])
        assert metrics["Status"] == "failed"


# ---------------------------------------------------------------------------
# R66 — RQ7 tranche enters at i+1, not i
# ---------------------------------------------------------------------------

class TestRq7EntryTiming:
    def test_signal_day_return_is_not_earned(self, rq7):
        n_dates, n_assets = 30, 5
        dates = pd.bdate_range("2024-01-01", periods=n_dates)
        cols = [f"A{i}" for i in range(n_assets)]
        returns = pd.DataFrame(-0.01, index=dates, columns=cols)
        # A0 crashes over rows 1..4 so it has the lowest trailing signal at
        # i=5, but a huge row-0 return keeps it out of the book at i=4.
        returns.loc[dates[0], cols[0]] = 10.0
        returns.loc[dates[1]:dates[4], cols[0]] = -0.30
        # Other assets drift up, dominating A0 at i=4.
        returns.loc[dates[0], cols[1:]] = 0.0
        returns.loc[dates[1]:dates[5], cols[1:]] = 0.10
        # Decision day i=5: A0's forward return is +30%. The signal at i=5
        # uses it (it is known at close i+1) but the tranche must not earn it.
        returns.loc[dates[5], cols[0]] = 0.30
        # The day after entry: A0 loses 50%.
        returns.loc[dates[6], cols[0]] = -0.50

        result = rq7.run_cell(returns, freq_days=1, hold_days=1)
        dr = result["daily_returns"]

        # On day 5 only the i=4 tranche (a non-A0 asset) is held: +10%.
        # The old same-day entry would have earned A0's +30% here.
        assert dr.loc[dates[5]] == pytest.approx(0.10, abs=1e-12)
        assert dr.loc[dates[5]] != pytest.approx(0.30, abs=1e-6)
        # The i=5 tranche enters at close(6) and earns the next day's -50%.
        assert dr.loc[dates[6]] == pytest.approx(-0.50, abs=1e-12)


# ---------------------------------------------------------------------------
# R67 — cost test enters at i+1 and charges sell stamp tax
# ---------------------------------------------------------------------------

class TestCostTestTiming:
    @pytest.fixture
    def market(self):
        # 10 dates: rebalance indices are [2, 4, 6] (the i=8 trade would exit
        # past the data end), so exactly three trades are measured and the
        # function's "< 3 trades -> zeros" guard does not fire.
        dates = pd.bdate_range("2024-01-01", periods=10)
        cols = [f"A{i}" for i in range(5)]
        returns = pd.DataFrame(0.0, index=dates, columns=cols)
        # A0 has the lowest signal at both decisions where returns are set;
        # ties at i=6 are harmless because every asset prices flat there.
        returns.loc[dates[2], cols[0]] = 0.10
        returns.loc[dates[2], cols[1:]] = 0.50
        returns.loc[dates[4], cols[0]] = 0.10
        returns.loc[dates[4], cols[1:]] = 0.50

        prices = pd.DataFrame(100.0, index=dates, columns=cols)
        # Trade 1 (decision i=2): correct entry close(3)=100 -> exit close(5)=100.
        # Trade 2 (decision i=4): correct entry close(5)=100 -> exit
        # close(7)=110 (+10%). A same-day entry would buy at close(4)=50 and
        # sell at close(6)=100 (+100%).
        # Trade 3 (decision i=6): entry close(7)=110 -> exit close(9)=110.
        prices.loc[dates[4], cols[0]] = 50.0
        prices.loc[dates[7], cols[0]] = 110.0
        prices.loc[dates[9], cols[0]] = 110.0
        return dates, cols, returns, prices

    def test_entry_is_next_day_close(self, cost_mod, market, monkeypatch):
        dates, cols, returns, prices = market
        monkeypatch.setattr(cost_mod, "HOLD_H", 2)
        monkeypatch.setattr(cost_mod, "SIGNAL_H", 1)
        monkeypatch.setattr(cost_mod, "SELECT_PCT", 0.2)

        result = cost_mod.backtest_with_cost(
            returns, prices, use_vol_filter=False,
            commission_bps=0, slippage_bps=0, stamp_tax_bps=0,
        )
        # Trade 2 earns +10% on 100 -> 110. The signal-day entry would have
        # bought at 50 and sold at 100 (+100%), giving total_return=1.0.
        assert result["total_return"] == pytest.approx(0.10, abs=1e-9)

    def test_sell_stamp_tax_is_charged(self, cost_mod, market, monkeypatch):
        dates, cols, returns, prices = market
        monkeypatch.setattr(cost_mod, "HOLD_H", 2)
        monkeypatch.setattr(cost_mod, "SIGNAL_H", 1)
        monkeypatch.setattr(cost_mod, "SELECT_PCT", 0.2)

        result = cost_mod.backtest_with_cost(
            returns, prices, use_vol_filter=False,
            commission_bps=0, slippage_bps=0, stamp_tax_bps=1000,  # 10% sell tax
        )
        # Independent compounding of the three trades:
        #   t1: 0% gross, -10% sell tax  -> 0.90
        #   t2: +10% gross, -10% sell tax of proceeds -> -1% -> 0.891
        #   t3: 0% gross, -10% sell tax  -> 0.8019
        # total_return is measured from the first recorded value (0.90).
        expected = (0.90 * 0.99 * 0.90) / 0.90 - 1
        assert result["total_return"] == pytest.approx(expected, abs=1e-9)
        assert expected == pytest.approx(-0.109, abs=1e-9)
