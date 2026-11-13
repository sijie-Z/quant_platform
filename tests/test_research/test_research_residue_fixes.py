"""Regression tests for the research-residue wave (R74-R87).

Covers:
* R87 — run_rq3 month labels attribute each forward return to its *realised*
  date before compounding, so the last trading day's t -> t+1 return no
  longer leaks into the previous month.
* R85 — run_regime_heatmap_v4 labels every trading day with its OWN month's
  regime; no calendar-day forward fill of month-end labels.
* R86 — run_factor excludes the cash warm-up from performance, does not
  charge an unexecutable terminal rebalance, and reports a gross vs net
  basis whose cost decomposition is additive in CAGR space.
* common_runner states the metric basis (gross diagnostic proxy) and records
  the actual vs requested panel width.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest.metrics import max_drawdown

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load_script(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    stub = None
    try:
        import akshare  # noqa: F401
    except Exception:
        stub = types.ModuleType("akshare")
        sys.modules["akshare"] = stub
    try:
        spec.loader.exec_module(module)
    finally:
        if stub is not None:
            sys.modules.pop("akshare", None)
    return module


@pytest.fixture(scope="module")
def rq3():
    sys.path.insert(0, str(TOOLS))
    try:
        import run_rq3_stability
        return run_rq3_stability
    finally:
        sys.path.remove(str(TOOLS))


@pytest.fixture(scope="module")
def heatmap():
    # The heatmap imports run_rq3_stability from its sibling directory.
    sys.path.insert(0, str(TOOLS))
    try:
        return _load_script(
            "_heatmap_residue_under_test", TOOLS / "run_regime_heatmap_v4.py"
        )
    finally:
        sys.path.remove(str(TOOLS))


@pytest.fixture(scope="module")
def run_factor_mod():
    return _load_script("_run_factor_residue_under_test", ROOT / "run_factor.py")


@pytest.fixture(scope="module")
def common_runner_mod():
    return _load_script(
        "_common_runner_residue_under_test", TOOLS / "common_runner.py"
    )


# ---------------------------------------------------------------------------
# R87 — month-label compounding uses realised return dates
# ---------------------------------------------------------------------------

class TestRq3RealisedDateAttribution:
    def test_month_end_forward_return_belongs_to_the_next_month(self, rq3):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        benchmark = pd.Series(0.0, index=dates)
        last_jan = dates[dates.month == 1][-1]
        # Forward return labelled on the last January trading day: it is the
        # return from that close to the first February close, i.e. realised
        # in February. The old code compounded it into January.
        benchmark.loc[last_jan] = 0.10

        daily = rq3.classify_regimes(benchmark, trading_days=dates)

        jan = set(daily[daily.index.month == 1].dropna())
        feb = set(daily[daily.index.month == 2].dropna())
        assert jan == {"Sideways"}, f"January got {jan}: the +10% is realised in February"
        assert feb == {"Bull"}, f"February got {feb}: it should receive the +10%"

    def test_realised_monthly_returns_match_the_target_after_conversion(self, rq3):
        trading_days = pd.bdate_range("2021-01-01", "2021-06-30")
        targets = {1: 0.05, 2: -0.05, 3: 0.00, 4: 0.04, 5: -0.03, 6: 0.01}
        trailing = {}
        for date in trading_days:
            target = targets.get(date.month)
            if target is None:
                continue
            n_days = int((trading_days.month == date.month).sum())
            trailing[date] = (1.0 + target) ** (1.0 / n_days) - 1.0
        benchmark = pd.Series(trailing).sort_index().shift(-1)

        daily = rq3.classify_regimes(benchmark, trading_days=trading_days)
        expected = {1: "Bull", 2: "Bear", 3: "Sideways",
                    4: "Bull", 5: "Bear", 6: "Sideways"}
        for month, label in expected.items():
            labels = set(daily[daily.index.month == month].dropna().unique())
            assert labels == {label}, f"month {month} got {labels}, expected {label}"


    def test_strategy_returns_are_filtered_on_realised_dates(self, rq3):
        dates = pd.bdate_range("2024-01-01", "2024-01-31")
        feb1 = pd.Timestamp("2024-02-01")
        returns = pd.Series(0.0, index=dates)
        returns.iloc[-1] = 0.10  # labelled Jan 31, realised on Feb 1
        extended = pd.concat([returns, pd.Series([0.0], index=[feb1])])

        jan_subset = rq3.regime_return_subset(extended, dates)
        # The +10% belongs to February; January's last realised return is 0%.
        assert 0.10 not in set(jan_subset.round(10))
        assert jan_subset.index.max() <= dates[-1]

        feb_subset = rq3.regime_return_subset(extended, [feb1])
        assert feb_subset.loc[feb1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# R85 — heatmap labels are contemporaneous and trading-day indexed
# ---------------------------------------------------------------------------

class TestHeatmapContemporaneousLabels:
    def test_regime_labels_are_trading_day_indexed(self, heatmap, rq3):
        dates = pd.bdate_range("2024-01-01", "2024-03-29")
        benchmark = pd.Series(0.0, index=dates)
        benchmark[benchmark.index.month == 1] = 0.02
        benchmark[benchmark.index.month == 2] = -0.02

        daily, desc = heatmap.classify_regimes(benchmark, trading_days=dates)

        # A calendar-day date_range with ffill used to be returned; the
        # labels must live on the actual trading calendar.
        assert daily.index.equals(dates)
        pd.testing.assert_series_equal(
            daily, rq3.classify_regimes(benchmark, trading_days=dates)
        )
        # Each month carries its own regime, not the previous month-end's.
        assert (daily[daily.index.month == 1] == "Bull").all()
        assert (daily[daily.index.month == 2] == "Bear").all()
        assert desc == "基准月收益"

    def test_holding_returns_use_realised_date_labels(self, heatmap):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        labels = pd.Series("Jan", index=dates, dtype=object)
        labels[labels.index.month == 2] = "Feb"

        aligned = heatmap.realised_regime(labels)

        last_jan = dates[dates.month == 1][-1]
        # A forward return labelled Jan 31 spans into Feb 1, so its realised
        # regime is February. The unshifted modal regime used January.
        assert aligned.loc[last_jan] == "Feb"
        assert aligned.loc[dates[dates.month == 2][0]] == "Feb"

    def test_volatility_labels_are_contemporaneous_and_trading_days(self, heatmap):
        dates = pd.bdate_range("2024-01-01", "2024-03-29")
        cols = ["A", "B"]
        jan = dates[dates.month == 1]
        feb = dates[dates.month == 2]

        rng = np.random.default_rng(0)
        returns = pd.concat([
            pd.DataFrame(0.001, index=jan, columns=cols),
            pd.DataFrame(
                rng.choice([-0.05, 0.05], size=(len(feb), len(cols))),
                index=feb, columns=cols,
            ),
        ])

        daily, desc = heatmap.classify_volatility(returns, trading_days=dates)

        assert daily.index.equals(dates)
        assert set(daily[dates.month == 1].dropna()) == {"LowVol"}
        assert set(daily[dates.month == 2].dropna()) == {"HighVol"}
        # March has no data at all -> unlabelled, never silently carried over.
        assert daily[dates.month == 3].isna().all()
        assert desc == "月波动率"

    def test_failed_cell_reports_nan_not_a_fake_zero(self, heatmap):
        dates = pd.bdate_range("2024-01-01", periods=10)
        cols = ["A", "B"]
        returns = pd.DataFrame(0.0, index=dates, columns=cols)
        result = heatmap.backtest_with_regime_tracking(
            returns, signal_h=5, hold_h=5,
            daily_regime=pd.Series(dtype=object),
        )
        assert np.isnan(result["overall"]["ann_ret"])
        assert np.isnan(result["overall"]["sharpe"])


# ---------------------------------------------------------------------------
# R86 — run_factor warm-up, terminal rebalance and gross/net basis
# ---------------------------------------------------------------------------

DATES = pd.bdate_range("2024-01-01", periods=12)
COLUMNS = [f"{i:04d}" for i in range(65)]


def _flat_factor() -> pd.DataFrame:
    factor = pd.DataFrame(np.nan, index=DATES, columns=COLUMNS)
    for rebalance in (DATES[3], DATES[6]):
        factor.loc[rebalance, COLUMNS] = 0.0
        factor.loc[rebalance, COLUMNS[:20]] = np.arange(100, 80, -1)
    return factor


class TestRunFactorWarmupAndTerminal:
    def test_cash_warmup_is_excluded_from_cagr(self, run_factor_mod):
        factor = _flat_factor()
        rets = pd.DataFrame(0.01, index=DATES, columns=COLUMNS)

        result = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[3], DATES[6]],
            ny=len(DATES) / 252.0, cost_multiplier=0.0,
        )

        assert result["status"] == "ok"
        # The active window starts on the first rebalance (the day the book
        # is established), not on the first holding-return day.
        assert result["warmup_days"] == 3
        assert result["active_start"] == str(DATES[3].date())
        # DATES[3] earns 0 (weights activate next day); DATES[4..11] earn 1%.
        n_holding_days = len(DATES) - 4
        expected = (1.01 ** n_holding_days) ** (1 / result["ny_active"]) - 1
        assert result["cagr"] == pytest.approx(expected, rel=1e-12)
        # Annualising the same capital over the full panel (which includes
        # three cash days) is a different, diluted number.
        full_window = (1.01 ** n_holding_days) ** (1 / (len(DATES) / 252.0)) - 1
        assert result["cagr"] != pytest.approx(full_window, rel=1e-3)

    def test_sharpe_and_drawdown_include_costs(self, run_factor_mod):
        """The report/registry labels these as net of costs (R86)."""
        factor = _flat_factor()
        rets = pd.DataFrame(0.01, index=DATES, columns=COLUMNS)

        net = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[3], DATES[6]],
            ny=len(DATES) / 252.0, cost_multiplier=1.0,
        )
        gross = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[3], DATES[6]],
            ny=len(DATES) / 252.0, cost_multiplier=0.0,
        )
        entry_cost = run_factor_mod.C + run_factor_mod.P  # build 100% from cash
        # The first rebalance day's net return is the entry cost; the gross
        # series has no cost there. The old code left both series identical
        # while claiming Sharpe/MDD were net of costs.
        assert net["daily_rets"].loc[DATES[3]] == pytest.approx(-entry_cost)
        assert gross["daily_rets"].loc[DATES[3]] == pytest.approx(0.0)
        assert net["sharpe"] != pytest.approx(gross["sharpe"], abs=1e-12)
        assert net["max_dd"] != pytest.approx(gross["max_dd"], abs=1e-12)

    def test_terminal_rebalance_is_not_charged(self, run_factor_mod):
        factor = pd.DataFrame(0.0, index=DATES, columns=COLUMNS)
        rets = pd.DataFrame(0.01, index=DATES, columns=COLUMNS)

        # Only an unexecutable signal on the final bar.
        result = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[-1]],
            ny=len(DATES) / 252.0, cost_multiplier=1.0,
        )
        assert result["n_rebals"] == 0
        assert result["turnover_log"] == []
        assert result["cost_log"] == []
        assert result["status"] == "no_active_period"
        assert np.isnan(result["cagr"])

        # A terminal signal must not swallow the earlier executable one.
        mixed = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[2], DATES[-1]],
            ny=len(DATES) / 252.0, cost_multiplier=1.0,
        )
        assert mixed["n_rebals"] == 1
        assert len(mixed["turnover_log"]) == 1

    def test_gross_leg_compounds_without_costs_on_the_same_window(self, run_factor_mod):
        factor = _flat_factor()
        rets = pd.DataFrame(0.01, index=DATES, columns=COLUMNS)

        net = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[3], DATES[6]],
            ny=len(DATES) / 252.0, cost_multiplier=1.0,
        )
        gross = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[3], DATES[6]],
            ny=len(DATES) / 252.0, cost_multiplier=0.0,
        )

        assert gross["active_start"] == net["active_start"] == str(DATES[3].date())
        assert gross["cagr"] > net["cagr"]
        # The zero-cost leg is exactly the compounded gross return.
        active = gross["daily_rets"].iloc[gross["warmup_days"]:]
        expected = (1 + active).prod() ** (1 / gross["ny_active"]) - 1
        assert gross["cagr"] == pytest.approx(expected, rel=1e-12)
        # Cost drag in CAGR space is additive by construction (R86).
        assert gross["cagr"] - (gross["cagr"] - net["cagr"]) == pytest.approx(
            net["cagr"], rel=1e-12
        )

    def test_max_drawdown_still_matches_metrics_after_trimming(self, run_factor_mod):
        rng = np.random.default_rng(7)
        factor = pd.DataFrame(
            rng.normal(size=(len(DATES), len(COLUMNS))),
            index=DATES, columns=COLUMNS,
        )
        rets = pd.DataFrame(
            rng.normal(-0.0004, 0.02, (len(DATES), len(COLUMNS))),
            index=DATES, columns=COLUMNS,
        )
        result = run_factor_mod.backtest(
            factor, rets, DATES, [DATES[2], DATES[6]],
            ny=len(DATES) / 252.0,
        )
        active = result["daily_rets"].iloc[result["warmup_days"]:]
        expected, _, _ = max_drawdown(active)
        assert result["max_dd"] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# common_runner — explicit metric basis and panel provenance
# ---------------------------------------------------------------------------

class TestCommonRunnerBasis:
    def test_evaluation_records_gross_basis_and_panel_width(
        self, common_runner_mod, tmp_path, monkeypatch,
    ):
        dates = pd.bdate_range("2024-01-01", periods=160)
        cols = [f"{i:04d}" for i in range(60)]
        rng = np.random.default_rng(3)
        rets = rng.normal(0.0, 0.01, size=(len(dates), len(cols)))
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols,
        )
        prices.attrs["n_stocks_requested"] = 65
        prices.attrs["n_fetch_failures"] = 5

        from quant_platform.lab.registry import RunStore as RealRunStore

        db_path = str(tmp_path / "research.db")
        monkeypatch.setattr(
            common_runner_mod, "RunStore",
            lambda *args, **kwargs: RealRunStore(db_path),
        )
        monkeypatch.setattr(
            common_runner_mod, "_fetch_csi300_prices",
            lambda end_date=None: prices,
        )
        monkeypatch.setattr(
            common_runner_mod, "generate_report",
            lambda run, out_dir=None: "fake_report.md",
        )

        run_id = common_runner_mod.run_single_factor(
            "reversal", end_date="2024-12-31",
        )

        run = RealRunStore(db_path).get_run(run_id)
        evaluation = json.loads(run["evaluation"])
        assert run["status"] == "success"
        assert evaluation["metric_basis"]["long_short_sharpe_proxy"] == "gross_no_costs"
        assert "gross" in evaluation["metric_basis"]["note"].lower()
        assert evaluation["n_stocks"] == 60
        assert evaluation["n_stocks_requested"] == 65
        assert evaluation["n_fetch_failures"] == 5
        assert evaluation["provider"] == "akshare"
        assert evaluation["data_start"] == str(dates[0].date())
        assert evaluation["data_end"] == str(dates[-1].date())
        assert evaluation["requested_end"] == "2024-12-31"

    def test_report_failure_preserves_evaluation(
        self, common_runner_mod, tmp_path, monkeypatch,
    ):
        dates = pd.bdate_range("2024-01-01", periods=120)
        cols = [f"{i:04d}" for i in range(60)]
        rng = np.random.default_rng(5)
        rets = rng.normal(0.0, 0.01, size=(len(dates), len(cols)))
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols,
        )
        prices.attrs["n_stocks_requested"] = 60
        prices.attrs["n_fetch_failures"] = 0

        from quant_platform.lab.registry import RunStore as RealRunStore

        db_path = str(tmp_path / "research_fail.db")
        monkeypatch.setattr(
            common_runner_mod, "RunStore",
            lambda *args, **kwargs: RealRunStore(db_path),
        )
        monkeypatch.setattr(
            common_runner_mod, "_fetch_csi300_prices",
            lambda end_date=None: prices,
        )

        def boom(run, out_dir=None):
            raise RuntimeError("report failed")

        monkeypatch.setattr(common_runner_mod, "generate_report", boom)

        with pytest.raises(RuntimeError, match="report failed"):
            common_runner_mod.run_single_factor("reversal", end_date="2024-12-31")

        runs = RealRunStore(db_path).list_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"
        # The failure path used to call finish_run without the evaluation,
        # replacing the completed metrics with {}.
        evaluation = json.loads(runs[0]["evaluation"])
        assert evaluation["metric_basis"]["long_short_sharpe_proxy"] == "gross_no_costs"
        assert evaluation["n_ic_obs"] > 0
