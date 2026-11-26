"""The research validation path must never let the holdout influence a choice.

The decisive test builds a market where a factor's *entire* predictive power
sits inside the holdout window (an AR(1) burst) while the rest of the sample is
noise. With the holdout disabled that factor is selected; with the holdout
enabled its holdout dates are removed before factor computation, so it must
not be selected and the report must say how many days were excluded.

The production steps are replaced with deterministic fakes so the test can
observe exactly which panels the real ``_run`` hands to the real factor
selection / walk-forward step; the selection and walk-forward implementations
under test are the production ones.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quant_platform.strategy.research_validation as rv_module
from quant_platform.backtest.walkforward import WalkForwardValidator
from quant_platform.factors.evaluation import ic_summary, rank_ic
from quant_platform.research.holdout import HoldoutSpec
from quant_platform.strategy.research_validation import (
    ResearchValidator,
    ValidationStep,
    mask_research_inputs,
)

_UNSET = object()


# ── fixtures / helpers ─────────────────────────────────────────────────────


def _market_data(seed=7, n_dates=700, n_assets=30, holdout_days=60):
    """Prices/returns where the last ``holdout_days`` carry an AR(1) burst.

    ``prices.pct_change().shift(1)`` (yesterday's return) predicts today's
    return only inside the holdout; outside it the returns are iid noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    assets = [f"A{i:02d}" for i in range(n_assets)]
    values = rng.normal(scale=0.01, size=(n_dates, n_assets))
    start = n_dates - holdout_days
    for t in range(start, n_dates):
        values[t] = 0.6 * values[t - 1] + rng.normal(scale=0.02, size=n_assets)
    returns = pd.DataFrame(values, index=dates, columns=assets)
    prices = 100.0 * (1.0 + returns).cumprod()
    benchmark = returns.mean(axis=1)
    return prices, returns, benchmark


def _factor_panels(prices, returns):
    """The two factor panels the fake compute step derives from its inputs.

    ``holdout_alpha`` only sees prices (yesterday's return), so masking the
    inputs removes its alpha. ``clean`` correlates with same-day returns and
    keeps its power after masking.
    """
    noise = pd.DataFrame(
        np.random.default_rng(123).normal(size=returns.shape) * 0.01,
        index=returns.index, columns=returns.columns,
    )
    return {
        "holdout_alpha": prices.pct_change().shift(1),
        "clean": returns + noise,
    }


def _ic_results(factors, returns):
    results = {}
    for name, panel in factors.items():
        summary = ic_summary(rank_ic(panel, returns))
        results[name] = {
            "mean_ic": summary["mean_ic"],
            "std_ic": summary["std_ic"],
            "icir": summary["icir"],
            "pearson_ic": float("nan"),
            "coverage": 1.0,
            "n_assets": int(panel.notna().any().sum()),
        }
    return results


def _completed(name, category, target, details=None):
    step = ValidationStep(name=name, category=category, target=target,
                          status="completed")
    step.details = details or {}
    return step


def _install_fake_steps(monkeypatch, prices, returns, benchmark, seen):
    """Replace the I/O steps; keep selection / walk-forward production code."""

    def fake_load(self, n_symbols):
        step = _completed("load_data", "data", "market", {
            "prices": prices, "returns": returns, "benchmark": benchmark,
            "metadata": None, "financials": None, "turnover": None,
            "config": SimpleNamespace(),
            "n_assets": len(prices.columns), "n_dates": len(prices),
            "data_source": "synthetic", "adjust": "qfq", "pit": False,
            "provenance_warnings": ["synthetic_data_not_market_data"],
        })
        seen["load"] = step
        return step

    def fake_register(self):
        return _completed(
            "register_factors", "factor", "registry",
            {"factor_names": ["holdout_alpha", "clean"], "n_factors": 2},
        )

    def fake_compute(self, prices, returns, financials, metadata, turnover,
                     config, factor_names, max_factors):
        # Record what the real _run handed down: this is the data the live
        # factor step would have seen.
        seen["compute_index"] = returns.index
        seen["compute_prices_index"] = prices.index
        factors = _factor_panels(prices, returns)
        step = _completed("compute_factors", "factor_eval", "all", {
            "processed_factors": factors,
            "ic_results": _ic_results(factors, returns),
            "sector_map": {}, "fin_unstacked": None,
            "n_factors_computed": len(factors),
            "n_factors_evaluated": len(factors),
        })
        return step

    def fake_save(self, *args, **kwargs):
        return _completed("save_to_factor_store", "storage", "factor_store",
                          {"records_saved": 2})

    def fake_gates(self, ic_results, wf_result, n_factors, prices=None):
        return _completed("strategy_gates", "gate", "validation",
                          {"gate_report": {"overall_status": "PASS"}})

    def fake_ranking(self):
        return _completed("factor_ranking", "analysis", "factor_store",
                          {"rankings": []})

    monkeypatch.setattr(ResearchValidator, "_step_load_data", fake_load)
    monkeypatch.setattr(ResearchValidator, "_step_register_factors", fake_register)
    monkeypatch.setattr(ResearchValidator, "_step_compute_factors", fake_compute)
    monkeypatch.setattr(ResearchValidator, "_step_save_to_store", fake_save)
    monkeypatch.setattr(ResearchValidator, "_step_run_gates", fake_gates)
    monkeypatch.setattr(ResearchValidator, "_step_factor_ranking", fake_ranking)
    monkeypatch.setattr(
        WalkForwardValidator, "run",
        lambda self, **kwargs: {
            "fold_metrics": [{"sharpe_ratio": 0.5}],
            "stability": {}, "aggregate_metrics": {},
        },
    )


def _run_validator(monkeypatch, *, holdout=_UNSET, n_dates=700, holdout_days=60):
    prices, returns, benchmark = _market_data(
        n_dates=n_dates, holdout_days=holdout_days,
    )
    seen: dict = {}
    _install_fake_steps(monkeypatch, prices, returns, benchmark, seen)
    kwargs = {} if holdout is _UNSET else {"holdout": holdout}
    validator = ResearchValidator(n_stocks=30, **kwargs)
    report = validator.run_quick(max_factors=2, max_folds=1, n_symbols=30)
    return report, seen


def _step(report, name):
    return next(s for s in report.steps if s.name == name)


def _selected(report):
    return _step(report, "walk_forward").details["factor_selection"]["selected"]


def _holdout_spec(n_dates=700, holdout_days=60):
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    return HoldoutSpec(dates[-holdout_days].date(), dates[-1].date())


# ── masking helper ─────────────────────────────────────────────────────────


class TestMaskResearchInputs:
    def test_drops_holdout_dates_and_preserves_order(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        assets = ["A", "B"]
        prices = pd.DataFrame(
            np.arange(20, dtype=float).reshape(10, 2), index=dates, columns=assets,
        )
        returns = prices / 100.0
        benchmark = pd.Series(np.arange(10, dtype=float), index=dates)
        turnover = prices.copy()
        financials = pd.DataFrame(
            {"value": np.arange(20, dtype=float)},
            index=pd.MultiIndex.from_product(
                [dates, assets], names=["date", "asset"],
            ),
        )
        spec = HoldoutSpec(dates[3].date(), dates[5].date())

        masked, info = mask_research_inputs(
            spec, prices=prices, returns=returns, benchmark=benchmark,
            financials=financials, turnover=turnover,
        )

        keep = list(dates[:2]) + list(dates[6:])
        # dates[2] (the last row before the holdout) is dropped too: its
        # forward t -> t+1 return label is the first holdout day's return.
        assert list(masked["prices"].index) == keep
        assert list(masked["returns"].index) == keep
        assert list(masked["benchmark"].index) == keep
        assert list(masked["turnover"].index) == keep
        assert keep == sorted(keep)  # original order preserved
        # (date, asset) MultiIndex: 3 holdout days x 2 assets dropped (plus the
        # boundary date for the forward-label guard).
        assert masked["financials"].index.get_level_values("date").unique().tolist() == keep
        assert info == {
            "spec": {"start_date": str(dates[3].date()),
                     "end_date": str(dates[5].date())},
            "excluded_days": 3,
        }

    def test_forward_label_boundary_row_is_dropped(self):
        """Returns[t] labels t -> t+1: the pre-holdout row leaks a holdout
        return and must be dropped; frames stay aligned and ordered."""
        dates = pd.bdate_range("2024-01-01", periods=6)
        returns = pd.DataFrame({"A": np.arange(6, dtype=float)}, index=dates)
        prices = returns.copy()
        spec = HoldoutSpec(dates[3].date(), dates[4].date())

        masked, info = mask_research_inputs(
            spec, prices=prices, returns=returns, benchmark=None,
            financials=None, turnover=None,
        )

        # Holdout rows 3-4 and the boundary row 2 are gone; 0-1 and 5 remain.
        assert list(masked["returns"].index) == [dates[0], dates[1], dates[5]]
        assert list(masked["prices"].index) == list(masked["returns"].index)
        assert info["excluded_days"] == 2

    def test_none_inputs_pass_through(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        returns = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4]}, index=dates)
        # Window starts on the first row: no boundary row before it.
        spec = HoldoutSpec(dates[0].date(), dates[1].date())

        masked, info = mask_research_inputs(
            spec, prices=None, returns=returns, benchmark=None,
            financials=None, turnover=None,
        )

        assert masked["prices"] is None
        assert masked["benchmark"] is None
        assert masked["financials"] is None
        assert masked["turnover"] is None
        assert len(masked["returns"]) == 2
        assert info["excluded_days"] == 2


# ── end-to-end isolation through the real _run ─────────────────────────────


class TestHoldoutResearchIsolation:
    def test_disabled_holdout_keeps_the_previous_selection(self, monkeypatch):
        report, _seen = _run_validator(monkeypatch, holdout=None)

        assert report.holdout is None
        assert "holdout" not in report.to_dict()
        assert "holdout" not in report.summary

        # Fixed-seed regression: with the holdout disabled the selection is
        # exactly what the untouched selector returns on the full panels.
        prices, returns, _benchmark = _market_data()
        factors = _factor_panels(prices, returns)
        expected = rv_module.select_walk_forward_factors(
            _ic_results(factors, returns), factors, returns,
        )["selected"]
        assert _selected(report) == expected == ["clean", "holdout_alpha"]

    def test_enabled_holdout_masks_inputs_and_drops_the_holdout_only_factor(
        self, monkeypatch,
    ):
        spec = _holdout_spec()
        report, seen = _run_validator(monkeypatch, holdout=spec)

        # The factor step only ever saw non-holdout dates.
        holdout_dates = pd.bdate_range("2020-01-02", periods=700)[-60:]
        assert seen["compute_index"].max() < holdout_dates[0]
        assert seen["compute_prices_index"].max() < holdout_dates[0]

        # The alpha that lives only inside the holdout is not selected ...
        assert _selected(report) == ["clean"]
        # ... and the report proves the exclusion happened.
        assert report.holdout == {
            "spec": spec.to_dict(), "excluded_days": 60,
        }
        assert report.to_dict()["holdout"] == report.holdout
        assert report.summary["holdout"] == report.holdout
        load_step = _step(report, "load_data")
        assert load_step.details["holdout_excluded_days"] == 60

    def test_configured_holdout_is_resolved_by_default(self, monkeypatch):
        spec = _holdout_spec()
        monkeypatch.setattr(
            rv_module, "load_configured_holdout", lambda: spec,
        )

        report, _seen = _run_validator(monkeypatch)  # no explicit holdout

        assert report.holdout["spec"] == spec.to_dict()
        assert _selected(report) == ["clean"]

    def test_holdout_outside_the_data_leaves_research_unchanged(self, monkeypatch):
        # Holdout entirely before the sample: nothing to exclude, selection
        # must match the disabled run (the holdout is still recorded).
        spec = HoldoutSpec("2019-01-01", "2019-06-30")
        report, _seen = _run_validator(monkeypatch, holdout=spec)

        assert report.holdout == {
            "spec": spec.to_dict(), "excluded_days": 0,
        }
        assert _selected(report) == ["clean", "holdout_alpha"]
        # No false "rows after the holdout" warning: nothing was excluded.
        load_step = _step(report, "load_data")
        assert not any(
            "holdout_not_at_tail" in warning for warning in load_step.warnings
        )

    def test_holdout_in_the_middle_warns_about_remaining_rows(self, monkeypatch):
        # A non-tail holdout is allowed but documented as a risk: rows after
        # the window can still span its (removed) price path in long-lookback
        # factors, so the report says so instead of hiding it.
        dates = pd.bdate_range("2020-01-02", periods=700)
        spec = HoldoutSpec(dates[400].date(), dates[459].date())
        report, _seen = _run_validator(monkeypatch, holdout=spec)

        load_step = _step(report, "load_data")
        assert any(
            "holdout_not_at_tail" in warning for warning in load_step.warnings
        )
        assert report.holdout["excluded_days"] == 60

    def test_insufficient_samples_after_exclusion_skip_walk_forward(
        self, monkeypatch,
    ):
        # 620 days - 60 holdout days - 1 boundary (forward-label) day = 559
        # remaining, below 504 train + 126 test.
        spec = _holdout_spec(n_dates=620, holdout_days=60)
        report, _seen = _run_validator(
            monkeypatch, holdout=spec, n_dates=620, holdout_days=60,
        )

        wf_step = _step(report, "walk_forward")
        assert wf_step.status == "skipped"
        assert wf_step.details["wf_result"] is None
        assert wf_step.details["wf_results"] == {}
        assert wf_step.details["holdout_skip"]["excluded_days"] == 60
        assert wf_step.details["holdout_skip"]["remaining_days"] == 559
        # The selection detail is still present, like the other skipped path.
        assert "factor_selection" in wf_step.details
        assert wf_step.details["factor_selection"]["selected"] == ["clean"]
        warning = " ".join(wf_step.warnings)
        assert "holdout excluded 60 trading days" in warning
        assert "required for walk-forward" in warning
        assert report.holdout["excluded_days"] == 60
        # A skip is not a pass: the report must say it needs review.
        assert report.summary["overall"] == "needs_review"

    def test_masked_factor_rows_never_reach_the_selection_step(
        self, monkeypatch,
    ):
        """Even if a factor step leaked holdout rows, _run strips them."""
        spec = _holdout_spec()
        prices, returns, benchmark = _market_data()
        seen: dict = {}
        _install_fake_steps(monkeypatch, prices, returns, benchmark, seen)

        original_compute = ResearchValidator._step_compute_factors

        def leaky_compute(self, prices, returns, financials, metadata, turnover,
                          config, factor_names, max_factors):
            # Simulate a buggy factor step that evaluated on the full panel:
            # glue the holdout rows back onto the masked panels and inject the
            # contaminated full-sample IC moments as well.
            step = original_compute(
                self, prices, returns, financials, metadata, turnover,
                config, factor_names, max_factors,
            )
            full_prices, full_returns, _benchmark = _market_data()
            full_factor = _factor_panels(full_prices, full_returns)["holdout_alpha"]
            step.details["processed_factors"]["holdout_alpha"] = full_factor
            step.details["ic_results"]["holdout_alpha"] = _ic_results(
                {"holdout_alpha": full_factor}, full_returns,
            )["holdout_alpha"]
            return step

        monkeypatch.setattr(
            ResearchValidator, "_step_compute_factors", leaky_compute,
        )
        validator = ResearchValidator(n_stocks=30, holdout=spec)
        report = validator.run_quick(max_factors=2, max_folds=1, n_symbols=30)

        passed = _step(report, "walk_forward").details["factor_selection"]
        # Selection recomputed the panels and the IC moments from the masked
        # data it received, so the leaked holdout alpha cannot win.
        assert passed["n_obs"]["holdout_alpha"] < len(prices)
        assert "holdout_alpha" not in passed["selected"]
