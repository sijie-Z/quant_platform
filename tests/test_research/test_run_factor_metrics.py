"""Regression tests for the research portfolio simulator's numerics.

Covers audit R56 (max drawdown was a log-equity cumsum divided by INITIAL,
printing "-0.00%" for every strategy), R57/R58 (sanity_check and
construction_attribution activated new weights before the day's return), and
R59 (turnover/cost counted only sold names, never buys).

The two tool scripts execute a data fetch at module import, so their pure
functions are extracted from the AST and executed against fabricated globals.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest.metrics import max_drawdown

ROOT = Path(__file__).resolve().parents[2]


def _load_run_factor():
    """Load run_factor.py without registering a flat first-party module."""
    path = ROOT / "run_factor.py"
    spec = importlib.util.spec_from_file_location("_run_factor_metrics_under_test", path)
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


run_factor = _load_run_factor()


def _extract_function(path: Path, name: str, namespace: dict):
    """Compile one top-level function out of a script that runs at import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ns = dict(namespace)
    exec(compile(module, str(path), "exec"), ns)
    return ns[name]


DATES = pd.bdate_range("2024-01-01", periods=12)
COLUMNS = [f"{i:04d}" for i in range(65)]


def _decision_data():
    """Two rebalances: all of first20, then 10 kept + 10 new names."""
    factor = pd.DataFrame(np.nan, index=DATES, columns=COLUMNS)
    rets = pd.DataFrame(0.0, index=DATES, columns=COLUMNS)

    old_book = COLUMNS[:20]
    keep = COLUMNS[:10]
    new = COLUMNS[20:30]

    factor.loc[DATES[2], COLUMNS] = 0.0
    factor.loc[DATES[2], old_book] = np.arange(100, 80, -1)

    factor.loc[DATES[4], COLUMNS] = 0.0
    factor.loc[DATES[4], keep] = np.arange(100, 90, -1)
    factor.loc[DATES[4], new] = np.arange(89, 79, -1)

    # On DATES[2] the old book loses 5% and the rest gains 1%. A same-day
    # activation would earn this return for the freshly chosen book; the
    # correct next-day activation earns 0 on the decision day.
    rets.loc[DATES[2], old_book] = -0.05
    rets.loc[DATES[2], COLUMNS[20:]] = 0.01
    # Next day the held (old) book gains 2%.
    rets.loc[DATES[3], old_book] = 0.02
    return factor, rets


def _globals_for_sim(rets, rebals, fwd_ret, extra=None):
    ns = {
        "pd": pd, "np": np, "max_drawdown": max_drawdown,
        "INITIAL": 1_000_000, "N_LONG": 20, "N_SHORT": 0,
        "C": 0.0003, "S": 0.001, "P": 0.0005,
        "rets": rets, "all_dates": DATES, "rebals": rebals,
        "ny": len(DATES) / 252.0, "fwd_ret": fwd_ret,
    }
    if extra:
        ns.update(extra)
    return ns


# ---------------------------------------------------------------------------
# R56 — max drawdown
# ---------------------------------------------------------------------------

class TestRunFactorMaxDrawdown:
    def test_matches_metrics_max_drawdown_on_random_returns(self):
        rng = np.random.default_rng(7)
        rets = pd.DataFrame(
            rng.normal(-0.0004, 0.02, (len(DATES), len(COLUMNS))),
            index=DATES, columns=COLUMNS,
        )
        factor = pd.DataFrame(
            rng.normal(size=(len(DATES), len(COLUMNS))),
            index=DATES, columns=COLUMNS,
        )
        result = run_factor.backtest(
            factor, rets, DATES, [DATES[2], DATES[6]], ny=len(DATES) / 252.0,
        )
        dr = result["daily_rets"]
        expected, _, _ = max_drawdown(dr)

        assert result["max_dd"] == pytest.approx(expected, abs=1e-12)
        # Independent calculation: compounded equity, initial wealth 1.0.
        wealth = np.concatenate([[1.0], (1 + dr.to_numpy()).cumprod()])
        independent = float((wealth / np.maximum.accumulate(wealth) - 1).min())
        assert result["max_dd"] == pytest.approx(independent, abs=1e-12)
        # The old formula produced ~-4e-7 here; a drawdown of a losing
        # strategy must be materially negative.
        assert result["max_dd"] < -0.001

    def test_first_day_loss_is_measured_from_inception(self):
        factor = pd.DataFrame(np.nan, index=DATES, columns=COLUMNS)
        rets = pd.DataFrame(0.0, index=DATES, columns=COLUMNS)
        factor.loc[DATES[1], COLUMNS[:20]] = np.arange(100, 80, -1)
        factor.loc[DATES[1], COLUMNS[20:]] = 0.0
        rets.loc[DATES[2], COLUMNS[:20]] = -0.10

        result = run_factor.backtest(
            factor, rets, DATES, [DATES[1]], ny=len(DATES) / 252.0,
            cost_multiplier=0.0,
        )
        assert result["max_dd"] == pytest.approx(-0.10, abs=1e-12)


# ---------------------------------------------------------------------------
# R57 — lookahead in sanity_check / construction_attribution
# ---------------------------------------------------------------------------

class TestSanityCheckNextDayActivation:
    @pytest.fixture(scope="class")
    def simulate(self):
        path = ROOT / "tools" / "sanity_check.py"
        factor, rets = _decision_data()
        fwd_ret = rets.shift(-1)
        ns = _globals_for_sim(rets, [DATES[2], DATES[4]], fwd_ret)
        return _extract_function(path, "run_factor", ns), factor

    def test_rebalance_day_return_uses_old_weights(self, simulate):
        run_factor_fn, factor = simulate
        result = run_factor_fn(factor)
        dr = result["daily_rets"]

        assert dr.loc[DATES[2]] == pytest.approx(0.0)
        assert dr.loc[DATES[3]] == pytest.approx(0.02)
        # The old code applied current_w before this block; -5% would show.
        assert dr.loc[DATES[2]] != pytest.approx(-0.05, abs=1e-6)

    def test_style_benchmark_also_next_day(self, simulate):
        run_factor_fn, factor = simulate
        result = run_factor_fn(factor)
        # Style captures only DATES[3]'s +2%: one day of the old book held.
        expected = (1.02) ** (252.0 / len(DATES)) - 1
        assert result["style_cagr"] == pytest.approx(expected, rel=1e-9)
        # Like-for-like: the style leg has no costs, so the gap uses the
        # strategy's GROSS cagr. Same book and schedule -> zero gap.
        assert result["gross_cagr"] == pytest.approx(result["style_cagr"], rel=1e-9)
        assert result["style_excess"] == pytest.approx(0.0, abs=1e-12)
        assert result["cagr"] < result["gross_cagr"]  # costs were charged

    def test_max_drawdown_matches_metrics(self, simulate):
        run_factor_fn, factor = simulate
        result = run_factor_fn(factor)
        expected, _, _ = max_drawdown(result["daily_rets"])
        assert result["max_dd"] == pytest.approx(expected, abs=1e-12)


class TestConstructionAttributionNextDayActivation:
    @pytest.fixture(scope="class")
    def simulate(self):
        path = ROOT / "tools" / "construction_attribution.py"
        factor, rets = _decision_data()
        fwd_ret = rets.shift(-1)
        ns = _globals_for_sim(rets, [DATES[2], DATES[4]], fwd_ret)
        return _extract_function(path, "backtest", ns), factor

    def test_rebalance_day_return_uses_old_weights(self, simulate):
        backtest_fn, factor = simulate
        result = backtest_fn(factor, cost_multiplier=0.0)
        dr = result["daily_rets"]

        assert dr.loc[DATES[2]] == pytest.approx(0.0)
        assert dr.loc[DATES[3]] == pytest.approx(0.02)

    def test_max_drawdown_matches_metrics(self, simulate):
        backtest_fn, factor = simulate
        result = backtest_fn(factor, cost_multiplier=0.0)
        expected, _, _ = max_drawdown(result["daily_rets"])
        assert result["max_dd"] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# R59 — buy-side turnover and costs
# ---------------------------------------------------------------------------

class TestTurnoverAndCosts:
    def test_initial_buy_is_charged_and_turnover_is_one_way(self):
        factor, rets = _decision_data()
        result = run_factor.backtest(
            factor, rets, DATES, [DATES[2], DATES[4]], ny=len(DATES) / 252.0,
            cost_multiplier=1.0,
        )
        C, P, S = run_factor.C, run_factor.P, run_factor.S

        # First rebalance: build the book from cash -> buy 100%, one-way 0.5.
        assert result["turnover_log"][0] == pytest.approx(0.5)
        assert result["cost_log"][0] == pytest.approx(1.0 * (C + P))

        # Second rebalance: 10 names sold, 10 bought, weights 1/20 each.
        assert result["turnover_log"][1] == pytest.approx(0.5)
        assert result["cost_log"][1] == pytest.approx(
            0.5 * (C + P) + 0.5 * (C + P + S)
        )

    def test_full_rotation_costs_both_sides(self):
        factor = pd.DataFrame(np.nan, index=DATES, columns=COLUMNS)
        rets = pd.DataFrame(0.0, index=DATES, columns=COLUMNS)
        factor.loc[DATES[2], COLUMNS] = 0.0
        factor.loc[DATES[2], COLUMNS[:20]] = np.arange(100, 80, -1)
        factor.loc[DATES[4], COLUMNS] = 0.0
        factor.loc[DATES[4], COLUMNS[20:40]] = np.arange(100, 80, -1)

        result = run_factor.backtest(
            factor, rets, DATES, [DATES[2], DATES[4]], ny=len(DATES) / 252.0,
            cost_multiplier=1.0,
        )
        C, P, S = run_factor.C, run_factor.P, run_factor.S
        assert result["turnover_log"][1] == pytest.approx(1.0)
        assert result["cost_log"][1] == pytest.approx((C + P) + (C + P + S))

    def test_sanity_check_charges_buys(self):
        path = ROOT / "tools" / "sanity_check.py"
        factor, rets = _decision_data()
        ns = _globals_for_sim(rets, [DATES[2], DATES[4]], rets.shift(-1))
        run_factor_fn = _extract_function(path, "run_factor", ns)
        result = run_factor_fn(factor)

        C, P, S = 0.0003, 0.0005, 0.001
        assert result["turnover_log"][0] == pytest.approx(0.5)
        assert result["cost_log"][0] == pytest.approx(1.0 * (C + P))
        assert result["cost_log"][1] == pytest.approx(
            0.5 * (C + P) + 0.5 * (C + P + S)
        )

    def test_construction_attribution_charges_buys(self):
        path = ROOT / "tools" / "construction_attribution.py"
        factor, rets = _decision_data()
        ns = _globals_for_sim(rets, [DATES[2], DATES[4]], rets.shift(-1))
        backtest_fn = _extract_function(path, "backtest", ns)
        result = backtest_fn(factor, cost_multiplier=1.0)

        C, P, S = 0.0003, 0.0005, 0.001
        assert result["turnover_log"][0] == pytest.approx(0.5)
        assert result["cost_log"][0] == pytest.approx(1.0 * (C + P))
        assert result["cost_log"][1] == pytest.approx(
            0.5 * (C + P) + 0.5 * (C + P + S)
        )
