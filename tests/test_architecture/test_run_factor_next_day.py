"""`run_factor.backtest` must not let a rebalance-day decision earn that day's
already-realized return.

The original loop computed weights from ``factor_panel.loc[date]`` (close of
``date``) and then multiplied them by ``rets.loc[date]`` -- the t-1 -> t
return that had already happened before the decision. The fix keeps the old
weights for the rebalance day's return and activates the new target weights
from the next trading day. ``style_benchmark`` had the same leak.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_run_factor():
    """Load run_factor.py without registering a flat first-party module.

    The repo root is the `quant_platform` package, so a bare `import
    run_factor` would recreate the dual-module identity that
    tests/test_architecture/test_import_identity.py forbids.
    """
    path = Path(__file__).resolve().parents[2] / "run_factor.py"
    spec = importlib.util.spec_from_file_location("_run_factor_under_test", path)
    module = importlib.util.module_from_spec(spec)

    # `run_factor` imports akshare at module scope, but its tests only touch
    # the backtest function. Install a stub for the duration of exec_module
    # and remove it again: a permanent sys.modules entry made unrelated tests
    # (trading.realtime) believe akshare was installed.
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

DATES = pd.bdate_range("2024-01-01", periods=5)
COLUMNS = [f"000{i:03d}" for i in range(65)]


def _factor_and_returns():
    factor = pd.DataFrame(0.0, index=DATES, columns=COLUMNS)
    rets = pd.DataFrame(0.0, index=DATES, columns=COLUMNS)

    first_book = COLUMNS[:20]
    second_book = COLUMNS[20:40]

    # Decision #1 (DATES[1]) ranks first_book on top; decision #2 (DATES[3])
    # ranks second_book on top.
    factor.loc[DATES[1], first_book] = np.arange(100, 80, -1)
    factor.loc[DATES[1], COLUMNS[20:]] = 1.0
    factor.loc[DATES[3], second_book] = np.arange(100, 80, -1)
    factor.loc[DATES[3], COLUMNS[:20]] = 1.0
    factor.loc[DATES[3], COLUMNS[40:]] = 0.5

    # On DATES[3] the old book gains and the new book loses, so old vs new
    # weights produce opposite returns that day.
    rets.loc[DATES[3], first_book] = 0.01
    rets.loc[DATES[3], second_book] = -0.01
    rets.loc[DATES[4], second_book] = 0.02
    return factor, rets


def test_rebalance_day_return_uses_old_weights():
    factor, rets = _factor_and_returns()

    result = run_factor.backtest(
        factor, rets, DATES, [DATES[1], DATES[3]], ny=len(DATES) / 252,
        cost_multiplier=0.0,
    )
    dr = result["daily_rets"]

    assert dr.loc[DATES[0]] == pytest.approx(0.0)
    assert dr.loc[DATES[1]] == pytest.approx(0.0)  # no portfolio before first decision
    assert dr.loc[DATES[2]] == pytest.approx(0.0)
    # Old weights (first_book) were held into DATES[3]; the newly computed
    # second_book weights must not earn that day's return.
    assert dr.loc[DATES[3]] == pytest.approx(0.01)
    # New weights take effect on the following trading day.
    assert dr.loc[DATES[4]] == pytest.approx(0.02)


def test_style_benchmark_also_activates_weights_next_day():
    factor, rets = _factor_and_returns()

    result = run_factor.style_benchmark(
        factor, rets, DATES, [DATES[1], DATES[3]], ny=len(DATES) / 252,
    )
    dr = result["daily_rets"]

    assert dr.loc[DATES[3]] == pytest.approx(0.01)
    assert dr.loc[DATES[4]] == pytest.approx(0.02)
