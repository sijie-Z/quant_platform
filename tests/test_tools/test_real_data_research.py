"""Tests for ``tools/real_data_research.py`` (R16).

Covers the taskbook acceptance points: an end-to-end smoke run on a synthetic
cache, the insufficient-data error path, the next-session return convention,
and the automatic neutralization skip when no industry map is supplied.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "real_data_research.py"


@pytest.fixture(scope="module")
def tool():
    """Load tools/real_data_research.py as a module without packaging it."""
    spec = importlib.util.spec_from_file_location("real_data_research_r16", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_panel(path: Path, n_days: int = 400, n_assets: int = 20, seed: int = 7) -> pd.DataFrame:
    """Write a synthetic (date, asset) daily cache with the real column set."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    assets = [f"{600000 + i:06d}" for i in range(n_assets)]
    pct = rng.normal(0.0004, 0.018, size=(n_days, n_assets))
    close = 10.0 * np.cumprod(1.0 + pct, axis=0)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    def flat(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1)

    frame = pd.DataFrame(
        {
            "open": flat(close * (1.0 + rng.normal(0.0, 0.002, size=close.shape))),
            "high": flat(close * 1.01),
            "low": flat(close * 0.99),
            "close": flat(close),
            "preclose": flat(np.vstack([close[:1], close[:-1]])),
            "volume": flat(rng.lognormal(12.0, 0.4, size=close.shape)),
            "amount": flat(rng.lognormal(16.0, 0.4, size=close.shape)),
            "turnover": flat(rng.lognormal(-2.0, 0.5, size=close.shape)),
            "pct_chg": flat(pct * 100.0),
            "pe_ttm": flat(rng.uniform(5.0, 50.0, size=close.shape)),
            "pb_mrq": flat(rng.uniform(0.5, 8.0, size=close.shape)),
        },
        index=index,
    )
    frame.to_parquet(path)
    return frame


def _outputs(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "run.json", tmp_path / "run.md"


def _reject_json_constant(value: str):
    raise AssertionError(f"non-finite JSON constant {value!r} in the report")


def test_end_to_end_smoke(tool, tmp_path):
    """Acceptance (a)/(d)/(e): small panel runs end to end, quickly, no industry map."""
    cache = tmp_path / "panel.parquet"
    write_panel(cache)
    json_out, md_out = _outputs(tmp_path)

    started = time.monotonic()
    exit_code = tool.main([
        "--cache", str(cache),
        "--train", "120", "--test", "60", "--step", "60",
        "--factors", "turnover_20d", "--top-k", "1",
        "--json-out", str(json_out), "--md-out", str(md_out),
    ])
    elapsed = time.monotonic() - started

    assert exit_code == 0
    # Strict JSON: NaN/Infinity tokens are invalid JSON and would fail here.
    summary = json.loads(
        json_out.read_text(encoding="utf-8"), parse_constant=_reject_json_constant,
    )
    assert summary["n_folds"] >= 2
    assert summary["oos_days"] > 0
    assert summary["factors_used"] == ["turnover_20d"]
    assert summary["config"]["factors_requested"] == ["turnover_20d"]
    for key in ("sharpe_ratio", "annual_return", "max_drawdown"):
        value = summary["oos_metrics"][key]
        assert value is not None, f"{key} should be finite in the report"
        assert math.isfinite(value)
    # No --industry-map: neutralization is skipped, not crashed on.
    assert summary["config"]["industry_neutralization"] is False
    assert summary["config"]["industry_map_note"].startswith("static latest")
    assert summary["dsr"]["num_trials"] == summary["n_folds"] * 1
    assert summary["dsr"]["num_trials_source"] == "fold_x_factors"
    assert summary["dsr"]["return_basis"] == "excess"
    # Both benchmarks exist on the same OOS window and are finite.
    for name in ("daily_ew", "buyhold"):
        for key in ("sharpe", "annual", "mdd"):
            value = summary["benchmark"][name][key]
            assert value is not None, f"benchmark {name}.{key} should be finite"
            assert math.isfinite(value)
    assert md_out.is_file()
    assert elapsed < 30.0, f"smoke run took {elapsed:.1f}s (budget 30s)"


def test_insufficient_data_exits_nonzero(tool, tmp_path, capsys):
    """Acceptance (b): panel shorter than train+test must fail loudly, not fake."""
    cache = tmp_path / "small.parquet"
    write_panel(cache, n_days=100, n_assets=10)
    json_out, md_out = _outputs(tmp_path)

    exit_code = tool.main([
        "--cache", str(cache),
        "--train", "200", "--test", "60",
        "--factors", "turnover_20d",
        "--json-out", str(json_out), "--md-out", str(md_out),
    ])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "not enough data" in captured.err.lower()
    assert not json_out.exists()
    assert not md_out.exists()


def test_returns_are_next_session_pct_chg(tool):
    """Acceptance (c): returns.iloc[t] == pct_chg(t+1)/100, i.e. shift(-1)."""
    dates = pd.bdate_range("2024-01-02", periods=4)
    assets = ["000001", "000002"]
    pct_chg = np.array([
        [1.0, -2.0],
        [10.0, 5.0],
        [-5.0, 0.0],
        [2.0, 3.0],
    ])
    close = 10.0 * np.cumprod(1.0 + pct_chg / 100.0, axis=0)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    frame = pd.DataFrame(
        {
            "close": close.reshape(-1),
            "volume": np.ones(8),
            "turnover": np.ones(8) * 0.01,
            "pct_chg": pct_chg.reshape(-1),
        },
        index=index,
    )

    _close, _volume, _turnover, returns, adj, pct = tool.build_panels(frame)

    expected = np.array([
        [0.10, 0.05],
        [-0.05, 0.00],
        [0.02, 0.03],
        [np.nan, np.nan],
    ])
    np.testing.assert_allclose(returns.to_numpy(dtype=float), expected, equal_nan=True)
    # Row t is the *next* session's return; the final row has no successor.
    assert returns.iloc[0].tolist() == [pytest.approx(0.10), pytest.approx(0.05)]
    assert returns.iloc[-1].isna().all()
    # ``pct`` is the same-day move, unshifted.
    np.testing.assert_allclose(pct.to_numpy(dtype=float), pct_chg / 100.0)
    # Adjusted path compounds pct_chg; blanked wherever close is missing.
    expected_adj = 100.0 * np.cumprod(1.0 + pct_chg / 100.0, axis=0)
    np.testing.assert_allclose(adj.to_numpy(dtype=float), expected_adj, rtol=1e-12)


def test_limit_gates_use_same_day_pct(tool):
    """Gates on day d must reflect day d's own move, not the forward return.

    Asset A: normal on day 0, limit-up on day 1. Asset B: limit-up on day 0.
    Asset C: normal on day 0, limit-down on day 1. The old gate panel was the
    forward return panel (``returns[t] = pct_chg[t+1]``), so day 0's gate read
    day 1's move -- a one-day look-ahead.
    """
    dates = pd.bdate_range("2024-01-02", periods=3)
    assets = ["A", "B", "C"]
    pct_chg = np.array([
        [1.0, 10.0, -1.0],
        [10.0, 1.0, -10.0],
        [0.0, 0.0, 0.0],
    ])
    close = 10.0 * np.cumprod(1.0 + pct_chg / 100.0, axis=0)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    frame = pd.DataFrame(
        {
            "close": close.reshape(-1),
            "volume": np.ones(9),
            "turnover": np.ones(9) * 0.01,
            "pct_chg": pct_chg.reshape(-1),
        },
        index=index,
    )
    close_p, volume_p, _turnover, returns, _adj, pct = tool.build_panels(frame)
    gates = tool.build_execution_gates(close_p, volume_p, pct)
    day0, day1 = dates[0], dates[1]

    # The forward panel *would* have blocked A on day 0 ...
    assert returns.loc[day0, "A"] == pytest.approx(0.10)
    # ... but the same-day gate must not: A was normal on day 0.
    assert gates["limit_up"].loc[day0].tolist() == [False, True, False]
    assert gates["limit_up"].loc[day1].tolist() == [True, False, False]
    assert gates["limit_down"].loc[day0].tolist() == [False, False, False]
    assert gates["limit_down"].loc[day1].tolist() == [False, False, True]


def test_benchmark_returns_same_oos_window_hand_computed(tool):
    """Daily-EW and buy&hold benchmarks, hand-computed on a tiny panel."""
    dates = pd.bdate_range("2024-01-02", periods=4)
    assets = ["A", "B"]
    pct = pd.DataFrame(
        [
            [0.00, 0.00],
            [0.10, -0.02],
            [-0.05, 0.04],
            [0.02, 0.10],
        ],
        index=dates, columns=assets,
    )
    adj = (1.0 + pct).cumprod() * 100.0
    returns = pct.shift(-1)
    oos = pd.DatetimeIndex(dates[1:])

    series = tool.compute_benchmark_returns(returns, adj, oos)

    # daily EW: row d = mean of the next session's moves.
    expected_ew = pd.Series([-0.005, 0.06, np.nan], index=dates[1:])
    np.testing.assert_allclose(
        series["daily_ew"].to_numpy(dtype=float),
        expected_ew.to_numpy(dtype=float), equal_nan=True,
    )
    # buy & hold: basket formed at day 1, weights drift; day 2's return is
    # 1.0565 / 0.995 - 1, not the daily-EW 0.06.
    expected_bh = pd.Series([-0.005, 1.0565 / 0.995 - 1.0, np.nan], index=dates[1:])
    np.testing.assert_allclose(
        series["buyhold"].to_numpy(dtype=float),
        expected_bh.to_numpy(dtype=float), equal_nan=True,
    )
    # Same clock as the strategy series.
    assert series["daily_ew"].index.equals(series["buyhold"].index)


def test_dsr_uses_excess_returns(tool):
    """DSR input is ``oos - rf/252``; the excess p is strictly larger."""
    rng = np.random.default_rng(3)
    oos = pd.Series(
        rng.normal(0.0005, 0.01, size=300),
        index=pd.bdate_range("2023-01-02", periods=300),
    )

    gross = tool.FactorValidator().deflated_sharpe_test(oos.to_numpy(), num_trials=10)
    excess = tool.deflated_sharpe_excess(oos, num_trials=10)

    assert excess.observed_sharpe < gross.observed_sharpe
    assert excess.deflated_p_value > gross.deflated_p_value


def test_fallback_counter_counts_one_per_event(tool):
    """One fallback emits one line per logger; the count must stay 1."""
    counter = tool._OptimizerFallbackCounter()
    engine_logger = logging.getLogger(tool.ENGINE_LOGGER_NAME)
    optimizer_logger = logging.getLogger(tool.OPTIMIZER_LOGGER_NAME)
    engine_logger.addHandler(counter)
    optimizer_logger.addHandler(counter)
    try:
        engine_logger.warning(
            "Rebalance 2020-01-31: optimizer fallback #1 - solver error",
        )
        assert counter.count == 1
        optimizer_logger.warning("Optimizer fallback to equal weight: solver error")
        # Same event seen by the second logger: not double-counted.
        assert counter.count == 1
    finally:
        engine_logger.removeHandler(counter)
        optimizer_logger.removeHandler(counter)


def test_num_trials_override_and_deterministic_json(tool, tmp_path):
    """--num-trials overrides the default, and identical inputs give identical JSON."""
    cache = tmp_path / "panel.parquet"
    write_panel(cache, n_days=300, n_assets=12, seed=11)
    payloads = []
    for run in range(2):
        json_out = tmp_path / f"run{run}.json"
        md_out = tmp_path / f"run{run}.md"
        exit_code = tool.main([
            "--cache", str(cache),
            "--train", "100", "--test", "50", "--step", "50",
            "--factors", "turnover_20d", "--top-k", "1",
            "--num-trials", "48",
            "--json-out", str(json_out), "--md-out", str(md_out),
        ])
        assert exit_code == 0
        payloads.append(json_out.read_bytes())

    assert payloads[0] == payloads[1]
    summary = json.loads(payloads[0])
    assert summary["dsr"]["num_trials"] == 48
    assert summary["dsr"]["num_trials_source"] == "cli"


def test_industry_map_loading(tool, tmp_path):
    """Optional industry map: 6-digit code lookup, unknown assets -> 'Unknown'."""
    path = tmp_path / "industry.parquet"
    pd.DataFrame(
        {"industry": ["Bank", "Tech"]},
        index=pd.Index(["000001", "600000"], name="code"),
    ).to_parquet(path)

    sectors = tool.load_industry_map(path, pd.Index(["000001", "600000", "999999"]))

    assert sectors.to_dict() == {
        "000001": "Bank",
        "600000": "Tech",
        "999999": "Unknown",
    }


def test_string_date_index_is_coerced(tool, tmp_path):
    """A string-dated cache is coerced to DatetimeIndex instead of failing deep inside."""
    dates = ["2024-01-02", "2024-01-03"]
    index = pd.MultiIndex.from_product([dates, ["000001"]], names=["date", "asset"])
    path = tmp_path / "string_dates.parquet"
    pd.DataFrame(
        {
            "close": [10.0, 10.1],
            "volume": [1000.0, 1100.0],
            "turnover": [0.01, 0.01],
            "pct_chg": [0.0, 1.0],
        },
        index=index,
    ).to_parquet(path)

    loaded = tool.load_panel_frame(path)

    assert isinstance(loaded.index.get_level_values("date"), pd.DatetimeIndex)
    assert loaded.index.get_level_values("date")[0] == pd.Timestamp("2024-01-02")


def test_integer_date_index_read_as_yyyymmdd(tool, tmp_path):
    """Integer YYYYMMDD dates parse as calendar dates, not epoch nanoseconds."""
    index = pd.MultiIndex.from_product(
        [[20240102, 20240103], ["000001"]], names=["date", "asset"],
    )
    path = tmp_path / "int_dates.parquet"
    pd.DataFrame(
        {
            "close": [10.0, 10.1],
            "volume": [1000.0, 1100.0],
            "turnover": [0.01, 0.01],
            "pct_chg": [0.0, 1.0],
        },
        index=index,
    ).to_parquet(path)

    loaded = tool.load_panel_frame(path)

    assert loaded.index.get_level_values("date")[0] == pd.Timestamp("2024-01-02")
    assert loaded.index.get_level_values("date")[1] == pd.Timestamp("2024-01-03")


