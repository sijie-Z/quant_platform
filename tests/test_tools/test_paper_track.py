"""Tests for ``tools/paper_track.py`` (R21).

Covers the taskbook acceptance points and the adversarial-review regressions:

a. Known two-day returns: 50/50 A(+10%)/B(0%) -> +5% on the drifted book,
   execution on the session after ``as_of``.
b. Rebalance-day costs match a hand-computed CostModel charge (commission
   with the CNY 5 floor, transfer fee, square-root impact; no sells -> no
   stamp tax) and NAV is reduced by exactly that amount. The small-notional
   case pins the CNY 5 commission floor.
c. Idempotency: the same inputs twice leave ``nav.csv`` byte-identical and
   row-count-stable; a longer cache appends only the new dates and leaves
   existing lines untouched.
d. Missing prices: a real-cache suspension encoding (repeated close, NaN
   pct_chg/volume) is flagged stale, keeps its weight (not silently zeroed,
   not dropped), and is only liquidated to cash after
   ``--max-stale-sessions`` consecutive misses.
e. ``as_of`` at/after the cache end -> non-zero exit and nothing written.
f. Benchmark column equals the equal-weight daily-rebalanced return of the
   small panel, on the same window.

Review regressions pinned here: ex-rights returns come from ``pct_chg`` (not
raw close ratios), zero/NaN volume is not tradable and cannot produce a
negative NAV, limit-up buys are blocked, integer/list ``as_of`` parsing,
orphan-ledger refusal, and the single-row ledger warning.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "paper_track.py"
DISCLAIMER = "paper tracking only; style-tilt demo, not validated alpha"


@pytest.fixture(scope="module")
def tool():
    """Load tools/paper_track.py as a module without packaging it."""
    spec = importlib.util.spec_from_file_location("paper_track_r21", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_panel(
    prices: dict[str, list[float]],
    pct_chg: dict[str, list[float]] | None = None,
    volume: dict[str, list[float]] | None = None,
    volume_shares: float = 1_000_000.0,
    start: str = "2024-01-02",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (close, volume, pct_chg) wide panels from per-asset paths.

    ``pct_chg`` defaults to the raw close change (fine when there are no
    ex-rights events); pass it explicitly to model suspensions or dividends.
    """
    lengths = {len(values) for values in prices.values()}
    assert len(lengths) == 1, "all assets must have the same number of rows"
    dates = pd.bdate_range(start, periods=lengths.pop())
    close = pd.DataFrame(prices, index=dates, dtype=float)
    if pct_chg is None:
        pct = close.pct_change(fill_method=None) * 100.0
    else:
        pct = pd.DataFrame(pct_chg, index=dates, dtype=float)
    if volume is None:
        vol = pd.DataFrame(volume_shares, index=dates, columns=close.columns)
    else:
        vol = pd.DataFrame(volume, index=dates, dtype=float)
    return close, vol, pct


def write_cache(
    path: Path, close: pd.DataFrame, volume: pd.DataFrame, pct_chg: pd.DataFrame,
) -> Path:
    """Write a (date, asset) MultiIndex parquet with the required columns."""
    close = close.sort_index()
    volume = volume.reindex_like(close)
    pct_chg = pct_chg.reindex_like(close)
    index = pd.MultiIndex.from_product(
        [close.index, close.columns], names=["date", "asset"],
    )
    frame = pd.DataFrame(
        {
            "close": close.to_numpy(dtype=float).reshape(-1),
            "volume": volume.to_numpy(dtype=float).reshape(-1),
            "pct_chg": pct_chg.to_numpy(dtype=float).reshape(-1),
        },
        index=index,
    )
    frame.to_parquet(path)
    return path


def write_target(path: Path, as_of, weights: dict[str, float]) -> Path:
    path.write_text(
        json.dumps({"as_of": as_of, "weights": weights}, indent=2),
        encoding="utf-8",
    )
    return path


def run_tool(tool, target: Path, cache: Path, out_dir: Path, **kwargs) -> int:
    argv = [
        "--target", str(target),
        "--cache", str(cache),
        "--out-dir", str(out_dir),
        "--capital", str(kwargs.pop("capital", 1_000_000.0)),
        "--cost-mode", kwargs.pop("cost_mode", "fixed"),
    ]
    if "max_stale_sessions" in kwargs:
        argv += ["--max-stale-sessions", str(kwargs.pop("max_stale_sessions"))]
    assert not kwargs, f"unexpected kwargs: {kwargs}"
    return tool.main(argv)


def read_nav(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "nav.csv", parse_dates=["date"])


def read_summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# a + f. Execution timing, drifted return, benchmark
# ---------------------------------------------------------------------------

def test_execution_next_session_and_drifted_return(tool, tmp_path):
    # d0 = as_of, d1 = execution; d1->d2 A +10% / B 0%; d2->d3 A -10% / B +20%.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0, 9.9], "B": [10.0, 10.0, 10.0, 12.0]}
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"

    assert run_tool(tool, target, cache, out_dir) == 0

    nav = read_nav(out_dir)
    summary = read_summary(out_dir)

    # Execution at the close of the session strictly after as_of.
    assert list(nav["date"]) == [dates[1], dates[2], dates[3]]
    assert summary["execution_date"] == dates[1].strftime("%Y-%m-%d")
    assert summary["as_of"] == dates[0].strftime("%Y-%m-%d")

    # The book is still 50/50 at the d1 close, so the d1->d2 return is 5%.
    assert nav.loc[1, "daily_return"] == pytest.approx(0.05, abs=1e-9)
    assert nav.loc[1, "nav"] / nav.loc[0, "nav"] - 1 == pytest.approx(0.05, abs=1e-9)
    # Weights drift: 52.38% A / 47.62% B, then -10% / +20% -> 4.2857%.
    assert nav.loc[2, "daily_return"] == pytest.approx(
        0.5 * 1.1 / 1.05 * -0.1 + 0.5 / 1.05 * 0.2, abs=1e-9,
    )
    assert nav.loc[2, "daily_return"] == pytest.approx(3 / 70, abs=1e-9)

    # f: benchmark = equal-weight daily-rebalanced mean of the same panel.
    assert nav.loc[0, "benchmark_daily_return"] == 0.0
    assert nav.loc[1, "benchmark_daily_return"] == pytest.approx(0.05, abs=1e-9)
    assert nav.loc[2, "benchmark_daily_return"] == pytest.approx(0.05, abs=1e-9)
    assert summary["benchmark"]["total_return"] == pytest.approx(
        1.05 * 1.05 - 1, abs=1e-9,
    )

    # NAV-path drawdown and its dates come straight from the ledger labels.
    # The worst dip here is the build-cost row against the initial capital, so
    # the peak is inception (None) and the trough is the execution row.
    assert summary["since_inception"]["max_drawdown"] == pytest.approx(
        float(nav["drawdown"].min()), abs=1e-9,
    )
    assert summary["since_inception"]["max_drawdown_peak"] is None
    assert summary["since_inception"]["max_drawdown_trough"] == dates[1].strftime(
        "%Y-%m-%d"
    )
    assert summary["disclaimer"] == DISCLAIMER
    assert summary["n_target_assets"] == 2
    assert summary["n_purchased_assets"] == 2
    assert nav["n_positions"].tolist() == [2, 2, 2]


# ---------------------------------------------------------------------------
# b. Rebalance-day costs, hand-computed
# ---------------------------------------------------------------------------

def test_build_cost_matches_hand_computed_cost_model(tool, tmp_path):
    # Constant prices give a zero-vol history; the impact rule clips daily
    # vol to 0.1%. The 6th session must be the execution day so the causal
    # rolling(20, min_periods=5).std() window has five returns.
    close, volume, pct = make_panel(
        {"A": [10.0] * 8, "B": [10.0] * 8}, volume_shares=1_000_000.0,
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[5].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    capital = 1_000_000.0
    notional = 0.5 * capital
    volume_cny = 1_000_000.0 * 10.0
    commission = max(notional * 0.0003, 5.0)
    transfer = notional * 0.00001
    impact = notional * math.sqrt(notional / volume_cny) * 0.001  # vol clipped
    expected_impact_costs = 2 * (commission + transfer + impact)

    out_dir = tmp_path / "out-impact"
    assert run_tool(
        tool, target, cache, out_dir, capital=capital, cost_mode="impact",
    ) == 0
    nav = read_nav(out_dir)
    summary = read_summary(out_dir)
    assert nav.loc[0, "date"] == dates[6]
    assert nav.loc[0, "costs_paid"] == pytest.approx(
        expected_impact_costs, abs=1e-6,
    )
    assert nav.loc[0, "nav"] == pytest.approx(capital - expected_impact_costs, abs=1e-6)
    assert nav.loc[0, "daily_return"] == pytest.approx(
        -expected_impact_costs / capital, abs=1e-9,
    )
    assert summary["costs"]["total_cny"] == pytest.approx(
        expected_impact_costs, abs=1e-6,
    )
    # A build has no sells, so the sell-side stamp tax must be zero.
    assert nav["costs_paid"].iloc[1:].eq(0.0).all()

    # Fixed mode: commission floor + transfer + 10bp slippage, no impact.
    fixed_expected = 2 * (commission + transfer + notional * 0.001)
    out_fixed = tmp_path / "out-fixed"
    assert run_tool(
        tool, target, cache, out_fixed, capital=capital, cost_mode="fixed",
    ) == 0
    fixed_nav = read_nav(out_fixed)
    assert fixed_nav.loc[0, "costs_paid"] == pytest.approx(fixed_expected, abs=1e-6)
    assert fixed_nav.loc[0, "nav"] == pytest.approx(capital - fixed_expected, abs=1e-6)


def test_commission_floor_applies_on_small_notional(tool, tmp_path):
    # 10k capital / 50-50 book -> 5k notional per asset: 0.03% is 1.5 CNY, so
    # the CNY 5 per-order floor must bind (5 + 0.05 transfer + 5 slippage).
    close, volume, pct = make_panel({"A": [10.0] * 3, "B": [10.0] * 3})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir, capital=10_000.0) == 0
    nav = read_nav(out_dir)
    expected = 2 * (5.0 + 5_000.0 * 0.00001 + 5_000.0 * 0.001)
    assert expected == pytest.approx(20.1, abs=1e-12)
    assert nav.loc[0, "costs_paid"] == pytest.approx(expected, abs=1e-6)
    assert nav.loc[0, "nav"] == pytest.approx(10_000.0 - expected, abs=1e-6)


def test_variable_volatility_feeds_the_impact_cost(tool, tmp_path):
    # Pin the volatility input: pct_chg 1,-1,1,-1,0 (%) over the five-session
    # window at the execution close gives a sample daily std (ddof=1) of
    # 0.01, so the square-root impact is notional * sqrt(0.05) * 0.01 per
    # asset. A shift(1)/extra-day vol window or a fixed fallback would miss it.
    pct_pattern = [np.nan, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0]
    close, volume, pct = make_panel(
        {"A": [10.0] * 8, "B": [10.0] * 8},
        pct_chg={"A": pct_pattern, "B": pct_pattern},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[5].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    capital = 1_000_000.0
    out_dir = tmp_path / "out"
    assert run_tool(
        tool, target, cache, out_dir, capital=capital, cost_mode="impact",
    ) == 0
    notional = 0.5 * capital
    expected_impact = 2 * notional * math.sqrt(notional / 1e7) * 0.01
    assert expected_impact == pytest.approx(2_236.0679774997897, abs=1e-9)
    expected = 2 * (notional * 0.0003 + notional * 0.00001) + expected_impact
    nav = read_nav(out_dir)
    assert nav.loc[0, "costs_paid"] == pytest.approx(expected, abs=1e-6)
    assert nav.loc[0, "nav"] == pytest.approx(capital - expected, abs=1e-6)


# ---------------------------------------------------------------------------
# c. Idempotency and incremental accumulation
# ---------------------------------------------------------------------------

def test_idempotent_rerun_is_byte_identical(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0, 10.5], "B": [10.0, 10.0, 10.0, 10.2]}
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.6, "B": 0.4},
    )
    out_dir = tmp_path / "out"

    assert run_tool(tool, target, cache, out_dir) == 0
    nav_bytes = (out_dir / "nav.csv").read_bytes()
    summary_bytes = (out_dir / "summary.json").read_bytes()
    nav = read_nav(out_dir)

    assert run_tool(tool, target, cache, out_dir) == 0
    assert (out_dir / "nav.csv").read_bytes() == nav_bytes
    assert (out_dir / "summary.json").read_bytes() == summary_bytes
    assert len(read_nav(out_dir)) == len(nav)


def test_extended_cache_appends_only_new_dates(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.5, 10.4], "B": [10.0, 10.0, 10.1, 10.2]}
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.6, "B": 0.4},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    first_lines = (out_dir / "nav.csv").read_text(encoding="utf-8").splitlines()
    assert len(first_lines) == 1 + 3  # header + d1..d3

    # Feed two more cached sessions and rerun: only those dates are appended.
    close2, volume2, pct2 = make_panel(
        {"A": [10.0, 10.0, 10.5, 10.4, 10.9, 10.8],
         "B": [10.0, 10.0, 10.1, 10.2, 10.3, 10.4]}
    )
    write_cache(tmp_path / "cache.parquet", close2, volume2, pct2)
    assert run_tool(tool, target, cache, out_dir) == 0
    all_lines = (out_dir / "nav.csv").read_text(encoding="utf-8").splitlines()
    assert len(all_lines) == 1 + 5
    assert all_lines[: len(first_lines)] == first_lines
    nav = read_nav(out_dir)
    expected_dates = pd.bdate_range(dates[0], periods=6)
    assert list(nav["date"])[-2:] == [expected_dates[4], expected_dates[5]]


# ---------------------------------------------------------------------------
# d. Missing prices: stale flag, retained weight, delayed liquidation
# ---------------------------------------------------------------------------

def test_suspension_encoding_is_stale_not_zeroed(tool, tmp_path):
    # Real cache encoding: the close repeats, pct_chg and volume are NaN; the
    # resume-day pct_chg carries the whole gap return.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 10.0, 10.0],
         "B": [10.0, 10.0, 10.0, 12.0, 12.0]},
        pct_chg={"A": [np.nan, 0.0, 0.0, 0.0, 0.0],
                 "B": [np.nan, 0.0, np.nan, 20.0, 0.0]},
        volume={"A": [1e6] * 5, "B": [1e6, 1e6, np.nan, 1e6, 1e6]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0

    nav = read_nav(out_dir)
    # d2: B suspended -> held at its last close, flagged stale, still held.
    assert nav.loc[1, "stale_assets"] == 1
    assert nav.loc[1, "daily_return"] == pytest.approx(0.0, abs=1e-9)
    assert nav.loc[1, "n_positions"] == 2
    # d3: the resume-day pct_chg (+20%) is realized against the last close.
    assert nav.loc[2, "stale_assets"] == 0
    assert nav.loc[2, "daily_return"] == pytest.approx(0.5 * 0.2, abs=1e-9)
    assert nav.loc[2, "n_positions"] == 2


def test_nan_close_gap_return_on_resume(tool, tmp_path):
    # The tool also supports panels that encode a gap as a NaN close.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 10.0, 10.0],
         "B": [10.0, 10.0, np.nan, 12.0, 12.0]},
        pct_chg={"A": [np.nan, 0.0, 0.0, 0.0, 0.0],
                 "B": [np.nan, 0.0, np.nan, 20.0, 0.0]},
        volume={"A": [1e6] * 5, "B": [1e6, 1e6, np.nan, 1e6, 1e6]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    nav = read_nav(out_dir)
    assert nav.loc[1, "stale_assets"] == 1
    assert nav.loc[2, "daily_return"] == pytest.approx(0.5 * 0.2, abs=1e-9)


def test_nan_close_with_valid_pct_is_stale(tool, tmp_path):
    # A NaN close with a valid pct_chg is still stale: there is no price to
    # mark or liquidate at, so the session must not book a return.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, np.nan, 10.0], "B": [10.0, 10.0, 10.0, 10.0]},
        pct_chg={"A": [np.nan, 0.0, 5.0, 0.0], "B": [np.nan, 0.0, 0.0, 0.0]},
        volume={"A": [1e6] * 4, "B": [1e6] * 4},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    nav = read_nav(out_dir)
    assert nav.loc[1, "stale_assets"] == 1
    assert nav.loc[1, "daily_return"] == pytest.approx(0.0, abs=1e-9)
    assert nav.loc[1, "n_positions"] == 2


def test_long_stale_liquidates_to_cash_after_limit(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0] * 5, "B": [10.0] * 5},
        pct_chg={"A": [np.nan] + [0.0] * 4,
                 "B": [np.nan, 0.0, np.nan, np.nan, np.nan]},
        volume={"A": [1e6] * 5, "B": [1e6, 1e6, np.nan, np.nan, np.nan]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir, max_stale_sessions=2) == 0

    nav = read_nav(out_dir)
    # d2 is the first miss; d3 is the second consecutive miss -> liquidated.
    assert nav.loc[1, "stale_assets"] == 1
    assert nav.loc[1, "n_positions"] == 2
    assert nav.loc[2, "stale_assets"] == 1  # stale during the session
    assert nav.loc[2, "n_positions"] == 1  # then converted to cash
    summary = read_summary(out_dir)
    assert any("liquidated" in warning for warning in summary["warnings"])
    # Cash earns nothing and A is flat: NAV freezes from the liquidation on.
    assert nav.loc[2, "daily_return"] == pytest.approx(0.0, abs=1e-9)
    assert nav.loc[3, "n_positions"] == 1


def test_max_stale_sessions_zero_disables_liquidation(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0] * 5, "B": [10.0] * 5},
        pct_chg={"A": [np.nan] + [0.0] * 4,
                 "B": [np.nan, 0.0, np.nan, np.nan, np.nan]},
        volume={"A": [1e6] * 5, "B": [1e6, 1e6, np.nan, np.nan, np.nan]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir, max_stale_sessions=0) == 0
    nav = read_nav(out_dir)
    assert nav["n_positions"].tolist() == [2, 2, 2, 2]
    assert nav["stale_assets"].tolist() == [0, 1, 1, 1]
    assert read_summary(out_dir)["warnings"] == []


def test_target_asset_without_execution_price_stays_in_cash(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0], "B": [10.0, np.nan, 12.0]},
        pct_chg={"A": [np.nan, 0.0, 10.0], "B": [np.nan, np.nan, 20.0]},
        volume={"A": [1e6] * 3, "B": [1e6, np.nan, 1e6]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0

    summary = read_summary(out_dir)
    assert summary["n_purchased_assets"] == 1
    assert summary["cash_weight_at_build"] == pytest.approx(0.5, abs=1e-12)
    assert any(
        "no valid close on the execution date" in warning
        for warning in summary["warnings"]
    )
    nav = read_nav(out_dir)
    assert nav.loc[0, "n_positions"] == 1
    # Only the A half participates: +10% -> +5% of the portfolio.
    assert nav.loc[1, "daily_return"] == pytest.approx(0.05, abs=1e-9)


def test_zero_volume_execution_blocks_buy(tool, tmp_path):
    # R9 tradable_mask: volume == 0 means the name could not have been bought.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0], "B": [10.0, 10.0, 10.0]},
        pct_chg={"A": [np.nan, 0.0, 10.0], "B": [np.nan, 0.0, 0.0]},
        volume={"A": [1e6, 1e6, 1e6], "B": [1e6, 0.0, 1e6]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    summary = read_summary(out_dir)
    assert summary["n_purchased_assets"] == 1
    assert any(
        "zero/NaN volume" in warning for warning in summary["warnings"]
    )
    nav = read_nav(out_dir)
    assert nav.loc[1, "daily_return"] == pytest.approx(0.05, abs=1e-9)


def test_all_zero_volume_execution_exits_nonzero(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0], "B": [10.0, 10.0]},
        volume={"A": [0.0, 0.0], "B": [0.0, 0.0]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 2
    assert not out_dir.exists()


def test_limit_up_execution_blocks_buy(tool, tmp_path):
    # R19a: pct_chg > 9.8% on the execution close blocks the buy.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0], "B": [10.0, 11.05, 11.05]},
        pct_chg={"A": [np.nan, 0.0, 10.0], "B": [np.nan, 10.5, 0.0]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    summary = read_summary(out_dir)
    assert summary["n_purchased_assets"] == 1
    assert any("limit-up" in warning for warning in summary["warnings"])
    nav = read_nav(out_dir)
    assert nav.loc[0, "n_positions"] == 1


# ---------------------------------------------------------------------------
# Ex-rights return basis (review blocker B1)
# ---------------------------------------------------------------------------

def test_ex_dividend_uses_pct_chg_not_raw_close(tool, tmp_path):
    # 000429-style ex-dividend: previous close 14.00, adjusted preclose 13.48,
    # close 13.00 -> pct_chg -3.5608%, raw close ratio -7.1429%.
    close, volume, pct = make_panel(
        {"A": [14.0, 14.0, 13.0], "B": [10.0, 10.0, 10.0]},
        pct_chg={"A": [np.nan, 0.0, -3.5608], "B": [np.nan, 0.0, 0.0]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    nav = read_nav(out_dir)
    assert nav.loc[1, "daily_return"] == pytest.approx(0.5 * -0.035608, abs=1e-9)
    assert nav.loc[1, "benchmark_daily_return"] == pytest.approx(
        0.5 * -0.035608, abs=1e-9,
    )
    summary = read_summary(out_dir)
    assert "pct_chg" in summary["return_basis"]


# ---------------------------------------------------------------------------
# e. Data-insufficiency / malformed-input exits
# ---------------------------------------------------------------------------

def test_as_of_at_cache_end_exits_nonzero_and_writes_nothing(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[-1].strftime("%Y-%m-%d"), {"A": 1.0},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 2
    assert not (out_dir / "nav.csv").exists()
    assert not (out_dir / "summary.json").exists()
    stderr = capsys.readouterr().err
    assert "no trading day after as_of" in stderr


def test_as_of_beyond_cache_exits_nonzero(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0]})
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(tmp_path / "target.json", "2030-01-01", {"A": 1.0})
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 2
    assert not out_dir.exists()
    stderr = capsys.readouterr().err
    assert "beyond the cache end" in stderr


def test_leverage_target_is_rejected(tool, tmp_path):
    close, volume, pct = make_panel({"A": [10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.7, "B": 0.7},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 2
    assert not out_dir.exists()


def test_integer_as_of_is_yyyymmdd(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]}, start="2025-06-25",
    )
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(tmp_path / "target.json", 20250630, {"A": 1.0})
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    summary = read_summary(out_dir)
    assert summary["as_of"] == "2025-06-30"
    assert summary["execution_date"] == "2025-07-01"


def test_timezone_aware_as_of_keeps_its_wall_clock_date(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]}, start="2025-06-25",
    )
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", "2025-06-30T00:00:00+08:00", {"A": 1.0},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    summary = read_summary(out_dir)
    assert summary["as_of"] == "2025-06-30"
    assert summary["execution_date"] == "2025-07-01"


def test_list_as_of_is_a_clean_error(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0]})
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(tmp_path / "target.json", [2025, 6, 30], {"A": 1.0})
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 2
    assert not out_dir.exists()
    stderr = capsys.readouterr().err
    assert "as_of" in stderr
    assert "Traceback" not in stderr


def test_execution_on_last_session_warns(tool, tmp_path):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[1].strftime("%Y-%m-%d"), {"A": 1.0},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    summary = read_summary(out_dir)
    assert summary["n_sessions"] == 1
    assert any(
        "last cached session" in warning for warning in summary["warnings"]
    )


# ---------------------------------------------------------------------------
# Ledger provenance guards
# ---------------------------------------------------------------------------

def test_summary_contract_fields(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.5, 10.4], "B": [10.0, 10.0, 10.1, 10.2]}
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.6, "B": 0.4},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    summary = read_summary(out_dir)
    for key in (
        "as_of", "execution_date", "n_sessions", "capital", "cost_mode",
        "target_sha256", "since_inception", "benchmark", "tracking", "costs",
        "warnings", "disclaimer", "ledger_version", "return_basis",
        "tradable_gate",
    ):
        assert key in summary, key
    for key in (
        "final_nav", "total_return", "annualized_return", "sharpe_ratio",
        "max_drawdown",
    ):
        assert key in summary["since_inception"], key
    for key in (
        "total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
    ):
        assert key in summary["benchmark"], key
    assert "difference" in summary["tracking"]
    assert "total_cny" in summary["costs"]
    assert summary["disclaimer"] == DISCLAIMER
    nav = read_nav(out_dir)
    assert summary["since_inception"]["final_nav"] == pytest.approx(
        float(nav["nav"].iloc[-1]), abs=1e-6,
    )


def test_orphan_nav_without_summary_is_refused(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.5, 10.4], "B": [10.0, 10.0, 10.1, 10.2]}
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.6, "B": 0.4},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    nav_bytes = (out_dir / "nav.csv").read_bytes()
    (out_dir / "summary.json").unlink()
    assert run_tool(tool, target, cache, out_dir) == 2
    assert (out_dir / "nav.csv").read_bytes() == nav_bytes
    assert not (out_dir / "summary.json").exists()


def test_different_target_in_same_out_dir_is_refused(tool, tmp_path):
    import hashlib

    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.5, 10.4], "B": [10.0, 10.0, 10.1, 10.2]}
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_a = write_target(
        tmp_path / "target_a.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.6, "B": 0.4},
    )
    sha_a = hashlib.sha256(target_a.read_bytes()).hexdigest()
    out_dir = tmp_path / "out"
    assert run_tool(tool, target_a, cache, out_dir) == 0
    nav_bytes = (out_dir / "nav.csv").read_bytes()
    target_b = write_target(
        tmp_path / "target_b.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.5, "B": 0.5},
    )
    assert run_tool(tool, target_b, cache, out_dir) == 2
    assert (out_dir / "nav.csv").read_bytes() == nav_bytes
    summary = read_summary(out_dir)
    assert summary["target_sha256"] == sha_a
    assert summary["target_file"].endswith("target_a.json")


def test_cost_exceeding_capital_is_rejected(tool, tmp_path):
    # 21 assets would each pay the CNY 5 floor; with 10 CNY of capital the
    # build cost exceeds the book and must abort instead of writing a
    # negative NAV (review blocker B2's second experiment).
    assets = {f"{600000 + i:06d}": [10.0, 10.0] for i in range(21)}
    close, volume, pct = make_panel(assets)
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    weight = 1.0 / 21.0
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {asset: weight for asset in assets},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir, capital=10.0) == 2
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Rolling mode (R24): a schedule of monthly targets -> one ledger
# ---------------------------------------------------------------------------

def write_target_files(
    directory: Path, targets: list[tuple[str, dict[str, float]]],
) -> list[Path]:
    """Write one JSON target per (as_of, weights) entry."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (as_of, weights) in enumerate(targets):
        path = directory / f"target_{index:02d}.json"
        path.write_text(
            json.dumps({"as_of": as_of, "weights": weights}, indent=2),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def run_roll_tool(
    tool,
    cache: Path,
    out_dir: Path,
    *,
    targets: list[Path] | None = None,
    targets_dir: Path | None = None,
    **kwargs,
) -> int:
    argv = [
        "--cache", str(cache),
        "--out-dir", str(out_dir),
        "--capital", str(kwargs.pop("capital", 1_000_000.0)),
        "--cost-mode", kwargs.pop("cost_mode", "fixed"),
    ]
    if targets_dir is not None:
        argv += ["--targets-dir", str(targets_dir)]
    else:
        assert targets is not None
        argv += ["--targets", ",".join(str(path) for path in targets)]
    if "max_stale_sessions" in kwargs:
        argv += ["--max-stale-sessions", str(kwargs.pop("max_stale_sessions"))]
    assert not kwargs, f"unexpected kwargs: {kwargs}"
    return tool.main(argv)


def test_roll_execution_and_rebalance_match_hand_computed_ledger(tool, tmp_path):
    # d0 = first target as_of; d1 = build. A jumps +10% on d2 and stays.
    # The second target is as_of d4, so it rebalances at the d5 close.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
         "B": [10.0] * 10},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.4, "B": 0.4}),
        (dates[4].strftime("%Y-%m-%d"), {"A": 0.2, "B": 0.5}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0

    nav = read_nav(out_dir)
    summary = read_summary(out_dir)

    # Execution is the first cached session strictly after each as_of.
    assert list(nav["date"]) == list(dates[1:])
    assert summary["mode"] == "rolling"
    assert summary["n_targets"] == 2
    assert summary["n_rebalances"] == 2
    assert summary["n_pending_targets"] == 0
    assert [record["execution_date"] for record in summary["rebalances"]] == [
        dates[1].strftime("%Y-%m-%d"), dates[5].strftime("%Y-%m-%d"),
    ]

    # Build at d1: 400k per asset; 0.03% + 0.001% + 10bp fixed per side.
    build_cost_per_asset = 400_000.0 * (0.0003 + 0.00001 + 0.001)
    build_cost = 2 * build_cost_per_asset
    assert nav.loc[0, "costs_paid"] == pytest.approx(build_cost, abs=1e-6)
    assert nav.loc[0, "nav"] == pytest.approx(1_000_000.0 - build_cost, abs=1e-6)
    assert nav.loc[0, "turnover"] == pytest.approx(0.4, abs=1e-9)

    # d2 drift: A +10% on the 400,000 CNY position held against the post-cost
    # NAV (targets are sized off the pre-cost value, so the realised weight is
    # 400,000 / 998,952); the benchmark is the EW mean (+5%).
    assert nav.loc[1, "daily_return"] == pytest.approx(
        40_000.0 / (1_000_000.0 - build_cost), abs=1e-9,
    )
    assert nav.loc[1, "benchmark_daily_return"] == pytest.approx(0.05, abs=1e-9)
    assert nav.loc[1, "nav"] == pytest.approx(1_038_952.0, abs=1e-6)

    # d5 rebalance at P = 1,038,952: sell A down to 20%, buy B up to 50%.
    pre = 1_038_952.0
    sell_a = 440_000.0 - 0.2 * pre
    buy_b = 0.5 * pre - 400_000.0
    sell_cost = sell_a * (0.0003 + 0.0005 + 0.00001 + 0.001)
    buy_cost = buy_b * (0.0003 + 0.00001 + 0.001)
    total_cost = sell_cost + buy_cost
    assert nav.loc[4, "costs_paid"] == pytest.approx(total_cost, abs=1e-6)
    assert nav.loc[4, "nav"] == pytest.approx(pre - total_cost, abs=1e-6)
    assert nav.loc[4, "daily_return"] == pytest.approx(-total_cost / pre, abs=1e-9)
    assert nav.loc[4, "turnover"] == pytest.approx(
        0.5 * (sell_a + buy_b) / pre, abs=1e-9,
    )
    # No drift after d5, no rebalance on the remaining sessions.
    assert nav.loc[5:, "daily_return"].eq(0.0).all()
    assert nav.loc[5:, "costs_paid"].eq(0.0).all()
    assert nav.loc[5:, "turnover"].eq(0.0).all()
    assert nav["n_positions"].tolist() == [2] * 9
    assert summary["disclaimer"] == DISCLAIMER


def test_roll_impact_cost_matches_square_root_rule(tool, tmp_path):
    # Constant prices -> zero vol, clipped to 0.1% by the impact rule; daily
    # CNY volume = 1e6 shares * 10 CNY = 1e7; 400k notional per asset.
    close, volume, pct = make_panel(
        {"A": [10.0] * 8, "B": [10.0] * 8}, volume_shares=1_000_000.0,
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    # The execution day is d6, so rolling(20, min_periods=5).std() at d6 has
    # five returns and the fixed-cost fallback is not used.
    write_target_files(target_dir, [
        (dates[5].strftime("%Y-%m-%d"), {"A": 0.4, "B": 0.4}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(
        tool, cache, out_dir, targets_dir=target_dir, cost_mode="impact",
    ) == 0
    nav = read_nav(out_dir)
    summary = read_summary(out_dir)
    notional = 400_000.0
    volume_cny = 1_000_000.0 * 10.0
    commission = max(notional * 0.0003, 5.0)
    transfer = notional * 0.00001
    impact = notional * math.sqrt(notional / volume_cny) * 0.001
    expected = 2 * (commission + transfer + impact)
    assert nav.loc[0, "date"] == dates[6]
    assert nav.loc[0, "costs_paid"] == pytest.approx(expected, abs=1e-6)
    assert nav.loc[0, "nav"] == pytest.approx(1_000_000.0 - expected, abs=1e-6)
    assert summary["cost_mode"] == "impact"


def test_roll_first_ledger_row_has_zero_benchmark(tool, tmp_path):
    # A -10% move (limit-down, so buying is allowed) on the execution session
    # happened before the book existed: the build row must not book a drift
    # return, and its benchmark must start flat (the same convention as the
    # single-target execution row). Without the fix the row would show -5%.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 10.0], "B": [10.0, 9.0, 9.0, 9.0]},
        pct_chg={"A": [np.nan, 0.0, 0.0, 0.0],
                 "B": [np.nan, -10.0, 0.0, 0.0]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0
    nav = read_nav(out_dir)
    summary = read_summary(out_dir)
    cost_rate = 0.0003 + 0.00001 + 0.001
    expected_cost = 1_000_000.0 * cost_rate / (1.0 + cost_rate)
    assert nav.loc[0, "daily_return"] == pytest.approx(
        -expected_cost / 1_000_000.0, abs=1e-9,
    )
    assert nav.loc[0, "benchmark_daily_return"] == 0.0
    # The next session's benchmark is the panel's own EW return (0 here).
    assert nav.loc[1, "benchmark_daily_return"] == pytest.approx(0.0, abs=1e-12)
    assert summary["benchmark"]["total_return"] == pytest.approx(0.0, abs=1e-12)


def test_roll_second_target_half_turnover_cost_is_hand_computed(tool, tmp_path):
    # Constant prices: the only P&L is costs. Target 1 buys A at 50%; target
    # 2 sells all of A and buys B at 50% -> one-sided turnover ~50%.
    close, volume, pct = make_panel(
        {"A": [10.0] * 6, "B": [10.0] * 6, "C": [10.0] * 6},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5}),
        (dates[2].strftime("%Y-%m-%d"), {"B": 0.5}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0

    nav = read_nav(out_dir)
    summary = read_summary(out_dir)
    build_cost = 500_000.0 * (0.0003 + 0.00001 + 0.001)
    pre = 1_000_000.0 - build_cost
    sell_a = 500_000.0
    buy_b = 0.5 * pre
    sell_cost = sell_a * (0.0003 + 0.0005 + 0.00001 + 0.001)
    buy_cost = buy_b * (0.0003 + 0.00001 + 0.001)
    total_cost = sell_cost + buy_cost

    assert nav.loc[0, "costs_paid"] == pytest.approx(build_cost, abs=1e-6)
    assert nav.loc[2, "costs_paid"] == pytest.approx(total_cost, abs=1e-6)
    assert nav.loc[2, "nav"] == pytest.approx(pre - total_cost, abs=1e-6)
    assert nav.loc[2, "turnover"] == pytest.approx(
        0.5 * (sell_a + buy_b) / pre, abs=1e-9,
    )
    assert summary["rebalances"][1]["sell_cny"] == pytest.approx(sell_a, abs=1e-6)
    assert summary["rebalances"][1]["buy_cny"] == pytest.approx(buy_b, abs=1e-6)
    # Stamp tax is charged on the sell side only: its 0.05% is inside sell_cost.
    assert summary["rebalances"][1]["costs_cny"] == pytest.approx(
        total_cost, abs=1e-6,
    )


def test_roll_idempotent_rebuild_is_byte_identical(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.1, 10.3, 10.2, 10.4, 10.3, 10.5, 10.4],
         "B": [10.0, 9.9, 10.0, 10.2, 10.1, 10.3, 10.2, 10.2]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.3}),
        (dates[2].strftime("%Y-%m-%d"), {"A": 0.3, "B": 0.5}),
        (dates[4].strftime("%Y-%m-%d"), {"A": 0.6, "B": 0.1}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0
    nav_bytes = (out_dir / "nav.csv").read_bytes()
    summary_bytes = (out_dir / "summary.json").read_bytes()

    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0
    assert (out_dir / "nav.csv").read_bytes() == nav_bytes
    assert (out_dir / "summary.json").read_bytes() == summary_bytes

    # The explicit --targets list is sorted by as_of, so it must produce the
    # same bytes as --targets-dir even when passed in reverse order.
    out_list = tmp_path / "out-list"
    files = sorted(target_dir.glob("*.json"), reverse=True)
    assert run_roll_tool(tool, cache, out_list, targets=files) == 0
    assert (out_list / "nav.csv").read_bytes() == nav_bytes
    assert (out_list / "summary.json").read_bytes() == summary_bytes


def test_roll_duplicate_as_of_names_both_files(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    paths = write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5}),
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.6}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 2
    stderr = capsys.readouterr().err
    assert paths[0].name in stderr and paths[1].name in stderr
    assert "duplicate target as_of" in stderr
    assert not out_dir.exists()


def test_roll_invalid_targets_are_rejected_with_the_file_name(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0], "B": [10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    as_of = dates[0].strftime("%Y-%m-%d")

    cases = {
        "leverage": {"A": 0.7, "B": 0.7},
        "negative": {"A": -0.1, "B": 0.5},
        "unknown": {"A": 0.5, "ZZZZZZ": 0.5},
    }
    for name, weights in cases.items():
        target_dir = tmp_path / name
        (path,) = write_target_files(target_dir, [(as_of, weights)])
        out_dir = tmp_path / f"out-{name}"
        assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 2
        stderr = capsys.readouterr().err
        assert path.name in stderr, (name, stderr)
        assert not out_dir.exists()
        if name == "unknown":
            assert "not present in the cached panel" in stderr
        if name == "negative":
            assert "negative" in stderr
        if name == "leverage":
            assert "sum to" in stderr


def test_roll_target_beyond_cache_is_rejected(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0]})
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    (path,) = write_target_files(target_dir, [("2030-01-01", {"A": 0.5})])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 2
    stderr = capsys.readouterr().err
    assert path.name in stderr
    assert "beyond the cache end" in stderr
    assert not out_dir.exists()


def test_roll_pending_tail_target_is_reported_not_executed(tool, tmp_path):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0, 10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5}),
        # as_of == last cached session: no session after it yet.
        (dates[-1].strftime("%Y-%m-%d"), {"A": 0.4}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0
    summary = read_summary(out_dir)
    assert summary["n_targets"] == 2
    assert summary["n_rebalances"] == 1
    assert summary["n_pending_targets"] == 1
    assert summary["pending_targets"][0]["as_of"] == dates[-1].strftime("%Y-%m-%d")
    assert any("pending target" in warning for warning in summary["warnings"])
    nav = read_nav(out_dir)
    assert list(nav["date"]) == list(dates[1:])


def test_roll_duplicate_execution_session_is_rejected(tool, tmp_path, capsys):
    # Start on a Friday: as_of Friday and Saturday both execute on Monday.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0]}, start="2024-01-05",
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    paths = write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5}),
        ("2024-01-06", {"A": 0.6}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 2
    stderr = capsys.readouterr().err
    assert "duplicate execution sessions" in stderr
    assert paths[0].name in stderr and paths[1].name in stderr
    assert not out_dir.exists()


def test_roll_gross_one_build_scales_buys_to_fund_costs(tool, tmp_path):
    # A 100%-gross book cannot spend all the cash and still pay its costs:
    # buys are scaled by 1/(1+cost_rate) and cash ends at exactly zero.
    close, volume, pct = make_panel({"A": [10.0] * 4, "B": [10.0] * 4})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5, "B": 0.5}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0

    nav = read_nav(out_dir)
    summary = read_summary(out_dir)
    record = summary["rebalances"][0]
    # All-linear fixed costs: 30bp commission + 1bp transfer + 10bp slippage.
    cost_rate = 0.0003 + 0.00001 + 0.001
    expected_scale = 1.0 / (1.0 + cost_rate)
    assert record["buy_scale"] == pytest.approx(expected_scale, abs=1e-9)
    expected_cost = 1_000_000.0 * cost_rate / (1.0 + cost_rate)
    assert record["costs_cny"] == pytest.approx(expected_cost, abs=1e-4)
    # The cash-budget tolerance (1e-12 relative) leaves a micro-yuan sliver;
    # everything above it must match the closed-form scale exactly.
    assert record["buy_cny"] == pytest.approx(1_000_000.0 - expected_cost, abs=1e-5)
    assert nav.loc[0, "nav"] == pytest.approx(
        1_000_000.0 - expected_cost, abs=1e-5,
    )
    assert nav.loc[0, "turnover"] == pytest.approx(
        0.5 * (1_000_000.0 - expected_cost) / 1_000_000.0, abs=1e-9,
    )
    assert nav.loc[0, "n_positions"] == 2


def test_roll_blocked_sell_and_limit_up_buy_stay_cash_constrained(tool, tmp_path):
    # Target 2 wants 90% B and 10% C while A is limit-down (cannot sell) and
    # C is limit-up (cannot buy): buys for B are scaled to the available cash.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 9.0, 9.0, 9.0],
         "B": [10.0] * 6,
         "C": [10.0, 10.0, 10.0, 11.0, 11.0, 11.0]},
        pct_chg={
            "A": [np.nan, 0.0, 0.0, -10.0, 0.0, 0.0],
            "B": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],
            "C": [np.nan, 0.0, 0.0, 10.0, 0.0, 0.0],
        },
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.5}),
        (dates[2].strftime("%Y-%m-%d"), {"B": 0.9, "C": 0.1}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0

    summary = read_summary(out_dir)
    record = summary["rebalances"][1]
    assert record["blocked_sells"] == ["A"]
    assert record["blocked_buys"] == ["C"]
    assert record["buy_scale"] < 0.6
    # A dropped 10% before the rebalance: P = 450,000 (A) + 499,345 (cash).
    pre = 450_000.0 + (1_000_000.0 - 500_000.0 * (0.0003 + 0.00001 + 0.001))
    desired_b = 0.9 * pre
    assert record["buy_cny"] + record["costs_cny"] <= 499_345.0 + 2e-3
    assert record["buy_cny"] < desired_b
    assert any("cash constraint scaled" in w for w in summary["warnings"])
    assert any("limit-down" in w for w in summary["warnings"])
    assert any("limit-up" in w for w in summary["warnings"])
    nav = read_nav(out_dir)
    assert nav.loc[3, "n_positions"] == 2  # A (stuck) + B
    assert summary["since_inception"]["final_nav"] == pytest.approx(
        float(nav["nav"].iloc[-1]), abs=1e-6,
    )


def test_roll_zero_scale_reports_no_executed_buys(tool, tmp_path):
    # Fully invested book, then a 100% target switch while the old holding is
    # limit-down and cash is zero: the buy scale collapses to 0 and the record
    # must not claim a buy that never happened.
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 10.0, 9.0, 9.0, 9.0],
         "B": [10.0] * 6},
        pct_chg={"A": [np.nan, 0.0, 0.0, -10.0, 0.0, 0.0],
                 "B": [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 1.0}),
        (dates[2].strftime("%Y-%m-%d"), {"B": 1.0}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0
    summary = read_summary(out_dir)
    record = summary["rebalances"][1]
    assert record["blocked_sells"] == ["A"]
    assert record["buy_scale"] == 0.0
    assert record["buy_cny"] == 0.0
    assert record["n_buys"] == 0
    assert record["sell_cny"] == 0.0
    assert any("cash constraint scaled" in w for w in summary["warnings"])


def test_roll_refuses_a_different_config_in_the_same_out_dir(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0] * 4, "B": [10.0] * 4})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [
        (dates[0].strftime("%Y-%m-%d"), {"A": 0.4, "B": 0.4}),
    ])
    out_dir = tmp_path / "out"
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 0
    nav_bytes = (out_dir / "nav.csv").read_bytes()

    assert run_roll_tool(
        tool, cache, out_dir, targets_dir=target_dir, capital=2_000_000.0,
    ) == 2
    stderr = capsys.readouterr().err
    assert "different capital" in stderr
    assert (out_dir / "nav.csv").read_bytes() == nav_bytes

    assert run_roll_tool(
        tool, cache, out_dir, targets_dir=target_dir, cost_mode="impact",
    ) == 2
    stderr = capsys.readouterr().err
    assert "different cost mode" in stderr
    assert (out_dir / "nav.csv").read_bytes() == nav_bytes


def test_roll_refuses_to_mix_with_a_single_target_ledger(tool, tmp_path, capsys):
    close, volume, pct = make_panel({"A": [10.0, 10.0, 10.0, 10.0]})
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    out_dir = tmp_path / "out"
    single = write_target(
        tmp_path / "single.json", dates[0].strftime("%Y-%m-%d"), {"A": 0.5},
    )
    assert run_tool(tool, single, cache, out_dir) == 0
    target_dir = tmp_path / "targets"
    write_target_files(target_dir, [(dates[0].strftime("%Y-%m-%d"), {"A": 0.5})])
    assert run_roll_tool(tool, cache, out_dir, targets_dir=target_dir) == 2
    stderr = capsys.readouterr().err
    assert "not a rolling ledger" in stderr


# ---------------------------------------------------------------------------
# e. Single-target mode is byte-identical to the pre-rolling behavior
# ---------------------------------------------------------------------------

def test_single_target_mode_matches_pre_rolling_golden_bytes(tool, tmp_path):
    close, volume, pct = make_panel(
        {"A": [10.0, 10.0, 11.0, 10.5], "B": [10.0, 10.0, 10.0, 10.2]},
    )
    dates = close.index
    cache = write_cache(tmp_path / "cache.parquet", close, volume, pct)
    target = write_target(
        tmp_path / "target.json", dates[0].strftime("%Y-%m-%d"),
        {"A": 0.6, "B": 0.4},
    )
    out_dir = tmp_path / "out"
    assert run_tool(tool, target, cache, out_dir) == 0
    golden = (
        "date,nav,daily_return,benchmark_daily_return,cum_return,drawdown,"
        "n_positions,stale_assets,costs_paid\n"
        "2024-01-03,998690.000000,-0.0013100000,0.0000000000,"
        "-0.0013100000,-0.0013100000,2,0,1310.000000\n"
        "2024-01-04,1058611.400000,0.0600000000,0.0500000000,"
        "0.0586114000,0.0000000000,2,0,0.000000\n"
        "2024-01-05,1036640.220000,-0.0207547170,-0.0127272727,"
        "0.0366402200,-0.0207547170,2,0,0.000000\n"
    )
    assert (out_dir / "nav.csv").read_text(encoding="utf-8") == golden
    summary = read_summary(out_dir)
    assert summary["ledger_version"] == 2
    assert "mode" not in summary
    assert summary["since_inception"]["final_nav"] == pytest.approx(
        1_036_640.22, abs=1e-6,
    )
