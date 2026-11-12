from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "forward_evaluation.py"


@pytest.fixture()
def evaluation():
    spec = importlib.util.spec_from_file_location("forward_evaluation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_b1_alignment_excludes_pre_inception_return(evaluation):
    dates = pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28"])
    raw_b1 = pd.Series([0.025, -0.01, 0.02], index=dates)
    aligned = evaluation.align_b1_returns(raw_b1, dates)
    assert aligned.tolist() == [0.0, -0.01, 0.02]
    assert (1.0 + aligned).prod() - 1.0 == pytest.approx(0.0098)


def test_b1_alignment_fails_when_a_ledger_date_is_missing(evaluation):
    dates = pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28"])
    b1 = pd.Series([0.025, 0.02], index=dates[[0, 2]])
    with pytest.raises(RuntimeError, match="does not cover every"):
        evaluation.align_b1_returns(b1, dates)


def test_metrics_cumulative_matches_daily_compounding(evaluation):
    nav = pd.Series(
        [1_998_000.0, 2_003_000.0, 2_001_000.0],
        index=pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28"]),
    )
    result = evaluation.metrics(nav)
    returns = result["daily_returns"]
    assert (1.0 + returns).prod() - 1.0 == pytest.approx(
        result["cum_return"], abs=1e-12,
    )
    # Initial capital is the first drawdown peak, so a first-day loss counts.
    first_day_only_loss = evaluation.metrics(pd.Series(
        [1_990_000.0], index=pd.to_datetime(["2026-10-26"]),
    ))
    assert first_day_only_loss["max_drawdown"] == pytest.approx(-0.005)


def test_metrics_rejects_non_finite_nav(evaluation):
    with pytest.raises(RuntimeError, match="non-finite"):
        evaluation.metrics(pd.Series(
            [2_000_000.0, float("nan")],
            index=pd.to_datetime(["2026-10-26", "2026-10-27"]),
        ))


def test_st_cache_requires_coverage_through_evaluation_end(
    evaluation, tmp_path, monkeypatch,
):
    cache = tmp_path / "st.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2026-10-26", "2026-11-01"]),
        "asset": ["000001", "000001"],
        "is_st": [False, False],
    }).to_parquet(cache)
    monkeypatch.setattr(evaluation, "ST_CACHE", cache)
    import baostock as bs

    monkeypatch.setattr(bs, "login", lambda: type("R", (), {"error_code": "1", "error_msg": "offline"})())
    with pytest.raises(RuntimeError, match="login failed"):
        evaluation.st_panel(pd.Timestamp("2026-10-26"), pd.Timestamp("2026-11-12"))


def test_b1_rejects_missing_pit_st_observation(evaluation, tmp_path, monkeypatch):
    merged = tmp_path / "merged.parquet"
    pd.DataFrame({
        "date": pd.to_datetime([
            "2026-10-23", "2026-10-26", "2026-10-27",
            "2026-10-23", "2026-10-26", "2026-10-27",
        ]),
        "asset": ["000001", "000001", "000001", "000002", "000002", "000002"],
        "close": [100.0, 101.0, 102.0, 50.0, 51.0, 52.0],
    }).to_parquet(merged)
    st_cache = tmp_path / "st.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2026-10-26", "2026-10-27"]),
        "asset": ["000001", "000001"],
        "is_st": [False, False],
    }).to_parquet(st_cache)
    monkeypatch.setattr(evaluation, "MERGED", merged)
    monkeypatch.setattr(evaluation, "ST_CACHE", st_cache)

    monkeypatch.setattr(
        evaluation, "st_panel",
        lambda start, end: pd.read_parquet(st_cache),
    )
    with pytest.raises(RuntimeError, match="PIT ST mask missing"):
        evaluation.b1_returns(pd.Timestamp("2026-10-26"), pd.Timestamp("2026-10-26"))

