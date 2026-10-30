"""Tests for the indicative real-time target monitor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "forward_realtime.py"


@pytest.fixture()
def realtime():
    spec = importlib.util.spec_from_file_location("forward_realtime_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spot() -> pd.DataFrame:
    return pd.DataFrame({
        "latest": [11.0, 9.6],
        "prev_close": [10.0, 10.0],
        "return": [0.10, -0.04],
    }, index=["000001", "600000"])


def test_snapshot_is_gross_and_not_an_official_fill(realtime):
    weights = pd.Series({"000001": 0.5, "600000": 0.25})
    result = realtime.strategy_snapshot("turnover", weights, _spot(), 0.02)
    assert result["gross_return_since_previous_close"] == pytest.approx(0.04)
    assert result["indicative_nav_gross"] == pytest.approx(2_080_000.0)
    assert result["indicative_excess_vs_universe"] == pytest.approx(0.02)
    assert result["covered_weight"] == pytest.approx(0.75)
    assert result["complete"] is True


def test_missing_quote_is_explicit(realtime):
    weights = pd.Series({"000001": 0.5, "300000": 0.25})
    result = realtime.strategy_snapshot("turnover", weights, _spot(), None)
    assert result["complete"] is False
    assert result["missing_assets"] == ["300000"]
    assert result["gross_return_since_previous_close"] == pytest.approx(0.05)
    assert result["indicative_excess_vs_universe"] is None


def test_official_book_marks_actual_positions_and_cash(realtime):
    book = {
        "as_of": "2026-10-26",
        "cash": 1000.0,
        "positions_value": 2000.0,
        "positions": [
            {"code": "000001", "value": 1000.0,
             "reference_price": 10.0, "shares": 100.0},
            {"code": "600000", "value": 1000.0,
             "reference_price": 20.0, "shares": 50.0},
        ],
    }
    result = realtime.book_snapshot(book, _spot())
    assert result["basis"] == "official_book"
    assert result["official_nav"] == pytest.approx(3000.0)
    assert result["indicative_nav"] == pytest.approx(2580.0)
    assert result["return_since_official_close"] == pytest.approx(-0.14)
    assert result["quoted_assets"] == 2
    assert result["complete"] is True


def test_official_book_uses_reference_price_for_missing_quote(realtime):
    book = {
        "as_of": "2026-10-26", "cash": 0.0, "positions_value": 1000.0,
        "positions": [{"code": "300000", "value": 1000.0,
                       "reference_price": 10.0, "shares": 100.0}],
    }
    result = realtime.book_snapshot(book, _spot())
    assert result["indicative_nav"] == pytest.approx(1000.0)
    assert result["stale_assets"] == ["300000"]
    assert result["complete"] is False


def test_transient_snapshot_failure_does_not_end_session(realtime, tmp_path, monkeypatch):
    payload = {
        "timestamp": "2026-10-27T10:00:00",
        "universe": {"equal_weight_return": 0.01},
        "strategies": {
            "turnover": {
                "gross_return_since_previous_close": 0.02,
                "indicative_nav_gross": 2_040_000.0,
                "quoted_assets": 1,
                "complete": True,
            }
        },
    }
    calls = {"count": 0}

    def flaky_snapshot():
        calls["count"] += 1
        if calls["count"] == 1:
            raise IndexError("temporary provider response")
        return payload

    monkeypatch.setattr(realtime, "snapshot", flaky_snapshot)
    monkeypatch.setattr(realtime.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys, "argv",
        ["forward_realtime.py", "--interval", "1", "--count", "3",
         "--out-dir", str(tmp_path)],
    )
    realtime.main()
    assert len(pd.read_csv(tmp_path / "snapshots.csv")) == 2
    assert len((tmp_path / "errors.log").read_text(encoding="utf-8").splitlines()) == 1


def test_three_consecutive_snapshot_failures_stop_without_stale_output(
    realtime, tmp_path, monkeypatch,
):
    def failed_snapshot():
        raise IndexError("provider still unavailable")

    monkeypatch.setattr(realtime, "snapshot", failed_snapshot)
    monkeypatch.setattr(realtime.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys, "argv",
        ["forward_realtime.py", "--interval", "1", "--count", "3",
         "--out-dir", str(tmp_path)],
    )
    with pytest.raises(RuntimeError, match="3 consecutive"):
        realtime.main()
    assert not (tmp_path / "snapshots.csv").exists()
    assert len((tmp_path / "errors.log").read_text(encoding="utf-8").splitlines()) == 3


def test_snapshots_stay_in_scratch_and_append(realtime, tmp_path):
    payload = {
        "timestamp": "2026-10-26T16:00:00",
        "universe": {"equal_weight_return": 0.02},
        "strategies": {
            "turnover": {
                "gross_return_since_previous_close": 0.04,
                "indicative_nav_gross": 2_080_000.0,
                "quoted_assets": 2,
                "complete": True,
            }
        },
    }
    realtime.write_snapshot(payload, tmp_path)
    realtime.write_snapshot(payload, tmp_path)
    assert (tmp_path / "latest.json").exists()
    rows = pd.read_csv(tmp_path / "snapshots.csv")
    assert len(rows) == 2
    assert set(rows["strategy"]) == {"turnover"}
