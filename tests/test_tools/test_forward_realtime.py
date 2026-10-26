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
