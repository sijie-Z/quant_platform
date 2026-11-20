"""Regression tests for ``tools/fetch_valuation.py``."""

from __future__ import annotations

import importlib.util
import sys
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "fetch_valuation.py"


@pytest.fixture()
def valuation():
    spec = importlib.util.spec_from_file_location("fetch_valuation_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _root(valuation, tmp_path, monkeypatch, codes=("600519",)):
    stocks = tmp_path / "wide" / "stocks"
    stocks.mkdir(parents=True)
    for code in codes:
        (stocks / f"{code}.parquet").write_bytes(b"")
    out = tmp_path / "valuation"
    per_asset = out / "stocks"
    per_asset.mkdir(parents=True)
    monkeypatch.setattr(valuation, "STOCKS", stocks)
    monkeypatch.setattr(valuation, "OUT", out)
    monkeypatch.setattr(valuation, "PER_ASSET", per_asset)
    monkeypatch.setattr(valuation, "MERGED", out / "valuation_wide.parquet")
    return out, per_asset


def _fake_akshare(pe_rows, pb_rows):
    def fake(symbol, indicator, period):
        rows = pe_rows if indicator.startswith("市盈率") else pb_rows
        return pd.DataFrame({
            "date": [row[0] for row in rows],
            "value": [row[1] for row in rows],
        })

    return SimpleNamespace(stock_zh_valuation_baidu=fake)


def test_refresh_appends_only_new_rows_and_drops_future(
    valuation, tmp_path, monkeypatch,
):
    out, per_asset = _root(valuation, tmp_path, monkeypatch)
    existing = pd.DataFrame({
        "date": [pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-24")],
        "pe_ttm": [10.0, 10.5], "pb_mrq": [1.0, 1.1], "asset": ["600519"] * 2,
    })
    existing.to_parquet(per_asset / "600519.parquet", index=False)
    today = dt.date(2026, 10, 28)
    rows = [
        (pd.Timestamp("2026-10-24"), 99.0),
        (pd.Timestamp("2026-10-28"), 20.43),
        (pd.Timestamp("2026-10-31"), 30.0),  # future row must be dropped
    ]
    monkeypatch.setitem(
        sys.modules, "akshare", _fake_akshare(rows, rows),
    )
    frame = valuation.fetch("600519")
    appended = valuation.refresh_asset("600519", frame, today)
    assert appended == 1
    stored = pd.read_parquet(per_asset / "600519.parquet")
    dates = pd.to_datetime(stored["date"]).dt.date.tolist()
    assert dt.date(2026, 10, 28) in dates
    assert dt.date(2026, 10, 31) not in dates
    assert stored[stored["date"] == pd.Timestamp("2026-10-24")]["pe_ttm"].iloc[0] == 10.5
    # idempotent re-run appends nothing
    assert valuation.refresh_asset("600519", frame, today) == 0


def test_main_exits_zero_and_rebuilds_merged(
    valuation, tmp_path, monkeypatch,
):
    out, per_asset = _root(valuation, tmp_path, monkeypatch)
    monkeypatch.setitem(
        sys.modules, "akshare",
        _fake_akshare(
            [(pd.Timestamp.today().normalize(), 12.0)],
            [(pd.Timestamp.today().normalize(), 1.2)],
        ),
    )
    monkeypatch.setattr(sys, "argv", ["fetch_valuation.py"])
    assert valuation.main() == 0
    assert (out / "valuation_wide.parquet").is_file()
    vintage = __import__("json").loads((out / "vintage.json").read_text())
    assert vintage["assets"] == 1


def test_main_flags_widespread_failures(valuation, tmp_path, monkeypatch):
    _root(valuation, tmp_path, monkeypatch, codes=("600519", "000001"))

    def broken(**kwargs):
        raise ConnectionError("vendor down")

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_valuation_baidu=broken),
    )
    monkeypatch.setattr(sys, "argv", ["fetch_valuation.py"])
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    assert valuation.main() == 2


def test_rebuild_merged_ignores_out_of_universe_assets(
    valuation, tmp_path, monkeypatch,
):
    out, per_asset = _root(valuation, tmp_path, monkeypatch, codes=("600519",))
    for code in ("600519", "999999"):
        pd.DataFrame({
            "date": [pd.Timestamp("2026-10-28")],
            "pe_ttm": [10.0], "pb_mrq": [1.0], "asset": [code],
        }).to_parquet(per_asset / f"{code}.parquet", index=False)
    total = valuation.rebuild_merged()
    assert total == 1
    merged = pd.read_parquet(out / "valuation_wide.parquet").reset_index()
    assert set(merged["asset"]) == {"600519"}
