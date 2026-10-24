"""S19: the monitor script used to fail at import outside the repo root,
monitor a fixed 2025 sample, overwrite its history and always exit 0."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from quant_platform.monitoring import run_monitor

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_from_a_clean_cwd_does_not_touch_results(tmp_path):
    """`results/monitor.log` was opened at import, relative to the cwd."""
    script = REPO_ROOT / "monitoring" / "run_monitor.py"
    code = (
        "import runpy; "
        f"runpy.run_path(r'{script}', run_name='not_main')"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "FileNotFoundError" not in completed.stderr
    assert not (tmp_path / "results").exists()


def test_data_window_ends_today_by_default(monkeypatch):
    monkeypatch.delenv("QUANT_MONITOR_END_DATE", raising=False)
    monkeypatch.delenv("QUANT_MONITOR_START_DATE", raising=False)

    start, end = run_monitor._data_window()

    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    assert end == today
    assert start == "2018-01-01"


def test_data_window_is_configurable(monkeypatch):
    monkeypatch.setenv("QUANT_MONITOR_START_DATE", "2020-01-01")
    monkeypatch.setenv("QUANT_MONITOR_END_DATE", "2020-12-31")

    assert run_monitor._data_window() == ("2020-01-01", "2020-12-31")


def test_summary_appends_instead_of_overwriting(tmp_path):
    path = tmp_path / "monitor_summary.csv"
    row = {"date": "2026-01-01", "sharpe": 1.0}

    run_monitor._append_summary(pd.DataFrame([row]), path=path)
    run_monitor._append_summary(pd.DataFrame([{**row, "date": "2026-01-02"}]), path=path)

    stored = pd.read_csv(path)
    assert list(stored["date"]) == ["2026-01-01", "2026-01-02"]


class _FakePipeline:
    returns = None

    def __init__(self, **_kwargs):
        pass

    def run(self):
        pass

    def get_close(self):
        return self.returns.apply(lambda column: 10 + column.cumsum())


class _FakeTrader:
    totals: list[float] = []

    def __init__(self, **_kwargs):
        self.equity_curve = [{"total": total} for total in self.totals]
        self.trades = []

    def load_data(self, returns, prices):
        pass

    def run(self):
        pass


def _patch_main(monkeypatch, tmp_path, totals, returns):
    monkeypatch.setattr(run_monitor, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_monitor, "_LOG_FILE", tmp_path / "monitor.log")
    pipeline_cls = type("FakePipeline", (_FakePipeline,), {"returns": returns})
    monkeypatch.setattr(run_monitor, "DataPipeline", pipeline_cls)
    monkeypatch.setattr(run_monitor, "BaostockDataProvider", lambda **_kw: object())
    trader_cls = type("FakeTrader", (_FakeTrader,), {"totals": totals})
    monkeypatch.setattr(run_monitor, "ReversalPaperTrader", trader_cls)


def _returns(n_days=60, n_assets=2, drift=0.001):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(
        rng.normal(drift, 0.005, (n_days, n_assets)),
        index=dates,
        columns=[f"S{i}" for i in range(n_assets)],
    )


def test_main_exits_nonzero_when_alerts_are_raised(monkeypatch, tmp_path):
    _patch_main(
        monkeypatch, tmp_path,
        totals=[100.0, 99.0, 98.0, 97.0, 96.0],
        returns=_returns(drift=-0.01),
    )

    assert run_monitor.main() == 2


def test_main_exits_zero_when_nominal(monkeypatch, tmp_path):
    _patch_main(
        monkeypatch, tmp_path,
        totals=[100.0, 101.0, 100.5, 101.5, 102.0],
        returns=_returns(drift=0.002),
    )

    assert run_monitor.main() == 0


def test_main_exits_nonzero_when_data_loading_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(run_monitor, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_monitor, "_LOG_FILE", tmp_path / "monitor.log")

    class BrokenPipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            raise ConnectionError("provider down")

    monkeypatch.setattr(run_monitor, "DataPipeline", BrokenPipeline)
    monkeypatch.setattr(run_monitor, "BaostockDataProvider", lambda **_kw: object())

    assert run_monitor.main() == 1
