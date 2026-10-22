"""CLI sweep must refuse to touch holdout dates before running a backtest."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_platform.main as main
from quant_platform.utils.config import load_config


def _raw_config(start="", end="", data_window=("2021-01-01", "2025-12-31")):
    return {
        "data": {"start_date": data_window[0], "end_date": data_window[1]},
        "research": {"holdout": {"start_date": start, "end_date": end}},
    }


def _args(**overrides):
    base = dict(
        config=None,
        optimizers="equal_weight",
        frequencies="monthly",
        n_stocks="200",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_sweep_refuses_overlap_before_loading_data(monkeypatch, capsys):
    config = load_config(raw=_raw_config(
        "2024-07-01", "2024-12-31",
        data_window=("2021-01-01", "2024-07-01"),
    ))
    monkeypatch.setattr(main, "load_config", lambda _path=None: config)

    calls: list[str] = []

    def must_not_run(*args, **kwargs):
        calls.append("called")
        raise AssertionError("sweep must not start on holdout dates")

    monkeypatch.setattr(main, "_load_data", must_not_run)
    monkeypatch.setattr(main, "_run_backtest", must_not_run)

    rc = main.cmd_sweep(_args())

    captured = capsys.readouterr()
    assert rc != 0
    assert "holdout" in captured.err
    assert "disable the holdout" in captured.err
    assert calls == []
    # Nothing may be written to stdout either: no table, no warning-after-run.
    assert "PARAMETER SWEEP RESULTS" not in captured.out


def test_cmd_sweep_reports_invalid_holdout_config(monkeypatch, capsys):
    config = SimpleNamespace(
        data=SimpleNamespace(start_date="2021-01-01", end_date="2025-12-31"),
        research=SimpleNamespace(
            holdout=SimpleNamespace(
                start_date="2024-12-31", end_date="2024-01-01",
            ),
        ),
    )
    monkeypatch.setattr(main, "load_config", lambda _path=None: config)
    monkeypatch.setattr(
        main, "_load_data",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not load data")),
    )

    rc = main.cmd_sweep(_args())

    captured = capsys.readouterr()
    assert rc != 0
    assert "strictly before" in captured.err


def test_cmd_sweep_runs_when_window_is_disjoint(monkeypatch, capsys):
    config = load_config(raw=_raw_config(
        "2024-07-01", "2024-12-31",
        data_window=("2021-01-01", "2024-06-30"),
    ))
    monkeypatch.setattr(main, "load_config", lambda _path=None: config)
    monkeypatch.setattr(
        main, "_load_data",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), None, None, None, None),
    )
    monkeypatch.setattr(
        main, "_compute_factors", lambda *a, **k: ({}, {}, None, None),
    )
    monkeypatch.setattr(main, "_generate_signal", lambda *a, **k: None)
    monkeypatch.setattr(main, "_execution_gates", lambda metadata: None)
    series = np.tile(np.array([0.002 + 0.01, 0.002 - 0.01]), 250)
    monkeypatch.setattr(
        main, "_run_backtest",
        lambda *a, **k: {
            "summary": {
                "sharpe_ratio": 0.5, "annual_return": 0.1,
                "max_drawdown": -0.2, "sortino_ratio": 1.0, "win_rate": 0.5,
            },
            "daily_returns": pd.Series(series),
        },
    )

    rc = main.cmd_sweep(_args())

    out = capsys.readouterr().out
    assert rc == 0
    assert "PARAMETER SWEEP RESULTS" in out
