"""CLI sweep must reuse the shared DSR/BH helper and print the verdict."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import quant_platform.main as main


def _oscillating_returns(mean_daily: float, amplitude: float = 0.01, n: int = 500) -> np.ndarray:
    assert n % 2 == 0
    return np.tile(np.array([mean_daily + amplitude, mean_daily - amplitude]), n // 2)


def _install_fakes(monkeypatch, series) -> None:
    config = SimpleNamespace(
        portfolio=SimpleNamespace(optimizer="equal_weight"),
        backtest=SimpleNamespace(rebalance_frequency="monthly"),
        universe=SimpleNamespace(n_stocks=0),
    )
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

    def fake_backtest(*args, **kwargs):
        return {
            "summary": {
                "sharpe_ratio": 0.5,
                "annual_return": 0.10,
                "max_drawdown": -0.20,
                "sortino_ratio": 1.0,
                "win_rate": 0.5,
            },
            "daily_returns": pd.Series(series),
        }

    monkeypatch.setattr(main, "_run_backtest", fake_backtest)


def _args(**overrides):
    base = dict(
        config=None,
        optimizers="equal_weight",
        frequencies="monthly",
        n_stocks="200",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_sweep_prints_columns_and_significant_verdict(monkeypatch, capsys):
    _install_fakes(monkeypatch, _oscillating_returns(0.002))

    rc = main.cmd_sweep(_args())

    out = capsys.readouterr().out
    assert rc == 0
    assert "DSR p" in out and "BH Significant" in out
    assert "trials=1 best=equal_weight/monthly/200" in out
    assert "raw_p=" in out and "dsr_p=" in out
    assert "significant=True" in out
    assert "WARNING" not in out


def test_cmd_sweep_warns_when_correction_rejects_the_best(monkeypatch, capsys):
    # The same borderline Sharpe fails once it is selected from 30 trials.
    _install_fakes(monkeypatch, _oscillating_returns(0.0009))

    rc = main.cmd_sweep(_args(
        optimizers="equal_weight,mean_variance,risk_parity",
        frequencies="monthly,weekly",
        n_stocks="100,200,300,400,500",
    ))

    out = capsys.readouterr().out
    assert rc == 0
    assert "trials=30 best=equal_weight/monthly/100" in out
    assert "significant=False" in out
    assert "WARNING: the selected best does not survive multiple-testing" in out
