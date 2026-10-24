"""Tests for the sweep endpoint's DSR / BH multiple-testing correction.

The sweep grid reports N trials and picks the max-Sharpe row. Without a
correction that selection is biased; the endpoint now reports the deflated
p-value (DSR) plus a Benjamini-Hochberg mask per row, and warns when the
selected "best" does not survive either test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_platform.api import routes
from quant_platform.api.schemas import SweepResult


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _oscillating_returns(mean_daily: float, amplitude: float = 0.01, n: int = 500) -> np.ndarray:
    """Deterministic series with an exact mean and (near-exact) std.

    Alternating +A/-A keeps the sample mean at ``mean_daily`` and the sample
    std at ~``amplitude``, so the Sharpe and p-values do not depend on a
    random draw.
    """
    assert n % 2 == 0
    block = np.array([mean_daily + amplitude, mean_daily - amplitude])
    return np.tile(block, n // 2)


def _install_fakes(monkeypatch, series: np.ndarray, fail_optimizers: tuple[str, ...] = (),
                   no_returns_optimizers: tuple[str, ...] = ()) -> None:
    config = SimpleNamespace(universe=SimpleNamespace(n_stocks=0))
    monkeypatch.setattr(routes, "load_config", lambda _path=None: config)
    monkeypatch.setattr(
        routes, "_load_data",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), None, None, None, None),
    )
    monkeypatch.setattr(
        routes, "_compute_factors", lambda *a, **k: ({}, {}, None, None),
    )
    monkeypatch.setattr(routes, "_generate_signal", lambda *a, **k: None)

    def fake_backtest(config, signal, prices, returns, benchmark, sector_map,
                      financials, optimizer_override=None,
                      frequency_override=None, gates=None):
        if optimizer_override in fail_optimizers:
            raise RuntimeError("synthetic backtest failure")
        sharpe = float(np.mean(series) / np.std(series, ddof=1) * np.sqrt(252))
        if optimizer_override in no_returns_optimizers:
            sharp = 9.9  # attract the "best" pick without a return series
            daily = None
        else:
            sharp = sharpe
            daily = pd.Series(series)
        return {
            "summary": {
                "sharpe_ratio": sharp,
                "annual_return": 0.10,
                "max_drawdown": -0.20,
                "sortino_ratio": 1.0,
                "win_rate": 0.5,
            },
            "daily_returns": daily,
        }

    monkeypatch.setattr(routes, "_run_backtest", fake_backtest)


def _post_sweep(client, *, optimizers, frequencies, n_stocks_list):
    resp = client.post("/api/sweep", json={
        "optimizers": optimizers,
        "frequencies": frequencies,
        "n_stocks_list": n_stocks_list,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_sweep_reports_dsr_bh_and_no_note_when_best_passes(client, monkeypatch):
    series = _oscillating_returns(mean_daily=0.002)  # annualized Sharpe ~3.2
    _install_fakes(monkeypatch, series)

    body = _post_sweep(
        client, optimizers=["equal_weight"], frequencies=["monthly"],
        n_stocks_list=[200],
    )

    assert body["num_trials"] == 1
    row = body["table"][0]
    assert isinstance(row["DSR p"], float)
    assert row["BH Significant"] is True

    assert body["best_dsr"]["config"] == "equal_weight/monthly/200"
    assert body["best_dsr"]["significant"] is True
    assert body["best_dsr"]["deflated_p"] < 0.05
    # N=1 means no E[max SR] deflation: the expected max is exactly zero.
    assert body["best_dsr"]["expected_max"] == 0.0
    assert body["selection_note"] is None
    # best_params keeps its original max-Sharpe meaning.
    assert body["best_params"]["optimizer"] == "equal_weight"


def test_more_trials_flip_the_selection_verdict(client, monkeypatch):
    """The "can go red" control: a borderline Sharpe passes as one trial but
    fails once the grid's 30 trials are accounted for."""
    series = _oscillating_returns(mean_daily=0.0009)  # raw p ~0.02
    _install_fakes(monkeypatch, series)

    single = _post_sweep(
        client, optimizers=["equal_weight"], frequencies=["monthly"],
        n_stocks_list=[200],
    )
    assert single["num_trials"] == 1
    assert single["best_dsr"]["raw_p"] < 0.05
    assert single["best_dsr"]["significant"] is True
    assert single["selection_note"] is None

    grid = _post_sweep(
        client,
        optimizers=["equal_weight", "mean_variance", "risk_parity"],
        frequencies=["monthly", "weekly"],
        n_stocks_list=[100, 200, 300, 400, 500],
    )
    assert grid["num_trials"] == 30
    assert grid["best_dsr"]["raw_p"] < 0.05       # undeflated test still passes
    assert grid["best_dsr"]["deflated_p"] > 0.05  # deflation kills it
    assert grid["best_dsr"]["significant"] is False
    # BH consumes the raw p-values, not the deflated ones: with 30 trials the
    # largest rank threshold is alpha, so the shared raw p still clears BH.
    assert grid["best_dsr"]["bh_significant"] is True
    assert grid["best_dsr"]["dsr_significant"] is False
    assert grid["best_dsr"]["expected_max"] > 0
    assert grid["selection_note"]
    assert "does not survive multiple-testing correction" in grid["selection_note"]

    # Every successful row carries both new columns.
    for row in grid["table"]:
        assert "DSR p" in row
        assert isinstance(row["BH Significant"], bool)


def test_failed_configs_are_not_counted_as_trials(client, monkeypatch):
    series = _oscillating_returns(mean_daily=0.002)
    _install_fakes(monkeypatch, series, fail_optimizers=("mean_variance",))

    body = _post_sweep(
        client, optimizers=["equal_weight", "mean_variance"],
        frequencies=["monthly"], n_stocks_list=[200],
    )

    assert body["num_trials"] == 1
    by_optimizer = {row["Optimizer"]: row for row in body["table"]}
    assert by_optimizer["mean_variance"]["Sharpe"] == "ERR"
    assert "DSR p" not in by_optimizer["mean_variance"]
    assert by_optimizer["equal_weight"]["BH Significant"] is True
    assert body["best_dsr"]["config"] == "equal_weight/monthly/200"


def test_successful_config_without_daily_returns_is_kept_but_flagged(client, monkeypatch):
    """A successful backtest with no usable series must not be dropped or
    turned into an ERR row; if it is the best row, the note says it could
    not be validated and it is not counted as a DSR trial."""
    series = _oscillating_returns(mean_daily=0.002)
    _install_fakes(monkeypatch, series, no_returns_optimizers=("mean_variance",))

    body = _post_sweep(
        client, optimizers=["equal_weight", "mean_variance"],
        frequencies=["monthly"], n_stocks_list=[200],
    )

    # Only the config with a usable series contributes a trial.
    assert body["num_trials"] == 1
    by_optimizer = {row["Optimizer"]: row for row in body["table"]}
    missing = by_optimizer["mean_variance"]
    assert missing["Sharpe"] == 9.9          # still a valid table row
    assert "DSR p" not in missing            # but no invented p-value

    # mean_variance wins on table Sharpe; its verdict is unavailable.
    assert body["best_params"]["optimizer"] == "mean_variance"
    assert body["best_dsr"] is None
    assert body["selection_note"]
    assert "could not be validated" in body["selection_note"]
    # The other row still carries its own correction.
    assert by_optimizer["equal_weight"]["BH Significant"] is True


def test_sweep_result_schema_is_backward_compatible():
    result = SweepResult(table=[])
    assert result.best_params is None
    assert result.num_trials is None
    assert result.best_dsr is None
    assert result.selection_note is None
