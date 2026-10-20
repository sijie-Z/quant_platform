"""CLI contracts: every registered command runs, and `run` writes its artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_platform import main as cli


def test_every_registered_command_has_a_dispatch_branch():
    """Structural canary: `check-lookahead` was registered but undispatched.

    A command that parses and then falls through to `print_help(); return 1`
    is invisible to `--help` checks. This compares the parser registrations
    with the dispatch chain in the module source.
    """
    source = Path(cli.__file__).read_text(encoding="utf-8")
    registered = set(re.findall(
        r"subparsers\.add_parser\(\s*[\"']([^\"']+)", source,
    ))
    dispatched = set(re.findall(
        r"args\.command\s*==\s*[\"']([^\"']+)", source,
    ))

    assert registered, "no subcommands found -- the scanner is wrong"
    assert registered <= dispatched, (
        f"registered but never dispatched: {sorted(registered - dispatched)}"
    )


def test_check_lookahead_is_dispatched(monkeypatch, tmp_path):
    """The parser registered the command but main() had no dispatch branch.

    It fell through to `print_help(); return 1`, so the documented lookahead
    detector never ran. Patch the handler and assert dispatch reaches it.
    """
    called = {}

    def fake(args):
        called["threshold"] = args.threshold
        called["config"] = args.config
        return 0

    monkeypatch.setattr(cli, "cmd_check_lookahead", fake)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "check-lookahead",
        "--config", str(tmp_path / "cfg.yaml"),
        "--threshold", "0.001",
    ])

    assert cli.main() == 0
    assert called["threshold"] == pytest.approx(0.001)
    assert called["config"].endswith("cfg.yaml")


def _fake_config(tmp_path):
    return SimpleNamespace(
        output=SimpleNamespace(results_dir=str(tmp_path)),
        risk=SimpleNamespace(var=SimpleNamespace(confidence=0.95, horizon=1)),
    )


def _fake_results(periods: int = 80):
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2024-01-01", periods=periods)
    returns = pd.Series(rng.normal(0.0005, 0.01, periods), index=index)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    return {
        "daily_returns": returns,
        "benchmark_returns": returns * 0.5,
        "portfolio_values": (1 + returns).cumprod() * 10_000_000,
        "weights_history": {index[-1]: weights},
        "summary": {
            "total_return": 0.04,
            "annual_return": 0.12,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.05,
            "optimizer": "mean_variance",
            "n_rebalances": 4,
        },
    }


def _fake_ic_results():
    return {
        "momentum_1m": {
            "mean_ic": 0.02, "std_ic": 0.01, "icir": 0.3,
            "ic_positive_ratio": 0.55, "n_obs": 70,
        },
        "empty_factor": {
            "mean_ic": float("nan"), "std_ic": float("nan"),
            "icir": float("nan"), "ic_positive_ratio": float("nan"),
            "n_obs": 0,
        },
    }


def test_run_writes_html_and_metrics_json(tmp_path):
    """`run` promised an HTML report and `research report` needed metrics.json;
    neither was ever written."""
    metadata = pd.DataFrame({"sector": {"A": "Tech", "B": "Financials"}})

    html_path, metrics_path = cli._write_run_artifacts(
        _fake_results(), _fake_ic_results(), metadata, _fake_config(tmp_path),
    )

    assert html_path.exists() and html_path.stat().st_size > 0
    assert "<html" in html_path.read_text(encoding="utf-8").lower()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["performance"]["total_return"] == pytest.approx(0.04)
    assert len(metrics["daily_returns"]) == 80
    factor_names = {f["name"] for f in metrics["factors"]}
    assert factor_names == {"momentum_1m", "empty_factor"}
    # Non-finite values are JSON null, not the non-standard NaN token.
    empty = next(f for f in metrics["factors"] if f["name"] == "empty_factor")
    assert empty["mean_ic"] is None
    assert metrics["factor_contributions"] == {}


def test_research_report_falls_back_to_daily_returns_csv(tmp_path, capsys):
    """A result directory written before metrics.json existed must still work."""
    rng = np.random.default_rng(6)
    index = pd.bdate_range("2024-01-01", periods=60)
    pd.Series(rng.normal(0, 0.01, 60), index=index).to_csv(
        tmp_path / "daily_returns.csv",
    )

    args = SimpleNamespace(results_dir=str(tmp_path), subcommand="report")
    assert cli.cmd_research(args) == 0

    out = capsys.readouterr().out
    assert "computed performance from daily_returns.csv" in out
    assert "sharpe_ratio" in out


def test_screen_applies_rules_to_raw_values(monkeypatch, capsys):
    """`pe_ratio=lt:30` must mean PE < 30, not z-score < 30.

    The CLI passed the processed (standardized) panel, so every stock passed
    every threshold and the printed values were z-scores.
    """
    index = pd.bdate_range("2024-01-01", periods=5)
    index_assets = ["LOW", "MID", "HIGH"]
    raw_pe = pd.DataFrame(
        [[10.0, 20.0, 40.0]] * 5, index=index, columns=index_assets,
    )
    processed_pe = pd.DataFrame(
        [[0.5, 0.5, 0.5]] * 5, index=index, columns=index_assets,
    )
    raw = {"pe_ratio": raw_pe}
    processed = {"pe_ratio": processed_pe}

    prices = pd.DataFrame(
        np.ones((5, 3)), index=index, columns=index_assets,
    )
    monkeypatch.setattr(
        cli, "_load_data",
        lambda config, use_baostock=False: (
            prices, prices.pct_change().fillna(0), None, None, None, None,
        ),
    )

    def fake_compute(*args, return_raw=False, **kwargs):
        assert return_raw is True
        return processed, {}, None, None, raw

    monkeypatch.setattr(cli, "_compute_factors", fake_compute)

    args = SimpleNamespace(
        config=str(Path(cli.__file__).parent / "config" / "default.yaml"),
        rules="pe_ratio=lt:30",
        use_baostock=False,
        logic=None,
        min_stocks=1,
        max_stocks=None,
        output=None,
    )

    assert cli.cmd_screen(args) == 0
    out = capsys.readouterr().out
    assert "raw factor units" in out
    assert "Passing: 2 / 3" in out
    assert "HIGH" not in out


def _ml_args(subcommand: str):
    return SimpleNamespace(
        subcommand=subcommand,
        model="lightgbm",
        splits=3,
        config=str(Path(cli.__file__).parent / "config" / "default.yaml"),
    )


def _stub_pipeline(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(np.ones((5, 2)), index=index, columns=["A", "B"])
    monkeypatch.setattr(
        cli, "_load_data",
        lambda config: (prices, prices, None, None, None, None),
    )
    monkeypatch.setattr(
        cli, "_compute_factors", lambda *a, **k: ({}, {}, None, None),
    )


def test_ml_train_exits_nonzero_without_enough_data(monkeypatch, capsys):
    """A zero-sample model used to print IC 0.0000 and exit 0."""
    import quant_platform.alpha.ml_signal as ml

    class FakeGenerator:
        def __init__(self, config=None):
            pass

        def train(self, factors, returns):
            raise ml.InsufficientTrainingData("0 samples available")

    monkeypatch.setattr(ml, "MLSignalGenerator", FakeGenerator)
    _stub_pipeline(monkeypatch)

    assert cli.cmd_ml(_ml_args("train")) == 1
    assert "ML training failed" in capsys.readouterr().out


def test_ml_signal_exits_nonzero_on_an_empty_signal(monkeypatch, capsys):
    import quant_platform.alpha.ml_signal as ml

    class FakeGenerator:
        def __init__(self, config=None):
            pass

        def generate(self, factors, returns):
            return pd.DataFrame(
                np.nan, index=pd.bdate_range("2024-01-01", periods=2),
                columns=["A", "B"],
            )

    monkeypatch.setattr(ml, "MLSignalGenerator", FakeGenerator)
    _stub_pipeline(monkeypatch)

    assert cli.cmd_ml(_ml_args("signal")) == 1
    assert "ML signal is empty" in capsys.readouterr().out
