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


# ---------------------------------------------------------------------------
# CLI lifecycle (S14)
# ---------------------------------------------------------------------------

def test_cache_and_config_require_a_subcommand(monkeypatch):
    """`cache`/`config` with no subcommand used to exit 0 doing nothing."""
    for argv in (["main.py", "cache"], ["main.py", "config"]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 2


def test_cache_dir_is_accepted_after_the_subcommand(monkeypatch, tmp_path):
    """`cache clear --cache-dir X` was rejected (option was parent-only)."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", [
        "main.py", "cache", "clear", "--cache-dir", str(cache_dir),
    ])

    assert cli.main() == 0
    assert cache_dir.exists()


def test_cache_prune_subcommand_runs(monkeypatch, tmp_path):
    from quant_platform.utils.cache import PipelineCache

    cache_dir = tmp_path / "cache"
    PipelineCache(cache_dir).save_stage("data", "k", {"v": 1})

    monkeypatch.setattr(sys, "argv", [
        "main.py", "cache", "prune", "--cache-dir", str(cache_dir),
        "--max-entries", "1",
    ])
    assert cli.main() == 0


def test_missing_config_prints_a_message_not_a_traceback(
    monkeypatch, capsys, tmp_path,
):
    """A missing config printed a full traceback and exit 1."""
    monkeypatch.setattr(sys, "argv", [
        "main.py", "run", "--config", str(tmp_path / "missing.yaml"),
    ])

    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "not found" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_config_prints_an_actionable_error(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("universe:\n  n_stocks: -5\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "main.py", "run", "--config", str(bad), "--no-cache",
    ])

    assert cli.main() == 1
    err = capsys.readouterr().err
    assert "Config error" in err
    assert "n_stocks" in err
    assert "Traceback" not in err


def test_trade_days_zero_is_a_usage_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "trade", "--days", "0"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_trade_parser_accepts_qmt_live(monkeypatch):
    seen = {}

    def fake_cmd_trade(args):
        seen["broker"] = args.broker
        return 0

    monkeypatch.setattr(cli, "cmd_trade", fake_cmd_trade)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "trade", "--broker", "qmt_live", "--days", "1",
    ])

    assert cli.main() == 0
    assert seen["broker"] == "qmt_live"


def _trade_args(**overrides):
    args = SimpleNamespace(
        broker="qmt", days=1, universe="600519", cash=10_000_000,
        no_dual_track=True, seed=42, config=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _session_report(**overrides):
    from quant_platform.trading.live_runner import SessionReport

    values = dict(
        session_id="s1", start_date="2026-01-01", end_date="2026-01-01",
        days_traded=1, total_orders=3, total_fills=0, initial_capital=1.0,
        final_value=1.0, total_return_pct=0.0, annualized_return_pct=0.0,
        sharpe_ratio=0.0, max_drawdown_pct=0.0, avg_daily_volume=0.0,
        broker_type="qmt", status="failed",
    )
    values.update(overrides)
    return SessionReport(**values)


def test_cmd_trade_fails_when_nothing_fills(monkeypatch, capsys):
    """S5: a session where every order was rejected still exited 0."""
    import quant_platform.trading.live_runner as lr

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def set_universe(self, codes):
            pass

        def run(self, days, seed):
            return _session_report()

    monkeypatch.setattr(lr, "LiveRunner", FakeRunner)

    assert cli.cmd_trade(_trade_args()) == 1
    assert "no orders filled" in capsys.readouterr().err


def test_cmd_trade_fails_closed_when_the_broker_cannot_connect(
    monkeypatch, capsys,
):
    import quant_platform.trading.live_runner as lr

    class FakeRunner:
        def __init__(self, **kwargs):
            raise RuntimeError("Broker 'qmt' failed to connect")

    monkeypatch.setattr(lr, "LiveRunner", FakeRunner)

    assert cli.cmd_trade(_trade_args()) == 1
    assert "failed to connect" in capsys.readouterr().err


def test_cmd_trade_rejects_non_positive_days(capsys):
    assert cli.cmd_trade(_trade_args(days=0)) == 2
    assert "days" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Data provenance (S8)
# ---------------------------------------------------------------------------

def test_load_data_fails_closed_when_a_real_provider_is_unavailable(monkeypatch):
    import quant_platform.data.providers.baostock_provider as baostock_module
    from quant_platform.utils.config import load_config

    def unavailable():
        raise RuntimeError("no network")

    monkeypatch.setattr(baostock_module, "BaostockDataProvider", unavailable)
    config = load_config(raw={
        "universe": {"n_stocks": 5},
        "data": {"provider": "baostock"},
    })

    with pytest.raises(cli.DataProviderUnavailableError) as exc_info:
        cli._load_data(config)

    message = str(exc_info.value)
    assert "baostock" in message
    assert "--allow-synthetic-fallback" in message


def test_load_data_allows_explicit_fallback_and_records_provenance(monkeypatch):
    import quant_platform.data.pipeline as pipeline_module
    import quant_platform.data.providers.baostock_provider as baostock_module
    import quant_platform.data.providers.synthetic as synthetic_module
    from quant_platform.utils.config import load_config

    def unavailable():
        raise RuntimeError("no network")

    class FakeSynthetic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePipeline:
        def __init__(self, **kwargs):
            self.metadata = None
            self.returns = None
            self.benchmark = None
            self.financials = None

        def run(self):
            pass

        def get_factor_prices(self):
            return pd.DataFrame(
                {"A": [1.0]}, index=pd.DatetimeIndex(["2021-03-31"]),
            )

        def get_turnover(self):
            return None

        def get_sector_panel(self):
            return None

        def has_industry_history(self):
            return False

    monkeypatch.setattr(baostock_module, "BaostockDataProvider", unavailable)
    monkeypatch.setattr(synthetic_module, "SyntheticDataProvider", FakeSynthetic)
    monkeypatch.setattr(pipeline_module, "DataPipeline", FakePipeline)
    config = load_config(raw={
        "universe": {"n_stocks": 3},
        "data": {
            "provider": "baostock",
            "allow_synthetic_fallback": True,
            "start_date": "2021-01-01",
            "end_date": "2021-03-31",
        },
    })

    cli._load_data(config)

    assert cli._LAST_DATA_PROVIDER == "synthetic"
    provenance = cli._LAST_DATA_PROVENANCE
    assert provenance["requested_provider"] == "baostock"
    assert provenance["effective_provider"] == "synthetic"
    assert provenance["synthetic"] is True
    assert provenance["data_date"] == "2021-03-31"


def test_run_artifacts_record_provenance(tmp_path):
    """metrics.json and the HTML report must carry the data provenance."""
    provenance = {
        "requested_provider": "tushare",
        "effective_provider": "synthetic",
        "synthetic": True,
        "start_date": "2021-01-01",
        "end_date": "2025-12-31",
        "data_date": "2025-12-31",
    }
    html_path, metrics_path = cli._write_run_artifacts(
        _fake_results(), _fake_ic_results(), None,
        _fake_config(tmp_path), provenance=provenance,
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["provenance"] == provenance
    html = html_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in html
    assert "tushare" in html


# ---------------------------------------------------------------------------
# Per-run artifacts (S15)
# ---------------------------------------------------------------------------

def test_run_directories_and_publication_are_consistent(tmp_path):
    """Two concurrent runs write separate dirs; top-level copies always come
    from exactly one complete run (never a mix)."""
    import threading

    from quant_platform.reporting.dashboard import (
        create_run_dir,
        publish_run_artifacts,
        read_manifest,
        resolve_run_dir,
        write_manifest,
    )

    runs = []
    for tag in ("a", "b"):
        run_dir = create_run_dir(tmp_path, tag, "cafebabe")
        (run_dir / "daily_returns.csv").write_text(tag, encoding="utf-8")
        manifest = {"run_id": tag, "artifacts": ["daily_returns.csv"]}
        write_manifest(run_dir, manifest)
        runs.append((tag, run_dir, manifest))

    def publish(item):
        _tag, run_dir, manifest = item
        publish_run_artifacts(tmp_path, run_dir, manifest)

    threads = [threading.Thread(target=publish, args=(item,)) for item in runs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    content = (tmp_path / "daily_returns.csv").read_text(encoding="utf-8")
    pointer = json.loads(
        (tmp_path / "latest_manifest.json").read_text(encoding="utf-8")
    )
    pointed_dir = tmp_path / pointer["run_dir"]
    pointed_tag = read_manifest(pointed_dir)["run_id"]
    # The pointer and the top-level copy describe the same run.
    assert content == pointed_tag
    assert resolve_run_dir(tmp_path) == pointed_dir


def test_analyze_reads_the_run_manifest(tmp_path, capsys):
    from quant_platform.reporting.dashboard import create_run_dir, write_manifest

    run_dir = create_run_dir(tmp_path, "run1", "deadbeef")
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-01-01", periods=60)
    pd.Series(rng.normal(0, 0.01, 60), index=index).to_csv(
        run_dir / "daily_returns.csv",
    )
    write_manifest(run_dir, {"run_id": "run1", "artifacts": []})

    args = SimpleNamespace(results_dir=str(tmp_path))
    assert cli.cmd_analyze(args) == 0

    out = capsys.readouterr().out
    assert "Analyzing run" in out
    assert run_dir.name in out
