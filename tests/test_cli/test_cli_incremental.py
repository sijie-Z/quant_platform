"""CLI surface for incremental updates and the D32 freshness gate.

Covers: ``run --incremental`` argument plumbing, the one-line summary,
provider-capability fallback, ``cache refresh --incremental``, the
exchange-session freshness gate (warn for research, fail closed for decisive
runs) and the provenance/freshness record in ``metrics.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_platform.main as cli
from quant_platform.utils.cache import PipelineCache
from quant_platform.utils.config import load_config
from tests.test_cli.test_cli_contracts import (
    _fake_config,
    _fake_ic_results,
    _fake_results,
)


def _config(**data_overrides):
    raw = {
        "data": {
            "provider": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-01-09",
            **data_overrides,
        },
        "universe": {"n_stocks": 2},
    }
    return load_config(raw=raw)


# ----------------------------------------------------------------------
# Freshness gate
# ----------------------------------------------------------------------


def test_freshness_counts_exchange_sessions_and_warns_for_research(caplog):
    config = _config(max_staleness_days=1)
    provenance = {
        "data_date": "2024-01-05",       # Friday
        "end_date": "2024-01-09",        # Tuesday
        "incremental": {"new_sessions": 2},
    }
    with caplog.at_level(logging.WARNING, logger="quant_platform.main"):
        freshness = cli._enforce_freshness(
            provenance, config, context="research", fail_closed=False,
        )

    # Monday + Tuesday are two exchange sessions; the weekend does not count.
    assert freshness["data_age_days"] == 2
    assert freshness["stale"] is True
    assert freshness["max_staleness_days"] == 1
    # Surfaced both as a nested record and as top-level mirrors.
    assert provenance["freshness"]["stale"] is True
    assert provenance["data_age_days"] == 2
    assert provenance["stale"] is True
    assert "stale" in caplog.text.lower()


def test_freshness_gate_fails_closed_for_decisive_runs():
    config = _config(max_staleness_days=1)
    provenance = {"data_date": "2024-01-05", "end_date": "2024-01-09"}
    with pytest.raises(cli.DataStalenessError, match="refusing to run"):
        cli._enforce_freshness(
            provenance, config, context="decisive execute", fail_closed=True,
        )


def test_historical_backtest_is_not_stale_because_of_wall_clock():
    """A 2021 sample requested with a 2021 end date is age 0, not years old."""
    freshness = cli._freshness_from_values(
        last_complete_date="2021-03-31",
        data_date="2021-03-31",
        requested_end="2021-03-31",
        max_staleness_days=3,
    )
    assert freshness["data_age_days"] == 0
    assert freshness["stale"] is False


def test_incomplete_marker_is_reported_in_freshness():
    freshness = cli._freshness_from_values(
        last_complete_date="2024-01-08",
        data_date="2024-01-08",
        requested_end="2024-01-09",
        max_staleness_days=3,
        incomplete=True,
    )
    assert freshness["incomplete"] is True
    assert freshness["data_age_days"] == 1


def test_incremental_summary_line_format():
    provenance = {
        "data_date": "2024-01-11",
        "incremental": {"new_sessions": 3},
    }
    assert cli._incremental_summary_line(
        provenance, {"stale": False},
    ) == "data: 3 new sessions appended, data_date=2024-01-11, stale=no"
    assert cli._incremental_summary_line(
        {"data_date": "2024-01-08", "incremental": {"new_sessions": 0}},
        {"stale": True},
    ) == "data: 0 new sessions appended, data_date=2024-01-08, stale=yes"


# ----------------------------------------------------------------------
# Provider capability fallback
# ----------------------------------------------------------------------


class _FakeProvider:
    supports_incremental = False
    cache_status: dict = {}

    def __init__(self, updates=None):
        self.calls = []
        self._updates = updates or {}

    def update(self, start_date, end_date):
        self.calls.append((start_date, end_date))
        return dict(self._updates)


def test_unsupported_provider_falls_back_to_full_fetch_with_warning(caplog):
    resolution = {
        "provider": _FakeProvider(),
        "requested_provider": "synthetic",
        "effective_provider": "synthetic",
        "validated": False,
    }
    with caplog.at_level(logging.WARNING, logger="quant_platform.main"):
        summary = cli._run_incremental_update(resolution, _config())

    assert summary["supported"] is False
    assert summary["full_refetch"] is True
    assert summary["new_sessions"] == 0
    assert resolution["provider"].calls == []
    assert "does not support incremental updates" in caplog.text


def test_supported_provider_update_summary_is_returned():
    provider = _FakeProvider(updates={
        "performed": True, "new_sessions": 3,
        "last_complete_date": "2024-01-11", "incomplete": False,
        "reason": "appended 3 session(s)",
    })
    provider.supports_incremental = True
    resolution = {
        "provider": provider,
        "requested_provider": "tushare",
        "effective_provider": "tushare",
        "validated": False,
    }
    summary = cli._run_incremental_update(resolution, _config())

    assert provider.calls == [("2024-01-01", "2024-01-09")]
    assert summary["performed"] is True
    assert summary["new_sessions"] == 3
    assert summary["supported"] is True


def test_validated_runs_never_append(caplog):
    provider = _FakeProvider()
    provider.supports_incremental = True
    resolution = {
        "provider": provider,
        "requested_provider": "tushare",
        "effective_provider": "validated(tushare+adata)",
        "validated": True,
    }
    with caplog.at_level(logging.WARNING, logger="quant_platform.main"):
        summary = cli._run_incremental_update(resolution, _config())

    assert summary["supported"] is False
    assert provider.calls == []
    assert "validated" in caplog.text


# ----------------------------------------------------------------------
# _load_data_stage plumbing
# ----------------------------------------------------------------------


def _resolution(config):
    provider = _FakeProvider(updates={
        "performed": True, "new_sessions": 3,
        "last_complete_date": "2024-01-09", "incomplete": False,
        "reason": "appended 3 session(s)",
    })
    provider.supports_incremental = True
    return {
        "provider": provider,
        "provider_name": "synthetic",
        "requested_provider": "synthetic",
        "effective_provider": "synthetic",
        "identity": {
            "provider": "synthetic",
            "adjustment": "synthetic_adj_factor",
            "synthetic": True,
        },
        "universe": ["000001.SZ", "000002.SZ"],
        "allow_fallback": False,
        "failures": [],
        "validated": False,
    }


def _install_fake_load(monkeypatch, data, config):
    calls: list[int] = []

    def fake_load(config_, use_tushare=False, use_baostock=False, resolution=None):
        calls.append(1)
        cli._LAST_DATA_PROVENANCE = cli._build_data_manifest(
            resolution, config_, data_date="2024-01-09",
            payload_assets=list(data[0].columns),
            freshness={
                "last_complete_date": "2024-01-09",
                "data_date": "2024-01-09",
                "requested_end": str(config_.data.end_date),
                "data_age_days": 0,
                "incomplete": False,
            },
            incremental=(resolution or {}).get("incremental"),
        )
        return data

    monkeypatch.setattr(cli, "_load_data", fake_load)
    return calls


def _data_tuple():
    index = pd.DatetimeIndex(["2024-01-09"])
    prices = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=index)
    return prices, prices.pct_change(), None, None, None, None


def test_load_data_stage_incremental_calls_update_and_records_summary(
    tmp_path, monkeypatch,
):
    config = _config()
    resolution = _resolution(config)
    cache = PipelineCache(tmp_path)
    key = cli._data_cache_key(config, resolution)
    calls = _install_fake_load(monkeypatch, _data_tuple(), config)

    _tuple, provenance, hit = cli._load_data_stage(
        cache, key, config, resolution,
        read_cache=False, write_cache=False, incremental=True,
    )

    assert hit is False
    assert calls == [1]
    provider = resolution["provider"]
    assert provider.calls == [("2024-01-01", "2024-01-09")]
    assert provenance["incremental"]["new_sessions"] == 3
    assert provenance["freshness"]["last_complete_date"] == "2024-01-09"
    assert provenance["data_date"] == "2024-01-09"


def test_incremental_cache_entry_carries_freshness_in_manifest(
    tmp_path, monkeypatch,
):
    config = _config()
    resolution = _resolution(config)
    cache = PipelineCache(tmp_path)
    key = cli._data_cache_key(config, resolution)
    _install_fake_load(monkeypatch, _data_tuple(), config)

    cli._load_data_stage(
        cache, key, config, resolution,
        read_cache=False, write_cache=True, incremental=True,
    )

    payload, manifest = cache.load_stage_with_manifest(cli.DATA_STAGE, key)
    assert payload is not None
    assert manifest["last_complete_date"] == "2024-01-09"
    assert manifest["incremental"]["new_sessions"] == 3
    assert manifest["freshness"]["stale"] is False


# ----------------------------------------------------------------------
# CLI argument surface
# ----------------------------------------------------------------------


def test_run_accepts_incremental_flag(monkeypatch):
    captured = {}

    def fake_cmd_run(args):
        captured["incremental"] = args.incremental
        return 0

    monkeypatch.setattr(cli, "cmd_run", fake_cmd_run)
    monkeypatch.setattr(sys, "argv", ["main.py", "run", "--incremental"])
    assert cli.main() == 0
    assert captured["incremental"] is True


def test_run_incremental_flag_defaults_off(monkeypatch):
    captured = {}

    def fake_cmd_run(args):
        captured["incremental"] = args.incremental
        return 0

    monkeypatch.setattr(cli, "cmd_run", fake_cmd_run)
    monkeypatch.setattr(sys, "argv", ["main.py", "run"])
    assert cli.main() == 0
    assert captured["incremental"] is False


def test_cache_refresh_accepts_incremental_flag(monkeypatch):
    captured = {}

    def fake_refresh(args):
        captured["incremental"] = args.incremental
        return 0

    monkeypatch.setattr(cli, "_cmd_cache_refresh", fake_refresh)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "cache", "refresh", "--incremental"],
    )
    assert cli.main() == 0
    assert captured["incremental"] is True


def test_force_overrides_incremental(tmp_path, monkeypatch):
    """--force means full refetch; cmd_run must not run the append path."""
    import yaml

    raw = {
        "data": {
            "provider": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-01-09",
        },
        "universe": {"n_stocks": 2},
    }
    custom = tmp_path / "config.yaml"
    custom.write_text(yaml.safe_dump(raw), encoding="utf-8")

    stages = {}

    def fake_stage(cache, cache_key, config, resolution, **kwargs):
        stages.update(kwargs)
        return _data_tuple(), {"data_date": "2024-01-09"}, False

    def stop_factors(*_args, **_kwargs):
        raise RuntimeError("stop after the data stage")

    monkeypatch.setattr(cli, "_load_data_stage", fake_stage)
    monkeypatch.setattr(cli, "_compute_factors", stop_factors)
    args = SimpleNamespace(
        config=str(custom), use_baostock=False, use_tushare=False,
        force=True, no_cache=False, incremental=True,
        cache_dir=str(tmp_path / "cache_force_test"),
        allow_synthetic_fallback=False, description=None, save_factors=False,
    )
    with pytest.raises(RuntimeError, match="stop after the data stage"):
        cli.cmd_run(args)
    assert stages["incremental"] is False
    assert stages["read_cache"] is False


# ----------------------------------------------------------------------
# metrics.json provenance
# ----------------------------------------------------------------------


def test_metrics_json_carries_freshness_and_incremental(tmp_path):
    provenance = {
        "requested_provider": "tushare",
        "effective_provider": "tushare",
        "synthetic": False,
        "start_date": "2024-01-01",
        "end_date": "2024-01-11",
        "data_date": "2024-01-11",
        "last_complete_date": "2024-01-11",
        "data_age_days": 0,
        "stale": False,
        "incomplete": False,
        "freshness": {
            "last_complete_date": "2024-01-11",
            "data_date": "2024-01-11",
            "requested_end": "2024-01-11",
            "data_age_days": 0,
            "max_staleness_days": 3,
            "stale": False,
            "incomplete": False,
            "source": "incremental",
        },
        "incremental": {
            "supported": True,
            "performed": True,
            "full_refetch": False,
            "new_sessions": 3,
            "last_complete_date": "2024-01-11",
            "incomplete": False,
            "reason": "appended 3 session(s)",
        },
    }
    html_path, metrics_path = cli._write_run_artifacts(
        _fake_results(), _fake_ic_results(), None,
        _fake_config(tmp_path), provenance=provenance,
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    stored = metrics["provenance"]
    assert stored["last_complete_date"] == "2024-01-11"
    assert stored["data_age_days"] == 0
    assert stored["stale"] is False
    assert stored["freshness"]["source"] == "incremental"
    assert stored["incremental"]["new_sessions"] == 3
    assert stored["incremental"]["performed"] is True
    assert html_path.exists()
