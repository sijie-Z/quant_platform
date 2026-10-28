"""Operational regression tests for ``tools/forward_update.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "forward_update.py"


@pytest.fixture()
def forward_update():
    spec = importlib.util.spec_from_file_location("forward_update_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _merged(latest: str) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(latest), "000001")], names=["date", "asset"],
    )
    return pd.DataFrame({"close": [10.0]}, index=index)


def test_track_propagates_subprocess_exit_code(forward_update, tmp_path, monkeypatch):
    monkeypatch.setattr(
        forward_update.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="paper failed",
        ),
    )
    strategy = {"name": "turnover", "targets": tmp_path / "targets",
                "ledger": tmp_path / "ledger"}
    assert forward_update.track(strategy) == 2


def test_month_end_requires_st_snapshot(forward_update, tmp_path, monkeypatch):
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: True)
    strategy = {"name": "turnover", "targets": tmp_path / "targets"}
    with pytest.raises(RuntimeError, match="ST snapshot"):
        forward_update.ensure_monthly_target(
            strategy, pd.Timestamp("2026-10-30"), None,
        )


def test_target_generation_failure_is_not_silent(forward_update, tmp_path, monkeypatch):
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: True)
    monkeypatch.setattr(
        forward_update.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="signal failed",
        ),
    )
    st_file = tmp_path / "st.txt"
    st_file.write_text("000001\n", encoding="utf-8")
    strategy = {"name": "turnover", "factors": "turnover20d",
                "targets": tmp_path / "targets"}
    with pytest.raises(RuntimeError, match="target generation failed"):
        forward_update.ensure_monthly_target(
            strategy, pd.Timestamp("2026-10-30"), st_file,
        )


def test_main_skips_daily_st_snapshot_and_returns_success(
    forward_update, monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["forward_update.py", "--skip-refetch"])
    monkeypatch.setattr(
        forward_update, "rebuild", lambda: _merged("2026-10-26"),
    )
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: False)

    def unexpected_st_call():
        raise AssertionError("ST snapshot is only needed for month-end targets")

    monkeypatch.setattr(forward_update, "current_st_codes", unexpected_st_call)
    monkeypatch.setattr(
        forward_update, "ensure_monthly_target",
        lambda strategy, latest, st_file: None,
    )
    monkeypatch.setattr(forward_update, "track", lambda _strategy: 0)
    assert forward_update.main() == 0


def test_main_returns_nonzero_when_any_ledger_fails(forward_update, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["forward_update.py", "--skip-refetch"])
    monkeypatch.setattr(
        forward_update, "rebuild", lambda: _merged("2026-10-26"),
    )
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: False)
    monkeypatch.setattr(
        forward_update, "ensure_monthly_target",
        lambda strategy, latest, st_file: None,
    )
    calls = []

    def fail_one(strategy):
        calls.append(strategy["name"])
        return 2 if strategy["name"] == "turnover" else 0

    monkeypatch.setattr(forward_update, "track", fail_one)
    assert forward_update.main() == 1
    assert len(calls) == 2


def _st_test_root(forward_update, tmp_path, monkeypatch):
    root = tmp_path / "root"
    stocks = root / "stocks"
    stocks.mkdir(parents=True)
    for code in ("600519", "000001"):
        (stocks / f"{code}.parquet").write_bytes(b"")
    monkeypatch.setattr(forward_update, "ROOT", root)
    monkeypatch.setattr(forward_update, "STOCKS", stocks)
    forward_update._ST_CACHE.clear()
    forward_update._SPOT_CACHE.clear()
    forward_update._SPOT_FAILURE.clear()
    return root


def _market_spot(rows, fillers=120):
    """A market-wide-shaped spot frame: explicit rows plus traded fillers."""
    codes = [code for code, _ in rows] + [
        f"sz{index:06d}" for index in range(500, 500 + fillers)
    ]
    volumes = [volume for _, volume in rows] + [1000.0] * fillers
    return pd.DataFrame({"代码": codes, "成交量": volumes})


def test_st_snapshot_requires_complete_universe_coverage(
    forward_update, tmp_path, monkeypatch,
):
    root = _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    frame = pd.DataFrame({"代码": ["sh600519"], "名称": ["正常"]})
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: frame),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) is None
    assert not list((root / ".scratch").glob("**/st_codes_*.txt"))


def test_st_snapshot_rejects_empty_frame(forward_update, tmp_path, monkeypatch):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    frame = pd.DataFrame({"代码": [], "名称": []})
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: frame),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) is None


def test_st_snapshot_rejects_historical_target(forward_update, tmp_path, monkeypatch):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    assert forward_update.current_st_codes(pd.Timestamp("2026-10-23")) is None


def test_st_snapshot_writes_only_after_full_coverage(
    forward_update, tmp_path, monkeypatch,
):
    root = _st_test_root(forward_update, tmp_path, monkeypatch)
    frame = pd.DataFrame({
        "代码": ["sh600519", "sz000001"],
        "名称": ["正常", "ST测试"],
    })
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: frame),
    )
    path = forward_update.current_st_codes(pd.Timestamp.today())
    assert path is not None
    assert path.read_text(encoding="utf-8") == "000001"
    assert path.parent == root / ".scratch" / "orchestrate"


def test_stale_cache_reason_flags_unpublished_current_close(forward_update):
    import datetime as dt
    reason = forward_update.stale_cache_reason(
        pd.Timestamp("2026-10-23"), dt.date(2026, 10, 26),
    )
    assert reason is not None
    assert "2026-10-23" in reason
    assert "2026-10-26" in reason


def test_stale_cache_reason_allows_current_or_holiday_cache(forward_update):
    import datetime as dt
    assert forward_update.stale_cache_reason(
        pd.Timestamp("2026-10-26"), dt.date(2026, 10, 26),
    ) is None
    assert forward_update.stale_cache_reason(
        pd.Timestamp("2026-09-30"), dt.date(2026, 10, 1),
    ) is None


def test_stale_cache_reason_requires_previous_session_on_weekend(forward_update):
    import datetime as dt
    saturday = dt.date(2026, 10, 31)
    reason = forward_update.stale_cache_reason(
        pd.Timestamp("2026-10-29"), saturday,
    )
    assert reason is not None
    assert "2026-10-30" in reason
    assert forward_update.stale_cache_reason(
        pd.Timestamp("2026-10-30"), saturday,
    ) is None


def test_failed_refresh_reason_checks_watched_assets(forward_update, tmp_path):
    targets = tmp_path / "targets"
    targets.mkdir()
    (targets / "target_2026-10-23.json").write_text(
        '{"as_of": "2026-10-23", "weights": {"600519": 0.5}}', encoding="utf-8",
    )
    strategy = {"name": "turnover", "targets": targets,
                "ledger": tmp_path / "ledger"}
    reason = forward_update.failed_refresh_reason(
        [["600519", "timeout"]], [strategy], pd.Timestamp("2026-10-26"),
    )
    assert reason is not None and "600519" in reason
    assert forward_update.failed_refresh_reason(
        [["000001", "timeout"]], [strategy], pd.Timestamp("2026-10-26"),
    ) is None


def test_failed_refresh_reason_blocks_any_failure_before_month_end(forward_update):
    reason = forward_update.failed_refresh_reason(
        [["000001", "timeout"]], [], pd.Timestamp("2026-10-30"),
    )
    assert reason is not None and "month-end" in reason


def test_failed_refresh_reason_blocks_widespread_stale_responses(forward_update):
    failed = [[f"{index:06d}", "stale_response"] for index in range(6)]
    reason = forward_update.failed_refresh_reason(
        failed, [], pd.Timestamp("2026-10-27"),
    )
    assert reason is not None and "mixed-date" in reason


def test_refetch_retries_stale_response_and_only_writes_fresh_data(
    forward_update, tmp_path, monkeypatch,
):
    stocks = tmp_path / "stocks"
    stocks.mkdir()
    path = stocks / "000001.parquet"
    today = __import__("datetime").date.today()
    old_date = pd.Timestamp(today) - pd.Timedelta(days=1)
    pd.DataFrame({"date": [old_date], "close": [10.0]}).to_parquet(path, index=False)
    monkeypatch.setattr(forward_update, "STOCKS", stocks)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    import quant_platform.utils.exchange_calendar as calendar
    monkeypatch.setattr(calendar, "is_trading_day", lambda _day: True)
    calls = {"count": 0}

    def fake_daily(**_kwargs):
        calls["count"] += 1
        date = old_date if calls["count"] < 3 else pd.Timestamp(today)
        return pd.DataFrame({
            "date": [date], "open": [11.0], "high": [11.0], "low": [11.0],
            "close": [11.0], "volume": [1000.0], "amount": [11000.0],
            "turnover": [0.1],
        })

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_daily=fake_daily),
    )
    failed = forward_update.refetch()
    assert failed == []
    assert calls["count"] == 3
    result = pd.read_parquet(path)
    assert result["close"].iloc[0] == 11.0


def test_refetch_keeps_previous_file_when_response_stays_stale(
    forward_update, tmp_path, monkeypatch,
):
    stocks = tmp_path / "stocks"
    stocks.mkdir()
    path = stocks / "000001.parquet"
    today = __import__("datetime").date.today()
    old_date = pd.Timestamp(today) - pd.Timedelta(days=1)
    pd.DataFrame({"date": [old_date], "close": [10.0]}).to_parquet(path, index=False)
    monkeypatch.setattr(forward_update, "STOCKS", stocks)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    import quant_platform.utils.exchange_calendar as calendar
    monkeypatch.setattr(calendar, "is_trading_day", lambda _day: True)

    def fake_daily(**_kwargs):
        return pd.DataFrame({
            "date": [old_date], "open": [11.0], "high": [11.0], "low": [11.0],
            "close": [11.0], "volume": [1000.0], "amount": [11000.0],
            "turnover": [0.1],
        })

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_daily=fake_daily),
    )
    failed = forward_update.refetch()
    assert failed and failed[0][0] == "000001"
    assert failed[0][1].startswith("stale_response:")
    result = pd.read_parquet(path)
    assert result["close"].iloc[0] == 10.0


def test_overdue_month_ends_includes_earlier_gaps_and_uses_prior_target(
    forward_update, tmp_path,
):
    targets = tmp_path / "targets"
    targets.mkdir()
    for stamp in ("2026-08-31", "2026-10-30"):
        (targets / f"target_{stamp}.json").write_text(
            f'{{"as_of": "{stamp}", "weights": {{"600519": 0.5}}}}',
            encoding="utf-8",
        )
    overdue = forward_update.overdue_month_ends(
        targets, pd.Timestamp("2026-11-02"),
    )
    assert [value.strftime("%Y-%m-%d") for value in overdue] == ["2026-09-30"]
    assert forward_update.target_before(
        targets, pd.Timestamp("2026-09-30"),
    ) == targets / "target_2026-08-31.json"


def test_atomic_parquet_write_replaces_only_after_success(forward_update, tmp_path):
    path = tmp_path / "panel.parquet"
    frame = pd.DataFrame({"close": [1.0, 2.0]})
    forward_update.atomic_write_parquet(frame, path, index=False)
    assert pd.read_parquet(path).equals(frame)
    assert not path.with_name(path.name + ".tmp").exists()


def test_atomic_parquet_failure_preserves_previous_file(
    forward_update, tmp_path, monkeypatch,
):
    path = tmp_path / "panel.parquet"
    path.write_bytes(b"previous-good-data")

    def broken_write(_self, target, **_kwargs):
        Path(target).write_bytes(b"partial-new-data")
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken_write)
    with pytest.raises(RuntimeError, match="disk full"):
        forward_update.atomic_write_parquet(
            pd.DataFrame({"close": [3.0]}), path, index=False,
        )
    assert path.read_bytes() == b"previous-good-data"
    assert path.with_name(path.name + ".tmp").read_bytes() == b"partial-new-data"


def _strategy_with_target(tmp_path, stamp: str):
    targets = tmp_path / "targets"
    targets.mkdir()
    previous = targets / f"target_{stamp}.json"
    previous.write_text(
        f'{{"as_of": "{stamp}", "weights": {{"600519": 0.5}}}}',
        encoding="utf-8",
    )
    return {
        "name": "turnover", "factors": "turnover20d", "targets": targets,
    }


def test_main_detects_overdue_october_without_dated_st_snapshot(
    forward_update, tmp_path, monkeypatch, capsys,
):
    root = _st_test_root(forward_update, tmp_path, monkeypatch)
    strategy = _strategy_with_target(tmp_path, "2026-10-23")
    monkeypatch.setattr(forward_update, "STRATEGIES", [strategy])
    monkeypatch.setattr(sys, "argv", ["forward_update.py", "--skip-refetch"])
    monkeypatch.setattr(
        forward_update, "rebuild", lambda: _merged("2026-11-02"),
    )
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: False)
    current = root / ".scratch" / "orchestrate" / "st_codes_20261102.txt"
    current.parent.mkdir(parents=True)
    current.write_text("600519\n", encoding="utf-8")
    tracked = []
    monkeypatch.setattr(
        forward_update, "track",
        lambda strategy: tracked.append(strategy["name"]) or 0,
    )
    assert forward_update.main() == 1
    # The ledger still rebuilds with the existing book when the overdue
    # target is blocked; skipping it would silently freeze the NAV record.
    assert tracked == ["turnover"]
    output = capsys.readouterr().out
    assert "detected overdue month-end target 2026-10-30" in output
    assert "dated PIT ST snapshot" in output
    assert "snapshot_date == target as_of" in output
    assert not (strategy["targets"] / "target_2026-10-30.json").exists()


def test_main_catches_up_overdue_october_with_exact_dated_snapshot(
    forward_update, tmp_path, monkeypatch,
):
    root = _st_test_root(forward_update, tmp_path, monkeypatch)
    strategy = _strategy_with_target(tmp_path, "2026-10-23")
    snapshot_dir = root / ".scratch" / "orchestrate"
    snapshot_dir.mkdir(parents=True)
    dated = snapshot_dir / "st_codes_20261030.txt"
    dated.write_text("000001\n", encoding="utf-8")
    (snapshot_dir / "st_codes_20261102.txt").write_text(
        "600519\n", encoding="utf-8",
    )
    monkeypatch.setattr(forward_update, "STRATEGIES", [strategy])
    monkeypatch.setattr(sys, "argv", ["forward_update.py", "--skip-refetch"])
    monkeypatch.setattr(
        forward_update, "rebuild", lambda: _merged("2026-11-02"),
    )
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: False)
    commands = []

    def signal_success(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(forward_update.subprocess, "run", signal_success)
    tracked = []
    monkeypatch.setattr(
        forward_update, "track",
        lambda strategy: tracked.append(strategy["name"]) or 0,
    )
    assert forward_update.main() == 0
    assert tracked == ["turnover"]
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--as-of") + 1] == "2026-10-30"
    assert Path(command[command.index("--exclude-codes-file") + 1]) == dated
    assert Path(command[command.index("--prev-weights") + 1]) == (
        strategy["targets"] / "target_2026-10-23.json"
    )


def test_current_month_end_generates_with_supplied_snapshot(
    forward_update, tmp_path, monkeypatch,
):
    strategy = _strategy_with_target(tmp_path, "2026-10-23")
    snapshot = tmp_path / "current_st.txt"
    snapshot.write_text("000001\n", encoding="utf-8")
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: True)
    commands = []

    def signal_success(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(forward_update.subprocess, "run", signal_success)
    result = forward_update.ensure_monthly_target(
        strategy, pd.Timestamp("2026-10-30"), snapshot,
    )
    assert result == strategy["targets"] / "target_2026-10-30.json"
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--as-of") + 1] == "2026-10-30"
    assert Path(command[command.index("--exclude-codes-file") + 1]) == snapshot


def test_non_month_end_with_up_to_date_target_is_noop(
    forward_update, tmp_path, monkeypatch,
):
    strategy = _strategy_with_target(tmp_path, "2026-10-30")
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: False)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("an up-to-date non-month-end needs no ST or signal")

    monkeypatch.setattr(forward_update, "current_st_codes", unexpected_call)
    monkeypatch.setattr(forward_update.subprocess, "run", unexpected_call)
    assert forward_update.ensure_monthly_target(
        strategy, pd.Timestamp("2026-11-02"), None,
    ) is None


def test_st_snapshot_retries_transient_failures(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    frame = pd.DataFrame({
        "代码": ["sh600519", "sz000001"],
        "名称": ["正常", "ST测试"],
    })
    calls = {"count": 0}

    def flaky_spot():
        calls["count"] += 1
        if calls["count"] < 2:
            raise IndexError("list index out of range")
        return frame

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=flaky_spot),
    )
    path = forward_update.current_st_codes(pd.Timestamp.today())
    assert path is not None
    assert calls["count"] == 2
    assert path.read_text(encoding="utf-8") == "000001"


def test_st_snapshot_reuses_same_day_file_after_fetch_failure(
    forward_update, tmp_path, monkeypatch,
):
    import datetime as dt

    root = _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    stamp = dt.date.today().strftime("%Y%m%d")
    path = root / ".scratch" / "orchestrate" / f"st_codes_{stamp}.txt"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(f"{index:06d}" for index in range(1, 13)),
                    encoding="utf-8")

    def always_fail():
        raise IndexError("list index out of range")

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=always_fail),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) == path


def test_st_snapshot_fails_closed_without_same_day_file(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def always_fail():
        raise IndexError("list index out of range")

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=always_fail),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) is None


def _stale_daily_refetch(forward_update, tmp_path, monkeypatch):
    """One stock whose daily response stays at yesterday's bar."""
    stocks = tmp_path / "stocks"
    stocks.mkdir()
    path = stocks / "000001.parquet"
    today = __import__("datetime").date.today()
    old_date = pd.Timestamp(today) - pd.Timedelta(days=1)
    pd.DataFrame({"date": [old_date], "close": [10.0]}).to_parquet(path, index=False)
    monkeypatch.setattr(forward_update, "STOCKS", stocks)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    import quant_platform.utils.exchange_calendar as calendar
    monkeypatch.setattr(calendar, "is_trading_day", lambda _day: True)

    def fake_daily(**_kwargs):
        return pd.DataFrame({
            "date": [old_date], "open": [11.0], "high": [11.0], "low": [11.0],
            "close": [11.0], "volume": [1000.0], "amount": [11000.0],
            "turnover": [0.1],
        })

    return path, old_date, fake_daily


def _install_akshare(monkeypatch, fake_daily, spot):
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_daily=fake_daily, stock_zh_a_spot=spot),
    )


def test_refetch_reclassifies_stale_response_confirmed_suspended(
    forward_update, tmp_path, monkeypatch,
):
    path, old_date, fake_daily = _stale_daily_refetch(
        forward_update, tmp_path, monkeypatch,
    )
    spot = _market_spot([("sz000001", 0.0)])
    _install_akshare(monkeypatch, fake_daily, lambda: spot)
    assert forward_update.refetch() == []
    result = pd.read_parquet(path)
    assert result["date"].iloc[0] == old_date
    assert result["close"].iloc[0] == 10.0


def test_refetch_keeps_stale_response_when_spot_shows_trading(
    forward_update, tmp_path, monkeypatch,
):
    _path, _old_date, fake_daily = _stale_daily_refetch(
        forward_update, tmp_path, monkeypatch,
    )
    spot = _market_spot([("sz000001", 500.0)])
    _install_akshare(monkeypatch, fake_daily, lambda: spot)
    failed = forward_update.refetch()
    assert failed and failed[0][0] == "000001"
    assert failed[0][1].startswith("stale_response:")


def test_refetch_keeps_stale_response_when_spot_unavailable(
    forward_update, tmp_path, monkeypatch,
):
    _path, _old_date, fake_daily = _stale_daily_refetch(
        forward_update, tmp_path, monkeypatch,
    )

    def broken_spot():
        raise ConnectionError("spot endpoint down")

    _install_akshare(monkeypatch, fake_daily, broken_spot)
    failed = forward_update.refetch()
    assert failed and failed[0][0] == "000001"
    assert failed[0][1].startswith("stale_response:")


def test_refetch_reclassifies_error_response_confirmed_suspended(
    forward_update, tmp_path, monkeypatch,
):
    _path, _old_date, _fake = _stale_daily_refetch(
        forward_update, tmp_path, monkeypatch,
    )

    def empty_daily(**_kwargs):
        return pd.DataFrame()

    spot = _market_spot([("sz000001", 0.0)])
    _install_akshare(monkeypatch, empty_daily, lambda: spot)
    assert forward_update.refetch() == []


def _wide_spot(codes, volumes):
    return pd.DataFrame({"代码": codes, "成交量": volumes})


def test_confirm_rejects_zeroed_market_wide_feed(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    failed = [["000001", "stale_response:2026-10-29"],
              ["600519", "ValueError: empty"]]
    codes = ["sz000001", "sh600519"] + [f"sz{index:06d}" for index in range(1, 300)]
    spot = _wide_spot(codes, [0.0] * len(codes))
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: spot),
    )
    remaining, suspended = forward_update.confirm_suspended_stale(failed)
    assert suspended == []
    assert remaining == failed


def test_confirm_requires_duplicate_rows_to_all_be_untraded(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    failed = [["000001", "stale_response:2026-10-29"]]
    spot = _market_spot([("sz000001", 0.0), ("sz000001", 500.0)])
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: spot),
    )
    remaining, suspended = forward_update.confirm_suspended_stale(failed)
    assert suspended == []
    assert remaining == failed


def test_confirm_treats_nan_volume_row_as_doubt(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    failed = [["000001", "stale_response:2026-10-29"]]
    spot = _market_spot([("sz000001", float("nan")), ("sz000001", 0.0)])
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: spot),
    )
    remaining, suspended = forward_update.confirm_suspended_stale(failed)
    assert suspended == []
    assert remaining == failed


def test_confirm_rejects_partial_realtime_frame(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    failed = [["000001", "stale_response:2026-10-29"]]
    spot = _wide_spot(["sz000001"], [0.0])
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: spot),
    )
    remaining, suspended = forward_update.confirm_suspended_stale(failed)
    assert suspended == []
    assert remaining == failed


def test_confirm_caps_mass_reclassification(forward_update, tmp_path, monkeypatch):
    root = tmp_path / "root"
    stocks = root / "stocks"
    stocks.mkdir(parents=True)
    codes = [f"{index:06d}" for index in range(1, 16)]
    for code in codes:
        (stocks / f"{code}.parquet").write_bytes(b"")
    monkeypatch.setattr(forward_update, "ROOT", root)
    monkeypatch.setattr(forward_update, "STOCKS", stocks)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    failed = [[code, "stale_response:2026-10-29"] for code in codes]
    spot = _market_spot([(f"sz{code}", 0.0) for code in codes])
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: spot),
    )
    remaining, suspended = forward_update.confirm_suspended_stale(failed)
    assert suspended == []
    assert remaining == failed


def test_st_snapshot_rejects_implausible_zero_match(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    frame = pd.DataFrame({
        "代码": ["sh600519", "sz000001"],
        "名称": ["正常", "另一只"],
    })
    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=lambda: frame),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) is None


def test_st_snapshot_reuse_rejects_malformed_same_day_file(
    forward_update, tmp_path, monkeypatch,
):
    import datetime as dt

    root = _st_test_root(forward_update, tmp_path, monkeypatch)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    stamp = dt.date.today().strftime("%Y%m%d")
    path = root / ".scratch" / "orchestrate" / f"st_codes_{stamp}.txt"
    path.parent.mkdir(parents=True)
    path.write_text("not-a-code", encoding="utf-8")

    def always_fail():
        raise IndexError("list index out of range")

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=always_fail),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) is None


def test_ensure_monthly_target_regenerates_corrupt_existing_target(
    forward_update, tmp_path, monkeypatch,
):
    strategy = _strategy_with_target(tmp_path, "2026-10-23")
    corrupt = strategy["targets"] / "target_2026-10-30.json"
    corrupt.write_text('{"weights": {}', encoding="utf-8")
    snapshot = tmp_path / "current_st.txt"
    snapshot.write_text("000001\n", encoding="utf-8")
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: True)
    commands = []

    def signal_success(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(forward_update.subprocess, "run", signal_success)
    forward_update.ensure_monthly_target(
        strategy, pd.Timestamp("2026-10-30"), snapshot,
    )
    assert len(commands) == 1
    assert commands[0][commands[0].index("--as-of") + 1] == "2026-10-30"


def test_overdue_month_ends_ignores_corrupt_known_target(
    forward_update, tmp_path,
):
    targets = tmp_path / "targets"
    targets.mkdir()
    for stamp in ("2026-08-31", "2026-10-30"):
        (targets / f"target_{stamp}.json").write_text(
            f'{{"as_of": "{stamp}", "weights": {{"600519": 0.5}}}}',
            encoding="utf-8",
        )
    (targets / "target_2026-09-30.json").write_text(
        '{"as_of": "2026-09-30", "weights": {"600', encoding="utf-8",
    )
    overdue = forward_update.overdue_month_ends(
        targets, pd.Timestamp("2026-11-02"),
    )
    assert [value.strftime("%Y-%m-%d") for value in overdue] == ["2026-09-30"]


def test_failed_refresh_reason_strict_override_blocks_single_failure(
    forward_update,
):
    reason = forward_update.failed_refresh_reason(
        [["000001", "stale_response"]], [], pd.Timestamp("2026-11-02"),
        strict_month_end=True,
    )
    assert reason is not None and "month-end" in reason


def test_main_captures_dated_st_before_gate_failure(
    forward_update, monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["forward_update.py"])
    monkeypatch.setattr(forward_update, "_month_end_session_today", lambda: True)
    monkeypatch.setattr(
        forward_update, "rebuild", lambda: _merged("2026-10-30"),
    )
    monkeypatch.setattr(forward_update, "stale_cache_reason",
                        lambda latest, today: None)
    monkeypatch.setattr(
        forward_update, "failed_refresh_reason",
        lambda failed, strategies, latest, strict=None: "blocked for test",
    )
    monkeypatch.setattr(forward_update, "refetch", lambda: [])
    calls = []

    def capture(as_of=None):
        calls.append(as_of)
        return None

    monkeypatch.setattr(forward_update, "current_st_codes", capture)
    assert forward_update.main() == 2
    assert len(calls) == 1


def test_main_survives_early_st_capture_crash(forward_update, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["forward_update.py"])
    monkeypatch.setattr(forward_update, "_month_end_session_today", lambda: True)

    def locked(_as_of=None):
        raise OSError("file locked by antivirus")

    monkeypatch.setattr(forward_update, "current_st_codes", locked)
    monkeypatch.setattr(forward_update, "refetch", lambda: [])
    monkeypatch.setattr(
        forward_update, "rebuild", lambda: _merged("2026-10-30"),
    )
    monkeypatch.setattr(forward_update, "stale_cache_reason",
                        lambda latest, today: None)
    monkeypatch.setattr(
        forward_update, "failed_refresh_reason", lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(forward_update, "is_month_end", lambda _latest: False)
    monkeypatch.setattr(
        forward_update, "ensure_monthly_target",
        lambda strategy, latest, st_file=None: None,
    )
    monkeypatch.setattr(forward_update, "track", lambda _strategy: 0)
    assert forward_update.main() == 0


def test_spot_snapshot_deadline_and_failure_memo(forward_update, monkeypatch):
    import time as time_module

    original_sleep = time_module.sleep
    monkeypatch.setattr(forward_update, "SPOT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)
    forward_update._SPOT_CACHE.clear()
    forward_update._SPOT_FAILURE.clear()
    calls = {"count": 0}

    def hung():
        calls["count"] += 1
        original_sleep(0.2)
        return pd.DataFrame({"代码": ["sz000001"], "成交量": [0.0]})

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=hung),
    )
    with pytest.raises(RuntimeError, match="unavailable after 2 attempts"):
        forward_update.today_spot_snapshot()
    assert calls["count"] == 2
    with pytest.raises(RuntimeError, match="recent failure cached"):
        forward_update.today_spot_snapshot()
    assert calls["count"] == 2


def test_st_snapshot_does_not_multiply_fetch_attempts(
    forward_update, tmp_path, monkeypatch,
):
    _st_test_root(forward_update, tmp_path, monkeypatch)
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)
    calls = {"count": 0}

    def down():
        calls["count"] += 1
        raise ConnectionError("feed down")

    monkeypatch.setitem(
        sys.modules, "akshare",
        SimpleNamespace(stock_zh_a_spot=down),
    )
    assert forward_update.current_st_codes(pd.Timestamp.today()) is None
    assert calls["count"] == 2
