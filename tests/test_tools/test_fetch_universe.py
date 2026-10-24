"""Tests for ``tools/fetch_universe.py`` (R27).

The network is mocked everywhere: fake AKShare/Baostock objects are injected
through the tool's lazy import hooks. Covers the taskbook acceptance points:
universe sampling, column mapping, retry/backoff, all-failed exit code 2,
breakpoint resume plus ``--refresh``, aggregation/deduplication, determinism,
and the optional Baostock source.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "fetch_universe.py"


@pytest.fixture(scope="module")
def tool():
    """Load tools/fetch_universe.py as a module without packaging it."""
    spec = importlib.util.spec_from_file_location("fetch_universe_r27", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install(tool, monkeypatch, *, ak=None, bs=None):
    """Patch the tool's lazy imports and sleep so no real I/O happens."""
    monkeypatch.setattr(tool, "_sleep", lambda seconds: None)
    if ak is not None:
        monkeypatch.setattr(tool, "_import_akshare", lambda: ak)
    if bs is not None:
        monkeypatch.setattr(tool, "_import_baostock", lambda: bs)


def run_cli(tool, out_dir, *, codes=None, universe_file=None, source="akshare", extra=(), retries=3):
    argv = [
        "--source",
        source,
        "--start",
        "2020-01-01",
        "--end",
        "2020-12-31",
        "--out-dir",
        str(out_dir),
        "--sleep",
        "0",
        "--retries",
        str(retries),
    ]
    if codes is not None:
        argv += ["--codes", codes]
    if universe_file is not None:
        argv += ["--universe-file", str(universe_file), "--max-stocks", "10"]
    argv += list(extra)
    return tool.main(argv)


def hist_frame(code: str = "600000", close: float = 10.0, dates=None) -> pd.DataFrame:
    """An AKShare-shaped response with Chinese columns in a shuffled order."""
    dates = pd.bdate_range("2020-01-02", periods=5) if dates is None else dates
    n = len(dates)
    close_series = np.linspace(close, close + 1.0, n)
    return pd.DataFrame(
        {
            "股票代码": code,
            "日期": dates,
            "收盘": close_series,
            "开盘": close_series - 0.1,
            "最高": close_series + 0.2,
            "最低": close_series - 0.2,
            "成交量": np.arange(n, dtype=float) + 1000.0,
            "成交额": np.arange(n, dtype=float) * 1000.0 + 50000.0,
            "振幅": np.full(n, 2.0),
            "涨跌幅": np.linspace(-1.0, 1.0, n),
            "涨跌额": np.linspace(-0.1, 0.1, n),
            "换手率": np.full(n, 1.5),
        }
    )


class FakeAK:
    """Minimal stand-in for the akshare module.

    A response may be a DataFrame, an Exception (raised on every call), or a
    list of those (consumed one per call) to script a retry sequence.
    """

    def __init__(self, responses=None, default=None):
        self.responses = dict(responses or {})
        self.default = default
        self.calls: list[dict[str, str]] = []

    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust):
        self.calls.append(
            {
                "symbol": symbol,
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "adjust": adjust,
            }
        )
        value = self.responses.get(symbol, self.default)
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise AssertionError(f"unexpected symbol: {symbol}")
        return value.copy()


class FakeQueryResult:
    def __init__(self, rows, fields):
        self.error_code = "0"
        self.error_msg = ""
        self.fields = fields
        self._rows = list(rows)
        self._position = 0

    def next(self):
        if self._position < len(self._rows):
            self._position += 1
            return True
        return False

    def get_row_data(self):
        return self._rows[self._position - 1]


class FakeBS:
    def __init__(self, rows=None, login_ok=True):
        self.rows = list(rows or [])
        self.login_ok = login_ok
        self.logins = 0
        self.logouts = 0
        self.queries: list[dict[str, str]] = []

    def login(self):
        self.logins += 1
        if self.login_ok:
            return SimpleNamespace(error_code="0", error_msg="success")
        return SimpleNamespace(error_code="10001001", error_msg="login failed")

    def logout(self):
        self.logouts += 1
        return SimpleNamespace(error_code="0", error_msg="success")

    def query_history_k_data_plus(self, code, fields, start_date, end_date, frequency, adjustflag):
        self.queries.append(
            {
                "code": code,
                "fields": fields,
                "start_date": start_date,
                "end_date": end_date,
                "frequency": frequency,
                "adjustflag": adjustflag,
            }
        )
        return FakeQueryResult(self.rows, fields.split(","))


def write_metadata(path: Path, n: int = 110, delisted: tuple[int, ...] = ()) -> tuple[Path, list[str]]:
    codes = [f"{index:06d}" for index in range(n)]
    frame = pd.DataFrame(
        {
            "sector": "test",
            "market_cap_group": "mid",
            "listing_date": pd.Timestamp("2000-01-01"),
            "delisting_date": pd.NaT,
        },
        index=pd.Index(codes, name="asset"),
    )
    delisted_codes = [codes[index] for index in delisted]
    frame.loc[delisted_codes, "delisting_date"] = pd.Timestamp("2020-01-01")
    frame.to_parquet(path)
    return path, delisted_codes


def write_seed_stock(tool, stocks_dir: Path, code: str, close: float = 10.0) -> None:
    frame = tool._normalize_frame(hist_frame(code, close=close), tool.AKSHARE_COLUMN_MAP, code)
    tool.write_stock(frame, code, stocks_dir)


# ---------------------------------------------------------------------------
# a. universe sampling
# ---------------------------------------------------------------------------


def test_universe_sampling_evenly_spaced_and_listed_only(tool, tmp_path, monkeypatch):
    delisted_positions = tuple(range(5, 105, 10))  # 10 delisted -> 100 listed
    meta_path, delisted_codes = write_metadata(tmp_path / "metadata.parquet", n=110, delisted=delisted_positions)

    listed = tool.load_listed_codes(meta_path)
    assert len(listed) == 100
    assert not set(delisted_codes) & set(listed)
    assert listed == sorted(listed)

    picked = tool.sample_evenly(listed, 10)
    assert picked == listed[::11]
    assert len(picked) == 10

    fake = FakeAK(default=hist_frame())
    install(tool, monkeypatch, ak=fake)
    out = tmp_path / "out"
    rc = run_cli(tool, out, universe_file=meta_path)
    assert rc == 0
    assert [call["symbol"] for call in fake.calls] == picked
    assert [path.stem for path in sorted((out / "stocks").glob("*.parquet"))] == picked

    # --codes overrides the metadata filter, even for a delisted name.
    fake2 = FakeAK(default=hist_frame())
    install(tool, monkeypatch, ak=fake2)
    rc = run_cli(tool, tmp_path / "out_codes", codes=delisted_codes[0])
    assert rc == 0
    assert [call["symbol"] for call in fake2.calls] == [delisted_codes[0]]


def test_sample_evenly_edges(tool):
    codes = [f"{index:06d}" for index in range(7)]
    assert tool.sample_evenly(codes, 100) == codes
    assert tool.sample_evenly(codes, 1) == [codes[0]]
    assert tool.sample_evenly(codes, 3) == [codes[0], codes[3], codes[6]]
    assert tool.normalize_code("600000.SH") == "600000"
    assert tool.normalize_code("sh.600000") == "600000"
    assert tool.normalize_code(1) == "000001"


# ---------------------------------------------------------------------------
# b. successful fetch and column mapping
# ---------------------------------------------------------------------------


def test_fetch_success_maps_columns_and_writes_parquet(tool, tmp_path, monkeypatch):
    raw = hist_frame("600519", close=100.0)
    fake = FakeAK(responses={"600519": raw})
    install(tool, monkeypatch, ak=fake)
    out = tmp_path / "out"

    rc = run_cli(tool, out, codes="600519", retries=0)
    assert rc == 0
    assert fake.calls == [
        {
            "symbol": "600519",
            "period": "daily",
            "start_date": "20200101",
            "end_date": "20201231",
            "adjust": "hfq",
        }
    ]

    stock_path = out / "stocks" / "600519.parquet"
    assert stock_path.exists()
    saved = pd.read_parquet(stock_path)
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["日期"]),
            "asset": "600519",
            "open": raw["开盘"].astype(float),
            "high": raw["最高"].astype(float),
            "low": raw["最低"].astype(float),
            "close": raw["收盘"].astype(float),
            "volume": raw["成交量"].astype(float),
            "amount": raw["成交额"].astype(float),
            "pct_chg": raw["涨跌幅"].astype(float),
            "turnover": raw["换手率"].astype(float),
        }
    )
    assert list(saved.columns) == tool.OUTPUT_COLUMNS
    assert (saved["asset"] == "600519").all()
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)

    failures = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert failures["count"] == 0 and failures["failures"] == []

    wide = pd.read_parquet(out / "daily_wide.parquet")
    assert wide.index.names == ["date", "asset"]
    assert list(wide.columns) == tool.VALUE_COLUMNS
    assert len(wide) == len(raw)


def test_empty_and_missing_columns_are_classified_failures(tool):
    empty = pd.DataFrame(columns=list(tool.AKSHARE_COLUMN_MAP))
    with pytest.raises(tool.FetchError) as info:
        tool._normalize_frame(empty, tool.AKSHARE_COLUMN_MAP, "600000")
    assert info.value.kind == "EmptyData"

    missing = hist_frame().drop(columns=["收盘"])
    with pytest.raises(tool.FetchError) as info:
        tool._normalize_frame(missing, tool.AKSHARE_COLUMN_MAP, "600000")
    assert info.value.kind == "MissingColumns"


# ---------------------------------------------------------------------------
# c. retry with exponential backoff
# ---------------------------------------------------------------------------


def test_retry_twice_then_success(tool, tmp_path, monkeypatch):
    fake = FakeAK(
        responses={
            "600000": [
                ConnectionError("proxy reset"),
                ConnectionError("proxy reset"),
                hist_frame("600000"),
            ]
        }
    )
    install(tool, monkeypatch, ak=fake)
    out = tmp_path / "out"

    rc = run_cli(tool, out, codes="600000", retries=3)
    assert rc == 0
    assert len(fake.calls) == 3
    assert (out / "stocks" / "600000.parquet").exists()
    failures = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert failures["failures"] == []


def test_backoff_is_exponential_and_bounded(tool):
    waits: list[float] = []
    calls = {"count": 0}

    def always_fail():
        calls["count"] += 1
        raise TimeoutError("slow")

    frame, attempts, error = tool.fetch_with_retries(always_fail, retries=2, sleep_seconds=0.5, sleep_fn=waits.append)
    assert frame is None
    assert attempts == 3
    assert error[0] == "TimeoutError"
    assert waits == [0.5, 1.0]
    assert calls["count"] == 3


# ---------------------------------------------------------------------------
# d. all failures -> exit code 2
# ---------------------------------------------------------------------------


def test_all_failures_exit_code_2_and_failures_json(tool, tmp_path, monkeypatch):
    fake = FakeAK(
        responses={
            "600000": ConnectionError("network down"),
            "000001": TimeoutError("connect timeout"),
        }
    )
    install(tool, monkeypatch, ak=fake)
    out = tmp_path / "out"

    rc = run_cli(tool, out, codes="600000,000001", retries=2)
    assert rc == 2
    assert len(fake.calls) == 6  # 2 codes x (2 retries + 1 attempt)
    assert not list((out / "stocks").glob("*.parquet"))
    assert not (out / "daily_wide.parquet").exists()

    payload = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert payload["count"] == 2
    by_code = {entry["code"]: entry for entry in payload["failures"]}
    assert by_code["600000"]["error_type"] == "ConnectionError"
    assert "network down" in by_code["600000"]["reason"]
    assert by_code["000001"]["attempts"] == 3
    assert [entry["code"] for entry in payload["failures"]] == ["600000", "000001"]


# ---------------------------------------------------------------------------
# e. breakpoint resume and --refresh
# ---------------------------------------------------------------------------


def test_resume_skips_existing_and_refresh_forces_refetch(tool, tmp_path, monkeypatch):
    out = tmp_path / "out"
    stocks_dir = out / "stocks"
    for code in ("600001", "600002", "600003"):
        write_seed_stock(tool, stocks_dir, code)

    fake = FakeAK(responses={"600004": hist_frame("600004"), "600005": hist_frame("600005")})
    install(tool, monkeypatch, ak=fake)
    rc = run_cli(tool, out, codes="600001,600002,600003,600004,600005")
    assert rc == 0
    assert [call["symbol"] for call in fake.calls] == ["600004", "600005"]

    wide = pd.read_parquet(out / "daily_wide.parquet")
    assert sorted(wide.index.get_level_values("asset").unique()) == ["600001", "600002", "600003", "600004", "600005"]
    assert len(wide) == 25  # 5 stocks x 5 sessions

    # --refresh refetches one existing code and replaces its file.
    fake2 = FakeAK(responses={"600002": hist_frame("600002", close=20.0)})
    install(tool, monkeypatch, ak=fake2)
    rc = run_cli(tool, out, codes="600001,600002,600003,600004,600005", extra=["--refresh", "600002"])
    assert rc == 0
    assert [call["symbol"] for call in fake2.calls] == ["600002"]
    refreshed = pd.read_parquet(out / "stocks" / "600002.parquet")
    assert pytest.approx(refreshed["close"].iloc[0]) == 20.0
    # A refresh code outside the selection still works when its file exists.
    fake3 = FakeAK(responses={"600001": hist_frame("600001", close=30.0)})
    install(tool, monkeypatch, ak=fake3)
    rc = run_cli(tool, out, codes="600004,600005", extra=["--refresh", "600001"])
    assert rc == 0
    assert [call["symbol"] for call in fake3.calls] == ["600001"]
    assert pytest.approx(pd.read_parquet(out / "stocks" / "600001.parquet")["close"].iloc[0]) == 30.0

    # A refresh code with no file and outside the selection is a warning no-op.
    fake4 = FakeAK(default=hist_frame())
    install(tool, monkeypatch, ak=fake4)
    rc = run_cli(tool, out, codes="600004,600005", extra=["--refresh", "999999"])
    assert rc == 0
    assert fake4.calls == []


# ---------------------------------------------------------------------------
# f. aggregation and deduplication
# ---------------------------------------------------------------------------


def test_aggregate_merges_and_deduplicates(tool, tmp_path):
    out = tmp_path / "out"
    stocks_dir = out / "stocks"
    stocks_dir.mkdir(parents=True)

    first = tool._normalize_frame(hist_frame("600000"), tool.AKSHARE_COLUMN_MAP, "600000")
    duplicate = first.iloc[[0]].copy()
    duplicate["close"] = 99.0
    duplicate["open"] = 99.0
    pd.concat([first, duplicate], ignore_index=True).to_parquet(stocks_dir / "600000.parquet", index=False)

    second = tool._normalize_frame(hist_frame("000001"), tool.AKSHARE_COLUMN_MAP, "000001")
    second.to_parquet(stocks_dir / "000001.parquet", index=False)

    panel = tool.aggregate_stocks(stocks_dir, out / "daily_wide.parquet")
    assert panel is not None
    assert panel.index.names == ["date", "asset"]
    assert list(panel.columns) == tool.VALUE_COLUMNS
    assert len(panel) == 10
    assert not panel.index.duplicated().any()

    first_date = first["date"].iloc[0]
    assert (first_date, "600000") in panel.index
    assert pytest.approx(panel.loc[(first_date, "600000"), "close"]) == 99.0
    assert panel.index.is_monotonic_increasing


def test_aggregate_returns_none_without_files(tool, tmp_path):
    stocks_dir = tmp_path / "empty" / "stocks"
    stocks_dir.mkdir(parents=True)
    assert tool.aggregate_stocks(stocks_dir, tmp_path / "daily_wide.parquet") is None
    assert not (tmp_path / "daily_wide.parquet").exists()


# ---------------------------------------------------------------------------
# g. determinism
# ---------------------------------------------------------------------------


def test_two_runs_are_deterministic_except_timestamp(tool, tmp_path, monkeypatch):
    payloads = []
    panels = []
    for run_index, name in enumerate(("out1", "out2")):
        fake = FakeAK(responses={"600000": hist_frame("600000"), "000001": ConnectionError("permanent outage")})
        install(tool, monkeypatch, ak=fake)
        out = tmp_path / name
        rc = run_cli(tool, out, codes="600000,000001", retries=1)
        assert rc == 0
        assert len(fake.calls) == 3  # 1 success attempt + 1 failure code x 2 attempts
        payload = json.loads((out / "failures.json").read_text(encoding="utf-8"))
        payload.pop("generated_at")
        payloads.append(payload)
        panels.append(pd.read_parquet(out / "daily_wide.parquet"))
    assert payloads[0] == payloads[1]
    pd.testing.assert_frame_equal(panels[0], panels[1])


# ---------------------------------------------------------------------------
# robustness / regression cases raised in review
# ---------------------------------------------------------------------------


def test_read_error_is_reported_and_skipped(tool, tmp_path, monkeypatch):
    out = tmp_path / "out"
    stocks_dir = out / "stocks"
    write_seed_stock(tool, stocks_dir, "600002")
    (stocks_dir / "600001.parquet").write_bytes(b"definitely not parquet")

    fake = FakeAK(responses={"600000": ConnectionError("down")})
    install(tool, monkeypatch, ak=fake)
    rc = run_cli(tool, out, codes="600000", retries=0)
    assert rc == 0  # 600002 still provides data; the run does not crash on the bad file

    payload = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert {entry["code"] for entry in payload["failures"]} == {"600000", "600001"}
    read_error = next(entry for entry in payload["failures"] if entry["code"] == "600001")
    assert read_error["error_type"] == "ReadError"
    wide = pd.read_parquet(out / "daily_wide.parquet")
    assert sorted(wide.index.get_level_values("asset").unique()) == ["600002"]


def test_only_unreadable_files_exit_2(tool, tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "stocks").mkdir(parents=True)
    (out / "stocks" / "600001.parquet").write_bytes(b"garbage")
    fake = FakeAK(responses={"600000": ConnectionError("down")})
    install(tool, monkeypatch, ak=fake)
    rc = run_cli(tool, out, codes="600000", retries=0)
    assert rc == 2
    payload = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert {entry["error_type"] for entry in payload["failures"]} == {"ReadError", "ConnectionError"}


def test_codes_and_refresh_must_be_six_digits(tool, tmp_path, monkeypatch):
    fake = FakeAK(default=hist_frame())
    install(tool, monkeypatch, ak=fake)

    rc = run_cli(tool, tmp_path / "out_traversal", codes="../../../EVIL")
    assert rc == 2
    assert fake.calls == []
    assert not (tmp_path / "EVIL.parquet").exists()

    rc = run_cli(tool, tmp_path / "out_refresh", codes="600000", extra=["--refresh", "../x"])
    assert rc == 2
    assert fake.calls == []


def test_text_delisting_date_sentinels(tool, tmp_path):
    codes = ["000001", "000002", "000003", "000004", "000005", "000006"]
    frame = pd.DataFrame(
        {"delisting_date": [None, "", "null", "NaN", "-", "2020-01-01"]},
        index=pd.Index(codes, name="asset"),
    )
    frame["delisting_date"] = frame["delisting_date"].astype(object)
    path = tmp_path / "metadata_text.parquet"
    frame.to_parquet(path)
    assert tool.load_listed_codes(path) == codes[:5]


def test_backoff_waits_are_capped(tool):
    waits: list[float] = []

    def always_fail():
        raise RuntimeError("boom")

    _, attempts, error = tool.fetch_with_retries(always_fail, retries=2, sleep_seconds=1000.0, sleep_fn=waits.append)
    assert attempts == 3
    assert error[0] == "RuntimeError"
    assert waits == [60.0, 60.0]


def test_keyboard_interrupt_is_not_swallowed(tool):
    calls = {"count": 0}

    def interrupt():
        calls["count"] += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tool.fetch_with_retries(interrupt, retries=3, sleep_seconds=0.0, sleep_fn=lambda seconds: None)
    assert calls["count"] == 1


def test_refresh_failure_keeps_the_old_file(tool, tmp_path, monkeypatch):
    out = tmp_path / "out"
    stocks_dir = out / "stocks"
    write_seed_stock(tool, stocks_dir, "600001", close=10.0)

    fake = FakeAK(responses={"600001": ConnectionError("refresh outage")})
    install(tool, monkeypatch, ak=fake)
    rc = run_cli(tool, out, codes="600001", extra=["--refresh", "600001"], retries=1)
    assert rc == 0
    assert len(fake.calls) == 2
    saved = pd.read_parquet(stocks_dir / "600001.parquet")
    assert saved["close"].iloc[0] == pytest.approx(10.0)
    payload = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["failures"][0]["attempts"] == 2


def test_write_error_is_recorded_and_run_continues(tool, tmp_path, monkeypatch):
    fake = FakeAK(responses={"600001": hist_frame("600001"), "600002": hist_frame("600002")})
    install(tool, monkeypatch, ak=fake)
    original_write = tool.write_stock

    def flaky_write(frame, code, stocks_dir):
        if code == "600001":
            raise OSError("disk full")
        return original_write(frame, code, stocks_dir)

    monkeypatch.setattr(tool, "write_stock", flaky_write)
    out = tmp_path / "out"
    rc = run_cli(tool, out, codes="600001,600002", retries=0)
    assert rc == 0
    payload = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert payload["failures"][0]["code"] == "600001"
    assert payload["failures"][0]["error_type"] == "WriteError"
    assert (out / "stocks" / "600002.parquet").exists()


def test_empty_universe_warns_and_exits_2(tool, tmp_path, monkeypatch, capsys):
    meta_path, _ = write_metadata(tmp_path / "metadata_empty.parquet", n=3, delisted=(0, 1, 2))
    fake = FakeAK(default=hist_frame())
    install(tool, monkeypatch, ak=fake)
    rc = run_cli(tool, tmp_path / "out", universe_file=meta_path)
    assert rc == 2
    assert fake.calls == []
    assert "0 codes" in capsys.readouterr().err


def test_zero_success_warns_about_stale_panel(tool, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    out.mkdir(parents=True)
    pd.DataFrame({"stale": [1]}).to_parquet(out / "daily_wide.parquet")
    fake = FakeAK(responses={"600000": ConnectionError("down")})
    install(tool, monkeypatch, ak=fake)
    rc = run_cli(tool, out, codes="600000", retries=0)
    assert rc == 2
    assert "stale" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# optional Baostock source
# ---------------------------------------------------------------------------


def test_baostock_source_maps_fields_and_logs_out(tool, tmp_path, monkeypatch):
    rows = [
        ["2020-01-02", "10.0", "10.5", "9.8", "10.2", "1000", "50000", "1.5", "2.0"],
        ["2020-01-03", "10.2", "10.8", "10.0", "10.7", "1100", "60000", "-0.5", "2.1"],
    ]
    bs = FakeBS(rows=rows)
    install(tool, monkeypatch, bs=bs)
    out = tmp_path / "out"

    rc = run_cli(tool, out, codes="600000", source="baostock")
    assert rc == 0
    assert bs.logins == 1 and bs.logouts == 1
    query = bs.queries[0]
    assert query["code"] == "sh.600000"
    assert query["adjustflag"] == "1"
    assert query["start_date"] == "2020-01-01" and query["end_date"] == "2020-12-31"

    saved = pd.read_parquet(out / "stocks" / "600000.parquet")
    assert list(saved.columns) == tool.OUTPUT_COLUMNS
    assert saved["close"].tolist() == [10.2, 10.7]
    assert saved["turnover"].tolist() == [2.0, 2.1]
    assert pd.api.types.is_datetime64_any_dtype(saved["date"])


def test_baostock_login_failure_exits_1_after_two_attempts(tool, tmp_path, monkeypatch):
    bs = FakeBS(rows=[], login_ok=False)
    install(tool, monkeypatch, bs=bs)
    out = tmp_path / "out"

    rc = run_cli(tool, out, codes="600000", source="baostock")
    assert rc == 1
    assert bs.logins == 2
    assert bs.queries == []
    assert not (out / "daily_wide.parquet").exists()


def test_baostock_fully_resumed_pool_skips_login(tool, tmp_path, monkeypatch):
    out = tmp_path / "out"
    write_seed_stock(tool, out / "stocks", "600000")
    bs = FakeBS(rows=[], login_ok=False)  # would fail if a login were attempted
    install(tool, monkeypatch, bs=bs)
    rc = run_cli(tool, out, codes="600000", source="baostock")
    assert rc == 0
    assert bs.logins == 0 and bs.logouts == 0
    assert bs.queries == []
