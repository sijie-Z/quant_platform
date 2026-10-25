"""Tests for ``tools/daily_signal.py`` (R17).

Covers the taskbook acceptance points: PIT truncation at ``as_of`` (a fake
future row must not change a single weight), EP / turnover formulas against
hand-computed values, industry neutralisation on/off, the insufficient-data
exit path, byte-identical determinism, and the "weights only" output contract.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "daily_signal.py"

N_DAYS = 300
N_ASSETS = 30


@pytest.fixture(scope="module")
def tool():
    """Load tools/daily_signal.py as a module without packaging it."""
    spec = importlib.util.spec_from_file_location("daily_signal_r17", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flat(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def make_panel(
    n_days: int = N_DAYS, n_assets: int = N_ASSETS, seed: int = 7,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    """Synthetic (date, asset) daily cache with the real column set."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    assets = [f"{600000 + i:06d}" for i in range(n_assets)]
    pct = rng.normal(0.0004, 0.018, size=(n_days, n_assets))
    close = 10.0 * np.cumprod(1.0 + pct, axis=0)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    frame = pd.DataFrame(
        {
            "open": _flat(close * (1.0 + rng.normal(0.0, 0.002, size=close.shape))),
            "high": _flat(close * 1.01),
            "low": _flat(close * 0.99),
            "close": _flat(close),
            "preclose": _flat(np.vstack([close[:1], close[:-1]])),
            "volume": _flat(rng.lognormal(12.0, 0.4, size=close.shape)),
            "amount": _flat(rng.lognormal(16.0, 0.4, size=close.shape)),
            "turnover": _flat(rng.lognormal(-2.0, 0.5, size=close.shape)),
            "pct_chg": _flat(pct * 100.0),
            "pe_ttm": _flat(rng.uniform(5.0, 50.0, size=close.shape)),
            "pb_mrq": _flat(rng.uniform(0.5, 8.0, size=close.shape)),
        },
        index=index,
    )
    return frame, dates, assets


def append_future_rows(frame: pd.DataFrame, n_future: int = 5) -> pd.DataFrame:
    """Append post-as_of rows with violent moves (a leak would be obvious)."""
    last = frame.index.get_level_values("date").max()
    assets = frame.index.get_level_values("asset").unique()
    future_dates = pd.bdate_range(last + pd.Timedelta(days=1), periods=n_future)
    rows = []
    for _date in future_dates:
        for asset in assets:
            row = frame.loc[(last, asset)].copy()
            row["pct_chg"] = 50.0
            row["close"] = row["close"] * 1.5
            row["pe_ttm"] = 999.0
            row["pb_mrq"] = 999.0
            row["turnover"] = 99.0
            rows.append(row)
    index = pd.MultiIndex.from_product(
        [future_dates, assets], names=["date", "asset"],
    )
    extra = pd.DataFrame(rows, index=index)[frame.columns]
    return pd.concat([frame, extra]).sort_index()


def write_industry_map(
    path: Path, assets: list[str], n_sectors: int = 6,
) -> None:
    """Cyclic industry map; six sectors keep the 30% cap non-binding by default."""
    sectors = [f"S{i}" for i in range(n_sectors)]
    labels = [sectors[i % n_sectors] for i in range(len(assets))]
    pd.DataFrame(
        {"industry": labels}, index=pd.Index(assets, name="code"),
    ).to_parquet(path)


@pytest.fixture(scope="module")
def panels_dir(tmp_path_factory):
    """Shared synthetic cache (300 x 30) plus a future-row variant."""
    directory = tmp_path_factory.mktemp("r17_daily_signal")
    frame, dates, assets = make_panel()
    frame.to_parquet(directory / "panel.parquet")
    append_future_rows(frame).to_parquet(directory / "panel_future.parquet")
    write_industry_map(directory / "industry.parquet", assets)
    small_dates = set(dates[:100])
    small_mask = frame.index.get_level_values("date").isin(small_dates)
    frame.loc[small_mask].to_parquet(directory / "panel_small.parquet")
    return directory, dates[-1]


def run_tool(
    tool, cache: Path, out_dir: Path, *extra: str, as_of: str,
) -> tuple[int, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "targets.json"
    csv_path = out_dir / "targets.csv"
    argv = [
        "--cache", str(cache),
        "--as-of", as_of,
        "--factors", "ep,turnover20d",
        "--max-weight", "0.05",
        "--max-turnover", "0.30",
        "--capital", "2000000",
        "--out-json", str(json_path),
        "--out-csv", str(csv_path),
        *extra,
    ]
    return tool.main(argv), json_path, csv_path


def _reject_json_constant(value: str):
    raise AssertionError(f"non-finite JSON constant {value!r} in targets JSON")


def load_targets(path: Path) -> dict:
    """Strict JSON: NaN/Infinity tokens are rejected, not silently parsed."""
    return json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant,
    )


def test_pit_future_rows_do_not_change_weights(tool, panels_dir, tmp_path):
    """Acceptance (a): post-as_of rows with violent moves change nothing."""
    directory, as_of = panels_dir
    industry = directory / "industry.parquet"
    as_of_text = as_of.date().isoformat()

    code_base, json_base, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "base",
        "--industry-map", str(industry), as_of=as_of_text,
    )
    code_future, json_future, _ = run_tool(
        tool, directory / "panel_future.parquet", tmp_path / "future",
        "--industry-map", str(industry), as_of=as_of_text,
    )
    assert code_base == 0
    assert code_future == 0

    base = load_targets(json_base)
    future = load_targets(json_future)
    # Bit-for-bit identical weights: the future rows never entered the slice.
    assert future["weights"] == base["weights"]
    assert future["signal_date"] == base["signal_date"] == as_of_text
    assert future["data_end"] == base["data_end"] == as_of_text
    assert (
        future["diagnostics"]["factor_ic"] == base["diagnostics"]["factor_ic"]
    )
    # PIT evidence: the IC window ends strictly before as_of.
    window = base["diagnostics"]["factor_ic"]["window"]
    assert window is not None
    assert window["end"] < as_of_text
    assert window["n"] > 0


def test_ep_and_turnover_formulas(tool):
    """Acceptance (b): ep = 1/PE (PE<=0 -> NaN), turnover20d = 20d mean."""
    assets = ["000001", "000002", "000003", "000004"]
    dates = pd.bdate_range("2023-01-02", periods=25)

    pe = pd.DataFrame(
        [[10.0, 20.0, -5.0, 0.0], [5.0, np.nan, 25.0, -1.0], [8.0, 12.0, 16.0, 100.0]],
        index=dates[:3], columns=assets,
    )
    pb = pd.DataFrame(
        [[2.0, 4.0, 0.0, -1.0], [5.0, np.nan, 8.0, 10.0], [1.0, 2.0, 3.0, 4.0]],
        index=dates[:3], columns=assets,
    )
    close = pd.DataFrame(10.0, index=dates[:3], columns=assets)
    pct = pd.DataFrame(0.0, index=dates[:3], columns=assets)
    long = pd.concat(
        [close.stack(), pct.stack(), pe.stack(), pb.stack()],
        axis=1, keys=["close", "pct_chg", "pe_ttm", "pb_mrq"],
    )
    long.index = long.index.set_names(["date", "asset"])
    panels = tool.build_panels(long)
    ep = tool.raw_factor_panel("ep", panels)
    bp = tool.raw_factor_panel("bp", panels)
    expected_ep = (1.0 / pe.where(pe > 0)).reindex_like(ep)
    expected_bp = (1.0 / pb.where(pb > 0)).reindex_like(bp)
    pd.testing.assert_frame_equal(ep, expected_ep, check_dtype=False, check_names=False)
    pd.testing.assert_frame_equal(bp, expected_bp, check_dtype=False, check_names=False)

    # turnover20d: two assets, 25 sessions; A = 1..25, B = 10 constant.
    assets2 = ["000001", "000002"]
    idx2 = pd.MultiIndex.from_product([dates, assets2], names=["date", "asset"])
    values_a = np.arange(1.0, 26.0)
    flat_turnover = np.column_stack([values_a, np.full(25, 10.0)]).reshape(-1)
    turnover = pd.DataFrame({"turnover": flat_turnover}, index=idx2)
    turnover["close"] = 10.0
    turnover["pct_chg"] = 0.0
    panels2 = tool.build_panels(turnover)
    turnover20d = tool.raw_factor_panel("turnover20d", panels2)
    assert math.isnan(turnover20d.loc[dates[18], "000001"])
    assert turnover20d.loc[dates[19], "000001"] == pytest.approx(10.5)
    assert turnover20d.loc[dates[-1], "000001"] == pytest.approx(15.5)
    assert turnover20d.loc[dates[-1], "000002"] == pytest.approx(10.0)


def test_build_signal_missing_factor_is_neutral(tool):
    """A missing factor value is a zero, never a per-asset renormalisation."""
    date = pd.Timestamp("2023-03-01")
    assets = ["000001", "000002", "000003", "000004"]
    ep = pd.DataFrame(
        [[np.nan, 0.0, 2.5, np.nan]], index=[date], columns=assets,
    )
    turnover = pd.DataFrame(
        [[-2.0, -2.0, 0.0, np.nan]], index=[date], columns=assets,
    )
    composite = tool.build_signal(
        {"ep": ep, "turnover20d": turnover},
        {"ep": 0.5, "turnover20d": -0.5},
        date,
    )
    # EP missing (000001) must score exactly like EP = 0 (000002).
    assert composite["000001"] == pytest.approx(composite["000002"])
    assert composite["000001"] == pytest.approx(1.0)
    assert composite["000003"] == pytest.approx(1.25)
    # Only an asset that is NaN for every weighted factor gets no weight.
    assert math.isnan(composite["000004"])


def test_industry_neutralization_on_off(tool, panels_dir, tmp_path):
    """Acceptance (c): both modes give finite long-only weights within caps."""
    directory, as_of = panels_dir
    as_of_text = as_of.date().isoformat()
    cache = directory / "panel.parquet"

    code_on, json_on, _ = run_tool(
        tool, cache, tmp_path / "on",
        "--industry-map", str(directory / "industry.parquet"), as_of=as_of_text,
    )
    code_off, json_off, _ = run_tool(
        tool, cache, tmp_path / "off",
        "--max-sector", "1.0", as_of=as_of_text,
    )
    assert code_on == 0
    assert code_off == 0

    on = load_targets(json_on)
    off = load_targets(json_off)
    for payload, neutralized in ((on, True), (off, False)):
        weights = payload["weights"]
        assert weights, "expected a non-empty book"
        assert payload["provenance"]["industry_neutralization"] is neutralized
        values = np.array(list(weights.values()), dtype=float)
        assert np.isfinite(values).all()
        assert (values > 0.0).all()
        assert values.max() <= 0.05 + 1e-9
        total = values.sum()
        assert 0.999 <= total <= 1.0 + 1e-6
        assert abs(payload["diagnostics"]["gross_weight"] - total) < 1e-9
        assert payload["diagnostics"]["covariance_sessions"] == 252
        assert payload["diagnostics"]["covariance_input_sessions"] == 299
    assert not any(
        "industry map" in warning for warning in on["diagnostics"]["warnings"]
    )
    assert any(
        "industry map" in warning for warning in off["diagnostics"]["warnings"]
    )


def test_insufficient_data_exits_nonzero(tool, panels_dir, tmp_path, capsys):
    """Acceptance (d): fewer than 252 sessions -> non-zero, no half-written files."""
    directory, _ = panels_dir
    json_path = tmp_path / "targets.json"
    csv_path = tmp_path / "targets.csv"
    code = tool.main([
        "--cache", str(directory / "panel_small.parquet"),
        "--as-of", "2022-05-20",
        "--out-json", str(json_path),
        "--out-csv", str(csv_path),
    ])
    assert code != 0
    captured = capsys.readouterr()
    assert "insufficient" in captured.err.lower()
    assert not json_path.exists()
    assert not csv_path.exists()


def test_deterministic_json(tool, panels_dir, tmp_path):
    """Acceptance (e): same input twice -> byte-identical JSON."""
    directory, as_of = panels_dir
    as_of_text = as_of.date().isoformat()
    args = ("--industry-map", str(directory / "industry.parquet"))
    code_a, json_a, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "a", *args, as_of=as_of_text,
    )
    code_b, json_b, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "b", *args, as_of=as_of_text,
    )
    assert code_a == code_b == 0
    assert json_a.read_text(encoding="utf-8") == json_b.read_text(encoding="utf-8")


def test_outputs_are_weights_only(tool, panels_dir, tmp_path):
    """Acceptance (f): weights + target notionals, no performance reporting."""
    directory, as_of = panels_dir
    code, json_path, csv_path = run_tool(
        tool, directory / "panel.parquet", tmp_path / "schema",
        "--industry-map", str(directory / "industry.parquet"),
        as_of=as_of.date().isoformat(),
    )
    assert code == 0
    payload = load_targets(json_path)
    assert set(payload) == {
        "as_of", "data_end", "factors", "weights", "signal_date",
        "provenance", "diagnostics",
    }
    for forbidden in ("sharpe", "annual_return", "max_drawdown", "nav", "pnl"):
        assert forbidden not in json.dumps(payload).lower()

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "code,weight,target_value_cny"
    assert len(lines) - 1 == len(payload["weights"])
    capital = payload["provenance"]["capital_cny"]
    for line in lines[1:]:
        code_text, weight_text, value_text = line.split(",")
        assert code_text in payload["weights"]
        assert float(weight_text) == pytest.approx(payload["weights"][code_text])
        assert float(value_text) == pytest.approx(
            payload["weights"][code_text] * capital, abs=0.01,
        )


def test_prev_weights_and_turnover_cap(tool, panels_dir, tmp_path):
    """First build vs a prior book: flags, bare-map parsing, turnover cap."""
    directory, as_of = panels_dir
    as_of_text = as_of.date().isoformat()
    industry = str(directory / "industry.parquet")
    cache = directory / "panel.parquet"

    code_first, json_first, _ = run_tool(
        tool, cache, tmp_path / "first",
        "--industry-map", industry, as_of=as_of_text,
    )
    assert code_first == 0
    first = load_targets(json_first)
    assert first["diagnostics"]["first_build"] is True
    # First build: turnover against a zero book is 0.5 * sum(|w|).
    assert first["diagnostics"]["turnover"] == pytest.approx(
        0.5 * first["diagnostics"]["gross_weight"], abs=1e-6,
    )
    # The turnover cap cannot apply against a zero book; that must be loud.
    assert any(
        "first build" in warning and "turnover" in warning
        for warning in first["diagnostics"]["warnings"]
    )

    # Chain the previous targets.json into the next run.
    code_chain, json_chain, _ = run_tool(
        tool, cache, tmp_path / "chain",
        "--industry-map", industry, "--prev-weights", str(json_first),
        as_of=as_of_text,
    )
    assert code_chain == 0
    chained = load_targets(json_chain)
    assert chained["diagnostics"]["first_build"] is False
    assert chained["diagnostics"]["turnover"] <= 0.30 + 1e-6
    # The optimiser re-solves with the cost terms active, so the book stays
    # put to solver tolerance rather than to the last bit.
    assert set(chained["weights"]) == set(first["weights"])
    for code, weight in first["weights"].items():
        assert chained["weights"][code] == pytest.approx(weight, abs=1e-5)

    # A bare {code: weight} mapping is accepted and behaves identically.
    bare = tmp_path / "bare_prev.json"
    bare.write_text(json.dumps(first["weights"]), encoding="utf-8")
    code_bare, json_bare, _ = run_tool(
        tool, cache, tmp_path / "bare",
        "--industry-map", industry, "--prev-weights", str(bare),
        as_of=as_of_text,
    )
    assert code_bare == 0
    assert load_targets(json_bare)["weights"] == chained["weights"]

    # A deliberately different prior book must be traded toward, not dumped:
    # the one-sided turnover cap holds and every weight stays within bounds.
    codes = sorted(first["weights"])[:20]
    different = tmp_path / "different_prev.json"
    different.write_text(json.dumps({code: 0.05 for code in codes}), encoding="utf-8")
    code_cap, json_cap, _ = run_tool(
        tool, cache, tmp_path / "cap",
        "--industry-map", industry, "--prev-weights", str(different),
        "--max-turnover", "0.30", as_of=as_of_text,
    )
    assert code_cap == 0
    capped = load_targets(json_cap)
    assert capped["diagnostics"]["turnover"] <= 0.30 + 1e-6
    values = np.array(list(capped["weights"].values()), dtype=float)
    assert (values >= 0.0).all()
    assert values.max() <= 0.05 + 1e-9
    assert values.sum() <= 1.0 + 1e-6


def test_exclude_codes_file_drops_candidates(tool, panels_dir, tmp_path):
    """ST-style exclusion: listed codes never reach the target book."""
    directory, as_of = panels_dir
    frame = pd.read_parquet(directory / "panel.parquet")
    assets = frame.index.get_level_values("asset").unique()
    excluded = sorted({str(assets[0]), str(assets[1])})
    exclude_path = tmp_path / "exclude.txt"
    # Mixed formats exercise _normalize_code (shXXXXXX and bare 6 digits).
    exclude_path.write_text(
        f"sh{excluded[0]}\n{excluded[1]}\n", encoding="utf-8",
    )
    rc, json_path, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "excluded",
        "--exclude-codes-file", str(exclude_path),
        "--industry-map", str(directory / "industry.parquet"),
        as_of=as_of.strftime("%Y-%m-%d"),
    )
    assert rc == 0
    payload = load_targets(json_path)
    weights = set(payload["weights"])
    assert not (weights & set(excluded))
    diagnostics = payload["diagnostics"]
    assert set(diagnostics["excluded_dropped"]) == set(excluded)
    assert payload["provenance"]["excluded_codes_count"] == 2
    assert any("excluded" in warning for warning in diagnostics["warnings"])


def test_stale_cache_warns(tool, panels_dir, tmp_path):
    """A cache that ends long before --as-of must not pass silently."""
    directory, _ = panels_dir
    code, json_path, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "stale",
        "--max-sector", "1.0", as_of="2023-06-30",
    )
    assert code == 0
    payload = load_targets(json_path)
    assert payload["signal_date"] == "2023-02-24"
    assert payload["data_end"] == "2023-02-24"
    assert payload["diagnostics"]["data_lag_days"] > 7
    assert any(
        "stale cache" in warning for warning in payload["diagnostics"]["warnings"]
    )


def test_invalid_cli_inputs_fail_cleanly(tool, panels_dir, tmp_path, capsys):
    """Bad --as-of / --capital exit 2 without tracebacks or half-written files."""
    directory, as_of = panels_dir
    cache = directory / "panel.parquet"
    json_path = tmp_path / "bad.json"
    csv_path = tmp_path / "bad.csv"

    code = tool.main([
        "--cache", str(cache),
        "--as-of", "not-a-date",
        "--out-json", str(json_path),
        "--out-csv", str(csv_path),
    ])
    assert code == 2
    assert "could not parse --as-of" in capsys.readouterr().err
    assert not json_path.exists()

    with pytest.raises(SystemExit) as excinfo:
        tool.main([
            "--cache", str(cache),
            "--as-of", as_of.date().isoformat(),
            "--capital", "nan",
            "--out-json", str(json_path),
            "--out-csv", str(csv_path),
        ])
    assert excinfo.value.code == 2
    assert not json_path.exists()


def test_prev_over_cap_is_a_clear_input_error(
    tool, panels_dir, tmp_path, capsys,
):
    """An over-cap prior book that the turnover cap cannot repair fails loudly."""
    directory, as_of = panels_dir
    as_of_text = as_of.date().isoformat()
    industry = str(directory / "industry.parquet")
    cache = directory / "panel.parquet"

    code_first, json_first, _ = run_tool(
        tool, cache, tmp_path / "first",
        "--industry-map", industry, as_of=as_of_text,
    )
    assert code_first == 0
    codes = sorted(load_targets(json_first)["weights"])[:20]
    prev = {codes[0]: 0.5}
    prev.update({code: 0.025 for code in codes[1:]})
    prev_path = tmp_path / "over_cap_prev.json"
    prev_path.write_text(json.dumps(prev), encoding="utf-8")

    json_path = tmp_path / "targets.json"
    csv_path = tmp_path / "targets.csv"
    code = tool.main([
        "--cache", str(cache),
        "--as-of", as_of_text,
        "--factors", "ep,turnover20d",
        "--industry-map", industry,
        "--prev-weights", str(prev_path),
        "--out-json", str(json_path),
        "--out-csv", str(csv_path),
    ])
    assert code == 2
    assert "prev weights exceed --max-weight" in capsys.readouterr().err
    assert not json_path.exists()
    assert not csv_path.exists()


# ---------------------------------------------------------------------------
# R26: portfolio-level sector cap and construction modes
# ---------------------------------------------------------------------------

def test_project_sector_cap_scales_two_50pct_sectors(tool):
    """Acceptance (a): two sectors at 50% each -> each <= 30%, caps projected."""
    weights = pd.Series({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    sectors = pd.Series({"a": "X", "b": "X", "c": "Y", "d": "Y"})
    projected = tool.project_sector_cap(weights, sectors, 0.30)
    totals = tool.sector_weights(projected, sectors)
    assert float(totals.max()) <= 0.30 + 1e-12
    assert totals["X"] == pytest.approx(0.30)
    assert totals["Y"] == pytest.approx(0.30)
    assert float(projected.sum()) == pytest.approx(0.60)
    # Projection only scales down; the per-name cap can never be breached.
    assert float(projected.max()) <= 0.25 + 1e-12


def test_two_sector_book_is_capped_or_fails_loudly(
    tool, panels_dir, tmp_path, capsys,
):
    """Acceptance (a, end to end): a 2-sector universe cannot fund a 30% cap."""
    directory, as_of = panels_dir
    as_of_text = as_of.date().isoformat()
    assets = list(pd.read_parquet(directory / "industry.parquet").index)
    split = len(assets) // 2
    two_sector = pd.DataFrame(
        {"industry": ["X"] * split + ["Y"] * (len(assets) - split)},
        index=pd.Index(assets, name="code"),
    )
    two_sector.to_parquet(tmp_path / "industry_two.parquet")

    code, json_path, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "two",
        "--industry-map", str(tmp_path / "industry_two.parquet"),
        as_of=as_of_text,
    )
    if code == 0:
        payload = load_targets(json_path)
        diagnostics = payload["diagnostics"]
        assert diagnostics["max_sector_weight"] <= 0.30 + 1e-6
        assert diagnostics["sector_cap"]["n_sectors_holdings"] <= 2
        # Two 30% caps cannot fund a gross-1 book: the fallback is explicit.
        assert diagnostics["optimizer"]["fallback_count"] >= 1
        assert any(
            "fell back" in warning
            for warning in diagnostics["warnings"]
        )
        assert payload["diagnostics"]["gross_weight"] <= 0.60 + 1e-6
    else:
        error = capsys.readouterr().err.lower()
        assert "sector" in error
        assert "cap" in error or "infeasible" in error


def test_default_construction_is_mvo(tool, panels_dir, tmp_path):
    """Acceptance (b): no --construction -> R25's recommended MVO default."""
    directory, as_of = panels_dir
    code, json_path, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "default",
        "--industry-map", str(directory / "industry.parquet"),
        as_of=as_of.date().isoformat(),
    )
    assert code == 0
    payload = load_targets(json_path)
    assert payload["provenance"]["construction"] == "mvo"
    assert payload["provenance"]["top_n"] is None
    assert payload["diagnostics"]["optimizer"]["name"] == "MeanVarianceOptimizer"


def test_topn_ew_is_equal_weight_and_capped(tool, panels_dir, tmp_path):
    """Acceptance (b): topn_ew outputs equal weights within the caps."""
    directory, as_of = panels_dir
    code, json_path, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "ew",
        "--industry-map", str(directory / "industry.parquet"),
        "--construction", "topn_ew", "--top-n", "20",
        as_of=as_of.date().isoformat(),
    )
    assert code == 0
    payload = load_targets(json_path)
    assert payload["provenance"]["construction"] == "topn_ew"
    assert payload["provenance"]["top_n"] == 20
    values = np.array(list(payload["weights"].values()), dtype=float)
    assert len(values) == 20
    assert np.allclose(values, 1.0 / 20.0, atol=1e-9)
    assert float(values.max()) <= 0.05 + 1e-9
    assert payload["diagnostics"]["max_sector_weight"] <= 0.30 + 1e-6


def test_topn_rank_weights_are_monotone(tool):
    """Acceptance (b): top-N rank weights decrease with the signal rank."""
    assets = ["a", "b", "c", "d", "e", "f", "g"]
    alpha = pd.Series(
        {"a": 0.4, "b": 0.2, "c": 0.0, "d": -0.2, "e": -0.4, "f": 0.1, "g": -0.1}
    )
    cov = pd.DataFrame(
        np.eye(len(assets)), index=assets, columns=assets,
    )
    constraints = tool.PortfolioConstraints(
        max_weight=0.30, max_sector_exposure=0.30, max_turnover=0.30,
    )
    weights, info = tool.build_simple_book(
        "topn_rank", alpha, cov, None, None, constraints, 5, [],
    )
    assert info["selected"] == 5
    selected = ["a", "b", "f", "c", "g"]
    ordered = [float(weights[name]) for name in selected]
    assert ordered == sorted(ordered, reverse=True)
    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights.max()) <= 0.30 + 1e-12
    assert float(weights.min()) >= 0.0


def test_risk_parity_construction_is_finite_and_capped(
    tool, panels_dir, tmp_path,
):
    """Acceptance (b): risk_parity weights are finite and inside both caps.

    The synthetic fixture has homogeneous volatilities, so risk parity is
    close to equal weight here; non-degeneracy is covered separately by
    ``test_risk_parity_is_non_degenerate_with_heterogeneous_covariance``.
    """
    directory, as_of = panels_dir
    code, json_path, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "rp",
        "--industry-map", str(directory / "industry.parquet"),
        "--construction", "risk_parity", "--top-n", "20",
        as_of=as_of.date().isoformat(),
    )
    assert code == 0
    payload = load_targets(json_path)
    values = np.array(list(payload["weights"].values()), dtype=float)
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()
    assert float(values.max()) <= 0.05 + 1e-6
    assert float(values.sum()) <= 1.0 + 1e-6
    assert payload["diagnostics"]["max_sector_weight"] <= 0.30 + 1e-6
    assert payload["diagnostics"]["optimizer"]["name"] == "RiskParityOptimizer"


def test_risk_parity_is_non_degenerate_with_heterogeneous_covariance(tool):
    """Risk parity must differ from equal weight when volatilities differ."""
    assets = [f"a{i}" for i in range(10)]
    alpha = pd.Series(np.linspace(0.4, -0.4, len(assets)), index=assets)
    variances = np.linspace(0.01, 0.10, len(assets))
    cov = pd.DataFrame(
        np.diag(variances), index=assets, columns=assets,
    )
    constraints = tool.PortfolioConstraints(
        max_weight=0.20, max_sector_exposure=0.30, max_turnover=0.30,
    )
    weights, info = tool.build_simple_book(
        "risk_parity", alpha, cov, None, None, constraints, len(assets), [],
    )
    assert info["fallback_count"] == 0
    values = weights.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert float(values.sum()) == pytest.approx(1.0)
    assert float(values.max()) <= 0.20 + 1e-6
    assert float(np.ptp(values)) > 1e-3


def test_mvo_receives_the_real_sector_map(
    tool, panels_dir, tmp_path, monkeypatch,
):
    """Acceptance (a, instrumented): the solver gets the real multi-sector map."""
    directory, as_of = panels_dir
    captured: dict = {}
    real_optimize = tool.MeanVarianceOptimizer.optimize

    def spy(self, signal, cov_matrix=None, prices=None, prev_weights=None,
            sector_map=None, adv=None, aum=None):
        captured["sector_map"] = (
            None if sector_map is None else sector_map.copy()
        )
        captured["max_sector_exposure"] = self.constraints.max_sector_exposure
        return real_optimize(
            self, signal, cov_matrix, prices, prev_weights, sector_map, adv, aum,
        )

    monkeypatch.setattr(tool.MeanVarianceOptimizer, "optimize", spy)
    code, _, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "spy",
        "--industry-map", str(directory / "industry.parquet"),
        as_of=as_of.date().isoformat(),
    )
    assert code == 0
    sector_map = captured["sector_map"]
    assert sector_map is not None
    assert sector_map.nunique() > 1
    # The baseline path passed sector_map=None (cap silently skipped); the
    # fixture map has S0..S5, so the solver must receive multiple industries.
    assert "A" not in set(sector_map.unique())
    assert captured["max_sector_exposure"] == pytest.approx(0.30)


def test_deterministic_simple_construction(tool, panels_dir, tmp_path):
    """Acceptance (c): a top-N construction is byte-identical across runs."""
    directory, as_of = panels_dir
    args = (
        "--industry-map", str(directory / "industry.parquet"),
        "--construction", "topn_rank", "--top-n", "20",
    )
    code_a, json_a, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "ra", *args,
        as_of=as_of.date().isoformat(),
    )
    code_b, json_b, _ = run_tool(
        tool, directory / "panel.parquet", tmp_path / "rb", *args,
        as_of=as_of.date().isoformat(),
    )
    assert code_a == code_b == 0
    assert json_a.read_text(encoding="utf-8") == json_b.read_text(encoding="utf-8")


def test_chained_targets_respect_sector_cap(tool, panels_dir, tmp_path):
    """Acceptance (e): monthly chaining keeps every target's sectors <= 30%."""
    directory, _ = panels_dir
    frame = pd.read_parquet(directory / "panel.parquet")
    dates = sorted(pd.Index(frame.index.get_level_values("date")).unique())
    picks = [dates[-40], dates[-20], dates[-1]]
    industry = str(directory / "industry.parquet")
    cache = directory / "panel.parquet"
    previous: Path | None = None
    for index, session in enumerate(picks):
        extra = ["--industry-map", industry]
        if previous is not None:
            extra += ["--prev-weights", str(previous)]
        code, json_path, _ = run_tool(
            tool, cache, tmp_path / f"chain{index}", *extra,
            as_of=pd.Timestamp(session).date().isoformat(),
        )
        assert code == 0
        payload = load_targets(json_path)
        diagnostics = payload["diagnostics"]
        assert diagnostics["max_sector_weight"] <= 0.30 + 1e-6
        if index > 0:
            assert diagnostics["first_build"] is False
            assert diagnostics["turnover"] <= 0.30 + 1e-6
        previous = json_path
