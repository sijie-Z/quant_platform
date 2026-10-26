"""Indicative real-time monitor for the frozen forward targets.

This is deliberately separate from the official next-session-close ledger.
It marks the current target weights against Sina's realtime previous-close and
latest prices, producing an intraday *indicative* gross return only. It never
writes docs/research/paper_track ledgers and never simulates an official fill.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "docs" / "research" / "paper_track"
BASE = PAPER_ROOT / "forward_targets"
LEDGERS = {
    "turnover": PAPER_ROOT / "forward_2026_turnover",
    "ep_turnover": PAPER_ROOT / "forward_2026",
}
STOCKS = ROOT / ".scratch" / "wide_sina_2026" / "stocks"
DEFAULT_OUT = ROOT / ".scratch" / "forward_realtime"


def load_targets() -> dict[str, pd.Series]:
    targets = {}
    for name in ("turnover", "ep_turnover"):
        paths = sorted((BASE / name).glob("target_*.json"))
        if not paths:
            raise RuntimeError(f"no active target for {name}")
        payload = json.loads(paths[-1].read_text(encoding="utf-8"))
        targets[name] = pd.Series(payload["weights"], dtype=float)
    return targets


def load_book(name: str) -> dict | None:
    """Return the official end-of-day book when the ledger provides one."""
    path = LEDGERS[name] / "summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    book = payload.get("book")
    if not isinstance(book, dict) or not isinstance(book.get("positions"), list):
        return None
    if not isinstance(book.get("cash"), (int, float)):
        return None
    return book


def spot_panel() -> pd.DataFrame:
    import akshare as ak

    frame = ak.stock_zh_a_spot()
    code_col = "代码" if "代码" in frame.columns else "code"
    price_col = "最新价" if "最新价" in frame.columns else "price"
    prev_col = "昨收" if "昨收" in frame.columns else "prev_close"
    if frame.empty or not {code_col, price_col, prev_col}.issubset(frame.columns):
        raise RuntimeError("realtime spot response is empty or malformed")
    panel = pd.DataFrame({
        "code": frame[code_col].astype(str).str[-6:],
        "latest": pd.to_numeric(frame[price_col], errors="coerce"),
        "prev_close": pd.to_numeric(frame[prev_col], errors="coerce"),
    }).drop_duplicates("code").set_index("code")
    panel["return"] = panel["latest"] / panel["prev_close"] - 1.0
    return panel.replace([float("inf"), float("-inf")], pd.NA)


def strategy_snapshot(name: str, weights: pd.Series, spot: pd.DataFrame,
                      benchmark_return: float | None) -> dict:
    aligned = spot.reindex(weights.index)
    valid = (aligned["return"].notna() & (aligned["prev_close"] > 0)).fillna(False)
    missing = sorted(weights.index[~valid])
    gross_return = float((weights[valid] * aligned.loc[valid, "return"]).sum())
    covered_weight = float(weights[valid].sum())
    cash_weight = max(0.0, 1.0 - float(weights.sum()))
    indicative_nav = 2_000_000.0 * (1.0 + gross_return)
    return {
        "basis": "target_weights",
        "target_as_of": "2026-10-23",
        "target_assets": int(len(weights)),
        "quoted_assets": int(valid.sum()),
        "missing_assets": missing,
        "covered_weight": covered_weight,
        "cash_weight": cash_weight,
        "gross_return_since_previous_close": gross_return,
        "indicative_nav_gross": indicative_nav,
        "indicative_excess_vs_universe": (
            gross_return - benchmark_return if benchmark_return is not None else None
        ),
        "complete": not missing,
    }


def book_snapshot(book: dict, spot: pd.DataFrame) -> dict:
    """Mark the actual official positions and cash to the latest quotes."""
    cash = float(book["cash"])
    positions_value = float(book.get(
        "positions_value",
        sum(float(position["value"]) for position in book["positions"]),
    ))
    official_nav = cash + positions_value
    current_value = 0.0
    quoted = 0
    stale = []
    for position in book["positions"]:
        code = position["code"]
        reference = float(position["reference_price"])
        shares = position.get("shares")
        if shares is None:
            shares = float(position["value"]) / reference
        latest = spot["latest"].get(code, float("nan"))
        if pd.notna(latest) and float(latest) > 0:
            price = float(latest)
            quoted += 1
        else:
            price = reference
            stale.append(code)
        current_value += float(shares) * price
    indicative_nav = cash + current_value
    return {
        "basis": "official_book",
        "official_ledger_date": book["as_of"],
        "official_nav": official_nav,
        "indicative_nav": indicative_nav,
        "return_since_official_close": indicative_nav / official_nav - 1.0,
        "n_positions": len(book["positions"]),
        "quoted_assets": quoted,
        "stale_assets": stale,
        "cash": cash,
        "complete": not stale,
    }


def snapshot() -> dict:
    targets = load_targets()
    spot = spot_panel()
    universe = sorted({path.stem[-6:] for path in STOCKS.glob("*.parquet")})
    universe_returns = spot["return"].reindex(universe).dropna()
    benchmark_return = (
        float(universe_returns.mean()) if not universe_returns.empty else None
    )
    strategies = {}
    for name, weights in targets.items():
        book = load_book(name)
        if book is not None:
            strategies[name] = book_snapshot(book, spot)
        else:
            strategies[name] = strategy_snapshot(
                name, weights, spot, benchmark_return,
            )
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "indicative_realtime_monitor",
        "official_ledger": False,
        "price_basis": "latest/raw and previous-close/raw; reference only, not an execution price",
        "costs": "target mode excludes costs; official-book mode carries the official cash and booked costs",
        "source": "sina stock_zh_a_spot",
        "universe": {"assets": len(universe), "quoted": int(universe_returns.size),
                     "equal_weight_return": benchmark_return},
        "strategies": strategies,
    }


def write_snapshot(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    rows = []
    for name, values in payload["strategies"].items():
        rows.append({
            "timestamp": payload["timestamp"],
            "strategy": name,
            "basis": values.get("basis", "target_weights"),
            "gross_return": values.get(
                "gross_return_since_previous_close",
                values.get("return_since_official_close"),
            ),
            "indicative_nav_gross": values.get(
                "indicative_nav_gross", values.get("indicative_nav"),
            ),
            "universe_ew_return": payload["universe"]["equal_weight_return"],
            "quoted_assets": values["quoted_assets"],
            "complete": values["complete"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "snapshots.csv", mode="a",
                              header=not (out_dir / "snapshots.csv").exists(),
                              index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=0,
                        help="seconds between snapshots; 0 means one snapshot")
    parser.add_argument("--count", type=int, default=1,
                        help="number of snapshots when --interval is used")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    iterations = args.count if args.interval > 0 else 1
    for index in range(iterations):
        payload = snapshot()
        write_snapshot(payload, args.out_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if index + 1 < iterations:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
