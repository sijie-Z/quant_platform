"""Measure what the BUG-28 execution-timing change did to the numbers.

The engine used to apply new weights at the signal day's own close, earning the
move into the very close the signal was read from. It now trades at the next
trading day's close, which is what the module docstring, ASHARE_PITFALLS.md and
the T+1 notes had always claimed.

This is the third change to the numbers in this repository (after the cost
model and the covariance window), so the migration gets evidence rather than a
sentence in a pull request. Both timings run **in one process, on one dataset,
one config and one seed** -- the only difference between the two runs is
`execution_timing`, which isolates the change to the thing being changed.

Usage:
    python tools/execution_timing_migration.py --out migration.json

`signal_close` reproduces the pre-change semantics exactly and exists only for
this comparison. Do not use it for research.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))        # repo root, for `import main`
sys.path.insert(0, str(_HERE.parent.parent.parent))  # its parent, so `quant_platform` resolves to the repo

from quant_platform.backtest.cost_model import CostModel  # noqa: E402
from quant_platform.backtest.engine import BacktestEngine  # noqa: E402
from quant_platform.portfolio.constraints import PortfolioConstraints  # noqa: E402
from quant_platform.utils.config import load_config  # noqa: E402

import main as cli  # noqa: E402

TIMINGS = ("signal_close", "next_close")


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _run(config, signal, prices, returns, benchmark, sector_map, financials, timing: str):
    """`main._run_backtest`, with the execution-timing switch exposed."""
    constraints = PortfolioConstraints(
        long_only=config.portfolio.constraints.long_only,
        max_weight=config.portfolio.constraints.max_weight,
        max_sector_exposure=config.portfolio.constraints.max_sector_exposure,
        max_turnover=config.portfolio.constraints.max_turnover,
        lot_size=config.portfolio.constraints.lot_size,
    )
    cost_model = CostModel(
        commission=config.costs.commission,
        stamp_tax=config.costs.stamp_tax,
        slippage=config.costs.slippage,
        slippage_model=config.costs.slippage_model,
    )
    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        rebalance_frequency=config.backtest.rebalance_frequency,
        cost_model=cost_model,
        constraints=constraints,
        optimizer=config.portfolio.optimizer,
        benchmark=config.backtest.benchmark,
        covariance_method=config.portfolio.covariance.method,
        covariance_lookback=config.portfolio.covariance.lookback,
        execution_timing=timing,
    )
    return engine.run(
        signal=signal, prices=prices, returns=returns,
        benchmark_returns=benchmark, sector_map=sector_map, financials=financials,
    )


METRIC_KEYS = (
    "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown", "total_return", "win_rate",
    "information_ratio", "excess_return", "n_rebalances",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="execution_timing_migration.json")
    args = ap.parse_args()

    config = load_config()
    prices, returns, benchmark, metadata, financials, turnover = cli._load_data(
        config, use_tushare=False
    )
    processed_factors, _ic, sector_map, fin_unstacked = cli._compute_factors(
        prices, returns, financials, metadata, turnover, config=config
    )
    signal = cli._generate_signal(
        config, processed_factors, returns, prices=prices, volume=turnover
    )

    runs = {
        t: _run(config, signal, prices, returns, benchmark, sector_map, fin_unstacked, t)
        for t in TIMINGS
    }
    old, new = runs["signal_close"], runs["next_close"]

    old_daily, new_daily = old["daily_returns"], new["daily_returns"]
    common = old_daily.index.intersection(new_daily.index)
    delta = (new_daily.loc[common] - old_daily.loc[common]).dropna()

    report = {
        "baseline_commit": _git_rev(),
        "timings": {"old": "signal_close", "new": "next_close"},
        "data": {
            "provider": config.data.provider,
            "embedded_alpha": _synthetic_flag(config),
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
            "n_assets": int(prices.shape[1]) if hasattr(prices, "shape") else None,
            "n_days": int(len(common)),
        },
        "config": {
            "optimizer": config.portfolio.optimizer,
            "alpha_method": config.alpha.method,
            "rebalance_frequency": config.backtest.rebalance_frequency,
            "covariance_method": config.portfolio.covariance.method,
            "slippage_model": config.costs.slippage_model,
        },
        "metrics": {},
        "daily_delta": {
            "n_common_days": int(len(delta)),
            "mean": float(delta.mean()) if len(delta) else None,
            "median": float(delta.median()) if len(delta) else None,
            "max_abs": float(delta.abs().max()) if len(delta) else None,
            "std": float(delta.std()) if len(delta) else None,
            "n_nonzero": int((delta.abs() > 1e-12).sum()),
        },
        "cumulative": {
            "old": float((1 + old_daily).prod() - 1),
            "new": float((1 + new_daily).prod() - 1),
        },
        "turnover": {
            "old_sum": float(old["turnover_history"].sum()) if old.get("turnover_history") is not None else None,
            "new_sum": float(new["turnover_history"].sum()) if new.get("turnover_history") is not None else None,
            "old_first_date": str(old["turnover_history"].index[0].date()) if len(old["turnover_history"]) else None,
            "new_first_date": str(new["turnover_history"].index[0].date()) if len(new["turnover_history"]) else None,
        },
    }

    for key in METRIC_KEYS:
        o, n = old["summary"].get(key), new["summary"].get(key)
        if o is None and n is None:
            continue
        entry = {"old": _as_float(o), "new": _as_float(n)}
        if entry["old"] is not None and entry["new"] is not None:
            entry["delta"] = entry["new"] - entry["old"]
        report["metrics"][key] = entry

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten to {args.out}")
    return 0


def _as_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _synthetic_flag(config):
    """`config.data.synthetic` is declared as a dataclass but the loader leaves
    it a dict, so accept either."""
    syn = config.data.synthetic
    if isinstance(syn, dict):
        return syn.get("embedded_alpha")
    return getattr(syn, "embedded_alpha", None)


if __name__ == "__main__":
    raise SystemExit(main())
