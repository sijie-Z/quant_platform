"""Offline synthetic factor runner (P0 validation path).

Validates the research spine without any network dependency:
    synthetic prices -> factor -> cross-sectional rank IC -> Registry -> report

Usage:
    python -m lab.runs.synthetic_factor_run momentum_12m
    python -m lab.runs.synthetic_factor_run list
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quant_platform.factors.evaluation import ic_summary  # noqa: E402
from quant_platform.factors.technical import register_all as register_technical  # noqa: E402
from quant_platform.lab.registry import DEFAULT_DB, RunStore  # noqa: E402
from quant_platform.lab.reports import generate_report  # noqa: E402


def available_factors() -> list[str]:
    register_technical()
    from quant_platform.factors.registry import get_registry
    return sorted(get_registry().list_all())


def run_synthetic_factor(
    factor_name: str,
    n_stocks: int = 5,
    n_days: int = 500,
    seed: int = 7,
    db_path: str | Path = DEFAULT_DB,
    out_dir: str | Path = "data/reports",
) -> str:
    """Run one technical factor offline and persist the research record.

    Args:
        factor_name: Registered factor to evaluate.
        n_stocks: Width of the synthetic panel.
        n_days: Length of the synthetic panel.
        seed: RNG seed, for reproducibility.
        db_path: Research Registry to write into. Defaults to the real one.
            Tests MUST override this and `out_dir` -- `research_runs` is the
            project's research knowledge base, not a scratch table.
        out_dir: Directory for the generated Markdown report.
    """
    register_technical()
    from quant_platform.factors.registry import get_registry

    cls = get_registry().get(factor_name)
    if cls is None:
        raise ValueError(
            f"Unknown factor: {factor_name}. Available: {available_factors()}"
        )

    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.02, size=(n_days, n_stocks))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_stocks)])

    factor = cls().compute(panel)
    fwd = panel.pct_change(fill_method=None).shift(-1)

    ic_vals = []
    for dt in factor.index:
        f = factor.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) >= 3:
            ic = f.loc[common].rank().corr(r.loc[common].rank(), method="pearson")
            if not np.isnan(ic):
                ic_vals.append((dt, ic))

    ic_series = pd.Series(
        [v for _, v in ic_vals],
        index=pd.DatetimeIndex([d for d, _ in ic_vals]),
        name="rank_ic",
    )
    stats = ic_summary(ic_series) if len(ic_series) else {}

    store = RunStore(str(db_path))
    run_id = store.begin_run(f"synthetic_{factor_name}", {
        "data_source": "synthetic",
        "factor": factor_name,
        "factor_params": getattr(cls(), "params", {}),
        "warnings": [],
    })
    evaluation = {
        "n_ic_obs": int(len(ic_series)),
        "ic_mean": round(float(stats.get("mean_ic", float("nan"))), 4) if stats else None,
        "icir": round(float(stats.get("icir", float("nan"))), 4) if stats else None,
        "ic_positive_ratio": round(float(stats.get("ic_positive_ratio", float("nan"))), 4) if stats else None,
    }
    store.finish_run(run_id, status="success", evaluation=evaluation, report_path=None, warnings=[])
    report_path = generate_report(store.get_run(run_id), out_dir=str(out_dir))
    store.finish_run(run_id, status="success", evaluation=evaluation, report_path=report_path, warnings=[])

    print(f"[synthetic_factor_run] {factor_name}: IC={evaluation['ic_mean']} "
          f"ICIR={evaluation['icir']} run={run_id} report={report_path}")
    return run_id


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print("Available factors:")
        for name in available_factors():
            print(f"  {name}")
        return 0
    run_synthetic_factor(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
