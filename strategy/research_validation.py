"""Research Validation Sprint — unified evidence quality assessment.

Integrates IC evaluation, walk-forward validation, Factor Store persistence,
and Strategy Gates into a single comprehensive validation run.

Inspired by quawn's Bounded Research Validation Sprint design.

Modes:
  - quick:  Limited factors × limited symbols × limited folds. Minutes.
  - full:   All factors × full universe × default folds. Hours.

Output: Evidence quality report with per-factor and per-strategy scores.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from quant_platform.research.holdout import HoldoutSpec, load_configured_holdout
from quant_platform.utils.logging import get_logger
from quant_platform.utils.research_window import resolve_end_date

logger = get_logger(__name__)

# Walk-forward window used by :meth:`ResearchValidator._step_walk_forward`.
# Held as constants so the "holdout left too little data" check can never
# drift from the validator configuration it protects.
_WF_TRAIN_PERIOD = 504
_WF_TEST_PERIOD = 126

# Sentinel: "holdout not passed explicitly -> resolve it from the platform
# configuration". An explicit ``None`` means the caller disabled it.
_HOLDOUT_UNSET = object()


def mask_research_inputs(
    spec: HoldoutSpec,
    *,
    prices: pd.DataFrame | None,
    returns: pd.DataFrame | None,
    benchmark: pd.Series | None,
    financials: pd.DataFrame | None,
    turnover: pd.DataFrame | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop holdout dates from every date-indexed research input.

    The holdout must never take part in a choice, so it is removed *before*
    factor computation, IC evaluation, factor screening and walk-forward —
    not merely before the final metrics. Index order of the remaining rows is
    preserved (the mask keeps the original sequence). ``financials`` is a
    ``(date, asset)`` MultiIndex panel, hence the ``level="date"`` mask;
    ``None`` inputs pass through.

    Returns ``(masked_inputs, holdout_info)`` where ``holdout_info`` is the
    ``{"spec": {...}, "excluded_days": N}`` record written into the report.
    """
    excluded = int(spec.excluded_days(returns.index)) if returns is not None else 0
    masked = {
        "prices": spec.mask_frame(prices),
        "returns": spec.mask_frame(returns),
        "benchmark": spec.mask_frame(benchmark),
        "financials": spec.mask_frame(financials, level="date"),
        "turnover": spec.mask_frame(turnover),
    }
    info = {"spec": spec.to_dict(), "excluded_days": excluded}
    return masked, info


def _panel_ic_results(
    processed_factors: dict, returns: pd.DataFrame | None,
) -> tuple[dict, list[str]]:
    """IC moments for each processed factor panel (shared by step 3 and R3).

    Returns ``(ic_results, warnings)``; a factor whose evaluation fails is
    reported as a warning and omitted rather than stored as a fake zero.
    """
    from quant_platform.factors.evaluation import (
        ic_summary,
        pearson_ic,
        rank_ic,
    )

    ic_results: dict = {}
    warnings: list[str] = []
    for name, factor in processed_factors.items():
        try:
            ic = rank_ic(factor, returns)
            summary = ic_summary(ic)
            # `ic_summary` only sees the IC series; pearson IC and coverage
            # need the factor panel. The old code read keys ic_summary never
            # returns, so every saved evaluation row carried pearson_ic=0
            # and coverage=0.
            try:
                p_ic = pearson_ic(factor, returns)
                pearson_mean = (
                    float(p_ic.mean()) if len(p_ic) else float("nan")
                )
            except Exception:
                # A failed Pearson computation is missing data, not 0.
                pearson_mean = float("nan")
            coverage = (
                float(factor.notna().mean().mean()) if factor.size else 0.0
            )
            ic_results[name] = {
                "mean_ic": summary["mean_ic"],
                "std_ic": summary.get("std_ic", 0.0),
                "icir": summary["icir"],
                "pearson_ic": pearson_mean,
                "coverage": coverage,
                "n_assets": int(factor.notna().any().sum()),
            }
        except Exception as exc:
            warnings.append(f"IC evaluation failed for {name}: {exc}")
    return ic_results, warnings


def _finite(value: Any) -> float:
    """Coerce a metric to float without inventing a zero for missing data."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def signal_date_for_panel(factor: pd.DataFrame | None) -> str | None:
    """Last date on which the factor panel has an observed value.

    The old writer stamped every evaluation row with ``self.end_date`` (a
    configured constant, not data), so the factor store was not a time series
    — the live database held 154 rows all dated 2022-01-01. The evaluation
    date must come from the panel that was actually evaluated.
    """
    if factor is None or len(factor) == 0:
        return None
    observed = factor.notna().any(axis=1)
    if not bool(observed.any()):
        return None
    last = factor.index[observed.to_numpy()][-1]
    try:
        return str(pd.Timestamp(last).date())
    except (TypeError, ValueError):
        return str(last)


def stability_from_fold_metrics(fold_metrics: list[dict] | None) -> dict:
    """Walk-forward stability statistics computed from real fold metrics.

    The old writer persisted ``sharpe_mean/std/consistency/positive_folds``
    as zeros for every factor. That made 30% of the factor health score
    structurally zero and made a factor that was never walk-forward validated
    look like one that was measured and found unstable. When there are no
    usable folds the statistics are NaN/None and ``status="unavailable"`` —
    the caller stores that flag in the row warnings, so missing data never
    masquerades as a measured zero.
    """
    sharpes: list[float] = []
    skipped = 0
    for metric in fold_metrics or []:
        value = metric.get("sharpe_ratio")
        try:
            value = float(value)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if math.isfinite(value):
            sharpes.append(value)
        else:
            skipped += 1

    if not sharpes:
        return {
            "status": "unavailable",
            "sharpe_mean": float("nan"),
            "sharpe_std": float("nan"),
            "sharpe_consistency": float("nan"),
            "positive_folds": None,
            "total_folds": None,
            "n_skipped_folds": skipped,
        }

    values = np.asarray(sharpes, dtype=float)
    return {
        "status": "computed",
        "sharpe_mean": float(values.mean()),
        "sharpe_std": float(values.std()),
        "sharpe_consistency": float((values > 0).mean()),
        "positive_folds": int((values > 0).sum()),
        "total_folds": int(values.size),
        "n_skipped_folds": skipped,
    }


def select_walk_forward_factors(
    ic_results: dict,
    processed_factors: dict,
    returns: pd.DataFrame,
    *,
    max_factors: int = 2,
    significance_level: float = 0.05,
) -> dict:
    """Pick walk-forward factors after an IC t-test with BH-FDR control.

    The old selector took the top ``max_factors`` by full-sample ``|ICIR|``
    with no significance test, so a noise factor could reach the
    walk-forward stage (and its OOS numbers were then selection-biased).

    Per-factor significance is a two-sided t-test on the IC series,
    approximated in the normal tail:

        t = mean(IC) / std(IC) * sqrt(n) = ICIR * sqrt(n)

    ``ic_results`` stores only scalar moments (mean/std/ICIR), not the IC
    series or its length, so the very same Rank IC series is recomputed with
    :func:`quant_platform.factors.evaluation.rank_ic` to obtain ``n``. This
    is an approximation: the IC series is treated as iid (no autocorrelation
    adjustment) and the t distribution is approximated by the normal one
    (differences are negligible for the hundreds of daily ICs a real panel
    produces). It is recorded as such in the returned detail.

    Selection: among factors whose raw p-values survive Benjamini-Hochberg
    at ``significance_level``, take the top ``max_factors`` by ``|ICIR|``.
    If some survive but fewer than ``max_factors``, take all survivors. If
    none survive, fall back to the old top-``|ICIR|`` selection and set
    ``selection_warning`` — the fallback is explicit, never silent.

    Returns a detail dict with ``method``, ``p_values``, ``n_obs``,
    ``survivors``, ``selected``, ``fallback`` and ``selection_warning``.
    """
    from quant_platform.factors.evaluation import rank_ic
    from quant_platform.research.validation import FactorValidator

    validator = FactorValidator(significance_level=significance_level)
    names = list(ic_results)
    p_values: dict[str, float] = {}
    n_obs: dict[str, int] = {}

    for name in names:
        factor = processed_factors.get(name)
        if factor is None:
            p_values[name] = 1.0
            n_obs[name] = 0
            continue
        ic = rank_ic(factor, returns)
        ic = ic[np.isfinite(np.asarray(ic, dtype=float))]
        n = int(len(ic))
        n_obs[name] = n
        icir = _finite(ic_results[name].get("icir", float("nan")))
        if n < 2 or not math.isfinite(icir):
            p_values[name] = 1.0
            continue
        t_stat = icir * math.sqrt(n)
        # Two-sided normal tail (selection uses |ICIR|, so sign does not
        # matter). erfc(|t|/sqrt(2)) is P(|Z| >= |t|).
        p_values[name] = min(1.0, math.erfc(abs(t_stat) / math.sqrt(2.0)))

    if names:
        mask = validator.multiple_testing_correction(
            [p_values[n] for n in names], method="bh",
        )
        survivors = [n for n, ok in zip(names, mask, strict=True) if bool(ok)]
    else:
        survivors = []

    selected = sorted(
        survivors,
        key=lambda n: abs(_finite(ic_results[n].get("icir", 0.0))),
        reverse=True,
    )[:max_factors]

    fallback = False
    selection_warning = None
    if not selected and names:
        fallback = True
        selection_warning = (
            "no factor survives BH-FDR; falling back to top-|ICIR|"
        )
        # Fallback keeps the original top-|ICIR| rule, but a factor whose
        # ICIR is undefined (fewer than MIN_IC_OBSERVATIONS ICs, hence NaN)
        # cannot be ranked and must not win by sort accident. If no factor
        # has a finite ICIR at all, preserve the historical ordering so a
        # short-sample run still walks forward exactly as it used to.
        finite = [
            (name, info) for name, info in ic_results.items()
            if math.isfinite(_finite(info.get("icir", float("nan"))))
        ]
        if finite:
            ranked = sorted(
                finite, key=lambda item: abs(item[1]["icir"]), reverse=True,
            )
        else:
            ranked = sorted(
                ic_results.items(), key=lambda x: abs(x[1].get("icir", 0)),
                reverse=True,
            )
        selected = [name for name, _ in ranked[:max_factors]]

    return {
        "method": "full-sample ICIR two-sided t-test + BH-FDR",
        "significance_level": significance_level,
        "p_value_method": (
            "normal approximation to t = ICIR*sqrt(n_obs); n_obs is the "
            "length of the recomputed Rank IC series; IC autocorrelation "
            "is not modelled"
        ),
        "p_values": p_values,
        "n_obs": n_obs,
        "survivors": survivors,
        "selected": selected,
        "fallback": fallback,
        "selection_warning": selection_warning,
    }


def persist_validation_results(
    ic_results: dict,
    processed_factors: dict,
    wf_results: dict | None = None,
    *,
    store=None,
    data_source: str = "synthetic",
    adjust: str = "qfq",
    pit: bool | None = False,
    validity: str = "valid",
    warnings: list[str] | None = None,
) -> dict:
    """Persist factor evaluations and stability diagnostics honestly (R11).

    Each row carries the last *observed* date of its factor panel (so repeated
    runs form a real time series and different panels produce distinct dates),
    walk-forward statistics computed from the actual fold metrics, and the
    lineage columns added by R10 (data source, adjustment, PIT, validity,
    warnings). Returns a summary of what was written/skipped.
    """
    from quant_platform.factors.store import (
        FactorEvalRecord,
        FactorResearchStore,
        FactorStabilityRecord,
    )

    store = store if store is not None else FactorResearchStore()
    wf_results = wf_results or {}
    base_warnings = list(warnings or [])

    saved = 0
    skipped: list[str] = []
    stability_status: dict[str, str] = {}
    signal_dates: dict[str, str] = {}

    for fname, summary in ic_results.items():
        signal_date = signal_date_for_panel(processed_factors.get(fname))
        if signal_date is None:
            skipped.append(f"{fname}: no dated factor observations")
            continue

        signal_dates[fname] = signal_date
        run_id = f"research_validation_{signal_date}"
        stability = stability_from_fold_metrics(
            (wf_results.get(fname) or {}).get("fold_metrics")
        )
        stability_status[fname] = stability["status"]

        row_warnings = list(base_warnings)
        if stability["status"] != "computed":
            row_warnings.append(
                "stability_unavailable: no usable walk-forward folds"
            )
        elif stability["n_skipped_folds"]:
            row_warnings.append(
                f"stability_skipped_folds: {stability['n_skipped_folds']}"
            )

        store.save_evaluation(FactorEvalRecord(
            factor_name=fname,
            signal_date=signal_date,
            rank_ic=_finite(summary.get("mean_ic")),
            pearson_ic=_finite(summary.get("pearson_ic", 0)),
            icir=_finite(summary.get("icir")),
            coverage=_finite(summary.get("coverage", 0)),
            n_assets=int(summary.get("n_assets") or 0),
            run_id=run_id,
            data_source=data_source,
            adjust=adjust,
            pit=pit,
            validity=validity,
            warnings=row_warnings,
        ))
        store.save_stability(FactorStabilityRecord(
            factor_name=fname,
            ic_mean=_finite(summary.get("mean_ic")),
            ic_std=_finite(summary.get("std_ic", 0)),
            icir=_finite(summary.get("icir")),
            sharpe_mean=stability["sharpe_mean"],
            sharpe_std=stability["sharpe_std"],
            sharpe_consistency=stability["sharpe_consistency"],
            positive_folds=stability["positive_folds"],
            total_folds=stability["total_folds"],
            coverage=_finite(summary.get("coverage", 0)),
            run_id=run_id,
            data_source=data_source,
            adjust=adjust,
            pit=pit,
            validity=validity,
            warnings=row_warnings,
        ))
        saved += 1

    return {
        "records_saved": saved,
        "signal_dates": signal_dates,
        "stability_status": stability_status,
        "skipped": skipped,
    }


@dataclass
class ValidationStep:
    """Record of one validation step execution."""
    name: str
    category: str  # factor_eval, walk_forward, strategy_gate, etc.
    target: str
    status: str  # completed, skipped, failed, timeout
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "target": self.target,
            "status": self.status,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "warnings": self.warnings,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Complete research validation report."""
    mode: str
    start_time: str
    end_time: str = ""
    total_runtime: float = 0.0
    steps: list[ValidationStep] = field(default_factory=list)
    factor_rankings: list[dict] = field(default_factory=list)
    gate_result: dict | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    warnings_global: list[str] = field(default_factory=list)
    #: ``{"spec": {...}, "excluded_days": N}`` when a holdout was active for
    #: this run, else ``None`` (disabled runs keep their exact old shape).
    holdout: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        report = {
            "mode": self.mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_runtime": round(self.total_runtime, 3),
            "n_steps": len(self.steps),
            "n_warnings": len(self.warnings_global) + sum(
                len(s.warnings) for s in self.steps
            ),
            "steps": [s.to_dict() for s in self.steps],
            "factor_rankings": self.factor_rankings[:10],
            "gate_result": self.gate_result,
            "summary": self.summary,
            "warnings_global": self.warnings_global,
        }
        # Only present when a holdout was actually applied: a disabled run
        # must serialise exactly as it did before this feature existed.
        if self.holdout is not None:
            report["holdout"] = self.holdout
        return report

    def print_report(self) -> str:
        lines = []
        runtime_str = f"{self.total_runtime:.1f}s"
        lines.append(f"\n{'='*60}")
        lines.append(f"  Research Validation ({self.mode} mode)")
        lines.append(f"  Runtime: {runtime_str}")
        lines.append(f"  Steps:   {len(self.steps)}")
        if self.holdout is not None:
            spec = self.holdout.get("spec", {})
            lines.append(
                f"  Holdout: {spec.get('start_date')}..{spec.get('end_date')} "
                f"({self.holdout.get('excluded_days', 0)} trading days excluded "
                "from selection and walk-forward)"
            )
        lines.append(f"{'='*60}")

        for s in self.steps:
            icon = {"completed": "v", "skipped": "-", "failed": "x", "timeout": "!"}
            rt = f"({s.runtime_seconds:.1f}s)" if s.runtime_seconds > 0.1 else ""
            lines.append(f"  [{icon.get(s.status, '?')}] {s.name:<35} {s.status:<10} {rt}")
            for w in s.warnings[:3]:
                lines.append(f"       WARN: {w}")
            if s.error:
                lines.append(f"       ERR: {s.error}")

        if self.factor_rankings:
            lines.append(f"\n  --- Factor Rankings (top {min(5, len(self.factor_rankings))}) ---")
            for f in self.factor_rankings[:5]:
                lines.append(
                    f"    {f['factor_name']:<25} IC={f.get('mean_ic', 0):.4f}  "
                    f"ICIR={f.get('icir', 0):.3f}  Score={f.get('health_score', 0):.3f}"
                )

        if self.gate_result:
            lines.append(f"\n  --- Strategy Gates: {self.gate_result.get('overall_status', 'N/A')} ---")

        if self.warnings_global:
            lines.append("\n  Warnings:")
            for w in self.warnings_global:
                lines.append(f"    WARN: {w}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class ResearchValidator:
    """Orchestrate research validation across factors and strategies.

    Usage:
        validator = ResearchValidator()
        report = validator.run_quick()
        # or
        report = validator.run_full()
    """

    def __init__(
        self,
        n_stocks: int = 500,
        start_date: str = "2021-01-01",
        end_date: str | None = None,
        timeout_seconds: float = 600,
        factor_store_path: str | None = None,
        *,
        holdout: HoldoutSpec | None | object = _HOLDOUT_UNSET,
    ):
        self.n_stocks = n_stocks
        self.start_date = start_date
        # Resolved at run time (QUANT_RESEARCH_END_DATE or today): the old
        # hardcoded 2025-12-31 default silently described a stale window.
        self.end_date = resolve_end_date(end_date)
        self.timeout_seconds = timeout_seconds
        #: None keeps the historical default (data/factor_research.db). Tests
        #: and scratch runs must point this at a temporary database: a quick
        #: validation sprint is a synthetic smoke test, not research evidence.
        self.factor_store_path = factor_store_path
        #: Final-evaluation window. ``_HOLDOUT_UNSET`` (the default) means
        #: "resolve from the platform configuration" so the CLI research path
        #: is protected without extra wiring; an explicit ``None`` disables
        #: the holdout regardless of config; a spec applies it directly.
        self._holdout_explicit = holdout is not _HOLDOUT_UNSET
        self.holdout: HoldoutSpec | None = (
            None if holdout is _HOLDOUT_UNSET else holdout  # type: ignore[assignment]
        )

    def _resolved_holdout(self) -> HoldoutSpec | None:
        """The holdout in force for this run (see ``__init__``)."""
        if self._holdout_explicit:
            return self.holdout
        return load_configured_holdout()

    def run_quick(
        self,
        max_factors: int = 3,
        max_folds: int = 2,
        n_symbols: int = 100,
    ) -> ValidationReport:
        """Quick validation: limited scope, fast turnaround."""
        return self._run(
            mode="quick",
            max_factors=max_factors,
            max_folds=max_folds,
            n_symbols=n_symbols,
        )

    def run_full(
        self,
        max_folds: int = 5,
    ) -> ValidationReport:
        """Full validation: all factors, full universe, standard folds."""
        return self._run(
            mode="full",
            max_factors=999,
            max_folds=max_folds,
            n_symbols=self.n_stocks,
        )

    def _run(
        self,
        mode: str,
        max_factors: int,
        max_folds: int,
        n_symbols: int,
    ) -> ValidationReport:
        start_time = datetime.now().isoformat()
        t0 = time.time()

        report = ValidationReport(mode=mode, start_time=start_time)
        steps: list[ValidationStep] = []
        global_warnings: list[str] = []

        # Step 1: Load data
        step1 = self._step_load_data(n_symbols)
        steps.append(step1)
        if step1.status != "completed":
            report.steps = steps
            report.end_time = datetime.now().isoformat()
            report.total_runtime = time.time() - t0
            report.warnings_global = global_warnings
            return report

        prices = step1.details.get("prices")
        returns = step1.details.get("returns")
        benchmark = step1.details.get("benchmark")
        metadata = step1.details.get("metadata")
        financials = step1.details.get("financials")
        turnover = step1.details.get("turnover")
        config = step1.details.get("config")

        # ── Holdout discipline ───────────────────────────────────────────
        # A configured holdout is reserved for the final evaluation only.
        # Remove its dates from every research input *now*, before any factor
        # is computed: the factor panels, the IC moments that drive selection
        # and the walk-forward folds then cannot see holdout data at all.
        holdout_spec = self._resolved_holdout()
        holdout_info = None
        if holdout_spec is not None:
            masked, holdout_info = mask_research_inputs(
                holdout_spec,
                prices=prices, returns=returns, benchmark=benchmark,
                financials=financials, turnover=turnover,
            )
            prices = masked["prices"]
            returns = masked["returns"]
            benchmark = masked["benchmark"]
            financials = masked["financials"]
            turnover = masked["turnover"]
            # Keep the load step's payload holdout-free as well, so the report
            # itself cannot carry the reserved window into later analysis.
            step1.details.update(masked)
            step1.details["holdout_excluded_days"] = holdout_info["excluded_days"]
            if returns is not None:
                step1.details["n_dates"] = len(returns)
            step1.warnings.append(
                "holdout_excluded_days: "
                f"{holdout_info['excluded_days']} "
                f"({holdout_info['spec']['start_date']}.."
                f"{holdout_info['spec']['end_date']})"
            )
        report.holdout = holdout_info

        # Step 2: Register and get factors
        step2 = self._step_register_factors()
        steps.append(step2)
        factor_names = step2.details.get("factor_names", [])

        # Step 3: Compute factors (up to max_factors)
        step3 = self._step_compute_factors(
            prices, returns, financials, metadata, turnover,
            config, factor_names, max_factors,
        )
        steps.append(step3)
        if step3.status != "completed":
            report.steps = steps
            report.end_time = datetime.now().isoformat()
            report.total_runtime = time.time() - t0
            report.warnings_global = global_warnings
            return report

        processed_factors = step3.details.get("processed_factors", {})
        ic_results = step3.details.get("ic_results", {})
        sector_map = step3.details.get("sector_map")
        fin_unstacked = step3.details.get("fin_unstacked")

        # Defence in depth: factor panels were computed from masked prices
        # already, but strip any holdout rows again before selection so a
        # holdout date can never reach the screening step. The IC moments that
        # drive selection are then recomputed from the masked panels: the
        # scalars stored by step 3 must describe exactly what selection sees,
        # never a full-sample evaluation that might have touched the holdout.
        # Index order is preserved; a disabled holdout skips this block.
        if (
            holdout_spec is not None
            and holdout_info["excluded_days"] > 0
            and processed_factors
            and returns is not None
        ):
            processed_factors = {
                name: holdout_spec.mask_frame(panel)
                for name, panel in processed_factors.items()
            }
            step3.details["processed_factors"] = processed_factors
            ic_results, ic_warnings = _panel_ic_results(
                processed_factors, returns,
            )
            step3.details["ic_results"] = ic_results
            step3.warnings.extend(ic_warnings)
            step3.warnings.append(
                "ic_moments_recomputed_after_holdout_exclusion"
            )

        # Step 4: Run walk-forward on each factor (up to max_folds)
        step4 = self._step_walk_forward(
            processed_factors, ic_results, prices, returns,
            benchmark, sector_map, fin_unstacked,
            factor_names[:max_factors], max_folds,
            holdout_excluded_days=(
                holdout_info["excluded_days"] if holdout_info is not None else 0
            ),
        )
        steps.append(step4)

        # Step 5: Save to Factor Store
        step5 = self._step_save_to_store(
            ic_results, processed_factors,
            step4.details.get("wf_results", {}),
            data_source=step1.details.get("data_source", "synthetic"),
            adjust=step1.details.get("adjust", "qfq"),
            pit=step1.details.get("pit", False),
            warnings=step1.details.get("provenance_warnings", []),
        )
        steps.append(step5)

        # Step 6: Run Strategy Gates
        wf_result = step4.details.get("wf_result")
        step6 = self._step_run_gates(
            ic_results, wf_result, len(processed_factors), prices=prices,
        )
        steps.append(step6)
        report.gate_result = step6.details.get("gate_report")

        # Step 7: Factor ranking
        step7 = self._step_factor_ranking()
        steps.append(step7)
        report.factor_rankings = step7.details.get("rankings", [])

        # Compile summary. "Pass" has to describe the gate report and the
        # required steps, not merely the absence of a crashed step: a failed
        # IC-quality gate or a skipped walk-forward used to report "pass" and
        # exit 0.
        n_completed = sum(1 for s in steps if s.status == "completed")
        n_failed = sum(1 for s in steps if s.status == "failed")
        skipped_steps = [s.name for s in steps if s.status == "skipped"]
        n_warnings = sum(len(s.warnings) for s in steps) + len(global_warnings)
        gate_status = (report.gate_result or {}).get("overall_status", "")
        needs_review = (
            n_failed > 0
            or bool(skipped_steps)
            or gate_status in ("FAIL", "REJECTED")
        )
        report.summary = {
            "mode": mode,
            "steps_completed": n_completed,
            "steps_failed": n_failed,
            "steps_skipped": len(skipped_steps),
            "skipped_steps": skipped_steps,
            "gate_status": gate_status,
            "total_warnings": n_warnings,
            "n_factors_evaluated": len(ic_results),
            "overall": "needs_review" if needs_review else "pass",
        }
        if holdout_info is not None:
            # Mirrored into the summary so the report reader can tell this run
            # never looked at the holdout. Disabled runs keep the old shape.
            report.summary["holdout"] = holdout_info

        report.steps = steps
        report.end_time = datetime.now().isoformat()
        report.total_runtime = time.time() - t0
        report.warnings_global = global_warnings
        return report

    # ── Step implementations ──

    def _step_load_data(self, n_symbols: int) -> ValidationStep:
        """Load market data."""
        step = ValidationStep(
            name="load_data", category="data", target="market",
            status="running",
        )
        t0 = time.time()
        try:
            from types import SimpleNamespace

            from quant_platform.data.pipeline import DataPipeline
            from quant_platform.data.providers.synthetic import SyntheticDataProvider

            config = SimpleNamespace()
            config.data = SimpleNamespace()
            config.data.start_date = self.start_date
            config.data.end_date = self.end_date
            config.data.provider = "synthetic"
            config.universe = SimpleNamespace()
            config.universe.n_stocks = n_symbols
            config.universe.exclude_st = True
            config.universe.exclude_suspended = True
            config.data.synthetic = SimpleNamespace()
            config.data.synthetic.embedded_alpha = False

            provider = SyntheticDataProvider(
                n_stocks=n_symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                embedded_alpha=False,
            )
            pipeline = DataPipeline(
                provider=provider,
                start_date=config.data.start_date,
                end_date=config.data.end_date,
                exclude_st=config.universe.exclude_st,
                exclude_suspended=config.universe.exclude_suspended,
            )
            pipeline.run()

            prices = pipeline.get_close()
            returns = pipeline.returns
            bench = pipeline.benchmark
            meta = pipeline.metadata
            fin = pipeline.financials
            turn = pipeline.get_turnover()

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {
                "prices": prices, "returns": returns,
                "benchmark": bench, "metadata": meta,
                "financials": fin, "turnover": turn,
                "config": config,
                "n_assets": len(prices.columns), "n_dates": len(prices),
                # Provenance written into every factor-store row. Synthetic
                # data is not market data and the simulated universe is not
                # point-in-time, so downstream readers can never mistake a
                # validation-sprint row for a real-data result.
                "data_source": "synthetic",
                "adjust": "qfq",
                "pit": False,
                "provenance_warnings": ["synthetic_data_not_market_data"],
            }
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_register_factors(self) -> ValidationStep:
        """Register all factors in the registry."""
        step = ValidationStep(
            name="register_factors", category="factor", target="registry",
            status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.factors.fundamental import register_all as register_fundamental
            from quant_platform.factors.registry import get_registry
            from quant_platform.factors.technical import register_all as register_technical

            register_technical()
            register_fundamental()
            registry = get_registry()
            names = registry.list_all()

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {"factor_names": names, "n_factors": len(names)}
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_compute_factors(
        self, prices, returns, financials, metadata, turnover,
        config, factor_names, max_factors: int,
    ) -> ValidationStep:
        """Compute and evaluate factors."""
        step = ValidationStep(
            name="compute_factors", category="factor_eval", target="all",
            status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.factors.processing import process_factor
            from quant_platform.factors.registry import get_registry

            # Only compute top N factors
            names = factor_names[:max_factors]
            registry = get_registry()

            fin_unstacked = None
            if financials is not None:
                fin_unstacked = financials.unstack("asset")

            sector_map = metadata.get("sector", {}) if metadata is not None else {}
            mcap = fin_unstacked.get("market_cap") if fin_unstacked is not None else None

            raw_factors = {}
            for name in names:
                cls = registry.get(name)
                if cls is None:
                    continue
                try:
                    inst = cls()
                    kwargs = {}
                    if turnover is not None:
                        kwargs["turnover"] = turnover
                    if hasattr(inst, 'category') and inst.category.value == "fundamental" and fin_unstacked is not None:
                        result = inst.run(prices, fin_unstacked, **kwargs)
                    else:
                        result = inst.run(prices, **kwargs)
                    raw_factors[result.name] = result.values
                except Exception as e:
                    step.warnings.append(f"Failed to compute {name}: {e}")

            processed = {}
            for name, factor in raw_factors.items():
                processed[name] = process_factor(
                    factor, sector_map=sector_map, market_cap=mcap,
                )

            ic_results, ic_warnings = _panel_ic_results(processed, returns)
            step.warnings.extend(ic_warnings)

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {
                "processed_factors": processed,
                "ic_results": ic_results,
                "sector_map": sector_map,
                "fin_unstacked": fin_unstacked,
                "n_factors_computed": len(raw_factors),
                "n_factors_evaluated": len(ic_results),
            }
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_walk_forward(
        self, processed_factors, ic_results, prices, returns,
        benchmark, sector_map, fin_unstacked,
        factor_names, max_folds: int,
        *,
        holdout_excluded_days: int = 0,
    ) -> ValidationStep:
        """Run walk-forward validation on top factors."""
        step = ValidationStep(
            name="walk_forward", category="walk_forward", target="top_factors",
            status="running",
        )
        t0 = time.time()
        try:
            # If the holdout left fewer days than a single walk-forward window
            # needs, say so and skip: never fabricate OOS metrics, and never
            # report a "completed" walk-forward that measured nothing.
            required_days = _WF_TRAIN_PERIOD + _WF_TEST_PERIOD
            if (
                holdout_excluded_days > 0
                and returns is not None
                and len(returns) < required_days
            ):
                step.status = "skipped"
                step.warnings.append(
                    f"holdout excluded {holdout_excluded_days} trading days; "
                    f"only {len(returns)} days remain, fewer than the "
                    f"{required_days} required for walk-forward "
                    f"({_WF_TRAIN_PERIOD} train + {_WF_TEST_PERIOD} test). "
                    "No walk-forward was run."
                )
                step.details = {
                    "wf_result": None,
                    "wf_results": {},
                    "holdout_skip": {
                        "excluded_days": int(holdout_excluded_days),
                        "remaining_days": int(len(returns)),
                        "required_days": required_days,
                    },
                }
                step.runtime_seconds = time.time() - t0
                return step

            # Pick walk-forward factors by ICIR, but only among factors whose
            # full-sample IC series is significant after a t-test and BH-FDR
            # control. Selection details are kept for audit.
            factor_selection = select_walk_forward_factors(
                ic_results, processed_factors, returns,
            )
            step.details["factor_selection"] = factor_selection
            top_factors = factor_selection["selected"]
            if top_factors:
                step.warnings.append(
                    "walk-forward factors were selected by full-sample ICIR "
                    f"({', '.join(top_factors)}); their OOS metrics are "
                    "selection-biased"
                )
            if factor_selection["fallback"]:
                step.warnings.append(factor_selection["selection_warning"])

            if not top_factors:
                step.status = "skipped"
                step.runtime_seconds = time.time() - t0
                step.details = {
                    "wf_result": None,
                    "wf_results": {},
                    "factor_selection": factor_selection,
                }
                return step

            from quant_platform.backtest.walkforward import WalkForwardValidator

            wf_aggregate = None
            wf_results: dict[str, dict] = {}
            for fname in top_factors:
                factor_data = processed_factors.get(fname)
                if factor_data is None:
                    continue

                validator = WalkForwardValidator(
                    train_period=_WF_TRAIN_PERIOD,
                    test_period=_WF_TEST_PERIOD,
                    step_size=126,
                    mode="expanding",
                )
                wf_result = validator.run(
                    signal=factor_data,
                    prices=prices,
                    returns=returns,
                    benchmark_returns=benchmark,
                    sector_map=sector_map,
                    financials=fin_unstacked,
                    engine_kwargs={},
                    factors={fname: factor_data},
                    alpha_kwargs={"method": "equal_weight"},
                )
                if wf_aggregate is None:
                    wf_aggregate = wf_result
                # Keep the fold metrics (plain dicts) so the store writer can
                # compute real stability statistics; the old writer discarded
                # them and persisted zeros instead.
                wf_results[fname] = {
                    "fold_metrics": list(wf_result.get("fold_metrics", [])),
                    "stability": dict(wf_result.get("stability", {})),
                    "n_folds": len(wf_result.get("fold_metrics", [])),
                }
                step.details[f"wf_{fname}"] = {
                    "n_folds": len(wf_result.get("fold_metrics", [])),
                    "oos_sharpe": wf_result.get("aggregate_metrics", {}).get(
                        "sharpe_ratio", float("nan")
                    ),
                }

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details["wf_result"] = wf_aggregate
            step.details["wf_results"] = wf_results
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_save_to_store(
        self, ic_results: dict, processed_factors: dict,
        wf_results: dict | None = None,
        *, data_source: str = "synthetic", adjust: str = "qfq",
        pit: bool | None = False, warnings: list[str] | None = None,
    ) -> ValidationStep:
        """Save evaluation results to Factor Research Store."""
        step = ValidationStep(
            name="save_to_factor_store", category="storage",
            target="factor_store", status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.factors.store import FactorResearchStore

            store = (
                FactorResearchStore(self.factor_store_path)
                if self.factor_store_path else FactorResearchStore()
            )
            result = persist_validation_results(
                ic_results, processed_factors, wf_results,
                store=store, data_source=data_source, adjust=adjust,
                pit=pit, warnings=warnings,
            )

            if result["records_saved"]:
                step.status = "completed"
            else:
                step.status = "skipped"
                step.warnings.append(
                    "no factor evaluations could be dated; nothing persisted"
                )
            step.runtime_seconds = time.time() - t0
            step.details = result
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_run_gates(
        self, ic_results: dict, wf_result: dict | None, n_factors: int,
        prices=None,
    ) -> ValidationStep:
        """Run Strategy Gates on validation results."""
        step = ValidationStep(
            name="strategy_gates", category="gate",
            target="validation", status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.strategy.gates import (
                GateRunner,
                count_assets_with_data,
            )

            n_assets_total, n_assets_with_data = count_assets_with_data(prices)
            runner = GateRunner()
            report = runner.run(
                strategy_name="research_validation",
                ic_results=ic_results,
                wf_results=wf_result,
                n_factors=n_factors,
                n_params=n_factors,
                n_assets_total=n_assets_total,
                n_assets_with_data=n_assets_with_data,
            )

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {"gate_report": report.to_dict()}
            step.warnings = report.warnings
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_factor_ranking(self) -> ValidationStep:
        """Get factor rankings from the store."""
        step = ValidationStep(
            name="factor_ranking", category="analysis",
            target="factor_store", status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.factors.store import FactorResearchStore

            store = (
                FactorResearchStore(self.factor_store_path)
                if self.factor_store_path else FactorResearchStore()
            )
            rankings = store.get_factor_ranking()

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {"rankings": rankings}
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step
