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

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


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

    def to_dict(self) -> dict:
        return {
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

    def print_report(self) -> str:
        lines = []
        runtime_str = f"{self.total_runtime:.1f}s"
        lines.append(f"\n{'='*60}")
        lines.append(f"  Research Validation ({self.mode} mode)")
        lines.append(f"  Runtime: {runtime_str}")
        lines.append(f"  Steps:   {len(self.steps)}")
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
            lines.append(f"\n  Warnings:")
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
        end_date: str = "2025-12-31",
        timeout_seconds: float = 600,
    ):
        self.n_stocks = n_stocks
        self.start_date = start_date
        self.end_date = end_date
        self.timeout_seconds = timeout_seconds

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

        # Step 4: Run walk-forward on each factor (up to max_folds)
        step4 = self._step_walk_forward(
            processed_factors, ic_results, prices, returns,
            benchmark, sector_map, fin_unstacked,
            factor_names[:max_factors], max_folds,
        )
        steps.append(step4)

        # Step 5: Save to Factor Store
        step5 = self._step_save_to_store(ic_results, processed_factors)
        steps.append(step5)

        # Step 6: Run Strategy Gates
        wf_result = step4.details.get("wf_result")
        step6 = self._step_run_gates(
            ic_results, wf_result, len(processed_factors),
        )
        steps.append(step6)
        report.gate_result = step6.details.get("gate_report")

        # Step 7: Factor ranking
        step7 = self._step_factor_ranking()
        steps.append(step7)
        report.factor_rankings = step7.details.get("rankings", [])

        # Compile summary
        n_completed = sum(1 for s in steps if s.status == "completed")
        n_failed = sum(1 for s in steps if s.status == "failed")
        n_warnings = sum(len(s.warnings) for s in steps) + len(global_warnings)
        report.summary = {
            "mode": mode,
            "steps_completed": n_completed,
            "steps_failed": n_failed,
            "total_warnings": n_warnings,
            "n_factors_evaluated": len(ic_results),
            "overall": "pass" if n_failed == 0 else "needs_review",
        }

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
            from quant_platform.data.pipeline import DataPipeline
            from quant_platform.data.providers.synthetic import SyntheticDataProvider
            from types import SimpleNamespace

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
            from quant_platform.factors.technical import register_all as register_technical
            from quant_platform.factors.fundamental import register_all as register_fundamental
            from quant_platform.factors.registry import get_registry

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
            from quant_platform.factors.evaluation import rank_ic, ic_summary
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

            ic_results = {}
            for name, factor in processed.items():
                try:
                    ic = rank_ic(factor, returns)
                    summary = ic_summary(ic)
                    ic_results[name] = {
                        "mean_ic": summary["mean_ic"],
                        "icir": summary["icir"],
                        "pearson_ic": summary.get("pearson_ic", 0),
                        "coverage": summary.get("coverage", 0),
                    }
                except Exception as e:
                    step.warnings.append(f"IC evaluation failed for {name}: {e}")

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
    ) -> ValidationStep:
        """Run walk-forward validation on top factors."""
        step = ValidationStep(
            name="walk_forward", category="walk_forward", target="top_factors",
            status="running",
        )
        t0 = time.time()
        try:
            # Pick top 2 factors by ICIR for walk-forward
            sorted_factors = sorted(
                ic_results.items(), key=lambda x: abs(x[1].get("icir", 0)), reverse=True
            )
            top_factors = [f[0] for f in sorted_factors[:min(2, len(sorted_factors))]]

            if not top_factors:
                step.status = "skipped"
                step.runtime_seconds = time.time() - t0
                step.details = {"wf_result": None}
                return step

            from quant_platform.backtest.walkforward import WalkForwardValidator

            wf_aggregate = None
            for fname in top_factors:
                factor_data = processed_factors.get(fname)
                if factor_data is None:
                    continue

                validator = WalkForwardValidator(
                    train_period=504,
                    test_period=126,
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
                step.details[f"wf_{fname}"] = {
                    "n_folds": len(wf_result.get("fold_metrics", [])),
                    "oos_sharpe": wf_result.get("aggregate_metrics", {}).get("sharpe_ratio", 0),
                }

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details["wf_result"] = wf_aggregate
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_save_to_store(
        self, ic_results: dict, processed_factors: dict,
    ) -> ValidationStep:
        """Save evaluation results to Factor Research Store."""
        step = ValidationStep(
            name="save_to_factor_store", category="storage",
            target="factor_store", status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.factors.store import FactorResearchStore, FactorEvalRecord

            store = FactorResearchStore()
            saved = 0
            for fname, summary in ic_results.items():
                store.save_evaluation(FactorEvalRecord(
                    factor_name=fname,
                    signal_date=self.end_date,
                    rank_ic=summary.get("mean_ic", 0),
                    pearson_ic=summary.get("pearson_ic", 0),
                    icir=summary.get("icir", 0),
                    coverage=summary.get("coverage", 0),
                    n_assets=0,
                    run_id=f"research_validation_{self.end_date}",
                ))
                saved += 1

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {"records_saved": saved}
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step

    def _step_run_gates(
        self, ic_results: dict, wf_result: dict | None, n_factors: int,
    ) -> ValidationStep:
        """Run Strategy Gates on validation results."""
        step = ValidationStep(
            name="strategy_gates", category="gate",
            target="validation", status="running",
        )
        t0 = time.time()
        try:
            from quant_platform.strategy.gates import GateRunner, GateConfig

            runner = GateRunner()
            report = runner.run(
                strategy_name="research_validation",
                ic_results=ic_results,
                wf_results=wf_result,
                n_factors=n_factors,
                n_params=n_factors,
                n_assets_total=100,
                n_assets_with_data=100,
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

            store = FactorResearchStore()
            rankings = store.get_factor_ranking()

            step.status = "completed"
            step.runtime_seconds = time.time() - t0
            step.details = {"rankings": rankings}
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.runtime_seconds = time.time() - t0
        return step
