"""Strategy Evaluation Gates — deterministic offline quality control.

Each gate checks one dimension of strategy quality:
- IC/ICIR thresholds
- Walk-forward OOS stability
- Data coverage
- Drawdown/risk limits
- Complexity

Gates produce PASS / WARNING / FAIL / REJECTED statuses.
Thresholds are configurable but gates themselves are not bypassable.

Inspired by quawn's Strategy Evaluation Gates design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Status Constants ──

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"
REJECTED = "REJECTED"
SKIPPED = "SKIPPED"

FINAL_STATUS_ORDER = {PASS: 0, SKIPPED: 1, WARNING: 2, FAIL: 3, REJECTED: 4}

# ── Config ──


@dataclass
class GateConfig:
    """Configurable thresholds for strategy evaluation gates.

    These are research diagnostics, not return guarantees.
    """

    # IC quality
    minimum_ic: float = 0.02
    minimum_rank_ic: float = 0.02
    minimum_icir: float = 0.0
    minimum_factor_history_count: int = 3

    # Walk-forward
    minimum_walk_forward_folds: int = 3
    minimum_test_sharpe: float = 0.20
    maximum_train_test_gap: float = 0.50

    # Data quality
    minimum_price_coverage: float = 0.80
    minimum_fundamental_coverage: float = 0.30

    # Risk
    maximum_drawdown: float = 0.25
    kill_drawdown: float = 0.35
    maximum_turnover: float = 5.0
    maximum_cost_drag: float = 0.05

    # Complexity
    maximum_factor_count: int = 10
    maximum_parameter_count: int = 30

    # Regime
    minimum_regime_sample: int = 30

    # Schema
    reject_on_schema_error: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> GateConfig:
        if not data:
            return cls()
        allowed = cls.__dataclass_fields__
        kwargs = {}
        for key, value in data.items():
            if key in allowed:
                default = allowed[key].default
                if isinstance(default, bool):
                    kwargs[key] = bool(value)
                elif isinstance(default, int):
                    kwargs[key] = int(value)
                elif isinstance(default, float):
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "minimum_ic": self.minimum_ic,
            "minimum_rank_ic": self.minimum_rank_ic,
            "minimum_icir": self.minimum_icir,
            "minimum_factor_history_count": self.minimum_factor_history_count,
            "minimum_walk_forward_folds": self.minimum_walk_forward_folds,
            "minimum_test_sharpe": self.minimum_test_sharpe,
            "maximum_train_test_gap": self.maximum_train_test_gap,
            "minimum_price_coverage": self.minimum_price_coverage,
            "minimum_fundamental_coverage": self.minimum_fundamental_coverage,
            "maximum_drawdown": self.maximum_drawdown,
            "maximum_turnover": self.maximum_turnover,
            "maximum_cost_drag": self.maximum_cost_drag,
            "maximum_factor_count": self.maximum_factor_count,
            "maximum_parameter_count": self.maximum_parameter_count,
            "minimum_regime_sample": self.minimum_regime_sample,
            "reject_on_schema_error": self.reject_on_schema_error,
        }


# ── Gate Result ──


@dataclass
class GateResult:
    """Result from a single gate check."""
    gate_name: str
    status: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    current_value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        return {
            "gate_name": self.gate_name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "current_value": self.current_value,
            "threshold": self.threshold,
        }


@dataclass
class GateReport:
    """Complete gate evaluation report."""
    strategy_name: str
    overall_status: str
    gate_results: list[GateResult]
    config: GateConfig = field(default_factory=GateConfig)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "overall_status": self.overall_status,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "config": self.config.to_dict(),
            "warnings": self.warnings,
            "summary": self.summary,
        }

    def print_report(self) -> str:
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  Strategy Gates: {self.strategy_name}")
        lines.append(f"  Overall Status: {self.overall_status}")
        lines.append(f"{'='*60}")

        for gr in self.gate_results:
            icon = {"PASS": "+", "WARNING": "!", "FAIL": "x", "REJECTED": "XX", "SKIPPED": "-"}
            lines.append(f"  [{icon.get(gr.status, '?')}] {gr.gate_name:<30} {gr.status:<10} {gr.message}")

        if self.warnings:
            lines.append(f"\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    WARN: {w}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# ── Individual Gate Checks ──


def _overall_status(results: list[GateResult]) -> str:
    """Determine overall status from all gate results."""
    has_rejected = any(r.status == REJECTED for r in results)
    has_fail = any(r.status == FAIL for r in results)
    has_warning = any(r.status == WARNING for r in results)

    if has_rejected:
        return REJECTED
    if has_fail:
        return FAIL
    if has_warning:
        return WARNING
    return PASS


def check_ic_quality(
    ic_results: dict[str, dict],
    config: GateConfig,
) -> GateResult:
    """Check factor IC/ICIR against minimum thresholds."""
    if not ic_results:
        return GateResult(
            gate_name="ic_quality",
            status=SKIPPED,
            message="No IC results available",
        )

    n_factors = len(ic_results)
    n_above_ic = sum(1 for s in ic_results.values() if abs(s.get("mean_ic", 0)) >= config.minimum_ic)
    n_above_icir = sum(1 for s in ic_results.values() if abs(s.get("icir", 0)) >= config.minimum_icir)

    details = {
        "n_factors": n_factors,
        "n_above_minimum_ic": n_above_ic,
        "n_above_minimum_icir": n_above_icir,
    }

    best_factor = max(ic_results, key=lambda k: abs(ic_results[k].get("icir", 0)))
    best_icir = ic_results[best_factor].get("icir", 0)

    if n_above_icir < 1:
        return GateResult(
            gate_name="ic_quality",
            status=FAIL,
            message=f"No factor meets minimum ICIR={config.minimum_icir}",
            details=details,
            current_value=n_above_icir,
            threshold=1,
        )

    if n_above_ic < config.minimum_factor_history_count:
        return GateResult(
            gate_name="ic_quality",
            status=WARNING,
            message=f"Only {n_above_ic}/{n_factors} factors meet minimum IC={config.minimum_ic}",
            details=details,
            current_value=n_above_ic,
            threshold=config.minimum_factor_history_count,
        )

    return GateResult(
        gate_name="ic_quality",
        status=PASS,
        message=f"{n_above_icir}/{n_factors} factors pass ICIR gate. Best: {best_factor} ({best_icir:.4f})",
        details=details,
    )


def check_walk_forward(
    wf_results: dict | None,
    config: GateConfig,
) -> GateResult:
    """Check walk-forward OOS stability."""
    if not wf_results:
        return GateResult(
            gate_name="walk_forward",
            status=SKIPPED,
            message="No walk-forward data available",
        )

    n_folds = len(wf_results.get("fold_metrics", []))
    oos_sharpe = wf_results.get("aggregate_metrics", {}).get("sharpe_ratio", 0)

    details = {
        "n_folds": n_folds,
        "oos_sharpe": round(oos_sharpe, 4),
    }

    if n_folds < config.minimum_walk_forward_folds:
        return GateResult(
            gate_name="walk_forward",
            status=WARNING,
            message=f"Only {n_folds} folds (min {config.minimum_walk_forward_folds})",
            details=details,
            current_value=n_folds,
            threshold=config.minimum_walk_forward_folds,
        )

    if oos_sharpe < config.minimum_test_sharpe:
        return GateResult(
            gate_name="walk_forward",
            status=WARNING,
            message=f"OOS Sharpe {oos_sharpe:.3f} below threshold {config.minimum_test_sharpe}",
            details=details,
            current_value=oos_sharpe,
            threshold=config.minimum_test_sharpe,
        )

    return GateResult(
        gate_name="walk_forward",
        status=PASS,
        message=f"{n_folds} folds, OOS Sharpe={oos_sharpe:.3f}",
        details=details,
    )


def check_drawdown_risk(
    backtest_summary: dict | None,
    config: GateConfig,
) -> GateResult:
    """Check maximum drawdown against limits."""
    if not backtest_summary:
        return GateResult(
            gate_name="drawdown_risk",
            status=SKIPPED,
            message="No backtest results available",
        )

    max_dd = abs(backtest_summary.get("max_drawdown", 0))
    sharpe = backtest_summary.get("sharpe_ratio", 0)

    details = {
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
    }

    if max_dd > config.kill_drawdown:
        return GateResult(
            gate_name="drawdown_risk",
            status=REJECTED,
            message=f"Drawdown {max_dd:.1%} exceeds kill threshold {config.kill_drawdown:.0%}",
            details=details,
            current_value=max_dd,
            threshold=config.kill_drawdown,
        )

    if max_dd > config.maximum_drawdown:
        return GateResult(
            gate_name="drawdown_risk",
            status=FAIL,
            message=f"Drawdown {max_dd:.1%} exceeds limit {config.maximum_drawdown:.0%}",
            details=details,
            current_value=max_dd,
            threshold=config.maximum_drawdown,
        )

    if sharpe < 0:
        return GateResult(
            gate_name="drawdown_risk",
            status=WARNING,
            message=f"Negative Sharpe ({sharpe:.2f}) with drawdown {max_dd:.1%}",
            details=details,
        )

    return GateResult(
        gate_name="drawdown_risk",
        status=PASS,
        message=f"Max DD {max_dd:.1%}, Sharpe {sharpe:.2f}",
        details=details,
    )


def check_complexity(
    n_factors: int,
    n_params: int,
    config: GateConfig,
) -> GateResult:
    """Check strategy complexity (factor count, parameter count)."""
    details = {
        "n_factors": n_factors,
        "n_params": n_params,
    }

    if n_factors > config.maximum_factor_count:
        return GateResult(
            gate_name="complexity",
            status=WARNING,
            message=f"{n_factors} factors exceeds max {config.maximum_factor_count}",
            details=details,
            current_value=n_factors,
            threshold=config.maximum_factor_count,
        )

    if n_params > config.maximum_parameter_count:
        return GateResult(
            gate_name="complexity",
            status=WARNING,
            message=f"{n_params} total params exceeds max {config.maximum_parameter_count}",
            details=details,
            current_value=n_params,
            threshold=config.maximum_parameter_count,
        )

    return GateResult(
        gate_name="complexity",
        status=PASS,
        message=f"{n_factors} factors, {n_params} params",
        details=details,
    )


def check_data_coverage(
    n_assets_total: int,
    n_assets_with_data: int,
    config: GateConfig,
) -> GateResult:
    """Check data coverage ratio."""
    if n_assets_total == 0:
        return GateResult(
            gate_name="data_coverage",
            status=SKIPPED,
            message="No universe defined",
        )

    coverage = n_assets_with_data / max(n_assets_total, 1)
    details = {
        "total_assets": n_assets_total,
        "with_data": n_assets_with_data,
        "coverage": round(coverage, 4),
    }

    if coverage < config.minimum_price_coverage:
        return GateResult(
            gate_name="data_coverage",
            status=FAIL,
            message=f"Coverage {coverage:.1%} below threshold {config.minimum_price_coverage:.0%}",
            details=details,
            current_value=coverage,
            threshold=config.minimum_price_coverage,
        )

    return GateResult(
        gate_name="data_coverage",
        status=PASS,
        message=f"Coverage {coverage:.1%} ({n_assets_with_data}/{n_assets_total})",
        details=details,
    )


# ── Gate Runner ──


class GateRunner:
    """Orchestrate strategy evaluation gates."""

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()

    def run(
        self,
        strategy_name: str = "default",
        ic_results: dict[str, dict] | None = None,
        wf_results: dict | None = None,
        backtest_summary: dict | None = None,
        n_factors: int = 0,
        n_params: int = 0,
        n_assets_total: int = 0,
        n_assets_with_data: int = 0,
    ) -> GateReport:
        """Run all applicable gates."""
        results: list[GateResult] = []

        results.append(check_ic_quality(ic_results or {}, self.config))
        results.append(check_walk_forward(wf_results, self.config))
        results.append(check_drawdown_risk(backtest_summary, self.config))
        results.append(check_complexity(n_factors, n_params, self.config))
        results.append(check_data_coverage(n_assets_total, n_assets_with_data, self.config))

        overall = _overall_status(results)

        warnings = []
        for r in results:
            if r.status == WARNING:
                warnings.append(f"[{r.gate_name}] {r.message}")

        return GateReport(
            strategy_name=strategy_name,
            overall_status=overall,
            gate_results=results,
            config=self.config,
            warnings=warnings,
            summary={
                "total_gates": len(results),
                "pass": sum(1 for r in results if r.status == PASS),
                "warning": sum(1 for r in results if r.status == WARNING),
                "fail": sum(1 for r in results if r.status == FAIL),
                "rejected": sum(1 for r in results if r.status == REJECTED),
                "skipped": sum(1 for r in results if r.status == SKIPPED),
            },
        )

    def run_from_backtest_results(
        self,
        strategy_name: str,
        ic_results: dict[str, dict] | None = None,
        backtest_result: dict | None = None,
        wf_result: dict | None = None,
        n_factors: int = 0,
        n_params: int = 0,
    ) -> GateReport:
        """Run gates using results from a full pipeline run."""
        summary = None
        if backtest_result:
            summary = backtest_result.get("summary")

        n_assets_total = 0
        n_assets_with_data = 0
        if backtest_result and "weights_history" in backtest_result:
            wh = backtest_result["weights_history"]
            if wh:
                first_weights = next(iter(wh.values()))
                n_assets_with_data = len(first_weights)
                n_assets_total = n_assets_with_data

        return self.run(
            strategy_name=strategy_name,
            ic_results=ic_results,
            wf_results=wf_result,
            backtest_summary=summary,
            n_factors=n_factors,
            n_params=n_params,
            n_assets_total=n_assets_total,
            n_assets_with_data=n_assets_with_data,
        )
