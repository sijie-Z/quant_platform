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

import math
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

    # IC quality. `ic_results` values come from `factors.evaluation.ic_summary`
    # on a rank-IC series, so `mean_ic` is the cross-sectional rank IC.
    minimum_ic: float = 0.02
    minimum_icir: float = 0.2
    minimum_factors_passing_ic: int = 3

    # Walk-forward
    minimum_walk_forward_folds: int = 3
    minimum_test_sharpe: float = 0.20

    # Data quality
    minimum_price_coverage: float = 0.80

    # Risk
    maximum_drawdown: float = 0.25
    kill_drawdown: float = 0.35
    maximum_turnover: float = 5.0
    maximum_cost_drag: float = 0.05

    # Complexity
    maximum_factor_count: int = 10
    maximum_parameter_count: int = 30

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
            "minimum_icir": self.minimum_icir,
            "minimum_factors_passing_ic": self.minimum_factors_passing_ic,
            "minimum_walk_forward_folds": self.minimum_walk_forward_folds,
            "minimum_test_sharpe": self.minimum_test_sharpe,
            "minimum_price_coverage": self.minimum_price_coverage,
            "maximum_drawdown": self.maximum_drawdown,
            "kill_drawdown": self.kill_drawdown,
            "maximum_turnover": self.maximum_turnover,
            "maximum_cost_drag": self.maximum_cost_drag,
            "maximum_factor_count": self.maximum_factor_count,
            "maximum_parameter_count": self.maximum_parameter_count,
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
            lines.append("\n  Warnings:")
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


def _metric(stats: dict, key: str) -> float:
    """Read a metric defensively: JSON round-trips can turn NaN into None."""
    try:
        value = float(stats.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def check_ic_quality(
    ic_results: dict[str, dict],
    config: GateConfig,
) -> GateResult:
    """Check factor IC/ICIR against minimum thresholds.

    A threshold of 0 disables that criterion. The old default for
    `minimum_icir` was 0, so `abs(icir) >= 0` was true for every factor and
    the gate could never fail; it also reported factors as "passing the ICIR
    gate" when no ICIR requirement was in force.

    IC and ICIR are compared by absolute value: a factor with a stable
    negative IC is usable through a negative weight in the alpha combination.
    """
    if not ic_results:
        return GateResult(
            gate_name="ic_quality",
            status=SKIPPED,
            message="No IC results available",
        )

    n_factors = len(ic_results)
    icir_enabled = config.minimum_icir > 0

    n_above_ic = sum(
        1 for s in ic_results.values()
        if abs(_metric(s, "mean_ic")) >= config.minimum_ic
    )
    n_above_icir = (
        sum(
            1 for s in ic_results.values()
            if abs(_metric(s, "icir")) >= config.minimum_icir
        )
        if icir_enabled else 0
    )

    details = {
        "n_factors": n_factors,
        "n_above_minimum_ic": n_above_ic,
        "n_above_minimum_icir": n_above_icir,
        "icir_gate_enabled": icir_enabled,
    }

    best_factor = max(ic_results, key=lambda k: abs(_metric(ic_results[k], "icir")))
    best_icir = _metric(ic_results[best_factor], "icir")

    if icir_enabled and n_above_icir < 1:
        return GateResult(
            gate_name="ic_quality",
            status=FAIL,
            message=f"No factor meets minimum ICIR={config.minimum_icir}",
            details=details,
            current_value=n_above_icir,
            threshold=1,
        )

    if n_above_ic < config.minimum_factors_passing_ic:
        return GateResult(
            gate_name="ic_quality",
            status=WARNING,
            message=(
                f"Only {n_above_ic}/{n_factors} factors meet minimum "
                f"IC={config.minimum_ic}"
            ),
            details=details,
            current_value=n_above_ic,
            threshold=config.minimum_factors_passing_ic,
        )

    icir_note = (
        f"{n_above_icir}/{n_factors} factors pass ICIR gate. "
        if icir_enabled
        else "ICIR gate disabled (threshold 0). "
    )
    return GateResult(
        gate_name="ic_quality",
        status=PASS,
        message=f"{icir_note}Best: {best_factor} ({best_icir:.4f})",
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


def count_assets_with_data(prices: Any, min_coverage: float = 0.5) -> tuple[int, int]:
    """Return (universe size, number of assets with enough price history).

    Both gate callers used to pass the same number for both arguments, which
    made the coverage gate compare the universe with itself and always pass.
    An asset counts as having data only when at least `min_coverage` of its
    rows are non-null.
    """
    if prices is None or getattr(prices, "empty", True):
        return 0, 0
    coverage = prices.notna().mean()
    return int(len(coverage)), int((coverage >= min_coverage).sum())


def check_trading_costs(
    backtest_summary: dict | None,
    config: GateConfig,
) -> GateResult:
    """Check annual turnover and cost drag reported by the backtest.

    Turnover is one-sided, the same convention as `PortfolioConstraints`.
    Both limits were declared in `GateConfig` but nothing read them.
    """
    if not backtest_summary:
        return GateResult(
            gate_name="trading_costs",
            status=SKIPPED,
            message="No backtest results available",
        )

    def _finite(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    turnover = _finite(backtest_summary.get("annual_turnover"))
    cost_drag = _finite(backtest_summary.get("annual_cost_drag"))
    if turnover is None and cost_drag is None:
        return GateResult(
            gate_name="trading_costs",
            status=SKIPPED,
            message="Backtest summary does not report turnover or cost drag",
        )

    details = {
        "annual_turnover": round(float(turnover), 4) if turnover is not None else None,
        "annual_cost_drag": round(float(cost_drag), 4) if cost_drag is not None else None,
    }

    if cost_drag is not None and cost_drag > config.maximum_cost_drag:
        return GateResult(
            gate_name="trading_costs",
            status=FAIL,
            message=(
                f"Annual cost drag {cost_drag:.2%} exceeds limit "
                f"{config.maximum_cost_drag:.2%}"
            ),
            details=details,
            current_value=cost_drag,
            threshold=config.maximum_cost_drag,
        )

    if turnover is not None and turnover > config.maximum_turnover:
        return GateResult(
            gate_name="trading_costs",
            status=WARNING,
            message=(
                f"Annual turnover {turnover:.2f}x exceeds guideline "
                f"{config.maximum_turnover:.2f}x"
            ),
            details=details,
            current_value=turnover,
            threshold=config.maximum_turnover,
        )

    return GateResult(
        gate_name="trading_costs",
        status=PASS,
        message=(
            f"Annual turnover {turnover:.2f}x, cost drag {cost_drag:.2%}"
            if turnover is not None and cost_drag is not None
            else "Trading costs within limits"
        ),
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
        results.append(check_trading_costs(backtest_summary, self.config))
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
        n_assets_total: int = 0,
        n_assets_with_data: int = 0,
    ) -> GateReport:
        """Run gates using results from a full pipeline run.

        Coverage must be supplied by the caller when known. It used to be
        derived as `len(first_weights)` for both sides, which made the gate
        compare a number with itself and always pass; when the universe size
        is unknown the gate now reports SKIPPED instead.
        """
        summary = None
        if backtest_result:
            summary = backtest_result.get("summary")

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
