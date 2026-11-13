"""The shipped Grafana dashboard must only query metrics something emits.

S7: 11 of 16 panels queried metric names with zero producers, so the panels
were permanently empty; the API-request legend also used ``{{endpoint}}``
while the exporter labels the series ``path``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "monitoring" / "grafana_dashboard.json"

# PromQL functions/aggregators/keywords that are not metric names.
_PROMQL_WORDS = {
    "rate", "irate", "increase", "delta", "idelta", "deriv", "predict_linear",
    "histogram_quantile", "sum", "avg", "min", "max", "count", "stddev",
    "stdvar", "topk", "bottomk", "quantile", "abs", "ceil", "floor", "round",
    "clamp_max", "clamp_min", "vector", "time", "by", "without", "on",
    "ignoring", "group_left", "group_right", "offset", "bool", "and", "or",
    "unless", "le", "m", "s", "h", "d", "w", "y",
}

_IDENTIFIER = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")


def _load_dashboard() -> dict:
    with open(DASHBOARD, encoding="utf-8") as f:
        return json.load(f)


def _exprs() -> list[tuple[str, str]]:
    """(panel title, expr) for every target of every panel."""
    out = []
    for panel in _load_dashboard().get("panels", []):
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if expr:
                out.append((panel.get("title", "?"), expr))
    return out


def _metric_tokens(expr: str) -> set[str]:
    return {
        token for token in _IDENTIFIER.findall(expr)
        if token not in _PROMQL_WORDS
    }


def _production_source_text() -> str:
    """Concatenated non-test Python sources (metric producers live there)."""
    chunks = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {"tests", "__pycache__", ".venv", "node_modules", "dist"}:
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def _candidates(metric: str) -> set[str]:
    """A histogram family is exposed with _bucket/_sum/_count suffixes."""
    found = {metric}
    for suffix in ("_bucket", "_sum", "_count"):
        if metric.endswith(suffix):
            found.add(metric[: -len(suffix)])
    return found


def test_dashboard_has_panels_and_expressions():
    exprs = _exprs()
    assert len(exprs) >= 10


def test_every_panel_expression_has_a_producer():
    sources = _production_source_text()
    missing = []
    for title, expr in _exprs():
        for metric in _metric_tokens(expr):
            if not any(candidate in sources for candidate in _candidates(metric)):
                missing.append(f"{title}: {metric} ({expr})")
    assert not missing, (
        "Grafana panels query metrics no production code emits:\n"
        + "\n".join(sorted(missing))
    )


def test_api_request_legend_uses_the_path_label():
    """The HTTP middleware labels the series ``path``; ``endpoint`` was a lie."""
    text = DASHBOARD.read_text(encoding="utf-8")
    assert "{{endpoint}}" not in text
    assert "{{path}}" in text


@pytest.mark.parametrize("metric", [
    "data_provider_success_total",
    "data_provider_failures_total",
    "trading_active_positions",
    "trading_portfolio_pnl",
    "risk_level",
    "risk_kill_switch_active",
    "factor_ic_value",
    "ml_model_test_ic",
    "ml_model_test_icir",
    "eventbus_events_published_total",
    "eventbus_events_delivered_total",
    "eventbus_dead_letters_total",
    "store_orders_total",
    "store_trades_total",
])
def test_previously_unbacked_metric_is_emitted(metric):
    """Pin the 11 originally-missing metric names to real producers."""
    assert metric in _production_source_text()
