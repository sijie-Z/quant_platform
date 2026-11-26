"""Production-grade HTML backtest report generator.

Generates a self-contained HTML report with:
- Performance summary (all metrics)
- Equity curve with benchmark
- Drawdown analysis
- Monthly returns heatmap
- Factor IC analysis
- Risk decomposition
- Walk-forward validation
- Monte Carlo confidence intervals
- Regime detection results
- Trade cost analysis
- Position history

The report is a single HTML file with embedded CSS/JS (ECharts CDN).

Every metric shown in the KPI strip comes from the same ``performance`` /
``risk`` mapping that is written to ``metrics.json`` -- the report never
recomputes a headline number. Windows are stated explicitly: the total-return
KPI is the full strategy history ("since inception"), while benchmark and
excess metrics are computed on the common strategy/benchmark window and
report their coverage.
"""

from __future__ import annotations

import html as html_lib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: Below this many observations ``backtest.metrics.all_metrics`` returns an
#: empty mapping. The report must say n/a instead of rendering the 0.0
#: defaults as if they had been measured.
MIN_OBSERVATIONS = 10


def _h(value) -> str:
    """Escape a data-derived string for HTML text/attribute context."""
    return html_lib.escape(str(value), quote=True)


def _json_for_script(value) -> str:
    """JSON that is safe inside a ``<script>`` block.

    Plain ``json.dumps`` leaves ``</script>`` intact, so a provider or run
    field containing ``</script><img src=x onerror=...>`` broke out of the
    block (and a value with ``</title>`` broke out of the title). Escaping
    ``<``/``>``/``&`` as unicode escapes keeps the payload inert while the
    decoded JSON value is unchanged.
    """
    return (
        json.dumps(value)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


def _finite(value) -> float | None:
    """A finite float, or None for missing/non-numeric/non-finite values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_pct(value) -> str:
    """Signed return formatter (gains carry an explicit +)."""
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:+.2f}%"


def _format_unsigned_pct(value) -> str:
    """Unsigned formatter for volatility / tracking error."""
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.2f}%"


def _format_loss_pct(value) -> str:
    """VaR/CVaR (positive = loss) without a signed-return '+' prefix.

    The raw value keeps its sign (a negative VaR means the quantile is a
    gain); the dashboard prints the same number, so both reports agree.
    """
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.2f}%"


def _format_ratio_pct(value, decimals: int = 1) -> str:
    """A [0, 1] proportion (win rate, IC+ ratio) as an unsigned percent."""
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.{decimals}f}%"


def _format_num(value, decimals: int = 2) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:,.{decimals}f}"


def _format_signed(value, decimals: int = 4) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:+.{decimals}f}"


def _format_int(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{int(number):,}"


def _css_sign(value) -> str:
    number = _finite(value)
    if number is None:
        return ""
    return "pos" if number >= 0 else "neg"


def _clean_series(values) -> list:
    """JSON-safe series: non-finite values become null (chart gaps)."""
    return [_finite(v) for v in (values or [])]


def _var_basis(risk: dict) -> str:
    """Label suffix from the configured confidence/horizon, never a constant."""
    confidence = _finite(risk.get("confidence"))
    horizon = _finite(risk.get("horizon_days"))
    parts = []
    if confidence is not None and 0 < confidence < 1:
        parts.append(f"{confidence * 100:g}%")
    if horizon is not None and horizon > 0:
        parts.append(f"{horizon:g}-day")
    return f" ({', '.join(parts)})" if parts else ""


def _kpi_card(metric: str, label: str, value: str, css: str = "") -> str:
    return (
        f'<div class="kpi" data-kpi="{metric}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {css}">{value}</div></div>'
    )


def _monthly_has_values(monthly) -> bool:
    if not isinstance(monthly, dict):
        return False
    for row in monthly.get("data") or []:
        if isinstance(row, (list, tuple)) and any(_finite(v) is not None for v in row):
            return True
    return False


def _resolve_rolling_axis(rolling: list, rolling_dates, dates: list) -> tuple[list, str]:
    """Return (x-axis labels, empty-state note) for the rolling-Sharpe chart.

    The API stores ``rolling_sharpe_dates`` next to ``rolling_sharpe``; using
    ``dates.slice(-n)`` mislabelled the early points whenever the two series
    were downsampled differently. When the producer supplies explicit dates
    they win. A suffix of ``dates`` is only used for the legacy CLI payload,
    where the rolling series is a plain suffix of the strategy dates (a
    rolling(252) on the same series drops exactly the leading 251 rows).
    """
    note = "Rolling Sharpe unavailable: fewer than 252 observations."
    if not rolling:
        return [], note
    supplied = [str(d) for d in (rolling_dates or [])]
    if supplied and len(supplied) == len(rolling):
        return supplied, ""
    if len(rolling) == len(dates) and dates:
        return list(dates), ""
    if 0 < len(rolling) <= len(dates):
        return list(dates[-len(rolling):]), ""
    return [], (
        f"Rolling Sharpe omitted: {len(rolling)} points cannot be date-aligned "
        f"to {len(dates)} strategy dates."
    )


def generate_html_report(
    results: dict,
    output_path: str = "results/backtest_report.html",
    title: str = "Quant Platform — Backtest Report",
) -> str:
    """Generate a comprehensive HTML backtest report.

    Args:
        results: dict with keys: performance, risk, factors, chart_data, etc.
        output_path: where to save the HTML file
        title: report title

    Returns:
        Path to the generated HTML file.
    """
    perf = results.get("performance") or {}
    risk = results.get("risk") or {}
    factors = results.get("factors") or []
    chart = results.get("chart_data") or {}
    exposure = results.get("exposure") or {}
    stress = results.get("stress_tests") or []
    provenance = results.get("provenance") or {}

    dates = [str(d) for d in (chart.get("dates") or [])]
    equity = _clean_series(chart.get("equity"))
    drawdown = _clean_series(chart.get("drawdown"))

    n_obs_value = _finite(perf.get("total_days"))
    if n_obs_value is None and dates:
        n_obs_value = float(len(dates))
    n_obs = int(n_obs_value) if n_obs_value is not None else None
    insufficient = n_obs is not None and n_obs < MIN_OBSERVATIONS
    no_data = n_obs is None and not perf and not dates
    # n/a, never 0.0, when the sample cannot support a metric.
    metric_src = {} if insufficient else perf
    risk_src = {} if insufficient else risk

    # --- Provenance / header -------------------------------------------------
    provider_label = provenance.get("effective_provider") or "N/A"
    if provenance.get("synthetic"):
        provider_label += " (SYNTHETIC)"
    requested_provider = provenance.get("requested_provider") or "N/A"
    data_range = ""
    if provenance.get("start_date") or provenance.get("end_date"):
        data_range = (
            f"{provenance.get('start_date', '?')} → "
            f"{provenance.get('end_date', '?')} "
            f"(data date: {provenance.get('data_date') or 'n/a'})"
        )

    optimizer = str(perf.get("optimizer") or "N/A")
    effective_optimizer = perf.get("effective_optimizer")
    optimizer_line = optimizer
    if effective_optimizer and str(effective_optimizer) != optimizer:
        optimizer_line += f" → {effective_optimizer}"
    n_fallbacks = _finite(perf.get("n_optimizer_fallbacks"))
    if n_fallbacks is not None:
        fallback_rate = _finite(perf.get("optimizer_fallback_rate"))
        optimizer_line += f" ({int(n_fallbacks)} fallbacks"
        if fallback_rate is not None:
            optimizer_line += f", {fallback_rate * 100:.1f}% of rebalances"
        optimizer_line += ")"

    # --- Return windows (R90) ------------------------------------------------
    full_total = _finite(metric_src.get("strategy_total_return_full_window"))
    common_total = _finite(metric_src.get("total_return"))
    coverage = _finite(metric_src.get("benchmark_coverage"))

    # Some producers (currently the API run record, api/schemas.py) drop the
    # full-window/coverage fields and keep only the common-window
    # ``total_return``. The equity curve's final value is the full-history
    # total; when it disagrees with ``total_return`` the KPI is a
    # benchmark-window number and must be labelled as such.
    last_equity = _finite(equity[-1]) if equity else None
    chart_total = None if last_equity is None else last_equity - 1.0
    total_matches_chart = (
        common_total is not None and chart_total is not None
        and abs(chart_total - common_total) <= 1e-6
    )

    window_parts = []
    if full_total is not None:
        window_parts.append("Total Return = since inception (full strategy history)")
    if coverage is not None and coverage < 0.9999:
        window_parts.append(
            f"benchmark-based metrics (excess/IR/TE) = common window "
            f"({coverage * 100:.1f}% coverage)"
        )
    elif (
        full_total is None and coverage is None and common_total is not None
        and chart_total is not None and not total_matches_chart
    ):
        window_parts.append(
            "Total Return = common strategy/benchmark window "
            "(coverage unavailable in this payload)"
        )
    window_note = "; ".join(window_parts)

    kpi_cards = []
    if full_total is not None:
        kpi_cards.append(_kpi_card(
            "total_return", "Total Return (since inception)",
            _format_pct(full_total), _css_sign(full_total),
        ))
    elif common_total is not None and (
        (coverage is not None and coverage < 0.9999)
        or (coverage is None and chart_total is not None and not total_matches_chart)
    ):
        kpi_cards.append(_kpi_card(
            "total_return", "Total Return (benchmark window)",
            _format_pct(common_total), _css_sign(common_total),
        ))
    else:
        kpi_cards.append(_kpi_card(
            "total_return", "Total Return",
            _format_pct(common_total), _css_sign(common_total),
        ))

    if (
        coverage is not None and coverage < 0.9999
        and full_total is not None and common_total is not None
    ):
        kpi_cards.append(_kpi_card(
            "benchmark_window_return", "Benchmark-Window Return",
            _format_pct(common_total), _css_sign(common_total),
        ))

    if coverage is not None and coverage < 0.9999:
        coverage_value = f"{coverage * 100:.1f}%"
        bench_days = _finite(metric_src.get("benchmark_days"))
        total_days = _finite(metric_src.get("total_days"))
        if bench_days is not None and total_days is not None and total_days > 0:
            coverage_value += f" ({int(bench_days)}/{int(total_days)} days)"
        kpi_cards.append(_kpi_card(
            "benchmark_coverage", "Benchmark Coverage", coverage_value,
        ))

    annual_return = _finite(metric_src.get("annual_return"))
    sharpe = _finite(metric_src.get("sharpe_ratio"))
    sortino = _finite(metric_src.get("sortino_ratio"))
    max_dd = _finite(metric_src.get("max_drawdown"))
    calmar = _finite(metric_src.get("calmar_ratio"))
    win_rate = _finite(metric_src.get("win_rate"))
    info_ratio = _finite(metric_src.get("information_ratio"))
    annual_vol = _finite(metric_src.get("annual_volatility"))

    kpi_cards.extend([
        _kpi_card("annual_return", "Annual Return",
                  _format_pct(annual_return), _css_sign(annual_return)),
        _kpi_card("sharpe_ratio", "Sharpe Ratio", _format_num(sharpe), "accent"),
        _kpi_card("sortino_ratio", "Sortino Ratio", _format_num(sortino), "accent"),
        _kpi_card("max_drawdown", "Max Drawdown", _format_pct(max_dd), "neg"),
        _kpi_card("calmar_ratio", "Calmar Ratio", _format_num(calmar)),
        _kpi_card("win_rate", "Win Rate", _format_ratio_pct(win_rate)),
        _kpi_card("information_ratio", "Information Ratio",
                  _format_num(info_ratio), "accent"),
        _kpi_card("annual_volatility", "Annual Vol", _format_unsigned_pct(annual_vol)),
        _kpi_card("n_rebalances", "Rebalances",
                  _format_int(metric_src.get("n_rebalances"))),
    ])

    # VaR/CVaR labels come from the configured confidence/horizon (R97), and
    # the loss magnitude is never rendered as a signed return (R101).
    var_basis = _var_basis(risk)
    kpi_cards.extend([
        _kpi_card("historical_var", f"Historical VaR{var_basis}",
                  _format_loss_pct(risk_src.get("historical_var")), "neg"),
        _kpi_card("historical_cvar", f"Historical CVaR{var_basis}",
                  _format_loss_pct(risk_src.get("historical_cvar")), "neg"),
    ])

    notice = ""
    if insufficient:
        notice = (
            f'<div class="notice" data-notice="insufficient-data">'
            f"Insufficient data: n={n_obs} observations "
            f"(minimum {MIN_OBSERVATIONS} required); performance and risk "
            f"metrics are shown as n/a.</div>"
        )
    elif no_data:
        notice = (
            '<div class="notice" data-notice="no-data">No data: no performance '
            "metrics or chart series were supplied.</div>"
        )

    # --- Benchmark alignment (R91) ------------------------------------------
    benchmark = _clean_series(chart.get("benchmark"))
    benchmark_dates = [str(d) for d in (chart.get("benchmark_dates") or [])]
    # Positional plotting is only safe when the benchmark provably shares the
    # strategy clock: either explicit dates, or a 100% overlap marker. Equal
    # lengths alone are not proof -- the API downsamples dates and benchmark
    # independently, so a gapped benchmark can coincidentally end up with the
    # same number of points and still be drawn one day off.
    coverage_marker = _finite(perf.get("benchmark_coverage"))
    benchmark_note = ""
    if benchmark:
        if not dates:
            benchmark_note = (
                "Benchmark series omitted: the report carries no strategy dates "
                "to align it to."
            )
            benchmark = []
        elif benchmark_dates and len(benchmark_dates) == len(benchmark):
            by_date = dict(zip(benchmark_dates, benchmark, strict=False))
            benchmark = [by_date.get(date) for date in dates]
        elif (
            len(benchmark) == len(dates)
            and coverage_marker is not None and coverage_marker >= 0.9999
        ):
            pass  # verified 100% overlap: same clock, same points
        else:
            raw_points = len(chart.get("benchmark") or [])
            if len(benchmark) != len(dates):
                benchmark_note = (
                    f"Benchmark series omitted: {raw_points} points cannot be "
                    f"date-aligned to {len(dates)} strategy dates."
                )
            else:
                benchmark_note = (
                    "Benchmark series omitted: date alignment cannot be "
                    "verified (benchmark coverage <100% or unavailable); "
                    "supply benchmark_dates to plot it."
                )
            benchmark = []
        if benchmark and all(value is None for value in benchmark):
            benchmark = []
            benchmark_note = (
                "Benchmark series omitted: no overlapping dates with the strategy."
            )
    has_benchmark = any(value is not None for value in benchmark)

    # --- Rolling Sharpe axis (R92) ------------------------------------------
    rolling_values = [
        value for value in _clean_series(chart.get("rolling_sharpe"))
        if value is not None
    ]
    rolling_axis, rolling_note = _resolve_rolling_axis(
        rolling_values,
        chart.get("rolling_sharpe_dates"),
        dates,
    )
    rolling_sharpe = rolling_values[-len(rolling_axis):] if rolling_axis else []

    monthly = chart.get("monthly_returns") or {}
    has_monthly = _monthly_has_values(monthly)

    # --- Factor table (R99: non-finite statistics render as n/a) ------------
    factor_rows = ""
    for f in factors:
        name = f.get("name", "")
        ic = _finite(f.get("mean_ic"))
        icir = _finite(f.get("icir"))
        ic_positive = _finite(f.get("ic_positive_ratio"))
        std_ic = _finite(f.get("std_ic"))
        if icir is None:
            color = "#6b7a8d"
        elif icir > 0.3:
            color = "#22c55e"
        elif icir > 0:
            color = "#fbbf24"
        else:
            color = "#ef4444"
        factor_rows += f"""
        <tr>
          <td style="font-weight:600;color:#4da6ff">{_h(name)}</td>
          <td style="font-family:monospace">{_format_signed(ic, 4)}</td>
          <td style="font-family:monospace;color:{color};font-weight:700">{_format_signed(icir, 3)}</td>
          <td style="font-family:monospace">{_format_ratio_pct(ic_positive)}</td>
          <td style="font-family:monospace">{_format_num(std_ic, 4)}</td>
        </tr>"""

    # --- Sector exposure rows ------------------------------------------------
    sector_rows = ""
    sectors = exposure.get("sectors") or {}
    for sector, weight in sorted(
        ((name, _finite(weight)) for name, weight in sectors.items()),
        key=lambda item: -(item[1] if item[1] is not None else float("-inf")),
    )[:10]:
        weight_value = weight or 0.0
        bar_width = min(weight_value * 300, 100)
        sector_rows += f"""
        <tr>
          <td>{_h(sector)}</td>
          <td style="font-family:monospace">{_format_unsigned_pct(weight)}</td>
          <td><div style="background:#4da6ff;height:6px;width:{bar_width}%;border-radius:3px"></div></td>
        </tr>"""

    # --- Stress test rows ----------------------------------------------------
    stress_rows = ""
    for s in stress:
        ret = _finite(s.get("cumulative_return"))
        dd = _finite(s.get("max_drawdown"))
        ret_color = "#ef4444" if ret is not None and ret < 0 else "#22c55e"
        stress_rows += f"""
        <tr>
          <td>{_h(s.get('scenario', ''))}</td>
          <td style="font-family:monospace;color:{ret_color}">{_format_pct(ret)}</td>
          <td style="font-family:monospace;color:#ef4444">{_format_pct(dd)}</td>
        </tr>"""

    dates_json = _json_for_script(dates)
    equity_json = _json_for_script(equity)
    benchmark_json = _json_for_script(benchmark)
    drawdown_json = _json_for_script(drawdown)
    sharpe_json = _json_for_script(rolling_sharpe)
    rolling_dates_json = _json_for_script(rolling_axis)
    monthly_json = _json_for_script(monthly)

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    period_start = _h(dates[0]) if dates else "n/a"
    period_end = _h(dates[-1]) if dates else "n/a"
    observations = "n/a" if n_obs is None else str(n_obs)

    benchmark_legend = ", 'Benchmark'" if has_benchmark else ""
    benchmark_series = ""
    if has_benchmark:
        benchmark_series = (
            ", { name: 'Benchmark', type: 'line', data: benchmark, smooth: 0.2, "
            "symbol: 'none', lineStyle: { color: '#6b7a8d', width: 1, type: 'dashed' } }"
        )

    if dates and equity:
        equity_js = """const eqChart = echarts.init(document.getElementById('equityChart'));
eqChart.setOption({
  tooltip: { trigger: 'axis' },
  legend: { data: ['Strategy'__LEGEND__], textStyle: { color: '#6b7a8d', fontSize: 10 }, top: 0, right: 0 },
  grid: { top: 30, right: 16, bottom: 28, left: 50 },
  xAxis: { type: 'category', data: dates, axisLabel: { color: '#6b7a8d', fontSize: 9 }, axisLine: { lineStyle: { color: '#1e2a3a' } } },
  yAxis: { type: 'value', axisLabel: { color: '#6b7a8d', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e2a3a' } } },
  series: [
    { name: 'Strategy', type: 'line', data: equity, smooth: 0.2, symbol: 'none', lineStyle: { color: '#4da6ff', width: 2 },
       areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(77,166,255,0.2)'},{offset:1,color:'rgba(77,166,255,0.01)'}]) } }__BENCHMARK_SERIES__
  ]
});"""
        equity_js = equity_js.replace("__LEGEND__", benchmark_legend)
        equity_js = equity_js.replace("__BENCHMARK_SERIES__", benchmark_series)
    else:
        equity_js = "const eqChart = null;"

    if dates and drawdown:
        dd_js = """const ddChart = echarts.init(document.getElementById('ddChart'));
ddChart.setOption({
  tooltip: { trigger: 'axis', formatter: p => p[0].name + '<br/>DD: ' + (p[0].value*100).toFixed(2) + '%' },
  grid: { top: 8, right: 16, bottom: 28, left: 50 },
  xAxis: { type: 'category', data: dates, axisLabel: { color: '#6b7a8d', fontSize: 9 }, axisLine: { lineStyle: { color: '#1e2a3a' } } },
  yAxis: { type: 'value', axisLabel: { color: '#6b7a8d', fontSize: 9, formatter: v => (v*100).toFixed(0)+'%' }, splitLine: { lineStyle: { color: '#1e2a3a' } } },
  series: [{ type: 'line', data: drawdown, smooth: 0.2, symbol: 'none', lineStyle: { color: '#ef4444', width: 1 },
    areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(239,68,68,0.3)'},{offset:1,color:'rgba(239,68,68,0.01)'}]) } }]
});"""
    else:
        dd_js = "const ddChart = null;"

    if rolling_sharpe and rolling_axis:
        sharpe_js = """const shChart = echarts.init(document.getElementById('sharpeChart'));
shChart.setOption({
  tooltip: { trigger: 'axis' },
  grid: { top: 8, right: 16, bottom: 24, left: 40 },
  xAxis: { type: 'category', data: rollingDates, axisLabel: { color: '#6b7a8d', fontSize: 8 }, axisLine: { lineStyle: { color: '#1e2a3a' } } },
  yAxis: { type: 'value', axisLabel: { color: '#6b7a8d', fontSize: 9 }, splitLine: { lineStyle: { color: '#1e2a3a' } } },
  series: [{ type: 'line', data: rollingSharpe, smooth: 0.2, symbol: 'none', lineStyle: { color: '#fbbf24', width: 1.5 },
    areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(251,191,36,0.15)'},{offset:1,color:'rgba(251,191,36,0.01)'}]) } }]
});"""
        sharpe_placeholder = ""
    else:
        sharpe_js = "const shChart = null;"
        sharpe_placeholder = f'<div class="empty-state">{_h(rolling_note)}</div>'

    if has_monthly:
        monthly_js = """const mChart = echarts.init(document.getElementById('monthlyChart'));
mChart.setOption({
  tooltip: { formatter: p => months[p.data[0]] + ' ' + years[p.data[1]] + ': ' + (p.data[2]*100).toFixed(2) + '%' },
  grid: { top: 8, right: 12, bottom: 24, left: 40 },
  xAxis: { type: 'category', data: months, axisLabel: { color: '#6b7a8d', fontSize: 9 }, axisLine: { lineStyle: { color: '#1e2a3a' } } },
  yAxis: { type: 'category', data: years, axisLabel: { color: '#6b7a8d', fontSize: 9 }, axisLine: { lineStyle: { color: '#1e2a3a' } } },
  visualMap: { min: -0.1, max: 0.1, calculable: false, orient: 'horizontal', left: 'center', bottom: 0, show: false,
    inRange: { color: ['#ef4444', '#111827', '#22c55e'] } },
  series: [{ type: 'heatmap', data: heatData, label: { show: true, fontSize: 8, color: '#c9d1d9',
    formatter: p => (p.data[2]*100).toFixed(1) + '%' } }]
});"""
        monthly_placeholder = ""
    else:
        monthly_js = "const mChart = null;"
        monthly_placeholder = '<div class="empty-state">No monthly returns available.</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<!-- Defense in depth: the report escapes all data-derived strings, but a CSP
     still blocks external exfiltration and plugin/object loads if an embedding
     site ever injects content. Inline scripts are required by the charts. -->
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'">
<title>{_h(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0e17; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #1e2a3a; }}
  .header h1 {{ font-size: 20px; color: #4da6ff; font-weight: 700; letter-spacing: 1px; }}
  .header .meta {{ font-size: 11px; color: #6b7a8d; text-align: right; line-height: 1.5; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 20px; }}
  .kpi {{ background: #111827; border: 1px solid #1e2a3a; border-radius: 8px; padding: 12px; text-align: center; }}
  .kpi-label {{ font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
  .kpi-value {{ font-size: 18px; font-weight: 700; font-family: 'SF Mono', 'Fira Code', monospace; color: #e6edf3; }}
  .kpi-value.pos {{ color: #22c55e; }}
  .kpi-value.neg {{ color: #ef4444; }}
  .kpi-value.accent {{ color: #4da6ff; }}
  .notice {{ background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.4); color: #fbbf24; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 12px; }}
  .empty-state {{ display: flex; align-items: center; justify-content: center; height: 100%; color: #6b7a8d; font-size: 11px; text-align: center; padding: 12px; }}
  .section {{ background: #111827; border: 1px solid #1e2a3a; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .section-title {{ font-size: 11px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
  .section-title::before {{ content: ''; width: 6px; height: 6px; border-radius: 50%; background: #4da6ff; }}
  .chart {{ width: 100%; height: 300px; }}
  .chart-sm {{ width: 100%; height: 200px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
  th {{ padding: 6px 10px; font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.6px; text-align: left; border-bottom: 1px solid #1e2a3a; background: #0d1117; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #161b22; color: #c9d1d9; }}
  tr:hover td {{ background: rgba(77,166,255,0.04); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .badge {{ font-size: 9px; font-weight: 700; padding: 2px 8px; border-radius: 3px; letter-spacing: 0.3px; }}
  .badge-green {{ color: #22c55e; background: rgba(34,197,94,0.1); }}
  .badge-yellow {{ color: #fbbf24; background: rgba(251,191,36,0.1); }}
  .badge-red {{ color: #ef4444; background: rgba(239,68,68,0.1); }}
</style>
</head>
<body>

<div class="header">
  <h1>{_h(title)}</h1>
  <div class="meta">
    Generated: {generated_at}<br>
    Period: {period_start} → {period_end}<br>
    Observations: {observations}<br>
    Optimizer: {_h(optimizer_line)}<br>
    Provider: {_h(provider_label)} (requested: {_h(requested_provider)})<br>
    Data: {_h(data_range or 'n/a')}{f'<br>Windows: {_h(window_note)}' if window_note else ''}</div>
</div>

{notice}

<!-- KPI Strip -->
<div class="kpi-grid">{''.join(kpi_cards)}</div>

<!-- Equity + Drawdown -->
<div class="two-col">
  <div class="section">
    <div class="section-title">Equity Curve</div>
    {f'<div class="notice">{_h(benchmark_note)}</div>' if benchmark_note else ''}
    <div id="equityChart" class="chart">{'' if dates and equity else '<div class="empty-state">No equity series available.</div>'}</div>
  </div>
  <div class="section">
    <div class="section-title">Drawdown</div>
    <div id="ddChart" class="chart">{'' if dates and drawdown else '<div class="empty-state">No drawdown series available.</div>'}</div>
  </div>
</div>

<!-- Rolling Sharpe + Monthly Returns -->
<div class="two-col">
  <div class="section">
    <div class="section-title">Rolling Sharpe (252d, rf=0)</div>
    <div id="sharpeChart" class="chart-sm">{sharpe_placeholder}</div>
  </div>
  <div class="section">
    <div class="section-title">Monthly Returns</div>
    <div id="monthlyChart" class="chart-sm">{monthly_placeholder}</div>
  </div>
</div>

<!-- Factor IC Analysis -->
<div class="section">
  <div class="section-title">Factor IC Analysis</div>
  <table>
    <thead><tr><th>Factor</th><th>Mean IC</th><th>ICIR</th><th>IC+ Ratio</th><th>IC Std</th></tr></thead>
    <tbody>{factor_rows}</tbody>
  </table>
</div>

<!-- Sector Exposure -->
<div class="two-col">
  <div class="section">
    <div class="section-title">Sector Exposure</div>
    <table>
      <thead><tr><th>Sector</th><th>Weight</th><th></th></tr></thead>
      <tbody>{sector_rows}</tbody>
    </table>
  </div>
  <div class="section">
    <div class="section-title">Stress Tests</div>
    <table>
      <thead><tr><th>Scenario</th><th>Return</th><th>Max DD</th></tr></thead>
      <tbody>{stress_rows}</tbody>
    </table>
  </div>
</div>

<!-- Risk Metrics -->
<div class="section">
  <div class="section-title">Risk Metrics</div>
  <div class="kpi-grid">
    <div class="kpi" data-kpi="risk_historical_var"><div class="kpi-label">Historical VaR{var_basis}</div><div class="kpi-value neg">{_format_loss_pct(risk_src.get('historical_var'))}</div></div>
    <div class="kpi" data-kpi="risk_parametric_var"><div class="kpi-label">Parametric VaR{var_basis}</div><div class="kpi-value neg">{_format_loss_pct(risk_src.get('parametric_var'))}</div></div>
    <div class="kpi" data-kpi="risk_cvar"><div class="kpi-label">CVaR{var_basis}</div><div class="kpi-value neg">{_format_loss_pct(risk_src.get('historical_cvar'))}</div></div>
    <div class="kpi" data-kpi="tracking_error"><div class="kpi-label">Tracking Error</div><div class="kpi-value">{_format_unsigned_pct(metric_src.get('tracking_error'))}</div></div>
    <div class="kpi" data-kpi="excess_return"><div class="kpi-label">Excess Return (benchmark window)</div><div class="kpi-value {_css_sign(metric_src.get('excess_return'))}">{_format_pct(metric_src.get('excess_return'))}</div></div>
    <div class="kpi" data-kpi="effective_n"><div class="kpi-label">Effective N</div><div class="kpi-value accent">{_format_num(exposure.get('effective_n'), 1)}</div></div>
  </div>
</div>

<script>
const dates = {dates_json};
const equity = {equity_json};
const benchmark = {benchmark_json};
const drawdown = {drawdown_json};
const rollingSharpe = {sharpe_json};
const rollingDates = {rolling_dates_json};
const monthlyData = {monthly_json};

// Equity Chart
{equity_js}

// Drawdown Chart
{dd_js}

// Rolling Sharpe (x-axis comes from rolling_sharpe_dates when supplied; the
// old dates.slice(-rollingSharpe.length) mislabelled downsampled series and
// threw a TypeError when rollingSharpe was null, blanking the monthly chart).
{sharpe_js}

// Monthly Heatmap
const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
// `monthly_returns` is {{years: [...], months: [...], data: [[...]]}}.
// Position values by their actual month, not by array index.
const monthly = monthlyData || {{}};
const years = Array.isArray(monthly.years) ? monthly.years.map(String) : [];
const monthNumbers = Array.isArray(monthly.months) && monthly.months.length
  ? monthly.months
  : [1,2,3,4,5,6,7,8,9,10,11,12];
const heatData = [];
years.forEach((y, yi) => {{
  const row = (monthly.data && monthly.data[yi]) || [];
  row.forEach((v, mi) => {{
    const month = Number(monthNumbers[mi]) || (mi + 1);
    if (v !== null && v !== undefined) heatData.push([month - 1, yi, v]);
  }});
}});
{monthly_js}

window.addEventListener('resize', () => {{ [eqChart, ddChart, shChart, mChart].forEach(c => c && c.resize()); }});
</script>

<div style="text-align:center;padding:24px;color:#3a4a5e;font-size:10px">
  Generated by A-Share Multi-Factor Quant Platform &bull; {generated_at}
</div>

</body>
</html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("HTML report generated: %s", output)
    return str(output)
