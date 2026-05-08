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
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def _format_pct(v: float | None) -> str:
    if v is None: return "--"
    return f"{v * 100:+.2f}%"


def _format_num(v: float | None, decimals: int = 2) -> str:
    if v is None: return "--"
    return f"{v:,.{decimals}f}"


def _format_bps(v: float | None) -> str:
    if v is None: return "--"
    return f"{v:.1f} bps"


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
    perf = results.get("performance", {})
    risk = results.get("risk", {})
    factors = results.get("factors", [])
    chart = results.get("chart_data", {})
    exposure = results.get("exposure", {})
    stress = results.get("stress_tests", [])

    dates_json = json.dumps(chart.get("dates", []))
    equity_json = json.dumps(chart.get("equity", []))
    benchmark_json = json.dumps(chart.get("benchmark", []))
    drawdown_json = json.dumps(chart.get("drawdown", []))
    sharpe_json = json.dumps(chart.get("rolling_sharpe", []))
    monthly_json = json.dumps(chart.get("monthly_returns", {}))

    # Factor table rows
    factor_rows = ""
    for f in factors:
        ic = f.get("mean_ic", 0)
        icir = f.get("icir", 0)
        color = "#22c55e" if icir > 0.3 else "#fbbf24" if icir > 0 else "#ef4444"
        factor_rows += f"""
        <tr>
          <td style="font-weight:600;color:#4da6ff">{f.get('name','')}</td>
          <td style="font-family:monospace">{ic:.4f}</td>
          <td style="font-family:monospace;color:{color};font-weight:700">{icir:.3f}</td>
          <td style="font-family:monospace">{f.get('ic_positive_ratio',0)*100:.1f}%</td>
          <td style="font-family:monospace">{f.get('std_ic',0):.4f}</td>
        </tr>"""

    # Sector exposure rows
    sector_rows = ""
    sectors = exposure.get("sectors", {})
    for sector, weight in sorted(sectors.items(), key=lambda x: -x[1])[:10]:
        bar_width = min(weight * 300, 100)
        sector_rows += f"""
        <tr>
          <td>{sector}</td>
          <td style="font-family:monospace">{weight*100:.2f}%</td>
          <td><div style="background:#4da6ff;height:6px;width:{bar_width}%;border-radius:3px"></div></td>
        </tr>"""

    # Stress test rows
    stress_rows = ""
    for s in stress:
        ret_color = "#ef4444" if s.get("cumulative_return", 0) < 0 else "#22c55e"
        stress_rows += f"""
        <tr>
          <td>{s.get('scenario','')}</td>
          <td style="font-family:monospace;color:{ret_color}">{s.get('cumulative_return',0)*100:.2f}%</td>
          <td style="font-family:monospace;color:#ef4444">{s.get('max_drawdown',0)*100:.2f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0e17; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #1e2a3a; }}
  .header h1 {{ font-size: 20px; color: #4da6ff; font-weight: 700; letter-spacing: 1px; }}
  .header .meta {{ font-size: 11px; color: #6b7a8d; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 20px; }}
  .kpi {{ background: #111827; border: 1px solid #1e2a3a; border-radius: 8px; padding: 12px; text-align: center; }}
  .kpi-label {{ font-size: 9px; font-weight: 700; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
  .kpi-value {{ font-size: 18px; font-weight: 700; font-family: 'SF Mono', 'Fira Code', monospace; color: #e6edf3; }}
  .kpi-value.pos {{ color: #22c55e; }}
  .kpi-value.neg {{ color: #ef4444; }}
  .kpi-value.accent {{ color: #4da6ff; }}
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
  <h1>{title}</h1>
  <div class="meta">
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
    Period: {chart.get('dates', [''])[0]} → {chart.get('dates', [''])[-1]}<br>
    Optimizer: {perf.get('optimizer', 'N/A')}
  </div>
</div>

<!-- KPI Strip -->
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Total Return</div><div class="kpi-value {'pos' if (perf.get('total_return',0) or 0) >= 0 else 'neg'}">{_format_pct(perf.get('total_return'))}</div></div>
  <div class="kpi"><div class="kpi-label">Annual Return</div><div class="kpi-value {'pos' if (perf.get('annual_return',0) or 0) >= 0 else 'neg'}">{_format_pct(perf.get('annual_return'))}</div></div>
  <div class="kpi"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-value accent">{_format_num(perf.get('sharpe_ratio'), 2)}</div></div>
  <div class="kpi"><div class="kpi-label">Sortino Ratio</div><div class="kpi-value accent">{_format_num(perf.get('sortino_ratio'), 2)}</div></div>
  <div class="kpi"><div class="kpi-label">Max Drawdown</div><div class="kpi-value neg">{_format_pct(perf.get('max_drawdown'))}</div></div>
  <div class="kpi"><div class="kpi-label">Calmar Ratio</div><div class="kpi-value">{_format_num(perf.get('calmar_ratio'), 2)}</div></div>
  <div class="kpi"><div class="kpi-label">Win Rate</div><div class="kpi-value">{(perf.get('win_rate',0) or 0)*100:.1f}%</div></div>
  <div class="kpi"><div class="kpi-label">Information Ratio</div><div class="kpi-value accent">{_format_num(perf.get('information_ratio'), 2)}</div></div>
  <div class="kpi"><div class="kpi-label">Annual Vol</div><div class="kpi-value">{_format_pct(perf.get('annual_volatility'))}</div></div>
  <div class="kpi"><div class="kpi-label">Rebalances</div><div class="kpi-value">{perf.get('n_rebalances', 0)}</div></div>
  <div class="kpi"><div class="kpi-label">VaR (95%)</div><div class="kpi-value neg">{_format_pct(risk.get('historical_var'))}</div></div>
  <div class="kpi"><div class="kpi-label">CVaR (95%)</div><div class="kpi-value neg">{_format_pct(risk.get('historical_cvar'))}</div></div>
</div>

<!-- Equity + Drawdown -->
<div class="two-col">
  <div class="section">
    <div class="section-title">Equity Curve</div>
    <div id="equityChart" class="chart"></div>
  </div>
  <div class="section">
    <div class="section-title">Drawdown</div>
    <div id="ddChart" class="chart"></div>
  </div>
</div>

<!-- Rolling Sharpe + Monthly Returns -->
<div class="two-col">
  <div class="section">
    <div class="section-title">Rolling Sharpe (252d)</div>
    <div id="sharpeChart" class="chart-sm"></div>
  </div>
  <div class="section">
    <div class="section-title">Monthly Returns</div>
    <div id="monthlyChart" class="chart-sm"></div>
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
    <div class="kpi"><div class="kpi-label">Historical VaR</div><div class="kpi-value neg">{_format_pct(risk.get('historical_var'))}</div></div>
    <div class="kpi"><div class="kpi-label">Parametric VaR</div><div class="kpi-value neg">{_format_pct(risk.get('parametric_var'))}</div></div>
    <div class="kpi"><div class="kpi-label">CVaR</div><div class="kpi-value neg">{_format_pct(risk.get('historical_cvar'))}</div></div>
    <div class="kpi"><div class="kpi-label">Tracking Error</div><div class="kpi-value">{_format_pct(perf.get('tracking_error'))}</div></div>
    <div class="kpi"><div class="kpi-label">Excess Return</div><div class="kpi-value {'pos' if (perf.get('excess_return',0) or 0) >= 0 else 'neg'}">{_format_pct(perf.get('excess_return'))}</div></div>
    <div class="kpi"><div class="kpi-label">Effective N</div><div class="kpi-value accent">{_format_num(exposure.get('effective_n'), 1)}</div></div>
  </div>
</div>

<script>
const dates = {dates_json};
const equity = {equity_json};
const benchmark = {benchmark_json};
const drawdown = {drawdown_json};
const rollingSharpe = {sharpe_json};
const monthlyData = {monthly_json};

// Equity Chart
const eqChart = echarts.init(document.getElementById('equityChart'));
eqChart.setOption({{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ data: ['Strategy', 'Benchmark'], textStyle: {{ color: '#6b7a8d', fontSize: 10 }}, top: 0, right: 0 }},
  grid: {{ top: 30, right: 16, bottom: 28, left: 50 }},
  xAxis: {{ type: 'category', data: dates, axisLabel: {{ color: '#6b7a8d', fontSize: 9 }}, axisLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  yAxis: {{ type: 'value', axisLabel: {{ color: '#6b7a8d', fontSize: 9 }}, splitLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  series: [
    {{ name: 'Strategy', type: 'line', data: equity, smooth: 0.2, symbol: 'none', lineStyle: {{ color: '#4da6ff', width: 2 }},
       areaStyle: {{ color: new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:'rgba(77,166,255,0.2)'}},{{offset:1,color:'rgba(77,166,255,0.01)'}}]) }} }},
    {{ name: 'Benchmark', type: 'line', data: benchmark, smooth: 0.2, symbol: 'none', lineStyle: {{ color: '#6b7a8d', width: 1, type: 'dashed' }} }}
  ]
}});

// Drawdown Chart
const ddChart = echarts.init(document.getElementById('ddChart'));
ddChart.setOption({{
  tooltip: {{ trigger: 'axis', formatter: p => p[0].name + '<br/>DD: ' + (p[0].value*100).toFixed(2) + '%' }},
  grid: {{ top: 8, right: 16, bottom: 28, left: 50 }},
  xAxis: {{ type: 'category', data: dates, axisLabel: {{ color: '#6b7a8d', fontSize: 9 }}, axisLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  yAxis: {{ type: 'value', axisLabel: {{ color: '#6b7a8d', fontSize: 9, formatter: v => (v*100).toFixed(0)+'%' }}, splitLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  series: [{{ type: 'line', data: drawdown, smooth: 0.2, symbol: 'none', lineStyle: {{ color: '#ef4444', width: 1 }},
    areaStyle: {{ color: new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:'rgba(239,68,68,0.3)'}},{{offset:1,color:'rgba(239,68,68,0.01)'}}]) }} }}]
}});

// Rolling Sharpe
const shChart = echarts.init(document.getElementById('sharpeChart'));
shChart.setOption({{
  tooltip: {{ trigger: 'axis' }},
  grid: {{ top: 8, right: 16, bottom: 24, left: 40 }},
  xAxis: {{ type: 'category', data: dates.slice(-rollingSharpe.length), axisLabel: {{ color: '#6b7a8d', fontSize: 8 }}, axisLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  yAxis: {{ type: 'value', axisLabel: {{ color: '#6b7a8d', fontSize: 9 }}, splitLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  series: [{{ type: 'line', data: rollingSharpe, smooth: 0.2, symbol: 'none', lineStyle: {{ color: '#fbbf24', width: 1.5 }},
    areaStyle: {{ color: new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:'rgba(251,191,36,0.15)'}},{{offset:1,color:'rgba(251,191,36,0.01)'}}]) }} }}]
}});

// Monthly Heatmap
const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const years = Object.keys(monthlyData).sort();
const heatData = [];
years.forEach((y, yi) => {{
  (monthlyData[y] || []).forEach((v, mi) => {{
    if (v !== null && v !== undefined) heatData.push([mi, yi, v]);
  }});
}});
const mChart = echarts.init(document.getElementById('monthlyChart'));
mChart.setOption({{
  tooltip: {{ formatter: p => months[p.data[0]] + ' ' + years[p.data[1]] + ': ' + (p.data[2]*100).toFixed(2) + '%' }},
  grid: {{ top: 8, right: 12, bottom: 24, left: 40 }},
  xAxis: {{ type: 'category', data: months, axisLabel: {{ color: '#6b7a8d', fontSize: 9 }}, axisLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  yAxis: {{ type: 'category', data: years, axisLabel: {{ color: '#6b7a8d', fontSize: 9 }}, axisLine: {{ lineStyle: {{ color: '#1e2a3a' }} }} }},
  visualMap: {{ min: -0.1, max: 0.1, calculable: false, orient: 'horizontal', left: 'center', bottom: 0, show: false,
    inRange: {{ color: ['#ef4444', '#111827', '#22c55e'] }} }},
  series: [{{ type: 'heatmap', data: heatData, label: {{ show: true, fontSize: 8, color: '#c9d1d9',
    formatter: p => (p.data[2]*100).toFixed(1) + '%' }} }}]
}});

window.addEventListener('resize', () => {{ eqChart.resize(); ddChart.resize(); shChart.resize(); mChart.resize(); }});
</script>

<div style="text-align:center;padding:24px;color:#3a4a5e;font-size:10px">
  Generated by A-Share Multi-Factor Quant Platform &bull; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

</body>
</html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logger.info("HTML report generated: %s", output)
    return str(output)
