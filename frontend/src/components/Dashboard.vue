<template>
  <div>
    <!-- Page Header -->
    <div class="section-header">
      <div>
        <div class="section-title">Backtest Dashboard</div>
        <div class="section-subtitle">Configure, run, and analyze multi-factor strategies</div>
      </div>
      <div class="flex-row gap-sm">
        <button class="btn btn-secondary btn-sm" @click="loadDemo" :disabled="running">
          &#9889; Demo Data
        </button>
        <button class="btn btn-secondary btn-sm" @click="exportResults" :disabled="!result">
          &#9744; Export
        </button>
      </div>
    </div>

    <!-- Run Configuration Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Pipeline Configuration
        </div>
        <div class="flex-row gap-sm">
          <span class="tag tag-accent" v-if="runId">RUN {{ runId }}</span>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label for="cfg-stocks">Stock Universe</label>
          <select id="cfg-stocks" v-model.number="config.n_stocks">
            <option :value="100">100 stocks</option>
            <option :value="200">200 stocks</option>
            <option :value="300">300 stocks</option>
            <option :value="500">500 stocks</option>
          </select>
          <div class="form-hint">Larger universe = longer runtime</div>
        </div>
        <div class="form-group">
          <label for="cfg-optimizer">Portfolio Optimizer</label>
          <select id="cfg-optimizer" v-model="config.optimizer">
            <option value="equal_weight">Equal Weight (1/N)</option>
            <option value="mean_variance">Mean-Variance (MVO)</option>
            <option value="risk_parity">Risk Parity</option>
          </select>
        </div>
        <div class="form-group">
          <label for="cfg-alpha">Alpha Synthesis</label>
          <select id="cfg-alpha" v-model="config.alpha_method">
            <option value="equal_weight">Equal Weight</option>
            <option value="ic_weighted">IC Weighted</option>
            <option value="icir_weighted">ICIR Weighted</option>
          </select>
        </div>
        <div class="form-group">
          <label for="cfg-freq">Rebalance Frequency</label>
          <select id="cfg-freq" v-model="config.rebalance_frequency">
            <option value="monthly">Monthly</option>
            <option value="weekly">Weekly</option>
          </select>
        </div>
        <div class="form-group">
          <label for="cfg-cov">Covariance Method</label>
          <select id="cfg-cov" v-model="config.covariance_method">
            <option value="ledoit_wolf">Ledoit-Wolf Shrinkage</option>
            <option value="sample">Sample Covariance</option>
            <option value="ewma">EWMA</option>
          </select>
        </div>
      </div>

      <div class="flex-between mt-2">
        <div class="flex-row gap-sm">
          <button class="btn btn-primary" :disabled="running" @click="startRun">
            <span v-if="!running">&#9654; Run Pipeline</span>
            <span v-else>
              <span class="status-spinner" style="display:inline-block;"></span>
              Running...
            </span>
          </button>
          <span v-if="running" class="text-muted text-sm">{{ progress }}% &middot; {{ stage }}</span>
        </div>
        <span v-if="!running && result" class="tag tag-green">&#10003; Completed</span>
      </div>
    </div>

    <!-- Progress Card -->
    <Transition name="tab-content">
      <div v-if="running" class="card">
        <div class="card-header">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Pipeline Progress
          </div>
          <span class="status-badge running">
            <span class="status-spinner"></span>
            {{ stageLabel }}
          </span>
        </div>
        <div class="progress-stages">
          <div
            v-for="(s, i) in stages" :key="s"
            :class="['progress-stage', stageIdx > i ? 'done' : '', stageIdx === i ? 'active' : '']"
            :title="s"
          ></div>
        </div>
        <div class="progress-bar" role="progressbar" :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100">
          <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        </div>
        <div class="status-row">
          <span class="text-xs text-dim">Run {{ runId }}</span>
          <span class="text-mono text-accent">{{ progress }}%</span>
        </div>
      </div>
    </Transition>

    <!-- Error -->
    <Transition name="tab-content">
      <div v-if="error" role="alert" class="alert alert-error">
        <span aria-hidden="true">&#10007;</span>
        <div>
          <div style="font-weight:600;">Pipeline Error</div>
          <div class="text-sm mt-1" style="opacity:0.85;">{{ error }}</div>
        </div>
      </div>
    </Transition>

    <!-- Loading Skeleton -->
    <template v-if="running && !result">
      <div class="card">
        <div class="card-title"><span class="card-title-dot"></span>Performance Metrics</div>
        <div class="skeleton-metrics mt-3">
          <div class="skeleton skeleton-metric" v-for="i in 8" :key="i"></div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title"><span class="card-title-dot"></span>Equity Curve</div>
          <div class="skeleton skeleton-chart mt-3"></div>
        </div>
        <div class="card">
          <div class="card-title"><span class="card-title-dot"></span>Drawdown</div>
          <div class="skeleton skeleton-chart mt-3"></div>
        </div>
      </div>
    </template>

    <!-- Results -->
    <template v-if="result">
      <!-- Performance Metrics -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Performance Metrics
          </div>
          <span class="text-xs text-dim">{{ result.performance?.total_days }} trading days &middot; {{ result.performance?.n_rebalances }} rebalances</span>
        </div>
        <div class="metrics-grid">
          <div class="metric-card" v-for="m in metricCards" :key="m.label">
            <div class="metric-card-top">
              <div class="metric-icon" aria-hidden="true">{{ m.icon }}</div>
              <Sparkline
                v-if="m.sparkline?.length"
                :data="m.sparkline"
                :color="m.color === 'positive' ? '#34d399' : m.color === 'negative' ? '#f87171' : '#fb923c'"
                :width="60"
                :height="24"
              />
            </div>
            <div :class="['metric-value', m.color]">{{ m.displayValue }}</div>
            <div v-if="m.trendLabel" :class="['metric-trend', m.trend > 0 ? 'positive' : 'negative']">
              {{ m.trend > 0 ? '&#9650;' : '&#9660;' }} {{ m.trendLabel }}
            </div>
            <div class="metric-label">{{ m.label }}</div>
          </div>
        </div>
      </div>

      <!-- Charts Row 1: Equity + Drawdown -->
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Equity Curve
          </div>
          <div class="chart-wrapper">
            <div ref="equityChartRef" class="chart-container" role="img" aria-label="Equity curve chart"></div>
          </div>
        </div>
        <div class="card">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Drawdown
          </div>
          <div class="chart-wrapper">
            <div ref="drawdownChartRef" class="chart-container" role="img" aria-label="Drawdown chart"></div>
          </div>
        </div>
      </div>

      <!-- Chart Row 2: Rolling Sharpe -->
      <div class="card" v-if="result.chart_data?.rolling_sharpe?.length">
        <div class="card-title">
          <span class="card-title-dot"></span>
          Rolling Sharpe Ratio (252-Day)
        </div>
        <div class="chart-wrapper">
          <div ref="sharpeChartRef" class="chart-container large" role="img" aria-label="Rolling Sharpe ratio chart"></div>
        </div>
      </div>

      <!-- Risk + Exposure Row -->
      <div class="grid-2">
        <div class="card" v-if="result.risk">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Risk Metrics
          </div>
          <div class="table-container">
            <table>
              <thead>
                <tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>Historical VaR (95%)</td>
                  <td class="negative text-mono">{{ (result.risk.historical_var * 100).toFixed(2) }}%</td>
                  <td><span class="tag tag-orange">Daily Risk</span></td>
                </tr>
                <tr>
                  <td>Parametric VaR (95%)</td>
                  <td class="negative text-mono">{{ (result.risk.parametric_var * 100).toFixed(2) }}%</td>
                  <td><span class="tag tag-orange">Normal Dist</span></td>
                </tr>
                <tr>
                  <td>Historical CVaR (95%)</td>
                  <td class="negative text-mono">{{ (result.risk.historical_cvar * 100).toFixed(2) }}%</td>
                  <td><span class="tag tag-red">Tail Risk</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="card" v-if="result.exposure">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Portfolio Exposure
          </div>
          <div class="table-container">
            <table>
              <thead>
                <tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>Holdings</td>
                  <td class="text-mono">{{ result.exposure.n_assets }}</td>
                  <td><span class="tag tag-accent">Diversified</span></td>
                </tr>
                <tr>
                  <td>Effective N</td>
                  <td class="text-mono">{{ result.exposure.effective_n.toFixed(1) }}</td>
                  <td><span class="tag tag-accent">Concentration</span></td>
                </tr>
                <tr>
                  <td>Top 5 Concentration</td>
                  <td class="text-mono">{{ (result.exposure.top5_concentration * 100).toFixed(1) }}%</td>
                  <td><span :class="['tag', result.exposure.top5_concentration > 0.3 ? 'tag-red' : 'tag-green']">
                    {{ result.exposure.top5_concentration > 0.3 ? 'High' : 'OK' }}
                  </span></td>
                </tr>
                <tr>
                  <td>Top 10 Concentration</td>
                  <td class="text-mono">{{ (result.exposure.top10_concentration * 100).toFixed(1) }}%</td>
                  <td><span :class="['tag', result.exposure.top10_concentration > 0.5 ? 'tag-orange' : 'tag-green']">
                    {{ result.exposure.top10_concentration > 0.5 ? 'Watch' : 'OK' }}
                  </span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Stress Tests -->
      <div class="card" v-if="result.stress_tests?.length">
        <div class="card-header">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Stress Tests
          </div>
          <span class="text-xs text-dim">Scenario-based risk analysis</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr><th>Scenario</th><th>Cumulative Return</th><th>Max Drawdown</th><th>Severity</th></tr>
            </thead>
            <tbody>
              <tr v-for="st in result.stress_tests" :key="st.scenario">
                <td class="highlight-number">{{ st.scenario }}</td>
                <td :class="st.cumulative_return >= 0 ? 'positive' : 'negative'">
                  {{ (st.cumulative_return * 100).toFixed(2) }}%
                </td>
                <td class="negative text-mono">{{ (st.max_drawdown * 100).toFixed(2) }}%</td>
                <td>
                  <span :class="['tag', Math.abs(st.max_drawdown) > 0.4 ? 'tag-red' : Math.abs(st.max_drawdown) > 0.2 ? 'tag-orange' : 'tag-green']">
                    {{ Math.abs(st.max_drawdown) > 0.4 ? 'Severe' : Math.abs(st.max_drawdown) > 0.2 ? 'Moderate' : 'Mild' }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Factor IC Rankings (compact) -->
      <div class="card" v-if="result.factors?.length">
        <div class="card-header">
          <div class="card-title">
            <span class="card-title-dot"></span>
            Factor IC Rankings
          </div>
          <span class="text-xs text-dim">Sorted by |ICIR|</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr><th>#</th><th>Factor</th><th>Mean IC</th><th>ICIR</th><th>IC>0 %</th></tr>
            </thead>
            <tbody>
              <tr v-for="(f, i) in result.factors.slice(0, 10)" :key="f.name">
                <td class="text-dim">{{ i + 1 }}</td>
                <td class="highlight-number">{{ f.name }}</td>
                <td :class="f.mean_ic > 0.02 ? 'positive' : f.mean_ic < -0.02 ? 'negative' : ''">
                  {{ Number(f.mean_ic).toFixed(4) }}
                </td>
                <td>
                  <span
                    :class="['icir-bar', Math.abs(Number(f.icir)) >= 0.2 ? 'positive-bar' : 'negative-bar']"
                    :style="{ width: Math.min(Math.abs(Number(f.icir)) * 80, 100) + 'px' }"
                  ></span>
                  <span :class="Math.abs(Number(f.icir)) > 0.3 ? 'positive' : Math.abs(Number(f.icir)) > 0.15 ? 'neutral' : ''">
                    {{ Number(f.icir).toFixed(2) }}
                  </span>
                </td>
                <td :class="Number(f.ic_positive_ratio) > 0.55 ? 'positive' : Number(f.ic_positive_ratio) < 0.45 ? 'negative' : ''">
                  {{ (Number(f.ic_positive_ratio) * 100).toFixed(1) }}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>

    <!-- Empty state -->
    <div v-if="!result && !running" class="empty-state">
      <div class="empty-icon">&#9656;</div>
      <h3>Ready to Backtest</h3>
      <p>Configure your strategy parameters above and click "Run Pipeline" to start a full backtest.</p>
      <p class="mt-3">
        <button class="btn btn-primary" @click="startRun">&#9654; Run Pipeline</button>
        <button class="btn btn-secondary ml-2" @click="loadDemo">&#9889; Load Demo</button>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, nextTick, onBeforeUnmount } from 'vue'
import * as echarts from 'echarts'
import { runPipeline, getRunStatus, getRunResult, getDemo } from '../api/index.js'
import Sparkline from './Sparkline.vue'

const emit = defineEmits(['toast'])

const stages = ['data', 'factors', 'alpha', 'backtest', 'report', 'done']
const stageLabels = {
  data: 'Loading Data', factors: 'Computing Factors', alpha: 'Generating Alpha',
  backtest: 'Running Backtest', report: 'Building Report', done: 'Complete',
}

const config = reactive({
  n_stocks: 300,
  optimizer: 'mean_variance',
  alpha_method: 'icir_weighted',
  rebalance_frequency: 'monthly',
  covariance_method: 'ledoit_wolf',
  start_date: '2021-01-01',
  end_date: '2025-12-31',
  use_tushare: false,
})

const running = ref(false)
const runId = ref(null)
const progress = ref(0)
const stage = ref('')
const error = ref(null)
const result = ref(null)
let pollTimer = null
let pollFailCount = 0

const equityChartRef = ref(null)
const drawdownChartRef = ref(null)
const sharpeChartRef = ref(null)
let chartInstances = []
let resizeObservers = []
let animFrameIds = []

const stageIdx = computed(() => stages.indexOf(stage.value))
const stageLabel = computed(() => stageLabels[stage.value] || stage.value)

const animatedValues = reactive({})

function animateValue(key, target) {
  const start = animatedValues[key] ?? 0
  const duration = 700
  const startTime = performance.now()
  function step(now) {
    const elapsed = now - startTime
    const t = Math.min(elapsed / duration, 1)
    const eased = 1 - Math.pow(1 - t, 3)
    animatedValues[key] = start + (target - start) * eased
    if (t < 1) {
      const id = requestAnimationFrame(step)
      animFrameIds.push(id)
    } else {
      animatedValues[key] = target
    }
  }
  const id = requestAnimationFrame(step)
  animFrameIds.push(id)
}

// Generate sparkline data from equity curve (downsample to 30 points)
function makeSparkline(data, n = 30) {
  if (!data?.length || data.length < 2) return []
  const step = Math.max(1, Math.floor(data.length / n))
  const sampled = []
  for (let i = 0; i < data.length; i += step) sampled.push(data[i])
  if (sampled[sampled.length - 1] !== data[data.length - 1]) sampled.push(data[data.length - 1])
  return sampled
}

const equitySparkline = computed(() => makeSparkline(result.value?.chart_data?.equity))
const drawdownSparkline = computed(() => makeSparkline(result.value?.chart_data?.drawdown?.map(v => Math.abs(v))))

const metricCards = computed(() => {
  if (!result.value?.performance) return []
  const p = result.value.performance

  const items = [
    { key: 'total_return',  icon: '📈', label: 'Total Return',  value: p.total_return,      trend: p.total_return,  fmt: v => (v * 100).toFixed(2) + '%', color: p.total_return >= 0 ? 'positive' : 'negative', sparkline: equitySparkline.value },
    { key: 'annual_return', icon: '📊', label: 'Annual Return', value: p.annual_return,     trend: p.annual_return, fmt: v => (v * 100).toFixed(2) + '%', color: p.annual_return >= 0 ? 'positive' : 'negative', sparkline: null },
    { key: 'annual_vol',    icon: '〰', label: 'Annual Vol',    value: p.annual_volatility, trend: null,            fmt: v => (v * 100).toFixed(2) + '%', color: 'neutral', sparkline: null },
    { key: 'sharpe',        icon: '⚡', label: 'Sharpe Ratio',  value: p.sharpe_ratio,      trend: p.sharpe_ratio,  fmt: v => v.toFixed(2), color: p.sharpe_ratio >= 1 ? 'positive' : p.sharpe_ratio >= 0 ? 'neutral' : 'negative', sparkline: null },
    { key: 'sortino',       icon: '📐', label: 'Sortino Ratio', value: p.sortino_ratio,     trend: null,            fmt: v => v.toFixed(2), color: p.sortino_ratio >= 0 ? 'positive' : 'negative', sparkline: null },
    { key: 'max_dd',        icon: '⬇', label: 'Max Drawdown',  value: p.max_drawdown,      trend: null,            fmt: v => (v * 100).toFixed(2) + '%', color: 'negative', sparkline: drawdownSparkline.value },
    { key: 'win_rate',      icon: '🎯', label: 'Win Rate',      value: p.win_rate,          trend: p.win_rate,      fmt: v => (v * 100).toFixed(1) + '%', color: p.win_rate >= 0.5 ? 'positive' : 'negative', sparkline: null },
    { key: 'calmar',        icon: '⚖', label: 'Calmar Ratio',   value: p.calmar_ratio,      trend: null,            fmt: v => v.toFixed(2), color: p.calmar_ratio >= 0 ? 'positive' : 'negative', sparkline: null },
  ]

  items.forEach(item => animateValue(item.key, Number(item.value) || 0))

  return items.map(item => ({
    icon: item.icon,
    label: item.label,
    displayValue: item.fmt(animatedValues[item.key] ?? item.value),
    color: item.color,
    trend: item.trend,
    trendLabel: item.trend != null && Math.abs(item.trend) > 0.05 ? (Math.abs(item.trend) * 100).toFixed(1) + '%' : '',
    sparkline: item.sparkline,
  }))
})

async function startRun() {
  error.value = null
  result.value = null
  running.value = true
  progress.value = 0
  stage.value = 'data'
  pollFailCount = 0

  try {
    const status = await runPipeline({ ...config })
    runId.value = status.run_id
    startPolling()
    emit('toast', { message: 'Pipeline started: ' + status.run_id, type: 'info' })
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    running.value = false
    emit('toast', { message: 'Failed to start pipeline', type: 'error' })
  }
}

function startPolling() {
  pollTimer = setInterval(async () => {
    try {
      const status = await getRunStatus(runId.value)
      pollFailCount = 0
      progress.value = status.progress
      stage.value = status.stage

      if (status.status === 'completed') {
        clearInterval(pollTimer)
        running.value = false
        try {
          const data = await getRunResult(runId.value)
          result.value = data
          emit('toast', { message: 'Pipeline completed successfully', type: 'success' })
          await nextTick()
          renderCharts()
        } catch (e) {
          error.value = 'Failed to fetch result: ' + (e.response?.data?.detail || e.message)
          emit('toast', { message: 'Failed to fetch result', type: 'error' })
        }
      } else if (status.status === 'failed') {
        clearInterval(pollTimer)
        running.value = false
        error.value = status.error || 'Pipeline failed'
        emit('toast', { message: 'Pipeline failed', type: 'error' })
      }
    } catch (_) {
      pollFailCount++
      if (pollFailCount >= 5) {
        clearInterval(pollTimer)
        running.value = false
        error.value = 'Lost connection to server (5 consecutive failures)'
        emit('toast', { message: 'Connection lost', type: 'error' })
      }
    }
  }, 1500)
}

async function loadDemo() {
  error.value = null
  try {
    result.value = await getDemo()
    emit('toast', { message: 'Demo data loaded', type: 'success' })
    await nextTick()
    renderCharts()
  } catch (e) {
    error.value = e.response?.data?.detail || e.message
    emit('toast', { message: 'Failed to load demo', type: 'error' })
  }
}

function exportResults() {
  if (!result.value) return
  const data = JSON.stringify(result.value, null, 2)
  const blob = new Blob([data], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `backtest-${runId.value || 'demo'}.json`
  a.click()
  URL.revokeObjectURL(url)
  emit('toast', { message: 'Results exported', type: 'success' })
}

function disposeCharts() {
  chartInstances.forEach(c => { try { c.dispose() } catch (_) {} })
  chartInstances = []
  resizeObservers.forEach(o => { try { o.disconnect() } catch (_) {} })
  resizeObservers = []
}

function cancelAnimations() {
  animFrameIds.forEach(id => cancelAnimationFrame(id))
  animFrameIds = []
}

function initChart(containerRef) {
  if (!containerRef.value) return null
  const chart = echarts.init(containerRef.value, null, { renderer: 'canvas' })
  chartInstances.push(chart)
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(containerRef.value)
    resizeObservers.push(ro)
  }
  return chart
}

const chartBase = {
  textStyle: { color: '#8b9dc0', fontFamily: 'Inter, -apple-system, sans-serif' },
  legend: { textStyle: { color: '#8b9dc0', fontSize: 11 }, bottom: 0, itemGap: 20 },
  grid: { left: 56, right: 20, top: 16, bottom: 56 },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(15, 24, 41, 0.95)',
    borderColor: '#1c2d4a',
    borderWidth: 1,
    textStyle: { color: '#e8edf5', fontSize: 12, fontFamily: 'Inter, sans-serif' },
    axisPointer: { type: 'cross', lineStyle: { color: '#243756', type: 'dashed' }, crossStyle: { color: '#243756' } },
    extraCssText: 'backdrop-filter: blur(8px); border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);',
  },
  toolbox: {
    right: 12,
    top: 0,
    itemSize: 14,
    iconStyle: { borderColor: '#556882' },
    emphasis: { iconStyle: { borderColor: '#8b9dc0' } },
    feature: {
      saveAsImage: { title: 'Save', pixelRatio: 2 },
      dataZoom: { title: { zoom: 'Zoom', back: 'Reset' } },
      restore: { title: 'Reset' },
    },
  },
}

function makeAxis(dates) {
  return {
    type: 'category', data: dates,
    axisLine: { lineStyle: { color: '#1c2d4a' } },
    axisLabel: { color: '#556882', fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 8)) },
    axisTick: { show: false },
  }
}

function makeValueAxis(fmt) {
  return {
    type: 'value',
    axisLine: { show: false },
    axisLabel: { color: '#556882', fontSize: 10, formatter: fmt || (v => v.toFixed(2)) },
    splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
    axisTick: { show: false },
  }
}

function makeDataZoom(dates) {
  return [
    {
      type: 'inside',
      start: 0,
      end: 100,
      zoomOnMouseWheel: true,
      moveOnMouseMove: true,
    },
    {
      type: 'slider',
      start: 0,
      end: 100,
      height: 20,
      bottom: 4,
      borderColor: '#1c2d4a',
      backgroundColor: '#0a1121',
      fillerColor: 'rgba(77,166,255,0.08)',
      handleStyle: { color: '#4da6ff', borderColor: '#4da6ff' },
      textStyle: { color: '#556882', fontSize: 10 },
      dataBackground: {
        lineStyle: { color: '#1c2d4a' },
        areaStyle: { color: 'rgba(77,166,255,0.05)' },
      },
      selectedDataBackground: {
        lineStyle: { color: '#4da6ff' },
        areaStyle: { color: 'rgba(77,166,255,0.1)' },
      },
    },
  ]
}

function renderCharts() {
  if (!result.value?.chart_data) return
  disposeCharts()
  const cd = result.value.chart_data

  // ── Equity Curve ──
  const eqChart = initChart(equityChartRef)
  if (eqChart) {
    const series = [{
      name: 'Strategy', type: 'line', data: cd.equity, smooth: 0.3,
      lineStyle: { color: '#4da6ff', width: 2 },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: 'rgba(77,166,255,0.18)' },
        { offset: 1, color: 'rgba(77,166,255,0.01)' },
      ])},
      symbol: 'none', emphasis: { focus: 'series' },
    }]
    if (cd.benchmark?.length) {
      series.push({
        name: 'Benchmark', type: 'line', data: cd.benchmark, smooth: 0.3,
        lineStyle: { color: '#3a4d6a', width: 1.5, type: 'dashed' },
        symbol: 'none', emphasis: { focus: 'series' },
      })
    }
    eqChart.setOption({
      ...chartBase,
      xAxis: makeAxis(cd.dates),
      yAxis: makeValueAxis(),
      dataZoom: makeDataZoom(cd.dates),
      series,
    })
  }

  // ── Drawdown ──
  const ddChart = initChart(drawdownChartRef)
  if (ddChart && cd.drawdown?.length) {
    ddChart.setOption({
      ...chartBase,
      xAxis: makeAxis(cd.dates),
      yAxis: { ...makeValueAxis(v => v + '%'), max: 0 },
      dataZoom: makeDataZoom(cd.dates),
      series: [{
        name: 'Drawdown', type: 'line', data: cd.drawdown.map(v => (Number(v) * 100).toFixed(2)),
        smooth: 0.3,
        lineStyle: { color: '#f87171', width: 1.5 },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(248,113,113,0.25)' },
          { offset: 1, color: 'rgba(248,113,113,0.0)' },
        ])},
        symbol: 'none',
      }],
      tooltip: { ...chartBase.tooltip, valueFormatter: v => v + '%' },
    })
  }

  // ── Rolling Sharpe ──
  const rsChart = initChart(sharpeChartRef)
  if (rsChart && cd.rolling_sharpe?.length) {
    rsChart.setOption({
      ...chartBase,
      xAxis: makeAxis(cd.rolling_sharpe_dates || []),
      yAxis: makeValueAxis(),
      dataZoom: makeDataZoom(cd.rolling_sharpe_dates || []),
      series: [{
        name: 'Rolling Sharpe', type: 'line',
        data: cd.rolling_sharpe.map(v => Number(v).toFixed(2)),
        smooth: 0.3,
        lineStyle: { color: '#fb923c', width: 2 },
        symbol: 'none',
        markLine: {
          silent: true,
          data: [{ yAxis: 0, lineStyle: { color: '#3a4d6a', type: 'dashed' } }],
          symbol: 'none', label: { show: false },
        },
      }],
    })
  }
}

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
  cancelAnimations()
  disposeCharts()
})

defineExpose({ startRun, loadDemo, exportResults })
</script>
