<template>
  <div class="terminal-grid">
    <!-- Row 0: KPI Strip -->
    <KpiStrip :performance="perf" :risk="risk" />

    <!-- Row 1: Equity + Drawdown -->
    <div class="tg-row-main">
      <Panel title="EQUITY CURVE" dotColor="#4da6ff" class="tg-equity">
        <template #actions>
          <span class="panel-tag" v-if="perf">{{ (perf.total_return * 100).toFixed(1) }}% total</span>
        </template>
        <div v-if="loading" class="tg-skeleton-chart"></div>
        <div v-else ref="equityRef" class="tg-chart"></div>
      </Panel>
      <Panel title="DRAWDOWN" dotColor="#f87171" class="tg-drawdown">
        <template #actions>
          <span class="panel-tag neg" v-if="perf">{{ (perf.max_drawdown * 100).toFixed(1) }}% max</span>
        </template>
        <div v-if="loading" class="tg-skeleton-chart"></div>
        <div v-else ref="drawdownRef" class="tg-chart"></div>
      </Panel>
    </div>

    <!-- Row 2: Factor Heatmap + Risk + Holdings -->
    <div class="tg-row-secondary">
      <Panel title="FACTOR IC HEATMAP" dotColor="#34d399" class="tg-factors">
        <template #actions>
          <span class="panel-tag" v-if="factors.length">{{ factors.length }} factors</span>
        </template>
        <div v-if="loading" class="tg-skeleton-block"></div>
        <FactorHeatmap v-else :factors="factors" />
      </Panel>

      <Panel title="RISK ANALYSIS" dotColor="#fb923c" class="tg-risk">
        <div v-if="loading" class="tg-skeleton-block"></div>
        <RiskGauges v-else :risk="risk" :stressTests="stressTests" />
      </Panel>

      <Panel title="PORTFOLIO EXPOSURE" dotColor="#a78bfa" class="tg-holdings">
        <div v-if="loading" class="tg-skeleton-block"></div>
        <HoldingsPanel v-else :exposure="exposure" />
      </Panel>
    </div>

    <!-- Row 3: Distribution + Excess Return + Holdings Table -->
    <div class="tg-row-analytics">
      <Panel title="RETURN DISTRIBUTION" dotColor="#818cf8" class="tg-dist">
        <template #actions>
          <span class="panel-tag" v-if="chartData?.return_distribution">
            μ={{ chartData.return_distribution.mean?.toFixed(2) }}% σ={{ chartData.return_distribution.std?.toFixed(2) }}%
          </span>
        </template>
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <ReturnDistribution v-else :distribution="chartData?.return_distribution" />
      </Panel>

      <Panel title="EXCESS CUMULATIVE" dotColor="#34d399" class="tg-excess">
        <template #actions>
          <span class="panel-tag" v-if="perf?.excess_return != null">{{ (perf.excess_return * 100).toFixed(1) }}% excess</span>
        </template>
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <div v-else ref="excessRef" class="tg-chart-sm"></div>
      </Panel>

      <Panel title="TOP HOLDINGS" dotColor="#fb923c" class="tg-stock-holdings">
        <template #actions>
          <span class="panel-tag" v-if="exposure?.top_holdings?.length">{{ exposure.top_holdings.length }} positions</span>
        </template>
        <div v-if="loading" class="tg-skeleton-block"></div>
        <HoldingsTable v-else :holdings="exposure?.top_holdings || []" />
      </Panel>
    </div>

    <!-- Row 4: Factor Scatter + Attribution + Turnover -->
    <div class="tg-row-research">
      <Panel title="FACTOR SCATTER" dotColor="#818cf8" class="tg-scatter">
        <template #actions>
          <span class="panel-tag" v-if="chartData?.factor_scatter?.length">{{ chartData.factor_scatter.length }} factors</span>
        </template>
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <FactorScatter v-else :scatterData="chartData?.factor_scatter || []" />
      </Panel>

      <Panel title="P&L ATTRIBUTION" dotColor="#34d399" class="tg-attribution">
        <template #actions>
          <span class="panel-tag" v-if="chartData?.attribution?.length">{{ chartData.attribution.length }} factors</span>
        </template>
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <AttributionWaterfall v-else :attribution="chartData?.attribution || []" />
      </Panel>

      <Panel title="TURNOVER ANALYSIS" dotColor="#a78bfa" class="tg-turnover">
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <TurnoverChart v-else :turnover="chartData?.turnover || []" />
      </Panel>
    </div>

    <!-- Row 5: Drawdown Periods -->
    <div class="tg-row-dd">
      <Panel title="DRAWDOWN PERIODS" dotColor="#f87171" class="tg-dd-table" full>
        <template #actions>
          <span class="panel-tag neg" v-if="chartData?.drawdown_periods?.length">
            {{ chartData.drawdown_periods.filter(p => !p.recovered).length }} ongoing
          </span>
        </template>
        <div v-if="loading" class="tg-skeleton-block"></div>
        <DrawdownPeriods v-else :periods="chartData?.drawdown_periods || []" />
      </Panel>
    </div>

    <!-- Row 5b: Factor Correlation + IC Decay -->
    <div class="tg-row-corr">
      <Panel title="FACTOR CORRELATION" dotColor="#a78bfa" class="tg-corr">
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <FactorCorrelation v-else :factors="factors" />
      </Panel>
      <Panel title="IC DECAY CURVE" dotColor="#34d399" class="tg-decay">
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <ICDecay v-else :factors="factors" />
      </Panel>
    </div>

    <!-- Row 6: Monthly Heatmap + Rolling Sharpe -->
    <div class="tg-row-bottom">
      <Panel title="MONTHLY RETURNS" dotColor="#22d3ee" class="tg-monthly">
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <div v-else ref="monthlyRef" class="tg-chart-sm"></div>
      </Panel>
      <Panel title="ROLLING SHARPE (252D)" dotColor="#fbbf24" class="tg-sharpe">
        <div v-if="loading" class="tg-skeleton-chart-sm"></div>
        <div v-else ref="sharpeRef" class="tg-chart-sm"></div>
      </Panel>
    </div>

    <!-- Row 7: Walk-Forward + Monte Carlo + Risk Decomposition -->
    <div class="tg-row-advanced">
      <Panel title="WALK-FORWARD" dotColor="#34d399" class="tg-wf">
        <WalkForward :runId="lastRunId" @toast="$emit('toast', $event)" />
      </Panel>
      <Panel title="MONTE CARLO" dotColor="#fbbf24" class="tg-mc">
        <MonteCarlo :runId="lastRunId" @toast="$emit('toast', $event)" />
      </Panel>
      <Panel title="RISK DECOMPOSITION" dotColor="#a78bfa" class="tg-rd">
        <RiskDecomposition :runId="lastRunId" @toast="$emit('toast', $event)" />
      </Panel>
    </div>

    <!-- Row 8: Regime Detection + Risk Monitor -->
    <div class="tg-row-regime">
      <Panel title="MARKET REGIME" dotColor="#22d3ee" class="tg-regime">
        <RegimeDetector :runId="lastRunId" @toast="$emit('toast', $event)" />
      </Panel>
      <Panel title="RISK MONITOR" dotColor="#ef4444" class="tg-riskmon">
        <RiskMonitor @toast="$emit('toast', $event)" />
      </Panel>
    </div>

    <!-- Row 9: Multi-Strategy + Data Quality + Report -->
    <div class="tg-row-multi">
      <Panel title="MULTI-STRATEGY" dotColor="#4da6ff" class="tg-multistrat">
        <MultiStrategy @toast="$emit('toast', $event)" />
      </Panel>
      <Panel title="DATA QUALITY" dotColor="#22c55e" class="tg-dataquality">
        <DataQuality :runId="lastRunId" @toast="$emit('toast', $event)" />
      </Panel>
    </div>

    <!-- Row 10: Report Generation -->
    <div class="tg-row-report">
      <Panel title="REPORT" dotColor="#fbbf24" class="tg-report-panel" full>
        <div class="tg-report-bar">
          <span class="tg-report-label">Generate a self-contained HTML backtest report with all charts, metrics, and analysis.</span>
          <button class="tg-report-btn" @click="downloadReport" :disabled="!lastRunId">
            <span v-if="generatingReport" class="tg-report-spinner"></span>
            <span v-else>Download HTML Report</span>
          </button>
          <span v-if="reportPath" class="tg-report-path">Saved: {{ reportPath }}</span>
        </div>
      </Panel>
    </div>

    <!-- Row 11: System Log -->
    <div class="tg-row-log">
      <Panel title="SYSTEM LOG" dotColor="#8b9dc0" class="tg-log" full>
        <SystemLog :entries="logEntries" />
      </Panel>
    </div>

    <!-- Run Controls Overlay -->
    <Transition name="fade">
      <div v-if="showRunPanel" class="tg-run-overlay" @click.self="showRunPanel = false">
        <div class="tg-run-panel">
          <div class="tg-run-header">
            <span>PIPELINE CONFIGURATION</span>
            <button class="tg-close-btn" @click="showRunPanel = false">&times;</button>
          </div>
          <div class="tg-run-body">
            <div class="tg-run-row">
              <div class="tg-run-field">
                <label>Universe</label>
                <select v-model.number="runConfig.n_stocks">
                  <option :value="100">100 stocks</option>
                  <option :value="200">200 stocks</option>
                  <option :value="300">300 stocks</option>
                  <option :value="500">500 stocks</option>
                </select>
              </div>
              <div class="tg-run-field">
                <label>Optimizer</label>
                <select v-model="runConfig.optimizer">
                  <option value="equal_weight">Equal Weight</option>
                  <option value="mean_variance">Mean-Variance</option>
                  <option value="risk_parity">Risk Parity</option>
                </select>
              </div>
              <div class="tg-run-field">
                <label>Alpha</label>
                <select v-model="runConfig.alpha_method">
                  <option value="equal_weight">Equal Weight</option>
                  <option value="ic_weighted">IC Weighted</option>
                  <option value="icir_weighted">ICIR Weighted</option>
                </select>
              </div>
              <div class="tg-run-field">
                <label>Frequency</label>
                <select v-model="runConfig.rebalance_frequency">
                  <option value="monthly">Monthly</option>
                  <option value="weekly">Weekly</option>
                </select>
              </div>
              <div class="tg-run-field">
                <label>Covariance</label>
                <select v-model="runConfig.covariance_method">
                  <option value="ledoit_wolf">Ledoit-Wolf</option>
                  <option value="sample">Sample</option>
                  <option value="ewma">EWMA</option>
                </select>
              </div>
              <div class="tg-run-field">
                <label>Data Source</label>
                <select v-model="runConfig.data_source">
                  <option value="synthetic">Synthetic (fast)</option>
                  <option value="baostock">Baostock (real A-share)</option>
                </select>
              </div>
            </div>
            <!-- Progress bar during run -->
            <div v-if="running" class="tg-run-progress">
              <div class="tg-run-progress-stages">
                <div
                  v-for="(s, i) in stages" :key="s"
                  :class="['tg-run-stage', stageIdx > i ? 'done' : '', stageIdx === i ? 'active' : '']"
                >{{ s }}</div>
              </div>
              <div class="tg-run-progress-bar">
                <div class="tg-run-progress-fill" :style="{ width: progress + '%' }"></div>
              </div>
              <div class="tg-run-progress-info">
                <span class="tg-run-stage-label">{{ stageLabel }}</span>
                <span class="tg-run-pct">{{ progress }}%</span>
              </div>
            </div>
            <div class="tg-run-actions">
              <button class="btn btn-primary" @click="startRun" :disabled="running">
                <span v-if="!running">&#9654; Run Pipeline</span>
                <span v-else><span class="status-spinner" style="display:inline-block;"></span> Running...</span>
              </button>
              <button class="btn btn-secondary" @click="loadDemo" :disabled="running">&#9889; Demo Data</button>
              <button v-if="running" class="btn btn-danger" @click="cancelRun">Cancel</button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import Panel from './Panel.vue'
import KpiStrip from './KpiStrip.vue'
import FactorHeatmap from './FactorHeatmap.vue'
import RiskGauges from './RiskGauges.vue'
import HoldingsPanel from './HoldingsPanel.vue'
import HoldingsTable from './HoldingsTable.vue'
import ReturnDistribution from './ReturnDistribution.vue'
import FactorScatter from './FactorScatter.vue'
import TurnoverChart from './TurnoverChart.vue'
import AttributionWaterfall from './AttributionWaterfall.vue'
import DrawdownPeriods from './DrawdownPeriods.vue'
import FactorCorrelation from './FactorCorrelation.vue'
import ICDecay from './ICDecay.vue'
import WalkForward from './WalkForward.vue'
import MonteCarlo from './MonteCarlo.vue'
import RiskDecomposition from './RiskDecomposition.vue'
import RegimeDetector from './RegimeDetector.vue'
import RiskMonitor from './RiskMonitor.vue'
import MultiStrategy from './MultiStrategy.vue'
import DataQuality from './DataQuality.vue'
import SystemLog from './SystemLog.vue'
import { runPipeline, getRunStatus, getRunResult, getDemo, createStatusSocket, generateReport } from '../api/index.js'

const emit = defineEmits(['toast'])

// State
const perf = ref(null)
const risk = ref(null)
const factors = ref([])
const exposure = ref(null)
const stressTests = ref([])
const chartData = ref(null)
const logEntries = ref([])
const showRunPanel = ref(false)
const running = ref(false)
const lastRunId = ref('')
const progress = ref(0)
const stage = ref('')
const loading = ref(false)

const stages = ['data', 'factors', 'alpha', 'backtest', 'report', 'done']
const stageLabels = {
  data: 'Loading Data', factors: 'Computing Factors', alpha: 'Generating Alpha',
  backtest: 'Running Backtest', report: 'Building Report', done: 'Complete',
}
const stageIdx = computed(() => stages.indexOf(stage.value))
const stageLabel = computed(() => stageLabels[stage.value] || stage.value)

const runConfig = reactive({
  n_stocks: 300,
  optimizer: 'mean_variance',
  alpha_method: 'icir_weighted',
  rebalance_frequency: 'monthly',
  covariance_method: 'ledoit_wolf',
  start_date: '2021-01-01',
  end_date: '2025-12-31',
  use_tushare: false,
  data_source: 'synthetic',
})

let pollTimer = null
let runId = null
let ws = null
const generatingReport = ref(false)
const reportPath = ref('')

async function downloadReport() {
  if (!lastRunId.value) {
    emit('toast', { message: 'Run a pipeline first', type: 'error' })
    return
  }
  generatingReport.value = true
  try {
    const result = await generateReport(lastRunId.value)
    reportPath.value = result.path
    emit('toast', { message: `Report saved: ${result.path}`, type: 'success' })
    log('info', `HTML report generated: ${result.path}`)
  } catch (e) {
    emit('toast', { message: 'Report generation failed', type: 'error' })
  } finally {
    generatingReport.value = false
  }
}

// Chart refs
const equityRef = ref(null)
const drawdownRef = ref(null)
const sharpeRef = ref(null)
const monthlyRef = ref(null)
const excessRef = ref(null)
let chartInstances = []
let resizeObservers = []

// Logging
function log(level, message) {
  const time = new Date().toLocaleTimeString('en-US', { hour12: false })
  logEntries.value = [...logEntries.value.slice(-199), { time, level, message }]
}

// Pipeline
async function startRun() {
  showRunPanel.value = false
  running.value = true
  loading.value = true
  progress.value = 0
  stage.value = 'data'
  log('info', 'Starting pipeline...')

  try {
    const params = { ...runConfig }
    params.use_baostock = params.data_source === 'baostock'
    params.use_tushare = params.data_source === 'tushare'
    delete params.data_source
    const status = await runPipeline(params)
    runId = status.run_id
    log('info', `Pipeline ${runId} queued`)
    startPolling()
  } catch (e) {
    running.value = false
    loading.value = false
    log('error', `Failed to start: ${e.message}`)
    emit('toast', { message: 'Failed to start pipeline', type: 'error' })
  }
}

function startPolling() {
  let failCount = 0
  pollTimer = setInterval(async () => {
    try {
      const status = await getRunStatus(runId)
      failCount = 0
      progress.value = status.progress
      stage.value = status.stage

      if (status.status === 'completed') {
        clearInterval(pollTimer)
        running.value = false
        log('success', 'Pipeline completed')

        try {
          const data = await getRunResult(runId)
          applyResults(data)
          lastRunId.value = runId
          emit('toast', { message: 'Pipeline completed', type: 'success' })
        } catch (e) {
          loading.value = false
          log('error', `Failed to fetch results: ${e.message}`)
        }
      } else if (status.status === 'failed') {
        clearInterval(pollTimer)
        running.value = false
        loading.value = false
        log('error', `Pipeline failed: ${status.error || 'unknown'}`)
        emit('toast', { message: 'Pipeline failed', type: 'error' })
      }
    } catch {
      failCount++
      if (failCount >= 5) {
        clearInterval(pollTimer)
        running.value = false
        loading.value = false
        log('error', 'Lost connection to server')
      }
    }
  }, 1500)
}

function cancelRun() {
  if (pollTimer) clearInterval(pollTimer)
  running.value = false
  loading.value = false
  log('warn', 'Pipeline cancelled by user')
  emit('toast', { message: 'Pipeline cancelled', type: 'info' })
}

async function loadDemo() {
  showRunPanel.value = false
  loading.value = true
  log('info', 'Loading demo data...')
  try {
    const data = await getDemo()
    applyResults(data)
    log('success', `Demo loaded: ${data.factors?.length || 0} factors, ${data.chart_data?.dates?.length || 0} days`)
    emit('toast', { message: 'Demo data loaded', type: 'success' })
  } catch (e) {
    loading.value = false
    log('error', `Demo failed: ${e.message}`)
    emit('toast', { message: 'Failed to load demo', type: 'error' })
  }
}

function applyResults(data) {
  perf.value = data.performance || null
  risk.value = data.risk || null
  factors.value = data.factors || []
  exposure.value = data.exposure || null
  stressTests.value = data.stress_tests || []
  chartData.value = data.chart_data || null
  loading.value = false

  nextTick(() => renderCharts())
}

// Charts
function disposeCharts() {
  chartInstances.forEach(c => { try { c.dispose() } catch (_) {} })
  chartInstances = []
  resizeObservers.forEach(o => { try { o.disconnect() } catch (_) {} })
  resizeObservers = []
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
  textStyle: { color: '#8b9dc0', fontFamily: 'Inter, -apple-system, sans-serif', fontSize: 10 },
  grid: { left: 48, right: 14, top: 24, bottom: 36 },
  legend: {
    show: true,
    top: 2,
    right: 60,
    textStyle: { color: '#8b9dc0', fontSize: 10 },
    itemWidth: 12,
    itemHeight: 2,
    itemGap: 14,
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(15, 24, 41, 0.95)',
    borderColor: '#1c2d4a',
    borderWidth: 1,
    textStyle: { color: '#e8edf5', fontSize: 11 },
    axisPointer: { type: 'cross', lineStyle: { color: '#243756', type: 'dashed' } },
    extraCssText: 'backdrop-filter: blur(8px); border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);',
  },
}

function makeAxis(dates) {
  return {
    type: 'category', data: dates,
    axisLine: { lineStyle: { color: '#1c2d4a' } },
    axisLabel: { color: '#556882', fontSize: 9, interval: Math.max(1, Math.floor(dates.length / 8)) },
    axisTick: { show: false },
  }
}

function makeValueAxis(fmt) {
  return {
    type: 'value',
    axisLine: { show: false },
    axisLabel: { color: '#556882', fontSize: 9, formatter: fmt || null },
    splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
    axisTick: { show: false },
  }
}

function makeDataZoom(dates) {
  return [
    { type: 'inside', start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true },
    {
      type: 'slider', start: 0, end: 100, height: 16, bottom: 2,
      borderColor: '#1c2d4a', backgroundColor: '#0a1121',
      fillerColor: 'rgba(77,166,255,0.06)',
      handleStyle: { color: '#4da6ff', borderColor: '#4da6ff' },
      textStyle: { color: '#556882', fontSize: 9 },
      dataBackground: { lineStyle: { color: '#1c2d4a' }, areaStyle: { color: 'rgba(77,166,255,0.03)' } },
      selectedDataBackground: { lineStyle: { color: '#4da6ff' }, areaStyle: { color: 'rgba(77,166,255,0.08)' } },
    },
  ]
}

function renderCharts() {
  if (!chartData.value) return
  disposeCharts()
  const cd = chartData.value

  // ── Equity Curve ──
  const eqChart = initChart(equityRef)
  if (eqChart) {
    const series = [{
      name: 'Strategy', type: 'line', data: cd.equity, smooth: 0.3,
      lineStyle: { color: '#4da6ff', width: 1.5 },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: 'rgba(77,166,255,0.15)' },
        { offset: 1, color: 'rgba(77,166,255,0.01)' },
      ])},
      symbol: 'none',
    }]
    if (cd.benchmark?.length) {
      series.push({
        name: 'Benchmark', type: 'line', data: cd.benchmark, smooth: 0.3,
        lineStyle: { color: '#3a4d6a', width: 1, type: 'dashed' },
        symbol: 'none',
      })
    }
    eqChart.setOption({
      ...chartBase,
      xAxis: makeAxis(cd.dates),
      yAxis: makeValueAxis(v => v.toFixed(2)),
      dataZoom: makeDataZoom(cd.dates),
      series,
    })
  }

  // ── Drawdown ──
  const ddChart = initChart(drawdownRef)
  if (ddChart && cd.drawdown?.length) {
    ddChart.setOption({
      ...chartBase,
      legend: { show: false },
      xAxis: makeAxis(cd.dates),
      yAxis: { ...makeValueAxis(v => v + '%'), max: 0 },
      dataZoom: makeDataZoom(cd.dates),
      series: [{
        name: 'Drawdown', type: 'line',
        data: cd.drawdown.map(v => (Number(v) * 100).toFixed(2)),
        smooth: 0.3,
        lineStyle: { color: '#f87171', width: 1 },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(248,113,113,0.2)' },
          { offset: 1, color: 'rgba(248,113,113,0.0)' },
        ])},
        symbol: 'none',
      }],
      tooltip: { ...chartBase.tooltip, valueFormatter: v => v + '%' },
    })
  }

  // ── Rolling Sharpe ──
  const rsChart = initChart(sharpeRef)
  if (rsChart && cd.rolling_sharpe?.length) {
    rsChart.setOption({
      ...chartBase,
      legend: { show: false },
      grid: { left: 48, right: 14, top: 10, bottom: 28 },
      xAxis: makeAxis(cd.rolling_sharpe_dates || []),
      yAxis: makeValueAxis(v => v.toFixed(1)),
      dataZoom: makeDataZoom(cd.rolling_sharpe_dates || []),
      series: [{
        name: 'Sharpe', type: 'line',
        data: cd.rolling_sharpe.map(v => Number(v).toFixed(2)),
        smooth: 0.3,
        lineStyle: { color: '#fbbf24', width: 1.5 },
        symbol: 'none',
        markLine: {
          silent: true,
          data: [
            { yAxis: 0, lineStyle: { color: '#3a4d6a', type: 'dashed' }, label: { show: false } },
            { yAxis: 1, lineStyle: { color: 'rgba(52,211,153,0.3)', type: 'dotted' }, label: { show: false } },
          ],
          symbol: 'none',
        },
      }],
    })
  }

  // ── Excess Cumulative Return ──
  const exChart = initChart(excessRef)
  if (exChart && cd.excess_cumulative?.length) {
    exChart.setOption({
      ...chartBase,
      legend: { show: false },
      grid: { left: 48, right: 14, top: 10, bottom: 28 },
      xAxis: makeAxis(cd.dates),
      yAxis: makeValueAxis(v => v.toFixed(3)),
      dataZoom: makeDataZoom(cd.dates),
      series: [{
        name: 'Excess', type: 'line',
        data: cd.excess_cumulative.map(v => Number(v).toFixed(4)),
        smooth: 0.3,
        lineStyle: { color: '#34d399', width: 1.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(52,211,153,0.15)' },
            { offset: 1, color: 'rgba(52,211,153,0.01)' },
          ]),
        },
        symbol: 'none',
        markLine: {
          silent: true,
          data: [{ yAxis: 1, lineStyle: { color: '#3a4d6a', type: 'dashed' }, label: { show: false } }],
          symbol: 'none',
        },
      }],
      tooltip: { ...chartBase.tooltip, valueFormatter: v => v },
    })
  }

  // ── Monthly Returns Heatmap ──
  const mChart = initChart(monthlyRef)
  if (mChart && cd.monthly_returns) {
    const mr = cd.monthly_returns
    const years = mr.years || []
    const months = mr.months || []
    const data = []
    let minVal = Infinity, maxVal = -Infinity

    for (let yi = 0; yi < years.length; yi++) {
      for (let mi = 0; mi < months.length; mi++) {
        const val = mr.data?.[yi]?.[mi]
        if (val != null && !isNaN(val)) {
          const pct = (val * 100)
          data.push([mi, yi, Number(pct.toFixed(2))])
          minVal = Math.min(minVal, pct)
          maxVal = Math.max(maxVal, pct)
        }
      }
    }

    const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 1)

    mChart.setOption({
      ...chartBase,
      legend: { show: false },
      grid: { left: 48, right: 14, top: 10, bottom: 28 },
      xAxis: {
        type: 'category',
        data: months.map(m => ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1] || m),
        axisLine: { lineStyle: { color: '#1c2d4a' } },
        axisLabel: { color: '#556882', fontSize: 9 },
        axisTick: { show: false },
        splitArea: { show: false },
      },
      yAxis: {
        type: 'category',
        data: years,
        axisLine: { lineStyle: { color: '#1c2d4a' } },
        axisLabel: { color: '#556882', fontSize: 9 },
        axisTick: { show: false },
        splitArea: { show: false },
      },
      visualMap: {
        min: -absMax,
        max: absMax,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        show: false,
        inRange: {
          color: ['#dc2626', '#991b1b', '#450a0a', '#0a0a0a', '#052e16', '#166534', '#22c55e'],
        },
      },
      series: [{
        name: 'Monthly Return',
        type: 'heatmap',
        data,
        emphasis: { itemStyle: { borderColor: '#e8edf5', borderWidth: 1 } },
        label: {
          show: true,
          fontSize: 9,
          fontFamily: 'JetBrains Mono, monospace',
          color: '#e8edf5',
          formatter: p => p.value[2] != null ? p.value[2].toFixed(1) : '',
        },
      }],
      tooltip: {
        ...chartBase.tooltip,
        formatter: p => {
          const m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][p.value[0]]
          return `<b>${m} ${years[p.value[1]]}</b><br/>Return: ${p.value[2]}%`
        },
      },
    })
  }

  // Link all charts for synchronized zooming
  if (chartInstances.length > 1) {
    echarts.connect(chartInstances)
  }
}

// Keyboard shortcuts
function onKeydown(e) {
  // Don't trigger shortcuts when typing in inputs
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return

  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault()
    showRunPanel.value = !showRunPanel.value
    return
  }
  if (e.key === 'Escape') {
    showRunPanel.value = false
    return
  }

  switch (e.key.toLowerCase()) {
    case 'r':
      e.preventDefault()
      showRunPanel.value = true
      break
    case 'd':
      e.preventDefault()
      loadDemo()
      break
    case 'e':
      e.preventDefault()
      exportResults()
      break
  }
}

function exportResults() {
  if (!perf.value) {
    emit('toast', { message: 'No results to export', type: 'info' })
    return
  }
  const data = JSON.stringify({ performance: perf.value, risk: risk.value, factors: factors.value, exposure: exposure.value }, null, 2)
  const blob = new Blob([data], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `backtest-${new Date().toISOString().slice(0,10)}.json`
  a.click()
  URL.revokeObjectURL(url)
  log('info', 'Results exported')
  emit('toast', { message: 'Results exported', type: 'success' })
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  log('info', 'Quant Terminal initialized')
  log('info', 'Shortcuts: R=Run, D=Demo, E=Export, Ctrl+K=Command')
  log('info', 'Click Demo Data or press D to see results instantly')

  // WebSocket for real-time pipeline status
  try {
    ws = createStatusSocket(
      (data) => {
        if (data.type === 'status' && data.run_id === runId) {
          progress.value = data.progress || 0
          stage.value = data.stage || ''
          if (data.status === 'completed') {
            running.value = false
            log('success', 'Pipeline completed (WS)')
            getRunResult(runId).then(result => {
              applyResults(result)
              lastRunId.value = runId
              emit('toast', { message: 'Pipeline completed', type: 'success' })
            }).catch(e => {
              loading.value = false
              log('error', `Failed to fetch results: ${e.message}`)
            })
          } else if (data.status === 'failed') {
            running.value = false
            loading.value = false
            log('error', `Pipeline failed: ${data.error || 'unknown'}`)
            emit('toast', { message: 'Pipeline failed', type: 'error' })
          }
        }
      },
      () => { log('warn', 'WebSocket connection failed, using polling') }
    )
  } catch (_) {}
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (ws) { try { ws.close() } catch (_) {} }
  disposeCharts()
  window.removeEventListener('keydown', onKeydown)
})

defineExpose({ startRun, loadDemo, showRunPanel, exportResults })
</script>

<style scoped>
.terminal-grid {
  display: flex;
  flex-direction: column;
  gap: 3px;
  height: calc(100vh - 64px);
  padding: 3px;
  overflow: hidden;
}

/* Row layouts */
.tg-row-main {
  display: grid;
  grid-template-columns: 3fr 2fr;
  gap: 3px;
  flex: 2.5;
  min-height: 0;
}

.tg-row-secondary {
  display: grid;
  grid-template-columns: 5fr 4fr 4fr;
  gap: 3px;
  flex: 1.8;
  min-height: 0;
}

.tg-row-bottom {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3px;
  flex: 1.3;
  min-height: 0;
}

.tg-row-analytics {
  display: grid;
  grid-template-columns: 5fr 4fr 5fr;
  gap: 3px;
  flex: 1.5;
  min-height: 0;
}

.tg-row-research {
  display: grid;
  grid-template-columns: 5fr 4fr 4fr;
  gap: 3px;
  flex: 1.5;
  min-height: 0;
}

.tg-row-dd {
  flex: 0.6;
  min-height: 0;
}

.tg-row-corr {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3px;
  flex: 1.5;
  min-height: 0;
}

.tg-row-advanced {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 3px;
  flex: 2;
  min-height: 0;
}

.tg-row-regime {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 3px;
  flex: 1.2;
  min-height: 0;
}

.tg-row-multi {
  display: grid;
  grid-template-columns: 3fr 2fr;
  gap: 3px;
  flex: 1.8;
  min-height: 0;
}

.tg-row-report {
  flex: 0.4;
  min-height: 0;
}

.tg-report-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 100%;
  padding: 0 12px;
}

.tg-report-label {
  flex: 1;
  font-size: 10px;
  color: #6b7a8d;
}

.tg-report-btn {
  padding: 6px 16px;
  border-radius: 4px;
  border: 1px solid #fbbf24;
  background: transparent;
  color: #fbbf24;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}

.tg-report-btn:hover { background: rgba(251,191,36,0.1); }
.tg-report-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.tg-report-spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid #fbbf24;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

.tg-report-path {
  font-size: 10px;
  color: #22c55e;
  font-family: var(--font-mono);
}

.tg-row-log {
  flex: 0.6;
  min-height: 0;
}

/* Chart containers */
.tg-chart {
  width: 100%;
  height: 100%;
  min-height: 0;
}

.tg-chart-sm {
  width: 100%;
  height: 100%;
  min-height: 0;
}

/* Skeleton loaders */
.tg-skeleton-chart {
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.8s ease-in-out infinite;
  border-radius: var(--radius);
}

.tg-skeleton-chart-sm {
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.8s ease-in-out infinite;
  border-radius: var(--radius);
}

.tg-skeleton-block {
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.8s ease-in-out infinite;
  border-radius: var(--radius);
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Panel tags */
.panel-tag {
  font-size: 9px;
  font-weight: 600;
  font-family: var(--font-mono);
  color: var(--green);
  font-variant-numeric: tabular-nums;
}

.panel-tag.neg { color: var(--red); }

/* Run overlay */
.tg-run-overlay {
  position: fixed;
  inset: 0;
  background: rgba(6, 11, 20, 0.85);
  backdrop-filter: blur(10px);
  z-index: 500;
  display: flex;
  align-items: center;
  justify-content: center;
}

.tg-run-panel {
  background: var(--bg-card);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  width: 680px;
  max-width: 92vw;
  box-shadow: 0 16px 64px rgba(0,0,0,0.5);
}

.tg-run-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 1px;
  text-transform: uppercase;
}

.tg-close-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 20px;
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
  transition: color 0.15s;
}

.tg-close-btn:hover { color: var(--text-primary); }

.tg-run-body { padding: 16px; }

.tg-run-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  margin-bottom: 14px;
}

.tg-run-field label {
  display: block;
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.5px;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.tg-run-field select {
  width: 100%;
  padding: 7px 8px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--text-primary);
  font-size: 12px;
  font-family: inherit;
  transition: border-color 0.15s;
}

.tg-run-field select:focus { outline: none; border-color: var(--accent); }
.tg-run-field select option { background: var(--bg-secondary); color: var(--text-primary); }

/* Progress during run */
.tg-run-progress {
  margin-bottom: 14px;
}

.tg-run-progress-stages {
  display: flex;
  gap: 3px;
  margin-bottom: 6px;
}

.tg-run-stage {
  flex: 1;
  padding: 4px 0;
  text-align: center;
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.5px;
  text-transform: uppercase;
  background: var(--bg-secondary);
  border-radius: 4px;
  transition: all 0.3s ease;
}

.tg-run-stage.done {
  background: rgba(77, 166, 255, 0.1);
  color: var(--accent-dim);
}

.tg-run-stage.active {
  background: rgba(77, 166, 255, 0.15);
  color: var(--accent);
  box-shadow: 0 0 8px rgba(77, 166, 255, 0.2);
}

.tg-run-progress-bar {
  height: 3px;
  background: var(--bg-input);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 6px;
}

.tg-run-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-dim), var(--accent), var(--accent-bright));
  border-radius: 2px;
  transition: width 0.5s ease;
  box-shadow: 0 0 8px rgba(77, 166, 255, 0.3);
}

.tg-run-progress-info {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--text-muted);
}

.tg-run-stage-label {
  font-weight: 600;
  text-transform: capitalize;
}

.tg-run-pct {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}

.tg-run-actions {
  display: flex;
  gap: 8px;
}

/* Fade transition */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

@media (max-width: 1024px) {
  .tg-row-main { grid-template-columns: 1fr; }
  .tg-row-secondary { grid-template-columns: 1fr; }
  .tg-row-analytics { grid-template-columns: 1fr; }
  .tg-row-research { grid-template-columns: 1fr; }
  .tg-row-bottom { grid-template-columns: 1fr; }
  .tg-row-corr { grid-template-columns: 1fr; }
  .tg-row-advanced { grid-template-columns: 1fr; }
  .tg-row-regime { grid-template-columns: 1fr; }
  .terminal-grid { height: auto; min-height: 100vh; overflow-y: auto; }
  .tg-run-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
