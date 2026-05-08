<template>
  <div class="wf-view">
    <div class="wf-header">
      <div class="wf-title">
        <span class="wf-dot"></span>
        WALK-FORWARD VALIDATION
      </div>
      <div class="wf-actions">
        <select v-model="wfMode" class="wf-select">
          <option value="rolling">Rolling Window</option>
          <option value="expanding">Expanding Window</option>
        </select>
        <select v-model.number="trainPeriod" class="wf-select">
          <option :value="252">1Y Train</option>
          <option :value="504">2Y Train</option>
          <option :value="756">3Y Train</option>
        </select>
        <select v-model.number="testPeriod" class="wf-select">
          <option :value="63">3M Test</option>
          <option :value="126">6M Test</option>
          <option :value="252">1Y Test</option>
        </select>
        <button class="btn btn-sm btn-primary" @click="runWF" :disabled="loading">
          {{ loading ? 'Running...' : 'Run Walk-Forward' }}
        </button>
      </div>
    </div>

    <!-- Stability Summary -->
    <div class="wf-summary" v-if="result">
      <div class="wf-card">
        <div class="wf-card-label">Folds</div>
        <div class="wf-card-value">{{ result.n_folds }}</div>
      </div>
      <div class="wf-card">
        <div class="wf-card-label">Mean Sharpe</div>
        <div :class="['wf-card-value', result.stability?.mean_sharpe > 0 ? 'wf-pos' : 'wf-neg']">
          {{ result.stability?.mean_sharpe?.toFixed(2) }}
        </div>
      </div>
      <div class="wf-card">
        <div class="wf-card-label">Sharpe Std</div>
        <div class="wf-card-value">{{ result.stability?.std_sharpe?.toFixed(3) }}</div>
      </div>
      <div class="wf-card">
        <div class="wf-card-label">Consistency</div>
        <div :class="['wf-card-value', result.stability?.sharpe_consistency > 0.6 ? 'wf-pos' : 'wf-warn']">
          {{ (result.stability?.sharpe_consistency * 100)?.toFixed(0) }}%
        </div>
      </div>
      <div class="wf-card">
        <div class="wf-card-label">Positive Folds</div>
        <div class="wf-card-value wf-pos">
          {{ result.stability?.positive_folds }}/{{ result.stability?.total_folds }}
        </div>
      </div>
      <div class="wf-card">
        <div class="wf-card-label">Mean Return</div>
        <div :class="['wf-card-value', result.stability?.mean_return_pct > 0 ? 'wf-pos' : 'wf-neg']">
          {{ result.stability?.mean_return_pct?.toFixed(1) }}%
        </div>
      </div>
    </div>

    <!-- OOS Equity Chart -->
    <div class="wf-chart-section" v-if="result">
      <div ref="oosChartRef" class="wf-chart"></div>
    </div>

    <!-- Fold Details Table -->
    <div class="wf-table-wrap" v-if="result?.fold_details?.length">
      <table class="wf-tbl">
        <thead>
          <tr>
            <th>Fold</th>
            <th>Train Period</th>
            <th>Test Period</th>
            <th>OOS Days</th>
            <th>Sharpe</th>
            <th>Return %</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="f in result.fold_details" :key="f.fold" class="wf-row">
            <td class="wf-rank">{{ f.fold + 1 }}</td>
            <td class="wf-dim">{{ f.train }}</td>
            <td class="wf-dim">{{ f.test }}</td>
            <td class="wf-mono">{{ f.oos_days }}</td>
            <td :class="['wf-mono', f.sharpe > 0 ? 'wf-pos' : 'wf-neg']">
              {{ f.sharpe?.toFixed(2) }}
            </td>
            <td :class="['wf-mono', f.return_pct > 0 ? 'wf-pos' : 'wf-neg']">
              {{ f.return_pct > 0 ? '+' : '' }}{{ f.return_pct?.toFixed(2) }}%
            </td>
            <td>
              <span :class="['wf-badge', f.sharpe > 0.5 ? 'wf-badge-good' : f.sharpe > 0 ? 'wf-badge-ok' : 'wf-badge-bad']">
                {{ f.sharpe > 0.5 ? 'STRONG' : f.sharpe > 0 ? 'WEAK' : 'FAIL' }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Empty State -->
    <div v-if="!result && !loading" class="wf-empty">
      <div class="wf-empty-icon">&#9878;</div>
      <h3>Walk-Forward Validation</h3>
      <p>Run a pipeline first, then validate with out-of-sample testing. This is the gold standard for avoiding overfitting.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
import * as echarts from 'echarts'
import { runWalkForward } from '../api/index.js'

const emit = defineEmits(['toast'])
const props = defineProps({ runId: { type: String, default: '' } })

const loading = ref(false)
const result = ref(null)
const wfMode = ref('rolling')
const trainPeriod = ref(504)
const testPeriod = ref(126)
const oosChartRef = ref(null)
let oosChart = null
let resizeObs = null

async function runWF() {
  if (!props.runId) {
    emit('toast', { message: 'Run a pipeline first', type: 'error' })
    return
  }
  loading.value = true
  try {
    result.value = await runWalkForward({
      run_id: props.runId,
      mode: wfMode.value,
      train_period: trainPeriod.value,
      test_period: testPeriod.value,
    })
    await nextTick()
    renderOOSChart()
    emit('toast', { message: `Walk-forward: ${result.value.n_folds} folds completed`, type: 'success' })
  } catch (e) {
    emit('toast', { message: `Walk-forward failed: ${e.response?.data?.detail || e.message}`, type: 'error' })
  } finally {
    loading.value = false
  }
}

function renderOOSChart() {
  if (!oosChartRef.value || !result.value?.oos_equity?.length) return

  if (!oosChart) {
    oosChart = echarts.init(oosChartRef.value, null, { renderer: 'canvas' })
  }

  oosChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { top: 16, right: 16, bottom: 28, left: 50 },
    xAxis: {
      type: 'category',
      data: result.value.oos_dates,
      axisLabel: { color: '#8892a4', fontSize: 9 },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#8892a4', fontSize: 9, formatter: v => v.toFixed(2) },
      splitLine: { lineStyle: { color: '#1e2a3a' } },
    },
    series: [{
      type: 'line',
      data: result.value.oos_equity,
      smooth: 0.2,
      symbol: 'none',
      lineStyle: { color: '#4da6ff', width: 1.5 },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(77,166,255,0.2)' },
          { offset: 1, color: 'rgba(77,166,255,0.01)' },
        ]),
      },
    }],
  })
}

onMounted(() => {
  resizeObs = new ResizeObserver(() => oosChart?.resize())
  if (oosChartRef.value) resizeObs.observe(oosChartRef.value)
})
onBeforeUnmount(() => {
  resizeObs?.disconnect()
  oosChart?.dispose()
})
</script>

<style scoped>
.wf-view { display: flex; flex-direction: column; gap: 12px; height: 100%; }
.wf-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; flex-wrap: wrap; gap: 8px; }
.wf-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; letter-spacing: 0.5px; }
.wf-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 6px rgba(52,211,153,0.5); }
.wf-actions { display: flex; gap: 8px; align-items: center; }
.wf-select { background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 10px; padding: 4px 8px; }
.wf-summary { display: flex; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }
.wf-card { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 8px 12px; text-align: center; min-width: 100px; }
.wf-card-label { font-size: 9px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.wf-card-value { font-size: 16px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.wf-pos { color: var(--green); }
.wf-neg { color: var(--red); }
.wf-warn { color: var(--orange); }
.wf-chart-section { flex: 1; min-height: 120px; }
.wf-chart { width: 100%; height: 100%; }
.wf-table-wrap { flex: 1; overflow: auto; min-height: 0; border: 1px solid var(--border); border-radius: 6px; }
.wf-tbl { width: 100%; border-collapse: collapse; font-size: 10.5px; font-variant-numeric: tabular-nums; }
.wf-tbl th { padding: 6px 8px; font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.6px; text-align: left; border-bottom: 1px solid var(--border); background: var(--bg-card); position: sticky; top: 0; }
.wf-tbl td { padding: 4px 8px; border-bottom: 1px solid var(--border-subtle); color: var(--text-secondary); white-space: nowrap; }
.wf-row:hover td { background: rgba(77,166,255,0.04); }
.wf-rank { text-align: center; color: var(--text-dim); font-size: 9px; }
.wf-mono { font-family: var(--font-mono); }
.wf-dim { color: var(--text-dim); font-size: 9px; }
.wf-badge { font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 3px; letter-spacing: 0.3px; }
.wf-badge-good { color: var(--green); background: rgba(52,211,153,0.1); }
.wf-badge-ok { color: var(--orange); background: rgba(251,191,36,0.1); }
.wf-badge-bad { color: var(--red); background: rgba(239,68,68,0.1); }
.wf-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--text-muted); }
.wf-empty-icon { font-size: 48px; opacity: 0.2; }
.wf-empty h3 { font-size: 14px; color: var(--text-secondary); }
.wf-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }
</style>
