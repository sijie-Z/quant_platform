<template>
  <div class="rd-view">
    <div class="rd-header">
      <div class="rd-title">
        <span class="rd-dot"></span>
        FACTOR RISK DECOMPOSITION
      </div>
      <div class="rd-actions">
        <button class="btn btn-sm btn-primary" @click="decompose" :disabled="loading">
          {{ loading ? 'Analyzing...' : 'Decompose Risk' }}
        </button>
      </div>
    </div>

    <template v-if="result">
      <!-- Summary -->
      <div class="rd-summary">
        <div class="rd-card">
          <div class="rd-card-label">Total Risk</div>
          <div class="rd-card-value">{{ result.total_risk_pct?.toFixed(1) }}%</div>
        </div>
        <div class="rd-card">
          <div class="rd-card-label">Factor Risk</div>
          <div class="rd-card-value rd-accent">{{ result.factor_risk_pct?.toFixed(1) }}%</div>
        </div>
        <div class="rd-card">
          <div class="rd-card-label">Idiosyncratic</div>
          <div class="rd-card-value">{{ result.idiosyncratic_risk_pct?.toFixed(1) }}%</div>
        </div>
        <div class="rd-card">
          <div class="rd-card-label">R-Squared</div>
          <div class="rd-card-value rd-accent">{{ (result.r_squared * 100)?.toFixed(1) }}%</div>
        </div>
      </div>

      <!-- Risk Breakdown Chart -->
      <div ref="chartRef" class="rd-chart"></div>

      <!-- Factor Table -->
      <div class="rd-table-wrap">
        <table class="rd-tbl">
          <thead>
            <tr>
              <th>Factor</th>
              <th>Risk Share %</th>
              <th>Return (bps)</th>
              <th>Beta</th>
              <th>t-stat</th>
              <th>Type</th>
              <th>Risk Bar</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="f in result.factors" :key="f.factor" class="rd-row">
              <td class="rd-factor">{{ f.factor }}</td>
              <td class="rd-mono">{{ f.risk_share_pct?.toFixed(1) }}%</td>
              <td :class="['rd-mono', f.annual_return_bps > 0 ? 'rd-pos' : 'rd-neg']">
                {{ f.annual_return_bps > 0 ? '+' : '' }}{{ f.annual_return_bps?.toFixed(0) }}
              </td>
              <td class="rd-mono">{{ f.beta?.toFixed(3) }}</td>
              <td class="rd-mono">{{ f.t_stat?.toFixed(2) }}</td>
              <td>
                <span :class="['rd-type', f.contribution === 'alpha' ? 'rd-type-alpha' : 'rd-type-factor']">
                  {{ f.contribution }}
                </span>
              </td>
              <td class="rd-bar-cell">
                <div class="rd-bar-wrap">
                  <div
                    class="rd-bar-fill"
                    :class="f.contribution === 'alpha' ? 'rd-bar-alpha' : 'rd-bar-factor'"
                    :style="{ width: Math.min(f.risk_share_pct * 3, 100) + '%' }"
                  ></div>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>

    <!-- Empty State -->
    <div v-if="!result && !loading" class="rd-empty">
      <div class="rd-empty-icon">&#9670;</div>
      <h3>Factor Risk Decomposition</h3>
      <p>Run a pipeline first, then decompose portfolio risk into systematic factor exposure and idiosyncratic alpha.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import { decomposeRisk } from '../api/index.js'

const emit = defineEmits(['toast'])
const props = defineProps({ runId: { type: String, default: '' } })

const loading = ref(false)
const result = ref(null)
const chartRef = ref(null)
let chart = null
let resizeObs = null

async function decompose() {
  if (!props.runId) {
    emit('toast', { message: 'Run a pipeline first', type: 'error' })
    return
  }
  loading.value = true
  try {
    result.value = await decomposeRisk({ run_id: props.runId })
    await nextTick()
    renderChart()
    emit('toast', { message: 'Risk decomposition completed', type: 'success' })
  } catch (e) {
    emit('toast', { message: `Decomposition failed: ${e.response?.data?.detail || e.message}`, type: 'error' })
  } finally {
    loading.value = false
  }
}

function renderChart() {
  if (!chartRef.value || !result.value?.factors?.length) return

  if (!chart) {
    chart = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  }

  const factors = result.value.factors.filter(f => f.risk_share_pct > 0.5)
  const names = factors.map(f => f.factor)
  const values = factors.map(f => f.risk_share_pct)
  const colors = factors.map(f => f.contribution === 'alpha' ? '#a78bfa' : '#4da6ff')

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const p = params[0]
        return `${p.name}<br/>Risk Share: <b>${p.value.toFixed(1)}%</b>`
      },
    },
    grid: { top: 8, right: 16, bottom: 40, left: 100 },
    xAxis: {
      type: 'value',
      axisLabel: { color: '#8892a4', fontSize: 9, formatter: v => v + '%' },
      splitLine: { lineStyle: { color: '#1e2a3a' } },
    },
    yAxis: {
      type: 'category',
      data: names.reverse(),
      axisLabel: { color: '#8892a4', fontSize: 9 },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    series: [{
      type: 'bar',
      data: values.reverse().map((v, i) => ({
        value: v,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: colors[names.length - 1 - i] || '#4da6ff' },
            { offset: 1, color: 'rgba(77,166,255,0.3)' },
          ]),
        },
      })),
      label: {
        show: true,
        position: 'right',
        formatter: '{c}%',
        fontSize: 9,
        color: '#8892a4',
      },
    }],
  })
}

onMounted(() => {
  resizeObs = new ResizeObserver(() => chart?.resize())
  if (chartRef.value) resizeObs.observe(chartRef.value)
})
onBeforeUnmount(() => {
  resizeObs?.disconnect()
  chart?.dispose()
})
</script>

<style scoped>
.rd-view { display: flex; flex-direction: column; gap: 12px; height: 100%; }
.rd-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.rd-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; letter-spacing: 0.5px; }
.rd-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 6px rgba(77,166,255,0.4); }
.rd-actions { display: flex; gap: 8px; }
.rd-summary { display: flex; gap: 8px; flex-wrap: wrap; flex-shrink: 0; }
.rd-card { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 8px 12px; text-align: center; min-width: 100px; }
.rd-card-label { font-size: 9px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
.rd-card-value { font-size: 16px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.rd-accent { color: var(--accent); }
.rd-pos { color: var(--green); }
.rd-neg { color: var(--red); }
.rd-chart { flex: 0.8; min-height: 120px; }
.rd-table-wrap { flex: 1; overflow: auto; min-height: 0; border: 1px solid var(--border); border-radius: 6px; }
.rd-tbl { width: 100%; border-collapse: collapse; font-size: 10.5px; font-variant-numeric: tabular-nums; }
.rd-tbl th { padding: 6px 8px; font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.6px; text-align: left; border-bottom: 1px solid var(--border); background: var(--bg-card); position: sticky; top: 0; }
.rd-tbl td { padding: 4px 8px; border-bottom: 1px solid var(--border-subtle); color: var(--text-secondary); white-space: nowrap; }
.rd-row:hover td { background: rgba(77,166,255,0.04); }
.rd-factor { font-family: var(--font-mono); font-weight: 600; font-size: 10px; }
.rd-mono { font-family: var(--font-mono); }
.rd-type { font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 3px; text-transform: uppercase; }
.rd-type-factor { color: var(--accent); background: rgba(77,166,255,0.1); }
.rd-type-alpha { color: #a78bfa; background: rgba(167,139,250,0.1); }
.rd-bar-cell { width: 100px; }
.rd-bar-wrap { height: 4px; background: var(--bg-input); border-radius: 2px; overflow: hidden; }
.rd-bar-fill { height: 100%; border-radius: 2px; transition: width 0.4s ease; min-width: 2px; }
.rd-bar-factor { background: linear-gradient(90deg, var(--accent-dim), var(--accent)); }
.rd-bar-alpha { background: linear-gradient(90deg, rgba(167,139,250,0.3), #a78bfa); }
.rd-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--text-muted); }
.rd-empty-icon { font-size: 48px; opacity: 0.2; }
.rd-empty h3 { font-size: 14px; color: var(--text-secondary); }
.rd-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }
</style>
