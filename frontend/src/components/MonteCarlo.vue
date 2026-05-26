<template>
  <div class="mc-view">
    <div class="mc-header">
      <div class="mc-title">
        <span class="mc-dot"></span>
        {{ locale === 'zh-CN' ? '蒙特卡洛模拟' : 'MONTE CARLO SIMULATION' }}
      </div>
      <div class="mc-actions">
        <select v-model="mcMethod" class="mc-select">
          <option value="bootstrap">{{ locale === 'zh-CN' ? '分块自助法' : 'Block Bootstrap' }}</option>
          <option value="parametric">{{ locale === 'zh-CN' ? '学生t参数化' : 'Student-t Parametric' }}</option>
        </select>
        <select v-model.number="nSims" class="mc-select">
          <option :value="500">{{ locale === 'zh-CN' ? '500次模拟' : '500 sims' }}</option>
          <option :value="1000">{{ locale === 'zh-CN' ? '1000次模拟' : '1000 sims' }}</option>
          <option :value="5000">{{ locale === 'zh-CN' ? '5000次模拟' : '5000 sims' }}</option>
        </select>
        <button class="btn btn-sm btn-primary" @click="runMC" :disabled="loading">
          {{ loading ? (locale === 'zh-CN' ? '模拟中...' : 'Simulating...') : (locale === 'zh-CN' ? '运行模拟' : 'Run Simulation') }}
        </button>
      </div>
    </div>

    <!-- Results Grid -->
    <div class="mc-grid" v-if="result">
      <!-- Terminal Value -->
      <div class="mc-panel">
        <div class="mc-panel-title">{{ locale === 'zh-CN' ? '终值（1年）' : 'TERMINAL VALUE (1Y)' }}</div>
        <div class="mc-stat-row">
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '均值' : 'Mean' }}</span>
            <span class="mc-stat-value">{{ result.terminal_value?.mean?.toFixed(3) }}</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '中位数' : 'Median' }}</span>
            <span class="mc-stat-value">{{ result.terminal_value?.median?.toFixed(3) }}</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '5%分位' : '5th %ile' }}</span>
            <span class="mc-stat-value mc-neg">{{ result.terminal_value?.p5?.toFixed(3) }}</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '95%分位' : '95th %ile' }}</span>
            <span class="mc-stat-value mc-pos">{{ result.terminal_value?.p95?.toFixed(3) }}</span>
          </div>
        </div>
      </div>

      <!-- Annual Return -->
      <div class="mc-panel">
        <div class="mc-panel-title">{{ locale === 'zh-CN' ? '年化收益' : 'ANNUAL RETURN' }}</div>
        <div class="mc-stat-row">
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '均值' : 'Mean' }}</span>
            <span :class="['mc-stat-value', result.annual_return?.mean > 0 ? 'mc-pos' : 'mc-neg']">
              {{ (result.annual_return?.mean * 100)?.toFixed(1) }}%
            </span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? 'P(>0)' : 'P(>0)' }}</span>
            <span class="mc-stat-value">{{ (result.annual_return?.prob_positive * 100)?.toFixed(0) }}%</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? 'P(>10%)' : 'P(>10%)' }}</span>
            <span class="mc-stat-value">{{ (result.annual_return?.prob_10pct * 100)?.toFixed(0) }}%</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '5%分位' : '5th %ile' }}</span>
            <span class="mc-stat-value mc-neg">{{ (result.annual_return?.p5 * 100)?.toFixed(1) }}%</span>
          </div>
        </div>
      </div>

      <!-- Max Drawdown -->
      <div class="mc-panel">
        <div class="mc-panel-title">{{ locale === 'zh-CN' ? '最大回撤' : 'MAX DRAWDOWN' }}</div>
        <div class="mc-stat-row">
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '均值' : 'Mean' }}</span>
            <span class="mc-stat-value mc-neg">{{ (result.max_drawdown?.mean * 100)?.toFixed(1) }}%</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '最差' : 'Worst' }}</span>
            <span class="mc-stat-value mc-neg">{{ (result.max_drawdown?.worst * 100)?.toFixed(1) }}%</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? 'P(>20%)' : 'P(>20%)' }}</span>
            <span class="mc-stat-value mc-warn">{{ (result.max_drawdown?.prob_20pct * 100)?.toFixed(0) }}%</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? 'P(>30%)' : 'P(>30%)' }}</span>
            <span class="mc-stat-value mc-neg">{{ (result.max_drawdown?.prob_30pct * 100)?.toFixed(0) }}%</span>
          </div>
        </div>
      </div>

      <!-- Sharpe -->
      <div class="mc-panel" v-if="result.sharpe?.mean != null">
        <div class="mc-panel-title">{{ locale === 'zh-CN' ? '夏普比率' : 'SHARPE RATIO' }}</div>
        <div class="mc-stat-row">
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '均值' : 'Mean' }}</span>
            <span :class="['mc-stat-value', result.sharpe?.mean > 0 ? 'mc-pos' : 'mc-neg']">
              {{ result.sharpe?.mean?.toFixed(2) }}
            </span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '中位数' : 'Median' }}</span>
            <span class="mc-stat-value">{{ result.sharpe?.median?.toFixed(2) }}</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? 'P(>0)' : 'P(>0)' }}</span>
            <span class="mc-stat-value">{{ (result.sharpe?.prob_positive * 100)?.toFixed(0) }}%</span>
          </div>
          <div class="mc-stat">
            <span class="mc-stat-label">{{ locale === 'zh-CN' ? '标准差' : 'Std Dev' }}</span>
            <span class="mc-stat-value">{{ result.sharpe?.std?.toFixed(3) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Distribution Chart -->
    <div class="mc-chart-section" v-if="result">
      <div ref="distChartRef" class="mc-chart"></div>
    </div>

    <!-- Fitted Distribution Info -->
    <div class="mc-fit-info" v-if="result?.fitted_distribution">
      <span class="mc-fit-label">{{ locale === 'zh-CN' ? '拟合分布：' : 'Fitted:' }}</span>
      <span class="mc-fit-value">{{ locale === 'zh-CN' ? `学生t（自由度=${result.fitted_distribution.df}，位置=${result.fitted_distribution.loc}，尺度=${result.fitted_distribution.scale}）` : `Student-t (df=${result.fitted_distribution.df}, loc=${result.fitted_distribution.loc}, scale=${result.fitted_distribution.scale})` }}</span>
    </div>

    <!-- Empty State -->
    <div v-if="!result && !loading" class="mc-empty">
      <div class="mc-empty-icon">&#9858;</div>
      <h3>{{ locale === 'zh-CN' ? '蒙特卡洛模拟' : 'Monte Carlo Simulation' }}</h3>
      <p>{{ locale === 'zh-CN' ? '请先运行流水线，然后模拟数千种情景以估计置信区间和尾部风险。' : 'Run a pipeline first, then simulate thousands of scenarios to estimate confidence intervals and tail risks.' }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import { runMonteCarlo } from '../api/index.js'
import { useI18n } from '../i18n/index.js'

const { $t, locale } = useI18n()

const emit = defineEmits(['toast'])
const props = defineProps({ runId: { type: String, default: '' } })

const loading = ref(false)
const result = ref(null)
const mcMethod = ref('bootstrap')
const nSims = ref(1000)
const distChartRef = ref(null)
let distChart = null
let resizeObs = null

async function runMC() {
  if (!props.runId) {
    emit('toast', { message: locale.value === 'zh-CN' ? '请先运行流水线' : 'Run a pipeline first', type: 'error' })
    return
  }
  loading.value = true
  try {
    result.value = await runMonteCarlo({
      run_id: props.runId,
      method: mcMethod.value,
      n_simulations: nSims.value,
      horizon_days: 252,
    })
    await nextTick()
    renderDistChart()
    emit('toast', { message: locale.value === 'zh-CN' ? `蒙特卡洛：${nSims.value}次模拟已完成` : `Monte Carlo: ${nSims.value} simulations completed`, type: 'success' })
  } catch (e) {
    emit('toast', { message: locale.value === 'zh-CN' ? `模拟失败：${e.response?.data?.detail || e.message}` : `Simulation failed: ${e.response?.data?.detail || e.message}`, type: 'error' })
  } finally {
    loading.value = false
  }
}

function renderDistChart() {
  if (!distChartRef.value || !result.value?.terminal_values?.length) return

  if (!distChart) {
    distChart = echarts.init(distChartRef.value, null, { renderer: 'canvas' })
  }

  // Build histogram
  const vals = result.value.terminal_values
  const min = Math.min(...vals)
  const max = Math.max(...vals)
  const nBins = 50
  const binWidth = (max - min) / nBins
  const bins = new Array(nBins).fill(0)
  const labels = []

  for (let i = 0; i < nBins; i++) {
    labels.push((min + (i + 0.5) * binWidth).toFixed(3))
  }
  for (const v of vals) {
    const idx = Math.min(Math.floor((v - min) / binWidth), nBins - 1)
    if (idx >= 0) bins[idx]++
  }

  distChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { top: 16, right: 16, bottom: 28, left: 40 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#8892a4', fontSize: 8, interval: 4 },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#8892a4', fontSize: 9 },
      splitLine: { lineStyle: { color: '#1e2a3a' } },
    },
    series: [{
      type: 'bar',
      data: bins,
      itemStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(77,166,255,0.8)' },
          { offset: 1, color: 'rgba(77,166,255,0.2)' },
        ]),
      },
      barWidth: '90%',
    }],
  })
}

onMounted(() => {
  resizeObs = new ResizeObserver(() => distChart?.resize())
  if (distChartRef.value) resizeObs.observe(distChartRef.value)
})
onBeforeUnmount(() => {
  resizeObs?.disconnect()
  distChart?.dispose()
})
</script>

<style scoped>
.mc-view { display: flex; flex-direction: column; gap: 12px; height: 100%; }
.mc-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; flex-wrap: wrap; gap: 8px; }
.mc-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; letter-spacing: 0.5px; }
.mc-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--orange); box-shadow: 0 0 6px rgba(251,191,36,0.5); }
.mc-actions { display: flex; gap: 8px; align-items: center; }
.mc-select { background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 10px; padding: 4px 8px; }
.mc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; flex-shrink: 0; }
.mc-panel { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 10px 12px; }
.mc-panel-title { font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.mc-stat-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.mc-stat { display: flex; flex-direction: column; gap: 2px; }
.mc-stat-label { font-size: 9px; color: var(--text-dim); }
.mc-stat-value { font-size: 14px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.mc-pos { color: var(--green); }
.mc-neg { color: var(--red); }
.mc-warn { color: var(--orange); }
.mc-chart-section { flex: 1; min-height: 120px; }
.mc-chart { width: 100%; height: 100%; }
.mc-fit-info { font-size: 10px; color: var(--text-dim); display: flex; gap: 6px; flex-shrink: 0; }
.mc-fit-label { font-weight: 600; }
.mc-fit-value { font-family: var(--font-mono); }
.mc-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--text-muted); }
.mc-empty-icon { font-size: 48px; opacity: 0.2; }
.mc-empty h3 { font-size: 14px; color: var(--text-secondary); }
.mc-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }
</style>
