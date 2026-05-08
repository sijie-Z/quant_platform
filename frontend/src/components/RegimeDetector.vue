<template>
  <div class="regime-view">
    <div class="regime-header">
      <div class="regime-title">
        <span :class="['regime-dot', `regime-${regime?.overall_regime || 'neutral'}`]"></span>
        MARKET REGIME DETECTOR
      </div>
      <div class="regime-actions">
        <button class="btn btn-sm btn-primary" @click="detect" :disabled="loading">
          {{ loading ? 'Analyzing...' : 'Detect Regime' }}
        </button>
      </div>
    </div>

    <template v-if="regime">
      <!-- Composite Score -->
      <div class="regime-score-section">
        <div class="regime-score-ring">
          <svg viewBox="0 0 100 100" class="regime-ring-svg">
            <circle cx="50" cy="50" r="40" fill="none" stroke="var(--bg-input)" stroke-width="6" />
            <circle cx="50" cy="50" r="40" fill="none"
              :stroke="scoreColor"
              stroke-width="6"
              stroke-linecap="round"
              :stroke-dasharray="`${regime.composite_risk_score * 251} 251`"
              transform="rotate(-90 50 50)"
            />
          </svg>
          <div class="regime-score-label">
            <div class="regime-score-value">{{ (regime.composite_risk_score * 100).toFixed(0) }}</div>
            <div class="regime-score-sub">RISK</div>
          </div>
        </div>
        <div class="regime-score-info">
          <div :class="['regime-badge-lg', `regime-badge-${regime.overall_regime}`]">
            {{ regime.overall_regime?.toUpperCase() }}
          </div>
          <div class="regime-recommendation">{{ regime.recommendation }}</div>
        </div>
      </div>

      <!-- Sub-Regimes -->
      <div class="regime-grid">
        <!-- Volatility -->
        <div class="regime-panel">
          <div class="regime-panel-title">VOLATILITY</div>
          <div :class="['regime-badge', `vol-${regime.volatility?.regime}`]">
            {{ regime.volatility?.regime?.replace('_', ' ')?.toUpperCase() }}
          </div>
          <div class="regime-detail">
            <span>Current Vol</span>
            <span class="regime-mono">{{ (regime.volatility?.current_vol * 100)?.toFixed(1) }}%</span>
          </div>
          <div class="regime-detail">
            <span>Percentile</span>
            <span class="regime-mono">{{ (regime.volatility?.percentile * 100)?.toFixed(0) }}%</span>
          </div>
          <div class="regime-detail">
            <span>Confidence</span>
            <span class="regime-mono">{{ (regime.volatility?.confidence * 100)?.toFixed(0) }}%</span>
          </div>
        </div>

        <!-- Trend -->
        <div class="regime-panel">
          <div class="regime-panel-title">TREND</div>
          <div :class="['regime-badge', `trend-${regime.trend?.regime}`]">
            {{ regime.trend?.regime?.toUpperCase() }}
          </div>
          <div class="regime-detail">
            <span>MA Spread</span>
            <span class="regime-mono">{{ (regime.trend?.ma_spread * 100)?.toFixed(2) }}%</span>
          </div>
          <div class="regime-detail">
            <span>Crossovers</span>
            <span class="regime-mono">{{ regime.trend?.recent_crossovers }}</span>
          </div>
          <div class="regime-detail">
            <span>Confidence</span>
            <span class="regime-mono">{{ (regime.trend?.confidence * 100)?.toFixed(0) }}%</span>
          </div>
        </div>

        <!-- Correlation -->
        <div class="regime-panel">
          <div class="regime-panel-title">CORRELATION</div>
          <div :class="['regime-badge', `corr-${regime.correlation?.regime}`]">
            {{ regime.correlation?.regime?.replace('_', ' ')?.toUpperCase() }}
          </div>
          <div class="regime-detail">
            <span>Avg Corr</span>
            <span class="regime-mono">{{ regime.correlation?.avg_correlation?.toFixed(3) }}</span>
          </div>
          <div class="regime-detail">
            <span>Assets</span>
            <span class="regime-mono">{{ regime.correlation?.n_assets || 'N/A' }}</span>
          </div>
        </div>
      </div>

      <!-- Regime History -->
      <div class="regime-history" v-if="regime.history?.length">
        <div class="regime-history-title">REGIME HISTORY</div>
        <div ref="historyChartRef" class="regime-history-chart"></div>
      </div>
    </template>

    <!-- Empty State -->
    <div v-if="!regime && !loading" class="regime-empty">
      <div class="regime-empty-icon">&#9729;</div>
      <h3>Market Regime Detection</h3>
      <p>Run a pipeline first, then detect the current market regime using volatility, trend, and correlation analysis.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick, computed } from 'vue'
import * as echarts from 'echarts'
import { detectRegime } from '../api/index.js'

const emit = defineEmits(['toast'])
const props = defineProps({ runId: { type: String, default: '' } })

const loading = ref(false)
const regime = ref(null)
const historyChartRef = ref(null)
let historyChart = null
let resizeObs = null

const scoreColor = computed(() => {
  const s = regime.value?.composite_risk_score || 0
  if (s > 0.75) return '#ef4444'
  if (s > 0.5) return '#fbbf24'
  if (s > 0.25) return '#4da6ff'
  return '#22c55e'
})

async function detect() {
  if (!props.runId) {
    emit('toast', { message: 'Run a pipeline first', type: 'error' })
    return
  }
  loading.value = true
  try {
    regime.value = await detectRegime({ run_id: props.runId })
    await nextTick()
    renderHistory()
    emit('toast', { message: `Regime: ${regime.value.overall_regime}`, type: 'success' })
  } catch (e) {
    emit('toast', { message: `Detection failed: ${e.response?.data?.detail || e.message}`, type: 'error' })
  } finally {
    loading.value = false
  }
}

function renderHistory() {
  if (!historyChartRef.value || !regime.value?.history?.length) return

  if (!historyChart) {
    historyChart = echarts.init(historyChartRef.value, null, { renderer: 'canvas' })
  }

  const history = regime.value.history
  const dates = history.map(h => h.date)
  const volPct = history.map(h => h.vol_percentile)
  const colors = history.map(h => {
    if (h.volatility === 'extreme_volatility') return '#ef4444'
    if (h.volatility === 'high_volatility') return '#fbbf24'
    if (h.volatility === 'medium_volatility') return '#4da6ff'
    return '#22c55e'
  })

  historyChart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const idx = params[0].dataIndex
        const h = history[idx]
        return `${h.date}<br/>Vol: ${h.volatility}<br/>Trend: ${h.trend}<br/>Percentile: ${(h.vol_percentile * 100).toFixed(0)}%`
      },
    },
    grid: { top: 8, right: 16, bottom: 24, left: 40 },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: { color: '#8892a4', fontSize: 8, rotate: 45 },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    yAxis: {
      type: 'value',
      max: 1,
      axisLabel: { color: '#8892a4', fontSize: 9, formatter: v => (v * 100) + '%' },
      splitLine: { lineStyle: { color: '#1e2a3a' } },
    },
    series: [{
      type: 'bar',
      data: volPct.map((v, i) => ({
        value: v,
        itemStyle: { color: colors[i] },
      })),
      barWidth: '60%',
    }],
  })
}

onMounted(() => {
  resizeObs = new ResizeObserver(() => historyChart?.resize())
  if (historyChartRef.value) resizeObs.observe(historyChartRef.value)
})
onBeforeUnmount(() => {
  resizeObs?.disconnect()
  historyChart?.dispose()
})
</script>

<style scoped>
.regime-view { display: flex; flex-direction: column; gap: 12px; height: 100%; }
.regime-header { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.regime-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; letter-spacing: 0.5px; }
.regime-dot { width: 8px; height: 8px; border-radius: 50%; }
.regime-dot.risk_on { background: var(--green); box-shadow: 0 0 6px rgba(52,211,153,0.5); }
.regime-dot.neutral { background: var(--accent); box-shadow: 0 0 6px rgba(77,166,255,0.4); }
.regime-dot.cautious { background: var(--orange); box-shadow: 0 0 6px rgba(251,191,36,0.4); }
.regime-dot.risk_off { background: var(--red); box-shadow: 0 0 6px rgba(239,68,68,0.5); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
.regime-actions { display: flex; gap: 8px; }

.regime-score-section { display: flex; gap: 16px; align-items: center; flex-shrink: 0; }
.regime-score-ring { position: relative; width: 80px; height: 80px; flex-shrink: 0; }
.regime-ring-svg { width: 100%; height: 100%; }
.regime-score-label { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.regime-score-value { font-size: 20px; font-weight: 700; font-family: var(--font-mono); color: var(--text-primary); }
.regime-score-sub { font-size: 8px; color: var(--text-dim); letter-spacing: 1px; }
.regime-score-info { flex: 1; }
.regime-badge-lg { font-size: 14px; font-weight: 700; letter-spacing: 1px; margin-bottom: 4px; }
.regime-badge-lg.risk_on { color: var(--green); }
.regime-badge-lg.neutral { color: var(--accent); }
.regime-badge-lg.cautious { color: var(--orange); }
.regime-badge-lg.risk_off { color: var(--red); }
.regime-recommendation { font-size: 11px; color: var(--text-secondary); line-height: 1.5; }

.regime-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; flex-shrink: 0; }
.regime-panel { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 10px 12px; }
.regime-panel-title { font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.regime-badge { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 3px; display: inline-block; margin-bottom: 8px; letter-spacing: 0.3px; }
.vol-low_volatility { color: var(--green); background: rgba(52,211,153,0.1); }
.vol-medium_volatility { color: var(--accent); background: rgba(77,166,255,0.1); }
.vol-high_volatility { color: var(--orange); background: rgba(251,191,36,0.1); }
.vol-extreme_volatility { color: var(--red); background: rgba(239,68,68,0.1); }
.trend-bull { color: var(--green); background: rgba(52,211,153,0.1); }
.trend-bear { color: var(--red); background: rgba(239,68,68,0.1); }
.trend-sideways { color: var(--text-dim); background: rgba(100,100,100,0.1); }
.corr-normal_correlation { color: var(--green); background: rgba(52,211,153,0.1); }
.corr-stressed_correlation { color: var(--red); background: rgba(239,68,68,0.1); }
.regime-detail { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim); padding: 2px 0; }
.regime-mono { font-family: var(--font-mono); color: var(--text-secondary); }

.regime-history { flex: 1; min-height: 80px; }
.regime-history-title { font-size: 9px; font-weight: 700; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.regime-history-chart { width: 100%; height: 100%; }

.regime-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--text-muted); }
.regime-empty-icon { font-size: 48px; opacity: 0.2; }
.regime-empty h3 { font-size: 14px; color: var(--text-secondary); }
.regime-empty p { font-size: 12px; max-width: 400px; text-align: center; line-height: 1.6; }
</style>
