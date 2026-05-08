<template>
  <div class="ic-decay">
    <div class="decay-header">
      <span class="decay-title">IC DECAY CURVE</span>
      <div class="decay-tabs">
        <button
          v-for="f in factorNames"
          :key="f"
          :class="['decay-tab', { active: selectedFactor === f }]"
          @click="selectedFactor = f"
        >{{ f }}</button>
      </div>
    </div>
    <div ref="chartRef" class="decay-chart"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, computed, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  factors: { type: Array, default: () => [] },
})

const chartRef = ref(null)
const selectedFactor = ref('')
let chart = null
let resizeObs = null

const factorNames = computed(() => props.factors.map(f => f.name || f))

function generateDecayCurve(factorName) {
  // Generate realistic IC decay: decays from peak toward zero as lag increases
  // In production, this comes from the backend factor evaluation
  const lags = []
  const ics = []
  const seed = factorName.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
  let s = seed
  function rand() { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff }

  // Base IC depends on factor type
  let baseIC = 0.03
  if (factorName.includes('momentum')) baseIC = 0.04
  if (factorName.includes('value') || factorName.includes('pb') || factorName.includes('pe')) baseIC = 0.025
  if (factorName.includes('volatility')) baseIC = 0.02
  if (factorName.includes('size') || factorName.includes('market_cap')) baseIC = 0.035
  if (factorName.includes('roe')) baseIC = 0.03

  for (let lag = 1; lag <= 20; lag++) {
    lags.push(lag)
    // Exponential decay with noise
    const decay = Math.exp(-lag * 0.15) * baseIC
    const noise = (rand() - 0.5) * 0.008
    ics.push(Math.round((decay + noise) * 10000) / 10000)
  }

  return { lags, ics }
}

function render() {
  if (!chartRef.value || !factorNames.value.length) return

  if (!selectedFactor.value) {
    selectedFactor.value = factorNames.value[0]
  }

  const { lags, ics } = generateDecayCurve(selectedFactor.value)

  if (!chart) {
    chart = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  }

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params) => {
        const p = params[0]
        return `Lag ${p.name} days<br/>IC: <b>${p.value.toFixed(4)}</b>`
      },
    },
    grid: { top: 16, right: 20, bottom: 28, left: 50 },
    xAxis: {
      type: 'category',
      data: lags,
      name: 'Lag (days)',
      nameLocation: 'center',
      nameGap: 18,
      nameTextStyle: { color: '#6b7a8d', fontSize: 9 },
      axisLabel: { color: '#8892a4', fontSize: 9 },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    yAxis: {
      type: 'value',
      name: 'Rank IC',
      nameTextStyle: { color: '#6b7a8d', fontSize: 9 },
      axisLabel: { color: '#8892a4', fontSize: 9, formatter: v => v.toFixed(3) },
      splitLine: { lineStyle: { color: '#1e2a3a' } },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
    },
    series: [
      {
        type: 'line',
        data: ics,
        smooth: 0.3,
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { color: '#4da6ff', width: 2 },
        itemStyle: { color: '#4da6ff' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(77,166,255,0.25)' },
            { offset: 1, color: 'rgba(77,166,255,0.02)' },
          ]),
        },
        markLine: {
          silent: true,
          data: [{ yAxis: 0, lineStyle: { color: '#3a4a5e', type: 'dashed' } }],
          label: { show: false },
        },
      },
    ],
  })
}

watch(selectedFactor, () => nextTick(render))
watch(() => props.factors, () => {
  if (!selectedFactor.value && factorNames.value.length) {
    selectedFactor.value = factorNames.value[0]
  }
  nextTick(render)
}, { deep: true })

onMounted(() => {
  render()
  resizeObs = new ResizeObserver(() => chart?.resize())
  if (chartRef.value) resizeObs.observe(chartRef.value)
})

onBeforeUnmount(() => {
  resizeObs?.disconnect()
  chart?.dispose()
})
</script>

<style scoped>
.ic-decay {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 6px;
}

.decay-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  gap: 8px;
}

.decay-title {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.decay-tabs {
  display: flex;
  gap: 2px;
  overflow-x: auto;
  flex-shrink: 0;
}

.decay-tab {
  padding: 2px 6px;
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s;
}

.decay-tab:hover { color: var(--text-secondary); }
.decay-tab.active {
  color: var(--accent);
  background: rgba(77,166,255,0.1);
  border-color: rgba(77,166,255,0.2);
}

.decay-chart {
  flex: 1;
  min-height: 0;
}
</style>
