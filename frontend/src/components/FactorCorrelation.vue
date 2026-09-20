<template>
  <div class="factor-corr">
    <div class="corr-header">
      <span class="corr-title">FACTOR CORRELATION MATRIX</span>
    </div>
    <div ref="chartRef" class="corr-chart"></div>
    <div v-if="!ready" class="corr-empty">{{ message }}</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getFactorCorrelation } from '../api/index.js'

/**
 * The matrix comes from `/api/analysis/correlation`. It used to be generated
 * here, in the browser: a seeded LCG (seed 42) plus rules making factors whose
 * names contain "momentum" correlate by construction. That was the same
 * invention the endpoint itself was removed for (BUG-45), one layer up -- the
 * panel never called the API at all.
 */
const chartRef = ref(null)
const names = ref([])
const matrix = ref([])
const message = ref('Loading factor correlation...')
const ready = ref(false)
let chart = null
let resizeObs = null

async function load() {
  try {
    const data = await getFactorCorrelation()
    if (data?.available === false) {
      message.value = data.reason || 'Factor correlation is not available for this run.'
      return
    }
    names.value = data?.names ?? []
    matrix.value = data?.matrix ?? []
    if (!names.value.length || !matrix.value.length) {
      message.value = 'No factor correlation data for this run.'
      return
    }
    ready.value = true
  } catch (e) {
    message.value = `Factor correlation request failed: ${e?.message || e}`
  }
}

function render() {
  const n = names.value.length
  if (!chartRef.value || !ready.value || !n) return

  const data = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      data.push([j, i, matrix.value[i][j]])
    }
  }

  if (!chart) {
    chart = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  }

  chart.setOption({
    tooltip: {
      formatter: (p) => `${names.value[p.data[0]]} x ${names.value[p.data[1]]}: ${p.data[2].toFixed(3)}`,
    },
    grid: { top: 40, right: 12, bottom: 60, left: 70 },
    xAxis: {
      type: 'category',
      data: names.value,
      axisLabel: { rotate: 45, fontSize: 9, color: '#8892a4' },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
      splitArea: { show: false },
    },
    yAxis: {
      type: 'category',
      data: names.value,
      axisLabel: { fontSize: 9, color: '#8892a4' },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
      splitArea: { show: false },
    },
    visualMap: {
      min: -1,
      max: 1,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      itemWidth: 12,
      itemHeight: 100,
      textStyle: { color: '#6b7a8d', fontSize: 9 },
      inRange: {
        color: ['#ef4444', '#1e2a3a', '#22c55e'],
      },
    },
    series: [{
      type: 'heatmap',
      data: data,
      label: {
        show: n <= 12,
        fontSize: 8,
        color: '#c9d1d9',
        formatter: (p) => p.data[2].toFixed(2),
      },
      emphasis: {
        itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.4)' },
      },
    }],
  })
}

onMounted(async () => {
  await load()
  await nextTick()
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
.factor-corr {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 6px;
  position: relative;
}

.corr-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.corr-title {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.corr-chart {
  flex: 1;
  min-height: 0;
}

.corr-empty {
  position: absolute;
  inset: 20px 0 0 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 16px;
  text-align: center;
  font-size: 11px;
  line-height: 1.6;
  color: var(--text-dim);
  pointer-events: none;
}
</style>
