<template>
  <div class="factor-corr">
    <div class="corr-header">
      <span class="corr-title">FACTOR CORRELATION MATRIX</span>
    </div>
    <div ref="chartRef" class="corr-chart"></div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  factors: { type: Array, default: () => [] },
})

const chartRef = ref(null)
let chart = null
let resizeObs = null

function buildCorrelationMatrix(factorNames) {
  // Generate synthetic correlation matrix for demo (in production this comes from the backend)
  const n = factorNames.length
  const matrix = []
  const seed = 42
  let s = seed
  function rand() { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let val
      if (i === j) {
        val = 1.0
      } else if (j > i) {
        // Generate realistic low correlations (most factors are near-zero correlated)
        val = (rand() - 0.5) * 0.6
        // Momentum factors correlate with each other
        if (factorNames[i]?.includes('momentum') && factorNames[j]?.includes('momentum')) {
          val = 0.4 + rand() * 0.4
        }
        // Volatility factors correlate
        if (factorNames[i]?.includes('volatility') && factorNames[j]?.includes('volatility')) {
          val = 0.5 + rand() * 0.3
        }
      } else {
        val = matrix[j * n + i] // Symmetric
      }
      matrix.push(Math.round(val * 100) / 100)
    }
  }
  return matrix
}

function render() {
  if (!chartRef.value || !props.factors.length) return

  const names = props.factors.map(f => f.name || f)
  const matrix = buildCorrelationMatrix(names)
  const n = names.length
  const data = []

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      data.push([j, i, matrix[i * n + j]])
    }
  }

  if (!chart) {
    chart = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  }

  chart.setOption({
    tooltip: {
      formatter: (p) => `${names[p.data[0]]} x ${names[p.data[1]]}: ${p.data[2].toFixed(3)}`,
    },
    grid: { top: 40, right: 12, bottom: 60, left: 70 },
    xAxis: {
      type: 'category',
      data: names,
      axisLabel: { rotate: 45, fontSize: 9, color: '#8892a4' },
      axisLine: { lineStyle: { color: '#1e2a3a' } },
      splitArea: { show: false },
    },
    yAxis: {
      type: 'category',
      data: names,
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

watch(() => props.factors, () => nextTick(render), { deep: true })

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
.factor-corr {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 6px;
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
</style>
