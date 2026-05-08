<template>
  <div class="return-dist">
    <div v-if="!distribution" class="rd-empty">No distribution data</div>
    <template v-else>
      <div ref="chartRef" class="rd-chart"></div>
      <div class="rd-stats">
        <div class="rd-stat">
          <span class="rd-stat-label">Mean</span>
          <span class="rd-stat-value">{{ distribution.mean?.toFixed(3) }}%</span>
        </div>
        <div class="rd-stat">
          <span class="rd-stat-label">Std Dev</span>
          <span class="rd-stat-value">{{ distribution.std?.toFixed(3) }}%</span>
        </div>
        <div class="rd-stat">
          <span class="rd-stat-label">Skew</span>
          <span :class="['rd-stat-value', skewClass]">{{ distribution.skew?.toFixed(3) }}</span>
        </div>
        <div class="rd-stat">
          <span class="rd-stat-label">Kurt</span>
          <span :class="['rd-stat-value', kurtClass]">{{ distribution.kurtosis?.toFixed(3) }}</span>
        </div>
        <div class="rd-stat">
          <span class="rd-stat-label">Min</span>
          <span class="rd-stat-value rd-neg">{{ distribution.min?.toFixed(2) }}%</span>
        </div>
        <div class="rd-stat">
          <span class="rd-stat-label">Max</span>
          <span class="rd-stat-value rd-pos">{{ distribution.max?.toFixed(2) }}%</span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  distribution: { type: Object, default: null },
})

const chartRef = ref(null)
let chartInstance = null
let resizeObserver = null

const skewClass = computed(() => {
  const s = props.distribution?.skew
  if (s == null) return ''
  return s < -0.5 ? 'rd-neg' : s > 0.5 ? 'rd-pos' : ''
})

const kurtClass = computed(() => {
  const k = props.distribution?.kurtosis
  if (k == null) return ''
  return k > 3 ? 'rd-warn' : ''
})

function normalPDF(x, mean, std) {
  const z = (x - mean) / std
  return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI))
}

function renderChart() {
  if (!chartRef.value || !props.distribution) return
  if (chartInstance) chartInstance.dispose()

  chartInstance = echarts.init(chartRef.value, null, { renderer: 'canvas' })

  const d = props.distribution
  const edges = d.edges || []
  const counts = d.counts || []
  if (!edges.length || !counts.length) return

  // Build bar data: midpoint of each bin
  const barData = []
  const normalData = []
  const totalArea = counts.reduce((s, c) => s + c, 0)
  const binWidth = edges.length > 1 ? edges[1] - edges[0] : 1

  for (let i = 0; i < counts.length; i++) {
    const mid = (edges[i] + edges[i + 1]) / 2
    barData.push([mid, counts[i]])
    // Normal distribution overlay (scaled to match histogram area)
    const pdf = normalPDF(mid, d.mean, d.std)
    normalData.push([mid, pdf * totalArea * binWidth])
  }

  chartInstance.setOption({
    textStyle: { color: '#8b9dc0', fontFamily: 'Inter, -apple-system, sans-serif', fontSize: 10 },
    grid: { left: 40, right: 10, top: 10, bottom: 24 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 24, 41, 0.95)',
      borderColor: '#1c2d4a',
      borderWidth: 1,
      textStyle: { color: '#e8edf5', fontSize: 11 },
      extraCssText: 'backdrop-filter: blur(8px); border-radius: 6px;',
      formatter: params => {
        const bar = params.find(p => p.seriesName === 'Returns')
        const norm = params.find(p => p.seriesName === 'Normal')
        const x = params[0]?.value?.[0]?.toFixed(2)
        let s = `<b>${x}%</b><br/>`
        if (bar) s += `Count: ${bar.value[1]}<br/>`
        if (norm) s += `Normal: ${norm.value[1]?.toFixed(1)}`
        return s
      },
    },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#1c2d4a' } },
      axisLabel: { color: '#556882', fontSize: 9, formatter: v => v.toFixed(1) + '%' },
      axisTick: { show: false },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: '#556882', fontSize: 9 },
      splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
      axisTick: { show: false },
    },
    series: [
      {
        name: 'Returns',
        type: 'bar',
        data: barData,
        barWidth: '85%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(77,166,255,0.6)' },
            { offset: 1, color: 'rgba(77,166,255,0.15)' },
          ]),
          borderRadius: [1, 1, 0, 0],
        },
        markLine: {
          silent: true,
          symbol: 'none',
          data: [
            {
              xAxis: d.mean,
              lineStyle: { color: '#34d399', width: 1, type: 'solid' },
              label: {
                show: true,
                formatter: 'μ ' + d.mean?.toFixed(2) + '%',
                color: '#34d399',
                fontSize: 9,
                position: 'insideEndTop',
              },
            },
          ],
        },
      },
      {
        name: 'Normal',
        type: 'line',
        data: normalData,
        smooth: 0.4,
        lineStyle: { color: '#fbbf24', width: 1.5, type: 'dashed' },
        symbol: 'none',
      },
    ],
  })

  if (window.ResizeObserver) {
    resizeObserver = new ResizeObserver(() => chartInstance?.resize())
    resizeObserver.observe(chartRef.value)
  }
}

watch(() => props.distribution, () => nextTick(renderChart), { deep: true })
onMounted(() => { if (props.distribution) nextTick(renderChart) })
onBeforeUnmount(() => {
  if (chartInstance) chartInstance.dispose()
  if (resizeObserver) resizeObserver.disconnect()
})
</script>

<style scoped>
.return-dist {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 6px;
}

.rd-chart {
  flex: 1;
  min-height: 0;
  width: 100%;
}

.rd-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
  padding: 4px 0;
}

.rd-stat {
  display: flex;
  align-items: center;
  gap: 4px;
}

.rd-stat-label {
  font-size: 9px;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.3px;
  text-transform: uppercase;
}

.rd-stat-value {
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.rd-stat-value.rd-pos { color: var(--green); }
.rd-stat-value.rd-neg { color: var(--red); }
.rd-stat-value.rd-warn { color: var(--orange); }

.rd-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
