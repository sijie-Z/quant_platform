<template>
  <div class="turnover-chart">
    <div v-if="!turnover || !turnover.length" class="tc-empty">No turnover data</div>
    <template v-else>
      <div ref="chartRef" class="tc-chart"></div>
      <div class="tc-stats">
        <div class="tc-stat">
          <span class="tc-stat-label">Avg Turnover</span>
          <span class="tc-stat-value">{{ avgTurnover }}%</span>
        </div>
        <div class="tc-stat">
          <span class="tc-stat-label">Max Turnover</span>
          <span class="tc-stat-value tc-warn">{{ maxTurnover }}%</span>
        </div>
        <div class="tc-stat">
          <span class="tc-stat-label">Avg Trades</span>
          <span class="tc-stat-value">{{ avgTrades }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  turnover: { type: Array, default: () => [] },
})

const chartRef = ref(null)
let chartInstance = null
let resizeObserver = null

const avgTurnover = computed(() => {
  if (!props.turnover?.length) return '--'
  const avg = props.turnover.reduce((s, t) => s + t.turnover, 0) / props.turnover.length
  return (avg * 100).toFixed(1)
})

const maxTurnover = computed(() => {
  if (!props.turnover?.length) return '--'
  return (Math.max(...props.turnover.map(t => t.turnover)) * 100).toFixed(1)
})

const avgTrades = computed(() => {
  if (!props.turnover?.length) return '--'
  return Math.round(props.turnover.reduce((s, t) => s + t.n_trades, 0) / props.turnover.length)
})

function renderChart() {
  if (!chartRef.value || !props.turnover?.length) return
  if (chartInstance) chartInstance.dispose()

  chartInstance = echarts.init(chartRef.value, null, { renderer: 'canvas' })
  const dates = props.turnover.map(t => t.date)
  const values = props.turnover.map(t => (t.turnover * 100).toFixed(1))
  const trades = props.turnover.map(t => t.n_trades)

  chartInstance.setOption({
    textStyle: { color: '#8b9dc0', fontFamily: 'Inter, -apple-system, sans-serif', fontSize: 10 },
    grid: { left: 44, right: 44, top: 10, bottom: 28 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 24, 41, 0.95)',
      borderColor: '#1c2d4a',
      borderWidth: 1,
      textStyle: { color: '#e8edf5', fontSize: 11 },
      extraCssText: 'backdrop-filter: blur(8px); border-radius: 6px;',
      formatter: params => {
        const bar = params.find(p => p.seriesName === 'Turnover')
        const line = params.find(p => p.seriesName === 'Trades')
        let s = `<b>${params[0]?.axisValue}</b><br/>`
        if (bar) s += `Turnover: ${bar.value}%<br/>`
        if (line) s += `Trades: ${line.value}`
        return s
      },
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#1c2d4a' } },
      axisLabel: { color: '#556882', fontSize: 9, interval: Math.max(1, Math.floor(dates.length / 6)) },
      axisTick: { show: false },
    },
    yAxis: [
      {
        type: 'value',
        name: 'Turnover %',
        nameTextStyle: { color: '#556882', fontSize: 9 },
        axisLine: { show: false },
        axisLabel: { color: '#556882', fontSize: 9, formatter: v => v + '%' },
        splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
        axisTick: { show: false },
      },
      {
        type: 'value',
        name: 'Trades',
        nameTextStyle: { color: '#556882', fontSize: 9 },
        axisLine: { show: false },
        axisLabel: { color: '#556882', fontSize: 9 },
        splitLine: { show: false },
        axisTick: { show: false },
      },
    ],
    series: [
      {
        name: 'Turnover',
        type: 'bar',
        data: values,
        barWidth: '60%',
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(167,139,250,0.6)' },
            { offset: 1, color: 'rgba(167,139,250,0.1)' },
          ]),
          borderRadius: [2, 2, 0, 0],
        },
      },
      {
        name: 'Trades',
        type: 'line',
        yAxisIndex: 1,
        data: trades,
        smooth: 0.3,
        lineStyle: { color: '#fbbf24', width: 1.5 },
        symbol: 'circle',
        symbolSize: 3,
        itemStyle: { color: '#fbbf24' },
      },
    ],
  })

  if (window.ResizeObserver) {
    resizeObserver = new ResizeObserver(() => chartInstance?.resize())
    resizeObserver.observe(chartRef.value)
  }
}

watch(() => props.turnover, () => nextTick(renderChart), { deep: true })
onMounted(() => { if (props.turnover?.length) nextTick(renderChart) })
onBeforeUnmount(() => {
  if (chartInstance) chartInstance.dispose()
  if (resizeObserver) resizeObserver.disconnect()
})
</script>

<style scoped>
.turnover-chart {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 4px;
}

.tc-chart {
  flex: 1;
  min-height: 0;
  width: 100%;
}

.tc-stats {
  display: flex;
  gap: 14px;
  flex-shrink: 0;
  padding: 3px 0;
}

.tc-stat {
  display: flex;
  align-items: center;
  gap: 4px;
}

.tc-stat-label {
  font-size: 8px;
  font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.tc-stat-value {
  font-size: 10px;
  font-weight: 700;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.tc-warn { color: var(--orange); }

.tc-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
