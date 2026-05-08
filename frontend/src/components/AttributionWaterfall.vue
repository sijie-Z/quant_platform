<template>
  <div class="attribution">
    <div v-if="!attribution || !attribution.length" class="aw-empty">No attribution data</div>
    <template v-else>
      <div ref="chartRef" class="aw-chart"></div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  attribution: { type: Array, default: () => [] },
})

const chartRef = ref(null)
let chartInstance = null
let resizeObserver = null

function renderChart() {
  if (!chartRef.value || !props.attribution?.length) return
  if (chartInstance) chartInstance.dispose()

  chartInstance = echarts.init(chartRef.value, null, { renderer: 'canvas' })

  const factors = props.attribution.map(a => a.factor)
  const values = props.attribution.map(a => a.contribution_bps)
  const colors = values.map(v => v >= 0 ? '#34d399' : '#f87171')

  chartInstance.setOption({
    textStyle: { color: '#8b9dc0', fontFamily: 'Inter, -apple-system, sans-serif', fontSize: 10 },
    grid: { left: 100, right: 30, top: 10, bottom: 10 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15, 24, 41, 0.95)',
      borderColor: '#1c2d4a',
      borderWidth: 1,
      textStyle: { color: '#e8edf5', fontSize: 11 },
      extraCssText: 'backdrop-filter: blur(8px); border-radius: 6px;',
      formatter: params => {
        const p = params[0]
        const item = props.attribution[p.dataIndex]
        return `<b>${item.factor}</b><br/>Contribution: ${item.contribution_bps > 0 ? '+' : ''}${item.contribution_bps} bps<br/>Weight: ${(item.weight * 100).toFixed(1)}%<br/>Avg IC: ${item.avg_ic?.toFixed(4)}`
      },
    },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: '#1c2d4a' } },
      axisLabel: { color: '#556882', fontSize: 9, formatter: v => v + ' bps' },
      splitLine: { lineStyle: { color: '#152035', type: 'dashed' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'category',
      data: factors,
      axisLine: { lineStyle: { color: '#1c2d4a' } },
      axisLabel: { color: '#8b9dc0', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' },
      axisTick: { show: false },
      inverse: true,
    },
    series: [{
      type: 'bar',
      data: values.map((v, i) => ({
        value: v,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, v >= 0 ? 1 : -1, 0, [
            { offset: 0, color: v >= 0 ? 'rgba(52,211,153,0.2)' : 'rgba(248,113,113,0.2)' },
            { offset: 1, color: v >= 0 ? 'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)' },
          ]),
          borderRadius: v >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3],
        },
      })),
      barWidth: '55%',
      label: {
        show: true,
        position: 'right',
        fontSize: 9,
        fontFamily: 'JetBrains Mono, monospace',
        color: '#8b9dc0',
        formatter: p => {
          const v = p.value
          return (v > 0 ? '+' : '') + v.toFixed(1) + ' bps'
        },
      },
    }],
  })

  if (window.ResizeObserver) {
    resizeObserver = new ResizeObserver(() => chartInstance?.resize())
    resizeObserver.observe(chartRef.value)
  }
}

watch(() => props.attribution, () => nextTick(renderChart), { deep: true })
onMounted(() => { if (props.attribution?.length) nextTick(renderChart) })
onBeforeUnmount(() => {
  if (chartInstance) chartInstance.dispose()
  if (resizeObserver) resizeObserver.disconnect()
})
</script>

<style scoped>
.attribution {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.aw-chart {
  flex: 1;
  min-height: 0;
  width: 100%;
}

.aw-empty {
  color: var(--text-dim);
  font-size: 11px;
  text-align: center;
  padding: 20px;
}
</style>
